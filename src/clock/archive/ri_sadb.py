from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.clock.defect_balanced import (
    StepFn,
    VelocityFn,
    _call_velocity,
    _microbatch_map,
    _refined_step,
    estimate_refinement_order_and_defect,
    per_sample_l2_norm,
    smooth_profile,
)
from src.clock.profile import ClockProfile, build_clock_profile_from_alpha

RI_SADB_CLOCK_VERSION = 1
RI_SADB_FORMULA_VERSION = 1


@dataclass(frozen=True)
class TrajectoryGeometryStats:
    full_step_error: np.ndarray
    half_step_error: np.ndarray
    effective_order: np.ndarray
    delta_s: np.ndarray
    curvature: np.ndarray
    residual_perp_norm: np.ndarray
    residual_parallel_norm: np.ndarray


@dataclass(frozen=True)
class RISADBArtifacts:
    profile: ClockProfile
    defect_profile: np.ndarray
    smoothed_defect_profile: np.ndarray
    effective_order_profile: np.ndarray
    smoothed_effective_order_profile: np.ndarray
    interval_alpha_profile: np.ndarray
    geometry_profile: np.ndarray
    residual_perp_profile: np.ndarray
    residual_parallel_profile: np.ndarray


@dataclass(frozen=True)
class ShortWindowDefectStats:
    interval_arc_defect: np.ndarray
    interval_weight: np.ndarray
    window_len: int
    refine_factor: int
    q_prior: float
    defect_source: str
    status: str = "OK"


def _flatten_inner(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(tensor.shape[0], -1)


def _broadcast_per_sample(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return values.reshape((values.shape[0],) + (1,) * (target.ndim - 1))


def _per_sample_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (_flatten_inner(left) * _flatten_inner(right)).sum(dim=1)


def _interval_profile_to_nodes(interval_values: np.ndarray) -> np.ndarray:
    values = np.asarray(interval_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("interval profile must be a non-empty 1D array.")
    if len(values) == 1:
        return np.asarray([values[0], values[0]], dtype=np.float64)
    nodes = np.empty(len(values) + 1, dtype=np.float64)
    nodes[0] = values[0]
    nodes[-1] = values[-1]
    nodes[1:-1] = 0.5 * (values[:-1] + values[1:])
    return nodes


def _evaluate_velocity(
    velocity_fn: VelocityFn,
    sample: torch.Tensor,
    coordinate: float,
    sample_start: int,
    sample_stop: int,
) -> torch.Tensor:
    coordinate_tensor = torch.as_tensor(float(coordinate), device=sample.device, dtype=sample.dtype)
    return _call_velocity(velocity_fn, sample, coordinate_tensor, sample_start, sample_stop)


def collect_ri_sadb_stats(
    *,
    initial_sample: torch.Tensor,
    physical_grid: np.ndarray,
    velocity_fn: VelocityFn,
    step_fn: StepFn,
    observation_microbatch: int | None = None,
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> TrajectoryGeometryStats:
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")

    safe_eps = float(eps)
    current = initial_sample.detach()
    velocity_start = _microbatch_map(
        current,
        microbatch_size=observation_microbatch,
        fn=lambda batch, batch_start, batch_stop: _evaluate_velocity(
            velocity_fn,
            batch,
            float(grid[0]),
            batch_start,
            batch_stop,
        ),
    ).detach()

    full_errors: list[np.ndarray] = []
    half_errors: list[np.ndarray] = []
    delta_s_values: list[np.ndarray] = []
    curvature_values: list[np.ndarray] = []
    residual_perp_values: list[np.ndarray] = []
    residual_parallel_values: list[np.ndarray] = []

    for index in range(len(grid) - 1):
        start = float(grid[index])
        end = float(grid[index + 1])
        delta_u = end - start
        if abs(delta_u) <= safe_eps:
            raise ValueError("physical_grid must not contain repeated adjacent nodes.")

        full = _microbatch_map(
            current,
            microbatch_size=observation_microbatch,
            fn=lambda batch, batch_start, batch_stop, s=start, e=end: _refined_step(
                step_fn,
                batch,
                s,
                e,
                1,
                batch_start,
                batch_stop,
            ),
        )
        half = _microbatch_map(
            current,
            microbatch_size=observation_microbatch,
            fn=lambda batch, batch_start, batch_stop, s=start, e=end: _refined_step(
                step_fn,
                batch,
                s,
                e,
                2,
                batch_start,
                batch_stop,
            ),
        )
        quarter = _microbatch_map(
            current,
            microbatch_size=observation_microbatch,
            fn=lambda batch, batch_start, batch_stop, s=start, e=end: _refined_step(
                step_fn,
                batch,
                s,
                e,
                4,
                batch_start,
                batch_stop,
            ),
        )
        next_sample = quarter.detach()
        velocity_end = _microbatch_map(
            next_sample,
            microbatch_size=observation_microbatch,
            fn=lambda batch, batch_start, batch_stop, coordinate=end: _evaluate_velocity(
                velocity_fn,
                batch,
                coordinate,
                batch_start,
                batch_stop,
            ),
        ).detach()

        velocity_mid = 0.5 * (velocity_start + velocity_end)
        speed_start = per_sample_l2_norm(velocity_start)
        speed_end = per_sample_l2_norm(velocity_end)
        speed_mid_sq = _per_sample_dot(velocity_mid, velocity_mid)
        speed_mid = torch.sqrt(torch.clamp(speed_mid_sq, min=0.0))
        delta_s = 0.5 * (speed_start + speed_end) * abs(delta_u)

        acceleration = (velocity_end - velocity_start) / float(delta_u)
        accel_dot_velocity = _per_sample_dot(acceleration, velocity_mid)
        accel_parallel = _broadcast_per_sample(
            accel_dot_velocity / (speed_mid_sq + safe_eps),
            velocity_mid,
        ) * velocity_mid
        accel_perp = acceleration - accel_parallel
        curvature = per_sample_l2_norm(accel_perp) / (speed_mid_sq + safe_eps)

        tangent = velocity_mid / _broadcast_per_sample(speed_mid + safe_eps, velocity_mid)
        residual = full - half
        residual_parallel = _broadcast_per_sample(_per_sample_dot(residual, tangent), residual) * tangent
        residual_perp = residual - residual_parallel

        full_errors.append(per_sample_l2_norm(full - half).cpu().numpy())
        half_errors.append(per_sample_l2_norm(half - quarter).cpu().numpy())
        delta_s_values.append(delta_s.cpu().numpy())
        curvature_values.append(curvature.cpu().numpy())
        residual_perp_values.append(per_sample_l2_norm(residual_perp).cpu().numpy())
        residual_parallel_values.append(per_sample_l2_norm(residual_parallel).cpu().numpy())

        current = next_sample
        velocity_start = velocity_end

    full_error = np.stack(full_errors, axis=1)
    half_error = np.stack(half_errors, axis=1)
    effective_order, _ = estimate_refinement_order_and_defect(
        full_step_error=full_error,
        half_step_error=half_error,
        step_sizes=np.diff(grid),
        q_min=q_min,
        q_max=q_max,
        eps=safe_eps,
    )
    return TrajectoryGeometryStats(
        full_step_error=full_error,
        half_step_error=half_error,
        effective_order=effective_order,
        delta_s=np.maximum(np.stack(delta_s_values, axis=1), safe_eps),
        curvature=np.maximum(np.stack(curvature_values, axis=1), 0.0),
        residual_perp_norm=np.maximum(np.stack(residual_perp_values, axis=1), safe_eps),
        residual_parallel_norm=np.maximum(np.stack(residual_parallel_values, axis=1), safe_eps),
    )


def concatenate_ri_sadb_stats(items: Sequence[TrajectoryGeometryStats]) -> TrajectoryGeometryStats:
    if not items:
        raise ValueError("items must contain at least one TrajectoryGeometryStats object.")
    return TrajectoryGeometryStats(
        full_step_error=np.concatenate([item.full_step_error for item in items], axis=0),
        half_step_error=np.concatenate([item.half_step_error for item in items], axis=0),
        effective_order=np.concatenate([item.effective_order for item in items], axis=0),
        delta_s=np.concatenate([item.delta_s for item in items], axis=0),
        curvature=np.concatenate([item.curvature for item in items], axis=0),
        residual_perp_norm=np.concatenate([item.residual_perp_norm for item in items], axis=0),
        residual_parallel_norm=np.concatenate([item.residual_parallel_norm for item in items], axis=0),
    )


def distribute_short_window_arc_defect(
    *,
    delta_s: np.ndarray,
    window_residual: np.ndarray,
    window_len: int = 4,
    q_prior: float = 3.0,
    refine_factor: int = 2,
    eps: float = 1.0e-12,
    defect_source: str = "target_stork_short_window",
) -> ShortWindowDefectStats:
    arc = np.maximum(np.asarray(delta_s, dtype=np.float64), float(eps))
    residual = np.maximum(np.asarray(window_residual, dtype=np.float64), float(eps))
    if arc.ndim != 2:
        raise ValueError("delta_s must have shape [num_trajectories, num_intervals].")
    if residual.shape != arc.shape:
        raise ValueError("window_residual must have shape [num_trajectories, num_intervals].")
    width = max(int(window_len), 1)
    if width > arc.shape[1]:
        width = arc.shape[1]
    q_value = max(float(q_prior), 1.0 + float(eps))
    rho = max(abs(1.0 - 2.0 ** (1.0 - q_value)), float(eps))

    accumulated = np.zeros_like(arc, dtype=np.float64)
    coverage = np.zeros_like(arc, dtype=np.float64)
    for start in range(arc.shape[1]):
        stop = min(start + width, arc.shape[1])
        window_arc = np.maximum(np.sum(arc[:, start:stop], axis=1), float(eps))
        window_error = residual[:, start]
        window_defect = window_error / (np.power(window_arc + float(eps), q_value) * rho + float(eps))
        weights = arc[:, start:stop] / (window_arc[:, None] + float(eps))
        accumulated[:, start:stop] += weights * window_defect[:, None]
        coverage[:, start:stop] += weights

    interval_defect = accumulated / (coverage + float(eps))
    return ShortWindowDefectStats(
        interval_arc_defect=np.maximum(interval_defect, float(eps)),
        interval_weight=coverage,
        window_len=width,
        refine_factor=int(refine_factor),
        q_prior=q_value,
        defect_source=defect_source,
    )


def replace_ri_sadb_arc_defect(
    stats: TrajectoryGeometryStats,
    arc_defect: np.ndarray,
    *,
    q_prior: float,
    eps: float = 1.0e-12,
) -> TrajectoryGeometryStats:
    defect = np.maximum(np.asarray(arc_defect, dtype=np.float64), float(eps))
    delta_s = np.maximum(np.asarray(stats.delta_s, dtype=np.float64), float(eps))
    if defect.shape != delta_s.shape:
        raise ValueError("arc_defect must match stats.delta_s shape.")
    q = np.full_like(delta_s, max(float(q_prior), 1.0 + float(eps)), dtype=np.float64)
    rho = np.maximum(np.abs(1.0 - np.power(2.0, 1.0 - q)), float(eps))
    residual_perp = defect * (np.power(delta_s + float(eps), q) * rho + float(eps))
    return TrajectoryGeometryStats(
        full_step_error=np.asarray(stats.full_step_error, dtype=np.float64),
        half_step_error=np.asarray(stats.half_step_error, dtype=np.float64),
        effective_order=q,
        delta_s=delta_s,
        curvature=np.asarray(stats.curvature, dtype=np.float64),
        residual_perp_norm=np.maximum(residual_perp, float(eps)),
        residual_parallel_norm=np.full_like(delta_s, float(eps), dtype=np.float64),
    )


def build_ri_sadb_profile(
    physical_grid: np.ndarray,
    stats: TrajectoryGeometryStats,
    *,
    target_steps: int,
    eta: float = 0.25,
    beta: float = 0.0,
    ell_scale: str = "step",
    ri_agg: str = "mean",
    eps: float = 1.0e-12,
    q_min: float = 1.05,
    q_max: float = 6.0,
    smoothing_window: int = 1,
) -> RISADBArtifacts:
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")
    if int(target_steps) <= 0:
        raise ValueError("target_steps must be positive.")
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("eta must be in [0, 1].")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    if str(ell_scale).lower() != "step":
        raise ValueError("RI-SADB currently supports ell_scale='step' only.")
    if str(ri_agg).lower() != "mean":
        raise ValueError("RI-SADB currently supports ri_agg='mean' only.")

    safe_eps = float(eps)
    expected_shape = (stats.delta_s.shape[0], len(grid) - 1)
    for name, values in {
        "effective_order": stats.effective_order,
        "delta_s": stats.delta_s,
        "curvature": stats.curvature,
        "residual_perp_norm": stats.residual_perp_norm,
        "residual_parallel_norm": stats.residual_parallel_norm,
    }.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape [num_trajectories, len(physical_grid) - 1].")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf.")

    delta_s = np.maximum(np.asarray(stats.delta_s, dtype=np.float64), safe_eps)
    curvature = np.maximum(np.asarray(stats.curvature, dtype=np.float64), 0.0)
    q = np.clip(np.asarray(stats.effective_order, dtype=np.float64), max(float(q_min), 1.0 + safe_eps), float(q_max))
    residual_perp = np.maximum(np.asarray(stats.residual_perp_norm, dtype=np.float64), safe_eps)
    residual_parallel = np.maximum(np.asarray(stats.residual_parallel_norm, dtype=np.float64), safe_eps)

    path_length = np.maximum(np.sum(delta_s, axis=1, keepdims=True), safe_eps)
    ell = path_length / float(target_steps)
    geometry_weight = np.power(1.0 + np.square(ell * curvature), 0.25)
    n_geometry = geometry_weight / np.maximum(np.sum(geometry_weight * delta_s, axis=1, keepdims=True), safe_eps)

    residual_beta = np.sqrt(np.square(residual_perp) + float(beta) * np.square(residual_parallel))
    rho = np.maximum(np.abs(1.0 - np.power(2.0, 1.0 - q)), safe_eps)
    arc_defect = residual_beta / (np.power(delta_s + safe_eps, q) * rho + safe_eps)
    defect_weight = np.exp(
        (np.log(np.maximum(q - 1.0, safe_eps)) + np.log(np.maximum(arc_defect, safe_eps))) / q
    )
    n_defect = defect_weight / np.maximum(np.sum(defect_weight * delta_s, axis=1, keepdims=True), safe_eps)

    log_fused = (1.0 - float(eta)) * np.log(np.maximum(n_defect, safe_eps)) + float(eta) * np.log(
        np.maximum(n_geometry, safe_eps)
    )
    fused = np.exp(log_fused - np.max(log_fused, axis=1, keepdims=True))
    n_eta = fused / np.maximum(np.sum(fused * delta_s, axis=1, keepdims=True), safe_eps)

    delta_u = np.maximum(np.abs(np.diff(grid))[None, :], safe_eps)
    pulled_alpha = n_eta * delta_s / (delta_u + safe_eps)
    interval_alpha = np.maximum(np.mean(pulled_alpha, axis=0), safe_eps)
    if int(smoothing_window) > 1:
        interval_alpha = np.exp(smooth_profile(np.log(interval_alpha), int(smoothing_window)))
        interval_alpha = np.maximum(interval_alpha, safe_eps)
    node_alpha = _interval_profile_to_nodes(interval_alpha)
    if not np.all(np.isfinite(node_alpha)) or np.any(node_alpha <= 0.0):
        raise ValueError("RI-SADB produced an invalid alpha profile.")

    profile = build_clock_profile_from_alpha(grid, node_alpha)
    defect_profile = np.mean(arc_defect, axis=0)
    smoothed_defect = np.exp(smooth_profile(np.log(np.maximum(defect_profile, safe_eps)), int(smoothing_window)))
    effective_order_profile = np.mean(q, axis=0)
    smoothed_order = smooth_profile(effective_order_profile, int(smoothing_window))
    return RISADBArtifacts(
        profile=profile,
        defect_profile=defect_profile,
        smoothed_defect_profile=smoothed_defect,
        effective_order_profile=effective_order_profile,
        smoothed_effective_order_profile=smoothed_order,
        interval_alpha_profile=interval_alpha,
        geometry_profile=np.mean(curvature, axis=0),
        residual_perp_profile=np.mean(residual_perp, axis=0),
        residual_parallel_profile=np.mean(residual_parallel, axis=0),
    )

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
from src.clock.profile import ClockProfile

FP_CLOCK_VERSION = 8


@dataclass(frozen=True)
class FPTrajectoryStats:
    full_step_error: np.ndarray
    half_step_error: np.ndarray
    effective_order: np.ndarray
    delta_s: np.ndarray
    residual_perp_norm: np.ndarray


@dataclass(frozen=True)
class FPAnchoredReplayDetails:
    window_size: int
    window_residual_perp_norm: np.ndarray
    window_delta_s: np.ndarray
    window_effective_order: np.ndarray
    coverage: np.ndarray


@dataclass(frozen=True)
class FPClockArtifacts:
    profile: ClockProfile
    defect_profile: np.ndarray
    smoothed_defect_profile: np.ndarray
    effective_order_profile: np.ndarray
    smoothed_effective_order_profile: np.ndarray
    interval_alpha_profile: np.ndarray
    arc_length_profile: np.ndarray
    residual_perp_profile: np.ndarray


def _flatten_inner(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(tensor.shape[0], -1)


def _broadcast_per_sample(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return values.reshape((values.shape[0],) + (1,) * (target.ndim - 1))


def _per_sample_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (_flatten_inner(left) * _flatten_inner(right)).sum(dim=1)


def project_residual_to_frenet_normal(
    residual: torch.Tensor,
    tangent: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    tangent_norm = per_sample_l2_norm(tangent)
    unit_tangent = tangent / _broadcast_per_sample(tangent_norm + float(eps), tangent)
    residual_parallel = _broadcast_per_sample(_per_sample_dot(residual, unit_tangent), residual) * unit_tangent
    residual_perp = residual - residual_parallel
    return residual_perp, residual_parallel


def _as_trajectory_tensor(states: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(states, torch.Tensor):
        tensor = states.detach()
    else:
        if not states:
            raise ValueError("trajectory states must be non-empty.")
        tensor = torch.stack([state.detach() for state in states], dim=0)
    if tensor.ndim < 2:
        raise ValueError("trajectory states must have shape [nodes, batch, ...].")
    return tensor


def _window_l2_norm(values: torch.Tensor) -> np.ndarray:
    node_count, batch_count = values.shape[:2]
    return values.detach().float().reshape(node_count, batch_count, -1).norm(dim=2).cpu().numpy()


def collect_anchored_replay_stats(
    *,
    physical_grid: np.ndarray,
    reference_states: Sequence[torch.Tensor] | torch.Tensor,
    replay_1x_endpoints: Sequence[torch.Tensor] | torch.Tensor,
    replay_2x_endpoints: Sequence[torch.Tensor] | torch.Tensor,
    replay_4x_endpoints: Sequence[torch.Tensor] | torch.Tensor,
    window_size: int,
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> tuple[FPTrajectoryStats, FPAnchoredReplayDetails]:
    """Estimate FP clock stats from same-anchor coarse/half/quarter replay.

    Each replay endpoint must start from the same anchor state and cover the same
    native-coordinate window. This keeps the residual local to the target solver
    instead of comparing independently evolved complete trajectories.
    """
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two nodes.")
    if int(window_size) <= 0:
        raise ValueError("window_size must be positive.")
    if np.any(np.abs(np.diff(grid)) <= float(eps)):
        raise ValueError("physical_grid must not contain repeated adjacent nodes.")

    reference = _as_trajectory_tensor(reference_states)
    if reference.shape[0] != len(grid):
        raise ValueError("reference_states must have one state per physical_grid node.")
    replay_1x = _as_trajectory_tensor(replay_1x_endpoints).to(reference.device)
    replay_2x = _as_trajectory_tensor(replay_2x_endpoints).to(reference.device)
    replay_4x = _as_trajectory_tensor(replay_4x_endpoints).to(reference.device)

    safe_eps = float(eps)
    n_intervals = len(grid) - 1
    expected_replay_shape = (n_intervals,) + tuple(reference.shape[1:])
    for name, replay in {
        "replay_1x_endpoints": replay_1x,
        "replay_2x_endpoints": replay_2x,
        "replay_4x_endpoints": replay_4x,
    }.items():
        if tuple(replay.shape) != expected_replay_shape:
            raise ValueError(f"{name} must have shape [len(physical_grid)-1, batch, ...].")

    batch_count = int(reference.shape[1])
    interval_displacement = reference[1:] - reference[:-1]
    delta_s = np.maximum(_window_l2_norm(interval_displacement).T, safe_eps)
    q_lower = max(float(q_min), 1.0 + safe_eps)

    def finite_positive(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array) & (array > 0.0)]
        fallback = float(np.median(finite)) if finite.size else safe_eps
        return np.maximum(np.nan_to_num(array, nan=fallback, posinf=fallback, neginf=fallback), safe_eps)

    delta_s = finite_positive(delta_s)

    accumulated_defect = np.zeros((batch_count, n_intervals), dtype=np.float64)
    accumulated_q = np.zeros((batch_count, n_intervals), dtype=np.float64)
    coverage = np.zeros((batch_count, n_intervals), dtype=np.float64)
    window_residuals = np.zeros((batch_count, n_intervals), dtype=np.float64)
    window_delta_s = np.zeros((batch_count, n_intervals), dtype=np.float64)
    window_orders = np.zeros((batch_count, n_intervals), dtype=np.float64)

    max_window = min(int(window_size), n_intervals)
    for start in range(n_intervals):
        stop = min(start + max_window, n_intervals)
        node_stop = stop
        if node_stop <= start:
            continue

        tangent = reference[node_stop] - reference[start]

        full_residual = replay_1x[start] - replay_2x[start]
        half_residual = replay_2x[start] - replay_4x[start]
        residual_16, _ = project_residual_to_frenet_normal(
            full_residual,
            tangent,
            eps=safe_eps,
        )
        residual_32, _ = project_residual_to_frenet_normal(
            half_residual,
            tangent,
            eps=safe_eps,
        )
        full_error_norm = finite_positive(per_sample_l2_norm(full_residual).cpu().numpy())
        half_error_norm = finite_positive(per_sample_l2_norm(half_residual).cpu().numpy())
        residual_16_norm = finite_positive(per_sample_l2_norm(residual_16).cpu().numpy())
        residual_32_norm = finite_positive(per_sample_l2_norm(residual_32).cpu().numpy())
        q = np.clip(
            1.0 + np.log2((full_error_norm + safe_eps) / (half_error_norm + safe_eps)),
            q_lower,
            float(q_max),
        )
        q = np.clip(np.nan_to_num(q, nan=q_lower, posinf=float(q_max), neginf=q_lower), q_lower, float(q_max))
        ds_window = np.maximum(np.sum(delta_s[:, start:stop], axis=1), safe_eps)
        rho = np.maximum(np.abs(1.0 - np.power(2.0, 1.0 - q)), safe_eps)
        window_defect = finite_positive(residual_16_norm / (rho * np.power(ds_window + safe_eps, q) + safe_eps))

        weights = delta_s[:, start:stop] / ds_window[:, None]
        accumulated_defect[:, start:stop] += weights * window_defect[:, None]
        accumulated_q[:, start:stop] += weights * q[:, None]
        coverage[:, start:stop] += weights
        window_residuals[:, start] = residual_16_norm
        window_delta_s[:, start] = ds_window
        window_orders[:, start] = q

    missing = coverage <= safe_eps
    if np.any(missing):
        accumulated_defect[missing] = safe_eps
        accumulated_q[missing] = q_lower
        coverage[missing] = 1.0

    interval_defect = finite_positive(accumulated_defect / np.maximum(coverage, safe_eps))
    interval_q = np.clip(
        accumulated_q / np.maximum(coverage, safe_eps),
        q_lower,
        float(q_max),
    )
    interval_q = np.clip(
        np.nan_to_num(interval_q, nan=q_lower, posinf=float(q_max), neginf=q_lower),
        q_lower,
        float(q_max),
    )
    interval_rho = np.maximum(np.abs(1.0 - np.power(2.0, 1.0 - interval_q)), safe_eps)
    residual_perp = finite_positive(interval_defect * interval_rho * np.power(delta_s + safe_eps, interval_q))
    half_error = finite_positive(residual_perp / np.maximum(np.power(2.0, interval_q - 1.0), safe_eps))

    stats = FPTrajectoryStats(
        full_step_error=residual_perp.copy(),
        half_step_error=np.maximum(half_error, safe_eps),
        effective_order=interval_q,
        delta_s=delta_s,
        residual_perp_norm=residual_perp,
    )
    details = FPAnchoredReplayDetails(
        window_size=max_window,
        window_residual_perp_norm=np.maximum(window_residuals, safe_eps),
        window_delta_s=finite_positive(window_delta_s),
        window_effective_order=np.maximum(
            np.nan_to_num(window_orders, nan=q_lower, posinf=float(q_max), neginf=q_lower),
            q_lower,
        ),
        coverage=coverage,
    )
    return stats, details


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


def _build_interval_mass_profile(
    physical_grid: np.ndarray,
    interval_alpha: np.ndarray,
    *,
    eps: float,
) -> ClockProfile:
    grid = np.asarray(physical_grid, dtype=np.float64)
    alpha = np.asarray(interval_alpha, dtype=np.float64)
    if alpha.ndim != 1 or len(alpha) != len(grid) - 1:
        raise ValueError("interval_alpha must have length len(physical_grid) - 1.")

    node_alpha = _interval_profile_to_nodes(alpha)
    path_coordinate = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(grid)))])
    total_length = float(path_coordinate[-1])
    if total_length <= 0.0:
        raise ValueError("physical_grid must span a non-zero interval.")
    path_coordinate = path_coordinate / total_length
    interval_width = np.diff(path_coordinate)
    interval_mass = np.maximum(alpha, float(eps)) * interval_width
    interval_mass = interval_mass / np.maximum(float(np.sum(interval_mass)), float(eps))
    tau_profile = np.concatenate([[0.0], np.cumsum(interval_mass)])
    tau_profile[0] = 0.0
    tau_profile[-1] = 1.0

    density_integral = np.trapezoid(node_alpha, path_coordinate)
    if density_integral <= 0.0:
        raise ValueError("FP_CLOCK produced an invalid density profile.")
    density = node_alpha / density_integral
    return ClockProfile(
        physical_grid=grid,
        alpha_profile=node_alpha,
        density=density,
        tau_profile=tau_profile,
    )


def _evaluate_velocity(
    velocity_fn: VelocityFn,
    sample: torch.Tensor,
    coordinate: float,
    sample_start: int,
    sample_stop: int,
) -> torch.Tensor:
    coordinate_tensor = torch.as_tensor(float(coordinate), device=sample.device, dtype=torch.float32)
    return _call_velocity(velocity_fn, sample, coordinate_tensor, sample_start, sample_stop)


def collect_fp_clock_stats(
    *,
    initial_sample: torch.Tensor,
    physical_grid: np.ndarray,
    velocity_fn: VelocityFn,
    step_fn: StepFn,
    observation_microbatch: int | None = None,
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> FPTrajectoryStats:
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
    residual_perp_values: list[np.ndarray] = []

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
        delta_s = 0.5 * (speed_start + speed_end) * abs(delta_u)
        residual_perp, _ = project_residual_to_frenet_normal(full - half, velocity_mid, eps=safe_eps)

        full_errors.append(per_sample_l2_norm(full - half).cpu().numpy())
        half_errors.append(per_sample_l2_norm(half - quarter).cpu().numpy())
        delta_s_values.append(delta_s.cpu().numpy())
        residual_perp_values.append(per_sample_l2_norm(residual_perp).cpu().numpy())

        current = next_sample
        velocity_start = velocity_end

    full_error = np.stack(full_errors, axis=1)
    half_error = np.stack(half_errors, axis=1)
    effective_order, _ = estimate_refinement_order_and_defect(
        full_step_error=full_error,
        half_step_error=half_error,
        step_sizes=np.abs(np.diff(grid)),
        q_min=q_min,
        q_max=q_max,
        eps=safe_eps,
    )
    return FPTrajectoryStats(
        full_step_error=full_error,
        half_step_error=half_error,
        effective_order=effective_order,
        delta_s=np.maximum(np.stack(delta_s_values, axis=1), safe_eps),
        residual_perp_norm=np.maximum(np.stack(residual_perp_values, axis=1), safe_eps),
    )


def concatenate_fp_clock_stats(items: Sequence[FPTrajectoryStats]) -> FPTrajectoryStats:
    if not items:
        raise ValueError("items must contain at least one FPTrajectoryStats object.")
    return FPTrajectoryStats(
        full_step_error=np.concatenate([item.full_step_error for item in items], axis=0),
        half_step_error=np.concatenate([item.half_step_error for item in items], axis=0),
        effective_order=np.concatenate([item.effective_order for item in items], axis=0),
        delta_s=np.concatenate([item.delta_s for item in items], axis=0),
        residual_perp_norm=np.concatenate([item.residual_perp_norm for item in items], axis=0),
    )


def build_fp_clock_profile(
    physical_grid: np.ndarray,
    stats: FPTrajectoryStats,
    *,
    target_steps: int | None = None,
    eps: float = 1.0e-12,
    q_min: float = 1.05,
    q_max: float = 6.0,
    smoothing_window: int = 1,
) -> FPClockArtifacts:
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")
    if target_steps is not None and int(target_steps) <= 0:
        raise ValueError("target_steps must be positive when provided.")

    safe_eps = float(eps)
    expected_shape = (stats.delta_s.shape[0], len(grid) - 1)
    for name, values in {
        "effective_order": stats.effective_order,
        "delta_s": stats.delta_s,
        "residual_perp_norm": stats.residual_perp_norm,
    }.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape [num_trajectories, len(physical_grid) - 1].")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf.")

    delta_s = np.maximum(np.asarray(stats.delta_s, dtype=np.float64), safe_eps)
    q = np.clip(np.asarray(stats.effective_order, dtype=np.float64), max(float(q_min), 1.0 + safe_eps), float(q_max))
    residual_perp = np.maximum(np.asarray(stats.residual_perp_norm, dtype=np.float64), safe_eps)

    rho = np.maximum(np.abs(1.0 - np.power(2.0, 1.0 - q)), safe_eps)
    arc_defect = residual_perp / (np.power(delta_s + safe_eps, q) * rho + safe_eps)
    defect_weight = np.exp(
        (np.log(np.maximum(q - 1.0, safe_eps)) + np.log(np.maximum(arc_defect, safe_eps))) / q
    )
    arc_density = defect_weight / np.maximum(np.sum(defect_weight * delta_s, axis=1, keepdims=True), safe_eps)

    delta_u = np.maximum(np.abs(np.diff(grid))[None, :], safe_eps)
    pulled_alpha = arc_density * delta_s / (delta_u + safe_eps)
    interval_alpha = np.maximum(np.mean(pulled_alpha, axis=0), safe_eps)
    if int(smoothing_window) > 1:
        interval_alpha = np.exp(smooth_profile(np.log(interval_alpha), int(smoothing_window)))
        interval_alpha = np.maximum(interval_alpha, safe_eps)

    if not np.all(np.isfinite(interval_alpha)) or np.any(interval_alpha <= 0.0):
        raise ValueError("FP_CLOCK produced an invalid alpha profile.")

    profile = _build_interval_mass_profile(grid, interval_alpha, eps=safe_eps)
    defect_profile = np.mean(arc_defect, axis=0)
    smoothed_defect = np.exp(smooth_profile(np.log(np.maximum(defect_profile, safe_eps)), int(smoothing_window)))
    effective_order_profile = np.mean(q, axis=0)
    smoothed_order = smooth_profile(effective_order_profile, int(smoothing_window))
    return FPClockArtifacts(
        profile=profile,
        defect_profile=defect_profile,
        smoothed_defect_profile=smoothed_defect,
        effective_order_profile=effective_order_profile,
        smoothed_effective_order_profile=smoothed_order,
        interval_alpha_profile=interval_alpha,
        arc_length_profile=np.mean(delta_s, axis=0),
        residual_perp_profile=np.mean(residual_perp, axis=0),
    )

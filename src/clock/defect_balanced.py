from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.clock.profile import ClockProfile, build_clock_profile_from_alpha

StepFn = Callable[[torch.Tensor, float, float], torch.Tensor]
VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
DEFECT_BALANCED_CLOCK_VERSION = 1


@dataclass(frozen=True)
class StepRefinementStats:
    full_step_error: np.ndarray
    half_step_error: np.ndarray
    effective_order: np.ndarray
    defect_strength: np.ndarray


@dataclass(frozen=True)
class DefectBalancedProfileArtifacts:
    profile: ClockProfile
    defect_profile: np.ndarray
    smoothed_defect_profile: np.ndarray
    effective_order_profile: np.ndarray
    smoothed_effective_order_profile: np.ndarray
    interval_alpha_profile: np.ndarray


def smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("Expected a 1D profile for smoothing.")
    width = max(int(window), 1)
    if width <= 1 or len(data) <= 2:
        return data.copy()
    if width % 2 == 0:
        width += 1
    radius = width // 2
    padded = np.pad(data, (radius, radius), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def per_sample_l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(tensor.shape[0], -1).norm(dim=1)


def _microbatch_map(
    sample: torch.Tensor,
    *,
    microbatch_size: int | None,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if microbatch_size is None or microbatch_size <= 0 or microbatch_size >= sample.shape[0]:
        return fn(sample)
    outputs: list[torch.Tensor] = []
    for start in range(0, sample.shape[0], microbatch_size):
        stop = min(start + microbatch_size, sample.shape[0])
        outputs.append(fn(sample[start:stop]))
    return torch.cat(outputs, dim=0)


def euler_step(
    velocity_fn: VelocityFn,
    sample: torch.Tensor,
    coordinate_start: float,
    coordinate_end: float,
) -> torch.Tensor:
    step = float(coordinate_end) - float(coordinate_start)
    coordinate = torch.as_tensor(float(coordinate_start), device=sample.device, dtype=sample.dtype)
    return sample + velocity_fn(sample, coordinate) * step


def heun2_step(
    velocity_fn: VelocityFn,
    sample: torch.Tensor,
    coordinate_start: float,
    coordinate_end: float,
) -> torch.Tensor:
    step = float(coordinate_end) - float(coordinate_start)
    start = torch.as_tensor(float(coordinate_start), device=sample.device, dtype=sample.dtype)
    end = torch.as_tensor(float(coordinate_end), device=sample.device, dtype=sample.dtype)
    velocity_start = velocity_fn(sample, start)
    predicted = sample + velocity_start * step
    velocity_end = velocity_fn(predicted, end)
    return sample + 0.5 * (velocity_start + velocity_end) * step


def build_velocity_stepper(velocity_fn: VelocityFn, method: str) -> StepFn:
    normalized = str(method).lower().replace("-", "_")
    if normalized in {"euler", "flow_euler"}:
        return lambda sample, start, end: euler_step(velocity_fn, sample, start, end)
    if normalized in {"heun2", "flow_heun"}:
        return lambda sample, start, end: heun2_step(velocity_fn, sample, start, end)
    raise ValueError(f"Unsupported velocity-based refinement solver: {method}")


def _refined_step(step_fn: StepFn, sample: torch.Tensor, start: float, end: float, pieces: int) -> torch.Tensor:
    if pieces < 1:
        raise ValueError("pieces must be positive.")
    current = sample
    nodes = np.linspace(float(start), float(end), int(pieces) + 1, dtype=np.float64)
    for index in range(int(pieces)):
        current = step_fn(current, float(nodes[index]), float(nodes[index + 1]))
    return current


def estimate_refinement_order_and_defect(
    *,
    full_step_error: np.ndarray,
    half_step_error: np.ndarray,
    step_sizes: np.ndarray,
    q_min: float = 0.25,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(full_step_error, dtype=np.float64)
    half = np.asarray(half_step_error, dtype=np.float64)
    steps = np.asarray(step_sizes, dtype=np.float64)
    if full.shape != half.shape:
        raise ValueError("full_step_error and half_step_error must have matching shapes.")
    if full.ndim != 2:
        raise ValueError("step-refinement errors must have shape [num_trajectories, num_intervals].")
    if steps.ndim != 1 or steps.shape[0] != full.shape[1]:
        raise ValueError("step_sizes must have one value per interval.")
    if q_min <= 0.0 or q_max < q_min:
        raise ValueError("q_min and q_max must satisfy 0 < q_min <= q_max.")

    safe_eps = float(eps)
    raw_order = np.log2((full + safe_eps) / (half + safe_eps))
    effective_order = np.clip(raw_order, float(q_min), float(q_max))

    step_scale = np.power(np.maximum(np.abs(steps), safe_eps)[None, :], effective_order + 1.0)
    refinement_factor = np.maximum(1.0 - np.power(2.0, -effective_order), safe_eps)
    defect = full / np.maximum(step_scale * refinement_factor, safe_eps)
    return effective_order, np.maximum(defect, safe_eps)


def collect_step_refinement_stats(
    *,
    initial_sample: torch.Tensor,
    physical_grid: np.ndarray,
    step_fn: StepFn,
    observation_microbatch: int | None = None,
    q_min: float = 0.25,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> StepRefinementStats:
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")

    current = initial_sample.detach()
    full_errors: list[np.ndarray] = []
    half_errors: list[np.ndarray] = []

    for index in range(len(grid) - 1):
        start = float(grid[index])
        end = float(grid[index + 1])
        full = _microbatch_map(
            current,
            microbatch_size=observation_microbatch,
            fn=lambda batch, s=start, e=end: _refined_step(step_fn, batch, s, e, 1),
        )
        half = _microbatch_map(
            current,
            microbatch_size=observation_microbatch,
            fn=lambda batch, s=start, e=end: _refined_step(step_fn, batch, s, e, 2),
        )
        quarter = _microbatch_map(
            current,
            microbatch_size=observation_microbatch,
            fn=lambda batch, s=start, e=end: _refined_step(step_fn, batch, s, e, 4),
        )
        full_errors.append(per_sample_l2_norm(full - half).cpu().numpy())
        half_errors.append(per_sample_l2_norm(half - quarter).cpu().numpy())
        current = full.detach().clone()

    full_error = np.stack(full_errors, axis=1)
    half_error = np.stack(half_errors, axis=1)
    effective_order, defect_strength = estimate_refinement_order_and_defect(
        full_step_error=full_error,
        half_step_error=half_error,
        step_sizes=np.diff(grid),
        q_min=q_min,
        q_max=q_max,
        eps=eps,
    )
    return StepRefinementStats(
        full_step_error=full_error,
        half_step_error=half_error,
        effective_order=effective_order,
        defect_strength=defect_strength,
    )


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


def build_defect_balanced_profile(
    physical_grid: np.ndarray,
    stats: StepRefinementStats,
    *,
    smoothing_window: int = 1,
    eps: float = 1.0e-12,
) -> DefectBalancedProfileArtifacts:
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")

    defect = np.asarray(stats.defect_strength, dtype=np.float64)
    order = np.asarray(stats.effective_order, dtype=np.float64)
    expected_shape = (defect.shape[0], len(grid) - 1)
    if defect.ndim != 2 or defect.shape[1] != len(grid) - 1:
        raise ValueError("defect_strength must have shape [num_trajectories, len(physical_grid) - 1].")
    if order.shape != expected_shape:
        raise ValueError("effective_order must match defect_strength shape.")

    safe_eps = float(eps)
    defect_profile = np.sqrt(np.mean(np.square(np.maximum(defect, safe_eps)), axis=0))
    effective_order_profile = np.mean(order, axis=0)

    smoothed_log_defect = smooth_profile(np.log(np.maximum(defect_profile, safe_eps)), smoothing_window)
    smoothed_order = smooth_profile(effective_order_profile, smoothing_window)
    smoothed_defect = np.exp(smoothed_log_defect)
    interval_alpha = np.exp(smoothed_log_defect / np.maximum(smoothed_order + 1.0, safe_eps))
    node_alpha = _interval_profile_to_nodes(np.maximum(interval_alpha, safe_eps))
    profile = build_clock_profile_from_alpha(grid, node_alpha)
    return DefectBalancedProfileArtifacts(
        profile=profile,
        defect_profile=defect_profile,
        smoothed_defect_profile=smoothed_defect,
        effective_order_profile=effective_order_profile,
        smoothed_effective_order_profile=smoothed_order,
        interval_alpha_profile=interval_alpha,
    )

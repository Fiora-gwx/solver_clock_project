from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.clock.va import SharedClockProfile, build_clock_profile_from_alpha

LCS_SCHEDULE_IMPLEMENTATION_VERSION = 7

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class LcsProfileArtifacts:
    profile: SharedClockProfile
    kappa_profile: np.ndarray
    smoothed_kappa_profile: np.ndarray


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


def material_derivative_jvp(
    velocity_fn: VelocityFn,
    sample: torch.Tensor,
    coordinate: float,
) -> torch.Tensor:
    with torch.enable_grad():
        sample_input = sample.detach().clone().requires_grad_(True)
        coordinate_input = torch.as_tensor(float(coordinate), device=sample.device, dtype=sample.dtype).requires_grad_(True)

        def wrapped(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return velocity_fn(x, c)

        base_velocity = wrapped(sample_input, coordinate_input)
        _, derivative = torch.autograd.functional.jvp(
            wrapped,
            (sample_input, coordinate_input),
            (base_velocity.detach(), torch.ones_like(coordinate_input)),
            create_graph=False,
            strict=False,
        )
    return derivative.detach()


def euler_step(
    velocity_fn: VelocityFn,
    sample: torch.Tensor,
    coordinate_start: float,
    coordinate_end: float,
) -> torch.Tensor:
    dt = float(coordinate_end) - float(coordinate_start)
    velocity = velocity_fn(sample, torch.as_tensor(float(coordinate_start), device=sample.device, dtype=sample.dtype))
    return sample + velocity * dt


def heun2_step(
    velocity_fn: VelocityFn,
    sample: torch.Tensor,
    coordinate_start: float,
    coordinate_end: float,
) -> torch.Tensor:
    dt = float(coordinate_end) - float(coordinate_start)
    start = torch.as_tensor(float(coordinate_start), device=sample.device, dtype=sample.dtype)
    end = torch.as_tensor(float(coordinate_end), device=sample.device, dtype=sample.dtype)
    velocity_start = velocity_fn(sample, start)
    predicted = sample + velocity_start * dt
    velocity_end = velocity_fn(predicted, end)
    return sample + 0.5 * (velocity_start + velocity_end) * dt


def build_stepper(velocity_fn: VelocityFn, method: str) -> Callable[[torch.Tensor, float, float], torch.Tensor]:
    normalized = str(method).lower().replace("-", "_")
    if normalized in {"euler", "flow_euler"}:
        return lambda sample, start, end: euler_step(velocity_fn, sample, start, end)
    if normalized in {"heun2", "flow_heun"}:
        return lambda sample, start, end: heun2_step(velocity_fn, sample, start, end)
    raise ValueError(
        f"Unsupported generic LCS pilot solver: {method}. "
        "Native solver pilots such as STORK must be collected through the backend adapter."
    )


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


def collect_lcs_norms(
    *,
    initial_sample: torch.Tensor,
    physical_grid: np.ndarray,
    velocity_fn: VelocityFn,
    pilot_solver: str,
    order: int,
    observation_microbatch: int | None = None,
) -> np.ndarray:
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")
    if order != 1:
        raise ValueError("LCS-2 step-doubling profiles are disabled; only LCS-1 is supported.")

    stepper = build_stepper(velocity_fn, pilot_solver)
    current = initial_sample.detach()
    norms: list[np.ndarray] = []

    if order == 1:
        for index in range(len(grid)):
            derivative = _microbatch_map(
                current,
                microbatch_size=observation_microbatch,
                fn=lambda batch, coordinate=float(grid[index]): material_derivative_jvp(velocity_fn, batch, coordinate),
            )
            norms.append(per_sample_l2_norm(derivative).cpu().numpy())
            if index + 1 < len(grid):
                with torch.inference_mode():
                    current = _microbatch_map(
                        current,
                        microbatch_size=observation_microbatch,
                        fn=lambda batch, start=float(grid[index]), end=float(grid[index + 1]): stepper(batch, start, end),
                    )
                    current = current.detach().clone()
        return np.stack(norms, axis=1)

    raise AssertionError("unreachable")


def collect_velocity_norms(
    *,
    initial_sample: torch.Tensor,
    physical_grid: np.ndarray,
    velocity_fn: VelocityFn,
    pilot_solver: str,
    observation_microbatch: int | None = None,
) -> np.ndarray:
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")

    stepper = build_stepper(velocity_fn, pilot_solver)
    current = initial_sample.detach()
    norms: list[np.ndarray] = []
    for index in range(len(grid)):
        velocity = _microbatch_map(
            current,
            microbatch_size=observation_microbatch,
            fn=lambda batch, coordinate=float(grid[index]): velocity_fn(
                batch,
                torch.as_tensor(coordinate, device=batch.device, dtype=batch.dtype),
            ),
        )
        norms.append(per_sample_l2_norm(velocity).cpu().numpy())
        if index + 1 < len(grid):
            with torch.inference_mode():
                current = _microbatch_map(
                    current,
                    microbatch_size=observation_microbatch,
                    fn=lambda batch, start=float(grid[index]), end=float(grid[index + 1]): stepper(batch, start, end),
                )
    return np.stack(norms, axis=1)


def build_lcs_profile(
    physical_grid: np.ndarray,
    defect_norms: np.ndarray,
    *,
    order: int,
    smoothing_window: int = 1,
    eps: float = 1.0e-6,
) -> LcsProfileArtifacts:
    grid = np.asarray(physical_grid, dtype=np.float64)
    norms = np.asarray(defect_norms, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")
    if norms.ndim != 2 or norms.shape[1] != len(grid):
        raise ValueError("defect_norms must have shape [num_trajectories, len(physical_grid)].")
    if order != 1:
        raise ValueError("LCS-2 step-doubling profiles are disabled; only LCS-1 is supported.")

    kappa_profile = np.sqrt(np.mean(np.square(norms), axis=0))
    smoothed_kappa = smooth_profile(kappa_profile, smoothing_window)
    smoothed_kappa = np.maximum(smoothed_kappa, float(eps))
    alpha_profile = np.power(smoothed_kappa, 1.0 / float(order + 1))
    profile = build_clock_profile_from_alpha(grid, alpha_profile)
    return LcsProfileArtifacts(
        profile=profile,
        kappa_profile=kappa_profile,
        smoothed_kappa_profile=smoothed_kappa,
    )

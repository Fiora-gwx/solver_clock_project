from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from src.utils.nfe_budget import EffectiveNfePlan, resolve_effective_nfe_plan
from src.utils.schedule_bundle import ScheduleBundle


@dataclass(frozen=True)
class SharedClockProfile:
    physical_grid: np.ndarray
    alpha_profile: np.ndarray
    density: np.ndarray
    tau_profile: np.ndarray


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    areas = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(areas)])


def _path_coordinate(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("grid must be a 1D array with at least two points.")
    coord = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(values)))])
    total = float(coord[-1])
    if total <= 0.0:
        raise ValueError("grid must span a non-zero range.")
    return coord / total


def _normalize_density_from_profile(alpha_profile: np.ndarray, physical_grid: np.ndarray) -> np.ndarray:
    coordinate = _path_coordinate(physical_grid)
    integral = np.trapz(alpha_profile, coordinate)
    if integral <= 0:
        raise ValueError("Shared-clock alpha profile must have positive integral.")
    return alpha_profile / integral


def build_clock_profile_from_alpha(
    physical_grid: np.ndarray,
    alpha_profile: np.ndarray,
) -> SharedClockProfile:
    grid = np.asarray(physical_grid, dtype=np.float64)
    alpha = np.asarray(alpha_profile, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")
    if alpha.ndim != 1 or len(alpha) != len(grid):
        raise ValueError("alpha_profile must be a 1D array with the same length as physical_grid.")
    if np.any(alpha < 0.0):
        raise ValueError("alpha_profile must be non-negative.")

    density = _normalize_density_from_profile(alpha, grid)
    tau_profile = _cumulative_trapezoid(density, _path_coordinate(grid))
    tau_profile = tau_profile / tau_profile[-1]
    tau_profile[0] = 0.0
    tau_profile[-1] = 1.0
    return SharedClockProfile(
        physical_grid=grid,
        alpha_profile=alpha,
        density=density,
        tau_profile=tau_profile,
    )


def build_shared_clock_profile(
    physical_grid: np.ndarray,
    material_derivative_norms: np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> SharedClockProfile:
    grid = np.asarray(physical_grid, dtype=np.float64)
    norms = np.asarray(material_derivative_norms, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must be a 1D array with at least two points.")
    if norms.ndim != 2 or norms.shape[1] != len(grid):
        raise ValueError("material_derivative_norms must have shape [num_trajectories, len(physical_grid)].")

    kappa_profile = np.sqrt(np.mean(np.square(norms), axis=0))
    alpha_profile = np.sqrt(np.maximum(kappa_profile, float(eps)))
    return build_clock_profile_from_alpha(grid, alpha_profile)


def slice_profile_interval(
    profile: SharedClockProfile,
    coordinate_start: float,
    coordinate_end: float,
    *,
    atol: float = 1.0e-12,
) -> SharedClockProfile:
    grid = np.asarray(profile.physical_grid, dtype=np.float64)
    alpha = np.asarray(profile.alpha_profile, dtype=np.float64)
    if grid.ndim != 1 or alpha.ndim != 1 or len(grid) != len(alpha):
        raise ValueError("SharedClockProfile must contain aligned 1D physical_grid and alpha_profile arrays.")
    if len(grid) < 2:
        raise ValueError("SharedClockProfile must contain at least two grid points.")
    if np.isclose(coordinate_start, coordinate_end, atol=atol, rtol=0.0):
        raise ValueError("coordinate_start and coordinate_end must span a non-zero interval.")

    increasing_grid = grid[0] <= grid[-1]
    xp = grid if increasing_grid else grid[::-1]
    fp = alpha if increasing_grid else alpha[::-1]

    low = float(min(coordinate_start, coordinate_end))
    high = float(max(coordinate_start, coordinate_end))
    if low < float(xp[0]) - atol or high > float(xp[-1]) + atol:
        raise ValueError(
            f"Requested interval [{coordinate_start}, {coordinate_end}] lies outside the profile support "
            f"[{float(grid[0])}, {float(grid[-1])}]."
        )

    clipped_start = float(np.clip(coordinate_start, float(xp[0]), float(xp[-1])))
    clipped_end = float(np.clip(coordinate_end, float(xp[0]), float(xp[-1])))
    start_alpha = float(np.interp(clipped_start, xp, fp))
    end_alpha = float(np.interp(clipped_end, xp, fp))

    inner_mask = (xp > low + atol) & (xp < high - atol)
    cropped_grid = np.concatenate(
        [
            np.asarray([low], dtype=np.float64),
            xp[inner_mask],
            np.asarray([high], dtype=np.float64),
        ]
    )
    cropped_alpha = np.concatenate(
        [
            np.asarray([float(np.interp(low, xp, fp))], dtype=np.float64),
            fp[inner_mask],
            np.asarray([float(np.interp(high, xp, fp))], dtype=np.float64),
        ]
    )

    if clipped_start > clipped_end:
        cropped_grid = cropped_grid[::-1]
        cropped_alpha = cropped_alpha[::-1]

    cropped_grid[0] = clipped_start
    cropped_grid[-1] = clipped_end
    cropped_alpha[0] = start_alpha
    cropped_alpha[-1] = end_alpha
    return build_clock_profile_from_alpha(cropped_grid, cropped_alpha)


def materialize_schedule_nodes(
    profile: SharedClockProfile,
    plan: EffectiveNfePlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_count = plan.solver_steps
    tau_grid = np.linspace(0.0, 1.0, step_count + 1, dtype=np.float64)
    physical_nodes = np.interp(tau_grid, profile.tau_profile, profile.physical_grid)
    if profile.physical_grid[0] <= profile.physical_grid[-1]:
        xp = profile.physical_grid
        fp = profile.density
    else:
        xp = profile.physical_grid[::-1]
        fp = profile.density[::-1]
    density_nodes = np.interp(physical_nodes, xp, fp)
    g_grid = 1.0 / np.maximum(density_nodes, 1.0e-12)
    return tau_grid, physical_nodes, g_grid


def build_reparameterized_bundle(
    profile: SharedClockProfile,
    *,
    effective_nfe: int,
    solver_name: str,
    representation: str,
    schedule_family: str,
    meta: dict[str, object] | None = None,
    representation_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    time_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ScheduleBundle:
    plan = resolve_effective_nfe_plan(solver_name, effective_nfe)
    tau_grid, physical_nodes, g_grid = materialize_schedule_nodes(profile, plan)
    representation_grid = (
        np.asarray(representation_transform(physical_nodes), dtype=np.float64)
        if representation_transform is not None
        else physical_nodes.copy()
    )
    if representation_grid.shape != physical_nodes.shape:
        raise ValueError("representation_transform must preserve the physical node shape.")

    transformed_time_grid = (
        np.asarray(time_transform(physical_nodes), dtype=np.float64) if time_transform is not None else None
    )
    if transformed_time_grid is not None and transformed_time_grid.shape != physical_nodes.shape:
        raise ValueError("time_transform must preserve the physical node shape.")

    payload = dict(meta or {})
    payload.update(plan.to_meta())
    payload.setdefault("nfe", effective_nfe)
    payload.setdefault("representation", representation)
    payload.setdefault("schedule_family", schedule_family)
    payload.setdefault("dtau", float(1.0 / plan.solver_steps))
    # For scheduler compatibility, we export the current-step anchors and keep the full node grid alongside them.
    anchor_grid = representation_grid[:-1].copy()
    bundle_kwargs: dict[str, object] = {
        "tau_grid": tau_grid,
        "g_grid": g_grid,
        "meta": payload,
    }
    if representation == "timesteps":
        if transformed_time_grid is None:
            transformed_time_grid = representation_grid.copy()
        bundle_kwargs["time_grid"] = transformed_time_grid
        bundle_kwargs["timesteps"] = anchor_grid
    elif representation == "sigmas":
        if transformed_time_grid is not None:
            bundle_kwargs["timesteps"] = transformed_time_grid[:-1].copy()
            bundle_kwargs["time_grid"] = transformed_time_grid
        bundle_kwargs["sigmas"] = anchor_grid
        bundle_kwargs["sigma_grid"] = representation_grid
    else:
        raise ValueError(f"Unsupported reparameterized representation: {representation}")
    return ScheduleBundle(**bundle_kwargs)


def export_shared_clock_sweep(
    profile: SharedClockProfile,
    target_nfes: Iterable[int],
    *,
    output_root: str | Path,
    solver_name: str,
    representation: str,
    schedule_family: str,
    meta: dict[str, object] | None = None,
    representation_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    time_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[Path]:
    exported: list[Path] = []
    for effective_nfe in target_nfes:
        bundle = build_reparameterized_bundle(
            profile,
            effective_nfe=int(effective_nfe),
            solver_name=solver_name,
            representation=representation,
            schedule_family=schedule_family,
            meta=meta,
            representation_transform=representation_transform,
            time_transform=time_transform,
        )
        output_dir = Path(output_root) / f"nfe_{int(effective_nfe):03d}"
        exported.append(bundle.save(output_dir))
    return exported

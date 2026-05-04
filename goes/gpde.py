from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .aggregation import robust_aggregate
from .metrics import Metric
from .mixed_defect import mixed_normal_defect_sq
from .oracle import OracleData


@dataclass
class GPDEProfile:
    probe_grid: np.ndarray
    probe_steps: np.ndarray
    defects: np.ndarray
    coefficient_per_sample: np.ndarray
    aggregate_coefficient: np.ndarray
    monitor_density: np.ndarray
    q_estimate: float
    q_source: str
    fallback_counts: np.ndarray
    fallback_fraction: float
    metadata: dict[str, Any]


@dataclass
class GPDESchedule:
    u_schedule: np.ndarray
    selected_indices: list[int]
    interval_monitor_masses: np.ndarray
    objective: float
    total_monitor_mass: float
    snap_errors: np.ndarray
    metadata: dict[str, Any]


def default_q_for_solver(solver_name: str) -> float:
    normalized = str(solver_name).lower().replace("-", "_")
    if normalized in {"heun", "heun2", "midpoint", "flow_heun"}:
        return 6.0
    return 4.0


def parse_float_list(value: Any, *, default: tuple[float, ...]) -> list[float]:
    if value is None:
        return [float(item) for item in default]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [float(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def make_probe_grid(u_min: float, u_max: float, size: int) -> np.ndarray:
    if int(size) < 2:
        raise ValueError("probe grid size must be at least 2.")
    return np.linspace(float(u_min), float(u_max), int(size), dtype=np.float64)


def make_probe_steps(
    *,
    u_min: float,
    u_max: float,
    probe_grid_size: int,
    multipliers: Any = None,
    absolute_steps: Any = None,
) -> np.ndarray:
    if absolute_steps is not None:
        steps = np.asarray(parse_float_list(absolute_steps, default=()), dtype=np.float64)
    else:
        factors = np.asarray(parse_float_list(multipliers, default=(1.0, 2.0, 4.0)), dtype=np.float64)
        base = (float(u_max) - float(u_min)) / max(int(probe_grid_size) - 1, 1)
        steps = factors * base
    steps = np.unique(np.asarray(steps, dtype=np.float64))
    steps = steps[np.isfinite(steps) & (steps > 0.0)]
    if steps.size == 0:
        raise ValueError("At least one positive probe step is required.")
    return steps


def _fill_missing_profile(values: np.ndarray, *, floor: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(result) & (result > 0.0)
    if not np.any(finite):
        return np.full_like(result, float(floor), dtype=np.float64)
    indices = np.arange(result.size)
    result[~finite] = np.interp(indices[~finite], indices[finite], result[finite])
    return np.maximum(result, float(floor))


def _smooth_positive_profile(values: np.ndarray, window: int, *, floor: float) -> np.ndarray:
    data = np.maximum(np.asarray(values, dtype=np.float64), float(floor))
    width = max(int(window), 1)
    if width <= 1 or data.size <= 2:
        return data.copy()
    if width % 2 == 0:
        width += 1
    radius = width // 2
    padded = np.pad(np.log(data), (radius, radius), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.exp(np.convolve(padded, kernel, mode="valid"))


def estimate_global_q(
    defects: np.ndarray,
    probe_steps: np.ndarray,
    *,
    aggregation: dict[str, Any] | str,
    default_q: float,
    min_q: float = 0.25,
    max_q: float = 12.0,
) -> tuple[float, str]:
    values = np.asarray(defects, dtype=np.float64)
    steps = np.asarray(probe_steps, dtype=np.float64)
    xs: list[float] = []
    ys: list[float] = []
    for step_index, eta in enumerate(steps):
        samples = values[:, step_index, :].reshape(-1)
        samples = samples[np.isfinite(samples) & (samples > 0.0)]
        if samples.size == 0:
            continue
        try:
            aggregated = robust_aggregate(samples, aggregation)
        except ValueError:
            continue
        if np.isfinite(aggregated) and aggregated > 0.0:
            xs.append(float(np.log(eta)))
            ys.append(float(np.log(aggregated)))
    if len(set(xs)) < 2:
        return float(default_q), "default_insufficient_probe_steps"
    slope, _intercept = np.polyfit(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), deg=1)
    if not np.isfinite(slope):
        return float(default_q), "default_nonfinite_fit"
    return float(np.clip(slope, float(min_q), float(max_q))), "global_loglog_fit"


def evaluate_gpde_profile(
    solver: Any,
    oracle: OracleData,
    probe_grid: np.ndarray,
    probe_steps: np.ndarray,
    metric: Metric,
    *,
    rho: float = 0.1,
    aggregation: dict[str, Any] | str = "cvar",
    q_mode: str = "global_fit",
    fixed_q: float | None = None,
    default_q: float = 4.0,
    min_q: float = 0.25,
    max_q: float = 12.0,
    coefficient_floor: float = 1.0e-12,
    monitor_smoothing_window: int = 3,
    monitor_exponent: str = "q_root",
    eps: float = 1.0e-12,
    fallback_full_residual_on_tiny_tangent: bool = True,
) -> GPDEProfile:
    grid = np.asarray(probe_grid, dtype=np.float64)
    steps = np.asarray(probe_steps, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("probe_grid must be a one-dimensional array with at least two points.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("probe_grid must be strictly increasing in unified coordinate u.")
    if steps.ndim != 1 or steps.size < 1 or np.any(steps <= 0.0):
        raise ValueError("probe_steps must contain at least one positive step.")

    K = oracle.num_samples
    defects = np.full((grid.size, steps.size, K), np.nan, dtype=np.float64)
    fallback_counts = np.zeros((grid.size, steps.size), dtype=np.int64)
    fallback_possible = 0
    fallback_total = 0
    u_max = float(grid[-1])
    for index, u in enumerate(grid):
        x_a = oracle.state_at(float(u))
        for step_index, eta in enumerate(steps):
            b = float(u + eta)
            if b > u_max + 1.0e-12:
                continue
            b = min(b, u_max)
            x_b = oracle.state_at(b)
            v_b = oracle.tangent_at(b)
            x_hat = solver.single_edge_step_from_state(x_a.copy(), float(u), b, oracle.conditions)
            residual = x_hat - x_b
            defect = mixed_normal_defect_sq(
                residual,
                v_b,
                metric,
                b,
                rho=rho,
                eps=eps,
                fallback_full_residual_on_tiny_tangent=fallback_full_residual_on_tiny_tangent,
            )
            defects[index, step_index, :] = defect.values
            count = int(np.count_nonzero(defect.fallback_mask))
            fallback_counts[index, step_index] = count
            fallback_total += count
            fallback_possible += defect.fallback_mask.size

    mode = str(q_mode).lower().replace("-", "_")
    if mode == "fixed":
        if fixed_q is None or not np.isfinite(float(fixed_q)) or float(fixed_q) <= 0.0:
            raise ValueError("fixed q estimation requires a positive fixed_q value.")
        q_estimate = float(fixed_q)
        q_source = "fixed"
    else:
        q_estimate, q_source = estimate_global_q(
            defects,
            steps,
            aggregation=aggregation,
            default_q=default_q,
            min_q=min_q,
            max_q=max_q,
        )

    coeff_steps = defects / np.reshape(steps**q_estimate, (1, -1, 1))
    coefficient_per_sample = np.full((grid.size, K), np.nan, dtype=np.float64)
    for index in range(grid.size):
        valid = coeff_steps[index]
        valid = np.where(np.isfinite(valid) & (valid > 0.0), valid, np.nan)
        with np.errstate(all="ignore"):
            coefficient_per_sample[index, :] = np.nanmedian(valid, axis=0)

    aggregate = np.full(grid.size, np.nan, dtype=np.float64)
    for index in range(grid.size):
        samples = coefficient_per_sample[index]
        samples = samples[np.isfinite(samples) & (samples > 0.0)]
        if samples.size:
            aggregate[index] = robust_aggregate(samples, aggregation)
    aggregate = _fill_missing_profile(aggregate, floor=coefficient_floor)
    aggregate = _smooth_positive_profile(aggregate, monitor_smoothing_window, floor=coefficient_floor)
    if str(monitor_exponent).lower() in {"identity", "a", "coefficient"}:
        density = aggregate.copy()
        exponent_label = "identity"
    else:
        density = np.power(aggregate + float(coefficient_floor), 1.0 / q_estimate)
        exponent_label = "q_root"
    density = _fill_missing_profile(density, floor=coefficient_floor)

    fallback_fraction = float(fallback_total / fallback_possible) if fallback_possible else 0.0
    metadata = {
        "method": "GPDE",
        "profile_type": "oracle_start_probe_monitor",
        "num_probe_nodes": int(grid.size),
        "num_probe_steps": int(steps.size),
        "probe_steps": [float(item) for item in steps],
        "q_estimate": float(q_estimate),
        "q_source": q_source,
        "monitor_exponent": exponent_label,
        "coefficient_floor": float(coefficient_floor),
        "monitor_smoothing_window": int(monitor_smoothing_window),
        "tiny_tangent_fallback_fraction": fallback_fraction,
        "aggregation": aggregation if isinstance(aggregation, str) else dict(aggregation),
    }
    return GPDEProfile(
        probe_grid=grid,
        probe_steps=steps,
        defects=defects,
        coefficient_per_sample=coefficient_per_sample,
        aggregate_coefficient=aggregate,
        monitor_density=density,
        q_estimate=float(q_estimate),
        q_source=q_source,
        fallback_counts=fallback_counts,
        fallback_fraction=fallback_fraction,
        metadata=metadata,
    )


def cumulative_trapezoid(grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    x = np.asarray(grid, dtype=np.float64)
    y = np.asarray(density, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("grid and density must have matching shape.")
    increments = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    cumulative = np.concatenate([[0.0], np.cumsum(np.maximum(increments, 0.0))])
    return cumulative


def snap_schedule_to_grid(u_schedule: np.ndarray, admissible_grid: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray]:
    schedule = np.asarray(u_schedule, dtype=np.float64)
    grid = np.asarray(admissible_grid, dtype=np.float64)
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("admissible_grid must be strictly increasing.")
    if grid.size < schedule.size:
        raise ValueError("admissible_grid must contain at least target_nfe + 1 points.")
    indices: list[int] = [0]
    for pos in range(1, schedule.size - 1):
        remaining = schedule.size - 1 - pos
        lo = indices[-1] + 1
        hi = grid.size - 1 - remaining
        nearest = int(np.argmin(np.abs(grid - schedule[pos])))
        indices.append(min(max(nearest, lo), hi))
    indices.append(grid.size - 1)
    snapped = grid[np.asarray(indices, dtype=np.int64)]
    errors = snapped - schedule
    return snapped, indices, errors


def materialize_gpde_schedule(
    probe_grid: np.ndarray,
    monitor_density: np.ndarray,
    target_nfe: int,
    *,
    admissible_grid: np.ndarray | None = None,
) -> GPDESchedule:
    if int(target_nfe) < 1:
        raise ValueError("target_nfe must be positive.")
    grid = np.asarray(probe_grid, dtype=np.float64)
    density = np.asarray(monitor_density, dtype=np.float64)
    cumulative = cumulative_trapezoid(grid, density)
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        u_schedule = np.linspace(float(grid[0]), float(grid[-1]), int(target_nfe) + 1, dtype=np.float64)
    else:
        targets = np.linspace(0.0, total, int(target_nfe) + 1, dtype=np.float64)
        u_schedule = np.interp(targets, cumulative, grid)
        u_schedule[0] = float(grid[0])
        u_schedule[-1] = float(grid[-1])

    if admissible_grid is None:
        _snapped_for_indices, selected_indices, _index_snap_errors = snap_schedule_to_grid(u_schedule, grid)
        snap_errors = np.zeros_like(u_schedule)
    else:
        u_schedule, selected_indices, snap_errors = snap_schedule_to_grid(u_schedule, admissible_grid)

    if np.any(np.diff(u_schedule) <= 0.0):
        raise ValueError("GPDE materialized schedule is not strictly increasing.")
    interval_masses = np.diff(np.interp(u_schedule, grid, cumulative))
    objective = float(np.max(interval_masses)) if interval_masses.size else 0.0
    metadata = {
        "method": "GPDE",
        "optimizer": "monitor_inverse_cdf",
        "target_nfe": int(target_nfe),
        "total_monitor_mass": total,
        "monitor_objective": objective,
        "max_abs_snap_error": float(np.max(np.abs(snap_errors))) if snap_errors.size else 0.0,
        "mean_abs_snap_error": float(np.mean(np.abs(snap_errors))) if snap_errors.size else 0.0,
    }
    return GPDESchedule(
        u_schedule=np.asarray(u_schedule, dtype=np.float64),
        selected_indices=[int(item) for item in selected_indices],
        interval_monitor_masses=np.asarray(interval_masses, dtype=np.float64),
        objective=objective,
        total_monitor_mass=total,
        snap_errors=np.asarray(snap_errors, dtype=np.float64),
        metadata=metadata,
    )

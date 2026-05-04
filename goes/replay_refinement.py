from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .edge_evaluator import ReplayMetrics, evaluate_replay_metrics


@dataclass
class ReplayRefinementResult:
    u_schedule: np.ndarray
    metrics: ReplayMetrics
    history: list[dict[str, float]]


def smoothness_penalty(u_schedule: np.ndarray) -> float:
    schedule = np.asarray(u_schedule, dtype=np.float64)
    steps = np.diff(schedule)
    if steps.size < 2:
        return 0.0
    ratios = steps[1:] / np.maximum(steps[:-1], 1.0e-12)
    return float(np.sum(np.square(np.log(np.maximum(ratios, 1.0e-12)))))


def refine_schedule_blackbox(
    solver: Any,
    oracle: Any,
    u_schedule: np.ndarray,
    candidate_grid: np.ndarray,
    metric: Any,
    *,
    rho: float = 0.1,
    aggregation: dict[str, Any] | str = "trimmed_mean",
    rounds: int = 3,
    local_window: int = 8,
    lambda_final: float = 0.0,
    mu_smooth: float = 0.0,
    fallback_full_residual_on_tiny_tangent: bool = True,
) -> ReplayRefinementResult:
    grid = np.asarray(candidate_grid, dtype=np.float64)
    schedule = np.asarray(u_schedule, dtype=np.float64).copy()
    history: list[dict[str, float]] = []

    def objective(candidate: np.ndarray) -> tuple[float, ReplayMetrics]:
        metrics = evaluate_replay_metrics(
            solver,
            oracle,
            candidate,
            metric,
            rho=rho,
            aggregation=aggregation,
            fallback_full_residual_on_tiny_tangent=fallback_full_residual_on_tiny_tangent,
        )
        value = metrics.replay_loss + lambda_final * metrics.final_mse + mu_smooth * smoothness_penalty(candidate)
        return float(value), metrics

    current_value, current_metrics = objective(schedule)
    history.append({"round": 0.0, "objective": current_value, "final_mse": current_metrics.final_mse})
    for round_index in range(1, int(rounds) + 1):
        improved = False
        for pos in range(1, schedule.size - 1):
            center_idx = int(np.argmin(np.abs(grid - schedule[pos])))
            left_bound = int(np.searchsorted(grid, schedule[pos - 1], side="right"))
            right_bound = int(np.searchsorted(grid, schedule[pos + 1], side="left"))
            lo = max(left_bound, center_idx - int(local_window))
            hi = min(right_bound, center_idx + int(local_window) + 1)
            for grid_idx in range(lo, hi):
                candidate = schedule.copy()
                candidate[pos] = grid[grid_idx]
                if candidate[pos - 1] >= candidate[pos] or candidate[pos] >= candidate[pos + 1]:
                    continue
                value, metrics = objective(candidate)
                if value < current_value - 1.0e-12:
                    schedule = candidate
                    current_value = value
                    current_metrics = metrics
                    improved = True
        history.append(
            {"round": float(round_index), "objective": current_value, "final_mse": current_metrics.final_mse}
        )
        if not improved:
            break
    return ReplayRefinementResult(u_schedule=schedule, metrics=current_metrics, history=history)

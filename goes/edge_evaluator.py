from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .aggregation import robust_aggregate
from .metrics import Metric
from .mixed_defect import mixed_normal_defect_sq
from .oracle import OracleData


@dataclass
class EdgeCostTable:
    candidate_grid: np.ndarray
    edge_costs: np.ndarray
    per_sample_costs: np.ndarray
    fallback_counts: np.ndarray
    fallback_fraction: float
    metadata: dict[str, Any]


def evaluate_edge_table(
    solver: Any,
    oracle: OracleData,
    candidate_grid: np.ndarray,
    metric: Metric,
    *,
    rho: float = 0.1,
    aggregation: dict[str, Any] | str = "trimmed_mean",
    eps: float = 1.0e-12,
    fallback_full_residual_on_tiny_tangent: bool = True,
) -> EdgeCostTable:
    grid = np.asarray(candidate_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("candidate_grid must be a one-dimensional array with at least two points.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("candidate_grid must be strictly increasing in unified coordinate u.")

    size = grid.size
    K = oracle.num_samples
    edge_costs = np.full((size, size), np.inf, dtype=np.float64)
    per_sample = np.full((size, size, K), np.inf, dtype=np.float64)
    fallback_counts = np.zeros((size, size), dtype=np.int64)

    for j in range(size - 1):
        x_a = oracle.state_at(float(grid[j]))
        for l in range(j + 1, size):
            b = float(grid[l])
            x_b = oracle.state_at(b)
            v_b = oracle.tangent_at(b)
            x_hat = solver.single_edge_step_from_state(x_a.copy(), float(grid[j]), b, oracle.conditions)
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
            per_sample[j, l, :] = defect.values
            fallback_counts[j, l] = int(np.count_nonzero(defect.fallback_mask))
            edge_costs[j, l] = robust_aggregate(defect.values, aggregation)

    possible = K * (size * (size - 1) // 2)
    fallback_fraction = float(np.sum(fallback_counts) / possible) if possible else 0.0
    metadata = {
        "num_candidates": int(size),
        "num_candidate_intervals": int(size - 1),
        "num_calibration_samples": int(K),
        "rho": float(rho),
        "aggregation": aggregation if isinstance(aggregation, str) else dict(aggregation),
        "tiny_tangent_fallback_fraction": fallback_fraction,
    }
    return EdgeCostTable(
        candidate_grid=grid,
        edge_costs=edge_costs,
        per_sample_costs=per_sample,
        fallback_counts=fallback_counts,
        fallback_fraction=fallback_fraction,
        metadata=metadata,
    )


@dataclass
class ReplayMetrics:
    schedule: np.ndarray
    endpoint_costs: np.ndarray
    endpoint_mse: np.ndarray
    final_sample_mse: np.ndarray
    final_mse: float
    replay_loss: float
    fallback_fraction: float


def evaluate_replay_metrics(
    solver: Any,
    oracle: OracleData,
    u_schedule: np.ndarray,
    metric: Metric,
    *,
    rho: float = 0.1,
    aggregation: dict[str, Any] | str = "trimmed_mean",
    eps: float = 1.0e-12,
    fallback_full_residual_on_tiny_tangent: bool = True,
) -> ReplayMetrics:
    schedule = np.asarray(u_schedule, dtype=np.float64)
    if np.any(np.diff(schedule) <= 0.0):
        raise ValueError("u_schedule must be strictly increasing.")
    current = oracle.state_at(float(schedule[0]))
    endpoint_costs: list[float] = []
    endpoint_mse: list[float] = []
    final_sample_mse = np.asarray([], dtype=np.float64)
    fallback_total = 0
    fallback_possible = 0
    for a, b in zip(schedule[:-1], schedule[1:]):
        current = solver.single_edge_step_from_state(current, float(a), float(b), oracle.conditions)
        target = oracle.state_at(float(b))
        tangent = oracle.tangent_at(float(b))
        residual = current - target
        defect = mixed_normal_defect_sq(
            residual,
            tangent,
            metric,
            float(b),
            rho=rho,
            eps=eps,
            fallback_full_residual_on_tiny_tangent=fallback_full_residual_on_tiny_tangent,
        )
        sample_mse = np.mean(np.reshape(residual, (residual.shape[0], -1)) ** 2, axis=1)
        endpoint_costs.append(robust_aggregate(defect.values, aggregation))
        endpoint_mse.append(float(np.mean(sample_mse)))
        final_sample_mse = np.asarray(sample_mse, dtype=np.float64)
        fallback_total += int(np.count_nonzero(defect.fallback_mask))
        fallback_possible += defect.fallback_mask.size
    final_mse = endpoint_mse[-1] if endpoint_mse else 0.0
    replay_loss = float(max(endpoint_costs)) if endpoint_costs else 0.0
    fallback_fraction = float(fallback_total / fallback_possible) if fallback_possible else 0.0
    return ReplayMetrics(
        schedule=schedule,
        endpoint_costs=np.asarray(endpoint_costs, dtype=np.float64),
        endpoint_mse=np.asarray(endpoint_mse, dtype=np.float64),
        final_sample_mse=final_sample_mse,
        final_mse=float(final_mse),
        replay_loss=replay_loss,
        fallback_fraction=fallback_fraction,
    )

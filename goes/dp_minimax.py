from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class MinimaxPath:
    indices: list[int]
    objective: float
    total_cost: float
    edge_costs: list[float]


def _validate_edge_costs(edge_costs: np.ndarray, intervals: int) -> None:
    costs = np.asarray(edge_costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("edge_costs must be a square matrix.")
    if intervals < 1:
        raise ValueError("intervals must be positive.")
    if intervals > costs.shape[0] - 1:
        raise ValueError("intervals cannot exceed candidate interval count.")


def solve_minimax_schedule(
    edge_costs: np.ndarray,
    intervals: int,
    *,
    tie_break_sum_cost: bool = True,
    tie_tolerance: float = 1.0e-12,
) -> MinimaxPath:
    D = np.asarray(edge_costs, dtype=np.float64)
    _validate_edge_costs(D, intervals)
    M = D.shape[0] - 1
    N = int(intervals)
    inf = np.inf
    dp = np.full((N + 1, M + 1), inf, dtype=np.float64)
    sum_dp = np.full((N + 1, M + 1), inf, dtype=np.float64)
    prev = np.full((N + 1, M + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    sum_dp[0, 0] = 0.0

    for n in range(1, N + 1):
        for l in range(1, M + 1):
            if l < n:
                continue
            for j in range(n - 1, l):
                if not np.isfinite(dp[n - 1, j]) or not np.isfinite(D[j, l]):
                    continue
                candidate = max(dp[n - 1, j], D[j, l])
                candidate_sum = sum_dp[n - 1, j] + D[j, l]
                current = dp[n, l]
                improves_primary = candidate < current - tie_tolerance
                ties_primary = abs(candidate - current) <= tie_tolerance
                improves_secondary = tie_break_sum_cost and candidate_sum < sum_dp[n, l] - tie_tolerance
                if improves_primary or (ties_primary and improves_secondary):
                    dp[n, l] = candidate
                    sum_dp[n, l] = candidate_sum
                    prev[n, l] = j

    if prev[N, M] < 0:
        raise ValueError("No feasible min-max path found.")

    indices = [M]
    l = M
    for n in range(N, 0, -1):
        j = int(prev[n, l])
        if j < 0:
            raise ValueError("Failed to backtrack min-max path.")
        indices.append(j)
        l = j
    indices.reverse()
    if len(indices) != N + 1 or indices[0] != 0 or indices[-1] != M:
        raise AssertionError("Invalid DP path reconstruction.")
    if any(b <= a for a, b in zip(indices[:-1], indices[1:])):
        raise AssertionError("DP path is not strictly increasing.")

    selected_costs = [float(D[a, b]) for a, b in zip(indices[:-1], indices[1:])]
    return MinimaxPath(
        indices=indices,
        objective=float(dp[N, M]),
        total_cost=float(sum(selected_costs)),
        edge_costs=selected_costs,
    )


def brute_force_minimax(edge_costs: np.ndarray, intervals: int) -> MinimaxPath:
    D = np.asarray(edge_costs, dtype=np.float64)
    _validate_edge_costs(D, intervals)
    M = D.shape[0] - 1
    best: MinimaxPath | None = None
    for internal in combinations(range(1, M), int(intervals) - 1):
        indices = [0, *internal, M]
        selected = [float(D[a, b]) for a, b in zip(indices[:-1], indices[1:])]
        if not all(np.isfinite(selected)):
            continue
        objective = max(selected)
        total = sum(selected)
        path = MinimaxPath(indices=indices, objective=objective, total_cost=total, edge_costs=selected)
        if best is None or objective < best.objective - 1.0e-12:
            best = path
        elif best is not None and abs(objective - best.objective) <= 1.0e-12 and total < best.total_cost:
            best = path
    if best is None:
        raise ValueError("No feasible brute-force path found.")
    return best

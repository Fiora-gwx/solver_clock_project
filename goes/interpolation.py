from __future__ import annotations

import numpy as np


def linear_interpolate_grid(values: np.ndarray, grid: np.ndarray, query: float | np.ndarray) -> np.ndarray:
    """Linearly interpolate values with grid axis at position 1.

    values is expected to have shape [K, J, ...]. A scalar query returns
    [K, ...]; an array query with length Q returns [K, Q, ...].
    """

    data = np.asarray(values)
    nodes = np.asarray(grid, dtype=np.float64)
    if data.shape[1] != nodes.shape[0]:
        raise ValueError("values must have shape [K, len(grid), ...].")
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("grid must be strictly increasing.")

    q = np.asarray(query, dtype=np.float64)
    scalar_query = q.ndim == 0
    q_flat = q.reshape(-1)
    if np.any(q_flat < nodes[0] - 1.0e-12) or np.any(q_flat > nodes[-1] + 1.0e-12):
        raise ValueError("query is outside interpolation grid.")
    q_flat = np.clip(q_flat, nodes[0], nodes[-1])

    right = np.searchsorted(nodes, q_flat, side="right")
    right = np.clip(right, 1, len(nodes) - 1)
    left = right - 1
    denom = nodes[right] - nodes[left]
    weight = (q_flat - nodes[left]) / denom

    left_values = data[:, left, ...]
    right_values = data[:, right, ...]
    shape = (1, q_flat.shape[0]) + (1,) * (data.ndim - 2)
    out = (1.0 - weight.reshape(shape)) * left_values + weight.reshape(shape) * right_values
    if scalar_query:
        return out[:, 0, ...]
    return out.reshape((data.shape[0],) + q.shape + data.shape[2:])

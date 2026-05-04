from __future__ import annotations

from typing import Any

import numpy as np


def robust_aggregate(values: np.ndarray, config: dict[str, Any] | str = "trimmed_mean", **kwargs: Any) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("Cannot aggregate an empty array.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Edge costs must be finite before aggregation.")

    if isinstance(config, str):
        name = config
        trim_ratio = float(kwargs.get("trim_ratio", 0.10))
        alpha = float(kwargs.get("alpha", 0.80))
    else:
        name = str(config.get("name", "trimmed_mean"))
        trim_ratio = float(config.get("trim_ratio", config.get("trim_fraction", 0.10)))
        alpha = float(config.get("alpha", 0.80))

    if name in {"trimmed_mean_10pct", "trimmed_mean_10"}:
        name = "trimmed_mean"
        trim_ratio = 0.10

    if name == "mean":
        return float(np.mean(data))
    if name == "median":
        return float(np.median(data))
    if name == "trimmed_mean":
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must satisfy 0 <= trim_ratio < 0.5.")
        trim_count = int(np.floor(data.size * trim_ratio))
        if trim_count == 0 or 2 * trim_count >= data.size:
            return float(np.mean(data))
        sorted_data = np.sort(data)
        return float(np.mean(sorted_data[trim_count:-trim_count]))
    if name == "cvar":
        if not 0.0 <= alpha < 1.0:
            raise ValueError("cvar alpha must satisfy 0 <= alpha < 1.")
        sorted_data = np.sort(data)
        start = int(np.floor(alpha * sorted_data.size))
        start = min(start, sorted_data.size - 1)
        return float(np.mean(sorted_data[start:]))
    raise ValueError(f"Unsupported aggregation mode: {name}")


def aggregation_label(config: dict[str, Any]) -> str:
    name = str(config.get("name", "trimmed_mean"))
    if name == "trimmed_mean":
        pct = int(round(100.0 * float(config.get("trim_ratio", 0.10))))
        return f"trimmed_mean_{pct}pct"
    if name == "cvar":
        pct = int(round(100.0 * float(config.get("alpha", 0.80))))
        return f"cvar_{pct}pct"
    return name

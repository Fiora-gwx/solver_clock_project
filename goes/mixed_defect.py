from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import Metric


@dataclass
class MixedDefectResult:
    values: np.ndarray
    fallback_mask: np.ndarray
    full_residual: np.ndarray
    tangent_component: np.ndarray

    @property
    def fallback_fraction(self) -> float:
        if self.fallback_mask.size == 0:
            return 0.0
        return float(np.mean(self.fallback_mask.astype(np.float64)))


def mixed_normal_defect_sq(
    residual: np.ndarray,
    tangent: np.ndarray,
    metric: Metric,
    u: float,
    *,
    rho: float = 0.1,
    eps: float = 1.0e-12,
    fallback_full_residual_on_tiny_tangent: bool = True,
) -> MixedDefectResult:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1].")
    r = np.asarray(residual, dtype=np.float64)
    v = np.asarray(tangent, dtype=np.float64)
    if r.shape != v.shape:
        raise ValueError(f"residual and tangent shapes must match, got {r.shape} and {v.shape}.")

    full = metric.dot(r, r, u)
    vgv = metric.dot(v, v, u)
    fallback_mask = np.asarray(vgv <= eps, dtype=bool)
    safe_vgv = np.maximum(vgv, eps)
    tangent_dot = metric.dot(v, r, u) / np.sqrt(safe_vgv)
    tangent_component = tangent_dot * tangent_dot
    values = full - (1.0 - float(rho)) * tangent_component
    values = np.maximum(values, 0.0)

    if fallback_full_residual_on_tiny_tangent:
        values = np.where(fallback_mask, full, values)
    return MixedDefectResult(
        values=np.asarray(values, dtype=np.float64),
        fallback_mask=fallback_mask,
        full_residual=np.asarray(full, dtype=np.float64),
        tangent_component=np.asarray(tangent_component, dtype=np.float64),
    )


def mixed_defect_metadata(config: dict[str, Any], fallback_fraction: float) -> dict[str, Any]:
    return {
        "rho": float(config.get("rho", 0.1)),
        "eps": float(config.get("eps", 1.0e-12)),
        "fallback_full_residual_on_tiny_tangent": bool(
            config.get("fallback_full_residual_on_tiny_tangent", True)
        ),
        "tiny_tangent_fallback_fraction": float(fallback_fraction),
    }

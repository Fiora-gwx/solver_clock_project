from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .interpolation import linear_interpolate_grid


class Metric(Protocol):
    name: str

    def apply(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        ...

    def dot(self, x: np.ndarray, y: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        ...

    def norm_sq(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


def _sum_nonbatch(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    return np.sum(values.reshape(values.shape[0], -1), axis=1)


@dataclass
class IdentityMetric:
    eps: float = 1.0e-12
    name: str = "identity"

    def apply(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        del u
        return np.asarray(x, dtype=np.float64)

    def dot(self, x: np.ndarray, y: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        del u
        return _sum_nonbatch(np.asarray(x, dtype=np.float64) * np.asarray(y, dtype=np.float64))

    def norm_sq(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        return self.dot(x, x, u)

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "eps": self.eps}


@dataclass
class EDMScalarMetric:
    sigma_data: float = 0.5
    eps: float = 1.0e-12
    coordinate_name: str = "sigma"
    u_grid: np.ndarray | None = None
    sigma_grid: np.ndarray | None = None
    name: str = "edm_scalar"

    def __post_init__(self) -> None:
        if self.u_grid is None and self.sigma_grid is None:
            return
        if self.u_grid is None or self.sigma_grid is None:
            raise ValueError("EDMScalarMetric requires both u_grid and sigma_grid when either is provided.")
        u_grid = np.asarray(self.u_grid, dtype=np.float64)
        sigma_grid = np.asarray(self.sigma_grid, dtype=np.float64)
        if u_grid.ndim != 1 or sigma_grid.ndim != 1 or u_grid.shape != sigma_grid.shape:
            raise ValueError("EDMScalarMetric u_grid and sigma_grid must be one-dimensional arrays with matching shape.")
        if len(u_grid) < 2 or np.any(np.diff(u_grid) <= 0.0):
            raise ValueError("EDMScalarMetric u_grid must be strictly increasing.")
        self.u_grid = u_grid
        self.sigma_grid = np.maximum(sigma_grid, self.eps)

    def sigma(self, u: float | np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        if self.u_grid is not None and self.sigma_grid is not None:
            interpolated = linear_interpolate_grid(self.sigma_grid.reshape(1, -1), self.u_grid, values)[0]
            return np.maximum(interpolated, self.eps)
        if self.coordinate_name in {"negative_sigma", "negative_sigmas"}:
            return np.maximum(-values, self.eps)
        if self.coordinate_name == "log_sigma":
            return np.exp(values)
        if self.coordinate_name == "negative_log_sigma":
            return np.exp(-values)
        if self.coordinate_name == "logsnr":
            return np.exp(-0.5 * values)
        return np.maximum(values, self.eps)

    def weight(self, u: float | np.ndarray) -> np.ndarray:
        sigma = self.sigma(u)
        return 1.0 / (sigma * sigma + float(self.sigma_data) ** 2)

    def _broadcast_weight(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        weight = self.weight(u)
        if np.asarray(weight).ndim == 0:
            return np.asarray(weight, dtype=np.float64)
        return np.asarray(weight, dtype=np.float64).reshape((1, -1) + (1,) * (np.asarray(x).ndim - 2))

    def apply(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        return self._broadcast_weight(values, u) * values

    def dot(self, x: np.ndarray, y: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        return _sum_nonbatch(np.asarray(x, dtype=np.float64) * self.apply(y, u))

    def norm_sq(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        return self.dot(x, x, u)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sigma_data": self.sigma_data,
            "eps": self.eps,
            "coordinate_name": self.coordinate_name,
            "sigma_mapping": "interpolated_grid" if self.u_grid is not None else "coordinate",
            "sigma_grid_size": 0 if self.sigma_grid is None else int(len(self.sigma_grid)),
        }


@dataclass
class ChannelWhitenedMetric:
    u_grid: np.ndarray
    weights: np.ndarray
    eps: float = 1.0e-12
    min_weight: float = 1.0e-4
    max_weight: float = 1.0e4
    name: str = "channel_whitened"

    @classmethod
    def from_oracle(
        cls,
        states: np.ndarray,
        u_grid: np.ndarray,
        *,
        eps: float = 1.0e-12,
        min_weight: float = 1.0e-4,
        max_weight: float = 1.0e4,
    ) -> "ChannelWhitenedMetric":
        data = np.asarray(states, dtype=np.float64)
        if data.ndim < 3:
            raise ValueError("states must have shape [K, J, ...].")
        if data.ndim == 3:
            # Treat vector dimensions as channels.
            variance = np.var(data, axis=0)
        else:
            # [K, J, C, ...] -> [J, C]
            axes = (0,) + tuple(range(3, data.ndim))
            variance = np.var(data, axis=axes)
        weights = 1.0 / (variance + eps)
        weights = np.clip(weights, min_weight, max_weight)
        return cls(
            u_grid=np.asarray(u_grid, dtype=np.float64),
            weights=np.asarray(weights, dtype=np.float64),
            eps=float(eps),
            min_weight=float(min_weight),
            max_weight=float(max_weight),
        )

    def weight_at(self, u: float | np.ndarray) -> np.ndarray:
        weight_data = self.weights[None, ...]
        return linear_interpolate_grid(weight_data, self.u_grid, u)[0]

    def _reshape_weight(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        values = np.asarray(x)
        weight = np.asarray(self.weight_at(u), dtype=np.float64)
        if values.ndim == 1:
            return weight
        if values.ndim == 2:
            return weight.reshape((1,) + weight.shape)
        # [B, C, ...]
        if weight.ndim == 1:
            return weight.reshape((1, weight.shape[0]) + (1,) * (values.ndim - 2))
        return weight

    def apply(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        return self._reshape_weight(values, u) * values

    def dot(self, x: np.ndarray, y: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        return _sum_nonbatch(np.asarray(x, dtype=np.float64) * self.apply(y, u))

    def norm_sq(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        return self.dot(x, x, u)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "eps": self.eps,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "weight_shape": list(self.weights.shape),
        }


def make_metric(config: dict[str, Any], *, oracle: Any | None = None, coordinate: Any | None = None) -> Metric:
    name = str(config.get("name", "identity"))
    eps = float(config.get("eps", 1.0e-12))
    if name == "identity":
        return IdentityMetric(eps=eps)
    if name == "edm_scalar":
        coordinate_name = getattr(coordinate, "name", str(config.get("coordinate_name", "sigma")))
        return EDMScalarMetric(
            sigma_data=float(config.get("sigma_data", 0.5)),
            eps=eps,
            coordinate_name=coordinate_name,
            u_grid=config.get("u_grid"),
            sigma_grid=config.get("sigma_grid"),
        )
    if name == "channel_whitened":
        if oracle is None:
            raise ValueError("channel_whitened metric requires an oracle.")
        return ChannelWhitenedMetric.from_oracle(
            oracle.states,
            oracle.u_grid,
            eps=eps,
            min_weight=float(config.get("min_weight", 1.0e-4)),
            max_weight=float(config.get("max_weight", 1.0e4)),
        )
    raise ValueError(f"Unsupported GOES metric: {name}")

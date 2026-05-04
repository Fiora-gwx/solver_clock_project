from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CoordinateAdapter:
    """Map repository-native coordinates to the increasing GOES coordinate u."""

    name: str
    u_min: float
    u_max: float
    direction: str = "increasing"
    sigma_eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.direction not in {"increasing", "decreasing"}:
            raise ValueError("direction must be 'increasing' or 'decreasing'.")
        if not self.u_min < self.u_max:
            raise ValueError("u_min must be smaller than u_max.")

    def native_to_u(self, native: Any) -> np.ndarray:
        values = np.asarray(native, dtype=np.float64)
        if self.name in {"t", "identity", "u"}:
            return values
        if self.name == "sigma":
            return values
        if self.name == "log_sigma":
            return np.log(np.maximum(values, self.sigma_eps))
        if self.name == "logsnr":
            sigma = np.maximum(values, self.sigma_eps)
            return -2.0 * np.log(sigma)
        raise ValueError(f"Unsupported coordinate: {self.name}")

    def u_to_native(self, u: Any) -> np.ndarray:
        values = np.asarray(u, dtype=np.float64)
        if self.name in {"t", "identity", "u"}:
            return values
        if self.name == "sigma":
            return values
        if self.name == "log_sigma":
            return np.exp(values)
        if self.name == "logsnr":
            return np.exp(-0.5 * values)
        raise ValueError(f"Unsupported coordinate: {self.name}")

    def candidate_grid(self, size: int, grid_type: str = "uniform_in_u") -> np.ndarray:
        if size < 1:
            raise ValueError("candidate grid size must be positive.")
        if grid_type != "uniform_in_u":
            raise ValueError(f"Unsupported candidate grid type: {grid_type}")
        return np.linspace(self.u_min, self.u_max, int(size) + 1, dtype=np.float64)

    def native_schedule_from_u(self, u_schedule: Any) -> np.ndarray:
        native = self.u_to_native(u_schedule)
        if self.direction == "decreasing":
            return native[::-1].copy()
        return native

    def metadata(self) -> dict[str, Any]:
        return {
            "coordinate": self.name,
            "direction": self.direction,
            "u_min": self.u_min,
            "u_max": self.u_max,
        }


def make_coordinate_adapter(config: dict[str, Any]) -> CoordinateAdapter:
    return CoordinateAdapter(
        name=str(config.get("name", "t")),
        direction=str(config.get("direction", "increasing")),
        u_min=float(config.get("u_min", 0.0)),
        u_max=float(config.get("u_max", 1.0)),
    )

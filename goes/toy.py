from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ToyFlowModel:
    """Small deterministic flow ODE used for CPU GOES validation."""

    name: str = "toy_flow"
    state_shape: tuple[int, ...] = (2,)

    def drift(self, x: np.ndarray, u: float, condition: np.ndarray | float | None = None) -> np.ndarray:
        state = np.asarray(x, dtype=np.float64)
        flat = state.reshape(state.shape[0], -1)
        cond = np.zeros((state.shape[0],), dtype=np.float64)
        if condition is not None:
            cond = np.asarray(condition, dtype=np.float64).reshape(state.shape[0])

        out = np.zeros_like(flat)
        phase = 2.0 * np.pi * float(u) + cond
        damping = -0.08 - 0.04 * np.sin(phase)
        out += damping[:, None] * flat

        if flat.shape[1] >= 2:
            omega = 1.0 + 0.25 * np.cos(phase)
            out[:, 0] += -omega * flat[:, 1] + 0.05 * np.sin(phase)
            out[:, 1] += omega * flat[:, 0] + 0.05 * np.cos(phase)
        else:
            out[:, 0] += 0.05 * np.sin(phase)
        if flat.shape[1] > 2:
            idx = np.arange(flat.shape[1] - 2, dtype=np.float64)
            out[:, 2:] += 0.02 * np.sin(phase[:, None] + 0.5 * idx[None, :])
        return out.reshape(state.shape)

    @property
    def identifier(self) -> str:
        return f"{self.name}:{'x'.join(str(item) for item in self.state_shape)}"


@dataclass(frozen=True)
class ToySolver:
    name: str
    model: ToyFlowModel
    deterministic: bool = True
    theory_covered: bool = True
    history_mode: str = "one_step"

    def single_edge_step_from_state(
        self,
        x_a: np.ndarray,
        a: float,
        b: float,
        condition: np.ndarray | float | None = None,
    ) -> np.ndarray:
        h = float(b) - float(a)
        if h <= 0.0:
            raise ValueError("GPDE edge/probe steps require b > a in unified coordinate.")
        if self.name == "euler":
            return x_a + h * self.model.drift(x_a, a, condition)
        if self.name in {"heun", "heun2"}:
            k1 = self.model.drift(x_a, a, condition)
            pred = x_a + h * k1
            k2 = self.model.drift(pred, b, condition)
            return x_a + 0.5 * h * (k1 + k2)
        if self.name == "midpoint":
            k1 = self.model.drift(x_a, a, condition)
            mid = x_a + 0.5 * h * k1
            k2 = self.model.drift(mid, a + 0.5 * h, condition)
            return x_a + h * k2
        if self.name == "biased_euler":
            result = x_a + h * self.model.drift(x_a, a, condition)
            return result + 0.05 * h
        if self.name == "empirical_noisy_euler":
            return x_a + h * self.model.drift(x_a, a, condition)
        raise ValueError(f"Unsupported toy solver: {self.name}")

    def replay(self, x0: np.ndarray, u_schedule: np.ndarray, condition: np.ndarray | None = None) -> list[np.ndarray]:
        states = [np.asarray(x0, dtype=np.float64)]
        current = states[0]
        for a, b in zip(u_schedule[:-1], u_schedule[1:]):
            current = self.single_edge_step_from_state(current, float(a), float(b), condition)
            states.append(current)
        return states

    def compatibility_metadata(self) -> dict[str, Any]:
        if self.name == "empirical_noisy_euler":
            return {
                "deterministic_oracle_theory": False,
                "coverage_note": "empirical-only: solver is declared incompatible with deterministic-oracle theory.",
            }
        return {
            "deterministic_oracle_theory": bool(self.deterministic and self.theory_covered),
            "coverage_note": "theory-covered deterministic toy ODE solver.",
        }


def make_toy_model(config: dict[str, Any]) -> ToyFlowModel:
    shape = tuple(int(item) for item in config.get("state_shape", [2]))
    return ToyFlowModel(name=str(config.get("name", "toy_flow")), state_shape=shape)


def make_solver(name: str, model: ToyFlowModel, mode: str = "one_step") -> ToySolver:
    solver_name = str(name)
    theory_covered = solver_name != "empirical_noisy_euler"
    return ToySolver(name=solver_name, model=model, theory_covered=theory_covered, history_mode=str(mode))

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import stable_hash
from .interpolation import linear_interpolate_grid
from .toy import ToyFlowModel


@dataclass
class CalibrationSamples:
    initial_states: np.ndarray
    conditions: np.ndarray
    noise_seeds: np.ndarray
    split: str
    metadata: dict[str, Any]


@dataclass
class OracleData:
    states: np.ndarray
    tangents: np.ndarray
    u_grid: np.ndarray
    conditions: np.ndarray
    noise_seeds: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_samples(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_shape(self) -> tuple[int, ...]:
        return tuple(self.states.shape[2:])

    def state_at(self, u: float | np.ndarray) -> np.ndarray:
        return linear_interpolate_grid(self.states, self.u_grid, u)

    def tangent_at(self, u: float | np.ndarray) -> np.ndarray:
        return linear_interpolate_grid(self.tangents, self.u_grid, u)

    def condition(self, sample_index: int | None = None) -> np.ndarray:
        if sample_index is None:
            return self.conditions
        return self.conditions[int(sample_index)]


def make_calibration_samples(
    *,
    num_samples: int,
    seed: int,
    state_shape: tuple[int, ...],
    split: str,
) -> CalibrationSamples:
    rng = np.random.default_rng(int(seed))
    initial_states = rng.normal(size=(int(num_samples),) + tuple(state_shape)).astype(np.float64)
    conditions = rng.uniform(low=-np.pi, high=np.pi, size=(int(num_samples),)).astype(np.float64)
    noise_seeds = rng.integers(0, np.iinfo(np.int32).max, size=int(num_samples), dtype=np.int64)
    metadata = {
        "num_samples": int(num_samples),
        "seed": int(seed),
        "split": str(split),
        "initial_noise_hash": stable_hash(initial_states.tolist()),
        "condition_split_hash": stable_hash({"conditions": conditions.tolist(), "split": split}),
        "noise_seed_hash": stable_hash(noise_seeds.tolist()),
        "noise_seeds": [int(item) for item in noise_seeds.tolist()],
    }
    return CalibrationSamples(
        initial_states=initial_states,
        conditions=conditions,
        noise_seeds=noise_seeds,
        split=str(split),
        metadata=metadata,
    )


def _rk4_step(
    model: ToyFlowModel,
    x: np.ndarray,
    u: float,
    h: float,
    condition: np.ndarray,
) -> np.ndarray:
    k1 = model.drift(x, u, condition)
    k2 = model.drift(x + 0.5 * h * k1, u + 0.5 * h, condition)
    k3 = model.drift(x + 0.5 * h * k2, u + 0.5 * h, condition)
    k4 = model.drift(x + h * k3, u + h, condition)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_reference_ode(
    model: ToyFlowModel,
    samples: CalibrationSamples,
    *,
    u_min: float,
    u_max: float,
    ref_nfe: int,
    ref_grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_steps = max(int(ref_nfe), int(ref_grid_size) - 1)
    dense_grid = np.linspace(float(u_min), float(u_max), dense_steps + 1, dtype=np.float64)
    dense_states = np.empty((samples.initial_states.shape[0], dense_grid.shape[0]) + samples.initial_states.shape[1:])
    current = samples.initial_states.copy()
    dense_states[:, 0, ...] = current
    for idx, (a, b) in enumerate(zip(dense_grid[:-1], dense_grid[1:]), start=1):
        current = _rk4_step(model, current, float(a), float(b - a), samples.conditions)
        dense_states[:, idx, ...] = current

    u_grid = np.linspace(float(u_min), float(u_max), int(ref_grid_size), dtype=np.float64)
    states = linear_interpolate_grid(dense_states, dense_grid, u_grid)
    tangents = np.empty_like(states)
    for index, u_value in enumerate(u_grid):
        tangents[:, index, ...] = model.drift(states[:, index, ...], float(u_value), samples.conditions)
    return states, tangents, u_grid


def build_universal_oracle(
    model: ToyFlowModel,
    samples: CalibrationSamples,
    *,
    coordinate_metadata: dict[str, Any],
    ref_integrator: str,
    interpolation: str,
    ref_nfe: int,
    ref_grid_size: int,
    guidance_scale: float,
) -> OracleData:
    if ref_integrator != "rk4":
        raise ValueError("The toy GOES oracle currently supports ref_integrator: rk4.")
    if interpolation != "linear":
        raise ValueError("The toy GOES oracle currently supports interpolation: linear.")
    states, tangents, u_grid = solve_reference_ode(
        model,
        samples,
        u_min=float(coordinate_metadata["u_min"]),
        u_max=float(coordinate_metadata["u_max"]),
        ref_nfe=int(ref_nfe),
        ref_grid_size=int(ref_grid_size),
    )
    metadata = {
        "method": "GOES",
        "model_identifier": model.identifier,
        "ode_sampler_family": "toy_deterministic_flow_ode",
        "coordinate_mapping": coordinate_metadata,
        "ref_integrator": ref_integrator,
        "interpolation": interpolation,
        "ref_nfe": int(ref_nfe),
        "ref_grid_size": int(ref_grid_size),
        "ref_grid_hash": stable_hash(u_grid.tolist()),
        "condition_split_hash": samples.metadata["condition_split_hash"],
        "initial_noise_hash": samples.metadata["initial_noise_hash"],
        "noise_seed_hash": samples.metadata["noise_seed_hash"],
        "noise_seeds": samples.metadata["noise_seeds"],
        "cfg": {"guidance_scale": float(guidance_scale)},
        "dtype": "float64",
        "device": "cpu",
        "calibration": samples.metadata,
    }
    return OracleData(
        states=states,
        tangents=tangents,
        u_grid=u_grid,
        conditions=samples.conditions,
        noise_seeds=samples.noise_seeds,
        metadata=metadata,
    )

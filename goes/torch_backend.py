from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.clock.defect_balanced import (
    StepFn,
    VelocityFn,
    _call_step,
    _call_velocity,
    build_velocity_stepper,
)

from .config import repo_path, stable_hash
from .edge_evaluator import EdgeCostTable, evaluate_edge_table
from .metrics import Metric
from .oracle import OracleData
from .oracle_cache import load_oracle, make_oracle_key, save_oracle


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _tensor_to_numpy(values: torch.Tensor) -> np.ndarray:
    tensor = values.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy()


def _microbatch_velocity(
    velocity_fn: VelocityFn,
    sample: torch.Tensor,
    coordinate: float,
    *,
    microbatch_size: int | None = None,
) -> torch.Tensor:
    coordinate_tensor = torch.as_tensor(float(coordinate), device=sample.device, dtype=sample.dtype)
    if microbatch_size is None or microbatch_size <= 0 or microbatch_size >= sample.shape[0]:
        return _call_velocity(velocity_fn, sample, coordinate_tensor, 0, sample.shape[0])
    outputs: list[torch.Tensor] = []
    for start in range(0, sample.shape[0], int(microbatch_size)):
        stop = min(start + int(microbatch_size), sample.shape[0])
        outputs.append(_call_velocity(velocity_fn, sample[start:stop], coordinate_tensor, start, stop))
    return torch.cat(outputs, dim=0)


def _rk4_step(
    velocity_fn: VelocityFn,
    sample: torch.Tensor,
    start: float,
    end: float,
    *,
    microbatch_size: int | None = None,
) -> torch.Tensor:
    h = float(end) - float(start)
    k1 = _microbatch_velocity(velocity_fn, sample, start, microbatch_size=microbatch_size)
    k2 = _microbatch_velocity(velocity_fn, sample + 0.5 * h * k1, start + 0.5 * h, microbatch_size=microbatch_size)
    k3 = _microbatch_velocity(velocity_fn, sample + 0.5 * h * k2, start + 0.5 * h, microbatch_size=microbatch_size)
    k4 = _microbatch_velocity(velocity_fn, sample + h * k3, end, microbatch_size=microbatch_size)
    return sample + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _linear_interpolate_torch(states: torch.Tensor, grid: np.ndarray, query: np.ndarray) -> torch.Tensor:
    nodes = np.asarray(grid, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("grid must be strictly increasing.")
    if np.any(q < nodes[0] - 1.0e-12) or np.any(q > nodes[-1] + 1.0e-12):
        raise ValueError("query values must lie inside the grid.")
    q = np.clip(q, nodes[0], nodes[-1])
    right = np.searchsorted(nodes, q, side="right")
    right = np.clip(right, 1, len(nodes) - 1)
    left = right - 1
    weight = (q - nodes[left]) / (nodes[right] - nodes[left])
    left_tensor = states[left]
    right_tensor = states[right]
    weight_tensor = torch.as_tensor(weight, device=states.device, dtype=states.dtype).reshape(
        (-1,) + (1,) * (states.ndim - 1)
    )
    return (1.0 - weight_tensor) * left_tensor + weight_tensor * right_tensor


def build_torch_velocity_oracle(
    *,
    initial_sample: torch.Tensor,
    velocity_fn: VelocityFn,
    u_grid: np.ndarray,
    ref_nfe: int,
    metadata: dict[str, Any],
    ref_integrator: str = "rk4",
    conditions: np.ndarray | None = None,
    noise_seeds: np.ndarray | None = None,
    microbatch_size: int | None = None,
) -> OracleData:
    if str(ref_integrator) != "rk4":
        raise ValueError("GOES torch velocity oracle currently supports ref_integrator='rk4'.")
    grid = np.asarray(u_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("u_grid must be one-dimensional with at least two nodes.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("u_grid must be strictly increasing.")
    dense_steps = max(int(ref_nfe), len(grid) - 1)
    dense_grid = np.linspace(float(grid[0]), float(grid[-1]), dense_steps + 1, dtype=np.float64)

    current = initial_sample.detach().clone()
    dense_states: list[torch.Tensor] = [current.detach().clone()]
    with torch.no_grad():
        for start, end in zip(dense_grid[:-1], dense_grid[1:]):
            current = _rk4_step(
                velocity_fn,
                current,
                float(start),
                float(end),
                microbatch_size=microbatch_size,
            )
            dense_states.append(current.detach().clone())
        dense_tensor = torch.stack(dense_states, dim=0)
        states_by_node = _linear_interpolate_torch(dense_tensor, dense_grid, grid)
        tangents = []
        for index, u_value in enumerate(grid):
            tangent = _microbatch_velocity(
                velocity_fn,
                states_by_node[index],
                float(u_value),
                microbatch_size=microbatch_size,
            )
            tangents.append(tangent.detach().clone())
        tangents_by_node = torch.stack(tangents, dim=0)

    states = _tensor_to_numpy(states_by_node).transpose((1, 0) + tuple(range(2, states_by_node.ndim)))
    tangent_values = _tensor_to_numpy(tangents_by_node).transpose(
        (1, 0) + tuple(range(2, tangents_by_node.ndim))
    )
    batch = initial_sample.shape[0]
    if conditions is None:
        conditions_array = np.arange(batch, dtype=np.float64)
    else:
        conditions_array = np.asarray(conditions)
        if conditions_array.shape[0] != batch:
            raise ValueError("conditions must have leading dimension equal to the batch size.")
    if noise_seeds is None:
        noise_seed_array = np.arange(batch, dtype=np.int64)
    else:
        noise_seed_array = np.asarray(noise_seeds, dtype=np.int64)
        if noise_seed_array.shape[0] != batch:
            raise ValueError("noise_seeds must have leading dimension equal to the batch size.")
    payload = dict(metadata)
    payload.update(
        {
            "method": "GPDE",
            "legacy_method_alias": "GOES",
            "ref_integrator": str(ref_integrator),
            "interpolation": "linear",
            "ref_nfe": int(ref_nfe),
            "ref_grid_size": int(len(grid)),
            "ref_grid_hash": stable_hash(grid.tolist()),
            "initial_noise_hash": _array_hash(_tensor_to_numpy(initial_sample)),
            "condition_split_hash": _array_hash(np.asarray(conditions_array)),
            "noise_seed_hash": stable_hash(noise_seed_array.tolist()),
            "noise_seeds": [int(item) for item in noise_seed_array.tolist()],
            "dtype": str(initial_sample.dtype).replace("torch.", ""),
            "device": str(initial_sample.device),
            "microbatch_size": None if microbatch_size is None else int(microbatch_size),
        }
    )
    payload.setdefault("model_identifier", "torch_velocity_fn")
    payload.setdefault("ode_sampler_family", "deterministic_velocity_ode")
    payload.setdefault("coordinate_mapping", {"coordinate": "u", "direction": "increasing"})
    payload.setdefault("cfg", {})
    return OracleData(
        states=states,
        tangents=tangent_values,
        u_grid=grid,
        conditions=conditions_array,
        noise_seeds=noise_seed_array,
        metadata=payload,
    )


@dataclass
class TorchOracleCacheResult:
    oracle: OracleData
    cache_key: str
    cache_path: Path
    metadata_path: Path
    loaded_from_cache: bool
    elapsed_seconds: float


def build_or_load_torch_velocity_oracle(
    *,
    cache_dir: str | Path,
    initial_sample: torch.Tensor,
    velocity_fn: VelocityFn,
    u_grid: np.ndarray,
    ref_nfe: int,
    metadata: dict[str, Any],
    ref_integrator: str = "rk4",
    conditions: np.ndarray | None = None,
    noise_seeds: np.ndarray | None = None,
    microbatch_size: int | None = None,
    reuse: bool = True,
) -> TorchOracleCacheResult:
    started = time.time()
    grid = np.asarray(u_grid, dtype=np.float64)
    batch = initial_sample.shape[0]
    conditions_array = np.arange(batch, dtype=np.float64) if conditions is None else np.asarray(conditions)
    noise_seed_array = np.arange(batch, dtype=np.int64) if noise_seeds is None else np.asarray(noise_seeds, dtype=np.int64)
    key_metadata = dict(metadata)
    key_metadata.update(
        {
            "ref_integrator": str(ref_integrator),
            "interpolation": "linear",
            "ref_nfe": int(ref_nfe),
            "ref_grid_size": int(len(grid)),
            "ref_grid_hash": stable_hash(grid.tolist()),
            "initial_noise_hash": _array_hash(_tensor_to_numpy(initial_sample)),
            "condition_split_hash": _array_hash(conditions_array),
            "noise_seed_hash": stable_hash(noise_seed_array.tolist()),
            "dtype": str(initial_sample.dtype).replace("torch.", ""),
            "device": str(initial_sample.device),
        }
    )
    key_metadata.setdefault("model_identifier", "torch_velocity_fn")
    key_metadata.setdefault("ode_sampler_family", "deterministic_velocity_ode")
    key_metadata.setdefault("coordinate_mapping", {"coordinate": "u", "direction": "increasing"})
    key_metadata.setdefault("cfg", {})
    cache_key = make_oracle_key(key_metadata)
    root = repo_path(cache_dir)
    cache_path = root / f"{cache_key}.npz"
    metadata_path = root / f"{cache_key}.json"
    if bool(reuse) and cache_path.exists() and metadata_path.exists():
        oracle = load_oracle(root, cache_key)
        oracle.metadata["loaded_from_cache"] = True
        return TorchOracleCacheResult(oracle, cache_key, cache_path, metadata_path, True, time.time() - started)

    oracle = build_torch_velocity_oracle(
        initial_sample=initial_sample,
        velocity_fn=velocity_fn,
        u_grid=grid,
        ref_nfe=ref_nfe,
        metadata=metadata,
        ref_integrator=ref_integrator,
        conditions=conditions_array,
        noise_seeds=noise_seed_array,
        microbatch_size=microbatch_size,
    )
    oracle.metadata["loaded_from_cache"] = False
    save_oracle(root, cache_key, oracle)
    return TorchOracleCacheResult(oracle, cache_key, cache_path, metadata_path, False, time.time() - started)


@dataclass
class TorchStepSolver:
    name: str
    step_fn: StepFn
    device: torch.device
    dtype: torch.dtype

    def single_edge_step_from_state(
        self,
        x_a: np.ndarray,
        a: float,
        b: float,
        condition: np.ndarray | None = None,
    ) -> np.ndarray:
        del condition
        sample = torch.as_tensor(x_a, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            result = _call_step(self.step_fn, sample, float(a), float(b), 0, sample.shape[0])
        return _tensor_to_numpy(result)

    def compatibility_metadata(self) -> dict[str, Any]:
        return {
            "deterministic_oracle_theory": True,
            "coverage_note": "theory-covered when the wrapped velocity_fn represents the deterministic target ODE.",
        }


def make_torch_step_solver(
    *,
    name: str,
    velocity_fn: VelocityFn,
    device: torch.device | str,
    dtype: torch.dtype,
) -> TorchStepSolver:
    step_fn = build_velocity_stepper(velocity_fn, name)
    return TorchStepSolver(name=str(name), step_fn=step_fn, device=torch.device(device), dtype=dtype)


def evaluate_torch_velocity_edge_table(
    *,
    solver_name: str,
    velocity_fn: VelocityFn,
    oracle: OracleData,
    candidate_grid: np.ndarray,
    metric: Metric,
    rho: float = 0.1,
    aggregation: dict[str, Any] | str = "trimmed_mean",
    eps: float = 1.0e-12,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> EdgeCostTable:
    solver = make_torch_step_solver(name=solver_name, velocity_fn=velocity_fn, device=device, dtype=dtype)
    return evaluate_edge_table(
        solver,
        oracle,
        candidate_grid,
        metric,
        rho=rho,
        aggregation=aggregation,
        eps=eps,
    )

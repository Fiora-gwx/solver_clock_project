from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class CalibrationRecord:
    step_index: int
    timestep: float | None
    sample_norms: tuple[float, ...]

    @property
    def norm(self) -> float:
        return float(np.mean(self.sample_norms))


def _extract_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if hasattr(value, "sample"):
        return _extract_tensor(value.sample)
    if hasattr(value, "prev_sample"):
        return _extract_tensor(value.prev_sample)
    if hasattr(value, "__dict__"):
        return _extract_tensor(vars(value))
    return None


def _extract_timestep(args: tuple[Any, ...]) -> float | None:
    if len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].numel() > 0:
        return float(args[1].detach().flatten()[0].cpu().item())
    for item in args:
        if isinstance(item, torch.Tensor) and item.numel() == 1:
            return float(item.detach().cpu().item())
    return None


class ForwardNormCollector:
    def __init__(
        self,
        module: torch.nn.Module,
        *,
        norm_type: str = "l2",
        normalize_by_dim: bool = False,
    ) -> None:
        self._module = module
        self._norm_type = norm_type
        self._normalize_by_dim = normalize_by_dim
        self.records: list[CalibrationRecord] = []
        self._handle = None

    def _compute_sample_norms(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.detach().float().reshape(tensor.shape[0], -1)
        if self._norm_type == "l1":
            norms = flat.abs().sum(dim=1)
            if self._normalize_by_dim:
                norms = norms / flat.shape[1]
            return norms
        if self._norm_type == "l2":
            norms = flat.norm(dim=1)
            if self._normalize_by_dim:
                norms = norms / np.sqrt(flat.shape[1])
            return norms
        if self._norm_type == "l2_sq":
            norms = flat.pow(2).sum(dim=1)
            if self._normalize_by_dim:
                norms = norms / flat.shape[1]
            return norms
        if self._norm_type == "linf":
            return flat.abs().amax(dim=1)
        raise ValueError(f"Unsupported norm_type: {self._norm_type}")

    def _hook(self, _module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        tensor = _extract_tensor(output)
        if tensor is None:
            return
        sample_norms = tuple(float(item) for item in self._compute_sample_norms(tensor).cpu().tolist())
        timestep = _extract_timestep(args)
        self.records.append(
            CalibrationRecord(step_index=len(self.records), timestep=timestep, sample_norms=sample_norms)
        )

    def __enter__(self) -> "ForwardNormCollector":
        self._handle = self._module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def reduce_samples(values: tuple[float, ...], mode: str, *, trim_fraction: float = 0.1) -> float:
    data = np.asarray(values, dtype=np.float64)
    if mode == "mean":
        return float(np.mean(data))
    if mode == "median":
        return float(np.median(data))
    if mode == "trimmed_mean":
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must satisfy 0 <= trim_fraction < 0.5.")
        if len(data) <= 2:
            return float(np.mean(data))
        trim_count = int(np.floor(len(data) * trim_fraction))
        if trim_count == 0 or trim_count * 2 >= len(data):
            return float(np.mean(data))
        trimmed = np.sort(data)[trim_count:-trim_count]
        return float(np.mean(trimmed))
    raise ValueError(f"Unsupported sample_reduce mode: {mode}")


def reduce_cycle(values: list[float], mode: str) -> float:
    data = np.asarray(values, dtype=np.float64)
    if mode == "mean":
        return float(np.mean(data))
    if mode == "median":
        return float(np.median(data))
    raise ValueError(f"Unsupported cycle_reduce mode: {mode}")


def aggregate_by_cycle(
    records: list[CalibrationRecord],
    domain_values: np.ndarray,
    *,
    sample_reduce: str = "mean",
    cycle_reduce: str = "mean",
    trim_fraction: float = 0.1,
    profile_stat: str = "mean_reduced",
) -> tuple[np.ndarray, np.ndarray]:
    cycle_len = len(domain_values)
    if cycle_len == 0:
        raise ValueError("domain_values must be non-empty")
    if len(records) < cycle_len:
        raise ValueError(
            f"Expected at least {cycle_len} calibration records for one full cycle, got {len(records)}."
        )

    buckets: dict[int, list[float]] = defaultdict(list)
    raw_buckets: dict[int, list[float]] = defaultdict(list)
    for idx, record in enumerate(records):
        buckets[idx % cycle_len].append(
            reduce_samples(record.sample_norms, sample_reduce, trim_fraction=trim_fraction)
        )
        raw_buckets[idx % cycle_len].extend(float(item) for item in record.sample_norms)

    if profile_stat == "mean_reduced":
        proxy = np.asarray([reduce_cycle(buckets[index], cycle_reduce) for index in range(cycle_len)], dtype=np.float64)
    elif profile_stat == "rms_all":
        proxy = np.asarray(
            [
                float(np.sqrt(np.mean(np.square(np.asarray(raw_buckets[index], dtype=np.float64)))))
                for index in range(cycle_len)
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported profile_stat mode: {profile_stat}")
    return np.asarray(domain_values, dtype=np.float64), proxy

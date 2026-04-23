from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import dump_json, ensure_dir, load_json, resolve_repo_path


def scheduler_accepts(scheduler: Any, name: str) -> bool:
    return name in set(inspect.signature(scheduler.set_timesteps).parameters.keys())


@dataclass
class ScheduleBundle:
    timesteps: np.ndarray | None = None
    sigmas: np.ndarray | None = None
    time_grid: np.ndarray | None = None
    sigma_grid: np.ndarray | None = None
    tau_grid: np.ndarray | None = None
    g_grid: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = ("timesteps", "sigmas", "time_grid", "sigma_grid", "tau_grid", "g_grid")
        if all(getattr(self, name) is None for name in arrays):
            raise ValueError("ScheduleBundle requires at least one schedule array.")
        for name in arrays:
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, np.asarray(values, dtype=np.float64))

    @property
    def nfe(self) -> int:
        if "effective_nfe" in self.meta:
            return int(self.meta["effective_nfe"])
        source = self.timesteps if self.timesteps is not None else self.sigmas
        if source is None:
            source = self.time_grid if self.time_grid is not None else self.sigma_grid
        if source is None:
            raise ValueError("ScheduleBundle does not contain a primary schedule array.")
        return int(len(source))

    def save(self, output_dir: str | Path) -> Path:
        resolved = ensure_dir(output_dir)
        for field_name in ("timesteps", "sigmas", "time_grid", "sigma_grid", "tau_grid", "g_grid"):
            array_path = resolved / f"{field_name}.npy"
            values = getattr(self, field_name)
            if values is None:
                if array_path.exists():
                    array_path.unlink()
                continue
            np.save(array_path, values)
        payload = dict(self.meta)
        payload.setdefault("nfe", self.nfe)
        payload.setdefault("has_timesteps", self.timesteps is not None)
        payload.setdefault("has_sigmas", self.sigmas is not None)
        payload.setdefault("has_time_grid", self.time_grid is not None)
        payload.setdefault("has_sigma_grid", self.sigma_grid is not None)
        payload.setdefault("has_tau_grid", self.tau_grid is not None)
        payload.setdefault("has_g_grid", self.g_grid is not None)
        dump_json(payload, resolved / "meta.json")
        return resolved

    @classmethod
    def load(cls, input_dir: str | Path) -> "ScheduleBundle":
        resolved = resolve_repo_path(input_dir)
        meta_path = resolved / "meta.json"
        arrays = {}
        for field_name in ("timesteps", "sigmas", "time_grid", "sigma_grid", "tau_grid", "g_grid"):
            array_path = resolved / f"{field_name}.npy"
            arrays[field_name] = np.load(array_path) if array_path.exists() else None
        meta = load_json(meta_path) if meta_path.exists() else {}
        return cls(meta=meta, **arrays)

    def scheduler_kwargs(
        self,
        scheduler: Any | None = None,
        *,
        prefer: str = "sigmas",
        integer_timesteps: bool = False,
    ) -> dict[str, list[float] | list[int]]:
        if prefer not in {"sigmas", "timesteps"}:
            raise ValueError(f"Unsupported schedule preference: {prefer}")

        choices = [prefer, "timesteps" if prefer == "sigmas" else "sigmas"]
        for field_name in choices:
            values = getattr(self, field_name)
            if values is None:
                continue
            if scheduler is not None and not scheduler_accepts(scheduler, field_name):
                continue
            if field_name == "timesteps" and integer_timesteps:
                return {field_name: np.round(values).astype(np.int64).tolist()}
            return {field_name: values.tolist()}

        supported = []
        if scheduler is not None:
            for field_name in ("timesteps", "sigmas"):
                if scheduler_accepts(scheduler, field_name):
                    supported.append(field_name)
        supported_str = ", ".join(supported) if supported else "none"
        raise ValueError(f"No compatible schedule field found. Scheduler supports: {supported_str}")

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.schedule_bundle import ScheduleBundle

from .config import repo_path
from .schedules import GOES_SCHEDULE_IMPLEMENTATION_VERSION


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_vector(payload: dict[str, Any], key: str) -> np.ndarray:
    _require(key in payload, f"Missing `{key}` in GOES schedule payload.")
    values = np.asarray(payload[key], dtype=np.float64)
    _require(values.ndim == 1, f"`{key}` must be a one-dimensional array.")
    _require(np.all(np.isfinite(values)), f"`{key}` contains non-finite values.")
    return values


def _require_strictly_monotone(values: np.ndarray, name: str) -> None:
    if values.size <= 1:
        return
    diffs = np.diff(values)
    _require(
        bool(np.all(diffs > 0.0) or np.all(diffs < 0.0)),
        f"{name} must be strictly monotone.",
    )


def verify_schedule_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload.get("method") == "GOES", "Expected schedule payload method to be GOES.")
    _require(
        int(payload.get("schedule_implementation_version", -1)) == GOES_SCHEDULE_IMPLEMENTATION_VERSION,
        "GOES schedule implementation version is stale or missing.",
    )
    target_nfe = int(payload.get("target_nfe", -1))
    _require(target_nfe > 0, "`target_nfe` must be positive.")
    solver = str(payload.get("solver", ""))
    coordinate = str(payload.get("coordinate", ""))
    coordinate_direction = str(payload.get("coordinate_direction", ""))
    _require(bool(solver), "`solver` must be recorded.")
    _require(bool(coordinate), "`coordinate` must be recorded.")
    _require(bool(coordinate_direction), "`coordinate_direction` must be recorded.")

    u_schedule = _finite_vector(payload, "u_schedule")
    native_schedule = _finite_vector(payload, "native_schedule")
    _require(u_schedule.size == target_nfe + 1, "`u_schedule` must contain target_nfe + 1 points.")
    _require(native_schedule.size == target_nfe + 1, "`native_schedule` must contain target_nfe + 1 points.")
    _require(np.all(np.diff(u_schedule) > 0.0), "`u_schedule` must be strictly increasing.")

    selected_costs = np.asarray(payload.get("selected_edge_costs", []), dtype=np.float64)
    _require(selected_costs.ndim == 1, "`selected_edge_costs` must be one-dimensional.")
    _require(selected_costs.size == target_nfe, "`selected_edge_costs` must contain target_nfe values.")
    _require(np.all(np.isfinite(selected_costs)), "`selected_edge_costs` contains non-finite values.")

    selected_indices = payload.get("selected_indices")
    if selected_indices is not None:
        indices = np.asarray(selected_indices, dtype=np.int64)
        _require(indices.ndim == 1, "`selected_indices` must be one-dimensional.")
        _require(indices.size == target_nfe + 1, "`selected_indices` must contain target_nfe + 1 values.")
        _require(np.all(np.diff(indices) > 0), "`selected_indices` must be strictly increasing.")

    rho = float(payload.get("rho", np.nan))
    edge_objective = float(payload.get("edge_objective", np.nan))
    _require(np.isfinite(rho) and 0.0 <= rho <= 1.0, "`rho` must be finite and lie in [0, 1].")
    _require(np.isfinite(edge_objective), "`edge_objective` must be finite.")
    _require(bool(payload.get("oracle_cache_key")), "`oracle_cache_key` must be recorded.")
    _require(bool(payload.get("schedule_hash")), "`schedule_hash` must be recorded.")
    _require(bool(payload.get("aggregation")), "`aggregation` must be recorded.")
    _require(isinstance(payload.get("metric"), dict), "`metric` metadata must be recorded as an object.")

    return {
        "target_nfe": target_nfe,
        "solver": solver,
        "coordinate": coordinate,
        "coordinate_direction": coordinate_direction,
        "schedule_hash": str(payload["schedule_hash"]),
        "oracle_cache_key": str(payload["oracle_cache_key"]),
        "u_start": float(u_schedule[0]),
        "u_end": float(u_schedule[-1]),
        "edge_objective": edge_objective,
        "max_selected_edge_cost": float(np.max(selected_costs)) if selected_costs.size else 0.0,
    }


def verify_schedule_json(schedule_json: str | Path) -> dict[str, Any]:
    path = repo_path(schedule_json)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = verify_schedule_payload(payload)
    result["schedule_json"] = str(path)
    return result


def verify_schedule_bundle(bundle_dir: str | Path, *, expected_nfe: int | None = None) -> dict[str, Any]:
    path = repo_path(bundle_dir)
    bundle = ScheduleBundle.load(path)
    meta = bundle.meta
    _require(meta.get("schedule_family") == "GOES", "Expected ScheduleBundle schedule_family to be GOES.")
    _require(
        int(meta.get("schedule_implementation_version", -1)) == GOES_SCHEDULE_IMPLEMENTATION_VERSION,
        "GOES ScheduleBundle implementation version is stale or missing.",
    )
    effective_nfe = int(meta.get("effective_nfe", bundle.nfe))
    if expected_nfe is not None:
        _require(effective_nfe == int(expected_nfe), "ScheduleBundle effective_nfe does not match schedule.json.")
    _require(bool(meta.get("oracle_cache_key")), "ScheduleBundle must record oracle_cache_key.")
    _require(bool(meta.get("schedule_hash")), "ScheduleBundle must record schedule_hash.")

    has_time = bundle.time_grid is not None or bundle.timesteps is not None
    has_sigma = bundle.sigma_grid is not None or bundle.sigmas is not None
    _require(has_time or has_sigma, "ScheduleBundle must contain timestep or sigma arrays.")

    if bundle.time_grid is not None:
        _require(bundle.time_grid.ndim == 1, "time_grid must be one-dimensional.")
        _require(bundle.time_grid.size == effective_nfe + 1, "time_grid must contain effective_nfe + 1 points.")
        _require(np.all(np.isfinite(bundle.time_grid)), "time_grid contains non-finite values.")
        _require_strictly_monotone(bundle.time_grid, "time_grid")
    if bundle.timesteps is not None:
        _require(bundle.timesteps.ndim == 1, "timesteps must be one-dimensional.")
        _require(bundle.timesteps.size == effective_nfe, "timesteps must contain effective_nfe points.")
        _require(np.all(np.isfinite(bundle.timesteps)), "timesteps contains non-finite values.")
        _require_strictly_monotone(bundle.timesteps, "timesteps")
    if bundle.sigma_grid is not None:
        _require(bundle.sigma_grid.ndim == 1, "sigma_grid must be one-dimensional.")
        _require(bundle.sigma_grid.size == effective_nfe + 1, "sigma_grid must contain effective_nfe + 1 points.")
        _require(np.all(np.isfinite(bundle.sigma_grid)), "sigma_grid contains non-finite values.")
        _require_strictly_monotone(bundle.sigma_grid, "sigma_grid")
    if bundle.sigmas is not None:
        _require(bundle.sigmas.ndim == 1, "sigmas must be one-dimensional.")
        _require(bundle.sigmas.size == effective_nfe, "sigmas must contain effective_nfe points.")
        _require(np.all(np.isfinite(bundle.sigmas)), "sigmas contains non-finite values.")
        _require_strictly_monotone(bundle.sigmas, "sigmas")

    return {
        "bundle_dir": str(path),
        "effective_nfe": effective_nfe,
        "schedule_hash": str(meta["schedule_hash"]),
        "oracle_cache_key": str(meta["oracle_cache_key"]),
        "representation": str(meta.get("representation", "")),
    }


def verify_goes_schedule(
    schedule_json: str | Path,
    *,
    bundle_dir: str | Path | None = None,
) -> dict[str, Any]:
    schedule_result = verify_schedule_json(schedule_json)
    result: dict[str, Any] = {"schedule": schedule_result}
    if bundle_dir is not None:
        bundle_result = verify_schedule_bundle(bundle_dir, expected_nfe=int(schedule_result["target_nfe"]))
        _require(
            bundle_result["schedule_hash"] == schedule_result["schedule_hash"],
            "ScheduleBundle schedule_hash does not match schedule.json.",
        )
        _require(
            bundle_result["oracle_cache_key"] == schedule_result["oracle_cache_key"],
            "ScheduleBundle oracle_cache_key does not match schedule.json.",
        )
        result["bundle"] = bundle_result
    return result

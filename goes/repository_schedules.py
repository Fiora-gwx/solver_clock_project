from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.schedule_bundle import ScheduleBundle

from .config import repo_path
from .schedules import GPDE_SCHEDULE_IMPLEMENTATION_VERSION, GOES_SCHEDULE_IMPLEMENTATION_VERSION
from .verify import verify_schedule_payload


def schedule_json_to_bundle(
    schedule_json: str | Path,
    *,
    representation: str,
    backend: str,
    solver: str | None = None,
) -> ScheduleBundle:
    path = repo_path(schedule_json)
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    method = str(payload.get("method", ""))
    if method not in {"GPDE", "GOES"}:
        raise ValueError("Expected a GPDE or GOES schedule.json payload.")
    verify_schedule_payload(payload)
    native = np.asarray(payload["native_schedule"], dtype=np.float64)
    if native.ndim != 1 or native.size < 2:
        raise ValueError("GOES native_schedule must be a one-dimensional array with at least two nodes.")
    target_nfe = int(payload["target_nfe"])
    if native.size != target_nfe + 1:
        raise ValueError(
            f"{method} native_schedule should contain target_nfe + 1 nodes, got {native.size} for NFE {target_nfe}."
        )
    version = GPDE_SCHEDULE_IMPLEMENTATION_VERSION if method == "GPDE" else GOES_SCHEDULE_IMPLEMENTATION_VERSION
    meta = {
        "schedule_family": method,
        "legacy_schedule_family_alias": "GOES" if method == "GPDE" else "",
        "schedule_implementation_version": version,
        "backend": backend,
        "solver": solver or payload.get("solver", ""),
        "representation": representation,
        "coordinate": payload.get("coordinate", ""),
        "coordinate_direction": payload.get("coordinate_direction", ""),
        "oracle_cache_key": payload.get("oracle_cache_key", ""),
        "rho": payload.get("rho", ""),
        "metric": payload.get("metric", {}),
        "aggregation": payload.get("aggregation", ""),
        "edge_objective": payload.get("edge_objective", ""),
        "monitor_objective": payload.get("monitor_objective", ""),
        "selected_edge_costs": payload.get("selected_edge_costs", []),
        "selected_monitor_masses": payload.get("selected_monitor_masses", []),
        "schedule_hash": payload.get("schedule_hash", ""),
        "effective_nfe": target_nfe,
        "solver_steps": target_nfe,
    }
    for optional_key in (
        "dataset",
        "model_asset",
        "seed",
        "guidance_scale",
        "prompt_asset",
        "prompt_count",
        "pipeline_kind",
        "proxy_solver",
        "height",
        "width",
        "dtype",
        "physical_grid_mode",
        "calibration_config",
        "pilot_config",
        "oracle_config",
        "candidate_grid_config",
        "calibration_cost_estimate",
        "calibration_cost_unit",
        "calibration_cost_breakdown",
        "model_output_type",
        "sigma_floor",
    ):
        if optional_key in payload:
            meta[optional_key] = payload[optional_key]
    if representation == "timesteps":
        return ScheduleBundle(
            timesteps=native[:-1].copy(),
            time_grid=native.copy(),
            meta={**meta, "terminal_timestep": float(native[-1])},
        )
    if representation == "sigmas":
        return ScheduleBundle(
            sigmas=native[:-1].copy(),
            sigma_grid=native.copy(),
            meta={**meta, "terminal_sigma": float(native[-1])},
        )
    raise ValueError(f"Unsupported GOES bundle representation: {representation}")


def export_schedule_bundle(
    schedule_json: str | Path,
    output_dir: str | Path,
    *,
    representation: str,
    backend: str,
    solver: str | None = None,
) -> Path:
    bundle = schedule_json_to_bundle(schedule_json, representation=representation, backend=backend, solver=solver)
    return bundle.save(repo_path(output_dir))

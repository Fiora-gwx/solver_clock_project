from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.clock.goes import GPDE_SCHEDULE_IMPLEMENTATION_VERSION, GOES_SCHEDULE_IMPLEMENTATION_VERSION

from .aggregation import aggregation_label
from .config import stable_hash
from .coordinate import CoordinateAdapter
from .dp_minimax import MinimaxPath
from .logging_utils import dump_json, write_csv


def schedule_payload(
    *,
    solver_name: str,
    target_nfe: int,
    coordinate: CoordinateAdapter,
    u_schedule: np.ndarray,
    rho: float,
    metric_metadata: dict[str, Any],
    aggregation_config: dict[str, Any],
    oracle_cache_key: str,
    path: MinimaxPath,
) -> dict[str, Any]:
    native = coordinate.native_schedule_from_u(u_schedule)
    return {
        "method": "GOES",
        "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
        "solver": solver_name,
        "target_nfe": int(target_nfe),
        "coordinate": coordinate.name,
        "coordinate_direction": coordinate.direction,
        "u_schedule": [float(item) for item in u_schedule],
        "native_schedule": [float(item) for item in native],
        "rho": float(rho),
        "metric": metric_metadata,
        "aggregation": aggregation_label(aggregation_config),
        "oracle_cache_key": oracle_cache_key,
        "edge_objective": float(path.objective),
        "selected_edge_costs": [float(item) for item in path.edge_costs],
        "selected_indices": [int(item) for item in path.indices],
        "schedule_hash": stable_hash([float(item) for item in u_schedule]),
    }


def gpde_schedule_payload(
    *,
    solver_name: str,
    target_nfe: int,
    coordinate: CoordinateAdapter | Any,
    u_schedule: np.ndarray,
    rho: float,
    metric_metadata: dict[str, Any],
    aggregation_config: dict[str, Any],
    oracle_cache_key: str,
    profile_metadata: dict[str, Any],
    schedule_metadata: dict[str, Any],
    selected_indices: list[int],
    interval_monitor_masses: np.ndarray,
    snap_errors: np.ndarray | None = None,
    native_schedule: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if native_schedule is None:
        native = coordinate.native_schedule_from_u(u_schedule)
        coordinate_name = coordinate.name
        coordinate_direction = coordinate.direction
    else:
        native = np.asarray(native_schedule, dtype=np.float64)
        coordinate_name = getattr(coordinate, "name", "")
        coordinate_direction = getattr(coordinate, "direction", "")
    masses = np.asarray(interval_monitor_masses, dtype=np.float64)
    snaps = np.zeros_like(np.asarray(u_schedule, dtype=np.float64)) if snap_errors is None else np.asarray(snap_errors, dtype=np.float64)
    payload = {
        "method": "GPDE",
        "legacy_method_alias": "GOES",
        "schedule_implementation_version": GPDE_SCHEDULE_IMPLEMENTATION_VERSION,
        "solver": solver_name,
        "target_nfe": int(target_nfe),
        "coordinate": coordinate_name,
        "coordinate_direction": coordinate_direction,
        "u_schedule": [float(item) for item in np.asarray(u_schedule, dtype=np.float64)],
        "native_schedule": [float(item) for item in native],
        "rho": float(rho),
        "metric": metric_metadata,
        "aggregation": aggregation_label(aggregation_config),
        "oracle_cache_key": oracle_cache_key,
        "optimizer": "monitor_inverse_cdf",
        "edge_objective": float(schedule_metadata.get("monitor_objective", 0.0)),
        "monitor_objective": float(schedule_metadata.get("monitor_objective", 0.0)),
        "total_monitor_mass": float(schedule_metadata.get("total_monitor_mass", 0.0)),
        "selected_monitor_masses": [float(item) for item in masses],
        "selected_edge_costs": [float(item) for item in masses],
        "selected_indices": [int(item) for item in selected_indices],
        "snap_errors": [float(item) for item in snaps],
        "max_abs_snap_error": float(schedule_metadata.get("max_abs_snap_error", 0.0)),
        "mean_abs_snap_error": float(schedule_metadata.get("mean_abs_snap_error", 0.0)),
        "q_estimate": float(profile_metadata.get("q_estimate", np.nan)),
        "q_source": str(profile_metadata.get("q_source", "")),
        "monitor_exponent": str(profile_metadata.get("monitor_exponent", "q_root")),
        "probe_profile": dict(profile_metadata),
        "schedule_hash": stable_hash([float(item) for item in np.asarray(u_schedule, dtype=np.float64)]),
    }
    if extra:
        payload.update(extra)
    return payload


def save_schedule_outputs(
    run_dir: Path,
    *,
    payload: dict[str, Any],
    selected_indices: list[int],
    selected_edge_costs: list[float] | np.ndarray,
) -> None:
    dump_json(payload, run_dir / "schedule.json")
    dump_json(
        {
            "method": "GOES",
            "solver": payload["solver"],
            "target_nfe": payload["target_nfe"],
            "native_schedule": payload["native_schedule"],
            "schedule_hash": payload["schedule_hash"],
        },
        run_dir / "schedule_native.json",
    )
    rows = []
    u_schedule = payload["u_schedule"]
    interval_scores = [float(item) for item in selected_edge_costs]
    for edge_index, (start_idx, end_idx, cost) in enumerate(
        zip(selected_indices[:-1], selected_indices[1:], interval_scores)
    ):
        rows.append(
            {
                "edge_index": edge_index,
                "candidate_start_index": int(start_idx),
                "candidate_end_index": int(end_idx),
                "u_start": float(u_schedule[edge_index]),
                "u_end": float(u_schedule[edge_index + 1]),
                "edge_cost": float(cost),
                "interval_score": float(cost),
            }
        )
    write_csv(
        rows,
        run_dir / "selected_edges.csv",
        [
            "edge_index",
            "candidate_start_index",
            "candidate_end_index",
            "u_start",
            "u_end",
            "edge_cost",
            "interval_score",
        ],
    )

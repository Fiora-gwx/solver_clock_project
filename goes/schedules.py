from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.clock.goes import GOES_SCHEDULE_IMPLEMENTATION_VERSION

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


def save_schedule_outputs(
    run_dir: Path,
    *,
    payload: dict[str, Any],
    selected_indices: list[int],
    selected_edge_costs: list[float],
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
    for edge_index, (start_idx, end_idx, cost) in enumerate(
        zip(selected_indices[:-1], selected_indices[1:], selected_edge_costs)
    ):
        rows.append(
            {
                "edge_index": edge_index,
                "candidate_start_index": int(start_idx),
                "candidate_end_index": int(end_idx),
                "u_start": float(u_schedule[edge_index]),
                "u_end": float(u_schedule[edge_index + 1]),
                "edge_cost": float(cost),
            }
        )
    write_csv(
        rows,
        run_dir / "selected_edges.csv",
        ["edge_index", "candidate_start_index", "candidate_end_index", "u_start", "u_end", "edge_cost"],
    )

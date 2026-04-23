#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_yaml
from src.utils.schedule_bundle import ScheduleBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize published 10-step AYS assets into schedule bundles.")
    parser.add_argument(
        "--inventory",
        default="configs/reference_schedules/ays_published_10step.yaml",
        help="YAML inventory that stores published AYS schedules.",
    )
    parser.add_argument(
        "--output-root",
        default="schedules/ays_like/published",
        help="Directory root for materialized published AYS schedule bundles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = load_yaml(args.inventory)["published_ays_assets"]
    output_root = Path(args.output_root)

    for model_key, payload in inventory.items():
        nfe = int(payload["nfe"])
        sigma_grid = np.asarray(payload["noise_levels"], dtype=np.float64)
        timestep_indices = payload.get("timestep_indices")
        time_grid = None if timestep_indices is None else np.asarray(timestep_indices, dtype=np.float64)

        bundle = ScheduleBundle(
            sigmas=sigma_grid[:-1],
            sigma_grid=sigma_grid,
            timesteps=None if time_grid is None else time_grid[:-1],
            time_grid=time_grid,
            meta={
                "schedule_family": "ays_published",
                "source": "published_10step_table",
                "model_name": str(payload["model_name"]),
                "effective_nfe": nfe,
                "solver_steps": nfe,
                "step_methods": ["published_ays"] * nfe,
                "execution_backend": "external_asset",
                "published_asset": True,
                "terminal_sigma": float(sigma_grid[-1]),
                "terminal_timestep": None if time_grid is None else float(time_grid[-1]),
            },
        )
        bundle.save(output_root / model_key / f"nfe_{nfe:03d}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goes.repository_schedules import export_schedule_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a GOES schedule.json to the repository ScheduleBundle format.")
    parser.add_argument("--schedule-json", "--schedule", dest="schedule_json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--representation", choices=["timesteps", "sigmas"], required=True)
    parser.add_argument("--backend", choices=["pndm", "diffusers", "toy"], required=True)
    parser.add_argument("--solver", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_schedule_bundle(
        args.schedule_json,
        args.output_dir,
        representation=args.representation,
        backend=args.backend,
        solver=args.solver,
    )


if __name__ == "__main__":
    main()

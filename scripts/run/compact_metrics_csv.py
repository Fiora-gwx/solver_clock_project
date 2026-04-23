#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_yaml, resolve_repo_path
from src.utils.results import SUMMARY_FIELDNAMES, compact_result_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact experiment metrics CSVs to the standard summary schema.")
    parser.add_argument("csvs", nargs="*", help="CSV files to rewrite in place.")
    parser.add_argument("--glob", dest="csv_glob", help="Glob for CSV files relative to the repo root.")
    parser.add_argument("--experiment-config", help="Optional experiment config used to filter rows to valid solver/schedule pairs.")
    return parser.parse_args()


def canonical_schedule_label(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    mapping = {
        "base": "base",
        "linear": "linear",
        "ays": "ays",
        "lcs_1": "LCS-1",
        "v_b": "V_b",
        "a_a": "A_a",
        "a_b": "A_b",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported schedule name: {name}")
    return mapping[normalized]


def resolve_solver_schedule_overrides(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    solvers = [str(item) for item in config.get("solvers", [])]
    schedules = config.get("schedules") or config.get("variants") or ([config["schedule"]] if "schedule" in config else [])
    defaults = tuple(str(item) for item in schedules)
    raw_overrides = config.get("solver_schedules")
    if raw_overrides is None:
        return {solver: defaults for solver in solvers}
    if not isinstance(raw_overrides, Mapping):
        raise TypeError("`solver_schedules` must be a mapping when provided.")

    overrides: dict[str, tuple[str, ...]] = {}
    unknown_solvers = set(raw_overrides.keys()) - set(solvers)
    if unknown_solvers:
        raise ValueError(f"`solver_schedules` contains unknown solvers: {sorted(unknown_solvers)}")

    for solver in solvers:
        raw_schedules = raw_overrides.get(solver, defaults)
        if not isinstance(raw_schedules, (list, tuple)) or not raw_schedules:
            raise TypeError(f"`solver_schedules.{solver}` must be a non-empty list when provided.")
        overrides[solver] = tuple(str(item) for item in raw_schedules)
    return overrides


def schedule_family(schedule_name: str) -> str:
    if "[" in schedule_name:
        return schedule_name.split("[", 1)[0]
    return schedule_name


def build_row_filter(config_path: str | Path):
    config = load_yaml(config_path)
    solvers = {str(item) for item in config.get("solvers", [])}
    allowed = {
        solver: {canonical_schedule_label(item) for item in schedules}
        for solver, schedules in resolve_solver_schedule_overrides(config).items()
    }

    def keep_row(row: dict[str, Any]) -> bool:
        solver = str(row.get("solver", ""))
        if solver not in solvers:
            return False
        return schedule_family(str(row.get("schedule", ""))) in allowed.get(solver, set())

    return keep_row


def resolve_targets(args: argparse.Namespace) -> list[Path]:
    targets: list[Path] = []
    seen: set[Path] = set()
    for item in args.csvs:
        resolved = resolve_repo_path(item)
        if resolved not in seen:
            targets.append(resolved)
            seen.add(resolved)
    if args.csv_glob:
        for match in sorted(resolve_repo_path(".").glob(args.csv_glob)):
            if match not in seen:
                targets.append(match)
                seen.add(match)
    if not targets:
        raise ValueError("Provide at least one CSV path or --glob pattern.")
    return targets


def main() -> None:
    args = parse_args()
    targets = resolve_targets(args)
    keep_row = build_row_filter(args.experiment_config) if args.experiment_config else None
    for target in targets:
        compact_result_csv(target, keep_row=keep_row)
        print(f"[compact] {target} -> columns={','.join(SUMMARY_FIELDNAMES)}")


if __name__ == "__main__":
    main()

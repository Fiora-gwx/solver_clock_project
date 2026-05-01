#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import resolve_repo_path


PAIR_KEYS = ("model_asset", "solver", "nfe", "seed", "prompt_index")
GROUP_KEYS = ("model_asset", "solver", "nfe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SADB pilot-cost Pareto CSV from scoring details.")
    parser.add_argument("--input-csv", action="append", required=True, help="One or more detail scoring CSVs.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metric", default="image_reward")
    return parser.parse_args()


def load_rows(paths: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_path in paths:
        path = resolve_repo_path(raw_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def schedule_family(schedule: str) -> str:
    normalized = schedule.strip()
    lower = normalized.lower()
    if lower == "base":
        return "base"
    if lower in {"ays", "ays_like"}:
        return "AYS"
    if lower == "sadb" or lower.startswith("sadb["):
        return normalized if normalized.startswith("SADB") else "SADB" + normalized[4:]
    return normalized


def variant_label(schedule: str) -> str:
    if schedule.startswith("SADB[") and schedule.endswith("]"):
        return schedule.removeprefix("SADB[").removesuffix("]")
    return schedule


def numeric(row: dict[str, str], metric: str) -> float | None:
    raw = str(row.get(metric, "")).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def pair_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in PAIR_KEYS)


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in GROUP_KEYS)


def schedule_meta_path(row: dict[str, str]) -> Path | None:
    raw = str(row.get("schedule_dir", "")).strip()
    if not raw:
        return None
    meta_path = resolve_repo_path(raw) / "meta.json"
    return meta_path if meta_path.exists() else None


def calibration_cost_from_meta(row: dict[str, str]) -> int | None:
    meta_path = schedule_meta_path(row)
    if meta_path is None:
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    try:
        return int(meta["pilot_batch_size"]) * int(meta["pilot_num_batches"]) * int(meta["physical_grid_size"])
    except KeyError:
        return None


def win_rate_against(
    *,
    left_rows: dict[tuple[str, ...], dict[str, str]],
    right_rows: dict[tuple[str, ...], dict[str, str]],
    metric: str,
) -> float | None:
    common_keys = sorted(set(left_rows) & set(right_rows))
    wins = 0.0
    pairs = 0
    for key in common_keys:
        left = numeric(left_rows[key], metric)
        right = numeric(right_rows[key], metric)
        if left is None or right is None:
            continue
        pairs += 1
        if left > right:
            wins += 1.0
        elif left == right:
            wins += 0.5
    if pairs == 0:
        return None
    return wins / pairs


def mean_metric(rows: Iterable[dict[str, str]], metric: str) -> float | None:
    values = [value for row in rows if (value := numeric(row, metric)) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def blank_if_none(value: float | int | None) -> float | int | str:
    return "" if value is None else value


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    grouped: dict[tuple[str, ...], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[group_key(row)][schedule_family(str(row.get("schedule", "")))].append(row)

    output_rows: list[dict[str, Any]] = []
    for key, schedules in sorted(grouped.items()):
        model_asset, solver, nfe = key
        base_rows = {pair_key(row): row for row in schedules.get("base", [])}
        ays_rows = {pair_key(row): row for row in schedules.get("AYS", [])}
        for schedule in sorted(item for item in schedules if item == "SADB" or item.startswith("SADB[")):
            sadb_rows = schedules[schedule]
            sadb_by_key = {pair_key(row): row for row in sadb_rows}
            costs = [cost for row in sadb_rows if (cost := calibration_cost_from_meta(row)) is not None]
            metric_mean = mean_metric(sadb_rows, args.metric)
            base_win_rate = win_rate_against(left_rows=sadb_by_key, right_rows=base_rows, metric=args.metric) if base_rows else None
            ays_win_rate = win_rate_against(left_rows=sadb_by_key, right_rows=ays_rows, metric=args.metric) if ays_rows else None
            output_rows.append(
                {
                    "model_asset": model_asset,
                    "solver": solver,
                    "nfe": nfe,
                    "schedule": schedule,
                    "variant": variant_label(schedule),
                    "calibration_cost": statistics.median(costs) if costs else "",
                    f"{args.metric}_mean": blank_if_none(metric_mean),
                    "sadb_vs_base_win_rate": blank_if_none(base_win_rate),
                    "sadb_vs_ays_win_rate": blank_if_none(ays_win_rate),
                }
            )

    fieldnames = [
        "model_asset",
        "solver",
        "nfe",
        "schedule",
        "variant",
        "calibration_cost",
        f"{args.metric}_mean",
        "sadb_vs_base_win_rate",
        "sadb_vs_ays_win_rate",
    ]
    output_path = resolve_repo_path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()

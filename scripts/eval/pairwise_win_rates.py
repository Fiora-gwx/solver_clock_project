#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import resolve_repo_path


BASE_KEYS = ("model_asset", "solver", "nfe", "seed", "prompt_index")
GROUP_KEYS = ("model_asset", "solver", "nfe")
OPTIONAL_GROUP_KEYS = ("guidance_scale",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired schedule win rates from detail scoring CSVs.")
    parser.add_argument("--input-csv", action="append", required=True, help="Detail CSV from score_text_image_outputs.py.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metrics", default="clip_score,image_reward")
    return parser.parse_args()


def load_rows(paths: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_path in paths:
        path = resolve_repo_path(raw_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def parse_metrics(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def schedule_family(schedule: str) -> str:
    normalized = schedule.strip()
    lower = normalized.lower()
    if lower == "base":
        return "base"
    if lower in {"ays", "ays_like"}:
        return "AYS"
    if lower == "legacy_sadb" or lower.startswith("legacy_sadb["):
        return normalized if normalized.startswith("LEGACY_SADB") else "LEGACY_SADB" + normalized[11:]
    if lower == "fp_clock" or lower.startswith("fp_clock["):
        return normalized if normalized.startswith("FP_CLOCK") else "FP_CLOCK" + normalized[8:]
    return normalized


def paired_key(row: dict[str, str], optional_keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in (*BASE_KEYS, *optional_keys))


def group_key(row: dict[str, str], optional_keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in (*GROUP_KEYS, *optional_keys))


def numeric_value(row: dict[str, str], metric: str) -> float | None:
    raw = str(row.get(metric, "")).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def comparison_label(left: str, right: str) -> str:
    return f"{left} vs {right}"


def build_comparisons(schedules: set[str]) -> list[tuple[str, str]]:
    comparisons: list[tuple[str, str]] = []
    if "AYS" in schedules and "base" in schedules:
        comparisons.append(("AYS", "base"))
    adaptive_schedules = sorted(
        schedule
        for schedule in schedules
        if schedule == "LEGACY_SADB"
        or schedule.startswith("LEGACY_SADB[")
        or schedule == "FP_CLOCK"
        or schedule.startswith("FP_CLOCK[")
    )
    for schedule in adaptive_schedules:
        if "base" in schedules:
            comparisons.append((schedule, "base"))
        if "AYS" in schedules:
            comparisons.append((schedule, "AYS"))
    return comparisons


def summarize_deltas(
    *,
    rows_by_schedule: dict[str, dict[tuple[str, ...], dict[str, str]]],
    left_schedule: str,
    right_schedule: str,
    metric: str,
) -> dict[str, Any] | None:
    left_rows = rows_by_schedule[left_schedule]
    right_rows = rows_by_schedule[right_schedule]
    common_keys = sorted(set(left_rows) & set(right_rows))
    deltas: list[float] = []
    wins = 0.0
    for key in common_keys:
        left_value = numeric_value(left_rows[key], metric)
        right_value = numeric_value(right_rows[key], metric)
        if left_value is None or right_value is None:
            continue
        delta = left_value - right_value
        deltas.append(delta)
        if delta > 0.0:
            wins += 1.0
        elif delta == 0.0:
            wins += 0.5
    if not deltas:
        return None
    return {
        "num_pairs": len(deltas),
        "win_rate": wins / len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "median_delta": statistics.median(deltas),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    metrics = parse_metrics(args.metrics)
    optional_keys = tuple(key for key in OPTIONAL_GROUP_KEYS if any(str(row.get(key, "")).strip() for row in rows))

    grouped: dict[tuple[str, ...], dict[str, dict[tuple[str, ...], dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        schedule = schedule_family(str(row.get("schedule", "")))
        if not schedule:
            continue
        grouped[group_key(row, optional_keys)][schedule][paired_key(row, optional_keys)] = row

    output_rows: list[dict[str, Any]] = []
    for key, rows_by_schedule in sorted(grouped.items()):
        group_values = dict(zip((*GROUP_KEYS, *optional_keys), key))
        for left_schedule, right_schedule in build_comparisons(set(rows_by_schedule)):
            for metric in metrics:
                summary = summarize_deltas(
                    rows_by_schedule=rows_by_schedule,
                    left_schedule=left_schedule,
                    right_schedule=right_schedule,
                    metric=metric,
                )
                if summary is None:
                    continue
                output_rows.append(
                    {
                        "comparison": comparison_label(left_schedule, right_schedule),
                        "metric": metric,
                        "model_asset": group_values["model_asset"],
                        "solver": group_values["solver"],
                        "nfe": group_values["nfe"],
                        **{key: group_values[key] for key in optional_keys},
                        **summary,
                    }
                )

    fieldnames = [
        "comparison",
        "metric",
        "model_asset",
        "solver",
        "nfe",
        *optional_keys,
        "num_pairs",
        "win_rate",
        "mean_delta",
        "median_delta",
    ]
    output_path = resolve_repo_path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()

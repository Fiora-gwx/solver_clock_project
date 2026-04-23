#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import dump_yaml, load_yaml, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose the V_a ablation winner config from stage-A metrics.")
    parser.add_argument("--selection-config", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_selection_variants(path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    experiment = load_yaml(path)
    variants = experiment.get("clock_variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("selection config must define non-empty `clock_variants`.")

    baseline: dict[str, Any] | None = None
    factor_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in variants:
        if not isinstance(entry, dict):
            raise TypeError("Each `clock_variants` entry must be a mapping.")
        if entry.get("baseline"):
            baseline = entry
            continue
        group = entry.get("factor_group")
        if group:
            factor_groups[str(group)].append(entry)

    if baseline is None:
        raise ValueError("selection config must include one `clock_variants` entry with `baseline: true`.")
    return experiment, baseline, factor_groups


def load_metric_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with resolve_repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("fid"):
                continue
            rows.append(
                {
                    "schedule": row["schedule"],
                    "solver": row.get("solver", ""),
                    "nfe": int(row["nfe"]),
                    "fid": float(row["fid"]),
                }
            )
    if not rows:
        raise ValueError("No FID rows were found in the metrics CSV.")
    return rows


def select_group_winner(
    rows: list[dict[str, Any]],
    *,
    baseline_label: str,
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_labels = [baseline_label] + [str(item["label"]) for item in variants]
    schedule_names = {label: f"V_a[{label}]" for label in candidate_labels}
    filtered = [row for row in rows if row["schedule"] in schedule_names.values()]
    if not filtered:
        raise ValueError(f"No rows found for candidates: {candidate_labels}")

    by_nfe: dict[int, dict[str, float]] = defaultdict(dict)
    for row in filtered:
        for label, schedule_name in schedule_names.items():
            if row["schedule"] == schedule_name:
                by_nfe[row["nfe"]][label] = row["fid"]

    rank_totals: dict[str, float] = defaultdict(float)
    fid_totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for nfe, nfe_rows in by_nfe.items():
        missing = [label for label in candidate_labels if label not in nfe_rows]
        if missing:
            raise ValueError(f"Missing candidate rows at nfe={nfe}: {missing}")
        ranked = sorted(candidate_labels, key=lambda label: (nfe_rows[label], label))
        for rank, label in enumerate(ranked, start=1):
            rank_totals[label] += rank
            fid_totals[label] += nfe_rows[label]
            counts[label] += 1

    scored = sorted(
        candidate_labels,
        key=lambda label: (
            rank_totals[label] / counts[label],
            fid_totals[label] / counts[label],
            label,
        ),
    )
    winner_label = scored[0]
    if winner_label == baseline_label:
        return {"label": baseline_label, "factor_key": None, "factor_value": None}
    for entry in variants:
        if str(entry["label"]) == winner_label:
            return entry
    raise RuntimeError(f"Resolved winner {winner_label} was not found in variant entries.")


def main() -> None:
    args = parse_args()
    _experiment, baseline_entry, factor_groups = load_selection_variants(args.selection_config)
    rows = load_metric_rows(args.metrics_csv)

    baseline_path = str(baseline_entry["path"])
    payload = load_yaml(baseline_path)
    clock = payload.setdefault("clock", {})
    if not isinstance(clock, dict):
        raise TypeError("Baseline clock config must contain a `clock` mapping.")

    baseline_label = str(baseline_entry["label"])
    winners: dict[str, str] = {}
    for group_name, group_variants in factor_groups.items():
        winner = select_group_winner(rows, baseline_label=baseline_label, variants=group_variants)
        if winner.get("factor_key") is not None:
            clock[str(winner["factor_key"])] = winner["factor_value"]
        winners[group_name] = str(winner["label"])

    dump_yaml(payload, args.output)
    print(f"[va-ablation] wrote {args.output}")
    for group_name in sorted(winners):
        print(f"  {group_name}: {winners[group_name]}")


if __name__ == "__main__":
    main()

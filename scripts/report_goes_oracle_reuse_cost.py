#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import resolve_repo_path


FIELDNAMES = (
    "status",
    "oracle_cache_key",
    "backend",
    "dataset",
    "model_asset",
    "prompt_asset",
    "guidance_scale",
    "seed",
    "solver",
    "nfe",
    "schedule_dir",
    "schedule_hash",
    "edge_objective",
    "oracle_loaded_from_cache",
    "oracle_build_or_load_seconds",
    "per_schedule_oracle_model_eval_equivalents",
    "per_schedule_edge_model_eval_equivalents",
    "per_schedule_total_model_eval_equivalents",
    "schedules_sharing_cache_count",
    "solvers_sharing_cache",
    "nfes_sharing_cache",
    "shared_oracle_model_eval_equivalents",
    "separate_oracle_model_eval_equivalents",
    "shared_total_model_eval_equivalents",
    "separate_total_model_eval_equivalents",
    "saved_model_eval_equivalents",
    "shared_oracle_amortized_model_eval_equivalents",
    "note",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize GPDE/GOES schedule materialization cost and oracle reuse from exported schedule directories."
        )
    )
    parser.add_argument("roots", nargs="+", help="Schedule directories or roots containing GPDE/GOES schedule.json files.")
    parser.add_argument("--output-csv", required=True, help="Path for oracle_reuse_cost.csv.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def discover_schedule_jsons(roots: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = resolve_repo_path(root)
        candidates = [resolved] if resolved.name == "schedule.json" else sorted(resolved.rglob("schedule.json"))
        for candidate in candidates:
            if candidate not in seen:
                paths.append(candidate)
                seen.add(candidate)
    return paths


def finite_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def int_or_empty(value: Any) -> int | str:
    if value in ("", None):
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def str_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def infer_backend(schedule: dict[str, Any], run_metadata: dict[str, Any]) -> str:
    explicit = schedule.get("backend") or run_metadata.get("backend")
    if explicit:
        return str(explicit)
    if schedule.get("pipeline_kind") or schedule.get("prompt_asset"):
        return "diffusers"
    if schedule.get("dataset"):
        return "pndm"
    return ""


def cost_from_breakdown(breakdown: dict[str, Any]) -> tuple[int, int, int] | None:
    if not isinstance(breakdown, dict):
        return None
    total = int_or_empty(breakdown.get("total_model_eval_equivalents"))
    if total == "":
        return None
    num_samples = int_or_empty(breakdown.get("num_samples"))
    cfg_multiplier = int_or_empty(breakdown.get("cfg_multiplier"))
    oracle_cost_per_sample = int_or_empty(breakdown.get("oracle_cost_per_sample"))
    edge_cost_per_sample = int_or_empty(breakdown.get("edge_cost_per_sample", breakdown.get("probe_cost_per_sample")))
    if "" in {num_samples, cfg_multiplier, oracle_cost_per_sample, edge_cost_per_sample}:
        return None
    oracle_cost = int(num_samples) * int(cfg_multiplier) * int(oracle_cost_per_sample)
    edge_cost = int(num_samples) * int(cfg_multiplier) * int(edge_cost_per_sample)
    return oracle_cost, edge_cost, int(total)


def row_from_schedule(schedule_path: Path) -> dict[str, Any] | None:
    schedule = load_json(schedule_path)
    if str(schedule.get("method", "")).upper() not in {"GPDE", "GOES"}:
        return None
    run_metadata_path = schedule_path.parent / "run_metadata.json"
    run_metadata = load_json(run_metadata_path) if run_metadata_path.exists() else {}
    breakdown = schedule.get("calibration_cost_breakdown", {})
    costs = cost_from_breakdown(breakdown)
    status = "OK" if costs is not None and schedule.get("oracle_cache_key") else "MISSING_COST_METADATA"
    oracle_cost, edge_cost, total_cost = costs if costs is not None else ("", "", "")
    context = schedule.get("calibration_config", {})
    return {
        "status": status,
        "oracle_cache_key": str_or_empty(schedule.get("oracle_cache_key")),
        "backend": infer_backend(schedule, run_metadata),
        "dataset": str_or_empty(schedule.get("dataset")),
        "model_asset": str_or_empty(schedule.get("model_asset")),
        "prompt_asset": str_or_empty(schedule.get("prompt_asset")),
        "guidance_scale": str_or_empty(schedule.get("guidance_scale")),
        "seed": int_or_empty(schedule.get("seed") or context.get("seed")),
        "solver": str_or_empty(schedule.get("solver")),
        "nfe": int_or_empty(schedule.get("target_nfe")),
        "schedule_dir": str(schedule_path.parent),
        "schedule_hash": str_or_empty(schedule.get("schedule_hash")),
        "edge_objective": str_or_empty(schedule.get("edge_objective")),
        "oracle_loaded_from_cache": str_or_empty(run_metadata.get("oracle_loaded_from_cache")),
        "oracle_build_or_load_seconds": str_or_empty(run_metadata.get("oracle_build_or_load_seconds")),
        "per_schedule_oracle_model_eval_equivalents": oracle_cost,
        "per_schedule_edge_model_eval_equivalents": edge_cost,
        "per_schedule_total_model_eval_equivalents": total_cost,
        "note": "" if status == "OK" else "Missing oracle cache key or calibration_cost_breakdown.",
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("oracle_cache_key", ""))
        if key:
            grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        usable = [row for row in group if row.get("status") == "OK"]
        oracle_costs = [int(row["per_schedule_oracle_model_eval_equivalents"]) for row in usable]
        edge_costs = [int(row["per_schedule_edge_model_eval_equivalents"]) for row in usable]
        total_costs = [int(row["per_schedule_total_model_eval_equivalents"]) for row in usable]
        group_status = "OK" if len(usable) == len(group) and usable else "INCOMPLETE"
        if not usable:
            group_values = {
                "schedules_sharing_cache_count": len(group),
                "solvers_sharing_cache": ",".join(sorted({str(row.get("solver", "")) for row in group if row.get("solver")})),
                "nfes_sharing_cache": ",".join(sorted({str(row.get("nfe", "")) for row in group if row.get("nfe")})),
                "shared_oracle_model_eval_equivalents": "",
                "separate_oracle_model_eval_equivalents": "",
                "shared_total_model_eval_equivalents": "",
                "separate_total_model_eval_equivalents": "",
                "saved_model_eval_equivalents": "",
                "shared_oracle_amortized_model_eval_equivalents": "",
                "note": "No usable cost metadata for this oracle cache key.",
            }
        else:
            unique_oracle_costs = sorted(set(oracle_costs))
            shared_oracle_cost = max(unique_oracle_costs)
            separate_oracle_cost = sum(oracle_costs)
            shared_total = shared_oracle_cost + sum(edge_costs)
            separate_total = sum(total_costs)
            note = ""
            if len(unique_oracle_costs) > 1:
                group_status = "INCONSISTENT_ORACLE_COST"
                note = "Rows sharing an oracle cache key disagree on oracle cost; using max oracle cost for shared estimate."
            group_values = {
                "schedules_sharing_cache_count": len(group),
                "solvers_sharing_cache": ",".join(sorted({str(row.get("solver", "")) for row in group if row.get("solver")})),
                "nfes_sharing_cache": ",".join(sorted({str(row.get("nfe", "")) for row in group if row.get("nfe")})),
                "shared_oracle_model_eval_equivalents": shared_oracle_cost,
                "separate_oracle_model_eval_equivalents": separate_oracle_cost,
                "shared_total_model_eval_equivalents": shared_total,
                "separate_total_model_eval_equivalents": separate_total,
                "saved_model_eval_equivalents": separate_total - shared_total,
                "shared_oracle_amortized_model_eval_equivalents": shared_total / max(len(group), 1),
                "note": note,
            }
        for row in group:
            combined = dict(row)
            combined["status"] = group_status if row.get("status") == "OK" else row.get("status")
            if not combined.get("note"):
                combined["note"] = group_values["note"]
            combined.update({key: value for key, value in group_values.items() if key != "note"})
            output.append(combined)

    output.sort(
        key=lambda row: (
            str(row.get("oracle_cache_key", "")),
            str(row.get("solver", "")),
            int(row["nfe"]) if isinstance(row.get("nfe"), int) else 0,
            str(row.get("schedule_dir", "")),
        )
    )
    return output


def build_report(roots: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = [row for path in discover_schedule_jsons(roots) if (row := row_from_schedule(path)) is not None]
    if not rows:
        raise ValueError("No GOES schedule.json files found under the provided roots (GPDE aliases are accepted).")
    report = summarize_rows(rows)
    if not any(row.get("status") in {"OK", "INCONSISTENT_ORACLE_COST"} for row in report):
        raise ValueError("No GOES schedules with usable oracle reuse cost metadata were found (GPDE aliases are accepted).")
    return report


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = resolve_repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_report(args.roots)
    write_csv(args.output_csv, rows)
    print(json.dumps({"output_csv": str(resolve_repo_path(args.output_csv)), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

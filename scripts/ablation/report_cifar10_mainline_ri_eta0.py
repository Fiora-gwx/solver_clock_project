#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clock.archive.ri_sadb import RI_SADB_CLOCK_VERSION, RI_SADB_FORMULA_VERSION
from src.utils.config import load_yaml, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the CIFAR-10 mainline RI-SADB eta=0 comparison report.")
    parser.add_argument("--old-mainline-csv", default="outputs/cifar10_mainline/metrics/cifar10_mainline.csv")
    parser.add_argument("--ri-csv", default="outputs/cifar10_mainline_ri_eta0/metrics/cifar10_mainline_ri_eta0.csv")
    parser.add_argument("--experiment-config", default="configs/ablation/experiments/cifar10_mainline_ri_eta0.yaml")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--schedule-cache-root",
        default=None,
        help="Defaults to the experiment config schedule cache root.",
    )
    parser.add_argument("--output-report", default="outputs/cifar10_mainline_ri_eta0/reports/cifar10_mainline_ri_eta0_report.md")
    parser.add_argument("--strict", action="store_true", default=False)
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def as_float(value: Any) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def schedule_family(value: str) -> str:
    return value.split("[", 1)[0]


def ri_label_from_spec(value: str) -> str:
    if value.endswith("]") and "[" in value:
        return value[:-1].rsplit("[", 1)[1]
    return ""


def display_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def latest_metric_map(rows: list[dict[str, str]], *, schedule: str) -> dict[tuple[str, str, int], dict[str, str]]:
    result: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        if schedule_family(str(row.get("schedule", ""))) != schedule:
            continue
        nfe = as_int(row.get("nfe"))
        if nfe is None:
            continue
        key = (str(row.get("model_asset", "")), str(row.get("solver", "")), nfe)
        result[key] = row
    return result


def schedule_specs_for_solver(config: dict[str, Any], solver: str) -> list[str]:
    default = config.get("schedules") or config.get("variants") or [config["schedule"]]
    return [str(item) for item in config.get("solver_schedules", {}).get(solver, default)]


def expected_ri_entries(config: dict[str, Any]) -> list[tuple[str, str, int, str]]:
    models = [str(item) for item in config["model_assets"]]
    nfes = [int(item) for item in config.get("eval_nfes", config.get("nfes", []))]
    entries: list[tuple[str, str, int, str]] = []
    for solver in [str(item) for item in config["solvers"]]:
        labels = [
            ri_label_from_spec(spec)
            for spec in schedule_specs_for_solver(config, solver)
            if schedule_family(spec) == "RI_SADB"
        ]
        for label in labels:
            for model in models:
                for nfe in nfes:
                    entries.append((model, solver, nfe, label))
    return entries


def infer_schedule_cache_root(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    raw = execution.get("schedule_cache_root", config.get("schedule_cache_root"))
    if raw is None:
        raw = Path("outputs") / str(config["name"]) / "schedules"
    return Path(raw)


def expected_ri_output_dir(
    *,
    outputs_root: Path,
    experiment_name: str,
    model_asset: str,
    solver: str,
    label: str,
    nfe: int,
) -> Path:
    return (
        outputs_root
        / experiment_name
        / "samples"
        / "pndm"
        / "cifar10"
        / model_asset
        / solver
        / "RI_SADB"
        / label
        / f"nfe_{nfe:03d}"
    )


def expected_ri_schedule_dir(
    *,
    schedule_cache_root: Path,
    model_asset: str,
    solver: str,
    label: str,
    nfe: int,
) -> Path:
    return (
        schedule_cache_root
        / "RI_SADB"
        / "pndm"
        / "cifar10"
        / model_asset
        / solver
        / label
        / f"nfe_{nfe:03d}"
    )


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def array_is_monotone(path: Path) -> bool:
    values = np.load(path)
    if values.ndim != 1 or len(values) < 2:
        return True
    deltas = np.diff(values.astype(np.float64))
    return bool(np.all(deltas <= 1.0e-12) or np.all(deltas >= -1.0e-12))


def validate_ri_outputs(
    *,
    config: dict[str, Any],
    ri_rows: dict[tuple[str, str, int], dict[str, str]],
    outputs_root: Path,
    schedule_cache_root: Path,
) -> list[str]:
    issues: list[str] = []
    expected = expected_ri_entries(config)
    for model_asset, solver, nfe, label in expected:
        key = (model_asset, solver, nfe)
        row = ri_rows.get(key)
        if row is None:
            issues.append(f"missing RI metrics row: model={model_asset} solver={solver} nfe={nfe}")
            continue
        fid = as_float(row.get("fid"))
        if fid is None:
            issues.append(f"non-numeric RI FID: model={model_asset} solver={solver} nfe={nfe}")
        if as_int(row.get("num_samples")) != 50000:
            issues.append(
                f"unexpected RI num_samples: model={model_asset} solver={solver} nfe={nfe} "
                f"value={row.get('num_samples')}"
            )

        output_dir = expected_ri_output_dir(
            outputs_root=outputs_root,
            experiment_name=str(config["name"]),
            model_asset=model_asset,
            solver=solver,
            label=label,
            nfe=nfe,
        )
        manifest = load_manifest(resolve_repo_path(output_dir) / "run_manifest.json")
        schedule_dir = None
        if manifest is not None and manifest.get("schedule_dir"):
            schedule_dir = Path(str(manifest["schedule_dir"]))
        if schedule_dir is None:
            schedule_dir = expected_ri_schedule_dir(
                schedule_cache_root=schedule_cache_root,
                model_asset=model_asset,
                solver=solver,
                label=label,
                nfe=nfe,
            )
        resolved_schedule_dir = resolve_repo_path(schedule_dir)
        meta_path = resolved_schedule_dir / "meta.json"
        if not meta_path.exists():
            issues.append(f"missing RI schedule metadata: {schedule_dir}")
            continue
        meta = load_manifest(meta_path)
        expected_calibration_solver = "heun2" if label == "eta0_target_heun2" else "euler"
        checks = {
            "schedule_family": "RI_SADB",
            "eta": 0.0,
            "beta": 0.0,
            "calibration_solver": expected_calibration_solver,
            "ri_formula_version": RI_SADB_FORMULA_VERSION,
            "schedule_implementation_version": RI_SADB_CLOCK_VERSION,
        }
        assert meta is not None
        for field, expected_value in checks.items():
            if meta.get(field) != expected_value:
                issues.append(
                    f"RI metadata mismatch for {schedule_dir}: {field}={meta.get(field)!r}, "
                    f"expected {expected_value!r}"
                )
        for array_name in ("timesteps.npy", "time_grid.npy", "sigma_grid.npy"):
            array_path = resolved_schedule_dir / array_name
            if array_path.exists() and not array_is_monotone(array_path):
                issues.append(f"non-monotone RI schedule array: {array_path}")
    return issues


def winner(base: float | None, sadb: float | None, ri: float | None) -> str:
    candidates = [("base", base), ("SADB", sadb), ("RI_eta0", ri)]
    numeric = [(name, value) for name, value in candidates if value is not None]
    if len(numeric) != 3:
        return "NA"
    return min(numeric, key=lambda item: item[1])[0]


def aggregate_rows(rows: list[dict[str, Any]], key_name: str) -> list[list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["base_fid"] is None or row["sadb_fid"] is None or row["ri_fid"] is None:
            continue
        grouped[str(row[key_name])].append(row)
    table_rows: list[list[str]] = []
    sort_key = (lambda value: (as_int(value, 10**9), value)) if key_name == "nfe" else (lambda value: value)
    for key in sorted(grouped, key=sort_key):
        group = grouped[key]
        ri_wins = sum(1 for item in group if item["winner"] == "RI_eta0")
        table_rows.append(
            [
                key,
                str(len(group)),
                display_float(mean(item["base_fid"] for item in group)),
                display_float(mean(item["sadb_fid"] for item in group)),
                display_float(mean(item["ri_fid"] for item in group)),
                display_float(mean(item["ri_minus_sadb"] for item in group)),
                display_float(mean(item["ri_minus_base"] for item in group)),
                f"{ri_wins}/{len(group)}",
            ]
        )
    return table_rows


def main() -> None:
    args = parse_args()
    config = load_yaml(args.experiment_config)
    old_rows = read_rows(args.old_mainline_csv)
    new_rows = read_rows(args.ri_csv)
    base_rows = latest_metric_map(old_rows, schedule="base")
    sadb_rows = latest_metric_map(old_rows, schedule="SADB")
    ri_rows = latest_metric_map(new_rows, schedule="RI_SADB")

    comparison_rows: list[dict[str, Any]] = []
    for model_asset, solver, nfe, _label in expected_ri_entries(config):
        key = (model_asset, solver, nfe)
        base_fid = as_float(base_rows.get(key, {}).get("fid"))
        sadb_fid = as_float(sadb_rows.get(key, {}).get("fid"))
        ri_fid = as_float(ri_rows.get(key, {}).get("fid"))
        row_winner = winner(base_fid, sadb_fid, ri_fid)
        comparison_rows.append(
            {
                "model": model_asset,
                "solver": solver,
                "nfe": nfe,
                "base_fid": base_fid,
                "sadb_fid": sadb_fid,
                "ri_fid": ri_fid,
                "ri_minus_sadb": None if ri_fid is None or sadb_fid is None else ri_fid - sadb_fid,
                "ri_minus_base": None if ri_fid is None or base_fid is None else ri_fid - base_fid,
                "winner": row_winner,
            }
        )

    outputs_root = Path(args.outputs_root)
    schedule_cache_root = infer_schedule_cache_root(config, args.schedule_cache_root)
    validation_issues = validate_ri_outputs(
        config=config,
        ri_rows=ri_rows,
        outputs_root=outputs_root,
        schedule_cache_root=schedule_cache_root,
    )

    detail_table = markdown_table(
        [
            "model",
            "solver",
            "nfe",
            "base_fid",
            "SADB_fid",
            "RI_eta0_fid",
            "RI_minus_SADB",
            "RI_minus_base",
            "winner",
        ],
        (
            [
                str(row["model"]),
                str(row["solver"]),
                str(row["nfe"]),
                display_float(row["base_fid"]),
                display_float(row["sadb_fid"]),
                display_float(row["ri_fid"]),
                display_float(row["ri_minus_sadb"]),
                display_float(row["ri_minus_base"]),
                str(row["winner"]),
            ]
            for row in comparison_rows
        ),
    )
    aggregate_headers = [
        "group",
        "rows",
        "mean_base_fid",
        "mean_SADB_fid",
        "mean_RI_eta0_fid",
        "mean_RI_minus_SADB",
        "mean_RI_minus_base",
        "RI_wins",
    ]
    solver_table = markdown_table(aggregate_headers, aggregate_rows(comparison_rows, "solver"))
    model_table = markdown_table(aggregate_headers, aggregate_rows(comparison_rows, "model"))
    nfe_table = markdown_table(aggregate_headers, aggregate_rows(comparison_rows, "nfe"))

    complete_rows = sum(
        1
        for row in comparison_rows
        if row["base_fid"] is not None and row["sadb_fid"] is not None and row["ri_fid"] is not None
    )
    report = "\n\n".join(
        [
            "# CIFAR-10 Mainline RI-SADB eta=0 Report",
            (
                f"Base and SADB values are read from `{args.old_mainline_csv}`. "
                f"RI-SADB eta=0 values are read from `{args.ri_csv}`. "
                f"Complete comparison rows: {complete_rows}/{len(comparison_rows)}."
            ),
            (
                "STORK note: `stork4_1st` and `stork4_2nd` RI rows use the fixed Euler proxy defect "
                "(`eta0_fixed_euler_proxy`). They are not native STORK target short-window defect results."
            ),
            "## Detail\n\n" + detail_table,
            "## Aggregate by Solver\n\n" + solver_table,
            "## Aggregate by Model\n\n" + model_table,
            "## Aggregate by NFE\n\n" + nfe_table,
            "## Validation\n\n"
            + (
                "No RI output validation issues found."
                if not validation_issues
                else "\n".join(f"- {issue}" for issue in validation_issues)
            ),
        ]
    )
    output_path = resolve_repo_path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"[report] wrote {output_path}")

    if args.strict and validation_issues:
        raise SystemExit("RI output validation failed; see report for details.")


if __name__ == "__main__":
    main()

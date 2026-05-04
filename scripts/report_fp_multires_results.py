#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import resolve_repo_path


REQUIRED_FIELDS = (
    "backend",
    "dataset",
    "model",
    "model_asset",
    "solver",
    "schedule",
    "clock_label",
    "clock_family",
    "estimator",
    "nfe",
    "seed",
    "num_samples",
    "metric_name",
    "metric_value",
    "status",
    "error",
    "schedule_dir",
    "output_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FP_CLOCK[anchored_replay] smoke outputs.")
    parser.add_argument("--metrics-root", required=True)
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--command", action="append", default=[])
    return parser.parse_args()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--oneline"],
            cwd=resolve_repo_path("."),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"unavailable: {exc}"
    return result.stdout.strip()


def read_csv_rows(metrics_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(metrics_root.rglob("*.csv")):
        name = csv_path.name
        if name.endswith("_detail.csv") or name.endswith("_pairwise.csv") or name.endswith("_normalized.csv"):
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["_source_csv"] = str(csv_path)
                rows.append(row)
    return rows


def clock_fields(schedule: str) -> tuple[str, str, str]:
    label = ""
    if schedule.endswith("]") and "[" in schedule:
        base, label = schedule[:-1].split("[", 1)
    else:
        base = schedule
    if base == "FP_CLOCK":
        return label, "FP_CLOCK", label if label == "anchored_replay" else ""
    if base == "LEGACY_SADB":
        return label, "LEGACY_SADB", ""
    return label, base, ""


def first_present(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return str(value)
    return ""


def parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def normalize_status(row: dict[str, Any]) -> str:
    status = str(row.get("status", "") or "").upper()
    if status:
        return status
    return "FAILED" if str(row.get("error", "") or "") else "OK"


def metric_values(row: dict[str, Any]) -> list[tuple[str, Any]]:
    explicit_name = str(row.get("metric_name", "") or "")
    explicit_value = row.get("metric_value", "")
    metrics: list[tuple[str, Any]] = []
    if explicit_name:
        metrics.append((explicit_name, explicit_value))
    fid = first_present(row, "fid")
    if fid:
        metrics.append(("fid", fid))
    clip = first_present(row, "clip_score", "clip_score_mean")
    if clip:
        metrics.append(("clip_score", clip))
    reward = first_present(row, "image_reward", "image_reward_mean")
    if reward:
        metrics.append(("image_reward", reward))
    if not metrics:
        metrics.append((explicit_name, explicit_value))
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, Any]] = []
    for name, value in metrics:
        key = (str(name), str(value))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, value))
    return unique


def normalize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        schedule = first_present(row, "schedule")
        clock_label, clock_family, estimator = clock_fields(schedule)
        model_asset = first_present(row, "model_asset", "model")
        base = {
            "backend": first_present(row, "backend"),
            "dataset": first_present(row, "dataset"),
            "model": first_present(row, "model", "model_asset"),
            "model_asset": model_asset,
            "solver": first_present(row, "solver"),
            "schedule": schedule,
            "clock_label": first_present(row, "clock_label") or clock_label,
            "clock_family": first_present(row, "clock_family") or clock_family,
            "estimator": first_present(row, "estimator") or estimator,
            "nfe": first_present(row, "nfe"),
            "seed": first_present(row, "seed"),
            "guidance_scale": first_present(row, "guidance_scale"),
            "num_samples": first_present(row, "num_samples", "num_images"),
            "status": normalize_status(row),
            "error": first_present(row, "error"),
            "schedule_dir": first_present(row, "schedule_dir"),
            "output_dir": first_present(row, "output_dir"),
            "fid": first_present(row, "fid"),
            "clip_score": first_present(row, "clip_score", "clip_score_mean"),
            "image_reward": first_present(row, "image_reward", "image_reward_mean"),
            "_source_csv": first_present(row, "_source_csv"),
        }
        for metric_name, metric_value in metric_values(row):
            normalized = {
                **base,
                "metric_name": str(metric_name),
                "metric_value": "" if metric_value is None else metric_value,
            }
            identity = tuple(
                str(normalized.get(key, ""))
                for key in (
                    "backend",
                    "dataset",
                    "model_asset",
                    "solver",
                    "schedule",
                    "nfe",
                    "seed",
                    "guidance_scale",
                    "metric_name",
                )
            )
            existing = normalized_by_key.get(identity)
            if existing is None or (
                not parse_float(existing.get("metric_value")) and parse_float(normalized.get("metric_value")) is not None
            ):
                normalized_by_key[identity] = normalized
    return list(normalized_by_key.values())


def write_normalized_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(REQUIRED_FIELDS) + ["guidance_scale", "fid", "clip_score", "image_reward", "_source_csv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows: Iterable[dict[str, Any]], *, schedule: str, metric: str) -> float | None:
    values = [
        value
        for row in rows
        if row.get("schedule") == schedule
        and row.get("metric_name") == metric
        and row.get("status") == "OK"
        for value in [parse_float(row.get("metric_value"))]
        if value is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def schedule_order(schedule: str) -> int:
    order = {
        "base": 0,
        "AYS": 1,
        "ays": 1,
        "LEGACY_SADB": 2,
        "FP_CLOCK[anchored_replay]": 3,
        "FP_CLOCK": 3,
    }
    return order.get(schedule, 50)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def pndm_solver_table(rows: list[dict[str, Any]]) -> list[list[str]]:
    table: list[list[str]] = []
    pndm_rows = [row for row in rows if row.get("backend") == "pndm"]
    for solver in sorted({str(row.get("solver", "")) for row in pndm_rows if row.get("solver")}):
        solver_rows = [row for row in pndm_rows if row.get("solver") == solver]
        base = mean_metric(solver_rows, schedule="base", metric="fid")
        legacy = mean_metric(solver_rows, schedule="LEGACY_SADB", metric="fid")
        fp = mean_metric(solver_rows, schedule="FP_CLOCK[anchored_replay]", metric="fid")
        failures = [row for row in solver_rows if row.get("status") == "FAILED"]
        status = "FAILED" if failures and fp is None else ("PARTIAL" if failures else "OK")
        table.append(
            [
                "pndm",
                solver,
                fmt(base, 3),
                fmt(legacy, 3) if legacy is not None else "not run",
                fmt(fp, 3),
                fmt(None if base is None or fp is None else fp - base, 3),
                fmt(None if legacy is None or fp is None else fp - legacy, 3) if legacy is not None else "not run",
                status,
            ]
        )
    return table


def diffusers_model_solver_table(rows: list[dict[str, Any]]) -> list[list[str]]:
    table: list[list[str]] = []
    diff_rows = [row for row in rows if row.get("backend") == "diffusers"]
    keys = sorted({(str(row.get("model_asset", "")), str(row.get("solver", ""))) for row in diff_rows})
    for model, solver in keys:
        group = [row for row in diff_rows if row.get("model_asset") == model and row.get("solver") == solver]
        base_clip = mean_metric(group, schedule="base", metric="clip_score")
        base_reward = mean_metric(group, schedule="base", metric="image_reward")
        fp_clip = mean_metric(group, schedule="FP_CLOCK[anchored_replay]", metric="clip_score")
        fp_reward = mean_metric(group, schedule="FP_CLOCK[anchored_replay]", metric="image_reward")
        failures = [row for row in group if row.get("status") == "FAILED"]
        status = "FAILED" if failures and fp_clip is None and fp_reward is None else ("PARTIAL" if failures else "OK")
        table.append(
            [
                model,
                solver,
                fmt(base_clip),
                fmt(base_reward),
                fmt(fp_clip),
                fmt(fp_reward),
                fmt(None if base_clip is None or fp_clip is None else fp_clip - base_clip),
                fmt(None if base_reward is None or fp_reward is None else fp_reward - base_reward),
                status,
            ]
        )
    return table


def solver_schedule_matrix(rows: list[dict[str, Any]]) -> list[list[str]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        backend = str(row.get("backend", ""))
        solver = str(row.get("solver", ""))
        schedule = str(row.get("schedule", ""))
        if backend and solver and schedule:
            grouped[(backend, solver)].add(schedule)
    return [
        [backend, solver, ", ".join(sorted(schedules, key=schedule_order))]
        for (backend, solver), schedules in sorted(grouped.items())
    ]


def generation_status_table(rows: list[dict[str, Any]]) -> list[list[str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("backend", "")), str(row.get("solver", "")))].append(row)
    table: list[list[str]] = []
    for (backend, solver), group in sorted(grouped.items()):
        ok = len({(row.get("schedule"), row.get("nfe"), row.get("model_asset"), row.get("dataset")) for row in group if row.get("status") == "OK"})
        failed = len({(row.get("schedule"), row.get("nfe"), row.get("model_asset"), row.get("dataset")) for row in group if row.get("status") == "FAILED"})
        metric_ok = len(
            {
                (row.get("schedule"), row.get("nfe"), row.get("model_asset"), row.get("dataset"), row.get("metric_name"))
                for row in group
                if row.get("status") == "OK" and parse_float(row.get("metric_value")) is not None
            }
        )
        table.append([backend, solver, str(ok), str(failed), str(metric_ok)])
    return table


def find_fp_meta(samples_root: Path) -> list[Path]:
    roots = [samples_root, samples_root.parent]
    found: dict[Path, None] = {}
    for root in roots:
        if root.exists():
            for path in root.rglob("meta.json"):
                if "FP_CLOCK" in str(path) and "_profile_cache" not in path.parts:
                    found[path] = None
    return sorted(found)


def recursive_terms(value: Any, terms: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            for term in terms:
                key_text = str(key)
                if (term == "eta" and key_text == "eta") or (term != "eta" and term in key_text):
                    hits.append(term)
            hits.extend(recursive_terms(item, terms))
    elif isinstance(value, list):
        for item in value:
            hits.extend(recursive_terms(item, terms))
    elif isinstance(value, str):
        for term in terms:
            if (term == "eta" and value == "eta") or (term != "eta" and term in value):
                hits.append(term)
    return hits


def load_numpy_array(path: Path):
    try:
        import numpy as np
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def array_is_finite(path: Path) -> bool | None:
    array = load_numpy_array(path)
    if array is None:
        return None
    import numpy as np

    return bool(np.all(np.isfinite(array)))


def array_is_monotone(path: Path) -> bool | None:
    array = load_numpy_array(path)
    if array is None:
        return None
    import numpy as np

    flat = np.asarray(array, dtype=float).reshape(-1)
    if flat.size < 2:
        return True
    diffs = np.diff(flat)
    return bool(np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9))


def inspect_fp_meta(samples_root: Path) -> dict[str, Any]:
    paths = find_fp_meta(samples_root)
    required_failures: list[str] = []
    forbidden_hits: list[str] = []
    monotone_failures: list[str] = []
    finite_failures: list[str] = []
    prepare_by_solver: dict[str, int] = defaultdict(int)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        solver = str(meta.get("solver") or meta.get("target_solver") or "")
        if solver:
            prepare_by_solver[solver] += 1
        expected = {
            "schedule_family": "FP_CLOCK",
            "estimator": "anchored_replay",
            "calibration_nfes": [16],
        }
        for key, expected_value in expected.items():
            if meta.get(key) != expected_value:
                required_failures.append(f"{path}: {key}={meta.get(key)!r}")
        for key in ("native_coordinate", "window_len", "heun_omitted"):
            if key not in meta:
                required_failures.append(f"{path}: missing {key}")
        for term in sorted(set(recursive_terms(meta, ("eta", "RI_G", "RI_SADB", "n_geometry", "curvature")))):
            forbidden_hits.append(f"{path}: {term}")
        for array_name in ("tau_grid.npy", "time_grid.npy", "sigmas.npy", "sigma_grid.npy"):
            monotone = array_is_monotone(path.parent / array_name)
            if monotone is False:
                monotone_failures.append(f"{path.parent / array_name}")
        for array_name in ("tau_grid.npy", "g_grid.npy", "alpha_grid.npy", "density_grid.npy"):
            finite = array_is_finite(path.parent / array_name)
            if finite is False:
                finite_failures.append(f"{path.parent / array_name}")
    return {
        "count": len(paths),
        "prepare_by_solver": dict(sorted(prepare_by_solver.items())),
        "required_failures": required_failures,
        "forbidden_hits": forbidden_hits,
        "monotone_failures": monotone_failures,
        "finite_failures": finite_failures,
    }


def fp_prepare_table(rows: list[dict[str, Any]], meta_check: dict[str, Any]) -> list[list[str]]:
    prepare_by_solver = meta_check["prepare_by_solver"]
    solvers = sorted({str(row.get("solver", "")) for row in rows if row.get("schedule") == "FP_CLOCK[anchored_replay]"})
    table: list[list[str]] = []
    for solver in solvers:
        fp_rows = [row for row in rows if row.get("solver") == solver and row.get("schedule") == "FP_CLOCK[anchored_replay]"]
        failures = [row for row in fp_rows if row.get("status") == "FAILED"]
        table.append(
            [
                solver,
                "yes" if prepare_by_solver.get(solver, 0) else "no",
                str(prepare_by_solver.get(solver, 0)),
                "yes" if any(row.get("status") == "OK" for row in fp_rows) else "no",
                "yes" if any(parse_float(row.get("metric_value")) is not None for row in fp_rows) else "no",
                str(len(failures)),
            ]
        )
    return table


def failure_table(rows: list[dict[str, Any]], limit: int = 60) -> list[list[str]]:
    failures = [row for row in rows if row.get("status") == "FAILED" or row.get("error")]
    table: list[list[str]] = []
    for row in failures[:limit]:
        error = str(row.get("error", ""))
        if len(error) > 180:
            error = error[:177] + "..."
        table.append(
            [
                row.get("backend", ""),
                row.get("model_asset", ""),
                row.get("solver", ""),
                row.get("schedule", ""),
                row.get("nfe", ""),
                error,
            ]
        )
    return table


def log_error_summary(samples_root: Path, limit: int = 20) -> list[list[str]]:
    patterns = (
        "RuntimeError:",
        "ValueError:",
        "TypeError:",
        "NotImplementedError:",
        "CUDA out of memory",
        "ModuleNotFoundError:",
        "ImportError:",
    )
    counts: dict[str, int] = defaultdict(int)
    log_dirs = [path for path in samples_root.rglob("logs/*") if path.is_dir()]
    search_roots = [max(log_dirs, key=lambda path: path.name)] if log_dirs else [samples_root]
    log_paths = [log_path for root in search_roots for log_path in sorted(root.rglob("*.log"))]
    for log_path in log_paths:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if any(pattern in stripped for pattern in patterns):
                if len(stripped) > 220:
                    stripped = stripped[:217] + "..."
                counts[stripped] += 1
    return [[str(count), message] for message, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def main() -> None:
    args = parse_args()
    metrics_root = resolve_repo_path(args.metrics_root)
    samples_root = resolve_repo_path(args.samples_root)
    output_path = resolve_repo_path(args.output)
    rows = normalize_rows(read_csv_rows(metrics_root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_csv = output_path.with_name(output_path.stem + "_normalized.csv")
    write_normalized_csv(normalized_csv, rows)

    backends = sorted({str(row.get("backend", "")) for row in rows if row.get("backend")})
    solvers = sorted({str(row.get("solver", "")) for row in rows if row.get("solver")})
    schedules = sorted({str(row.get("schedule", "")) for row in rows if row.get("schedule")}, key=schedule_order)
    meta_check = inspect_fp_meta(samples_root)

    command_block = "\n".join(f"- `{command}`" for command in args.command) if args.command else "- not recorded"
    env_lines = [
        f"- report python: `{sys.executable}`",
        f"- metrics root: `{metrics_root}`",
        f"- samples root: `{samples_root}`",
    ]
    if any("sc-pndm" in command for command in args.command):
        env_lines.append("- generation env: `/home/gwx/miniconda3/envs/sc-pndm/bin/python`")
    if any("sc-diff" in command for command in args.command):
        env_lines.append("- generation env: `/home/gwx/miniconda3/envs/sc-diff/bin/python`")

    meta_status = "OK"
    if meta_check["required_failures"] or meta_check["forbidden_hits"] or meta_check["monotone_failures"] or meta_check["finite_failures"]:
        meta_status = "FAILED"

    lines = [
        "# FP Multiresolution Smoke Report",
        "",
        "## Actual Commands",
        command_block,
        "",
        "## Git Commit",
        f"`{git_commit()}`",
        "",
        "## Runtime Env",
        "\n".join(env_lines),
        "",
        "## Coverage",
        f"- backends: {', '.join(backends) if backends else 'none'}",
        f"- solvers: {', '.join(solvers) if solvers else 'none'}",
        f"- schedules: {', '.join(schedules) if schedules else 'none'}",
        f"- normalized CSV: `{normalized_csv}`",
        "",
        "## Solver Schedules",
        markdown_table(["backend", "solver", "schedules"], solver_schedule_matrix(rows)),
        "",
        "## FP Prepare / Generation / Metric Status",
        markdown_table(
            ["solver", "FP prepare", "meta count", "FP generation", "FP metric", "FP failures"],
            fp_prepare_table(rows, meta_check),
        ),
        "",
        "## Generation And Metric Counts",
        markdown_table(["backend", "solver", "generation OK", "generation failed", "metric rows OK"], generation_status_table(rows)),
        "",
    ]

    if "pndm" in backends:
        lines.extend(
            [
                "## PNDM Solver Comparison",
                "Metric is smoke FID over the configured small sample count, not final quality.",
                markdown_table(
                    [
                        "backend",
                        "solver",
                        "base metric",
                        "legacy metric",
                        "FP_CLOCK[anchored_replay] metric",
                        "FP-base",
                        "FP-legacy",
                        "status",
                    ],
                    pndm_solver_table(rows),
                ),
                "",
            ]
        )
    if "diffusers" in backends:
        lines.extend(
            [
                "## Diffusers Model Solver Comparison",
                markdown_table(
                    [
                        "model",
                        "solver",
                        "base CLIP",
                        "base ImageReward",
                        "FP CLIP",
                        "FP ImageReward",
                        "FP-base CLIP",
                        "FP-base ImageReward",
                        "status",
                    ],
                    diffusers_model_solver_table(rows),
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Heun Omission",
            "Heun/flow-heun are omitted for this smoke because the current effective-NFE contract supports vendor Heun only at odd effective NFE, while this smoke uses NFE 6 and 10.",
            "",
            "## FP Metadata Checks",
            f"- FP meta files: {meta_check['count']}",
            f"- status: {meta_status}",
            f"- required field failures: {len(meta_check['required_failures'])}",
            f"- forbidden eta/RI_G/RI_SADB/n_geometry/curvature hits: {len(meta_check['forbidden_hits'])}",
            f"- monotonicity failures: {len(meta_check['monotone_failures'])}",
            f"- finite alpha/density/tau failures: {len(meta_check['finite_failures'])}",
            "",
        ]
    )
    if meta_check["required_failures"]:
        lines.append("Required field failures:")
        lines.extend(f"- `{item}`" for item in meta_check["required_failures"][:30])
        lines.append("")
    if meta_check["forbidden_hits"]:
        lines.append("Forbidden metadata hits:")
        lines.extend(f"- `{item}`" for item in meta_check["forbidden_hits"][:30])
        lines.append("")
    if meta_check["monotone_failures"]:
        lines.append("Monotonicity failures:")
        lines.extend(f"- `{item}`" for item in meta_check["monotone_failures"][:30])
        lines.append("")
    if meta_check["finite_failures"]:
        lines.append("Finite-array failures:")
        lines.extend(f"- `{item}`" for item in meta_check["finite_failures"][:30])
        lines.append("")

    lines.extend(
        [
            "## Failures",
            "### Error Summary From Logs",
            markdown_table(["count", "message"], log_error_summary(samples_root)),
            "",
            "### Failure Rows",
            markdown_table(["backend", "model", "solver", "schedule", "nfe", "error"], failure_table(rows)),
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {normalized_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters.pndm import build_scheduler
from src.utils.config import load_json
from src.utils.nfe_budget import resolve_effective_nfe_plan
from src.utils.schedule_bundle import ScheduleBundle


RESULT_FIELDS = (
    "timestamp",
    "git_commit",
    "device",
    "dataset",
    "model_asset",
    "solver",
    "schedule",
    "nfe",
    "seed",
    "num_samples",
    "batch_size",
    "metric_name",
    "metric_value",
    "metric_available",
    "fid_reference",
    "runtime_sec",
    "status",
    "error",
    "artifact_path",
)

DIAGNOSTIC_FIELDS = (
    "solver",
    "schedule",
    "seed",
    "valid_alpha",
    "valid_tau",
    "monotone_tau",
    "monotone_nodes",
    "alpha_min",
    "alpha_max",
    "alpha_mean",
    "alpha_std",
    "tau_min_step",
    "tau_max_step",
    "node_min_step",
    "node_max_step",
    "step_ratio_max",
    "density_entropy",
    "status",
    "error",
)

AGG_FIELDS = (
    "dataset",
    "model_asset",
    "solver",
    "schedule",
    "nfe",
    "num_samples",
    "num_runs",
    "num_success",
    "success_rate",
    "fid_mean",
    "fid_std",
    "metric_mean",
    "metric_std",
    "runtime_mean",
    "runtime_std",
    "valid_schedule_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FP_CLOCK CIFAR-10 smoke suite.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--nfe", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--include-legacy-sadb", action="store_true")
    parser.add_argument("--include-fp-clock", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--solvers", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric", choices=["fid", "sample_std_proxy", "none"], default="sample_std_proxy")
    parser.add_argument("--compute-fid", action="store_true", default=False)
    parser.add_argument("--reference-stats-asset", default="pndm_fid_cifar10_train")
    parser.add_argument("--save-samples", default="true")
    parser.add_argument("--preview-samples", type=int, default=0)
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--model-asset", default="pndm_model_ddim_cifar10")
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets/cifar10.yaml")
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def command_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"torch_version": f"unavailable: {exc}", "cuda_available": False}
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception as exc:
        torchvision_version = f"unavailable: {exc}"
    return {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def resolve_device(requested: str) -> str:
    env = torch_environment()
    if requested == "auto":
        return "cuda" if bool(env.get("cuda_available")) else "cpu"
    if requested == "cuda" and not bool(env.get("cuda_available")):
        return "cpu"
    return requested


def default_output_dir(raw: str) -> Path:
    if raw:
        return (REPO_ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / f"fp_clock_smoke_{stamp}"


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def normalize_status_error(exc: BaseException) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text[:240] if text else exc.__class__.__name__


def nfe_for_solver(solver: str, requested_nfe: int) -> int:
    try:
        resolve_effective_nfe_plan(solver, requested_nfe)
        return int(requested_nfe)
    except ValueError as exc:
        if solver.lower().replace("-", "_") == "heun2" and "odd" in str(exc).lower():
            adjusted = int(requested_nfe) + 1
            resolve_effective_nfe_plan(solver, adjusted)
            return adjusted
        raise


def available_solvers(requested: list[str] | None) -> tuple[list[str], dict[str, str]]:
    candidates = requested or ["euler", "heun2"]
    solvers: list[str] = []
    mapping: dict[str, str] = {}
    for name in candidates:
        try:
            build_scheduler(name)
        except Exception:
            continue
        solvers.append(name)
        mapping[name] = name
    return solvers, mapping


def schedule_cases(args: argparse.Namespace, solvers: list[str]) -> list[dict[str, Any]]:
    include_all = not (args.include_base or args.include_legacy_sadb or args.include_fp_clock)
    schedules: list[str] = []
    if include_all or args.include_base:
        schedules.append("base")
    if include_all or args.include_legacy_sadb:
        schedules.append("LEGACY_SADB")
    if include_all or args.include_fp_clock:
        schedules.append("FP_CLOCK")
    return [{"solver": solver, "schedule": schedule} for solver in solvers for schedule in schedules]


def write_clock_config(path: Path, *, family: str, target_nfe: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "clock:",
        f"  family: {family}",
        f"  calibration_mode: {'fp_clock' if family == 'FP_CLOCK' else 'legacy_sadb'}",
        "  calibration_solver: target",
        "  model_output_type: epsilon",
        "  physical_grid_size: 17",
        "  pilot_batch_size: 4",
        "  pilot_num_batches: 1",
        "  pilot_observation_microbatch: 2",
        "  smoothing_window: 1",
        "  epsilon: 1.0e-12",
        "  q_min: 1.05",
        "  q_max: 6.0",
        "  warmup_steps: 1",
        f"  target_nfes: [{int(target_nfe)}]",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command: list[str], *, log_path: Path, commands: list[str]) -> None:
    printable = " ".join(command)
    commands.append(printable)
    append_log(log_path, f"$ {printable}")
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    append_log(log_path, process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {printable}")


def export_schedule(
    *,
    args: argparse.Namespace,
    case: dict[str, Any],
    seed: int,
    output_dir: Path,
    log_path: Path,
    commands: list[str],
) -> Path | None:
    if case["schedule"] == "base":
        return None

    case_nfe = nfe_for_solver(case["solver"], int(args.nfe))
    config_dir = output_dir / "clock_configs"
    config_path = config_dir / f"{case['schedule']}_{case['solver']}.yaml"
    write_clock_config(config_path, family=case["schedule"], target_nfe=case_nfe)
    schedule_root = output_dir / "schedules" / case["solver"] / case["schedule"] / f"seed_{seed:03d}"
    command = [
        sys.executable,
        "scripts/run/export_defect_clock_schedule.py",
        "--backend",
        "pndm",
        "--manifest",
        args.manifest,
        "--clock-config",
        str(config_path),
        "--output-root",
        str(schedule_root),
        "--profile-cache-root",
        str(output_dir / "cache" / case["schedule"] / case["solver"] / f"seed_{seed:03d}"),
        "--target-nfes",
        str(case_nfe),
        "--seed",
        str(int(seed)),
        "--solver",
        case["solver"],
        "--dataset-config",
        args.dataset_config,
        "--model-asset",
        args.model_asset,
    ]
    run_command(command, log_path=log_path, commands=commands)
    return schedule_root / f"nfe_{case_nfe:03d}"


def synthetic_nodes_for_base(solver: str, nfe: int) -> np.ndarray:
    try:
        plan = resolve_effective_nfe_plan(solver, nfe)
        steps = max(int(plan.solver_steps), 1)
    except Exception:
        steps = max(int(nfe), 1)
    return np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)


def monotone(values: np.ndarray, tol: float = 1.0e-12) -> bool:
    if values.size < 2:
        return True
    diffs = np.diff(values)
    return bool(np.all(diffs >= -tol) or np.all(diffs <= tol))


def entropy(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return float("nan")
    probs = positive / np.sum(positive)
    return float(-np.sum(probs * np.log(np.maximum(probs, 1.0e-300))))


def schedule_diagnostic(
    *,
    case: dict[str, Any],
    seed: int,
    bundle_dir: Path | None,
    nfe: int,
    status: str = "OK",
    error: str = "",
) -> dict[str, Any]:
    try:
        if bundle_dir is not None and (bundle_dir / "meta.json").exists():
            bundle = ScheduleBundle.load(bundle_dir)
            nodes = bundle.timesteps if bundle.timesteps is not None else bundle.sigmas
            if nodes is None:
                nodes = bundle.time_grid if bundle.time_grid is not None else bundle.sigma_grid
            if nodes is None:
                raise ValueError("bundle has no schedule nodes")
            tau = bundle.tau_grid if bundle.tau_grid is not None else np.linspace(0.0, 1.0, len(nodes), dtype=np.float64)
            alpha = 1.0 / np.maximum(np.asarray(bundle.g_grid, dtype=np.float64), 1.0e-12) if bundle.g_grid is not None else np.ones_like(np.asarray(nodes, dtype=np.float64))
        else:
            nodes = synthetic_nodes_for_base(case["solver"], nfe)
            tau = np.linspace(0.0, 1.0, len(nodes), dtype=np.float64)
            alpha = np.ones_like(nodes, dtype=np.float64)
        nodes = np.asarray(nodes, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        tau_diffs = np.abs(np.diff(tau)) if tau.size > 1 else np.asarray([0.0])
        node_diffs = np.abs(np.diff(nodes)) if nodes.size > 1 else np.asarray([0.0])
        positive_steps = node_diffs[node_diffs > 1.0e-12]
        step_ratio = (
            float(np.max(positive_steps) / max(float(np.min(positive_steps)), 1.0e-12))
            if positive_steps.size
            else 1.0
        )
        return {
            "solver": case["solver"],
            "schedule": case["schedule"],
            "seed": seed,
            "valid_alpha": bool(np.all(np.isfinite(alpha)) and np.all(alpha > 0.0)),
            "valid_tau": bool(np.all(np.isfinite(tau)) and tau.size > 0),
            "monotone_tau": monotone(tau),
            "monotone_nodes": monotone(nodes),
            "alpha_min": float(np.min(alpha)),
            "alpha_max": float(np.max(alpha)),
            "alpha_mean": float(np.mean(alpha)),
            "alpha_std": float(np.std(alpha)),
            "tau_min_step": float(np.min(tau_diffs)),
            "tau_max_step": float(np.max(tau_diffs)),
            "node_min_step": float(np.min(node_diffs)) if node_diffs.size else 0.0,
            "node_max_step": float(np.max(node_diffs)) if node_diffs.size else 0.0,
            "step_ratio_max": step_ratio,
            "density_entropy": entropy(alpha),
            "status": status,
            "error": error,
        }
    except Exception as exc:
        return {
            "solver": case["solver"],
            "schedule": case["schedule"],
            "seed": seed,
            "status": "FAILED",
            "error": normalize_status_error(exc),
        }


def sample_stat_proxy(image_dir: Path) -> tuple[str, float, bool]:
    try:
        from PIL import Image
    except Exception:
        return "none", float("nan"), False
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        return "none", float("nan"), False
    values = []
    for path in paths[:64]:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            values.append(float(np.std(arr)))
    if not values:
        return "none", float("nan"), False
    return "sample_std_proxy", float(np.mean(values)), True


def run_generation(
    *,
    args: argparse.Namespace,
    case: dict[str, Any],
    seed: int,
    schedule_dir: Path | None,
    output_dir: Path,
    log_path: Path,
    commands: list[str],
) -> tuple[str, float, bool, Path]:
    case_nfe = nfe_for_solver(case["solver"], int(args.nfe))
    artifact = output_dir / "samples" / case["solver"] / case["schedule"] / f"seed_{seed:03d}"
    command = [
        sys.executable,
        "scripts/run/run_pndm_experiment.py",
        "--manifest",
        args.manifest,
        "--dataset-config",
        args.dataset_config,
        "--model-asset",
        args.model_asset,
        "--solver",
        case["solver"],
        "--nfe",
        str(case_nfe),
        "--num-samples",
        str(int(args.num_samples)),
        "--batch-size",
        str(min(int(args.batch_size), int(args.num_samples))),
        "--seed",
        str(int(seed)),
        "--output-dir",
        str(artifact),
        "--summary-csv",
        str(output_dir / "project_summary.csv"),
        "--schedule-name",
        case["schedule"],
    ]
    if schedule_dir is not None:
        command.extend(["--schedule-dir", str(schedule_dir)])
    if bool(args.compute_fid) or str(args.metric) == "fid":
        command.append("--compute-fid")
        command.extend(["--reference-fid-asset", str(args.reference_stats_asset)])
        if not parse_bool(args.save_samples):
            command.append("--discard-samples")
        if int(args.preview_samples) > 0:
            command.extend(["--preview-samples", str(int(args.preview_samples))])
    run_command(command, log_path=log_path, commands=commands)
    if bool(args.compute_fid) or str(args.metric) == "fid":
        manifest_path = artifact / "run_manifest.json"
        manifest = load_json(manifest_path)
        fid_value = manifest.get("fid")
        if fid_value is None:
            raise RuntimeError(f"FID was requested but missing from {manifest_path}")
        return "fid", float(fid_value), True, artifact
    if str(args.metric) == "none":
        return "none", float("nan"), False, artifact
    metric_name, metric_value, metric_available = sample_stat_proxy(artifact)
    return metric_name, metric_value, metric_available, artifact


def base_result_row(
    *,
    args: argparse.Namespace,
    case: dict[str, Any],
    seed: int,
    timestamp: str,
    git_commit: str,
    device: str,
) -> dict[str, Any]:
    case_nfe = nfe_for_solver(case["solver"], int(args.nfe))
    return {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "device": device,
        "dataset": args.dataset,
        "model_asset": args.model_asset,
        "solver": case["solver"],
        "schedule": case["schedule"],
        "nfe": case_nfe,
        "seed": int(seed),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "metric_name": "none",
        "metric_value": float("nan"),
        "metric_available": False,
        "fid_reference": args.reference_stats_asset if (bool(args.compute_fid) or str(args.metric) == "fid") else "",
        "runtime_sec": 0.0,
        "status": "OK",
        "error": "",
        "artifact_path": "",
    }


def aggregate_results(raw_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diag_rates: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in diag_rows:
        key = (str(row.get("solver", "")), str(row.get("schedule", "")))
        valid = (
            str(row.get("valid_alpha", "")).lower() == "true" or row.get("valid_alpha") is True
        ) and (
            str(row.get("valid_tau", "")).lower() == "true" or row.get("valid_tau") is True
        ) and (
            str(row.get("monotone_tau", "")).lower() == "true" or row.get("monotone_tau") is True
        ) and (
            str(row.get("monotone_nodes", "")).lower() == "true" or row.get("monotone_nodes") is True
        )
        diag_rates[key].append(valid)

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            str(row["dataset"]),
            str(row["model_asset"]),
            str(row["solver"]),
            str(row["schedule"]),
            str(row["nfe"]),
            str(row["num_samples"]),
        )
        grouped[key].append(row)

    aggregate = []
    for key, rows in sorted(grouped.items()):
        metric_values = [
            float(row["metric_value"])
            for row in rows
            if str(row.get("metric_available", "")).lower() == "true"
            and str(row.get("metric_value", "")).lower() not in {"", "nan", "none"}
        ]
        runtimes = [float(row.get("runtime_sec", 0.0) or 0.0) for row in rows]
        valid_values = diag_rates.get((key[2], key[3]), [])
        aggregate.append(
            {
                "dataset": key[0],
                "model_asset": key[1],
                "solver": key[2],
                "schedule": key[3],
                "nfe": key[4],
                "num_samples": key[5],
                "num_runs": len(rows),
                "num_success": sum(1 for row in rows if row.get("status") == "OK"),
                "success_rate": sum(1 for row in rows if row.get("status") == "OK") / max(len(rows), 1),
                "fid_mean": statistics.fmean(metric_values) if metric_values else float("nan"),
                "fid_std": statistics.pstdev(metric_values) if len(metric_values) > 1 else 0.0 if metric_values else float("nan"),
                "metric_mean": statistics.fmean(metric_values) if metric_values else float("nan"),
                "metric_std": statistics.pstdev(metric_values) if len(metric_values) > 1 else 0.0 if metric_values else float("nan"),
                "runtime_mean": statistics.fmean(runtimes) if runtimes else float("nan"),
                "runtime_std": statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0 if runtimes else float("nan"),
                "valid_schedule_rate": sum(valid_values) / max(len(valid_values), 1) if valid_values else 0.0,
            }
        )
    return aggregate


def format_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(
    *,
    path: Path,
    args: argparse.Namespace,
    device: str,
    env: dict[str, Any],
    git_commit: str,
    dirty: str,
    commands: list[str],
    solvers: list[str],
    aggregate: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> None:
    result_rows = [
        [
            row.get("solver"),
            row.get("schedule"),
            row.get("nfe"),
            format_float(row.get("metric_mean")),
            format_float(row.get("metric_std")),
            row.get("num_samples"),
            format_float(row.get("success_rate")),
            format_float(row.get("valid_schedule_rate")),
        ]
        for row in aggregate
    ]
    failures = [row for row in raw_rows if row.get("status") == "FAILED"]
    lines = [
        "# FP_CLOCK CIFAR-10 Smoke Report",
        "",
        "## Environment",
        "",
        f"- python version: {platform.python_version()}",
        f"- torch version: {env.get('torch_version', '')}",
        f"- torchvision version: {env.get('torchvision_version', '')}",
        f"- cuda available: {env.get('cuda_available', False)}",
        f"- device used: {device}",
        f"- git commit: {git_commit}",
        f"- git status: {dirty}",
        f"- metric: {'fid' if bool(args.compute_fid) or str(args.metric) == 'fid' else args.metric}",
        f"- fid reference: {args.reference_stats_asset if bool(args.compute_fid) or str(args.metric) == 'fid' else ''}",
        f"- solvers: {', '.join(solvers)}",
        "- commands executed:",
    ]
    lines.extend([f"  - `{command}`" for command in commands] or ["  - none"])
    lines.extend(
        [
            "",
            "## Results",
            "",
            markdown_table(
                ["solver", "schedule", "nfe", "metric_mean", "metric_std", "num_samples", "success_rate", "valid_schedule_rate"],
                result_rows,
                limit=120,
            ),
            "",
            "## Failures",
            "",
            f"- failed rows: {len(failures)}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = default_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log_path.write_text("", encoding="utf-8")
    args.save_samples = parse_bool(args.save_samples)
    if str(args.metric) == "fid":
        args.compute_fid = True

    device = resolve_device(args.device)
    if device == "cpu":
        args.num_samples = int(args.num_samples or 32)
        args.seeds = args.seeds or [0]
    else:
        args.num_samples = int(args.num_samples or 64)
        args.seeds = args.seeds or [0, 1]

    env = torch_environment()
    git_commit = command_output(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    dirty = "dirty" if command_output(["git", "status", "--porcelain"]) else "clean"
    timestamp = datetime.now().isoformat(timespec="seconds")
    solvers, solver_mapping = available_solvers(args.solvers)
    cases = schedule_cases(args, solvers)
    commands: list[str] = []
    raw_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    append_log(log_path, f"output_dir={output_dir}")
    append_log(log_path, f"device={device}")
    append_log(log_path, f"solvers={solver_mapping}")

    for case in cases:
        for seed in args.seeds:
            row = base_result_row(
                args=args,
                case=case,
                seed=int(seed),
                timestamp=timestamp,
                git_commit=git_commit,
                device=device,
            )
            start = time.perf_counter()
            schedule_dir: Path | None = None
            status = "OK"
            error = ""
            try:
                if not args.schedule_only:
                    schedule_dir = export_schedule(
                        args=args,
                        case=case,
                        seed=int(seed),
                        output_dir=output_dir,
                        log_path=log_path,
                        commands=commands,
                    )
                    metric_name, metric_value, metric_available, artifact = run_generation(
                        args=args,
                        case=case,
                        seed=int(seed),
                        schedule_dir=schedule_dir,
                        output_dir=output_dir,
                        log_path=log_path,
                        commands=commands,
                    )
                    row["metric_name"] = metric_name
                    row["metric_value"] = metric_value
                    row["metric_available"] = metric_available
                    row["artifact_path"] = str(artifact)
            except Exception as exc:
                status = "FAILED"
                error = normalize_status_error(exc)
                append_log(log_path, f"FAILED {case['solver']} {case['schedule']} seed={seed}: {error}")
            row["runtime_sec"] = time.perf_counter() - start
            row["status"] = status
            row["error"] = error
            raw_rows.append(row)
            diagnostic_rows.append(
                schedule_diagnostic(
                    case=case,
                    seed=int(seed),
                    bundle_dir=schedule_dir,
                    nfe=int(args.nfe),
                    status=status,
                    error=error,
                )
            )

    aggregate = aggregate_results(raw_rows, diagnostic_rows)
    write_csv(output_dir / "results_raw.csv", RESULT_FIELDS, raw_rows)
    write_csv(output_dir / "schedule_diagnostics.csv", DIAGNOSTIC_FIELDS, diagnostic_rows)
    write_csv(output_dir / "results_aggregate.csv", AGG_FIELDS, aggregate)
    write_report(
        path=output_dir / "fp_clock_smoke_report.md",
        args=args,
        device=device,
        env=env,
        git_commit=git_commit,
        dirty=dirty,
        commands=commands,
        solvers=solvers,
        aggregate=aggregate,
        raw_rows=raw_rows,
    )
    print(output_dir)


if __name__ == "__main__":
    main()

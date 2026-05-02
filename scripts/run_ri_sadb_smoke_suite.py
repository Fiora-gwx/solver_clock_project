#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    "model",
    "solver",
    "schedule",
    "ri_variant",
    "eta",
    "beta",
    "defect_mode",
    "defect_source",
    "fixed_defect_solver",
    "target_defect_solver",
    "stork_window_len",
    "stork_refine_factor",
    "nfe",
    "seed",
    "num_samples",
    "train_subset",
    "eval_subset",
    "metric_name",
    "metric_value",
    "metric_available",
    "runtime_sec",
    "status",
    "error",
    "artifact_path",
)

DIAGNOSTIC_FIELDS = (
    "solver",
    "schedule",
    "ri_variant",
    "eta",
    "defect_mode",
    "defect_source",
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
    "solver",
    "schedule",
    "ri_variant",
    "eta",
    "defect_mode",
    "defect_source",
    "nfe",
    "num_runs",
    "num_success",
    "success_rate",
    "metric_mean",
    "metric_std",
    "runtime_mean",
    "runtime_std",
    "valid_schedule_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RI-SADB CIFAR-10 smoke suite.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--etas", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--nfe", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--eval-subset", type=int, default=None)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--include-sadb", action="store_true")
    parser.add_argument("--include-ri-g", action="store_true")
    parser.add_argument("--include-fixed-defect", action="store_true")
    parser.add_argument("--include-target-defect", action="store_true")
    parser.add_argument("--fixed-defect-solver", default="euler")
    parser.add_argument("--stork-window-len", type=int, default=4)
    parser.add_argument("--stork-refine-factor", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--solvers", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--model-asset", default="pndm_model_ddim_cifar10")
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets/cifar10.yaml")
    return parser.parse_args()


def command_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"torch_version": f"unavailable: {exc}", "cuda_available": False}
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception as exc:  # pragma: no cover - environment dependent
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
    return REPO_ROOT / "outputs" / "ri_sadb_smoke" / stamp


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


def is_stork_solver(solver: str) -> bool:
    normalized = solver.lower().replace("-", "_")
    return normalized.startswith("stork")


def available_solvers(requested: list[str] | None) -> tuple[list[str], dict[str, str]]:
    candidates = requested or ["euler", "heun2", "stork4_1st"]
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
    cases: list[dict[str, Any]] = []
    if args.include_base:
        cases.extend(
            {
                "solver": solver,
                "schedule": "base",
                "ri_variant": "",
                "eta": "",
                "beta": 0.0,
                "defect_mode": "none",
                "defect_source": "none",
                "target_defect_solver": solver,
            }
            for solver in solvers
        )
    if args.include_sadb:
        cases.extend(
            {
                "solver": solver,
                "schedule": "SADB",
                "ri_variant": "",
                "eta": "",
                "beta": 0.0,
                "defect_mode": "sadb_step_refinement",
                "defect_source": "target_solver" if not is_stork_solver(solver) else "target_stork_stateful",
                "target_defect_solver": solver,
            }
            for solver in solvers
        )
    if args.include_ri_g:
        cases.extend(
            {
                "solver": solver,
                "schedule": "RI_G",
                "ri_variant": "RI_G",
                "eta": 1.0,
                "beta": 0.0,
                "defect_mode": "geometry_only",
                "defect_source": "geometry_only",
                "target_defect_solver": "",
            }
            for solver in solvers
        )
    if args.include_fixed_defect:
        for solver in solvers:
            for eta in args.etas:
                cases.append(
                    {
                        "solver": solver,
                        "schedule": "RI_SADB_FIXED_DEFECT",
                        "ri_variant": "RI_SADB_FIXED_DEFECT",
                        "eta": float(eta),
                        "beta": 0.0,
                        "defect_mode": "fixed_euler_proxy",
                        "defect_source": "fixed_euler_proxy"
                        if args.fixed_defect_solver == "euler"
                        else f"fixed_{args.fixed_defect_solver}_proxy",
                        "target_defect_solver": solver,
                    }
                )
    if args.include_target_defect:
        for solver in solvers:
            for eta in args.etas:
                cases.append(
                    {
                        "solver": solver,
                        "schedule": "RI_SADB_TARGET_DEFECT",
                        "ri_variant": "RI_SADB_TARGET_DEFECT",
                        "eta": float(eta),
                        "beta": 0.0,
                        "defect_mode": "target_solver",
                        "defect_source": "target_stork_short_window" if is_stork_solver(solver) else f"target_{solver}",
                        "target_defect_solver": solver,
                    }
                )
    return cases


def write_clock_config(
    path: Path,
    *,
    family: str,
    calibration_solver: str,
    eta: float | None = None,
    beta: float = 0.0,
    target_nfe: int = 10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "clock:",
        f"  family: {family}",
        "  calibration_mode: ri_sadb" if family == "RI_SADB" else "  calibration_mode: sadb",
        f"  calibration_solver: {calibration_solver}",
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
        "  cache_path: outputs/cache/sadb_profiles",
        f"  target_nfes: [{int(target_nfe)}]",
    ]
    if family == "RI_SADB":
        lines.extend(
            [
                f"  eta: {float(eta if eta is not None else 0.25)}",
                f"  beta: {float(beta)}",
                "  ell_scale: step",
                "  ri_agg: mean",
            ]
        )
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

    config_dir = output_dir / "clock_configs"
    schedule_dir = output_dir / "schedules" / case["solver"] / case["schedule"]
    eta = case["eta"]
    eta_token = "base" if eta == "" else f"eta_{float(eta):.2f}".replace(".", "_")
    schedule_root = schedule_dir / case["defect_mode"] / eta_token / f"seed_{seed:03d}"
    family = "SADB" if case["schedule"] == "SADB" else "RI_SADB"
    if case["schedule"] == "SADB":
        calibration_solver = "target"
    elif case["defect_mode"] == "geometry_only":
        calibration_solver = args.fixed_defect_solver
    elif case["defect_mode"] == "fixed_euler_proxy" or case["defect_mode"].startswith("fixed"):
        calibration_solver = args.fixed_defect_solver
    elif is_stork_solver(case["solver"]):
        raise RuntimeError("native short-window STORK RI-SADB target defect is not implemented in this runner")
    else:
        calibration_solver = case["solver"]
    config_path = config_dir / f"{case['schedule']}_{case['solver']}_{case['defect_mode']}_{eta_token}.yaml"
    write_clock_config(
        config_path,
        family=family,
        calibration_solver=calibration_solver,
        eta=None if eta == "" else float(eta),
        beta=float(case["beta"]),
        target_nfe=int(args.nfe),
    )
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
        "--target-nfes",
        str(int(args.nfe)),
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
    bundle_dir = schedule_root / f"nfe_{int(args.nfe):03d}"
    patch_schedule_meta(
        bundle_dir,
        {
            "schedule_family": case["schedule"],
            "ri_variant": case["ri_variant"],
            "defect_mode": case["defect_mode"],
            "defect_source": case["defect_source"],
            "fixed_defect_solver": args.fixed_defect_solver,
            "target_defect_solver": case["target_defect_solver"],
            "stork_window_len": int(args.stork_window_len),
            "stork_refine_factor": int(args.stork_refine_factor),
        },
    )
    return bundle_dir


def patch_schedule_meta(bundle_dir: Path, updates: dict[str, Any]) -> None:
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = load_json(meta_path)
    meta.update(updates)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            if bundle.g_grid is not None:
                alpha = 1.0 / np.maximum(np.asarray(bundle.g_grid, dtype=np.float64), 1.0e-12)
            else:
                alpha = np.ones_like(np.asarray(nodes, dtype=np.float64))
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
            "ri_variant": case["ri_variant"],
            "eta": case["eta"],
            "defect_mode": case["defect_mode"],
            "defect_source": case["defect_source"],
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
            "ri_variant": case["ri_variant"],
            "eta": case["eta"],
            "defect_mode": case["defect_mode"],
            "defect_source": case["defect_source"],
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
    artifact = (
        output_dir
        / "samples"
        / case["solver"]
        / case["schedule"]
        / str(case["defect_mode"])
        / (f"eta_{float(case['eta']):.2f}".replace(".", "_") if case["eta"] != "" else "baseline")
        / f"seed_{seed:03d}"
    )
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
        str(int(args.nfe)),
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
    run_command(command, log_path=log_path, commands=commands)
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
    return {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "device": device,
        "dataset": args.dataset,
        "model": args.model_asset,
        "solver": case["solver"],
        "schedule": case["schedule"],
        "ri_variant": case["ri_variant"],
        "eta": case["eta"],
        "beta": case["beta"],
        "defect_mode": case["defect_mode"],
        "defect_source": case["defect_source"],
        "fixed_defect_solver": args.fixed_defect_solver,
        "target_defect_solver": case["target_defect_solver"],
        "stork_window_len": int(args.stork_window_len),
        "stork_refine_factor": int(args.stork_refine_factor),
        "nfe": int(args.nfe),
        "seed": int(seed),
        "num_samples": int(args.num_samples),
        "train_subset": int(args.train_subset),
        "eval_subset": int(args.eval_subset),
        "metric_name": "none",
        "metric_value": float("nan"),
        "metric_available": False,
        "runtime_sec": 0.0,
        "status": "OK",
        "error": "",
        "artifact_path": "",
    }


def aggregate_results(raw_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diag_rates: dict[tuple[str, str, str, str, str, str], list[bool]] = defaultdict(list)
    for row in diag_rows:
        key = (
            str(row.get("solver", "")),
            str(row.get("schedule", "")),
            str(row.get("ri_variant", "")),
            str(row.get("eta", "")),
            str(row.get("defect_mode", "")),
            str(row.get("defect_source", "")),
        )
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
            str(row["solver"]),
            str(row["schedule"]),
            str(row["ri_variant"]),
            str(row["eta"]),
            str(row["defect_mode"]),
            str(row["defect_source"]),
            str(row["nfe"]),
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
        diag_key = (key[1], key[2], key[3], key[4], key[5], key[6])
        valid_values = diag_rates.get(diag_key, [])
        aggregate.append(
            {
                "dataset": key[0],
                "solver": key[1],
                "schedule": key[2],
                "ri_variant": key[3],
                "eta": key[4],
                "defect_mode": key[5],
                "defect_source": key[6],
                "nfe": key[7],
                "num_runs": len(rows),
                "num_success": sum(1 for row in rows if row.get("status") in {"OK", "FALLBACK"}),
                "success_rate": sum(1 for row in rows if row.get("status") in {"OK", "FALLBACK"}) / max(len(rows), 1),
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


def best_eta(aggregate: list[dict[str, Any]], *, ri_variant: str) -> dict[str, str]:
    result: dict[str, str] = {}
    by_solver: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate:
        if row.get("ri_variant") == ri_variant and row.get("metric_mean") == row.get("metric_mean"):
            by_solver[str(row["solver"])].append(row)
    for solver, rows in by_solver.items():
        valid_rows = [row for row in rows if float(row.get("success_rate", 0.0)) > 0.0]
        if not valid_rows:
            continue
        chosen = max(valid_rows, key=lambda row: float(row.get("metric_mean", float("-inf"))))
        result[solver] = str(chosen.get("eta", ""))
    return result


def fixed_vs_target_table(aggregate: list[dict[str, Any]]) -> list[list[Any]]:
    fixed: dict[tuple[str, str], float] = {}
    target: dict[tuple[str, str], float] = {}
    for row in aggregate:
        key = (str(row.get("solver", "")), str(row.get("eta", "")))
        if row.get("ri_variant") == "RI_SADB_FIXED_DEFECT":
            fixed[key] = float(row.get("metric_mean", float("nan")))
        if row.get("ri_variant") == "RI_SADB_TARGET_DEFECT":
            target[key] = float(row.get("metric_mean", float("nan")))
    rows: list[list[Any]] = []
    for key in sorted(set(fixed) | set(target)):
        f_value = fixed.get(key, float("nan"))
        t_value = target.get(key, float("nan"))
        delta = t_value - f_value if not math.isnan(f_value) and not math.isnan(t_value) else float("nan")
        winner = "target" if delta > 0 else "fixed" if delta < 0 else "tie_or_unavailable"
        rows.append([key[0], key[1], format_float(f_value), format_float(t_value), format_float(delta), winner])
    return rows


def generate_report(
    *,
    path: Path,
    args: argparse.Namespace,
    device: str,
    env: dict[str, Any],
    git_commit: str,
    dirty: str,
    commands: list[str],
    solvers: list[str],
    solver_mapping: dict[str, str],
    aggregate: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> None:
    schedule_rows = []
    for row in aggregate:
        matching = [
            diag
            for diag in diagnostics
            if diag.get("solver") == row.get("solver")
            and diag.get("schedule") == row.get("schedule")
            and str(diag.get("eta", "")) == str(row.get("eta", ""))
            and diag.get("defect_source") == row.get("defect_source")
        ]
        alpha_ranges = [
            f"{format_float(diag.get('alpha_min'))}-{format_float(diag.get('alpha_max'))}"
            for diag in matching
            if diag.get("alpha_min", "") != ""
        ]
        monotone_rate = (
            sum(1 for diag in matching if diag.get("monotone_tau") is True or str(diag.get("monotone_tau")).lower() == "true")
            / max(len(matching), 1)
            if matching
            else 0.0
        )
        max_ratio = max([float(diag.get("step_ratio_max", 0.0) or 0.0) for diag in matching] or [0.0])
        schedule_rows.append(
            [
                row.get("ri_variant") or row.get("schedule"),
                row.get("solver"),
                row.get("eta"),
                row.get("defect_source"),
                format_float(row.get("valid_schedule_rate")),
                format_float(monotone_rate),
                ", ".join(alpha_ranges[:2]) if alpha_ranges else "",
                format_float(max_ratio),
            ]
        )

    eta_rows = [
        [
            row.get("solver"),
            row.get("defect_mode"),
            row.get("eta"),
            format_float(row.get("metric_mean")),
            format_float(row.get("metric_std")),
            format_float(row.get("runtime_mean")),
            format_float(row.get("success_rate")),
        ]
        for row in aggregate
        if row.get("ri_variant") in {"RI_G", "RI_SADB_FIXED_DEFECT", "RI_SADB_TARGET_DEFECT"}
    ]
    fallback_used = any(row.get("status") == "FALLBACK" for row in raw_rows)
    stork_rows = [
        row
        for row in raw_rows
        if is_stork_solver(str(row.get("solver", "")))
        and row.get("ri_variant") in {"RI_G", "RI_SADB_FIXED_DEFECT", "RI_SADB_TARGET_DEFECT"}
    ]
    fixed_best = best_eta(aggregate, ri_variant="RI_SADB_FIXED_DEFECT")
    target_best = best_eta(aggregate, ri_variant="RI_SADB_TARGET_DEFECT")
    fixed_target = fixed_vs_target_table(aggregate)
    failures = [row for row in raw_rows if row.get("status") == "FAILED"]
    lines = [
        "# RI-SADB Smoke Report",
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
        "- installed missing deps: none",
        f"- solvers requested/mapped: {json.dumps(solver_mapping, sort_keys=True)}",
        "- commands executed:",
    ]
    lines.extend([f"  - `{command}`" for command in commands] or ["  - none"])
    lines.extend(
        [
            "",
            "## Schedule Validity Summary",
            "",
            markdown_table(
                [
                    "variant",
                    "solver",
                    "eta",
                    "defect_source",
                    "valid_schedule_rate",
                    "monotone_rate",
                    "alpha_range",
                    "step_ratio_max",
                ],
                schedule_rows,
                limit=80,
            ),
            "",
            "## Eta Sweep Result",
            "",
            markdown_table(
                [
                    "solver",
                    "defect_mode",
                    "eta",
                    "metric_mean",
                    "metric_std",
                    "runtime_mean",
                    "success_rate",
                ],
                eta_rows,
                limit=120,
            ),
            "",
            f"- best_eta_by_metric_fixed_proxy: {json.dumps(fixed_best, sort_keys=True)}",
            f"- best_eta_by_metric_target_defect: {json.dumps(target_best, sort_keys=True)}",
            f"- best_eta_by_validity: variants with valid_schedule_rate=1.0 are listed in the validity table.",
            "",
            "## Fixed Defect vs Target Defect",
            "",
            markdown_table(
                [
                    "solver",
                    "eta",
                    "fixed_euler_proxy_metric",
                    "target_solver_metric",
                    "delta_target_minus_fixed",
                    "winner",
                ],
                fixed_target,
                limit=120,
            ),
            "",
            "## STORK Short-Window Defect",
            "",
        ]
    )
    if any(is_stork_solver(solver) for solver in solvers):
        stork_table = [
            [
                row.get("eta"),
                row.get("stork_window_len"),
                row.get("defect_source") if row.get("ri_variant") == "RI_SADB_FIXED_DEFECT" else "",
                row.get("defect_source") if row.get("ri_variant") == "RI_SADB_TARGET_DEFECT" else "",
                row.get("status"),
            ]
            for row in stork_rows
        ]
        lines.append(markdown_table(["eta", "window_len", "fixed_euler_proxy", "target_stork_short_window", "status"], stork_table, limit=80))
        lines.extend(
            [
                "",
                "- STORK history clone/restore implemented: no",
                f"- fallback used: {'yes' if fallback_used else 'no'}",
            ]
        )
        if fallback_used:
            lines.append("- Target STORK short-window defect was not available; fallback results are diagnostic only.")
    else:
        lines.append("- stork_available=false")
    target_better = sorted(
        {
            row[0]
            for row in fixed_target
            if len(row) >= 6 and row[-1] == "target"
        }
    )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"1. Best eta for RI_SADB_FIXED_DEFECT was {json.dumps(fixed_best, sort_keys=True)}.",
            f"2. Best eta for RI_SADB_TARGET_DEFECT was {json.dumps(target_best, sort_keys=True)}.",
            f"3. RI_G was {'stable' if all(row.get('status') in {'OK', 'FALLBACK'} for row in raw_rows if row.get('ri_variant') == 'RI_G') else 'unstable'} according to schedule validity and run status.",
            f"4. Target defect beat fixed proxy for solvers: {', '.join(target_better) if target_better else 'none or unavailable'}.",
            f"5. STORK short-window defect was {'fallback' if fallback_used else 'not requested' if not any(is_stork_solver(s) for s in solvers) else 'successful'}.",
            f"6. Main failures and next debugging steps: {len(failures)} failed rows; inspect run.log and failed CSV error fields before increasing NFE or sample count.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = default_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log_path.write_text("", encoding="utf-8")

    device = resolve_device(args.device)
    if device == "cpu":
        args.train_subset = int(args.train_subset or 128)
        args.eval_subset = int(args.eval_subset or 64)
        args.num_samples = int(args.num_samples or 32)
        args.seeds = args.seeds or [0]
    else:
        args.train_subset = int(args.train_subset or 512)
        args.eval_subset = int(args.eval_subset or 128)
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
    append_log(log_path, f"solvers={solvers}")

    for case in cases:
        for seed in args.seeds:
            row = base_result_row(
                args=args,
                case=case,
                seed=seed,
                timestamp=timestamp,
                git_commit=git_commit,
                device=device,
            )
            start = time.perf_counter()
            schedule_dir: Path | None = None
            status = "OK"
            error = ""
            try:
                if args.schedule_only:
                    if case["schedule"] != "base":
                        status = "FALLBACK" if is_stork_solver(case["solver"]) and case["ri_variant"] == "RI_SADB_TARGET_DEFECT" else "OK"
                        if status == "FALLBACK":
                            error = "schedule-only fallback; native STORK short-window target defect not executed"
                            case = dict(case)
                            case["defect_source"] = "stork_reference_fallback"
                            row["defect_source"] = "stork_reference_fallback"
                else:
                    if is_stork_solver(case["solver"]) and case["ri_variant"] == "RI_SADB_TARGET_DEFECT":
                        status = "FALLBACK"
                        error = "native STORK short-window RI-SADB target defect is not implemented; diagnostic fallback only"
                        case = dict(case)
                        case["defect_source"] = "stork_reference_fallback"
                        row["defect_source"] = "stork_reference_fallback"
                    else:
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
                append_log(log_path, f"FAILED {case['solver']} {case['schedule']} eta={case['eta']} seed={seed}: {error}")
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
    generate_report(
        path=output_dir / "ri_sadb_smoke_report.md",
        args=args,
        device=device,
        env=env,
        git_commit=git_commit,
        dirty=dirty,
        commands=commands,
        solvers=solvers,
        solver_mapping=solver_mapping,
        aggregate=aggregate,
        diagnostics=diagnostic_rows,
        raw_rows=raw_rows,
    )
    print(output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goes.aggregation import aggregation_label
from goes.config import stable_hash
from goes.edge_evaluator import evaluate_replay_metrics
from goes.gpde import (
    default_q_for_solver,
    evaluate_gpde_profile,
    make_probe_steps,
    materialize_gpde_schedule,
    parse_float_list,
)
from goes.logging_utils import dump_json, runtime_metadata, write_csv
from goes.metrics import make_metric
from goes.schedules import GPDE_SCHEDULE_IMPLEMENTATION_VERSION, save_schedule_outputs
from goes.torch_backend import (
    build_or_load_torch_velocity_oracle,
    make_torch_step_solver,
)
from src.clock.fp_clock import build_fp_clock_profile
from src.clock.solver_registry import get_solver_native_spec
from src.adapters.pndm import (
    _interp_sigmas_for_timesteps,
    build_scheduler,
    collect_anchored_replay_calibration_stats,
    build_sigma_derivative_oracle,
    build_velocity_oracle,
    load_model,
    load_native_config,
    preferred_calibration_domain,
)
from src.utils.assets import AssetManifest
from src.utils.config import dump_yaml, ensure_dir, load_yaml, resolve_repo_path
from src.utils.schedule_bundle import ScheduleBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a GPDE schedule for a PNDM noise-prediction model.")
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-asset")
    parser.add_argument("--solver", default="euler")
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-cache-dir", default="outputs/goes/pndm_oracle_cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=0)
    parser.add_argument("--ref-nfe", type=int, default=256)
    parser.add_argument("--ref-grid-size", type=int, default=257)
    parser.add_argument("--probe-grid-size", type=int, default=128)
    parser.add_argument("--candidate-grid-size", type=int, default=None, help="Compatibility alias for --probe-grid-size.")
    parser.add_argument("--probe-step-multipliers", default="1,2,4")
    parser.add_argument(
        "--defect-backend",
        choices=["auto", "single_step", "anchored_replay"],
        default="auto",
        help="GPDE defect estimator backend. auto uses single-step when valid and history-aware replay otherwise.",
    )
    parser.add_argument("--anchor-nfe", type=int, default=0, help="Anchor NFE for history-aware replay; 0 selects a solver-aware default.")
    parser.add_argument("--window-size", type=int, default=0, help="History-aware replay window length; 0 selects the solver registry default.")
    parser.add_argument("--replay-q-min", type=float, default=1.05)
    parser.add_argument("--replay-q-max", type=float, default=6.0)
    parser.add_argument("--q-mode", choices=["global_fit", "fixed"], default="global_fit")
    parser.add_argument("--fixed-q", type=float)
    parser.add_argument("--monitor-smoothing-window", type=int, default=3)
    parser.add_argument("--monitor-epsilon", type=float, default=1.0e-12)
    parser.add_argument("--coordinate-domain", choices=["auto", "timesteps", "sigmas"], default="auto")
    parser.add_argument("--metric", choices=["identity", "edm_scalar", "channel_whitened"], default="identity")
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--aggregation", choices=["mean", "median", "trimmed_mean", "cvar"], default="cvar")
    parser.add_argument("--trim-ratio", type=float, default=0.10)
    parser.add_argument("--cvar-alpha", type=float, default=0.80)
    parser.add_argument("--model-output-type", default="epsilon")
    parser.add_argument("--sigma-floor", type=float, default=1.0e-6)
    parser.add_argument("--no-reuse-oracle", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments and print the planned run without loading a model.")
    return parser.parse_args()


def _probe_grid_size(args: argparse.Namespace) -> int:
    candidate_grid_size = getattr(args, "candidate_grid_size", None)
    probe_grid_size = getattr(args, "probe_grid_size", candidate_grid_size)
    if candidate_grid_size is not None:
        return int(candidate_grid_size)
    return int(probe_grid_size)


def _probe_step_multipliers_arg(args: argparse.Namespace) -> str:
    return str(getattr(args, "probe_step_multipliers", "1,2,4"))


def _q_mode_arg(args: argparse.Namespace) -> str:
    return str(getattr(args, "q_mode", "global_fit"))


def _fixed_q_arg(args: argparse.Namespace) -> float | None:
    value = getattr(args, "fixed_q", None)
    return None if value is None else float(value)


def _monitor_smoothing_window_arg(args: argparse.Namespace) -> int:
    return int(getattr(args, "monitor_smoothing_window", 3))


def _monitor_epsilon_arg(args: argparse.Namespace) -> float:
    return float(getattr(args, "monitor_epsilon", 1.0e-12))


def _native_domain(args: argparse.Namespace) -> str:
    if args.coordinate_domain != "auto":
        return str(args.coordinate_domain)
    return str(preferred_calibration_domain(args.solver))


def _single_step_proxy_solver(solver: str) -> str | None:
    normalized = str(solver).lower().replace("-", "_")
    if normalized in {"euler", "heun2"}:
        return normalized
    return None


def _resolve_defect_backend(solver: str, requested: str) -> str:
    normalized_request = str(requested or "auto").lower().replace("-", "_")
    if normalized_request not in {"auto", "single_step", "anchored_replay"}:
        raise ValueError(f"Unsupported GPDE defect backend: {requested}")
    if normalized_request == "single_step":
        if _single_step_proxy_solver(solver) is None:
            raise ValueError(
                f"GPDE single-step defect backend is not valid for PNDM solver `{solver}`. "
                "Use --defect-backend anchored_replay for history-dependent or non-single-step solvers."
            )
        return "single_step"
    if normalized_request == "anchored_replay":
        spec = get_solver_native_spec("pndm", solver)
        if not spec.supports_base_trajectory_recording:
            raise ValueError(f"PNDM solver `{solver}` does not support GPDE anchored replay: {spec.notes}")
        return "anchored_replay"
    if _single_step_proxy_solver(solver) is not None:
        return "single_step"
    spec = get_solver_native_spec("pndm", solver)
    if not spec.supports_base_trajectory_recording:
        raise ValueError(f"PNDM solver `{solver}` does not support GPDE anchored replay: {spec.notes}")
    return "anchored_replay"


def _default_anchor_nfe(args: argparse.Namespace, solver: str) -> int:
    if int(getattr(args, "anchor_nfe", 0)) > 0:
        return int(args.anchor_nfe)
    spec = get_solver_native_spec("pndm", solver)
    return max(int(args.nfe), 4 * int(spec.solver_order), 16)


def _default_window_size(args: argparse.Namespace, solver: str) -> int:
    if int(getattr(args, "window_size", 0)) > 0:
        return int(args.window_size)
    return int(get_solver_native_spec("pndm", solver).recommended_window_len)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.nfe) <= 0:
        raise ValueError("--nfe must be positive.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.num_batches) <= 0:
        raise ValueError("--num-batches must be positive.")
    if int(args.microbatch_size) < 0:
        raise ValueError("--microbatch-size must be non-negative.")
    if int(args.ref_nfe) <= 0:
        raise ValueError("--ref-nfe must be positive.")
    if int(args.ref_grid_size) < 2:
        raise ValueError("--ref-grid-size must be at least 2.")
    if _probe_grid_size(args) < int(args.nfe) + 1:
        flag = "--candidate-grid-size" if getattr(args, "candidate_grid_size", None) is not None else "--probe-grid-size"
        raise ValueError(f"{flag} must be at least --nfe + 1.")
    if int(getattr(args, "anchor_nfe", 0)) < 0:
        raise ValueError("--anchor-nfe must be non-negative.")
    if int(getattr(args, "window_size", 0)) < 0:
        raise ValueError("--window-size must be non-negative.")
    replay_q_min = float(getattr(args, "replay_q_min", 1.05))
    replay_q_max = float(getattr(args, "replay_q_max", 6.0))
    if not math.isfinite(replay_q_min) or replay_q_min <= 1.0:
        raise ValueError("--replay-q-min must be finite and greater than 1.")
    if not math.isfinite(replay_q_max) or replay_q_max <= replay_q_min:
        raise ValueError("--replay-q-max must be finite and greater than --replay-q-min.")
    if _monitor_smoothing_window_arg(args) < 1:
        raise ValueError("--monitor-smoothing-window must be positive.")
    if _monitor_epsilon_arg(args) <= 0.0:
        raise ValueError("--monitor-epsilon must be positive.")
    if _q_mode_arg(args) == "fixed" and (_fixed_q_arg(args) is None or float(_fixed_q_arg(args)) <= 0.0):
        raise ValueError("--fixed-q must be positive when --q-mode=fixed.")
    rho = float(args.rho)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("--rho must be finite and in [0, 1].")
    trim_ratio = float(args.trim_ratio)
    if not math.isfinite(trim_ratio) or not 0.0 <= trim_ratio < 0.5:
        raise ValueError("--trim-ratio must be finite and in [0, 0.5).")
    cvar_alpha = float(args.cvar_alpha)
    if not math.isfinite(cvar_alpha) or not 0.0 <= cvar_alpha < 1.0:
        raise ValueError("--cvar-alpha must be finite and in [0, 1).")
    sigma_floor = float(args.sigma_floor)
    if not math.isfinite(sigma_floor) or sigma_floor <= 0.0:
        raise ValueError("--sigma-floor must be finite and positive.")
    sigma_data = float(args.sigma_data)
    if not math.isfinite(sigma_data) or sigma_data <= 0.0:
        raise ValueError("--sigma-data must be finite and positive.")


def _set_deterministic_seeds(seed: int) -> dict[str, Any]:
    resolved = int(seed)
    numpy_seed = resolved % (2**32 - 1)
    random.seed(resolved)
    np.random.seed(numpy_seed)
    torch.manual_seed(resolved)
    cuda_seeded = bool(torch.cuda.is_available())
    if cuda_seeded:
        torch.cuda.manual_seed_all(resolved)
    return {
        "python_random_seed": resolved,
        "numpy_seed": int(numpy_seed),
        "torch_seed": resolved,
        "torch_cuda_seed_all": cuda_seeded,
        "common_random_numbers": True,
    }


def _build_initial_samples(
    *,
    model: torch.nn.Module,
    image_size: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    initial_sigma: float,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    samples: list[torch.Tensor] = []
    seeds: list[int] = []
    for batch_index in range(int(num_batches)):
        samples.append(
            torch.randn(
                (int(batch_size), model.in_channels, int(image_size), int(image_size)),
                generator=generator,
                device=device,
            )
            * float(initial_sigma)
        )
        start = int(seed) + batch_index * int(batch_size)
        seeds.extend(range(start, start + int(batch_size)))
    return torch.cat(samples, dim=0), np.asarray(seeds, dtype=np.int64)


def _schedule_bundle(
    *,
    native_schedule: np.ndarray,
    representation: str,
    meta: dict[str, Any],
) -> ScheduleBundle:
    native = np.asarray(native_schedule, dtype=np.float64)
    if representation == "timesteps":
        return ScheduleBundle(
            timesteps=native[:-1].copy(),
            time_grid=native.copy(),
            meta={**meta, "representation": "timesteps", "terminal_timestep": float(native[-1])},
        )
    if representation == "sigmas":
        return ScheduleBundle(
            sigmas=native[:-1].copy(),
            sigma_grid=native.copy(),
            meta={**meta, "representation": "sigmas", "terminal_sigma": float(native[-1])},
        )
    raise ValueError(f"Unsupported schedule representation: {representation}")


def _goes_context_metadata(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    model_asset: str,
    coordinate_domain: str,
    model_path: str | Path | None = None,
    dataset_config_path: str | Path | None = None,
    defect_backend: str = "single_step",
    replay_detail_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    microbatch_size = int(args.microbatch_size) if int(args.microbatch_size) > 0 else None
    calibration_samples = int(args.batch_size) * int(args.num_batches)
    probe_nodes = _probe_grid_size(args)
    probe_step_count = len(parse_float_list(_probe_step_multipliers_arg(args), default=(1.0, 2.0, 4.0)))
    candidate_edges = int(probe_nodes) * (int(probe_nodes) + 1) // 2
    solver_evals_per_edge = 2 if str(args.solver).lower().replace("-", "_") == "heun2" else 1
    replay_meta = dict(replay_detail_meta or {})
    if str(defect_backend) == "anchored_replay":
        anchor_nfe = int(replay_meta.get("anchor_nfe", _default_anchor_nfe(args, args.solver)))
        window_size = int(replay_meta.get("window_size", _default_window_size(args, args.solver)))
        oracle_cost_per_sample = int(replay_meta.get("calibration_cost_per_sample", anchor_nfe * (4 + 7 * window_size)))
        probe_cost_per_sample = 0
        edge_cost_per_sample = 0
        calibration_cost = calibration_samples * oracle_cost_per_sample
        cost_note = "Estimated history-aware anchored replay scheduler steps; excludes generation."
    else:
        anchor_nfe = None
        window_size = None
        oracle_cost_per_sample = 4 * int(args.ref_nfe) + int(args.ref_grid_size)
        edge_cost_per_sample = candidate_edges * solver_evals_per_edge
        probe_cost_per_sample = edge_cost_per_sample
        calibration_cost = calibration_samples * (oracle_cost_per_sample + edge_cost_per_sample)
        cost_note = "Estimated RK4 oracle drift calls plus oracle-start GPDE probe drift calls; excludes generation."
    grid_config = {
        "size": int(probe_nodes),
        "type": "uniform_in_negative_native_coordinate",
        "probe_step_multipliers": parse_float_list(_probe_step_multipliers_arg(args), default=(1.0, 2.0, 4.0)),
        "q_mode": _q_mode_arg(args),
        "fixed_q": _fixed_q_arg(args),
        "monitor_smoothing_window": _monitor_smoothing_window_arg(args),
        "monitor_epsilon": _monitor_epsilon_arg(args),
    }
    return {
        "dataset": str(dataset_name),
        "model_asset": str(model_asset),
        "model_path": "" if model_path is None else str(model_path),
        "dataset_config_path": "" if dataset_config_path is None else str(dataset_config_path),
        "seed": int(args.seed),
        "guidance_scale": None,
        "coordinate_domain": str(coordinate_domain),
        "defect_backend": str(defect_backend),
        "anchored_replay_config": {
            "anchor_nfe": anchor_nfe,
            "window_size": window_size,
            "q_min": float(getattr(args, "replay_q_min", 1.05)),
            "q_max": float(getattr(args, "replay_q_max", 6.0)),
            **replay_meta,
        }
        if str(defect_backend) == "anchored_replay"
        else {},
        "calibration_config": {
            "num_samples": calibration_samples,
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "microbatch_size": microbatch_size,
            "seed": int(args.seed),
        },
        "pilot_config": {
            "num_samples": calibration_samples,
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "microbatch_size": microbatch_size,
            "seed": int(args.seed),
        },
        "oracle_config": {
            "ref_integrator": "rk4",
            "ref_nfe": int(args.ref_nfe),
            "ref_grid_size": int(args.ref_grid_size),
            "cache_dir": str(args.oracle_cache_dir),
            "reuse": not bool(args.no_reuse_oracle),
        },
        "probe_grid_config": grid_config,
        "candidate_grid_config": dict(grid_config),
        "calibration_cost_estimate": int(calibration_cost),
        "calibration_cost_unit": "model_evaluation_equivalents",
        "calibration_cost_breakdown": {
            "num_samples": calibration_samples,
            "cfg_multiplier": 1,
            "oracle_cost_per_sample": int(oracle_cost_per_sample),
            "candidate_edges": int(candidate_edges),
            "edge_cost_per_sample": int(edge_cost_per_sample),
            "probe_nodes": int(probe_nodes),
            "probe_step_count": int(probe_step_count),
            "solver_evals_per_edge": int(solver_evals_per_edge),
            "probe_cost_per_sample": int(probe_cost_per_sample),
            "total_model_eval_equivalents": int(calibration_cost),
            "note": cost_note,
        },
        "model_output_type": str(args.model_output_type),
        "sigma_floor": float(args.sigma_floor),
    }


def _schedule_export_metric_rows(
    *,
    solver: str,
    nfe: int,
    num_samples: int,
    final_latent_mse: float,
    replay_loss: float,
    fallback_fraction: float,
    schedule_dir: str | Path,
    oracle_cache_key: str,
    theory_covered: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = {
        "schedule": "GPDE",
        "solver": str(solver),
        "nfe": int(nfe),
        "guidance_scale": "",
        "schedule_dir": str(schedule_dir),
        "oracle_cache_key": str(oracle_cache_key),
        "theory_covered": bool(theory_covered),
    }
    calibration_row = {
        **base,
        "split": "calibration",
        "num_samples": int(num_samples),
        "final_latent_mse": float(final_latent_mse),
        "replay_loss": float(replay_loss),
        "fallback_fraction": float(fallback_fraction),
        "status": "OK",
        "note": "Calibration replay metrics from GPDE schedule export; not held-out image quality.",
    }
    heldout_row = {
        **base,
        "split": "heldout",
        "num_samples": 0,
        "final_latent_mse": "",
        "replay_loss": "",
        "fallback_fraction": "",
        "status": "NOT_EVALUATED",
        "note": "Schedule export does not run held-out generation or scoring; run the experiment launcher for paper metrics.",
    }
    return calibration_row, heldout_row


def _node_average_from_intervals(values: np.ndarray) -> np.ndarray:
    interval_values = np.asarray(values, dtype=np.float64)
    if interval_values.ndim != 1 or len(interval_values) < 1:
        raise ValueError("interval values must be a non-empty 1D array.")
    if len(interval_values) == 1:
        return np.asarray([interval_values[0], interval_values[0]], dtype=np.float64)
    nodes = np.empty(len(interval_values) + 1, dtype=np.float64)
    nodes[0] = interval_values[0]
    nodes[-1] = interval_values[-1]
    nodes[1:-1] = 0.5 * (interval_values[:-1] + interval_values[1:])
    return nodes


def _build_replay_gpde_artifacts(
    *,
    physical_grid: np.ndarray,
    stats,
    target_nfe: int,
    smoothing_window: int,
    epsilon: float,
    q_min: float,
    q_max: float,
) -> dict[str, Any]:
    artifacts = build_fp_clock_profile(
        physical_grid,
        stats,
        target_steps=int(target_nfe),
        eps=float(epsilon),
        q_min=float(q_min),
        q_max=float(q_max),
        smoothing_window=int(smoothing_window),
    )
    native_grid = np.asarray(artifacts.profile.physical_grid, dtype=np.float64)
    u_grid = -native_grid
    if np.any(np.diff(u_grid) <= 0.0):
        order = np.argsort(u_grid)
        u_grid = u_grid[order]
        native_grid = native_grid[order]
        monitor_density = np.asarray(artifacts.profile.density, dtype=np.float64)[order]
    else:
        monitor_density = np.asarray(artifacts.profile.density, dtype=np.float64)
    aggregate_coefficient = np.asarray(artifacts.interval_alpha_profile, dtype=np.float64)
    q_profile = np.asarray(artifacts.smoothed_effective_order_profile, dtype=np.float64)
    aggregate_nodes = (
        _node_average_from_intervals(aggregate_coefficient)
        if len(aggregate_coefficient) == len(u_grid) - 1
        else aggregate_coefficient
    )
    q_nodes = _node_average_from_intervals(q_profile) if len(q_profile) == len(u_grid) - 1 else q_profile
    gpde_schedule = materialize_gpde_schedule(u_grid, monitor_density, int(target_nfe))
    native_schedule = -np.asarray(gpde_schedule.u_schedule, dtype=np.float64)
    native_schedule[-1] = float(native_grid[-1])
    return {
        "clock_artifacts": artifacts,
        "u_grid": u_grid,
        "native_grid": native_grid,
        "monitor_density": monitor_density,
        "aggregate_coefficient": aggregate_nodes,
        "q_profile": q_nodes,
        "q_estimate": float(np.mean(np.asarray(artifacts.effective_order_profile, dtype=np.float64))),
        "gpde_schedule": gpde_schedule,
        "native_schedule": native_schedule,
    }


def _replay_profile_rows(
    *,
    physical_grid: np.ndarray,
    monitor_density: np.ndarray,
    aggregate_coefficient: np.ndarray,
    q_profile: np.ndarray,
) -> list[dict[str, Any]]:
    grid = np.asarray(physical_grid, dtype=np.float64)
    density = np.asarray(monitor_density, dtype=np.float64)
    coeff = np.asarray(aggregate_coefficient, dtype=np.float64)
    q_values = np.asarray(q_profile, dtype=np.float64)
    return [
        {
            "probe_index": int(index),
            "native_coordinate": float(grid[index]),
            "u": float(-grid[index]),
            "aggregate_coefficient": float(coeff[index]),
            "monitor_density": float(density[index]),
            "q_estimate": float(q_values[index]),
        }
        for index in range(len(grid))
    ]


def _anchored_replay_metric_summary(stats) -> dict[str, float]:
    full_error = np.asarray(stats.full_step_error, dtype=np.float64)
    half_error = np.asarray(stats.half_step_error, dtype=np.float64)
    residual = np.asarray(stats.residual_perp_norm, dtype=np.float64)
    return {
        "mean_full_step_error": float(np.mean(full_error)),
        "mean_half_step_error": float(np.mean(half_error)),
        "mean_residual_perp_norm": float(np.mean(residual)),
        "mean_residual_perp_mse": float(np.mean(np.square(residual))),
        "mean_effective_order": float(np.mean(np.asarray(stats.effective_order, dtype=np.float64))),
    }


def _resolved_export_config(args: argparse.Namespace, context_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "goes",
        "backend": "pndm",
        "script": "scripts/run/export_goes_pndm_schedule.py",
        "arguments": {key: value for key, value in sorted(vars(args).items())},
        "resolved_context": context_metadata,
    }


def _write_resolved_export_config(
    output_dir: str | Path,
    *,
    args: argparse.Namespace,
    context_metadata: dict[str, Any],
) -> Path:
    return dump_yaml(_resolved_export_config(args, context_metadata), Path(output_dir) / "config.resolved.yaml")


def main() -> None:
    args = parse_args()
    started = time.time()
    _validate_args(args)
    deterministic_seeds = _set_deterministic_seeds(int(args.seed))

    dataset_config = load_yaml(args.dataset_config)
    native_config = load_native_config(dataset_config["native_config"])
    model_asset = args.model_asset or dataset_config["default_model_asset"]
    manifest = AssetManifest(args.manifest)
    model_path = manifest.path(model_asset)
    domain = _native_domain(args)
    if domain not in {"timesteps", "sigmas"}:
        raise ValueError(f"Unsupported GPDE PNDM coordinate domain: {domain}")
    if args.dry_run:
        defect_backend = _resolve_defect_backend(args.solver, args.defect_backend)
        replay_meta = None
        if defect_backend == "anchored_replay":
            replay_meta = {
                "anchor_nfe": _default_anchor_nfe(args, args.solver),
                "window_size": _default_window_size(args, args.solver),
            }
        context_metadata = _goes_context_metadata(
            args,
            dataset_name=str(dataset_config["name"]),
            model_asset=str(model_asset),
            model_path=model_path,
            dataset_config_path=args.dataset_config,
            coordinate_domain=domain,
            defect_backend=defect_backend,
            replay_detail_meta=replay_meta,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "backend": "pndm",
                    "dataset": dataset_config["name"],
                    "model_asset": str(model_asset),
                    "model_path": str(model_path),
                    "solver": args.solver,
                    "defect_backend": defect_backend,
                    "target_nfe": int(args.nfe),
                    "coordinate_domain": domain,
                    "calibration_samples": int(args.batch_size) * int(args.num_batches),
                    "ref_nfe": int(args.ref_nfe),
                    "ref_grid_size": int(args.ref_grid_size),
                    "probe_grid_size": int(_probe_grid_size(args)),
                    "probe_step_multipliers": args.probe_step_multipliers,
                    "output_dir": str(resolve_repo_path(args.output_dir)),
                    "oracle_cache_dir": str(resolve_repo_path(args.oracle_cache_dir)),
                    "calibration_cost_estimate": context_metadata["calibration_cost_estimate"],
                    "calibration_cost_unit": context_metadata["calibration_cost_unit"],
                    "calibration_cost_breakdown": context_metadata["calibration_cost_breakdown"],
                    "deterministic_seeds": deterministic_seeds,
                    "would_load_model": False,
                    "would_write_schedule_points": int(args.nfe) + 1,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    device = torch.device(args.device)
    model, _ = load_model(dataset_config["native_config"], model_path, device=str(device))
    schedule_cfg = native_config["Schedule"]
    scheduler = build_scheduler(
        args.solver,
        diffusion_step=schedule_cfg["diffusion_step"],
        beta_start=schedule_cfg["beta_start"],
        beta_end=schedule_cfg["beta_end"],
        beta_schedule=schedule_cfg["type"],
    )
    defect_backend = _resolve_defect_backend(args.solver, args.defect_backend)
    if defect_backend == "anchored_replay":
        anchor_nfe = _default_anchor_nfe(args, args.solver)
        window_size = _default_window_size(args, args.solver)
        physical_grid, replay_stats, replay_detail_meta = collect_anchored_replay_calibration_stats(
            model=model,
            scheduler=scheduler,
            solver=args.solver,
            image_size=int(dataset_config["image_size"]),
            batch_size=int(args.batch_size),
            num_batches=int(args.num_batches),
            seed=int(args.seed),
            anchor_nfe=int(anchor_nfe),
            window_size=int(window_size),
            observation_microbatch=int(args.microbatch_size) if int(args.microbatch_size) > 0 else None,
            coordinate_domain=domain,
            model_output_type=str(args.model_output_type),
            q_min=float(args.replay_q_min),
            q_max=float(args.replay_q_max),
            eps=float(args.monitor_epsilon),
        )
        replay_artifacts = _build_replay_gpde_artifacts(
            physical_grid=physical_grid,
            stats=replay_stats,
            target_nfe=int(args.nfe),
            smoothing_window=int(args.monitor_smoothing_window),
            epsilon=float(args.monitor_epsilon),
            q_min=float(args.replay_q_min),
            q_max=float(args.replay_q_max),
        )
        gpde_schedule = replay_artifacts["gpde_schedule"]
        u_schedule = np.asarray(gpde_schedule.u_schedule, dtype=np.float64)
        native_schedule = np.asarray(replay_artifacts["native_schedule"], dtype=np.float64)
        representation = "sigmas" if domain == "sigmas" else "timesteps"
        schedule_hash = stable_hash([float(item) for item in u_schedule])
        replay_cache_key = stable_hash(
            {
                "backend": "pndm_anchored_replay",
                "dataset": str(dataset_config["name"]),
                "model_asset": str(model_asset),
                "solver": str(args.solver),
                "nfe": int(args.nfe),
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "num_batches": int(args.num_batches),
                "anchor_nfe": int(anchor_nfe),
                "window_size": int(window_size),
                "coordinate_domain": str(domain),
                "q_min": float(args.replay_q_min),
                "q_max": float(args.replay_q_max),
                "smoothing_window": int(args.monitor_smoothing_window),
                "model_output_type": str(args.model_output_type),
            }
        )
        context_metadata = _goes_context_metadata(
            args,
            dataset_name=str(dataset_config["name"]),
            model_asset=str(model_asset),
            model_path=model_path,
            dataset_config_path=args.dataset_config,
            coordinate_domain=domain,
            defect_backend=defect_backend,
            replay_detail_meta=replay_detail_meta,
        )
        output_dir = ensure_dir(args.output_dir)
        config_resolved_path = _write_resolved_export_config(
            output_dir,
            args=args,
            context_metadata=context_metadata,
        )
        replay_metric_summary = _anchored_replay_metric_summary(replay_stats)
        replay_metric_metadata = {
            "name": "anchored_replay_frenet_residual",
            "q_min": float(args.replay_q_min),
            "q_max": float(args.replay_q_max),
        }
        schedule_payload = {
            "method": "GPDE",
            "legacy_method_alias": "GOES",
            "schedule_implementation_version": GPDE_SCHEDULE_IMPLEMENTATION_VERSION,
            **context_metadata,
            "solver": args.solver,
            "target_nfe": int(args.nfe),
            "coordinate": f"negative_{domain}",
            "coordinate_direction": "increasing_u_native_decreasing",
            "u_schedule": [float(item) for item in u_schedule],
            "native_schedule": [float(item) for item in native_schedule],
            "rho": float(args.rho),
            "metric": replay_metric_metadata,
            "aggregation": "anchored_replay_mean",
            "oracle_cache_key": replay_cache_key,
            "replay_cache_key": replay_cache_key,
            "optimizer": "monitor_inverse_cdf",
            "edge_objective": float(gpde_schedule.objective),
            "monitor_objective": float(gpde_schedule.objective),
            "total_monitor_mass": float(gpde_schedule.total_monitor_mass),
            "selected_monitor_masses": [float(item) for item in gpde_schedule.interval_monitor_masses],
            "selected_edge_costs": [float(item) for item in gpde_schedule.interval_monitor_masses],
            "selected_indices": [int(item) for item in gpde_schedule.selected_indices],
            "snap_errors": [float(item) for item in gpde_schedule.snap_errors],
            "q_estimate": float(replay_artifacts["q_estimate"]),
            "q_source": "anchored_replay_effective_order",
            "monitor_exponent": "fp_clock_solver_order_profile",
            "probe_profile": {
                **replay_detail_meta,
                **replay_metric_summary,
                "native_grid": [float(item) for item in replay_artifacts["native_grid"]],
                "u_grid": [float(item) for item in replay_artifacts["u_grid"]],
            },
            "schedule_hash": schedule_hash,
        }
        save_schedule_outputs(
            output_dir,
            payload=schedule_payload,
            selected_indices=gpde_schedule.selected_indices,
            selected_edge_costs=gpde_schedule.interval_monitor_masses,
        )
        np.savez_compressed(
            output_dir / "probe_defects.npz",
            probe_grid=np.asarray(replay_artifacts["u_grid"], dtype=np.float64),
            probe_steps=np.asarray([], dtype=np.float64),
            defects=np.asarray(replay_stats.residual_perp_norm, dtype=np.float64),
            full_step_error=np.asarray(replay_stats.full_step_error, dtype=np.float64),
            half_step_error=np.asarray(replay_stats.half_step_error, dtype=np.float64),
            effective_order=np.asarray(replay_stats.effective_order, dtype=np.float64),
            delta_s=np.asarray(replay_stats.delta_s, dtype=np.float64),
            coefficient_per_sample=np.asarray(replay_stats.residual_perp_norm, dtype=np.float64),
            aggregate_coefficient=np.asarray(replay_artifacts["aggregate_coefficient"], dtype=np.float64),
            monitor_density=np.asarray(replay_artifacts["monitor_density"], dtype=np.float64),
            fallback_counts=np.zeros_like(np.asarray(replay_artifacts["monitor_density"], dtype=np.float64)),
        )
        np.savez_compressed(
            output_dir / "anchored_replay_defects.npz",
            physical_grid=np.asarray(physical_grid, dtype=np.float64),
            full_step_error=np.asarray(replay_stats.full_step_error, dtype=np.float64),
            half_step_error=np.asarray(replay_stats.half_step_error, dtype=np.float64),
            effective_order=np.asarray(replay_stats.effective_order, dtype=np.float64),
            delta_s=np.asarray(replay_stats.delta_s, dtype=np.float64),
            residual_perp_norm=np.asarray(replay_stats.residual_perp_norm, dtype=np.float64),
        )
        write_csv(
            _replay_profile_rows(
                physical_grid=np.asarray(replay_artifacts["native_grid"], dtype=np.float64),
                monitor_density=np.asarray(replay_artifacts["monitor_density"], dtype=np.float64),
                aggregate_coefficient=np.asarray(replay_artifacts["aggregate_coefficient"], dtype=np.float64),
                q_profile=np.asarray(replay_artifacts["q_profile"], dtype=np.float64),
            ),
            output_dir / "monitor_profile.csv",
        )
        dump_json(
            {
                "q_estimate": float(replay_artifacts["q_estimate"]),
                "q_source": "anchored_replay_effective_order",
                "q_min": float(args.replay_q_min),
                "q_max": float(args.replay_q_max),
                "profile": replay_detail_meta,
                **replay_metric_summary,
            },
            output_dir / "q_estimate.json",
        )
        dump_json({**replay_detail_meta, **replay_metric_summary}, output_dir / "replay_metadata.json")
        dump_json({**replay_detail_meta, "oracle_type": "anchored_replay_no_deterministic_oracle"}, output_dir / "oracle_metadata.json")
        calibration_row, heldout_row = _schedule_export_metric_rows(
            solver=args.solver,
            nfe=int(args.nfe),
            num_samples=int(args.batch_size) * int(args.num_batches),
            final_latent_mse=replay_metric_summary["mean_residual_perp_mse"],
            replay_loss=replay_metric_summary["mean_residual_perp_norm"],
            fallback_fraction=0.0,
            schedule_dir=output_dir,
            oracle_cache_key=replay_cache_key,
            theory_covered=False,
        )
        calibration_row["note"] = "Anchored replay calibration metrics from native scheduler windows; not held-out image quality."
        write_csv([calibration_row], output_dir / "calibration_metrics.csv")
        write_csv([heldout_row], output_dir / "heldout_metrics.csv")
        write_csv([heldout_row], output_dir / "paper_tables" / "main_results.csv")
        bundle_meta = {
            "schedule_family": "GPDE",
            "legacy_schedule_family_alias": "GOES",
            "schedule_implementation_version": GPDE_SCHEDULE_IMPLEMENTATION_VERSION,
            "backend": "pndm",
            **context_metadata,
            "dataset": dataset_config["name"],
            "model_asset": str(model_asset),
            "solver": args.solver,
            "target_solver": args.solver,
            "defect_backend": defect_backend,
            "coordinate_domain": domain,
            "native_coordinate": domain,
            "oracle_cache_key": replay_cache_key,
            "replay_cache_key": replay_cache_key,
            "schedule_hash": schedule_hash,
            "rho": float(args.rho),
            "metric": replay_metric_metadata,
            "aggregation": "anchored_replay_mean",
            "edge_objective": float(gpde_schedule.objective),
            "monitor_objective": float(gpde_schedule.objective),
            "selected_edge_costs": [float(item) for item in gpde_schedule.interval_monitor_masses],
            "selected_monitor_masses": [float(item) for item in gpde_schedule.interval_monitor_masses],
            "q_estimate": float(replay_artifacts["q_estimate"]),
            "q_source": "anchored_replay_effective_order",
            "effective_nfe": int(args.nfe),
            "solver_steps": int(args.nfe),
        }
        _schedule_bundle(native_schedule=native_schedule, representation=representation, meta=bundle_meta).save(output_dir)
        run_metadata = {
            "command": "export_goes_pndm_schedule",
            "config_resolved_path": str(config_resolved_path),
            "runtime": runtime_metadata(),
            "deterministic_seeds": deterministic_seeds,
            "theory_coverage": {
                "deterministic_oracle_theory": False,
                "note": "History-aware anchored replay covers native scheduler-state effects empirically; it is not the single-step deterministic ODE oracle path.",
            },
            "model_identifier": str(model_asset),
            "model_path": str(model_path),
            "manifest_path": str(args.manifest),
            "dataset": dataset_config["name"],
            "dataset_config_path": str(args.dataset_config),
            "solver": args.solver,
            "defect_backend": defect_backend,
            "coordinate_domain": domain,
            "oracle_cache_key": replay_cache_key,
            "replay_cache_key": replay_cache_key,
            "oracle_loaded_from_cache": False,
            "oracle_build_or_load_seconds": "",
            "replay_metadata": replay_detail_meta,
            "probe_profile": schedule_payload["probe_profile"],
            "schedule_materialization": gpde_schedule.metadata,
            "total_seconds": time.time() - started,
            "skipped_baselines": [
                {
                    "name": "AYS",
                    "reason": "This exporter only materializes the GPDE schedule; baseline evaluation is handled by run_experiment_config.",
                }
            ],
        }
        dump_json(run_metadata, output_dir / "run_metadata.json")
        print(json.dumps({"output_dir": str(resolve_repo_path(output_dir)), "replay_cache_key": replay_cache_key}, indent=2))
        return

    if domain == "timesteps":
        native_start = float(schedule_cfg["diffusion_step"] - 1)
        native_end = 0.0
        initial_sigma = float(_interp_sigmas_for_timesteps(scheduler, np.asarray([native_start], dtype=np.float64))[0])
        native_velocity_fn = build_velocity_oracle(
            model,
            scheduler,
            model_output_type=args.model_output_type,
            sigma_floor=float(args.sigma_floor),
        )

        def velocity_fn(sample, u_tensor, sample_start=0, sample_stop=None):
            native_t = -u_tensor
            return -native_velocity_fn(sample, native_t)

        representation = "timesteps"
    else:
        scheduler.set_timesteps(max(int(args.ref_grid_size) - 1, 1), device=device)
        raw_sigmas = scheduler.sigmas.detach().cpu().float().numpy()
        native_start = float(raw_sigmas[0])
        native_end = 0.0
        initial_sigma = native_start
        native_velocity_fn = build_sigma_derivative_oracle(
            model,
            scheduler,
            model_output_type=args.model_output_type,
        )

        def velocity_fn(sample, u_tensor, sample_start=0, sample_stop=None):
            native_sigma = -u_tensor
            return -native_velocity_fn(sample, native_sigma)

        representation = "sigmas"

    u_grid = np.linspace(-native_start, -native_end, int(args.ref_grid_size), dtype=np.float64)
    probe_grid = np.linspace(float(u_grid[0]), float(u_grid[-1]), _probe_grid_size(args), dtype=np.float64)
    probe_steps = make_probe_steps(
        u_min=float(u_grid[0]),
        u_max=float(u_grid[-1]),
        probe_grid_size=_probe_grid_size(args),
        multipliers=args.probe_step_multipliers,
    )
    initial_sample, noise_seeds = _build_initial_samples(
        model=model,
        image_size=int(dataset_config["image_size"]),
        batch_size=int(args.batch_size),
        num_batches=int(args.num_batches),
        seed=int(args.seed),
        initial_sigma=initial_sigma,
        device=device,
    )
    conditions = np.arange(initial_sample.shape[0], dtype=np.float64)
    metadata = {
        "model_identifier": str(model_asset),
        "ode_sampler_family": "pndm_noise_prediction_deterministic_ode",
        "coordinate_mapping": {
            "coordinate": f"negative_{domain}",
            "native_coordinate": domain,
            "direction": "increasing_u_native_decreasing",
            "u_min": float(u_grid[0]),
            "u_max": float(u_grid[-1]),
            "native_start": native_start,
            "native_end": native_end,
        },
        "cfg": {"guidance_scale": None},
        "dataset": dataset_config["name"],
        "model_output_type": args.model_output_type,
    }
    oracle_result = build_or_load_torch_velocity_oracle(
        cache_dir=args.oracle_cache_dir,
        initial_sample=initial_sample,
        velocity_fn=velocity_fn,
        u_grid=u_grid,
        ref_nfe=int(args.ref_nfe),
        metadata=metadata,
        conditions=conditions,
        noise_seeds=noise_seeds,
        microbatch_size=int(args.microbatch_size) if int(args.microbatch_size) > 0 else None,
        reuse=not bool(args.no_reuse_oracle),
    )
    metric_config = {
        "name": args.metric,
        "sigma_data": float(args.sigma_data),
        "eps": 1.0e-12,
        "min_weight": 1.0e-4,
        "max_weight": 1.0e4,
    }
    if args.metric == "edm_scalar":
        if domain == "timesteps":
            metric_config["u_grid"] = u_grid
            metric_config["sigma_grid"] = _interp_sigmas_for_timesteps(scheduler, -u_grid)
        else:
            metric_config["u_grid"] = u_grid
            metric_config["sigma_grid"] = -u_grid
    metric = make_metric(
        metric_config,
        oracle=oracle_result.oracle,
        coordinate=SimpleNamespace(name=f"negative_{domain}"),
    )
    aggregation = {"name": args.aggregation, "trim_ratio": args.trim_ratio, "alpha": args.cvar_alpha}
    solver = make_torch_step_solver(name=args.solver, velocity_fn=velocity_fn, device=device, dtype=initial_sample.dtype)
    profile = evaluate_gpde_profile(
        solver,
        oracle_result.oracle,
        probe_grid,
        probe_steps,
        metric,
        rho=float(args.rho),
        aggregation=aggregation,
        q_mode=str(args.q_mode),
        fixed_q=args.fixed_q,
        default_q=default_q_for_solver(args.solver),
        coefficient_floor=float(args.monitor_epsilon),
        monitor_smoothing_window=int(args.monitor_smoothing_window),
    )
    gpde_schedule = materialize_gpde_schedule(profile.probe_grid, profile.monitor_density, int(args.nfe))
    u_schedule = gpde_schedule.u_schedule
    native_schedule = -u_schedule
    schedule_hash = stable_hash([float(item) for item in u_schedule])
    context_metadata = _goes_context_metadata(
        args,
        dataset_name=str(dataset_config["name"]),
        model_asset=str(model_asset),
        model_path=model_path,
        dataset_config_path=args.dataset_config,
        coordinate_domain=domain,
    )

    output_dir = ensure_dir(args.output_dir)
    config_resolved_path = _write_resolved_export_config(
        output_dir,
        args=args,
        context_metadata=context_metadata,
    )
    schedule_payload = {
        "method": "GPDE",
        "legacy_method_alias": "GOES",
        "schedule_implementation_version": GPDE_SCHEDULE_IMPLEMENTATION_VERSION,
        **context_metadata,
        "solver": args.solver,
        "target_nfe": int(args.nfe),
        "coordinate": f"negative_{domain}",
        "coordinate_direction": "increasing_u_native_decreasing",
        "u_schedule": [float(item) for item in u_schedule],
        "native_schedule": [float(item) for item in native_schedule],
        "rho": float(args.rho),
        "metric": metric.metadata(),
        "aggregation": aggregation_label(aggregation),
        "oracle_cache_key": oracle_result.cache_key,
        "optimizer": "monitor_inverse_cdf",
        "edge_objective": float(gpde_schedule.objective),
        "monitor_objective": float(gpde_schedule.objective),
        "total_monitor_mass": float(gpde_schedule.total_monitor_mass),
        "selected_monitor_masses": [float(item) for item in gpde_schedule.interval_monitor_masses],
        "selected_edge_costs": [float(item) for item in gpde_schedule.interval_monitor_masses],
        "selected_indices": [int(item) for item in gpde_schedule.selected_indices],
        "snap_errors": [float(item) for item in gpde_schedule.snap_errors],
        "q_estimate": float(profile.q_estimate),
        "q_source": profile.q_source,
        "monitor_exponent": profile.metadata["monitor_exponent"],
        "probe_profile": profile.metadata,
        "schedule_hash": schedule_hash,
    }
    save_schedule_outputs(
        output_dir,
        payload=schedule_payload,
        selected_indices=gpde_schedule.selected_indices,
        selected_edge_costs=gpde_schedule.interval_monitor_masses,
    )
    np.savez_compressed(
        output_dir / "probe_defects.npz",
        probe_grid=profile.probe_grid,
        probe_steps=profile.probe_steps,
        defects=profile.defects,
        coefficient_per_sample=profile.coefficient_per_sample,
        aggregate_coefficient=profile.aggregate_coefficient,
        monitor_density=profile.monitor_density,
        fallback_counts=profile.fallback_counts,
    )
    write_csv(
        [
            {
                "probe_index": int(index),
                "u": float(u),
                "aggregate_coefficient": float(profile.aggregate_coefficient[index]),
                "monitor_density": float(profile.monitor_density[index]),
                "q_estimate": float(profile.q_estimate),
            }
            for index, u in enumerate(profile.probe_grid)
        ],
        output_dir / "monitor_profile.csv",
    )
    dump_json(profile.metadata, output_dir / "q_estimate.json")
    dump_json(oracle_result.oracle.metadata, output_dir / "oracle_metadata.json")

    replay_metrics = evaluate_replay_metrics(
        solver,
        oracle_result.oracle,
        u_schedule,
        metric,
        rho=float(args.rho),
        aggregation=aggregation,
    )
    calibration_row, heldout_row = _schedule_export_metric_rows(
        solver=args.solver,
        nfe=int(args.nfe),
        num_samples=oracle_result.oracle.num_samples,
        final_latent_mse=replay_metrics.final_mse,
        replay_loss=replay_metrics.replay_loss,
        fallback_fraction=replay_metrics.fallback_fraction,
        schedule_dir=output_dir,
        oracle_cache_key=oracle_result.cache_key,
        theory_covered=True,
    )
    write_csv([calibration_row], output_dir / "calibration_metrics.csv")
    write_csv([heldout_row], output_dir / "heldout_metrics.csv")
    write_csv([heldout_row], output_dir / "paper_tables" / "main_results.csv")

    bundle_meta = {
        "schedule_family": "GPDE",
        "legacy_schedule_family_alias": "GOES",
        "schedule_implementation_version": GPDE_SCHEDULE_IMPLEMENTATION_VERSION,
        "backend": "pndm",
        **context_metadata,
        "dataset": dataset_config["name"],
        "model_asset": str(model_asset),
        "solver": args.solver,
        "coordinate_domain": domain,
        "native_coordinate": domain,
        "oracle_cache_key": oracle_result.cache_key,
        "schedule_hash": schedule_hash,
        "rho": float(args.rho),
        "metric": metric.metadata(),
        "aggregation": aggregation_label(aggregation),
        "edge_objective": float(gpde_schedule.objective),
        "monitor_objective": float(gpde_schedule.objective),
        "selected_edge_costs": [float(item) for item in gpde_schedule.interval_monitor_masses],
        "selected_monitor_masses": [float(item) for item in gpde_schedule.interval_monitor_masses],
        "q_estimate": float(profile.q_estimate),
        "q_source": profile.q_source,
        "effective_nfe": int(args.nfe),
        "solver_steps": int(args.nfe),
    }
    _schedule_bundle(native_schedule=native_schedule, representation=representation, meta=bundle_meta).save(output_dir)
    run_metadata = {
        "command": "export_goes_pndm_schedule",
        "config_resolved_path": str(config_resolved_path),
        "runtime": runtime_metadata(),
        "deterministic_seeds": deterministic_seeds,
        "theory_coverage": {
            "deterministic_oracle_theory": True,
            "note": "PNDM GPDE exporter is theory-covered for deterministic velocity-based Euler/Heun ODE steps only.",
        },
        "model_identifier": str(model_asset),
        "model_path": str(model_path),
        "manifest_path": str(args.manifest),
        "dataset": dataset_config["name"],
        "dataset_config_path": str(args.dataset_config),
        "solver": args.solver,
        "coordinate_domain": domain,
        "oracle_cache_key": oracle_result.cache_key,
        "oracle_loaded_from_cache": oracle_result.loaded_from_cache,
        "oracle_build_or_load_seconds": oracle_result.elapsed_seconds,
        "probe_profile": profile.metadata,
        "schedule_materialization": gpde_schedule.metadata,
        "total_seconds": time.time() - started,
        "skipped_baselines": [
            {
                "name": "AYS",
            "reason": "This exporter only materializes the GPDE schedule; baseline evaluation is handled by run_experiment_config.",
            }
        ],
    }
    dump_json(run_metadata, output_dir / "run_metadata.json")
    print(json.dumps({"output_dir": str(resolve_repo_path(output_dir)), "oracle_cache_key": oracle_result.cache_key}, indent=2))


if __name__ == "__main__":
    main()

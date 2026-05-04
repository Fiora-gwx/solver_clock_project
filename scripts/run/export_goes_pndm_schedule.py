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
from src.adapters.pndm import (
    _interp_sigmas_for_timesteps,
    build_scheduler,
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
    parser.add_argument("--solver", default="euler", choices=["euler", "heun2"])
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
    return int(args.probe_grid_size if args.candidate_grid_size is None else args.candidate_grid_size)


def _native_domain(args: argparse.Namespace) -> str:
    if args.coordinate_domain != "auto":
        return str(args.coordinate_domain)
    return str(preferred_calibration_domain(args.solver))


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
        raise ValueError("--probe-grid-size must be at least --nfe + 1.")
    if int(args.monitor_smoothing_window) < 1:
        raise ValueError("--monitor-smoothing-window must be positive.")
    if float(args.monitor_epsilon) <= 0.0:
        raise ValueError("--monitor-epsilon must be positive.")
    if args.q_mode == "fixed" and (args.fixed_q is None or float(args.fixed_q) <= 0.0):
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
) -> dict[str, Any]:
    microbatch_size = int(args.microbatch_size) if int(args.microbatch_size) > 0 else None
    calibration_samples = int(args.batch_size) * int(args.num_batches)
    probe_nodes = _probe_grid_size(args)
    probe_step_count = len(parse_float_list(args.probe_step_multipliers, default=(1.0, 2.0, 4.0)))
    solver_evals_per_edge = 2 if str(args.solver).lower().replace("-", "_") == "heun2" else 1
    oracle_cost_per_sample = 4 * int(args.ref_nfe) + int(args.ref_grid_size)
    probe_cost_per_sample = probe_nodes * probe_step_count * solver_evals_per_edge
    calibration_cost = calibration_samples * (oracle_cost_per_sample + probe_cost_per_sample)
    return {
        "dataset": str(dataset_name),
        "model_asset": str(model_asset),
        "model_path": "" if model_path is None else str(model_path),
        "dataset_config_path": "" if dataset_config_path is None else str(dataset_config_path),
        "seed": int(args.seed),
        "guidance_scale": None,
        "coordinate_domain": str(coordinate_domain),
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
        "probe_grid_config": {
            "size": int(probe_nodes),
            "type": "uniform_in_negative_native_coordinate",
            "probe_step_multipliers": parse_float_list(args.probe_step_multipliers, default=(1.0, 2.0, 4.0)),
            "q_mode": str(args.q_mode),
            "fixed_q": None if args.fixed_q is None else float(args.fixed_q),
            "monitor_smoothing_window": int(args.monitor_smoothing_window),
            "monitor_epsilon": float(args.monitor_epsilon),
        },
        "calibration_cost_estimate": int(calibration_cost),
        "calibration_cost_unit": "model_evaluation_equivalents",
        "calibration_cost_breakdown": {
            "num_samples": calibration_samples,
            "cfg_multiplier": 1,
            "oracle_cost_per_sample": int(oracle_cost_per_sample),
            "probe_nodes": int(probe_nodes),
            "probe_step_count": int(probe_step_count),
            "solver_evals_per_edge": int(solver_evals_per_edge),
            "probe_cost_per_sample": int(probe_cost_per_sample),
            "total_model_eval_equivalents": int(calibration_cost),
            "note": "Estimated RK4 oracle drift calls plus oracle-start GPDE probe drift calls; excludes generation.",
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
        context_metadata = _goes_context_metadata(
            args,
            dataset_name=str(dataset_config["name"]),
            model_asset=str(model_asset),
            model_path=model_path,
            dataset_config_path=args.dataset_config,
            coordinate_domain=domain,
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

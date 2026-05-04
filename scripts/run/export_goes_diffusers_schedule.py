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
from src.adapters.diffusers import (
    _pipeline_kind,
    build_defect_sigma_grid,
    get_pipeline_device,
    load_pipeline,
    prepare_defect_batch,
    replace_scheduler,
)
from src.utils.assets import AssetManifest
from src.utils.config import dump_yaml, ensure_dir, load_json, resolve_repo_path
from src.utils.schedule_bundle import ScheduleBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a GPDE schedule for a project diffusers pipeline.")
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--model-asset", required=True)
    parser.add_argument("--prompt-asset", default="diffusers_smoke_prompts")
    parser.add_argument("--solver", default="flow_euler")
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-cache-dir", default="outputs/goes/diffusers_oracle_cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--ref-nfe", type=int, default=256)
    parser.add_argument("--ref-grid-size", type=int, default=257)
    parser.add_argument("--probe-grid-size", type=int, default=128)
    parser.add_argument("--candidate-grid-size", type=int, default=None, help="Compatibility alias for --probe-grid-size.")
    parser.add_argument("--probe-step-multipliers", default="1,2,4")
    parser.add_argument("--q-mode", choices=["global_fit", "fixed"], default="global_fit")
    parser.add_argument("--fixed-q", type=float)
    parser.add_argument("--monitor-smoothing-window", type=int, default=3)
    parser.add_argument("--monitor-epsilon", type=float, default=1.0e-12)
    parser.add_argument(
        "--physical-grid-mode",
        choices=["scheduler_sigmas", "linear_sigma", "log_sigma", "karras_sigma"],
        default="scheduler_sigmas",
    )
    parser.add_argument("--metric", choices=["identity", "edm_scalar", "channel_whitened"], default="identity")
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--aggregation", choices=["mean", "median", "trimmed_mean", "cvar"], default="cvar")
    parser.add_argument("--trim-ratio", type=float, default=0.10)
    parser.add_argument("--cvar-alpha", type=float, default=0.80)
    parser.add_argument("--no-reuse-oracle", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments and print the planned run without loading a pipeline.")
    return parser.parse_args()


def _probe_grid_size(args: argparse.Namespace) -> int:
    return int(args.probe_grid_size if args.candidate_grid_size is None else args.candidate_grid_size)


def _load_prompt_batch(manifest: AssetManifest, prompt_asset_or_path: str, total_samples: int) -> list[str]:
    prompt_path = manifest.path(prompt_asset_or_path) if manifest.has(prompt_asset_or_path) else prompt_asset_or_path
    prompts = load_json(prompt_path)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt asset must be a non-empty JSON list.")
    return [str(prompts[index % len(prompts)]) for index in range(int(total_samples))]


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.nfe) <= 0:
        raise ValueError("--nfe must be positive.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.num_batches) <= 0:
        raise ValueError("--num-batches must be positive.")
    if int(args.microbatch_size) < 0:
        raise ValueError("--microbatch-size must be non-negative.")
    if int(args.height) <= 0:
        raise ValueError("--height must be positive.")
    if int(args.width) <= 0:
        raise ValueError("--width must be positive.")
    guidance_scale = float(args.guidance_scale)
    if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
        raise ValueError("--guidance-scale must be finite and non-negative.")
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


def _theory_coverage_for_pipeline(kind: str, solver: str) -> dict[str, Any]:
    normalized_solver = str(solver).lower().replace("-", "_")
    flow_like = kind in {"flux", "sd3", "lumina2"} and normalized_solver in {
        "flow_euler",
        "flow_heun",
        "flow_dpm_solver",
        "flow_unipc",
    }
    if flow_like:
        return {
            "deterministic_oracle_theory": normalized_solver in {"flow_euler", "flow_heun"},
            "coverage_note": (
                "Flow pipeline with velocity-style defect batch. GPDE monitor probes are strict for velocity one-step "
                "Euler/Heun proxies; multistep solvers are outside this monitor-only implementation."
            ),
        }
    return {
        "deterministic_oracle_theory": False,
        "coverage_note": (
            "Empirical-only for VP/SD-style pipelines: adapter defect batches expose scheduler model outputs, "
            "not a solver-independent probability-flow ODE drift."
        ),
    }


def _validate_solver_pipeline_pair(kind: str, solver: str) -> str:
    normalized = str(solver).lower().replace("-", "_")
    flow_kinds = {"flux", "sd3", "lumina2"}
    vp_kinds = {"stable_diffusion", "sdxl", "deepfloyd_if"}
    if kind in flow_kinds:
        if normalized not in {"flow_euler", "flow_heun"}:
            raise ValueError(
                f"GPDE diffusers export for flow pipeline `{kind}` currently supports flow_euler/flow_heun only, "
                f"got `{solver}`. Multistep solvers require black-box replay refinement."
            )
        return "heun2" if normalized == "flow_heun" else "euler"
    if kind in vp_kinds:
        if normalized != "euler":
            raise ValueError(
                f"GPDE diffusers export for VP pipeline `{kind}` currently supports empirical euler only, "
                f"got `{solver}`. DPM/UniPC/SDE solvers need scheduler-history replay refinement."
            )
        return "euler"
    raise ValueError(f"Unsupported diffusers pipeline kind for GPDE export: {kind}")


def _schedule_bundle(native_sigmas: np.ndarray, meta: dict[str, Any]) -> ScheduleBundle:
    sigmas = np.asarray(native_sigmas, dtype=np.float64)
    return ScheduleBundle(
        sigmas=sigmas[:-1].copy(),
        sigma_grid=sigmas.copy(),
        meta={**meta, "representation": "sigmas", "terminal_sigma": float(sigmas[-1])},
    )


def _goes_context_metadata(
    args: argparse.Namespace,
    *,
    pipeline_kind: str,
    prompt_count: int,
    model_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
) -> dict[str, Any]:
    microbatch_size = int(args.microbatch_size) if int(args.microbatch_size) > 0 else None
    calibration_samples = int(args.batch_size) * int(args.num_batches)
    probe_nodes = _probe_grid_size(args)
    probe_step_count = len(parse_float_list(args.probe_step_multipliers, default=(1.0, 2.0, 4.0)))
    normalized_solver = str(args.solver).lower().replace("-", "_")
    solver_evals_per_edge = 2 if normalized_solver == "flow_heun" else 1
    cfg_multiplier = 2 if float(args.guidance_scale) != 1.0 else 1
    oracle_cost_per_sample = 4 * int(args.ref_nfe) + int(args.ref_grid_size)
    probe_cost_per_sample = probe_nodes * probe_step_count * solver_evals_per_edge
    calibration_cost = calibration_samples * cfg_multiplier * (oracle_cost_per_sample + probe_cost_per_sample)
    return {
        "model_asset": str(args.model_asset),
        "model_path": "" if model_path is None else str(model_path),
        "seed": int(args.seed),
        "guidance_scale": float(args.guidance_scale),
        "prompt_asset": str(args.prompt_asset),
        "prompt_path": "" if prompt_path is None else str(prompt_path),
        "prompt_count": int(prompt_count),
        "pipeline_kind": str(pipeline_kind),
        "height": int(args.height),
        "width": int(args.width),
        "dtype": str(args.dtype),
        "coordinate_domain": "sigmas",
        "physical_grid_mode": str(args.physical_grid_mode),
        "calibration_config": {
            "num_samples": calibration_samples,
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "microbatch_size": microbatch_size,
            "seed": int(args.seed),
            "guidance_scale": float(args.guidance_scale),
            "prompt_asset": str(args.prompt_asset),
            "prompt_count": int(prompt_count),
        },
        "pilot_config": {
            "num_samples": calibration_samples,
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "microbatch_size": microbatch_size,
            "seed": int(args.seed),
            "guidance_scale": float(args.guidance_scale),
            "prompt_asset": str(args.prompt_asset),
            "prompt_count": int(prompt_count),
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
            "type": "uniform_in_negative_sigma",
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
            "cfg_multiplier": int(cfg_multiplier),
            "oracle_cost_per_sample": int(oracle_cost_per_sample),
            "probe_nodes": int(probe_nodes),
            "probe_step_count": int(probe_step_count),
            "solver_evals_per_edge": int(solver_evals_per_edge),
            "probe_cost_per_sample": int(probe_cost_per_sample),
            "total_model_eval_equivalents": int(calibration_cost),
            "note": "Estimated RK4 oracle drift calls plus oracle-start GPDE probe drift calls; CFG multiplier counts unconditional/conditional branches and excludes generation.",
        },
    }


def _schedule_export_metric_rows(
    *,
    solver: str,
    nfe: int,
    guidance_scale: float,
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
        "guidance_scale": float(guidance_scale),
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
        "backend": "diffusers",
        "script": "scripts/run/export_goes_diffusers_schedule.py",
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

    manifest = AssetManifest(args.manifest)
    model_path = manifest.path(args.model_asset)
    prompt_path = manifest.path(args.prompt_asset) if manifest.has(args.prompt_asset) else args.prompt_asset
    prompts = _load_prompt_batch(manifest, args.prompt_asset, int(args.batch_size) * int(args.num_batches))
    if args.dry_run:
        context_metadata = _goes_context_metadata(
            args,
            pipeline_kind="dry_run_unvalidated",
            prompt_count=len(prompts),
            model_path=model_path,
            prompt_path=prompt_path,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "backend": "diffusers",
                    "model_asset": str(args.model_asset),
                    "model_path": str(model_path),
                    "prompt_asset": str(args.prompt_asset),
                    "prompt_path": str(prompt_path),
                    "prompt_count": len(prompts),
                    "solver": args.solver,
                    "target_nfe": int(args.nfe),
                    "guidance_scale": float(args.guidance_scale),
                    "height": int(args.height),
                    "width": int(args.width),
                    "calibration_samples": int(args.batch_size) * int(args.num_batches),
                    "ref_nfe": int(args.ref_nfe),
                    "ref_grid_size": int(args.ref_grid_size),
                    "probe_grid_size": int(_probe_grid_size(args)),
                    "probe_step_multipliers": args.probe_step_multipliers,
                    "physical_grid_mode": args.physical_grid_mode,
                    "output_dir": str(resolve_repo_path(args.output_dir)),
                    "oracle_cache_dir": str(resolve_repo_path(args.oracle_cache_dir)),
                    "calibration_cost_estimate": context_metadata["calibration_cost_estimate"],
                    "calibration_cost_unit": context_metadata["calibration_cost_unit"],
                    "calibration_cost_breakdown": context_metadata["calibration_cost_breakdown"],
                    "deterministic_seeds": deterministic_seeds,
                    "would_load_pipeline": False,
                    "would_write_schedule_points": int(args.nfe) + 1,
                    "solver_pipeline_validation": "requires pipeline class; run without --dry-run to validate exact support",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    pipeline = load_pipeline(model_path, device=args.device, dtype_name=args.dtype)
    kind = _pipeline_kind(pipeline)
    proxy_solver = _validate_solver_pipeline_pair(kind, args.solver)
    replace_scheduler(pipeline, args.solver)
    device = get_pipeline_device(pipeline)

    # Initialize scheduler sigmas before latent preparation, then build the
    # actual reference grid after the prompt-conditioned defect batch exists.
    build_defect_sigma_grid(
        pipeline,
        physical_grid_size=max(int(args.ref_grid_size), 3),
        height=int(args.height),
        width=int(args.width),
        physical_grid_mode=args.physical_grid_mode,
    )
    defect_batch = prepare_defect_batch(
        pipeline,
        prompt=prompts,
        batch_size=len(prompts),
        seed=int(args.seed),
        height=int(args.height),
        width=int(args.width),
        guidance_scale=float(args.guidance_scale),
    )
    sigma_grid = build_defect_sigma_grid(
        pipeline,
        physical_grid_size=int(args.ref_grid_size),
        height=int(args.height),
        width=int(args.width),
        physical_grid_mode=args.physical_grid_mode,
    )
    sigma_grid[0] = float(defect_batch.sigma_max)
    sigma_grid[-1] = 0.0
    if np.any(np.diff(sigma_grid) > 1.0e-8):
        raise RuntimeError("Diffusers GPDE sigma grid must be non-increasing.")
    u_grid = -np.asarray(sigma_grid, dtype=np.float64)
    if np.any(np.diff(u_grid) <= -1.0e-8):
        raise RuntimeError("Diffusers GPDE unified grid must be non-decreasing after sigma sign flip.")
    # Repair repeated terminal or scheduler nodes for interpolation.
    keep = np.concatenate([[True], np.diff(u_grid) > 1.0e-10])
    u_grid = u_grid[keep]
    if len(u_grid) < 2:
        raise RuntimeError("Diffusers GPDE reference grid collapsed to fewer than two nodes.")
    probe_grid = np.linspace(float(u_grid[0]), float(u_grid[-1]), _probe_grid_size(args), dtype=np.float64)
    probe_steps = make_probe_steps(
        u_min=float(u_grid[0]),
        u_max=float(u_grid[-1]),
        probe_grid_size=_probe_grid_size(args),
        multipliers=args.probe_step_multipliers,
    )

    def velocity_fn(sample, u_tensor, batch_start=0, batch_stop=None):
        sigma = -u_tensor
        return -defect_batch.velocity_fn(sample, sigma, batch_start, batch_stop)

    conditions = np.asarray(prompts)
    noise_seeds = np.arange(int(args.seed), int(args.seed) + len(prompts), dtype=np.int64)
    coverage = _theory_coverage_for_pipeline(kind, args.solver)
    metadata = {
        "model_identifier": str(args.model_asset),
        "ode_sampler_family": f"diffusers_{kind}_sigma_coordinate",
        "coordinate_mapping": {
            "coordinate": "negative_sigma",
            "native_coordinate": "sigmas",
            "direction": "increasing_u_native_decreasing",
            "u_min": float(u_grid[0]),
            "u_max": float(u_grid[-1]),
            "native_sigma_start": float(-u_grid[0]),
            "native_sigma_end": float(-u_grid[-1]),
            "physical_grid_mode": args.physical_grid_mode,
        },
        "cfg": {"guidance_scale": float(args.guidance_scale)},
        "prompt_split_hash": stable_hash(prompts),
        "prompt_asset": str(args.prompt_asset),
        "pipeline_kind": kind,
        **coverage,
    }
    oracle_result = build_or_load_torch_velocity_oracle(
        cache_dir=args.oracle_cache_dir,
        initial_sample=defect_batch.initial_latents,
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
        metric_config["u_grid"] = u_grid
        metric_config["sigma_grid"] = -u_grid
    metric = make_metric(
        metric_config,
        oracle=oracle_result.oracle,
        coordinate=SimpleNamespace(name="negative_sigma"),
    )
    aggregation = {"name": args.aggregation, "trim_ratio": args.trim_ratio, "alpha": args.cvar_alpha}
    solver_proxy = make_torch_step_solver(
        name=proxy_solver,
        velocity_fn=velocity_fn,
        device=device,
        dtype=defect_batch.initial_latents.dtype,
    )
    profile = evaluate_gpde_profile(
        solver_proxy,
        oracle=oracle_result.oracle,
        probe_grid=probe_grid,
        probe_steps=probe_steps,
        metric=metric,
        rho=float(args.rho),
        aggregation=aggregation,
        q_mode=str(args.q_mode),
        fixed_q=args.fixed_q,
        default_q=default_q_for_solver(proxy_solver),
        coefficient_floor=float(args.monitor_epsilon),
        monitor_smoothing_window=int(args.monitor_smoothing_window),
    )
    gpde_schedule = materialize_gpde_schedule(profile.probe_grid, profile.monitor_density, int(args.nfe))
    u_schedule = gpde_schedule.u_schedule
    native_sigmas = -u_schedule
    native_sigmas[-1] = 0.0

    output_dir = ensure_dir(args.output_dir)
    schedule_hash = stable_hash([float(item) for item in u_schedule])
    context_metadata = _goes_context_metadata(
        args,
        pipeline_kind=kind,
        prompt_count=len(prompts),
        model_path=model_path,
        prompt_path=prompt_path,
    )
    config_resolved_path = _write_resolved_export_config(
        output_dir,
        args=args,
        context_metadata=context_metadata,
    )
    payload = {
        "method": "GPDE",
        "legacy_method_alias": "GOES",
        "schedule_implementation_version": GPDE_SCHEDULE_IMPLEMENTATION_VERSION,
        **context_metadata,
        "solver": args.solver,
        "target_nfe": int(args.nfe),
        "coordinate": "negative_sigma",
        "coordinate_direction": "increasing_u_native_decreasing",
        "u_schedule": [float(item) for item in u_schedule],
        "native_schedule": [float(item) for item in native_sigmas],
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
        payload=payload,
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
        solver_proxy,
        oracle_result.oracle,
        u_schedule,
        metric,
        rho=float(args.rho),
        aggregation=aggregation,
    )
    calibration_row, heldout_row = _schedule_export_metric_rows(
        solver=args.solver,
        nfe=int(args.nfe),
        guidance_scale=float(args.guidance_scale),
        num_samples=oracle_result.oracle.num_samples,
        final_latent_mse=replay_metrics.final_mse,
        replay_loss=replay_metrics.replay_loss,
        fallback_fraction=replay_metrics.fallback_fraction,
        schedule_dir=output_dir,
        oracle_cache_key=oracle_result.cache_key,
        theory_covered=bool(coverage["deterministic_oracle_theory"]),
    )
    write_csv([calibration_row], output_dir / "calibration_metrics.csv")
    write_csv([heldout_row], output_dir / "heldout_metrics.csv")
    write_csv([heldout_row], output_dir / "paper_tables" / "main_results.csv")
    bundle_meta = {
        "schedule_family": "GPDE",
        "legacy_schedule_family_alias": "GOES",
        "schedule_implementation_version": GPDE_SCHEDULE_IMPLEMENTATION_VERSION,
        "backend": "diffusers",
        **context_metadata,
        "model_asset": str(args.model_asset),
        "solver": args.solver,
        "proxy_solver": proxy_solver,
        "pipeline_kind": kind,
        "coordinate_domain": "sigmas",
        "native_coordinate": "sigmas",
        "guidance_scale": float(args.guidance_scale),
        "prompt_asset": str(args.prompt_asset),
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
        **coverage,
    }
    _schedule_bundle(native_sigmas, bundle_meta).save(output_dir)
    run_metadata = {
        "command": "export_goes_diffusers_schedule",
        "config_resolved_path": str(config_resolved_path),
        "runtime": runtime_metadata(),
        "deterministic_seeds": deterministic_seeds,
        "model_identifier": str(args.model_asset),
        "model_path": str(model_path),
        "manifest_path": str(args.manifest),
        "pipeline_kind": kind,
        "solver": args.solver,
        "proxy_solver": proxy_solver,
        "guidance_scale": float(args.guidance_scale),
        "prompt_asset": str(args.prompt_asset),
        "prompt_path": str(prompt_path),
        "oracle_cache_key": oracle_result.cache_key,
        "oracle_loaded_from_cache": oracle_result.loaded_from_cache,
        "oracle_build_or_load_seconds": oracle_result.elapsed_seconds,
        "probe_profile": profile.metadata,
        "schedule_materialization": gpde_schedule.metadata,
        "total_seconds": time.time() - started,
        "theory_coverage": coverage,
        "skipped_baselines": [
            {
                "name": "AYS/base generation",
                "reason": "This exporter materializes a GPDE schedule; generation/evaluation remains in run_experiment_config.",
            }
        ],
    }
    dump_json(run_metadata, output_dir / "run_metadata.json")
    print(json.dumps({"output_dir": str(resolve_repo_path(output_dir)), "oracle_cache_key": oracle_result.cache_key}, indent=2))


if __name__ == "__main__":
    main()

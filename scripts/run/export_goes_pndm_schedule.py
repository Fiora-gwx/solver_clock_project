#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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
from goes.dp_minimax import solve_minimax_schedule
from goes.edge_evaluator import EdgeCostTable, evaluate_replay_metrics
from goes.logging_utils import dump_json, runtime_metadata, write_csv
from goes.metrics import make_metric
from goes.schedules import GOES_SCHEDULE_IMPLEMENTATION_VERSION, save_schedule_outputs
from goes.torch_backend import (
    build_or_load_torch_velocity_oracle,
    evaluate_torch_velocity_edge_table,
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
    parser = argparse.ArgumentParser(description="Export a GOES schedule for a PNDM noise-prediction model.")
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
    parser.add_argument("--candidate-grid-size", type=int, default=128)
    parser.add_argument("--coordinate-domain", choices=["auto", "timesteps", "sigmas"], default="auto")
    parser.add_argument("--metric", choices=["identity", "edm_scalar", "channel_whitened"], default="identity")
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--aggregation", choices=["mean", "median", "trimmed_mean", "cvar"], default="trimmed_mean")
    parser.add_argument("--trim-ratio", type=float, default=0.10)
    parser.add_argument("--cvar-alpha", type=float, default=0.80)
    parser.add_argument("--model-output-type", default="epsilon")
    parser.add_argument("--sigma-floor", type=float, default=1.0e-6)
    parser.add_argument("--no-reuse-oracle", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments and print the planned run without loading a model.")
    return parser.parse_args()


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
    if int(args.candidate_grid_size) < int(args.nfe):
        raise ValueError("--candidate-grid-size must be at least --nfe.")
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
    candidate_nodes = int(args.candidate_grid_size) + 1
    candidate_edges = candidate_nodes * (candidate_nodes - 1) // 2
    solver_evals_per_edge = 2 if str(args.solver).lower().replace("-", "_") == "heun2" else 1
    oracle_cost_per_sample = 4 * int(args.ref_nfe) + int(args.ref_grid_size)
    edge_cost_per_sample = candidate_edges * solver_evals_per_edge
    calibration_cost = calibration_samples * (oracle_cost_per_sample + edge_cost_per_sample)
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
        "candidate_grid_config": {
            "size": int(args.candidate_grid_size),
            "type": "uniform_in_negative_native_coordinate",
        },
        "calibration_cost_estimate": int(calibration_cost),
        "calibration_cost_unit": "model_evaluation_equivalents",
        "calibration_cost_breakdown": {
            "num_samples": calibration_samples,
            "cfg_multiplier": 1,
            "oracle_cost_per_sample": int(oracle_cost_per_sample),
            "candidate_edges": int(candidate_edges),
            "solver_evals_per_edge": int(solver_evals_per_edge),
            "edge_cost_per_sample": int(edge_cost_per_sample),
            "total_model_eval_equivalents": int(calibration_cost),
            "note": "Estimated RK4 oracle drift calls plus one-step edge replay drift calls; excludes generation.",
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
        "schedule": "GOES",
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
        "note": "Calibration replay metrics from GOES schedule export; not held-out image quality.",
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


def _edge_table_cache_metadata(
    *,
    args: argparse.Namespace,
    oracle_cache_key: str,
    candidate_grid: np.ndarray,
    metric_metadata: dict[str, Any],
    aggregation: dict[str, Any],
    coordinate_domain: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "cache_version": 1,
        "backend": "pndm",
        "method": "GOES",
        "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
        "solver": str(args.solver),
        "coordinate_domain": str(coordinate_domain),
        "oracle_cache_key": str(oracle_cache_key),
        "candidate_grid_size": int(args.candidate_grid_size),
        "candidate_grid_hash": stable_hash(np.asarray(candidate_grid, dtype=np.float64).tolist(), length=24),
        "metric": dict(metric_metadata),
        "rho": float(args.rho),
        "aggregation": dict(aggregation),
        "dtype": str(dtype).replace("torch.", ""),
        "model_output_type": str(args.model_output_type),
        "sigma_floor": float(args.sigma_floor),
    }


def _edge_table_cache_paths(cache_dir: str | Path, cache_key: str) -> tuple[Path, Path, Path]:
    root = ensure_dir(Path(cache_dir) / "edge_tables")
    return root / f"{cache_key}.npz", root / f"{cache_key}.json", root / f"{cache_key}.lock"


def _load_edge_table_from_cache(table_path: Path, metadata_path: Path) -> EdgeCostTable:
    metadata = json.loads(metadata_path.read_text())
    with np.load(table_path) as payload:
        edge_table_metadata = dict(metadata.get("edge_table_metadata", {}))
        edge_table_metadata["edge_cache_key"] = metadata["edge_cache_key"]
        edge_table_metadata["loaded_from_edge_cache"] = True
        return EdgeCostTable(
            candidate_grid=np.asarray(payload["candidate_grid"], dtype=np.float64),
            edge_costs=np.asarray(payload["edge_costs"], dtype=np.float64),
            per_sample_costs=np.asarray(payload["per_sample_costs"], dtype=np.float64),
            fallback_counts=np.asarray(payload["fallback_counts"], dtype=np.int64),
            fallback_fraction=float(metadata["fallback_fraction"]),
            metadata=edge_table_metadata,
        )


def _save_edge_table_to_cache(
    *,
    table: EdgeCostTable,
    table_path: Path,
    metadata_path: Path,
    cache_metadata: dict[str, Any],
    cache_key: str,
) -> None:
    ensure_dir(table_path.parent)
    temp_table_path = table_path.with_name(f"{table_path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temp_table_path,
        candidate_grid=table.candidate_grid,
        edge_costs=table.edge_costs,
        per_sample_costs=table.per_sample_costs,
        fallback_counts=table.fallback_counts,
    )
    os.replace(temp_table_path, table_path)
    dump_json(
        {
            **cache_metadata,
            "edge_cache_key": cache_key,
            "fallback_fraction": float(table.fallback_fraction),
            "edge_table_metadata": dict(table.metadata),
        },
        metadata_path,
    )


def _acquire_edge_cache_lock(lock_path: Path, *, stale_seconds: float = 6 * 60 * 60) -> bool:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return False
        if age > stale_seconds:
            lock_path.unlink(missing_ok=True)
        return False
    with os.fdopen(fd, "w") as handle:
        handle.write(f"pid={os.getpid()}\n")
        handle.write(f"started={time.time()}\n")
    return True


def _evaluate_or_load_edge_table(
    *,
    args: argparse.Namespace,
    velocity_fn,
    oracle,
    candidate_grid: np.ndarray,
    metric,
    aggregation: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    oracle_cache_key: str,
    coordinate_domain: str,
) -> EdgeCostTable:
    cache_metadata = _edge_table_cache_metadata(
        args=args,
        oracle_cache_key=oracle_cache_key,
        candidate_grid=candidate_grid,
        metric_metadata=metric.metadata(),
        aggregation=aggregation,
        coordinate_domain=coordinate_domain,
        dtype=dtype,
    )
    cache_key = stable_hash(cache_metadata, length=24)
    table_path, metadata_path, lock_path = _edge_table_cache_paths(args.oracle_cache_dir, cache_key)

    if table_path.exists() and metadata_path.exists():
        print(json.dumps({"edge_table_cache": "hit", "edge_cache_key": cache_key}), flush=True)
        return _load_edge_table_from_cache(table_path, metadata_path)

    acquired = False
    announced_wait = False
    try:
        while not acquired:
            acquired = _acquire_edge_cache_lock(lock_path)
            if acquired:
                break
            if table_path.exists() and metadata_path.exists():
                print(json.dumps({"edge_table_cache": "hit_after_wait", "edge_cache_key": cache_key}), flush=True)
                return _load_edge_table_from_cache(table_path, metadata_path)
            if not announced_wait:
                print(json.dumps({"edge_table_cache": "waiting", "edge_cache_key": cache_key}), flush=True)
                announced_wait = True
            time.sleep(10.0)

        if table_path.exists() and metadata_path.exists():
            print(json.dumps({"edge_table_cache": "hit_after_lock", "edge_cache_key": cache_key}), flush=True)
            return _load_edge_table_from_cache(table_path, metadata_path)

        print(json.dumps({"edge_table_cache": "miss", "edge_cache_key": cache_key}), flush=True)
        table = evaluate_torch_velocity_edge_table(
            solver_name=args.solver,
            velocity_fn=velocity_fn,
            oracle=oracle,
            candidate_grid=candidate_grid,
            metric=metric,
            rho=float(args.rho),
            aggregation=aggregation,
            device=device,
            dtype=dtype,
        )
        table.metadata["edge_cache_key"] = cache_key
        table.metadata["loaded_from_edge_cache"] = False
        _save_edge_table_to_cache(
            table=table,
            table_path=table_path,
            metadata_path=metadata_path,
            cache_metadata=cache_metadata,
            cache_key=cache_key,
        )
        return table
    finally:
        if acquired:
            lock_path.unlink(missing_ok=True)


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
        raise ValueError(f"Unsupported GOES PNDM coordinate domain: {domain}")
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
                    "candidate_grid_size": int(args.candidate_grid_size),
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
    candidate_grid = np.linspace(float(u_grid[0]), float(u_grid[-1]), int(args.candidate_grid_size) + 1)
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
    edge_table = _evaluate_or_load_edge_table(
        args=args,
        velocity_fn=velocity_fn,
        oracle=oracle_result.oracle,
        candidate_grid=candidate_grid,
        metric=metric,
        aggregation=aggregation,
        device=device,
        dtype=initial_sample.dtype,
        oracle_cache_key=oracle_result.cache_key,
        coordinate_domain=domain,
    )
    path = solve_minimax_schedule(edge_table.edge_costs, int(args.nfe))
    u_schedule = candidate_grid[path.indices]
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
        "method": "GOES",
        "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
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
        "edge_objective": float(path.objective),
        "selected_edge_costs": [float(item) for item in path.edge_costs],
        "selected_indices": [int(item) for item in path.indices],
        "schedule_hash": schedule_hash,
    }
    save_schedule_outputs(
        output_dir,
        payload=schedule_payload,
        selected_indices=path.indices,
        selected_edge_costs=path.edge_costs,
    )
    np.savez_compressed(
        output_dir / "edge_costs.npz",
        candidate_grid=edge_table.candidate_grid,
        edge_costs=edge_table.edge_costs,
        per_sample_costs=edge_table.per_sample_costs,
        fallback_counts=edge_table.fallback_counts,
    )
    dump_json(oracle_result.oracle.metadata, output_dir / "oracle_metadata.json")

    solver = make_torch_step_solver(name=args.solver, velocity_fn=velocity_fn, device=device, dtype=initial_sample.dtype)
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
        "schedule_family": "GOES",
        "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
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
        "edge_objective": float(path.objective),
        "selected_edge_costs": [float(item) for item in path.edge_costs],
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
            "note": "PNDM GOES exporter is theory-covered for deterministic velocity-based Euler/Heun ODE steps only.",
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
        "edge_table": edge_table.metadata,
        "total_seconds": time.time() - started,
        "skipped_baselines": [
            {
                "name": "AYS",
                "reason": "This exporter only materializes the GOES schedule; baseline evaluation is handled by run_experiment_config.",
            }
        ],
    }
    dump_json(run_metadata, output_dir / "run_metadata.json")
    print(json.dumps({"output_dir": str(resolve_repo_path(output_dir)), "oracle_cache_key": oracle_result.cache_key}, indent=2))


if __name__ == "__main__":
    main()

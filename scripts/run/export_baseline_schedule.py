#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters.pndm import (
    _interp_timesteps_for_sigmas,
    build_scheduler,
    load_native_config,
    preferred_schedule_representation,
)
from src.clock.baseline import BASELINE_SCHEDULE_IMPLEMENTATION_VERSION
from src.utils.assets import AssetManifest
from src.utils.config import load_yaml
from src.utils.nfe_budget import resolve_effective_nfe_plan
from src.utils.schedule_bundle import ScheduleBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export base or linear schedules into the unified schedule bundle format.")
    parser.add_argument("--backend", choices=["pndm", "diffusers"], required=True)
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--mode", choices=["base", "linear"], required=True)
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--solver", default="euler")
    parser.add_argument("--dataset-config")
    parser.add_argument("--model-asset")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def collapse_repeated_values(values: np.ndarray, expected_length: int) -> np.ndarray:
    collapsed: list[float] = []
    for item in np.asarray(values, dtype=np.float64).tolist():
        if not collapsed or not np.isclose(collapsed[-1], item):
            collapsed.append(float(item))
    result = np.asarray(collapsed, dtype=np.float64)
    if len(result) != expected_length:
        raise RuntimeError(
            f"Expected {expected_length} unique schedule values after collapsing repeats, got {len(result)}."
        )
    return result


def export_pndm(args: argparse.Namespace) -> None:
    plan = resolve_effective_nfe_plan(args.solver, args.nfe)
    dataset_config = load_yaml(args.dataset_config)
    native = load_native_config(dataset_config["native_config"])
    schedule_cfg = native["Schedule"]
    scheduler = build_scheduler(
        args.solver,
        diffusion_step=schedule_cfg["diffusion_step"],
        beta_start=schedule_cfg["beta_start"],
        beta_end=schedule_cfg["beta_end"],
        beta_schedule=schedule_cfg["type"],
    )
    scheduler.set_timesteps(plan.solver_steps)
    representation = preferred_schedule_representation(args.solver)
    if representation == "sigmas":
        raw_sigmas = getattr(scheduler, "sigmas", None)
        if raw_sigmas is None:
            raise RuntimeError("The selected PNDM solver does not expose sigma schedules.")
        sigma_values = raw_sigmas.detach().cpu().float().numpy() if hasattr(raw_sigmas, "detach") else np.asarray(raw_sigmas)
        base_sigmas = collapse_repeated_values(np.asarray(sigma_values[:-1], dtype=np.float64), expected_length=plan.solver_steps)
        if args.mode == "base":
            sigmas = base_sigmas
        else:
            sigmas = np.linspace(base_sigmas[0], base_sigmas[-1], plan.solver_steps, dtype=np.float64)
        sigma_grid = np.concatenate([sigmas, np.asarray([0.0], dtype=np.float64)])
        time_grid = _interp_timesteps_for_sigmas(scheduler, sigma_grid)
        bundle = ScheduleBundle(
            sigmas=sigmas,
            sigma_grid=sigma_grid,
            timesteps=time_grid[:-1],
            time_grid=time_grid,
            meta={
                "schedule_family": args.mode,
                "backend": "pndm",
                "dataset": dataset_config["name"],
                "solver": args.solver,
                "coordinate_domain": "sigmas",
                "representation": "sigmas",
                "terminal_sigma": float(sigma_grid[-1]),
                "schedule_implementation_version": BASELINE_SCHEDULE_IMPLEMENTATION_VERSION,
                **plan.to_meta(),
            },
        )
    else:
        base_timesteps = collapse_repeated_values(
            scheduler.timesteps.detach().cpu().float().numpy(),
            expected_length=plan.solver_steps,
        )
        if args.mode == "base":
            timesteps = base_timesteps
        else:
            timesteps = np.linspace(base_timesteps[0], base_timesteps[-1], plan.solver_steps, dtype=np.float64)
        bundle = ScheduleBundle(
            timesteps=timesteps,
            time_grid=np.concatenate([timesteps, np.asarray([0.0], dtype=np.float64)]),
            meta={
                "schedule_family": args.mode,
                "backend": "pndm",
                "dataset": dataset_config["name"],
                "solver": args.solver,
                "coordinate_domain": "timesteps",
                "representation": "timesteps",
                "terminal_timestep": 0.0,
                "schedule_implementation_version": BASELINE_SCHEDULE_IMPLEMENTATION_VERSION,
                **plan.to_meta(),
            },
        )
    bundle.save(args.output_dir)


def export_diffusers(args: argparse.Namespace) -> None:
    from src.adapters.diffusers import (
        get_pipeline_device,
        load_pipeline,
        replace_scheduler,
    )

    manifest = AssetManifest(args.manifest)
    model_path = manifest.path(args.model_asset)
    pipeline = load_pipeline(model_path, device="cpu", dtype_name=args.dtype)
    replace_scheduler(pipeline, args.solver)
    plan = resolve_effective_nfe_plan(args.solver, args.nfe)
    pipeline.scheduler.set_timesteps(plan.solver_steps, device=get_pipeline_device(pipeline))
    raw_sigmas = getattr(pipeline.scheduler, "sigmas", None)
    if raw_sigmas is None:
        raise RuntimeError("The selected modern diffusers solver does not expose sigma schedules.")
    sigma_values = raw_sigmas.detach().cpu().float().numpy() if hasattr(raw_sigmas, "detach") else np.asarray(raw_sigmas)
    base_sigmas = collapse_repeated_values(np.asarray(sigma_values[:-1], dtype=np.float64), expected_length=plan.solver_steps)
    if args.mode == "base":
        sigma_values = base_sigmas
    else:
        sigma_values = np.linspace(base_sigmas[0], base_sigmas[-1], plan.solver_steps, dtype=np.float64)
    bundle = ScheduleBundle(
        sigmas=sigma_values,
        sigma_grid=np.concatenate([sigma_values, np.asarray([0.0], dtype=np.float64)]),
        meta={
            "schedule_family": args.mode,
            "backend": "diffusers",
            "solver": args.solver,
            "model_asset": args.model_asset,
            "schedule_implementation_version": BASELINE_SCHEDULE_IMPLEMENTATION_VERSION,
            **plan.to_meta(),
        },
    )
    bundle.save(args.output_dir)


def main() -> None:
    args = parse_args()
    if args.backend == "pndm":
        if not args.dataset_config:
            raise ValueError("--dataset-config is required for backend=pndm")
        export_pndm(args)
    else:
        if not args.model_asset:
            raise ValueError("--model-asset is required for backend=diffusers")
        export_diffusers(args)


if __name__ == "__main__":
    main()

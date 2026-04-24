#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters.pndm import (
    _interp_sigmas_for_timesteps,
    _interp_timesteps_for_sigmas,
    build_pndm_native_coordinate_grid,
    build_pndm_sigma_grid,
    build_scheduler,
    collect_solver_refinement_stats,
    load_model,
    load_native_config,
    preferred_calibration_domain,
    preferred_schedule_representation,
)
from src.clock.defect_balanced import (
    DEFECT_BALANCED_CLOCK_VERSION,
    DefectBalancedProfileArtifacts,
    StepRefinementStats,
    build_defect_balanced_profile,
    build_velocity_stepper,
    collect_step_refinement_stats,
)
from src.clock.profile import ClockProfile, build_reparameterized_bundle, export_clock_sweep, slice_profile_interval
from src.utils.assets import AssetManifest
from src.utils.config import dump_json, ensure_dir, load_json, load_yaml, resolve_repo_path


SCHEDULE_FAMILY = "SADB"
ESTIMATOR_NAME = "step_refinement"
PROFILE_ARRAY_FILES = (
    "physical_grid.npy",
    "alpha_profile.npy",
    "density.npy",
    "tau_profile.npy",
    "defect_profile.npy",
    "smoothed_defect_profile.npy",
    "effective_order_profile.npy",
    "smoothed_effective_order_profile.npy",
    "interval_alpha_profile.npy",
    "meta.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export solver-aware defect-balanced schedule bundles.")
    parser.add_argument("--backend", choices=["pndm", "diffusers"], required=True)
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--clock-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-nfes", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--dataset-config")
    parser.add_argument("--model-asset")
    parser.add_argument("--prompt-asset", default="diffusers_smoke_prompts")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    return parser.parse_args()


def parse_target_nfes(raw: str, fallback: list[int]) -> list[int]:
    if not raw:
        return [int(item) for item in fallback]
    return [int(item) for item in raw.split(",") if item]


def load_clock_settings(path: str) -> dict[str, Any]:
    payload = load_yaml(path)
    clock = payload.get("clock", {})
    if not isinstance(clock, dict):
        raise TypeError("clock config must contain a `clock` mapping.")
    if str(clock.get("family", "")).upper() != SCHEDULE_FAMILY:
        raise ValueError(f"Defect-clock exporter expects `clock.family: {SCHEDULE_FAMILY}`.")
    clock["estimator"] = ESTIMATOR_NAME
    model_output_type = str(clock.get("model_output_type", "epsilon")).lower()
    if model_output_type not in {"epsilon", "v_prediction", "flow"}:
        raise ValueError("SADB expects clock.model_output_type to be one of: epsilon, v_prediction, flow.")
    clock["model_output_type"] = model_output_type
    clock["q_min"] = float(clock.get("q_min", 1.05))
    clock["q_max"] = float(clock.get("q_max", 6.0))
    return clock


def normalize_diffusers_solver(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if normalized == "heun2":
        return "flow_heun"
    if normalized == "euler":
        return "flow_euler"
    return normalized


def schedule_family_label() -> str:
    return SCHEDULE_FAMILY


def build_pndm_physical_grid(
    *,
    scheduler,
    coordinate_domain: str,
    diffusion_step: int,
    physical_grid_size: int,
) -> np.ndarray:
    if coordinate_domain == "timesteps":
        return np.linspace(
            float(diffusion_step - 1),
            0.0,
            physical_grid_size,
            dtype=np.float64,
        )
    if coordinate_domain == "sigmas":
        return build_pndm_sigma_grid(scheduler, physical_grid_size=physical_grid_size)
    raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")


def build_pndm_export_transforms(
    *,
    scheduler,
    coordinate_domain: str,
) -> tuple[object, object]:
    if coordinate_domain == "timesteps":
        return (
            lambda values: _interp_sigmas_for_timesteps(scheduler, values),
            lambda values: np.asarray(values, dtype=np.float64),
        )
    if coordinate_domain == "sigmas":
        return (
            lambda values: np.asarray(values, dtype=np.float64),
            lambda values: _interp_timesteps_for_sigmas(scheduler, values),
        )
    raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")


def resolve_calibration_solver(clock_config: dict[str, Any], target_solver: str) -> str:
    configured = clock_config.get("calibration_solver", clock_config.get("pilot_solver", "target"))
    if str(configured).lower() in {"", "target"}:
        return target_solver
    return str(configured)


def profile_cache_dir(
    *,
    cache_root: Path,
    backend: str,
    dataset_name: str | None,
    model_asset: str,
    solver: str,
    calibration_solver: str,
    physical_grid_size: int,
    pilot_batch_size: int,
    pilot_num_batches: int,
    pilot_observation_microbatch: int,
    smoothing_window: int,
    epsilon: float,
    q_min: float,
    q_max: float,
    seed: int,
    prompt_tag: str | None = None,
    height: int | None = None,
    width: int | None = None,
    guidance_scale: float | None = None,
    model_output_type: str | None = None,
    coordinate_domain: str | None = None,
) -> Path:
    parts = [backend, SCHEDULE_FAMILY, ESTIMATOR_NAME]
    if dataset_name:
        parts.append(dataset_name)
    parts.extend(
        [
            model_asset,
            f"solver_{solver}",
            f"calibration_{calibration_solver}",
            f"grid_{physical_grid_size}",
            f"batch_{pilot_batch_size}",
            f"batches_{pilot_num_batches}",
            f"obs_{pilot_observation_microbatch}",
            f"smooth_{smoothing_window}",
            f"eps_{epsilon:g}",
            f"q_{q_min:g}_{q_max:g}",
            f"seed_{seed}",
        ]
    )
    if prompt_tag:
        parts.append(f"prompt_{prompt_tag}")
    if height is not None and width is not None:
        parts.append(f"size_{height}x{width}")
    if guidance_scale is not None:
        parts.append(f"cfg_{guidance_scale:g}")
    if model_output_type:
        parts.append(f"model_output_{model_output_type}")
    if coordinate_domain:
        parts.append(f"domain_{coordinate_domain}")
    return cache_root.joinpath(*parts)


def save_profile(output_dir: Path, artifacts: DefectBalancedProfileArtifacts, meta: dict[str, Any]) -> None:
    ensure_dir(output_dir)
    np.save(output_dir / "physical_grid.npy", artifacts.profile.physical_grid)
    np.save(output_dir / "alpha_profile.npy", artifacts.profile.alpha_profile)
    np.save(output_dir / "density.npy", artifacts.profile.density)
    np.save(output_dir / "tau_profile.npy", artifacts.profile.tau_profile)
    np.save(output_dir / "defect_profile.npy", artifacts.defect_profile)
    np.save(output_dir / "smoothed_defect_profile.npy", artifacts.smoothed_defect_profile)
    np.save(output_dir / "effective_order_profile.npy", artifacts.effective_order_profile)
    np.save(output_dir / "smoothed_effective_order_profile.npy", artifacts.smoothed_effective_order_profile)
    np.save(output_dir / "interval_alpha_profile.npy", artifacts.interval_alpha_profile)
    dump_json(meta, output_dir / "meta.json")


def load_profile(input_dir: Path) -> ClockProfile:
    return ClockProfile(
        physical_grid=np.load(input_dir / "physical_grid.npy"),
        alpha_profile=np.load(input_dir / "alpha_profile.npy"),
        density=np.load(input_dir / "density.npy"),
        tau_profile=np.load(input_dir / "tau_profile.npy"),
    )


def profile_artifacts_exist(input_dir: Path) -> bool:
    return all((input_dir / name).exists() for name in PROFILE_ARRAY_FILES)


def semantic_meta_matches(cached_meta: dict[str, Any], expected_meta: dict[str, Any]) -> bool:
    for key, expected_value in expected_meta.items():
        if key not in cached_meta:
            return False
        if cached_meta[key] != expected_value:
            return False
    return True


def load_cached_profile_if_current(input_dir: Path, expected_meta: dict[str, Any]) -> ClockProfile | None:
    if not profile_artifacts_exist(input_dir):
        return None
    cached_meta = load_json(input_dir / "meta.json")
    if not semantic_meta_matches(cached_meta, expected_meta):
        return None
    return load_profile(input_dir)


def _build_profile_meta(
    *,
    backend: str,
    model_asset: str,
    solver: str,
    calibration_solver: str,
    physical_grid_size: int,
    pilot_batch_size: int,
    pilot_num_batches: int,
    pilot_observation_microbatch: int,
    epsilon: float,
    smoothing_window: int,
    q_min: float,
    q_max: float,
    model_output_type: str,
    coordinate_domain: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "backend": backend,
        "model_asset": model_asset,
        "schedule_family": SCHEDULE_FAMILY,
        "schedule_implementation_version": DEFECT_BALANCED_CLOCK_VERSION,
        "estimator": ESTIMATOR_NAME,
        "solver": solver,
        "calibration_solver": calibration_solver,
        "physical_grid_size": physical_grid_size,
        "pilot_batch_size": pilot_batch_size,
        "pilot_num_batches": pilot_num_batches,
        "pilot_observation_microbatch": pilot_observation_microbatch,
        "epsilon": epsilon,
        "smoothing_window": smoothing_window,
        "q_min": q_min,
        "q_max": q_max,
        "model_output_type": model_output_type,
        "coordinate_domain": coordinate_domain,
        "calibration_method": "solver_step_refinement_full_half_quarter",
    }
    if extra:
        meta.update(extra)
    return meta


def build_or_load_pndm_profile(
    *,
    manifest: AssetManifest,
    args: argparse.Namespace,
    clock_config: dict[str, Any],
) -> tuple[ClockProfile, Path, dict[str, Any]]:
    dataset_config = load_yaml(args.dataset_config)
    model_asset = args.model_asset or dataset_config["default_model_asset"]
    native_config = load_native_config(dataset_config["native_config"])
    model, _ = load_model(dataset_config["native_config"], manifest.path(model_asset), device="cuda")
    schedule_cfg = native_config["Schedule"]
    target_solver = str(args.solver or "euler")
    calibration_solver = resolve_calibration_solver(clock_config, target_solver)
    coordinate_domain = preferred_calibration_domain(calibration_solver)
    physical_grid_size = int(clock_config.get("physical_grid_size", 65))
    smoothing_window = int(clock_config.get("smoothing_window", 1))
    epsilon = float(clock_config.get("epsilon", 1.0e-12))
    q_min = float(clock_config["q_min"])
    q_max = float(clock_config["q_max"])
    pilot_batch_size = int(clock_config.get("pilot_batch_size", 8))
    pilot_num_batches = int(clock_config.get("pilot_num_batches", 4))
    pilot_observation_microbatch = int(clock_config.get("pilot_observation_microbatch", 4))
    cache_root = resolve_repo_path(clock_config.get("cache_path", "outputs/cache/sadb_profiles"))
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        backend="pndm",
        dataset_name=str(dataset_config["name"]),
        model_asset=str(model_asset),
        solver=target_solver,
        calibration_solver=calibration_solver,
        physical_grid_size=physical_grid_size,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        pilot_observation_microbatch=pilot_observation_microbatch,
        smoothing_window=smoothing_window,
        epsilon=epsilon,
        q_min=q_min,
        q_max=q_max,
        seed=args.seed,
        model_output_type=str(clock_config["model_output_type"]),
        coordinate_domain=coordinate_domain,
    )
    profile_meta = _build_profile_meta(
        backend="pndm",
        model_asset=str(model_asset),
        solver=target_solver,
        calibration_solver=calibration_solver,
        physical_grid_size=physical_grid_size,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        pilot_observation_microbatch=pilot_observation_microbatch,
        epsilon=epsilon,
        smoothing_window=smoothing_window,
        q_min=q_min,
        q_max=q_max,
        model_output_type=str(clock_config["model_output_type"]),
        coordinate_domain=coordinate_domain,
        extra={
            "dataset": dataset_config["name"],
            "pilot_data_source": "synthetic_noise_trajectories_only",
            "uses_dataset_samples": False,
        },
    )
    cached_profile = load_cached_profile_if_current(cache_dir, profile_meta)
    if cached_profile is not None:
        return cached_profile, cache_dir, profile_meta

    scheduler = build_scheduler(
        calibration_solver,
        diffusion_step=schedule_cfg["diffusion_step"],
        beta_start=schedule_cfg["beta_start"],
        beta_end=schedule_cfg["beta_end"],
        beta_schedule=schedule_cfg["type"],
    )
    physical_grid = build_pndm_physical_grid(
        scheduler=scheduler,
        coordinate_domain=coordinate_domain,
        diffusion_step=int(schedule_cfg["diffusion_step"]),
        physical_grid_size=physical_grid_size,
    )
    stats = collect_solver_refinement_stats(
        model=model,
        scheduler=scheduler,
        physical_grid=physical_grid,
        solver=calibration_solver,
        image_size=int(dataset_config["image_size"]),
        batch_size=pilot_batch_size,
        num_batches=pilot_num_batches,
        seed=args.seed,
        observation_microbatch=pilot_observation_microbatch,
        model_output_type=str(clock_config["model_output_type"]),
        sigma_floor=epsilon,
        coordinate_domain=coordinate_domain,
        q_min=q_min,
        q_max=q_max,
        eps=epsilon,
    )

    artifacts = build_defect_balanced_profile(
        physical_grid,
        stats,
        smoothing_window=smoothing_window,
        eps=epsilon,
    )
    save_profile(cache_dir, artifacts, profile_meta)
    return artifacts.profile, cache_dir, profile_meta


def build_or_load_diffusers_profile(
    *,
    manifest: AssetManifest,
    args: argparse.Namespace,
    clock_config: dict[str, Any],
) -> tuple[ClockProfile, Path, dict[str, Any]]:
    from src.adapters.diffusers import (
        build_defect_sigma_grid,
        get_pipeline_device,
        load_pipeline,
        prepare_defect_batch,
        replace_scheduler,
    )

    model_asset = str(args.model_asset)
    target_solver = normalize_diffusers_solver(str(args.solver or "flow_euler"))
    calibration_solver = normalize_diffusers_solver(resolve_calibration_solver(clock_config, target_solver))
    effective_model_output_type = "flow"
    physical_grid_size = int(clock_config.get("physical_grid_size", 65))
    smoothing_window = int(clock_config.get("smoothing_window", 1))
    epsilon = float(clock_config.get("epsilon", 1.0e-12))
    q_min = float(clock_config["q_min"])
    q_max = float(clock_config["q_max"])
    pilot_batch_size = int(clock_config.get("pilot_batch_size", 8))
    pilot_num_batches = int(clock_config.get("pilot_num_batches", 4))
    pilot_observation_microbatch = int(clock_config.get("pilot_observation_microbatch", 4))
    pilot_prompt_asset = str(clock_config.get("pilot_prompt_asset", args.prompt_asset))
    cache_root = resolve_repo_path(clock_config.get("cache_path", "outputs/cache/sadb_profiles"))
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        backend="diffusers",
        dataset_name=None,
        model_asset=model_asset,
        solver=target_solver,
        calibration_solver=calibration_solver,
        physical_grid_size=physical_grid_size,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        pilot_observation_microbatch=pilot_observation_microbatch,
        smoothing_window=smoothing_window,
        epsilon=epsilon,
        q_min=q_min,
        q_max=q_max,
        seed=args.seed,
        prompt_tag=Path(pilot_prompt_asset).stem,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        model_output_type=effective_model_output_type,
        coordinate_domain="sigmas",
    )
    profile_meta = _build_profile_meta(
        backend="diffusers",
        model_asset=model_asset,
        solver=target_solver,
        calibration_solver=calibration_solver,
        physical_grid_size=physical_grid_size,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        pilot_observation_microbatch=pilot_observation_microbatch,
        epsilon=epsilon,
        smoothing_window=smoothing_window,
        q_min=q_min,
        q_max=q_max,
        model_output_type=effective_model_output_type,
        coordinate_domain="sigmas",
        extra={
            "pilot_prompt_asset": pilot_prompt_asset,
            "uses_evaluation_prompts": False,
        },
    )
    cached_profile = load_cached_profile_if_current(cache_dir, profile_meta)
    if cached_profile is not None:
        return cached_profile, cache_dir, profile_meta

    prompts_path = manifest.path(pilot_prompt_asset) if manifest.has(pilot_prompt_asset) else pilot_prompt_asset
    prompts = load_json(prompts_path)
    prompt_text = str(prompts[0])
    prompt_batch = [prompt_text] * pilot_batch_size
    pipeline = load_pipeline(manifest.path(model_asset), device="cuda", dtype_name=args.dtype)
    replace_scheduler(pipeline, calibration_solver)
    physical_grid = build_defect_sigma_grid(
        pipeline,
        physical_grid_size=physical_grid_size,
        height=args.height,
        width=args.width,
    )
    stats_batches = []
    with torch.inference_mode():
        for batch_index in range(pilot_num_batches):
            batch = prepare_defect_batch(
                pipeline,
                prompt=prompt_batch,
                batch_size=pilot_batch_size,
                seed=args.seed + batch_index,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
            )
            step_fn = build_velocity_stepper(batch.velocity_fn, calibration_solver)
            stats_batches.append(
                collect_step_refinement_stats(
                    initial_sample=batch.initial_latents,
                    physical_grid=physical_grid,
                    step_fn=step_fn,
                    observation_microbatch=pilot_observation_microbatch,
                    q_min=q_min,
                    q_max=q_max,
                    eps=epsilon,
                )
            )
            device = get_pipeline_device(pipeline)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    stats = StepRefinementStats(
        full_step_error=np.concatenate([item.full_step_error for item in stats_batches], axis=0),
        half_step_error=np.concatenate([item.half_step_error for item in stats_batches], axis=0),
        effective_order=np.concatenate([item.effective_order for item in stats_batches], axis=0),
        defect_strength=np.concatenate([item.defect_strength for item in stats_batches], axis=0),
    )
    artifacts = build_defect_balanced_profile(
        physical_grid,
        stats,
        smoothing_window=smoothing_window,
        eps=epsilon,
    )
    save_profile(cache_dir, artifacts, profile_meta)
    return artifacts.profile, cache_dir, profile_meta


def export_pndm(args: argparse.Namespace) -> None:
    manifest = AssetManifest(args.manifest)
    clock_config = load_clock_settings(args.clock_config)
    dataset_config = load_yaml(args.dataset_config)
    model_asset = args.model_asset or dataset_config["default_model_asset"]
    target_nfes = parse_target_nfes(args.target_nfes, list(clock_config.get("target_nfes", [])))
    target_solver = str(args.solver or "euler")
    calibration_solver = resolve_calibration_solver(clock_config, target_solver)
    representation = preferred_schedule_representation(target_solver)
    coordinate_domain = preferred_calibration_domain(calibration_solver)
    profile, cache_dir, profile_meta = build_or_load_pndm_profile(
        manifest=manifest,
        args=args,
        clock_config=clock_config,
    )
    export_meta = {
        "backend": "pndm",
        "dataset": dataset_config["name"],
        "model_asset": model_asset,
        "solver": target_solver,
        "calibration_solver": calibration_solver,
        "coordinate_domain": coordinate_domain,
        "clock_config": args.clock_config,
        "clock_model_output_type": str(profile_meta.get("model_output_type", "flow")),
        "estimator": ESTIMATOR_NAME,
        "shared_profile_dir": str(cache_dir),
        "shared_profile_meta": profile_meta,
        "schedule_implementation_version": DEFECT_BALANCED_CLOCK_VERSION,
    }
    if representation == "sigmas":
        native_config = load_native_config(dataset_config["native_config"])
        schedule_cfg = native_config["Schedule"]
        target_scheduler = build_scheduler(
            target_solver,
            diffusion_step=schedule_cfg["diffusion_step"],
            beta_start=schedule_cfg["beta_start"],
            beta_end=schedule_cfg["beta_end"],
            beta_schedule=schedule_cfg["type"],
        )
        representation_transform, time_transform = build_pndm_export_transforms(
            scheduler=target_scheduler,
            coordinate_domain=coordinate_domain,
        )
        output_root = Path(args.output_root)
        for effective_nfe in target_nfes:
            active_profile = profile
            if coordinate_domain == "sigmas":
                native_coordinate_grid = build_pndm_native_coordinate_grid(
                    target_scheduler,
                    solver_name=target_solver,
                    effective_nfe=int(effective_nfe),
                    coordinate_domain=coordinate_domain,
                )
                active_profile = slice_profile_interval(
                    profile,
                    float(native_coordinate_grid[0]),
                    float(native_coordinate_grid[-1]),
                )
            bundle = build_reparameterized_bundle(
                active_profile,
                effective_nfe=int(effective_nfe),
                solver_name=target_solver,
                representation=representation,
                schedule_family=SCHEDULE_FAMILY,
                meta={
                    **export_meta,
                    "native_coordinate_start": float(active_profile.physical_grid[0]),
                    "native_coordinate_end": float(active_profile.physical_grid[-1]),
                },
                representation_transform=representation_transform,
                time_transform=time_transform,
            )
            bundle.save(output_root / f"nfe_{int(effective_nfe):03d}")
        return
    export_clock_sweep(
        profile,
        target_nfes,
        output_root=args.output_root,
        solver_name=target_solver,
        representation=representation,
        schedule_family=SCHEDULE_FAMILY,
        meta=export_meta,
    )


def export_diffusers(args: argparse.Namespace) -> None:
    manifest = AssetManifest(args.manifest)
    clock_config = load_clock_settings(args.clock_config)
    target_nfes = parse_target_nfes(args.target_nfes, list(clock_config.get("target_nfes", [])))
    profile, cache_dir, profile_meta = build_or_load_diffusers_profile(
        manifest=manifest,
        args=args,
        clock_config=clock_config,
    )
    export_clock_sweep(
        profile,
        target_nfes,
        output_root=args.output_root,
        solver_name=str(args.solver or "flow_euler"),
        representation="sigmas",
        schedule_family=SCHEDULE_FAMILY,
        meta={
            "backend": "diffusers",
            "model_asset": args.model_asset,
            "solver": args.solver or "flow_euler",
            "calibration_solver": profile_meta["calibration_solver"],
            "clock_config": args.clock_config,
            "clock_model_output_type": str(clock_config["model_output_type"]),
            "estimator": ESTIMATOR_NAME,
            "shared_profile_dir": str(cache_dir),
            "shared_profile_meta": profile_meta,
            "schedule_implementation_version": DEFECT_BALANCED_CLOCK_VERSION,
        },
    )


def main() -> None:
    args = parse_args()
    if args.backend == "pndm":
        if not args.dataset_config:
            raise ValueError("--dataset-config is required for backend=pndm")
        export_pndm(args)
        return
    if not args.model_asset:
        raise ValueError("--model-asset is required for backend=diffusers")
    export_diffusers(args)


if __name__ == "__main__":
    main()

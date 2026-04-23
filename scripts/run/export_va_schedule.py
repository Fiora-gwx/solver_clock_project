#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    build_scheduler,
    build_pndm_lambda_grid,
    build_pndm_sigma_grid,
    collect_shared_clock_velocity_norms,
    load_model,
    load_native_config,
    preferred_calibration_domain,
    preferred_schedule_representation,
)
from src.clock.lcs import collect_lcs_norms
from src.clock.va import (
    VA_SCHEDULE_IMPLEMENTATION_VERSION,
    SharedClockProfile,
    build_reparameterized_bundle,
    build_shared_clock_profile,
    export_shared_clock_sweep,
    slice_profile_interval,
)
from src.utils.assets import AssetManifest
from src.utils.config import dump_json, ensure_dir, load_json, load_yaml, resolve_repo_path


SHARED_CLOCK_CACHE_VERSION = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export shared-clock V_a schedule bundles.")
    parser.add_argument("--backend", choices=["pndm", "diffusers"], required=True)
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--clock-config", default="configs/clocks/V_a.yaml")
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
    proxy_type = str(clock.get("proxy_type", "material_derivative_norm")).lower()
    if proxy_type in {"dtv_norm", "acceleration_norm"}:
        proxy_type = "material_derivative_norm"
    if proxy_type != "material_derivative_norm":
        raise ValueError("Shared-clock V_a only supports proxy_type=material_derivative_norm.")
    clock["proxy_type"] = proxy_type
    model_output_type = str(clock.get("model_output_type", "epsilon")).lower()
    if model_output_type not in {"epsilon", "v_prediction", "flow"}:
        raise ValueError(
            "Shared-clock V_a expects clock.model_output_type to be one of: epsilon, v_prediction, flow."
        )
    clock["model_output_type"] = model_output_type
    return clock


def normalize_diffusers_pilot_solver(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if normalized == "heun2":
        return "flow_heun"
    if normalized == "euler":
        return "flow_euler"
    return normalized


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
    if coordinate_domain == "lambda":
        return build_pndm_lambda_grid(scheduler, physical_grid_size=physical_grid_size)
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
    if coordinate_domain == "lambda":
        return (
            lambda values: np.exp(-np.asarray(values, dtype=np.float64)),
            lambda values: _interp_timesteps_for_sigmas(
                scheduler,
                np.exp(-np.asarray(values, dtype=np.float64)),
                force_log_sigma=True,
            ),
        )
    raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")


def profile_cache_dir(
    *,
    cache_root: Path,
    backend: str,
    dataset_name: str | None,
    model_asset: str,
    pilot_solver: str,
    physical_grid_size: int,
    pilot_batch_size: int,
    pilot_num_batches: int,
    pilot_observation_microbatch: int,
    epsilon: float,
    seed: int,
    prompt_tag: str | None = None,
    height: int | None = None,
    width: int | None = None,
    guidance_scale: float | None = None,
    model_output_type: str | None = None,
    coordinate_domain: str | None = None,
) -> Path:
    parts = [backend]
    if dataset_name:
        parts.append(dataset_name)
    parts.extend(
        [
            model_asset,
            f"pilot_{pilot_solver}",
            f"grid_{physical_grid_size}",
            f"batch_{pilot_batch_size}",
            f"batches_{pilot_num_batches}",
            f"obs_{pilot_observation_microbatch}",
            f"eps_{epsilon:g}",
            f"seed_{seed}",
        ]
    )
    if prompt_tag:
        parts.append(f"prompt_{prompt_tag}")
    if height is not None and width is not None:
        parts.append(f"size_{height}x{width}")
    if guidance_scale is not None:
        parts.append(f"cfg_{guidance_scale}")
    if model_output_type:
        parts.append(f"model_output_{model_output_type}")
    if coordinate_domain:
        parts.append(f"domain_{coordinate_domain}")
    return cache_root.joinpath(*parts)


def save_profile(output_dir: Path, profile: SharedClockProfile, meta: dict[str, Any]) -> None:
    ensure_dir(output_dir)
    np.save(output_dir / "physical_grid.npy", profile.physical_grid)
    np.save(output_dir / "alpha_profile.npy", profile.alpha_profile)
    np.save(output_dir / "density.npy", profile.density)
    np.save(output_dir / "tau_profile.npy", profile.tau_profile)
    dump_json(meta, output_dir / "meta.json")


def load_profile(input_dir: Path) -> SharedClockProfile:
    return SharedClockProfile(
        physical_grid=np.load(input_dir / "physical_grid.npy"),
        alpha_profile=np.load(input_dir / "alpha_profile.npy"),
        density=np.load(input_dir / "density.npy"),
        tau_profile=np.load(input_dir / "tau_profile.npy"),
    )


def build_or_load_pndm_profile(
    *,
    manifest: AssetManifest,
    args: argparse.Namespace,
    clock_config: dict[str, Any],
) -> tuple[SharedClockProfile, Path, dict[str, Any]]:
    dataset_config = load_yaml(args.dataset_config)
    model_asset = args.model_asset or dataset_config["default_model_asset"]
    native_config = load_native_config(dataset_config["native_config"])
    model, _ = load_model(dataset_config["native_config"], manifest.path(model_asset), device="cuda")
    schedule_cfg = native_config["Schedule"]
    target_solver = str(args.solver or "euler")
    coordinate_domain = preferred_calibration_domain(target_solver)
    pilot_solver = str(clock_config.get("pilot_solver", "heun2"))
    physical_grid_size = int(clock_config.get("physical_grid_size", 65))
    pilot_batch_size = int(clock_config.get("pilot_batch_size", 8))
    pilot_num_batches = int(clock_config.get("pilot_num_batches", 4))
    pilot_observation_microbatch = int(clock_config.get("pilot_observation_microbatch", 4))
    epsilon = float(clock_config.get("epsilon", 1.0e-6))
    if physical_grid_size < 2:
        raise ValueError("physical_grid_size must be at least 2.")

    cache_root = resolve_repo_path(clock_config.get("cache_path", "outputs/cache/shared_clock_profiles"))
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        backend="pndm",
        dataset_name=str(dataset_config["name"]),
        model_asset=str(model_asset),
        pilot_solver=pilot_solver,
        physical_grid_size=physical_grid_size,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        pilot_observation_microbatch=pilot_observation_microbatch,
        epsilon=epsilon,
        seed=args.seed,
        model_output_type=str(clock_config.get("model_output_type", "epsilon")),
        coordinate_domain=coordinate_domain,
    )
    meta = {
        "backend": "pndm",
        "dataset": dataset_config["name"],
        "model_asset": model_asset,
        "solver": target_solver,
        "pilot_solver": pilot_solver,
        "physical_grid_size": physical_grid_size,
        "pilot_batch_size": pilot_batch_size,
        "pilot_num_batches": pilot_num_batches,
        "pilot_observation_microbatch": pilot_observation_microbatch,
        "epsilon": epsilon,
        "proxy_type": str(clock_config.get("proxy_type", "material_derivative_norm")),
        "model_output_type": str(clock_config.get("model_output_type", "epsilon")),
        "coordinate_domain": coordinate_domain,
        "estimator": "material_derivative",
        "pilot_data_source": "synthetic_noise_trajectories_only",
        "uses_dataset_samples": False,
        "cache_version": SHARED_CLOCK_CACHE_VERSION,
    }
    if (cache_dir / "meta.json").exists():
        cached_meta = load_json(cache_dir / "meta.json")
        if int(cached_meta.get("cache_version", -1)) == SHARED_CLOCK_CACHE_VERSION:
            return load_profile(cache_dir), cache_dir, meta

    scheduler = build_scheduler(
        pilot_solver,
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
    material_derivative_norms = collect_shared_clock_velocity_norms(
        model=model,
        scheduler=scheduler,
        physical_grid=physical_grid,
        pilot_solver=pilot_solver,
        image_size=int(dataset_config["image_size"]),
        batch_size=pilot_batch_size,
        num_batches=pilot_num_batches,
        seed=args.seed,
        observation_microbatch=pilot_observation_microbatch,
        model_output_type=str(clock_config.get("model_output_type", "epsilon")),
        sigma_floor=epsilon,
        coordinate_domain=coordinate_domain,
        quantity="material_derivative",
    )
    profile = build_shared_clock_profile(
        physical_grid=physical_grid,
        material_derivative_norms=material_derivative_norms,
        eps=epsilon,
    )
    save_profile(cache_dir, profile, meta)
    return profile, cache_dir, meta


def aggregate_diffusers_batch(records: list[CalibrationRecord], domain_values: np.ndarray) -> np.ndarray:
    if len(records) != len(domain_values):
        raise ValueError("Diffusers calibration record count must match its domain length.")
    sample_count = len(records[0].sample_norms)
    matrix = np.zeros((sample_count, len(domain_values)), dtype=np.float64)
    for index, record in enumerate(records):
        matrix[:, index] = np.asarray(record.sample_norms, dtype=np.float64)
    return matrix


def collapse_duplicate_columns(domain_values: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    collapsed_domain: list[float] = []
    collapsed_columns: list[np.ndarray] = []
    start = 0
    while start < len(domain_values):
        end = start + 1
        while end < len(domain_values) and np.isclose(domain_values[end], domain_values[start]):
            end += 1
        collapsed_domain.append(float(domain_values[start]))
        window = matrix[:, start:end]
        collapsed_columns.append(np.sqrt(np.mean(np.square(window), axis=1)))
        start = end
    return np.asarray(collapsed_domain, dtype=np.float64), np.stack(collapsed_columns, axis=1)


def build_or_load_diffusers_profile(
    *,
    manifest: AssetManifest,
    args: argparse.Namespace,
    clock_config: dict[str, Any],
) -> tuple[SharedClockProfile, Path, dict[str, Any]]:
    from src.adapters.diffusers import (
        build_lcs_sigma_grid,
        get_pipeline_device,
        load_pipeline,
        prepare_lcs_batch,
        replace_scheduler,
    )

    pilot_solver = normalize_diffusers_pilot_solver(str(clock_config.get("pilot_solver", "heun2")))
    effective_model_output_type = "flow"
    physical_grid_size = int(clock_config.get("physical_grid_size", 65))
    model_asset = str(args.model_asset)
    cache_root = resolve_repo_path(clock_config.get("cache_path", "outputs/cache/shared_clock_profiles"))
    pilot_batch_size = int(clock_config.get("pilot_batch_size", 8))
    pilot_num_batches = int(clock_config.get("pilot_num_batches", 4))
    pilot_observation_microbatch = int(clock_config.get("pilot_observation_microbatch", 4))
    epsilon = float(clock_config.get("epsilon", 1.0e-6))
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        backend="diffusers",
        dataset_name=None,
        model_asset=model_asset,
        pilot_solver=pilot_solver,
        physical_grid_size=physical_grid_size,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        pilot_observation_microbatch=pilot_observation_microbatch,
        epsilon=epsilon,
        seed=args.seed,
        prompt_tag=Path(str(clock_config.get("pilot_prompt_asset", "diffusers_smoke_prompts"))).stem,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        model_output_type=effective_model_output_type,
        coordinate_domain="sigmas",
    )
    meta = {
        "backend": "diffusers",
        "model_asset": model_asset,
        "pilot_solver": pilot_solver,
        "physical_grid_size": physical_grid_size,
        "pilot_batch_size": pilot_batch_size,
        "pilot_num_batches": pilot_num_batches,
        "pilot_observation_microbatch": pilot_observation_microbatch,
        "pilot_prompt_asset": str(clock_config.get("pilot_prompt_asset", "diffusers_smoke_prompts")),
        "uses_evaluation_prompts": False,
        "epsilon": epsilon,
        "proxy_type": str(clock_config.get("proxy_type", "material_derivative_norm")),
        "model_output_type": effective_model_output_type,
        "coordinate_domain": "sigmas",
        "estimator": "material_derivative",
        "cache_version": SHARED_CLOCK_CACHE_VERSION,
    }
    if (cache_dir / "meta.json").exists():
        cached_meta = load_json(cache_dir / "meta.json")
        if int(cached_meta.get("cache_version", -1)) == SHARED_CLOCK_CACHE_VERSION:
            return load_profile(cache_dir), cache_dir, meta

    pilot_prompt_asset = str(clock_config.get("pilot_prompt_asset", "diffusers_smoke_prompts"))
    prompts_path = manifest.path(pilot_prompt_asset) if manifest.has(pilot_prompt_asset) else pilot_prompt_asset
    prompts = load_json(prompts_path)
    pipeline = load_pipeline(manifest.path(model_asset), device="cuda", dtype_name=args.dtype)
    replace_scheduler(pipeline, pilot_solver)

    physical_grid = build_lcs_sigma_grid(
        pipeline,
        physical_grid_size=physical_grid_size,
        height=args.height,
        width=args.width,
    )
    batches: list[np.ndarray] = []
    batch_size = pilot_batch_size
    for batch_index in range(pilot_num_batches):
        prompt_batch = [str(prompts[0])] * batch_size
        batch = prepare_lcs_batch(
            pipeline,
            prompt=prompt_batch,
            batch_size=batch_size,
            seed=args.seed + batch_index,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
        )
        batches.append(
                collect_lcs_norms(
                    initial_sample=batch.initial_latents,
                    physical_grid=physical_grid,
                    velocity_fn=batch.velocity_fn,
                    pilot_solver=pilot_solver,
                    order=1,
                    observation_microbatch=pilot_observation_microbatch,
                )
            )
        device = get_pipeline_device(pipeline)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    material_derivative_norms = np.concatenate(batches, axis=0)
    profile = build_shared_clock_profile(
        physical_grid=physical_grid,
        material_derivative_norms=material_derivative_norms,
        eps=epsilon,
    )
    save_profile(cache_dir, profile, meta)
    return profile, cache_dir, meta


def export_pndm(args: argparse.Namespace) -> None:
    manifest = AssetManifest(args.manifest)
    clock_config = load_clock_settings(args.clock_config)
    dataset_config = load_yaml(args.dataset_config)
    model_asset = args.model_asset or dataset_config["default_model_asset"]
    target_nfes = parse_target_nfes(args.target_nfes, list(clock_config.get("target_nfes", [])))
    target_solver = str(args.solver or "euler")
    representation = preferred_schedule_representation(target_solver)
    coordinate_domain = preferred_calibration_domain(target_solver)
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
        "coordinate_domain": coordinate_domain,
        "clock_config": args.clock_config,
        "clock_model_output_type": str(clock_config.get("model_output_type", "epsilon")),
        "schedule_implementation_version": VA_SCHEDULE_IMPLEMENTATION_VERSION,
        "shared_profile_dir": str(cache_dir),
        "shared_profile_meta": profile_meta,
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
            if coordinate_domain in {"sigmas", "lambda"}:
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
                schedule_family="V_a",
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
    export_shared_clock_sweep(
        profile,
        target_nfes,
        output_root=args.output_root,
        solver_name=target_solver,
        representation=representation,
        schedule_family="V_a",
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
    export_shared_clock_sweep(
        profile,
        target_nfes,
        output_root=args.output_root,
        solver_name=str(args.solver or "flow_euler"),
        representation="sigmas",
        schedule_family="V_a",
        meta={
            "backend": "diffusers",
            "model_asset": args.model_asset,
            "solver": args.solver or "flow_euler",
            "clock_config": args.clock_config,
            "clock_model_output_type": str(profile_meta.get("model_output_type", "flow")),
            "schedule_implementation_version": VA_SCHEDULE_IMPLEMENTATION_VERSION,
            "shared_profile_dir": str(cache_dir),
            "shared_profile_meta": profile_meta,
        },
    )


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

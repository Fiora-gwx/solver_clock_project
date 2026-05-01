#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
    collect_velocity_curvature_calibration_stats,
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
DEFAULT_ESTIMATOR_NAME = "step_refinement"
VELOCITY_CURVATURE_ESTIMATOR_NAME = "velocity_curvature"
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
    estimator = str(clock.get("estimator", DEFAULT_ESTIMATOR_NAME)).lower().replace("-", "_")
    if estimator in {"curvature", "velocity_curvature", "velocity_curvature_q3"}:
        estimator = VELOCITY_CURVATURE_ESTIMATOR_NAME
    if estimator not in {DEFAULT_ESTIMATOR_NAME, VELOCITY_CURVATURE_ESTIMATOR_NAME}:
        raise ValueError(
            "SADB expects clock.estimator to be one of: step_refinement, velocity_curvature."
        )
    clock["estimator"] = estimator
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
    return normalized


def _diffusers_solver_uses_flow_prediction(name: str) -> bool:
    return normalize_diffusers_solver(name).startswith("flow_")


def _build_diffusers_sigma_to_timestep_transform(scheduler):
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError(f"Scheduler {scheduler.__class__.__name__} does not expose alphas_cumprod.")
    alphas = scheduler.alphas_cumprod.detach().float().cpu().numpy()
    train_sigmas = np.sqrt(np.clip(1.0 - alphas, 0.0, None) / np.clip(alphas, 1.0e-12, None))
    log_train_sigmas = np.log(np.clip(train_sigmas, 1.0e-10, None))
    train_timesteps = np.arange(len(train_sigmas), dtype=np.float64)

    def transform(sigmas: np.ndarray) -> np.ndarray:
        values = np.asarray(sigmas, dtype=np.float64)
        return np.interp(
            np.log(np.clip(values, 1.0e-10, None)),
            log_train_sigmas,
            train_timesteps,
        )

    return transform


def _build_diffusers_timestep_to_sigma_transform(scheduler):
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError(f"Scheduler {scheduler.__class__.__name__} does not expose alphas_cumprod.")
    alphas = scheduler.alphas_cumprod.detach().float().cpu().numpy()
    train_sigmas = np.sqrt(np.clip(1.0 - alphas, 0.0, None) / np.clip(alphas, 1.0e-12, None))
    train_timesteps = np.arange(len(train_sigmas), dtype=np.float64)

    def transform(timesteps: np.ndarray) -> np.ndarray:
        values = np.asarray(timesteps, dtype=np.float64)
        sigmas = np.interp(np.clip(values, 0.0, float(len(train_sigmas) - 1)), train_timesteps, train_sigmas)
        sigmas = np.asarray(sigmas, dtype=np.float64)
        sigmas[np.asarray(values) <= 0.0] = 0.0
        return sigmas

    return transform


def _diffusers_reference_time_grid(scheduler, *, effective_nfe: int, device: str = "cuda") -> np.ndarray:
    scheduler.set_timesteps(int(effective_nfe), device=device)
    timesteps = scheduler.timesteps.detach().float().cpu().numpy()
    return np.concatenate([timesteps, np.asarray([0.0], dtype=np.float64)])


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


def build_pndm_timestep_export_transform(*, scheduler, coordinate_domain: str):
    if coordinate_domain == "timesteps":
        return lambda values: np.asarray(values, dtype=np.float64)
    if coordinate_domain == "sigmas":
        return lambda values: _interp_timesteps_for_sigmas(scheduler, values)
    raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")


def _step_size_stats(nodes: np.ndarray, reference_nodes: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(nodes, dtype=np.float64)
    intervals = np.abs(np.diff(values))
    if intervals.size == 0:
        return {
            "max_dt": 0.0,
            "min_dt": 0.0,
            "max_neighbor_dt_ratio": 1.0,
            "max_dt_over_base_dt": 1.0,
        }
    if intervals.size > 1:
        neighbor = intervals[1:] / np.maximum(intervals[:-1], 1.0e-12)
        max_neighbor = float(np.max(neighbor))
    else:
        max_neighbor = 1.0
    if reference_nodes is not None:
        reference_intervals = np.abs(np.diff(np.asarray(reference_nodes, dtype=np.float64)))
        if reference_intervals.shape == intervals.shape:
            valid = reference_intervals > 1.0e-12
            max_over_base = float(np.max(intervals[valid] / reference_intervals[valid])) if np.any(valid) else 1.0
        else:
            max_over_base = 1.0
    else:
        max_over_base = 1.0
    return {
        "max_dt": float(np.max(intervals)),
        "min_dt": float(np.min(intervals)),
        "max_neighbor_dt_ratio": max_neighbor,
        "max_dt_over_base_dt": max_over_base,
    }


def _snap_descending_timesteps(values: np.ndarray, *, num_train_timesteps: int) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or len(raw) == 0:
        raise ValueError("Custom scheduler timesteps must be a non-empty 1D array.")
    max_timestep = int(num_train_timesteps) - 1
    if len(raw) > max_timestep:
        raise ValueError(f"Cannot snap {len(raw)} timesteps into a strictly descending training grid.")
    snapped = np.rint(raw).astype(np.int64)
    previous = max_timestep + 1
    for index in range(len(snapped)):
        lower = len(snapped) - index
        upper = previous - 1
        snapped[index] = int(np.clip(snapped[index], lower, upper))
        previous = int(snapped[index])
    if np.any(np.diff(snapped) >= 0):
        raise ValueError(f"Snapped timesteps must be strictly descending, got {snapped.tolist()}.")
    snap_error = np.abs(raw - snapped.astype(np.float64))
    return snapped.astype(np.float64), {
        "timestep_snap_enabled": True,
        "timestep_snap_max_abs_error": float(np.max(snap_error)),
        "timestep_snap_mean_abs_error": float(np.mean(snap_error)),
    }


def _limiter_reference_time_grid(reference_nodes: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_nodes, dtype=np.float64)
    if reference.ndim != 1 or len(reference) < 2:
        return reference.copy()
    intervals = np.abs(np.diff(reference))
    if np.any(intervals <= 1.0e-12):
        return np.linspace(float(reference[0]), float(reference[-1]), len(reference), dtype=np.float64)
    return reference.copy()


def limit_schedule_step_sizes(
    nodes: np.ndarray,
    reference_nodes: np.ndarray,
    *,
    max_dt_factor: float | None,
    max_neighbor_ratio: float | None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    values = np.asarray(nodes, dtype=np.float64)
    reference = np.asarray(reference_nodes, dtype=np.float64)
    if values.shape != reference.shape:
        raise ValueError("nodes and reference_nodes must have matching shapes for step-size limiting.")
    if values.ndim != 1 or len(values) < 2:
        return values.copy(), {"step_limiter_enabled": False}

    pre_stats = _step_size_stats(values, reference)
    if max_dt_factor is None and max_neighbor_ratio is None:
        return values.copy(), {
            "step_limiter_enabled": False,
            "pre_limit_max_dt": pre_stats["max_dt"],
            "pre_limit_min_dt": pre_stats["min_dt"],
            "pre_limit_max_neighbor_dt_ratio": pre_stats["max_neighbor_dt_ratio"],
            "pre_limit_max_dt_over_base_dt": pre_stats["max_dt_over_base_dt"],
            **pre_stats,
        }

    max_dt_factor = float(max_dt_factor) if max_dt_factor is not None else float("inf")
    max_neighbor_ratio = float(max_neighbor_ratio) if max_neighbor_ratio is not None else float("inf")
    if max_dt_factor <= 0.0:
        raise ValueError("max_dt_factor must be positive.")
    if max_neighbor_ratio <= 0.0:
        raise ValueError("max_neighbor_dt_ratio must be positive.")

    def satisfies(candidate: np.ndarray) -> bool:
        stats = _step_size_stats(candidate, reference)
        return (
            stats["max_dt_over_base_dt"] <= max_dt_factor + 1.0e-12
            and stats["max_neighbor_dt_ratio"] <= max_neighbor_ratio + 1.0e-12
        )

    if satisfies(values):
        limited = values.copy()
    else:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            candidate = (1.0 - mid) * values + mid * reference
            if satisfies(candidate):
                hi = mid
            else:
                lo = mid
        limited = (1.0 - hi) * values + hi * reference

    post_stats = _step_size_stats(limited, reference)
    return limited, {
        "step_limiter_enabled": True,
        "max_dt_factor": max_dt_factor,
        "max_neighbor_dt_ratio": max_neighbor_ratio,
        "pre_limit_max_dt": pre_stats["max_dt"],
        "pre_limit_min_dt": pre_stats["min_dt"],
        "pre_limit_max_neighbor_dt_ratio": pre_stats["max_neighbor_dt_ratio"],
        "pre_limit_max_dt_over_base_dt": pre_stats["max_dt_over_base_dt"],
        **post_stats,
    }


def _adaptive_s_schedule_meta(
    scheduler,
    time_grid: np.ndarray,
    *,
    enabled: bool,
    adaptive_s_max: int,
    adaptive_s_reference: str,
) -> dict[str, Any]:
    base_s = int(getattr(scheduler, "s", 50))
    values = np.asarray(time_grid, dtype=np.float64)
    intervals = np.abs(np.diff(values)) / float(scheduler.config.num_train_timesteps)
    base_dt = 1.0 / max(int(intervals.size), 1)
    requested_per_step: list[int] = []
    used_per_step: list[int] = []
    ms = None
    if hasattr(scheduler, "coeff_rock4") and hasattr(scheduler, "mdegr"):
        ms = scheduler.coeff_rock4()[0]
    for local_dt in intervals:
        if enabled:
            ratio = max(float(local_dt) / max(base_dt, 1.0e-12), 1.0)
            requested = min(max(math.ceil(base_s * math.sqrt(ratio)), base_s), int(adaptive_s_max))
        else:
            requested = base_s
        if ms is not None:
            used = int(scheduler.mdegr(requested, ms)[0])
        else:
            used = requested
        requested_per_step.append(int(requested))
        used_per_step.append(int(used))
    return {
        "adaptive_s_enabled": bool(enabled),
        "adaptive_s_base": base_s,
        "adaptive_s_max": int(adaptive_s_max),
        "adaptive_s_reference": str(adaptive_s_reference),
        "adaptive_s_base_dt": float(base_dt),
        "adaptive_s_requested_max": int(max(requested_per_step, default=base_s)),
        "adaptive_s_used_max": int(max(used_per_step, default=base_s)),
        "adaptive_s_requested_per_step": requested_per_step,
        "adaptive_s_used_per_step": used_per_step,
    }


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
    estimator: str = DEFAULT_ESTIMATOR_NAME,
    physical_grid_mode: str | None = None,
) -> Path:
    parts = [backend, SCHEDULE_FAMILY, estimator]
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
    if physical_grid_mode:
        parts.append(f"gridmode_{physical_grid_mode}")
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
    estimator: str = DEFAULT_ESTIMATOR_NAME,
    extra: dict[str, Any] | None = None,
    physical_grid_mode: str | None = None,
) -> dict[str, Any]:
    meta = {
        "backend": backend,
        "model_asset": model_asset,
        "schedule_family": SCHEDULE_FAMILY,
        "schedule_implementation_version": DEFECT_BALANCED_CLOCK_VERSION,
        "estimator": estimator,
        "solver": solver,
        "calibration_solver": calibration_solver,
        "physical_grid_size": physical_grid_size,
        "physical_grid_mode": physical_grid_mode,
        "pilot_batch_size": pilot_batch_size,
        "pilot_num_batches": pilot_num_batches,
        "pilot_observation_microbatch": pilot_observation_microbatch,
        "epsilon": epsilon,
        "smoothing_window": smoothing_window,
        "q_min": q_min,
        "q_max": q_max,
        "model_output_type": model_output_type,
        "coordinate_domain": coordinate_domain,
        "calibration_method": (
            "velocity_curvature_pilot_trajectory"
            if estimator == VELOCITY_CURVATURE_ESTIMATOR_NAME
            else "solver_step_refinement_full_half_quarter"
        ),
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
    estimator = str(clock_config["estimator"])
    coordinate_domain = preferred_calibration_domain(calibration_solver)
    physical_grid_size = int(clock_config.get("physical_grid_size", 65))
    smoothing_window = int(clock_config.get("smoothing_window", 1))
    epsilon = float(clock_config.get("epsilon", 1.0e-12))
    q_min = float(clock_config["q_min"])
    q_max = float(clock_config["q_max"])
    pilot_batch_size = int(clock_config.get("pilot_batch_size", 8))
    pilot_num_batches = int(clock_config.get("pilot_num_batches", 4))
    pilot_observation_microbatch = int(clock_config.get("pilot_observation_microbatch", 4))
    warmup_steps = int(clock_config.get("warmup_steps", 1))
    cache_root = resolve_repo_path(clock_config.get("cache_path", "outputs/cache/sadb_profiles"))
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        backend="pndm",
        dataset_name=str(dataset_config["name"]),
        model_asset=str(model_asset),
        solver=target_solver,
        calibration_solver=calibration_solver,
        estimator=estimator,
        physical_grid_size=physical_grid_size,
        physical_grid_mode=None,
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
    extra_meta = {
        "dataset": dataset_config["name"],
        "pilot_data_source": "synthetic_noise_trajectories_only",
        "uses_dataset_samples": False,
        "warmup_steps": warmup_steps,
    }
    if estimator == VELOCITY_CURVATURE_ESTIMATOR_NAME:
        extra_meta.update(
            {
                "velocity_curvature_q_const": float(clock_config.get("velocity_curvature_q_const", 3.0)),
                "velocity_curvature_pilot_solver": str(clock_config.get("velocity_curvature_pilot_solver", "heun2")),
                "velocity_curvature_pilot_pieces": int(clock_config.get("velocity_curvature_pilot_pieces", 4)),
                "velocity_curvature_defect_clip_quantile": clock_config.get(
                    "velocity_curvature_defect_clip_quantile", None
                ),
            }
        )
    profile_meta = _build_profile_meta(
        backend="pndm",
        model_asset=str(model_asset),
        solver=target_solver,
        calibration_solver=calibration_solver,
        estimator=estimator,
        physical_grid_size=physical_grid_size,
        physical_grid_mode=None,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        pilot_observation_microbatch=pilot_observation_microbatch,
        epsilon=epsilon,
        smoothing_window=smoothing_window,
        q_min=q_min,
        q_max=q_max,
        model_output_type=str(clock_config["model_output_type"]),
        coordinate_domain=coordinate_domain,
        extra=extra_meta,
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
    if estimator == VELOCITY_CURVATURE_ESTIMATOR_NAME:
        stats = collect_velocity_curvature_calibration_stats(
            model=model,
            scheduler=scheduler,
            physical_grid=physical_grid,
            image_size=int(dataset_config["image_size"]),
            batch_size=pilot_batch_size,
            num_batches=pilot_num_batches,
            seed=args.seed,
            observation_microbatch=pilot_observation_microbatch,
            model_output_type=str(clock_config["model_output_type"]),
            sigma_floor=epsilon,
            coordinate_domain=coordinate_domain,
            pilot_solver=str(clock_config.get("velocity_curvature_pilot_solver", "heun2")),
            pilot_pieces=int(clock_config.get("velocity_curvature_pilot_pieces", 4)),
            q_const=float(clock_config.get("velocity_curvature_q_const", 3.0)),
            eps=epsilon,
            defect_clip_quantile=clock_config.get("velocity_curvature_defect_clip_quantile", None),
        )
    else:
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
            warmup_steps=warmup_steps,
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
    estimator = str(clock_config["estimator"])
    if estimator == VELOCITY_CURVATURE_ESTIMATOR_NAME:
        raise ValueError("velocity_curvature calibration is currently implemented for backend=pndm only.")
    effective_model_output_type = "flow" if _diffusers_solver_uses_flow_prediction(target_solver) else str(clock_config["model_output_type"])
    physical_grid_size = int(clock_config.get("physical_grid_size", 65))
    physical_grid_mode = str(clock_config.get("physical_grid_mode", "scheduler_sigmas"))
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
        estimator=estimator,
        physical_grid_size=physical_grid_size,
        physical_grid_mode=physical_grid_mode,
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
        estimator=estimator,
        physical_grid_size=physical_grid_size,
        physical_grid_mode=physical_grid_mode,
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
            "guidance_scale": float(args.guidance_scale),
            "physical_grid_mode": physical_grid_mode,
        },
    )
    cached_profile = load_cached_profile_if_current(cache_dir, profile_meta)
    if cached_profile is not None:
        return cached_profile, cache_dir, profile_meta

    prompts_path = manifest.path(pilot_prompt_asset) if manifest.has(pilot_prompt_asset) else pilot_prompt_asset
    prompts = load_json(prompts_path)
    prompt_pool = [str(prompt) for prompt in prompts]
    pipeline = load_pipeline(manifest.path(model_asset), device="cuda", dtype_name=args.dtype)
    replace_scheduler(pipeline, calibration_solver)
    physical_grid = build_defect_sigma_grid(
        pipeline,
        physical_grid_size=physical_grid_size,
        height=args.height,
        width=args.width,
        physical_grid_mode=physical_grid_mode,
    )
    stats_batches = []
    with torch.inference_mode():
        for batch_index in range(pilot_num_batches):
            prompt_batch = [
                prompt_pool[(batch_index * pilot_batch_size + item_index) % len(prompt_pool)]
                for item_index in range(pilot_batch_size)
            ]
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
        "estimator": str(clock_config["estimator"]),
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
            max_dt_factor = clock_config.get("max_dt_factor")
            max_neighbor_ratio = clock_config.get("max_neighbor_dt_ratio")
            if max_dt_factor is not None or max_neighbor_ratio is not None:
                reference_time_grid = build_pndm_native_coordinate_grid(
                    target_scheduler,
                    solver_name=target_solver,
                    effective_nfe=int(effective_nfe),
                    coordinate_domain="timesteps",
                )
                limited_time_grid, limiter_meta = limit_schedule_step_sizes(
                    np.asarray(bundle.time_grid, dtype=np.float64),
                    _limiter_reference_time_grid(reference_time_grid),
                    max_dt_factor=None if max_dt_factor is None else float(max_dt_factor),
                    max_neighbor_ratio=None if max_neighbor_ratio is None else float(max_neighbor_ratio),
                )
                limited_sigma_grid = _interp_sigmas_for_timesteps(target_scheduler, limited_time_grid)
                limited_sigma_grid[-1] = 0.0
                bundle = type(bundle)(
                    timesteps=limited_time_grid[:-1].copy(),
                    time_grid=limited_time_grid,
                    sigmas=limited_sigma_grid[:-1].copy(),
                    sigma_grid=limited_sigma_grid,
                    tau_grid=bundle.tau_grid,
                    g_grid=bundle.g_grid,
                    meta={**bundle.meta, **limiter_meta},
                )
            else:
                schedule_stats = _step_size_stats(np.asarray(bundle.time_grid, dtype=np.float64))
                bundle.meta.update(
                    {
                        "step_limiter_enabled": False,
                        "max_dt": schedule_stats["max_dt"],
                        "min_dt": schedule_stats["min_dt"],
                        "max_neighbor_dt_ratio": schedule_stats["max_neighbor_dt_ratio"],
                        "max_dt_over_base_dt": schedule_stats["max_dt_over_base_dt"],
                    }
                )
            bundle.meta.update(
                _adaptive_s_schedule_meta(
                    target_scheduler,
                    np.asarray(bundle.time_grid, dtype=np.float64),
                    enabled=bool(clock_config.get("adaptive_s", False)),
                    adaptive_s_max=int(clock_config.get("adaptive_s_max", 100)),
                    adaptive_s_reference=str(clock_config.get("adaptive_s_reference", "base_dt")),
                )
            )
            bundle.save(output_root / f"nfe_{int(effective_nfe):03d}")
        return
    native_config = load_native_config(dataset_config["native_config"])
    schedule_cfg = native_config["Schedule"]
    target_scheduler = build_scheduler(
        target_solver,
        diffusion_step=schedule_cfg["diffusion_step"],
        beta_start=schedule_cfg["beta_start"],
        beta_end=schedule_cfg["beta_end"],
        beta_schedule=schedule_cfg["type"],
    )
    timestep_transform = build_pndm_timestep_export_transform(
        scheduler=target_scheduler,
        coordinate_domain=coordinate_domain,
    )
    output_root = Path(args.output_root)
    for effective_nfe in target_nfes:
        bundle = build_reparameterized_bundle(
            profile,
            effective_nfe=int(effective_nfe),
            solver_name=target_solver,
            representation=representation,
            schedule_family=SCHEDULE_FAMILY,
            meta=export_meta,
            representation_transform=timestep_transform,
            time_transform=timestep_transform,
        )
        max_dt_factor = clock_config.get("max_dt_factor")
        max_neighbor_ratio = clock_config.get("max_neighbor_dt_ratio")
        if max_dt_factor is not None or max_neighbor_ratio is not None:
            reference_time_grid = build_pndm_native_coordinate_grid(
                target_scheduler,
                solver_name=target_solver,
                effective_nfe=int(effective_nfe),
                coordinate_domain="timesteps",
            )
            limited_time_grid, limiter_meta = limit_schedule_step_sizes(
                np.asarray(bundle.time_grid, dtype=np.float64),
                _limiter_reference_time_grid(reference_time_grid),
                max_dt_factor=None if max_dt_factor is None else float(max_dt_factor),
                max_neighbor_ratio=None if max_neighbor_ratio is None else float(max_neighbor_ratio),
            )
            bundle = type(bundle)(
                timesteps=limited_time_grid[:-1].copy(),
                time_grid=limited_time_grid,
                tau_grid=bundle.tau_grid,
                g_grid=bundle.g_grid,
                meta={**bundle.meta, **limiter_meta},
            )
        bundle.save(output_root / f"nfe_{int(effective_nfe):03d}")


def export_diffusers(args: argparse.Namespace) -> None:
    manifest = AssetManifest(args.manifest)
    clock_config = load_clock_settings(args.clock_config)
    target_nfes = parse_target_nfes(args.target_nfes, list(clock_config.get("target_nfes", [])))
    profile, cache_dir, profile_meta = build_or_load_diffusers_profile(
        manifest=manifest,
        args=args,
        clock_config=clock_config,
    )
    time_transform = None
    target_scheduler = None
    if not _diffusers_solver_uses_flow_prediction(str(args.solver or "flow_euler")):
        from src.adapters.diffusers import load_pipeline, replace_scheduler

        pipeline = load_pipeline(manifest.path(args.model_asset), device="cuda", dtype_name=args.dtype)
        replace_scheduler(pipeline, str(args.solver or "flow_euler"))
        target_scheduler = pipeline.scheduler
        time_transform = _build_diffusers_sigma_to_timestep_transform(target_scheduler)
        timestep_to_sigma = _build_diffusers_timestep_to_sigma_transform(target_scheduler)
        exported = []
        for effective_nfe in target_nfes:
            bundle = build_reparameterized_bundle(
                profile,
                effective_nfe=int(effective_nfe),
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
                    "estimator": str(clock_config["estimator"]),
                    "physical_grid_mode": profile_meta.get("physical_grid_mode"),
                    "shared_profile_dir": str(cache_dir),
                    "shared_profile_meta": profile_meta,
                    "schedule_implementation_version": DEFECT_BALANCED_CLOCK_VERSION,
                },
                time_transform=time_transform,
            )
            max_dt_factor = clock_config.get("max_dt_factor")
            max_neighbor_ratio = clock_config.get("max_neighbor_dt_ratio")
            if max_dt_factor is not None or max_neighbor_ratio is not None:
                reference_time_grid = _diffusers_reference_time_grid(
                    target_scheduler,
                    effective_nfe=int(effective_nfe),
                    device="cuda",
                )
                limited_time_grid, limiter_meta = limit_schedule_step_sizes(
                    np.asarray(bundle.time_grid, dtype=np.float64),
                    _limiter_reference_time_grid(reference_time_grid),
                    max_dt_factor=None if max_dt_factor is None else float(max_dt_factor),
                    max_neighbor_ratio=None if max_neighbor_ratio is None else float(max_neighbor_ratio),
                )
                limited_sigma_grid = timestep_to_sigma(limited_time_grid)
                limited_sigma_grid[-1] = 0.0
                bundle = type(bundle)(
                    timesteps=limited_time_grid[:-1].copy(),
                    time_grid=limited_time_grid,
                    sigmas=limited_sigma_grid[:-1].copy(),
                    sigma_grid=limited_sigma_grid,
                    tau_grid=bundle.tau_grid,
                    g_grid=bundle.g_grid,
                    meta={**bundle.meta, **limiter_meta},
                )
            else:
                schedule_stats = _step_size_stats(np.asarray(bundle.time_grid, dtype=np.float64))
                bundle.meta.update(
                    {
                        "step_limiter_enabled": False,
                        "max_dt": schedule_stats["max_dt"],
                        "min_dt": schedule_stats["min_dt"],
                        "max_neighbor_dt_ratio": schedule_stats["max_neighbor_dt_ratio"],
                        "max_dt_over_base_dt": schedule_stats["max_dt_over_base_dt"],
                    }
                )
            if bundle.timesteps is not None:
                snapped_timesteps, snap_meta = _snap_descending_timesteps(
                    np.asarray(bundle.timesteps, dtype=np.float64),
                    num_train_timesteps=int(target_scheduler.config.num_train_timesteps),
                )
                snapped_time_grid = np.concatenate(
                    [snapped_timesteps, np.asarray([0.0], dtype=np.float64)]
                )
                snapped_sigma_grid = timestep_to_sigma(snapped_time_grid)
                snapped_sigma_grid[-1] = 0.0
                bundle = type(bundle)(
                    timesteps=snapped_timesteps,
                    time_grid=snapped_time_grid,
                    sigmas=snapped_sigma_grid[:-1].copy(),
                    sigma_grid=snapped_sigma_grid,
                    tau_grid=bundle.tau_grid,
                    g_grid=bundle.g_grid,
                    meta={**bundle.meta, **snap_meta},
                )
            output_dir = Path(args.output_root) / f"nfe_{int(effective_nfe):03d}"
            exported.append(bundle.save(output_dir))
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return
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
            "estimator": str(clock_config["estimator"]),
            "physical_grid_mode": profile_meta.get("physical_grid_mode"),
            "shared_profile_dir": str(cache_dir),
            "shared_profile_meta": profile_meta,
            "schedule_implementation_version": DEFECT_BALANCED_CLOCK_VERSION,
        },
        time_transform=time_transform,
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

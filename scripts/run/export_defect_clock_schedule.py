#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

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
    collect_anchored_replay_calibration_stats,
    collect_fp_clock_calibration_stats,
    collect_velocity_curvature_calibration_stats,
    collect_solver_refinement_stats,
    load_model,
    load_native_config,
    preferred_calibration_domain,
    preferred_schedule_representation,
)
from src.clock.fp_clock import (
    FP_CLOCK_VERSION,
    FPClockArtifacts,
    FPTrajectoryStats,
    build_fp_clock_profile,
    collect_fp_clock_stats,
    concatenate_fp_clock_stats,
)
from src.clock.solver_registry import get_solver_native_spec
from src.clock.defect_balanced import (
    DEFECT_BALANCED_CLOCK_VERSION,
    DefectBalancedProfileArtifacts,
    StepRefinementStats,
    build_defect_balanced_profile,
    build_velocity_stepper,
    collect_step_refinement_stats,
)
from src.clock.profile import ClockProfile, build_reparameterized_bundle, slice_profile_interval
from src.clock.transforms import (
    build_lambda_table,
    lambda_to_sigma,
    lambda_to_sigma_derivative,
    lambda_to_timestep,
    sigma_to_lambda,
)
from src.utils.assets import AssetManifest
from src.utils.config import dump_json, ensure_dir, load_json, load_yaml, resolve_repo_path
from src.utils.nfe_budget import resolve_effective_nfe_plan
from src.utils.schedule_bundle import ScheduleBundle


LEGACY_SADB_SCHEDULE_FAMILY = "LEGACY_SADB"
FP_CLOCK_SCHEDULE_FAMILY = "FP_CLOCK"
SCHEDULE_FAMILY = FP_CLOCK_SCHEDULE_FAMILY
DEFAULT_ESTIMATOR_NAME = "step_refinement"
VELOCITY_CURVATURE_ESTIMATOR_NAME = "velocity_curvature"
FP_CLOCK_ESTIMATOR_NAME = "fp_clock"
ANCHORED_REPLAY_ESTIMATOR_NAME = "anchored_replay"
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
    parser.add_argument(
        "--profile-cache-root",
        default="",
        help="Optional local profile cache root. Defaults to <output-root>/_profile_cache when clock.cache_path is unset.",
    )
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
    family = str(clock.get("family", SCHEDULE_FAMILY)).upper().replace("-", "_")
    if family not in {FP_CLOCK_SCHEDULE_FAMILY, LEGACY_SADB_SCHEDULE_FAMILY}:
        raise ValueError("Defect-clock exporter expects `clock.family` to be one of: FP_CLOCK, LEGACY_SADB.")
    clock["family"] = family
    if family == FP_CLOCK_SCHEDULE_FAMILY:
        estimator = str(clock.get("estimator", FP_CLOCK_ESTIMATOR_NAME)).lower().replace("-", "_")
        if estimator in {"window", "multires", "multiresolution", "target_solver", "trajectory_window"}:
            raise ValueError("trajectory_window FP_CLOCK has been retired; use anchored_replay.")
        if estimator not in {FP_CLOCK_ESTIMATOR_NAME, ANCHORED_REPLAY_ESTIMATOR_NAME}:
            raise ValueError("FP_CLOCK expects clock.estimator to be one of: fp_clock, anchored_replay.")
        clock["calibration_mode"] = "fp_clock"
    else:
        estimator = str(clock.get("estimator", DEFAULT_ESTIMATOR_NAME)).lower().replace("-", "_")
        if estimator in {"curvature", "velocity_curvature", "velocity_curvature_q3"}:
            estimator = VELOCITY_CURVATURE_ESTIMATOR_NAME
        if estimator not in {DEFAULT_ESTIMATOR_NAME, VELOCITY_CURVATURE_ESTIMATOR_NAME}:
            raise ValueError(
                "LEGACY_SADB expects clock.estimator to be one of: step_refinement, velocity_curvature."
            )
    clock["estimator"] = estimator
    model_output_type = str(clock.get("model_output_type", "epsilon")).lower()
    if model_output_type not in {"epsilon", "v_prediction", "flow"}:
        raise ValueError("clock.model_output_type must be one of: epsilon, v_prediction, flow.")
    clock["model_output_type"] = model_output_type
    clock["q_min"] = float(clock.get("q_min", 1.05))
    clock["q_max"] = float(clock.get("q_max", 6.0))
    coordinate_domain = str(clock.get("coordinate_domain", "lambda")).lower()
    if coordinate_domain not in {"lambda", "sigma", "sigmas", "timestep", "timesteps"}:
        raise ValueError("clock.coordinate_domain must be one of: lambda, sigma, timestep.")
    clock["coordinate_domain"] = {"sigmas": "sigma", "timesteps": "timestep"}.get(coordinate_domain, coordinate_domain)
    clock["prior_schedule"] = str(clock.get("prior_schedule", "none")).lower()
    clock["prior_blend"] = float(clock.get("prior_blend", 0.0))
    clock["density_temperature"] = float(clock.get("density_temperature", 1.0))
    clock["defect_reduce"] = str(clock.get("defect_reduce", "rms")).lower()
    clock["defect_quantile"] = float(clock.get("defect_quantile", 0.75))
    clock["q_shrinkage"] = float(clock.get("q_shrinkage", 0.0))
    if "q_prior" in clock:
        clock["q_prior"] = float(clock["q_prior"])
    clock["sde_noise_kappa"] = float(clock.get("sde_noise_kappa", 0.0))
    clock["sde_brownian_bridge"] = bool(clock.get("sde_brownian_bridge", False))
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


def _diffusers_reference_sigma_grid(scheduler, *, effective_nfe: int, device: str = "cuda") -> np.ndarray:
    scheduler.set_timesteps(int(effective_nfe), device=device)
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        raise RuntimeError(f"Scheduler {scheduler.__class__.__name__} does not expose sigmas.")
    values = sigmas.detach().float().cpu().numpy() if isinstance(sigmas, torch.Tensor) else np.asarray(sigmas, dtype=np.float64)
    if len(values) == int(effective_nfe):
        values = np.concatenate([values, np.asarray([0.0], dtype=np.float64)])
    if len(values) > int(effective_nfe) + 1:
        values = values[: int(effective_nfe) + 1]
    values = np.asarray(values, dtype=np.float64)
    values[-1] = 0.0
    return values


def schedule_family_label(clock_config: dict[str, Any] | None = None) -> str:
    if clock_config is None:
        return SCHEDULE_FAMILY
    return str(clock_config.get("family", SCHEDULE_FAMILY))


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
        if reference_intervals.shape == intervals.shape and reference_intervals.size > 0:
            base_dt = float(np.mean(reference_intervals))
            max_over_base = float(np.max(intervals) / max(base_dt, 1.0e-12))
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

    intervals = np.abs(np.diff(values))
    total = float(np.sum(intervals))
    if total <= 1.0e-12:
        limited = values.copy()
    else:
        base_dt = float(np.mean(np.abs(np.diff(reference))))
        max_dt = max_dt_factor * max(base_dt, 1.0e-12)
        limited_intervals = np.maximum(intervals, 1.0e-12)
        for _ in range(100):
            previous = limited_intervals.copy()
            limited_intervals = np.minimum(limited_intervals, max_dt)
            if np.isfinite(max_neighbor_ratio) and len(limited_intervals) > 1:
                for index in range(1, len(limited_intervals)):
                    limited_intervals[index] = min(
                        limited_intervals[index],
                        max_neighbor_ratio * max(limited_intervals[index - 1], 1.0e-12),
                    )
                for index in range(len(limited_intervals) - 2, -1, -1):
                    limited_intervals[index] = min(
                        limited_intervals[index],
                        max_neighbor_ratio * max(limited_intervals[index + 1], 1.0e-12),
                    )
            capped_total = float(np.sum(limited_intervals))
            if capped_total <= 1.0e-12:
                limited_intervals = np.full_like(limited_intervals, total / len(limited_intervals))
                break
            limited_intervals = limited_intervals * (total / capped_total)
            if np.max(np.abs(limited_intervals - previous)) <= 1.0e-10:
                break
        if np.max(limited_intervals) > max_dt + 1.0e-8:
            # If the requested limits are mutually infeasible after range preservation,
            # fall back to the largest feasible uniform range-preserving schedule.
            limited_intervals = np.full_like(limited_intervals, total / len(limited_intervals))
        direction = 1.0 if values[-1] >= values[0] else -1.0
        limited = values[0] + direction * np.concatenate([[0.0], np.cumsum(limited_intervals)])
        limited[-1] = values[-1]

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


def _clip_pndm_timestep_bundle_for_export(
    bundle: ScheduleBundle,
    *,
    diffusion_step: int,
    scheduler=None,
) -> ScheduleBundle:
    if bundle.time_grid is None:
        return bundle
    values = np.asarray(bundle.time_grid, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        return bundle

    max_timestep = float(int(diffusion_step) - 1)
    pre_min = float(np.min(values))
    pre_max = float(np.max(values))
    clipped = np.clip(values, 0.0, max_timestep)

    # Native DDIM/PNDM calibration may use a terminal coordinate below zero.
    # Exported scheduler timesteps must stay in the train-time range and remain
    # integer-distinct after the runtime adapter rounds them for diffusers.
    min_gap = 1.0
    clipped[-1] = 0.0
    if len(clipped) * min_gap > max_timestep + min_gap:
        clipped = np.linspace(max_timestep, 0.0, len(clipped), dtype=np.float64)
    else:
        clipped[0] = min(max(clipped[0], (len(clipped) - 1) * min_gap), max_timestep)
        for index in range(len(clipped) - 2, -1, -1):
            clipped[index] = min(max(clipped[index], clipped[index + 1] + min_gap), max_timestep)
        for index in range(1, len(clipped)):
            clipped[index] = min(clipped[index], clipped[index - 1] - min_gap)
        clipped[-1] = 0.0

    if np.any(np.diff(clipped) >= 0.0):
        clipped = np.linspace(float(clipped[0]), 0.0, len(clipped), dtype=np.float64)
    post_min = float(np.min(clipped))
    post_max = float(np.max(clipped))
    changed = bool(not np.allclose(values, clipped, rtol=0.0, atol=1.0e-9))
    meta = {
        **bundle.meta,
        "timestep_export_clip_enabled": changed,
        "timestep_export_clip_min_before": pre_min,
        "timestep_export_clip_max_before": pre_max,
        "timestep_export_clip_min_after": post_min,
        "timestep_export_clip_max_after": post_max,
    }
    sigma_grid = None
    sigmas = None if bundle.sigmas is None else bundle.sigmas.copy()
    if bundle.sigma_grid is not None:
        if scheduler is not None:
            sigma_grid = _interp_sigmas_for_timesteps(scheduler, clipped)
            sigma_grid[-1] = 0.0
            if sigmas is not None:
                sigmas = sigma_grid[:-1].copy()
        else:
            sigma_grid = bundle.sigma_grid.copy()
    return ScheduleBundle(
        timesteps=clipped[:-1].copy() if bundle.timesteps is not None else None,
        sigmas=None if sigmas is None else sigmas.copy(),
        time_grid=clipped,
        sigma_grid=sigma_grid,
        tau_grid=bundle.tau_grid,
        g_grid=bundle.g_grid,
        meta=meta,
    )


def _interval_midpoints(values: np.ndarray) -> np.ndarray:
    nodes = np.asarray(values, dtype=np.float64)
    return 0.5 * (nodes[:-1] + nodes[1:])


def _interval_sigma_profile(coordinate_grid: np.ndarray, *, coordinate_domain: str, sigma_transform) -> np.ndarray:
    midpoints = _interval_midpoints(coordinate_grid)
    if coordinate_domain == "lambda":
        return np.asarray(sigma_transform(midpoints), dtype=np.float64)
    return np.maximum(_interval_midpoints(coordinate_grid), 0.0)


def _prior_alpha_from_nodes(profile_grid: np.ndarray, prior_nodes: np.ndarray, *, eps: float) -> np.ndarray:
    grid = np.asarray(profile_grid, dtype=np.float64)
    nodes = np.asarray(prior_nodes, dtype=np.float64)
    if grid.ndim != 1 or nodes.ndim != 1 or len(grid) < 2 or len(nodes) < 2:
        raise ValueError("profile_grid and prior_nodes must be 1D arrays with at least two points.")
    grid_increasing = grid[0] <= grid[-1]
    xp = grid if grid_increasing else grid[::-1]
    prior = nodes if nodes[0] <= nodes[-1] else nodes[::-1]
    intervals = np.maximum(np.diff(prior), float(eps))
    interval_density = 1.0 / intervals
    centers = 0.5 * (prior[:-1] + prior[1:])
    if len(centers) == 1:
        alpha = np.full_like(xp, float(interval_density[0]), dtype=np.float64)
    else:
        alpha = np.interp(xp, centers, interval_density, left=interval_density[0], right=interval_density[-1])
    alpha = np.maximum(alpha, float(eps))
    return alpha if grid_increasing else alpha[::-1]


def _model_key_for_ays(model_asset: str) -> str | None:
    mapping = {
        "hf_stable_diffusion_15": "stable_diffusion_15",
        "hf_sdxl_base_10": "sdxl",
        "hf_deepfloyd_if_stage1": "deepfloyd_if_stage1",
    }
    return mapping.get(str(model_asset))


def build_prior_alpha(
    *,
    manifest: AssetManifest,
    model_asset: str,
    prior_schedule: str,
    profile_grid: np.ndarray,
    coordinate_domain: str,
    train_lambdas: np.ndarray | None,
    train_sigmas: np.ndarray | None,
    eps: float,
) -> np.ndarray | None:
    normalized = str(prior_schedule).lower()
    if normalized in {"", "none"}:
        return None
    if normalized == "base":
        return np.ones_like(np.asarray(profile_grid, dtype=np.float64))
    if normalized not in {"ays", "ays_like"}:
        raise ValueError("prior_schedule must be one of: ays, base, none.")
    if coordinate_domain != "lambda":
        raise ValueError("AYS prior blending is currently implemented for lambda coordinate schedules.")
    if train_lambdas is None or train_sigmas is None:
        raise ValueError("lambda tables are required to build an AYS prior in lambda coordinates.")
    model_key = _model_key_for_ays(model_asset)
    if model_key is None:
        raise ValueError(f"No published AYS prior mapping is available for model asset `{model_asset}`.")
    asset_key = f"ays_published_{model_key}_10step"
    if not manifest.has(asset_key):
        raise KeyError(f"Missing published AYS asset `{asset_key}` in manifest.")
    bundle = ScheduleBundle.load(manifest.path(asset_key))
    if bundle.sigma_grid is not None:
        prior_sigmas = np.asarray(bundle.sigma_grid, dtype=np.float64)
    elif bundle.sigmas is not None:
        prior_sigmas = np.concatenate([np.asarray(bundle.sigmas, dtype=np.float64), np.asarray([0.0])])
    else:
        raise ValueError(f"Published AYS bundle `{asset_key}` does not contain sigmas.")
    prior_lambdas = sigma_to_lambda(prior_sigmas, train_sigmas, train_lambdas)
    return _prior_alpha_from_nodes(profile_grid, prior_lambdas, eps=eps)


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


def _anchored_replay_cost_estimate(anchor_nfe: int, window_size: int) -> int:
    return int(anchor_nfe) * (4 + 7 * int(window_size))


def profile_cache_dir(
    *,
    cache_root: Path,
    schedule_family: str = SCHEDULE_FAMILY,
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
    defect_reduce: str | None = None,
    defect_quantile: float | None = None,
    q_prior: float | None = None,
    q_shrinkage: float | None = None,
    density_temperature: float | None = None,
    prior_schedule: str | None = None,
    prior_blend: float | None = None,
    sde_noise_kappa: float | None = None,
    target_nfe: int | None = None,
    target_steps: int | None = None,
    multires_nfes: Sequence[int] | None = None,
    anchor_nfe: int | None = None,
    window_size: int | None = None,
) -> Path:
    parts = [backend, str(schedule_family), estimator]
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
    if defect_reduce:
        parts.append(f"reduce_{defect_reduce}")
    if defect_quantile is not None:
        parts.append(f"dq_{defect_quantile:g}")
    if q_prior is not None:
        parts.append(f"qprior_{q_prior:g}")
    if q_shrinkage is not None:
        parts.append(f"qshrink_{q_shrinkage:g}")
    if density_temperature is not None:
        parts.append(f"temp_{density_temperature:g}")
    if prior_schedule and str(prior_schedule).lower() != "none":
        parts.append(f"prior_{prior_schedule}")
    if prior_blend is not None:
        parts.append(f"pblend_{prior_blend:g}")
    if sde_noise_kappa is not None:
        parts.append(f"sdek_{sde_noise_kappa:g}")
    if target_nfe is not None:
        parts.append(f"nfe_{int(target_nfe)}")
    if target_steps is not None:
        parts.append(f"steps_{int(target_steps)}")
    if multires_nfes is not None:
        parts.append("multires_" + "_".join(str(int(value)) for value in multires_nfes))
    if anchor_nfe is not None:
        parts.append(f"anchor_{int(anchor_nfe)}")
    if window_size is not None:
        parts.append(f"window_{int(window_size)}")
    return cache_root.joinpath(*parts)


def resolve_profile_cache_root(clock_config: dict[str, Any], args: argparse.Namespace) -> Path:
    configured = clock_config.get("cache_path")
    if configured:
        return resolve_repo_path(configured)
    configured_root = str(getattr(args, "profile_cache_root", "") or "")
    if configured_root:
        return resolve_repo_path(configured_root)
    return resolve_repo_path(Path(args.output_root) / "_profile_cache")


def save_profile(output_dir: Path, artifacts: DefectBalancedProfileArtifacts | FPClockArtifacts, meta: dict[str, Any]) -> None:
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
    if isinstance(artifacts, FPClockArtifacts):
        np.save(output_dir / "arc_length_profile.npy", artifacts.arc_length_profile)
        np.save(output_dir / "residual_perp_profile.npy", artifacts.residual_perp_profile)
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
    schedule_family: str = SCHEDULE_FAMILY,
    schedule_implementation_version: int | None = None,
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
    prior_schedule: str = "none",
    prior_blend: float = 0.0,
    density_temperature: float = 1.0,
    q_prior: float | None = None,
    q_shrinkage: float = 0.0,
    defect_reduce: str = "rms",
    defect_quantile: float = 0.75,
    sde_noise_kappa: float = 0.0,
    sde_brownian_bridge: bool = False,
    target_nfe: int | None = None,
    target_steps: int | None = None,
) -> dict[str, Any]:
    if schedule_implementation_version is None:
        schedule_implementation_version = (
            FP_CLOCK_VERSION if schedule_family == FP_CLOCK_SCHEDULE_FAMILY else DEFECT_BALANCED_CLOCK_VERSION
        )
    if schedule_family == FP_CLOCK_SCHEDULE_FAMILY and estimator == ANCHORED_REPLAY_ESTIMATOR_NAME:
        calibration_method = "anchored_quarter_replay_horizontal_defect"
    elif schedule_family == FP_CLOCK_SCHEDULE_FAMILY:
        calibration_method = "frenet_projected_richardson_arc_pullback"
    elif estimator == VELOCITY_CURVATURE_ESTIMATOR_NAME:
        calibration_method = "legacy_velocity_curvature_pilot_trajectory"
    else:
        calibration_method = "legacy_solver_step_refinement_full_half_quarter"
    meta = {
        "backend": backend,
        "model_asset": model_asset,
        "schedule_family": schedule_family,
        "schedule_implementation_version": schedule_implementation_version,
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
        "clock_version": f"fp_clock_v{FP_CLOCK_VERSION}" if schedule_family == FP_CLOCK_SCHEDULE_FAMILY else "legacy_sadb_v2",
        "prior_schedule": prior_schedule,
        "prior_blend": prior_blend,
        "density_temperature": density_temperature,
        "q_prior": q_prior,
        "q_shrinkage": q_shrinkage,
        "defect_reduce": defect_reduce,
        "defect_quantile": defect_quantile,
        "sde_noise_kappa": sde_noise_kappa,
        "sde_brownian_bridge": sde_brownian_bridge,
        "calibration_method": calibration_method,
    }
    if schedule_family == FP_CLOCK_SCHEDULE_FAMILY:
        meta.update(
            {
                "target_nfe": target_nfe,
                "target_steps": target_steps,
                "fp_clock_version": int(FP_CLOCK_VERSION),
                "defect_estimator": estimator,
            }
        )
    if extra:
        meta.update(extra)
    return meta


def build_or_load_pndm_profile(
    *,
    manifest: AssetManifest,
    args: argparse.Namespace,
    clock_config: dict[str, Any],
    target_nfe: int | None = None,
) -> tuple[ClockProfile, Path, dict[str, Any]]:
    dataset_config = load_yaml(args.dataset_config)
    model_asset = args.model_asset or dataset_config["default_model_asset"]
    native_config = load_native_config(dataset_config["native_config"])
    model, _ = load_model(dataset_config["native_config"], manifest.path(model_asset), device="cuda")
    schedule_cfg = native_config["Schedule"]
    target_solver = str(args.solver or "euler")
    calibration_solver = resolve_calibration_solver(clock_config, target_solver)
    schedule_family = str(clock_config.get("family", SCHEDULE_FAMILY))
    estimator = str(clock_config["estimator"])
    target_steps = None
    if schedule_family == FP_CLOCK_SCHEDULE_FAMILY:
        if target_nfe is None:
            raise ValueError("FP_CLOCK profile export requires a concrete target_nfe.")
        target_steps = resolve_effective_nfe_plan(target_solver, int(target_nfe)).solver_steps
        if calibration_solver != target_solver:
            raise ValueError("FP_CLOCK calibration_solver must resolve to the target solver.")
    anchored_spec = None
    anchor_nfe: int | None = None
    window_size: int | None = None
    if estimator == ANCHORED_REPLAY_ESTIMATOR_NAME:
        anchored_spec = get_solver_native_spec("pndm", calibration_solver)
        if not anchored_spec.supports_base_trajectory_recording:
            raise ValueError(f"PNDM solver `{calibration_solver}` cannot use anchored_replay FP calibration: {anchored_spec.notes}")
        coordinate_domain = str(anchored_spec.native_coordinate)
        anchor_nfe = int(clock_config.get("anchor_nfe", clock_config.get("calibration_nfe", 16)))
        window_size = int(clock_config.get("window_size", anchored_spec.recommended_window_len))
        physical_grid_size = int(anchor_nfe) + 1
        physical_grid_mode = "anchored_base"
    else:
        coordinate_domain = preferred_calibration_domain(calibration_solver)
        physical_grid_size = int(clock_config.get("physical_grid_size", 65))
        physical_grid_mode = None
    smoothing_window = int(clock_config.get("smoothing_window", 1))
    epsilon = float(clock_config.get("epsilon", 1.0e-12))
    q_min = float(clock_config["q_min"])
    q_max = float(clock_config["q_max"])
    pilot_batch_size = int(clock_config.get("pilot_batch_size", 8))
    pilot_num_batches = int(clock_config.get("pilot_num_batches", 4))
    pilot_observation_microbatch = int(clock_config.get("pilot_observation_microbatch", 4))
    warmup_steps = int(clock_config.get("warmup_steps", 1))
    cache_root = resolve_profile_cache_root(clock_config, args)
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        schedule_family=schedule_family,
        backend="pndm",
        dataset_name=str(dataset_config["name"]),
        model_asset=str(model_asset),
        solver=target_solver,
        calibration_solver=calibration_solver,
        estimator=estimator,
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
        physical_grid_mode=physical_grid_mode,
        anchor_nfe=anchor_nfe,
        window_size=window_size,
        target_nfe=target_nfe if schedule_family == FP_CLOCK_SCHEDULE_FAMILY else None,
        target_steps=target_steps,
    )
    extra_meta = {
        "dataset": dataset_config["name"],
        "pilot_data_source": "synthetic_noise_trajectories_only",
        "uses_dataset_samples": False,
        "warmup_steps": warmup_steps,
    }
    if estimator == ANCHORED_REPLAY_ESTIMATOR_NAME:
        assert anchored_spec is not None and anchor_nfe is not None and window_size is not None
        calibration_cost = int(
            pilot_batch_size
            * pilot_num_batches
            * _anchored_replay_cost_estimate(int(anchor_nfe), int(window_size))
        )
        extra_meta.update(
            {
                "defect_estimator": ANCHORED_REPLAY_ESTIMATOR_NAME,
                "anchor_nfe": int(anchor_nfe),
                "calibration_nfes": [int(anchor_nfe)],
                "grid_mode": "anchored_base",
                "physical_grid_mode": "anchored_base",
                "window_size": int(window_size),
                "window_len": int(window_size),
                "native_coordinate": coordinate_domain,
                "solver_order": int(anchored_spec.solver_order),
                "target_solver": target_solver,
                "target_nfe": int(target_nfe) if target_nfe is not None else None,
                "fp_clock_version": int(FP_CLOCK_VERSION),
                "replay_backend": "velocity_quarter_anchor",
                "reference_path": "quarter_refined_velocity",
                "q_estimator": "full_l2_replay_ratio",
                "residual_metric": "frenet_normal_replay_residual",
                "multistep_history_aware": False,
                "heun_omitted": bool(clock_config.get("heun_omitted", False)),
                "heun_omitted_reason": str(clock_config.get("heun_omitted_reason", "")),
                "calibration_cost_estimate": calibration_cost,
                "calibration_cost_unit": "model_evaluation_equivalents",
            }
        )
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
        schedule_family=schedule_family,
        backend="pndm",
        model_asset=str(model_asset),
        solver=target_solver,
        calibration_solver=calibration_solver,
        estimator=estimator,
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
        physical_grid_mode=physical_grid_mode,
        target_nfe=target_nfe if schedule_family == FP_CLOCK_SCHEDULE_FAMILY else None,
        target_steps=target_steps,
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
    physical_grid: np.ndarray
    if estimator == ANCHORED_REPLAY_ESTIMATOR_NAME:
        assert anchor_nfe is not None
        physical_grid, stats, detail_meta = collect_anchored_replay_calibration_stats(
            model=model,
            scheduler=scheduler,
            solver=calibration_solver,
            image_size=int(dataset_config["image_size"]),
            batch_size=pilot_batch_size,
            num_batches=pilot_num_batches,
            seed=args.seed,
            anchor_nfe=anchor_nfe,
            window_size=window_size,
            observation_microbatch=pilot_observation_microbatch,
            coordinate_domain=coordinate_domain,
            model_output_type=str(clock_config["model_output_type"]),
            q_min=q_min,
            q_max=q_max,
            eps=epsilon,
        )
        profile_meta = {**profile_meta, **detail_meta}
    else:
        physical_grid = build_pndm_physical_grid(
            scheduler=scheduler,
            coordinate_domain=coordinate_domain,
            diffusion_step=int(schedule_cfg["diffusion_step"]),
            physical_grid_size=physical_grid_size,
        )

    if estimator == FP_CLOCK_ESTIMATOR_NAME:
        stats = collect_fp_clock_calibration_stats(
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
    elif estimator == ANCHORED_REPLAY_ESTIMATOR_NAME:
        pass
    elif estimator == VELOCITY_CURVATURE_ESTIMATOR_NAME:
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

    if estimator in {FP_CLOCK_ESTIMATOR_NAME, ANCHORED_REPLAY_ESTIMATOR_NAME}:
        assert target_steps is not None
        artifacts = build_fp_clock_profile(
            physical_grid,
            stats,
            target_steps=target_steps,
            eps=epsilon,
            q_min=q_min,
            q_max=q_max,
            smoothing_window=smoothing_window,
        )
    else:
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
    target_nfe: int | None = None,
) -> tuple[ClockProfile, Path, dict[str, Any]]:
    from src.adapters.diffusers import (
        build_defect_sigma_grid,
        collect_anchored_replay_calibration_stats as collect_diffusers_anchored_replay_calibration_stats,
        get_pipeline_device,
        load_pipeline,
        prepare_defect_batch,
        replace_scheduler,
    )

    model_asset = str(args.model_asset)
    target_solver = normalize_diffusers_solver(str(args.solver or "flow_euler"))
    calibration_solver = normalize_diffusers_solver(resolve_calibration_solver(clock_config, target_solver))
    schedule_family = str(clock_config.get("family", SCHEDULE_FAMILY))
    estimator = str(clock_config["estimator"])
    if estimator == VELOCITY_CURVATURE_ESTIMATOR_NAME:
        raise ValueError("velocity_curvature calibration is currently implemented for backend=pndm only.")
    target_steps = None
    if schedule_family == FP_CLOCK_SCHEDULE_FAMILY:
        if target_nfe is None:
            raise ValueError("FP_CLOCK profile export requires a concrete target_nfe.")
        target_steps = resolve_effective_nfe_plan(target_solver, int(target_nfe)).solver_steps
        if calibration_solver != target_solver:
            raise ValueError("FP_CLOCK calibration_solver must resolve to the target solver.")
    effective_model_output_type = "flow" if _diffusers_solver_uses_flow_prediction(target_solver) else str(clock_config["model_output_type"])
    multires_nfes: tuple[int, ...] | None = None
    anchor_nfe: int | None = None
    window_size: int | None = None
    coordinate_domain = str(clock_config.get("coordinate_domain", "lambda"))
    if _diffusers_solver_uses_flow_prediction(target_solver) and coordinate_domain == "lambda":
        coordinate_domain = "sigma"
    if estimator == ANCHORED_REPLAY_ESTIMATOR_NAME:
        spec = get_solver_native_spec("diffusers", calibration_solver)
        if not spec.supports_base_trajectory_recording:
            raise ValueError(f"Diffusers solver `{calibration_solver}` cannot use anchored_replay FP calibration: {spec.notes}")
        anchor_nfe = int(clock_config.get("anchor_nfe", 16))
        window_size = int(clock_config.get("window_size", spec.recommended_window_len))
        if window_size < int(spec.solver_order):
            raise ValueError("anchored_replay window_size must be at least the solver order/history length.")
        coordinate_domain = str(clock_config.get("coordinate_domain", spec.native_coordinate))
        if coordinate_domain == "lambda":
            coordinate_domain = str(spec.native_coordinate)
        coordinate_domain = {"sigma": "sigmas", "timestep": "timesteps"}.get(coordinate_domain, coordinate_domain)
    physical_grid_size = int(clock_config.get("physical_grid_size", 65))
    physical_grid_mode = str(clock_config.get("physical_grid_mode", "scheduler_sigmas"))
    smoothing_window = int(clock_config.get("smoothing_window", 1))
    epsilon = float(clock_config.get("epsilon", 1.0e-12))
    default_q_prior = 2.2 if "sde" in target_solver else 3.0
    default_q_shrinkage = 0.6 if "sde" in target_solver else 0.5
    if "sde" in target_solver:
        q_min = float(clock_config.get("sde_q_min", 1.3))
        q_max = float(clock_config.get("sde_q_max", 3.5))
        q_prior = float(clock_config.get("sde_q_prior", default_q_prior))
        q_shrinkage = float(clock_config.get("sde_q_shrinkage", default_q_shrinkage))
    else:
        q_min = float(clock_config.get("q_min", 1.5))
        q_max = float(clock_config.get("q_max", 4.0))
        q_prior = float(clock_config.get("q_prior", default_q_prior))
        q_shrinkage = float(clock_config.get("q_shrinkage", default_q_shrinkage))
    defect_reduce = str(clock_config.get("defect_reduce", "quantile"))
    defect_quantile = float(clock_config.get("defect_quantile", 0.75))
    prior_schedule = str(clock_config.get("prior_schedule", "none")).lower()
    prior_blend = float(clock_config.get("prior_blend", 0.0))
    density_temperature = float(clock_config.get("density_temperature", 1.0))
    sde_noise_kappa = float(clock_config.get("sde_noise_kappa", 0.0 if "sde" not in target_solver else 0.3))
    sde_brownian_bridge = bool(clock_config.get("sde_brownian_bridge", False))
    pilot_batch_size = int(clock_config.get("pilot_batch_size", 8))
    pilot_num_batches = int(clock_config.get("pilot_num_batches", 4))
    pilot_observation_microbatch = int(clock_config.get("pilot_observation_microbatch", 4))
    pilot_prompt_asset = str(clock_config.get("pilot_prompt_asset", args.prompt_asset))
    if _diffusers_solver_uses_flow_prediction(target_solver) and prior_schedule in {"ays", "ays_like"}:
        prior_schedule = "none"
    cache_root = resolve_profile_cache_root(clock_config, args)
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        schedule_family=schedule_family,
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
        coordinate_domain=coordinate_domain,
        defect_reduce=defect_reduce,
        defect_quantile=defect_quantile,
        q_prior=q_prior,
        q_shrinkage=q_shrinkage,
        density_temperature=density_temperature,
        prior_schedule=prior_schedule,
        prior_blend=prior_blend,
        sde_noise_kappa=sde_noise_kappa,
        multires_nfes=multires_nfes,
        anchor_nfe=anchor_nfe,
        window_size=window_size,
        target_nfe=target_nfe if schedule_family == FP_CLOCK_SCHEDULE_FAMILY else None,
        target_steps=target_steps,
    )
    extra_meta = {
        "pilot_prompt_asset": pilot_prompt_asset,
        "uses_evaluation_prompts": False,
        "guidance_scale": float(args.guidance_scale),
        "physical_grid_mode": physical_grid_mode,
    }
    if estimator == ANCHORED_REPLAY_ESTIMATOR_NAME:
        assert anchor_nfe is not None and window_size is not None
        anchored_spec = get_solver_native_spec("diffusers", calibration_solver)
        calibration_cost = int(
            pilot_batch_size
            * pilot_num_batches
            * _anchored_replay_cost_estimate(int(anchor_nfe), int(window_size))
        )
        extra_meta.update(
            {
                "defect_estimator": ANCHORED_REPLAY_ESTIMATOR_NAME,
                "anchor_nfe": int(anchor_nfe),
                "calibration_nfes": [int(anchor_nfe)],
                "grid_mode": "anchored_base",
                "physical_grid_mode": "anchored_base",
                "window_size": int(window_size),
                "window_len": int(window_size),
                "native_coordinate": coordinate_domain,
                "solver_order": int(anchored_spec.solver_order),
                "target_solver": target_solver,
                "target_nfe": int(target_nfe) if target_nfe is not None else None,
                "fp_clock_version": int(FP_CLOCK_VERSION),
                "replay_backend": "scheduler_history_quarter_anchor",
                "reference_path": "quarter_refined_target_scheduler",
                "q_estimator": "full_l2_replay_ratio",
                "residual_metric": "frenet_normal_replay_residual",
                "multistep_history_aware": True,
                "heun_omitted": bool(clock_config.get("heun_omitted", False)),
                "heun_omitted_reason": str(clock_config.get("heun_omitted_reason", "")),
                "calibration_cost_estimate": calibration_cost,
                "calibration_cost_unit": "model_evaluation_equivalents",
            }
        )
    profile_meta = _build_profile_meta(
        schedule_family=schedule_family,
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
        coordinate_domain=coordinate_domain,
        prior_schedule=prior_schedule,
        prior_blend=prior_blend,
        density_temperature=density_temperature,
        q_prior=q_prior,
        q_shrinkage=q_shrinkage,
        defect_reduce=defect_reduce,
        defect_quantile=defect_quantile,
        sde_noise_kappa=sde_noise_kappa,
        sde_brownian_bridge=sde_brownian_bridge,
        target_nfe=target_nfe if schedule_family == FP_CLOCK_SCHEDULE_FAMILY else None,
        target_steps=target_steps,
        extra=extra_meta,
    )
    cached_profile = load_cached_profile_if_current(cache_dir, profile_meta)
    if cached_profile is not None:
        return cached_profile, cache_dir, profile_meta

    prompts_path = manifest.path(pilot_prompt_asset) if manifest.has(pilot_prompt_asset) else pilot_prompt_asset
    prompts = load_json(prompts_path)
    prompt_pool = [str(prompt) for prompt in prompts]
    pipeline = load_pipeline(manifest.path(model_asset), device="cuda", dtype_name=args.dtype)
    replace_scheduler(pipeline, calibration_solver)
    if estimator == ANCHORED_REPLAY_ESTIMATOR_NAME:
        assert anchor_nfe is not None and window_size is not None and target_steps is not None
        physical_grid, stats, detail_meta = collect_diffusers_anchored_replay_calibration_stats(
            pipeline=pipeline,
            solver=calibration_solver,
            prompt_pool=prompt_pool,
            batch_size=pilot_batch_size,
            num_batches=pilot_num_batches,
            seed=args.seed,
            anchor_nfe=int(anchor_nfe),
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            window_size=int(window_size),
            observation_microbatch=pilot_observation_microbatch,
            coordinate_domain=coordinate_domain,
            q_min=q_min,
            q_max=q_max,
            eps=epsilon,
        )
        profile_meta = {**profile_meta, **detail_meta}
        artifacts = build_fp_clock_profile(
            physical_grid,
            stats,
            target_steps=target_steps,
            eps=epsilon,
            q_min=q_min,
            q_max=q_max,
            smoothing_window=smoothing_window,
        )
        save_profile(cache_dir, artifacts, profile_meta)
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return artifacts.profile, cache_dir, profile_meta

    sigma_grid = build_defect_sigma_grid(
        pipeline,
        physical_grid_size=physical_grid_size,
        height=args.height,
        width=args.width,
        physical_grid_mode=physical_grid_mode,
    )
    train_timesteps = train_sigmas = train_lambdas = None
    sigma_to_lambda_transform = None
    lambda_to_sigma_transform = None
    if coordinate_domain == "lambda":
        train_timesteps, train_sigmas, train_lambdas = build_lambda_table(pipeline.scheduler)
        sigma_to_lambda_transform = lambda values: sigma_to_lambda(values, train_sigmas, train_lambdas)
        lambda_to_sigma_transform = lambda values: lambda_to_sigma(values, train_lambdas, train_sigmas)
        physical_grid = sigma_to_lambda_transform(sigma_grid)
    elif coordinate_domain == "sigma":
        physical_grid = sigma_grid
    elif coordinate_domain == "timestep":
        if not hasattr(pipeline.scheduler, "alphas_cumprod"):
            raise RuntimeError("timestep coordinate calibration requires alphas_cumprod.")
        sigma_to_time = _build_diffusers_sigma_to_timestep_transform(pipeline.scheduler)
        physical_grid = sigma_to_time(sigma_grid)
    else:
        raise ValueError(f"Unsupported diffusers coordinate_domain: {coordinate_domain}")
    if np.any(np.diff(physical_grid) <= 0.0) and coordinate_domain == "lambda":
        order = np.argsort(physical_grid)
        physical_grid = physical_grid[order]
        sigma_grid = sigma_grid[order]
    prior_alpha = build_prior_alpha(
        manifest=manifest,
        model_asset=model_asset,
        prior_schedule=prior_schedule,
        profile_grid=physical_grid,
        coordinate_domain=coordinate_domain,
        train_lambdas=train_lambdas,
        train_sigmas=train_sigmas,
        eps=epsilon,
    )
    sigma_profile = _interval_sigma_profile(
        physical_grid,
        coordinate_domain=coordinate_domain,
        sigma_transform=lambda_to_sigma_transform,
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
            velocity_fn = batch.velocity_fn
            if coordinate_domain == "lambda":
                assert train_lambdas is not None and train_sigmas is not None

                def velocity_lambda(sample, lamb, sample_start=0, sample_stop=None, base_velocity=velocity_fn):
                    sigma = lambda_to_sigma(
                        np.asarray([float(lamb.detach().float().cpu().item() if torch.is_tensor(lamb) else lamb)]),
                        train_lambdas,
                        train_sigmas,
                    )[0]
                    dsigma = lambda_to_sigma_derivative(
                        np.asarray([float(lamb.detach().float().cpu().item() if torch.is_tensor(lamb) else lamb)]),
                        train_lambdas,
                        train_sigmas,
                    )[0]
                    sigma_tensor = torch.as_tensor(float(sigma), device=sample.device, dtype=sample.dtype)
                    try:
                        sigma_velocity = base_velocity(sample, sigma_tensor, sample_start, sample_stop)
                    except TypeError:
                        sigma_velocity = base_velocity(sample, sigma_tensor)
                    return sigma_velocity * float(dsigma)

                velocity_fn = velocity_lambda
            step_fn = build_velocity_stepper(velocity_fn, calibration_solver)
            if estimator == FP_CLOCK_ESTIMATOR_NAME:
                stats_batches.append(
                    collect_fp_clock_stats(
                        initial_sample=batch.initial_latents,
                        physical_grid=physical_grid,
                        velocity_fn=velocity_fn,
                        step_fn=step_fn,
                        observation_microbatch=pilot_observation_microbatch,
                        q_min=q_min,
                        q_max=q_max,
                        eps=epsilon,
                    )
                )
            else:
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

    if estimator == FP_CLOCK_ESTIMATOR_NAME:
        assert target_steps is not None
        stats = concatenate_fp_clock_stats(stats_batches)
        artifacts = build_fp_clock_profile(
            physical_grid,
            stats,
            target_steps=target_steps,
            eps=epsilon,
            q_min=q_min,
            q_max=q_max,
            smoothing_window=smoothing_window,
        )
    else:
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
            defect_reduce=defect_reduce,
            defect_quantile=defect_quantile,
            q_prior=q_prior,
            q_shrinkage=q_shrinkage,
            q_min=q_min,
            q_max=q_max,
            density_temperature=density_temperature,
            prior_alpha=prior_alpha,
            prior_blend=prior_blend,
            sigma_profile=sigma_profile,
            solver_type="sde" if "sde" in target_solver else "ode",
            sde_noise_kappa=sde_noise_kappa,
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
    schedule_family = str(clock_config.get("family", SCHEDULE_FAMILY))
    representation = preferred_schedule_representation(target_solver)
    coordinate_domain = preferred_calibration_domain(calibration_solver)
    shared_profile: ClockProfile | None = None
    shared_cache_dir: Path | None = None
    shared_profile_meta: dict[str, Any] | None = None
    if schedule_family != FP_CLOCK_SCHEDULE_FAMILY:
        shared_profile, shared_cache_dir, shared_profile_meta = build_or_load_pndm_profile(
            manifest=manifest,
            args=args,
            clock_config=clock_config,
        )

    def profile_for_nfe(effective_nfe: int) -> tuple[ClockProfile, Path, dict[str, Any]]:
        if schedule_family == FP_CLOCK_SCHEDULE_FAMILY:
            return build_or_load_pndm_profile(
                manifest=manifest,
                args=args,
                clock_config=clock_config,
                target_nfe=int(effective_nfe),
            )
        assert shared_profile is not None and shared_cache_dir is not None and shared_profile_meta is not None
        return shared_profile, shared_cache_dir, shared_profile_meta

    def export_meta_for(cache_dir: Path, profile_meta: dict[str, Any]) -> dict[str, Any]:
        meta = {
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
            "schedule_implementation_version": int(profile_meta.get("schedule_implementation_version", DEFECT_BALANCED_CLOCK_VERSION)),
        }
        for key in (
            "defect_estimator",
            "multires_nfes",
            "anchor_nfe",
            "calibration_nfes",
            "grid_mode",
            "window_size",
            "window_len",
            "native_coordinate",
            "solver_order",
            "target_solver",
            "target_nfe",
            "target_steps",
            "fp_clock_version",
            "heun_omitted",
            "heun_omitted_reason",
            "calibration_cost_estimate",
            "calibration_cost_unit",
            "calibration_method",
            "replay_backend",
            "reference_path",
            "q_estimator",
            "residual_metric",
            "multistep_history_aware",
        ):
            if key in profile_meta:
                meta[key] = profile_meta[key]
        return meta
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
            target_scheduler.set_timesteps(
                resolve_effective_nfe_plan(target_solver, int(effective_nfe)).solver_steps,
                device=torch.device("cpu"),
            )
            profile, cache_dir, profile_meta = profile_for_nfe(int(effective_nfe))
            export_meta = export_meta_for(cache_dir, profile_meta)
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
                schedule_family=schedule_family,
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
            bundle = _clip_pndm_timestep_bundle_for_export(
                bundle,
                diffusion_step=int(schedule_cfg["diffusion_step"]),
                scheduler=target_scheduler,
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
        target_scheduler.set_timesteps(
            resolve_effective_nfe_plan(target_solver, int(effective_nfe)).solver_steps,
            device=torch.device("cpu"),
        )
        profile, cache_dir, profile_meta = profile_for_nfe(int(effective_nfe))
        export_meta = export_meta_for(cache_dir, profile_meta)
        bundle = build_reparameterized_bundle(
            profile,
            effective_nfe=int(effective_nfe),
            solver_name=target_solver,
            representation=representation,
            schedule_family=schedule_family,
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
        bundle = _clip_pndm_timestep_bundle_for_export(
            bundle,
            diffusion_step=int(schedule_cfg["diffusion_step"]),
            scheduler=target_scheduler,
        )
        bundle.save(output_root / f"nfe_{int(effective_nfe):03d}")


def export_diffusers(args: argparse.Namespace) -> None:
    manifest = AssetManifest(args.manifest)
    clock_config = load_clock_settings(args.clock_config)
    target_nfes = parse_target_nfes(args.target_nfes, list(clock_config.get("target_nfes", [])))
    schedule_family = str(clock_config.get("family", SCHEDULE_FAMILY))
    profile_cache: dict[int, tuple[ClockProfile, Path, dict[str, Any]]] = {}

    def profile_for_nfe(effective_nfe: int) -> tuple[ClockProfile, Path, dict[str, Any]]:
        cache_key = int(effective_nfe) if schedule_family == FP_CLOCK_SCHEDULE_FAMILY else -1
        if cache_key not in profile_cache:
            profile_cache[cache_key] = build_or_load_diffusers_profile(
                manifest=manifest,
                args=args,
                clock_config=clock_config,
                target_nfe=int(effective_nfe) if schedule_family == FP_CLOCK_SCHEDULE_FAMILY else None,
            )
        return profile_cache[cache_key]

    def clock_meta_fields(profile_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            key: profile_meta[key]
            for key in (
                "defect_estimator",
                "multires_nfes",
                "anchor_nfe",
                "calibration_nfes",
                "grid_mode",
                "window_size",
                "window_len",
                "native_coordinate",
                "solver_order",
                "target_solver",
                "target_nfe",
                "target_steps",
                "fp_clock_version",
                "heun_omitted",
                "heun_omitted_reason",
                "calibration_cost_estimate",
                "calibration_cost_unit",
                "calibration_cost_per_sample",
                "calibration_method",
                "replay_backend",
                "reference_path",
                "q_estimator",
                "residual_metric",
                "multistep_history_aware",
                "sde_variance_noise",
            )
            if key in profile_meta
        }

    profile, cache_dir, profile_meta = profile_for_nfe(int(target_nfes[0]))
    time_transform = None
    target_scheduler = None
    if not _diffusers_solver_uses_flow_prediction(str(args.solver or "flow_euler")):
        from src.adapters.diffusers import load_pipeline, replace_scheduler

        pipeline = load_pipeline(manifest.path(args.model_asset), device="cuda", dtype_name=args.dtype)
        replace_scheduler(pipeline, str(args.solver or "flow_euler"))
        target_scheduler = pipeline.scheduler
        coordinate_domain = str(profile_meta.get("coordinate_domain", "sigma"))
        train_timesteps = train_sigmas = train_lambdas = None
        if coordinate_domain == "lambda":
            train_timesteps, train_sigmas, train_lambdas = build_lambda_table(target_scheduler)
            representation_transform = lambda values: lambda_to_sigma(values, train_lambdas, train_sigmas)
            time_transform = lambda values: lambda_to_timestep(values, train_lambdas, train_timesteps)
        elif coordinate_domain == "timestep":
            representation_transform = _build_diffusers_timestep_to_sigma_transform(target_scheduler)
            time_transform = lambda values: np.asarray(values, dtype=np.float64)
        else:
            representation_transform = None
            time_transform = _build_diffusers_sigma_to_timestep_transform(target_scheduler)
        timestep_to_sigma = _build_diffusers_timestep_to_sigma_transform(target_scheduler)
        exported = []
        for effective_nfe in target_nfes:
            profile, cache_dir, profile_meta = profile_for_nfe(int(effective_nfe))
            bundle = build_reparameterized_bundle(
                profile,
                effective_nfe=int(effective_nfe),
                solver_name=str(args.solver or "flow_euler"),
                representation="sigmas",
                schedule_family=schedule_family,
                meta={
                    "backend": "diffusers",
                    "model_asset": args.model_asset,
                    "solver": args.solver or "flow_euler",
                    "calibration_solver": profile_meta["calibration_solver"],
                    "clock_config": args.clock_config,
                    "clock_model_output_type": str(clock_config["model_output_type"]),
                    "estimator": str(clock_config["estimator"]),
                    "physical_grid_mode": profile_meta.get("physical_grid_mode"),
                    "clock_version": profile_meta.get("clock_version"),
                    "coordinate_domain": coordinate_domain,
                    "prior_schedule": profile_meta.get("prior_schedule"),
                    "prior_blend": profile_meta.get("prior_blend"),
                    "density_temperature": profile_meta.get("density_temperature"),
                    "q_prior": profile_meta.get("q_prior"),
                    "q_shrinkage": profile_meta.get("q_shrinkage"),
                    "q_min": profile_meta.get("q_min"),
                    "q_max": profile_meta.get("q_max"),
                    "defect_reduce": profile_meta.get("defect_reduce"),
                    "defect_quantile": profile_meta.get("defect_quantile"),
                    "sde_noise_kappa": profile_meta.get("sde_noise_kappa"),
                    "sde_brownian_bridge": profile_meta.get("sde_brownian_bridge"),
                    "shared_profile_dir": str(cache_dir),
                    "shared_profile_meta": profile_meta,
                    "schedule_implementation_version": int(profile_meta.get("schedule_implementation_version", DEFECT_BALANCED_CLOCK_VERSION)),
                    **clock_meta_fields(profile_meta),
                },
                representation_transform=representation_transform,
                time_transform=time_transform,
            )
            max_dt_factor = clock_config.get("max_dt_factor")
            max_neighbor_ratio = clock_config.get("max_neighbor_dt_ratio")
            if max_dt_factor is not None or max_neighbor_ratio is not None:
                if coordinate_domain == "lambda":
                    assert train_sigmas is not None and train_lambdas is not None and train_timesteps is not None
                    reference_sigma_grid = _diffusers_reference_sigma_grid(
                        target_scheduler,
                        effective_nfe=int(effective_nfe),
                        device="cuda",
                    )
                    reference_lambda_grid = sigma_to_lambda(reference_sigma_grid, train_sigmas, train_lambdas)
                    lambda_grid = sigma_to_lambda(np.asarray(bundle.sigma_grid, dtype=np.float64), train_sigmas, train_lambdas)
                    limited_lambda_grid, limiter_meta = limit_schedule_step_sizes(
                        lambda_grid,
                        _limiter_reference_time_grid(reference_lambda_grid),
                        max_dt_factor=None if max_dt_factor is None else float(max_dt_factor),
                        max_neighbor_ratio=None if max_neighbor_ratio is None else float(max_neighbor_ratio),
                    )
                    limited_sigma_grid = lambda_to_sigma(limited_lambda_grid, train_lambdas, train_sigmas)
                    limited_time_grid = lambda_to_timestep(limited_lambda_grid, train_lambdas, train_timesteps)
                else:
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
    output_root = Path(args.output_root)
    for effective_nfe in target_nfes:
        profile, cache_dir, profile_meta = profile_for_nfe(int(effective_nfe))
        bundle = build_reparameterized_bundle(
            profile,
            effective_nfe=int(effective_nfe),
            solver_name=str(args.solver or "flow_euler"),
            representation="sigmas",
            schedule_family=schedule_family,
            meta={
                "backend": "diffusers",
                "model_asset": args.model_asset,
                "solver": args.solver or "flow_euler",
                "calibration_solver": profile_meta["calibration_solver"],
                "clock_config": args.clock_config,
                "clock_model_output_type": str(clock_config["model_output_type"]),
                "estimator": str(clock_config["estimator"]),
                "physical_grid_mode": profile_meta.get("physical_grid_mode"),
                "clock_version": profile_meta.get("clock_version"),
                "coordinate_domain": profile_meta.get("coordinate_domain"),
                "prior_schedule": profile_meta.get("prior_schedule"),
                "prior_blend": profile_meta.get("prior_blend"),
                "density_temperature": profile_meta.get("density_temperature"),
                "q_prior": profile_meta.get("q_prior"),
                "q_shrinkage": profile_meta.get("q_shrinkage"),
                "q_min": profile_meta.get("q_min"),
                "q_max": profile_meta.get("q_max"),
                "defect_reduce": profile_meta.get("defect_reduce"),
                "defect_quantile": profile_meta.get("defect_quantile"),
                "sde_noise_kappa": profile_meta.get("sde_noise_kappa"),
                "sde_brownian_bridge": profile_meta.get("sde_brownian_bridge"),
                "shared_profile_dir": str(cache_dir),
                "shared_profile_meta": profile_meta,
                "schedule_implementation_version": int(profile_meta.get("schedule_implementation_version", DEFECT_BALANCED_CLOCK_VERSION)),
                **clock_meta_fields(profile_meta),
            },
            time_transform=time_transform,
        )
        bundle.save(output_root / f"nfe_{int(effective_nfe):03d}")


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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clock.lcs import LCS_SCHEDULE_IMPLEMENTATION_VERSION
from src.clock.baseline import BASELINE_SCHEDULE_IMPLEMENTATION_VERSION
from src.clock.va import VA_SCHEDULE_IMPLEMENTATION_VERSION
from src.utils.config import dump_json, load_json, load_yaml, repo_root, resolve_repo_path
from src.utils.nfe_budget import normalize_solver_name, resolve_effective_nfe_plan
from src.utils.runtime_env import build_subprocess_env, command_preview, get_runtime_env, run_in_runtime_env


@dataclass(frozen=True)
class PrepareStep:
    key: str
    runtime_backend: str
    arguments: tuple[str, ...]
    output_path: Path


@dataclass(frozen=True)
class ExperimentInvocation:
    label: str
    runtime_backend: str
    run_arguments: tuple[str, ...]
    prepare_steps: tuple[PrepareStep, ...] = field(default_factory=tuple)
    output_dir: Path | None = None
    schedule_dir: Path | None = None
    materializable: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExecutionConfig:
    num_gpus: int
    gpu_ids: tuple[int, ...]
    prepare_schedules_first: bool
    prepare_gpu: int
    materialize_schedules: bool
    skip_existing: bool
    log_dir_root: Path
    schedule_cache_root: Path


@dataclass(frozen=True)
class ClockVariant:
    label: str | None
    config_path: str
    config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand and execute experiment YAMLs in the correct conda envs.")
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime_envs.yaml")
    parser.add_argument("--models-config", default="configs/models/modern_diffusers.yaml")
    parser.add_argument("--clock-config", default="configs/clocks/V_a.yaml")
    parser.add_argument("--ays-config", default="configs/clocks/AYS.yaml")
    parser.add_argument("--outputs-root", default="outputs/samples")
    parser.add_argument("--metrics-root", default="outputs/metrics")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--materialize-schedules", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-preview", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--distributed-child", action="store_true", default=False, help=argparse.SUPPRESS)
    return parser.parse_args()


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    raw = load_yaml(path)
    base_config = raw.get("base_config", {})
    if not isinstance(base_config, Mapping):
        raise TypeError("`base_config` must be a mapping when provided.")
    override = {key: value for key, value in raw.items() if key != "base_config"}
    return normalize_experiment_config(deep_merge_dicts(base_config, override))


def normalize_metric_names(raw_metrics: Any) -> list[str]:
    if raw_metrics is None:
        return []
    if isinstance(raw_metrics, str):
        return [raw_metrics]
    if not isinstance(raw_metrics, (list, tuple)):
        raise TypeError("`metrics` must be a string or list of strings.")
    return [str(item) for item in raw_metrics]


def normalize_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if "eval_nfes" in normalized and "nfes" not in normalized:
        normalized["nfes"] = normalized["eval_nfes"]

    if "metrics" in normalized:
        metrics = normalize_metric_names(normalized["metrics"])
    elif "metric" in normalized:
        metrics = normalize_metric_names([normalized["metric"]])
    else:
        metrics = []
    normalized["metrics"] = metrics
    if "metric" not in normalized and len(metrics) == 1:
        normalized["metric"] = metrics[0]
    return normalized


def wants_metric(experiment_config: Mapping[str, Any], metric_name: str) -> bool:
    desired = metric_name.lower()
    return any(str(metric).lower() == desired for metric in experiment_config.get("metrics", []))


def canonical_schedule_name(name: str) -> tuple[str, str]:
    normalized = name.lower().replace("-", "_")
    mapping = {
        "base": ("base", "base"),
        "linear": ("linear", "linear"),
        "ays": ("ays", "ays_like"),
        "v_a": ("V_a", "V_a"),
        "lcs_1": ("LCS-1", "LCS_1"),
        "lcs_2": ("LCS-2", "LCS_2"),
        "v_b": ("V_b", "V_b"),
        "a_a": ("A_a", "A_a"),
        "a_b": ("A_b", "A_b"),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported schedule name: {name}")
    return mapping[normalized]


def resolve_solver_schedule_overrides(
    experiment_config: Mapping[str, Any],
    *,
    solvers: list[str],
    default_schedules: list[Any],
) -> dict[str, tuple[str, ...]]:
    raw_overrides = experiment_config.get("solver_schedules")
    if raw_overrides is None:
        return {solver: tuple(str(item) for item in default_schedules) for solver in solvers}
    if not isinstance(raw_overrides, Mapping):
        raise TypeError("`solver_schedules` must be a mapping when provided.")

    normalized_defaults = tuple(str(item) for item in default_schedules)
    overrides: dict[str, tuple[str, ...]] = {}
    unknown_solvers = set(raw_overrides.keys()) - set(solvers)
    if unknown_solvers:
        raise ValueError(f"`solver_schedules` contains unknown solvers: {sorted(unknown_solvers)}")

    for solver in solvers:
        raw_schedules = raw_overrides.get(solver, normalized_defaults)
        if not isinstance(raw_schedules, (list, tuple)) or not raw_schedules:
            raise TypeError(f"`solver_schedules.{solver}` must be a non-empty list when provided.")
        overrides[solver] = tuple(str(item) for item in raw_schedules)
    return overrides


def validate_pndm_custom_dpm_schedule_semantics(
    solver_schedule_overrides: Mapping[str, tuple[str, ...]],
) -> None:
    lu_key = "dpm_solver_lu"
    default_key = "dpm_solver_default"
    if lu_key not in solver_schedule_overrides or default_key not in solver_schedule_overrides:
        return

    def active_custom_schedules(items: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            canonical_schedule_name(schedule_name)[0]
            for schedule_name in items
            if canonical_schedule_name(schedule_name)[0] != "base"
        )

    lu_custom = active_custom_schedules(solver_schedule_overrides[lu_key])
    default_custom = active_custom_schedules(solver_schedule_overrides[default_key])
    if not lu_custom or not default_custom:
        return

    raise ValueError(
        "Custom PNDM schedule injection collapses `dpm_solver_lu` and `dpm_solver_default` onto the same "
        "explicit sigma-grid execution path, so they should not both run with non-base schedules in the same "
        "experiment. Keep one variant `base`-only or remove it."
    )


def resolve_clock_variants(
    experiment_config: Mapping[str, Any],
    *,
    default_clock_config_path: str,
) -> tuple[ClockVariant, ...]:
    raw_variants = experiment_config.get("clock_variants")
    if raw_variants is None:
        return (ClockVariant(label=None, config_path=default_clock_config_path, config=load_yaml(default_clock_config_path)),)

    if not isinstance(raw_variants, (list, tuple)) or not raw_variants:
        raise TypeError("`clock_variants` must be a non-empty list when provided.")

    variants: list[ClockVariant] = []
    seen_labels: set[str] = set()
    for item in raw_variants:
        if isinstance(item, str):
            config_path = item
            label = Path(config_path).stem
        elif isinstance(item, Mapping):
            config_path = str(item.get("path") or item.get("clock_config") or "")
            if not config_path:
                raise ValueError("Each `clock_variants` entry must include `path` or `clock_config`.")
            label = str(item.get("label") or Path(config_path).stem)
        else:
            raise TypeError("Each `clock_variants` entry must be a string path or a mapping.")
        if label in seen_labels:
            raise ValueError(f"Duplicate clock variant label: {label}")
        seen_labels.add(label)
        variants.append(ClockVariant(label=label, config_path=config_path, config=load_yaml(config_path)))
    return tuple(variants)


def resolve_schedule_clock_configs(
    experiment_config: Mapping[str, Any],
    *,
    default_va_clock_config_path: str,
) -> dict[str, str]:
    configs = {
        "V_a": default_va_clock_config_path,
        "LCS-1": "configs/clocks/LCS_1.yaml",
        "LCS-2": "configs/clocks/LCS_2.yaml",
    }
    raw_mapping = experiment_config.get("schedule_clock_configs")
    if raw_mapping is None:
        return configs
    if not isinstance(raw_mapping, Mapping):
        raise TypeError("`schedule_clock_configs` must be a mapping when provided.")
    for raw_key, raw_value in raw_mapping.items():
        schedule_name, _ = canonical_schedule_name(str(raw_key))
        if schedule_name not in configs:
            raise ValueError(f"Unsupported materializable schedule in `schedule_clock_configs`: {raw_key}")
        if isinstance(raw_value, str):
            configs[schedule_name] = raw_value
            continue
        if isinstance(raw_value, Mapping):
            config_path = str(raw_value.get("path") or raw_value.get("clock_config") or "")
            if not config_path:
                raise ValueError(
                    f"`schedule_clock_configs.{raw_key}` must provide a string path or a mapping with `path` / `clock_config`."
                )
            configs[schedule_name] = config_path
            continue
        raise TypeError("Each `schedule_clock_configs` entry must be a string path or a mapping.")
    return configs


def active_clock_variants_for_schedule(
    schedule_name: str,
    *,
    va_clock_variants: tuple[ClockVariant, ...],
    schedule_clock_configs: Mapping[str, str],
) -> tuple[ClockVariant, ...]:
    if schedule_name == "V_a":
        return va_clock_variants
    if schedule_name in schedule_clock_configs:
        config_path = str(schedule_clock_configs[schedule_name])
        return (ClockVariant(label=None, config_path=config_path, config=load_yaml(config_path)),)
    return (ClockVariant(label=None, config_path="", config={}),)


def schedule_display_name(schedule_name: str, clock_label: str | None) -> str:
    if clock_label is None:
        return schedule_name
    return f"{schedule_name}[{clock_label}]"


def extend_schedule_parts(parts: tuple[str, ...], clock_label: str | None) -> tuple[str, ...]:
    if clock_label is None:
        return parts
    return (*parts, clock_label)


def pndm_schedule_parts(
    *,
    dataset_name: str,
    model_asset: str,
    solver: str,
    schedule_name: str,
    clock_label: str | None,
    share_ays_across_solvers: bool,
) -> tuple[str, ...]:
    if schedule_name == "ays" and share_ays_across_solvers and normalize_solver_name(solver) != "heun2":
        return extend_schedule_parts((dataset_name, model_asset, "shared"), clock_label)
    return extend_schedule_parts((dataset_name, model_asset, solver), clock_label)


def shared_schedule_root(schedule_folder: str, backend: str, *parts: str) -> Path:
    return Path("schedules") / schedule_folder / backend / Path(*parts)


def cached_schedule_root(cache_root: Path, schedule_folder: str, backend: str, *parts: str) -> Path:
    return cache_root / schedule_folder / backend / Path(*parts)


def is_materializable_schedule(backend: str, schedule_name: str) -> bool:
    materializable = {
        "pndm": {"linear", "V_a", "LCS-1", "LCS-2"},
        "diffusers": {"linear", "V_a", "LCS-1", "LCS-2"},
    }
    return schedule_name in materializable.get(backend, set())


def resolve_schedule_root(
    *,
    schedule_cache_root: Path,
    schedule_folder: str,
    backend: str,
    materializable: bool,
    parts: tuple[str, ...],
) -> Path:
    if materializable:
        return cached_schedule_root(schedule_cache_root, schedule_folder, backend, *parts)
    return shared_schedule_root(schedule_folder, backend, *parts)


def schedule_target_dir(root: Path, nfe: int) -> Path:
    return root / f"nfe_{int(nfe):03d}"


def infer_num_samples(experiment_config: Mapping[str, Any], dataset_config: Mapping[str, Any]) -> int:
    if "num_samples" in experiment_config:
        return int(experiment_config["num_samples"])
    name = str(experiment_config.get("name", ""))
    if name.startswith("smoke"):
        return int(dataset_config.get("smoke_num_samples", 100))
    if wants_metric(experiment_config, "fid"):
        return int(dataset_config.get("full_num_samples", dataset_config.get("smoke_num_samples", 100)))
    return int(dataset_config.get("smoke_num_samples", 100))


def infer_batch_size(experiment_config: Mapping[str, Any], dataset_config: Mapping[str, Any]) -> int:
    if "batch_size" in experiment_config:
        return int(experiment_config["batch_size"])
    name = str(experiment_config.get("name", ""))
    if name.startswith("smoke"):
        return int(dataset_config.get("smoke_batch_size", dataset_config.get("default_batch_size", 16)))
    return int(dataset_config.get("default_batch_size", dataset_config.get("smoke_batch_size", 16)))


def validate_effective_nfes(solvers: list[str], target_nfes: list[int]) -> None:
    for solver in solvers:
        for nfe in target_nfes:
            resolve_effective_nfe_plan(solver, nfe)


def build_pndm_prepare_steps(
    *,
    manifest_path: str,
    dataset_config_path: str,
    ays_config_path: str,
    schedule_cache_root: Path,
    dataset_name: str,
    model_asset: str,
    solver: str,
    schedule_name: str,
    schedule_folder: str,
    target_nfes: list[int],
    seed: int,
    clock_config_path: str,
    clock_label: str | None,
    share_ays_across_solvers: bool,
    ays_reference_solver: str,
    ays_consumer_solvers: tuple[str, ...],
) -> tuple[PrepareStep, ...]:
    if schedule_name == "base":
        return ()

    # Repo-owned PNDM AYS reproduction is intentionally retired. PNDM-side AYS
    # is now treated as an external asset only, so the launcher must not
    # regenerate it implicitly.
    if schedule_name == "ays":
        return ()

    if schedule_name == "linear":
        linear_parts = pndm_schedule_parts(
            dataset_name=dataset_name,
            model_asset=model_asset,
            solver=solver,
            schedule_name=schedule_name,
            clock_label=clock_label,
            share_ays_across_solvers=share_ays_across_solvers,
        )
        return tuple(
            PrepareStep(
                key=f"linear:pndm:{':'.join(linear_parts)}:{nfe}",
                runtime_backend="pndm",
                output_path=schedule_target_dir(
                    cached_schedule_root(schedule_cache_root, schedule_folder, "pndm", *linear_parts),
                    nfe,
                ),
                arguments=(
                    "scripts/run/export_baseline_schedule.py",
                    "--backend",
                    "pndm",
                    "--manifest",
                    manifest_path,
                    "--dataset-config",
                    dataset_config_path,
                    "--mode",
                    "linear",
                    "--solver",
                    solver,
                    "--nfe",
                    str(nfe),
                    "--output-dir",
                    str(
                        schedule_target_dir(
                            cached_schedule_root(
                                schedule_cache_root,
                                schedule_folder,
                                "pndm",
                                *linear_parts,
                            ),
                            nfe,
                        )
                    ),
                ),
            )
            for nfe in target_nfes
        )

    if schedule_name == "V_a":
        variant_parts = pndm_schedule_parts(
            dataset_name=dataset_name,
            model_asset=model_asset,
            solver=solver,
            schedule_name=schedule_name,
            clock_label=clock_label,
            share_ays_across_solvers=share_ays_across_solvers,
        )
        root = cached_schedule_root(schedule_cache_root, schedule_folder, "pndm", *variant_parts)
        return (
            PrepareStep(
                key=f"V_a:pndm:{':'.join(variant_parts)}",
                runtime_backend="pndm",
                output_path=root,
                arguments=(
                    "scripts/run/export_va_schedule.py",
                    "--backend",
                    "pndm",
                    "--manifest",
                    manifest_path,
                    "--clock-config",
                    clock_config_path,
                    "--dataset-config",
                    dataset_config_path,
                    "--model-asset",
                    model_asset,
                    "--solver",
                    solver,
                    "--seed",
                    str(seed),
                    "--target-nfes",
                    ",".join(str(nfe) for nfe in target_nfes),
                    "--output-root",
                    str(root),
                ),
            ),
        )

    if schedule_name in {"LCS-1", "LCS-2"}:
        variant_parts = pndm_schedule_parts(
            dataset_name=dataset_name,
            model_asset=model_asset,
            solver=solver,
            schedule_name=schedule_name,
            clock_label=clock_label,
            share_ays_across_solvers=share_ays_across_solvers,
        )
        root = cached_schedule_root(schedule_cache_root, schedule_folder, "pndm", *variant_parts)
        return (
            PrepareStep(
                key=f"{schedule_name}:pndm:{':'.join(variant_parts)}",
                runtime_backend="pndm",
                output_path=root,
                arguments=(
                    "scripts/run/export_lcs_schedule.py",
                    "--backend",
                    "pndm",
                    "--manifest",
                    manifest_path,
                    "--clock-config",
                    clock_config_path,
                    "--dataset-config",
                    dataset_config_path,
                    "--model-asset",
                    model_asset,
                    "--solver",
                    solver,
                    "--seed",
                    str(seed),
                    "--target-nfes",
                    ",".join(str(nfe) for nfe in target_nfes),
                    "--output-root",
                    str(root),
                ),
            ),
        )

    return ()


def build_diffusers_prepare_steps(
    *,
    manifest_path: str,
    schedule_cache_root: Path,
    model_key: str,
    model_asset: str,
    solver: str,
    schedule_name: str,
    schedule_folder: str,
    target_nfes: list[int],
    prompt_asset: str,
    seed: int,
    dtype_name: str,
    image_size: int,
    guidance_scale: float,
    clock_config_path: str,
    clock_label: str | None,
) -> tuple[PrepareStep, ...]:
    if schedule_name == "base":
        return ()

    if schedule_name == "linear":
        linear_parts = extend_schedule_parts((model_key, solver), clock_label)
        return tuple(
            PrepareStep(
                key=f"linear:diffusers:{':'.join(linear_parts)}:{nfe}",
                runtime_backend="diffusers",
                output_path=schedule_target_dir(
                    cached_schedule_root(schedule_cache_root, schedule_folder, "diffusers", *linear_parts), nfe
                ),
                arguments=(
                    "scripts/run/export_baseline_schedule.py",
                    "--backend",
                    "diffusers",
                    "--manifest",
                    manifest_path,
                    "--model-asset",
                    model_asset,
                    "--mode",
                    "linear",
                    "--solver",
                    solver,
                    "--dtype",
                    dtype_name,
                    "--nfe",
                    str(nfe),
                    "--output-dir",
                    str(
                        schedule_target_dir(
                            cached_schedule_root(schedule_cache_root, schedule_folder, "diffusers", *linear_parts),
                            nfe,
                        )
                    ),
                ),
            )
            for nfe in target_nfes
        )

    if schedule_name == "V_a":
        variant_parts = extend_schedule_parts((model_key, solver), clock_label)
        root = cached_schedule_root(schedule_cache_root, schedule_folder, "diffusers", *variant_parts)
        return (
            PrepareStep(
                key=f"V_a:diffusers:{':'.join(variant_parts)}",
                runtime_backend="diffusers",
                output_path=root,
                arguments=(
                    "scripts/run/export_va_schedule.py",
                    "--backend",
                    "diffusers",
                    "--manifest",
                    manifest_path,
                    "--clock-config",
                    clock_config_path,
                    "--model-asset",
                    model_asset,
                    "--prompt-asset",
                    prompt_asset,
                    "--solver",
                    solver,
                    "--seed",
                    str(seed),
                    "--dtype",
                    dtype_name,
                    "--height",
                    str(image_size),
                    "--width",
                    str(image_size),
                    "--guidance-scale",
                    str(guidance_scale),
                    "--target-nfes",
                    ",".join(str(nfe) for nfe in target_nfes),
                    "--output-root",
                    str(root),
                ),
            ),
        )

    if schedule_name in {"LCS-1", "LCS-2"}:
        variant_parts = extend_schedule_parts((model_key, solver), clock_label)
        root = cached_schedule_root(schedule_cache_root, schedule_folder, "diffusers", *variant_parts)
        return (
            PrepareStep(
                key=f"{schedule_name}:diffusers:{':'.join(variant_parts)}",
                runtime_backend="diffusers",
                output_path=root,
                arguments=(
                    "scripts/run/export_lcs_schedule.py",
                    "--backend",
                    "diffusers",
                    "--manifest",
                    manifest_path,
                    "--clock-config",
                    clock_config_path,
                    "--model-asset",
                    model_asset,
                    "--prompt-asset",
                    prompt_asset,
                    "--solver",
                    solver,
                    "--seed",
                    str(seed),
                    "--dtype",
                    dtype_name,
                    "--height",
                    str(image_size),
                    "--width",
                    str(image_size),
                    "--guidance-scale",
                    str(guidance_scale),
                    "--target-nfes",
                    ",".join(str(nfe) for nfe in target_nfes),
                    "--output-root",
                    str(root),
                ),
            ),
        )

    return ()


def build_pndm_invocations(
    experiment_config: Mapping[str, Any],
    *,
    manifest_path: str,
    ays_config_path: str,
    va_clock_variants: tuple[ClockVariant, ...],
    schedule_clock_configs: Mapping[str, str],
    schedule_cache_root: Path,
    outputs_root: str,
    metrics_root: str,
) -> list[ExperimentInvocation]:
    dataset_names = experiment_config.get("datasets") or [experiment_config["dataset"]]
    solvers = [str(item) for item in experiment_config.get("solvers", [])]
    schedules = experiment_config.get("schedules") or experiment_config.get("variants") or [experiment_config["schedule"]]
    target_nfes = [int(item) for item in experiment_config.get("nfes", [])]
    validate_effective_nfes(solvers, target_nfes)
    solver_schedule_overrides = resolve_solver_schedule_overrides(
        experiment_config,
        solvers=solvers,
        default_schedules=list(schedules),
    )
    validate_pndm_custom_dpm_schedule_semantics(solver_schedule_overrides)
    summary_csv = Path(metrics_root) / f"{experiment_config['name']}.csv"
    seed = int(experiment_config.get("seed", 0))
    save_samples = bool(experiment_config.get("save_samples", True))
    preview_samples = int(experiment_config.get("preview_samples", 0))
    if preview_samples < 0:
        raise ValueError("`preview_samples` must be non-negative.")
    invocations: list[ExperimentInvocation] = []
    share_ays_across_solvers = bool(experiment_config.get("share_pndm_ays_across_solvers", False))
    configured_reference_solver = experiment_config.get("ays_reference_solver")
    if configured_reference_solver is not None and str(configured_reference_solver) not in solvers:
        raise ValueError("`ays_reference_solver` must be present in the experiment `solvers` list.")

    for dataset_name in dataset_names:
        dataset_config_path = Path("configs/datasets") / f"{dataset_name}.yaml"
        dataset_config = load_yaml(dataset_config_path)
        model_assets = experiment_config.get("model_assets") or [dataset_config["default_model_asset"]]
        num_samples = infer_num_samples(experiment_config, dataset_config)
        batch_size = infer_batch_size(experiment_config, dataset_config)
        compute_fid = wants_metric(experiment_config, "fid")

        for model_asset in model_assets:
            ays_reference_solver = str(configured_reference_solver or solvers[0])
            for solver in solvers:
                active_schedules = solver_schedule_overrides[solver]
                for raw_schedule in active_schedules:
                    schedule_name, schedule_folder = canonical_schedule_name(str(raw_schedule))
                    active_clock_variants = active_clock_variants_for_schedule(
                        schedule_name,
                        va_clock_variants=va_clock_variants,
                        schedule_clock_configs=schedule_clock_configs,
                    )

                    for clock_variant in active_clock_variants:
                        active_ays_consumers = tuple(
                            item
                            for item in solvers
                            if not (
                                schedule_name == "ays"
                                and share_ays_across_solvers
                                and normalize_solver_name(item) == "heun2"
                            )
                            and any(
                                canonical_schedule_name(candidate)[0] == "ays"
                                for candidate in solver_schedule_overrides[item]
                            )
                        )
                        if schedule_name == "ays" and normalize_solver_name(solver) == "heun2":
                            active_ays_consumers = (solver,)
                        schedule_parts = pndm_schedule_parts(
                            dataset_name=dataset_name,
                            model_asset=str(model_asset),
                            solver=solver,
                            schedule_name=schedule_name,
                            clock_label=clock_variant.label,
                            share_ays_across_solvers=share_ays_across_solvers,
                        )
                        materializable = is_materializable_schedule("pndm", schedule_name)
                        prepare_steps = build_pndm_prepare_steps(
                            manifest_path=manifest_path,
                            dataset_config_path=str(dataset_config_path),
                            ays_config_path=ays_config_path,
                            schedule_cache_root=schedule_cache_root,
                            dataset_name=dataset_name,
                            model_asset=str(model_asset),
                            solver=solver,
                            schedule_name=schedule_name,
                            schedule_folder=schedule_folder,
                            target_nfes=target_nfes,
                            seed=seed,
                            clock_config_path=clock_variant.config_path,
                            clock_label=clock_variant.label,
                            share_ays_across_solvers=share_ays_across_solvers,
                            ays_reference_solver=ays_reference_solver,
                            ays_consumer_solvers=active_ays_consumers,
                        )
                        schedule_root_path = resolve_schedule_root(
                            schedule_cache_root=schedule_cache_root,
                            schedule_folder=schedule_folder,
                            backend="pndm",
                            materializable=materializable,
                            parts=schedule_parts,
                        )
                        display_name = schedule_display_name(schedule_name, clock_variant.label)

                        for nfe in target_nfes:
                            schedule_dir = None if schedule_name == "base" else schedule_target_dir(schedule_root_path, nfe)
                            notes: list[str] = []
                            if schedule_name == "ays" and share_ays_across_solvers:
                                notes.append(f"shared_ays_reference_solver={ays_reference_solver}")
                            if schedule_dir is not None and not resolve_repo_path(schedule_dir).exists():
                                notes.append(
                                    "schedule_missing_materializable" if materializable else "schedule_missing_external_asset"
                                )

                            output_dir = Path(outputs_root) / experiment_config["name"] / "pndm" / dataset_name / str(
                                model_asset
                            ) / solver / schedule_name
                            if clock_variant.label is not None:
                                output_dir = output_dir / clock_variant.label
                            output_dir = output_dir / f"nfe_{nfe:03d}"

                            run_arguments = [
                                "scripts/run/run_pndm_experiment.py",
                                "--manifest",
                                manifest_path,
                                "--dataset-config",
                                str(dataset_config_path),
                                "--model-asset",
                                str(model_asset),
                                "--solver",
                                solver,
                                "--nfe",
                                str(nfe),
                                "--num-samples",
                                str(num_samples),
                                "--batch-size",
                                str(batch_size),
                                "--seed",
                                str(seed),
                                "--schedule-name",
                                display_name,
                                "--output-dir",
                                str(output_dir),
                                "--summary-csv",
                                str(summary_csv),
                            ]
                            if compute_fid:
                                run_arguments.append("--compute-fid")
                            if not save_samples:
                                run_arguments.append("--discard-samples")
                            if preview_samples > 0:
                                run_arguments.extend(["--preview-samples", str(preview_samples)])
                            if schedule_dir is not None:
                                run_arguments.extend(["--schedule-dir", str(schedule_dir)])

                            invocations.append(
                                ExperimentInvocation(
                                    label=f"pndm:{dataset_name}:{model_asset}:{solver}:{display_name}:nfe_{nfe:03d}",
                                    runtime_backend="pndm",
                                    run_arguments=tuple(run_arguments),
                                    prepare_steps=prepare_steps,
                                    output_dir=output_dir,
                                    schedule_dir=schedule_dir,
                                    materializable=materializable,
                                    notes=tuple(notes),
                                )
                            )
    return invocations


def build_diffusers_invocations(
    experiment_config: Mapping[str, Any],
    *,
    manifest_path: str,
    va_clock_variants: tuple[ClockVariant, ...],
    schedule_clock_configs: Mapping[str, str],
    models_config_path: str,
    schedule_cache_root: Path,
    outputs_root: str,
    metrics_root: str,
    dtype_name: str,
) -> list[ExperimentInvocation]:
    model_catalog = load_yaml(models_config_path)["models"]
    model_keys = [str(item) for item in experiment_config.get("models", [])]
    solvers = [str(item) for item in experiment_config.get("solvers", [])]
    schedules = experiment_config.get("schedules") or [experiment_config["schedule"]]
    target_nfes = [int(item) for item in experiment_config.get("nfes", [])]
    validate_effective_nfes(solvers, target_nfes)
    solver_schedule_overrides = resolve_solver_schedule_overrides(
        experiment_config,
        solvers=solvers,
        default_schedules=list(schedules),
    )
    prompt_asset = str(experiment_config.get("prompt_asset", "diffusers_smoke_prompts"))
    summary_csv = Path(metrics_root) / f"{experiment_config['name']}.csv"
    seed = int(experiment_config.get("seed", 0))
    invocations: list[ExperimentInvocation] = []

    for model_key in model_keys:
        model_config = model_catalog[model_key]
        model_asset = str(model_config["asset_key"])
        image_size = int(model_config.get("image_size", 512))
        guidance_scale = float(model_config.get("guidance_scale", 3.5))

        for solver in solvers:
            active_schedules = solver_schedule_overrides[solver]
            for raw_schedule in active_schedules:
                schedule_name, schedule_folder = canonical_schedule_name(str(raw_schedule))
                active_clock_variants = active_clock_variants_for_schedule(
                    schedule_name,
                    va_clock_variants=va_clock_variants,
                    schedule_clock_configs=schedule_clock_configs,
                )

                for clock_variant in active_clock_variants:
                    materializable = is_materializable_schedule("diffusers", schedule_name)
                    prepare_steps = build_diffusers_prepare_steps(
                        manifest_path=manifest_path,
                        schedule_cache_root=schedule_cache_root,
                        model_key=model_key,
                        model_asset=model_asset,
                        solver=solver,
                        schedule_name=schedule_name,
                        schedule_folder=schedule_folder,
                        target_nfes=target_nfes,
                        prompt_asset=prompt_asset,
                        seed=seed,
                        dtype_name=dtype_name,
                        image_size=image_size,
                        guidance_scale=guidance_scale,
                        clock_config_path=clock_variant.config_path,
                        clock_label=clock_variant.label,
                    )
                    schedule_root_path = resolve_schedule_root(
                        schedule_cache_root=schedule_cache_root,
                        schedule_folder=schedule_folder,
                        backend="diffusers",
                        materializable=materializable,
                        parts=extend_schedule_parts((model_key, solver), clock_variant.label),
                    )
                    display_name = schedule_display_name(schedule_name, clock_variant.label)

                    for nfe in target_nfes:
                        schedule_dir = None if schedule_name == "base" else schedule_target_dir(schedule_root_path, nfe)
                        notes: list[str] = []
                        if schedule_dir is not None and not resolve_repo_path(schedule_dir).exists():
                            notes.append(
                                "schedule_missing_materializable" if materializable else "schedule_missing_external_asset"
                            )

                        output_dir = Path(outputs_root) / experiment_config["name"] / "diffusers" / model_key / solver / schedule_name
                        if clock_variant.label is not None:
                            output_dir = output_dir / clock_variant.label
                        output_dir = output_dir / f"nfe_{nfe:03d}"
                        run_arguments = [
                            "scripts/run/run_diffusers_experiment.py",
                            "--manifest",
                            manifest_path,
                            "--model-asset",
                            model_asset,
                            "--solver",
                            solver,
                            "--prompt-asset",
                            prompt_asset,
                            "--nfe",
                            str(nfe),
                            "--seed",
                            str(seed),
                            "--schedule-name",
                            display_name,
                            "--output-dir",
                            str(output_dir),
                            "--summary-csv",
                            str(summary_csv),
                            "--dtype",
                            dtype_name,
                            "--height",
                            str(image_size),
                            "--width",
                            str(image_size),
                            "--guidance-scale",
                            str(guidance_scale),
                        ]
                        if schedule_dir is not None:
                            run_arguments.extend(["--schedule-dir", str(schedule_dir)])

                        invocations.append(
                            ExperimentInvocation(
                                label=f"diffusers:{model_key}:{solver}:{display_name}:nfe_{nfe:03d}",
                                runtime_backend="diffusers",
                                run_arguments=tuple(run_arguments),
                                prepare_steps=prepare_steps,
                                output_dir=output_dir,
                                schedule_dir=schedule_dir,
                                materializable=materializable,
                                notes=tuple(notes),
                            )
                        )
    return invocations


def build_invocations(
    args: argparse.Namespace,
    experiment_config: Mapping[str, Any],
    *,
    execution_config: ExecutionConfig,
) -> list[ExperimentInvocation]:
    clock_config_path = str(experiment_config.get("clock_config", args.clock_config))
    va_clock_variants = resolve_clock_variants(experiment_config, default_clock_config_path=clock_config_path)
    schedule_clock_configs = resolve_schedule_clock_configs(
        experiment_config,
        default_va_clock_config_path=clock_config_path,
    )
    ays_config_path = str(experiment_config.get("ays_config", args.ays_config))
    models_config_path = str(experiment_config.get("models_config", args.models_config))
    backend = experiment_config["backend"]
    if backend == "pndm":
        return build_pndm_invocations(
            experiment_config,
            manifest_path=args.manifest,
            ays_config_path=ays_config_path,
            va_clock_variants=va_clock_variants,
            schedule_clock_configs=schedule_clock_configs,
            schedule_cache_root=execution_config.schedule_cache_root,
            outputs_root=args.outputs_root,
            metrics_root=args.metrics_root,
        )
    if backend == "diffusers":
        return build_diffusers_invocations(
            experiment_config,
            manifest_path=args.manifest,
            va_clock_variants=va_clock_variants,
            schedule_clock_configs=schedule_clock_configs,
            models_config_path=models_config_path,
            schedule_cache_root=execution_config.schedule_cache_root,
            outputs_root=args.outputs_root,
            metrics_root=args.metrics_root,
            dtype_name=args.dtype,
        )
    raise ValueError(f"Unsupported backend in experiment config: {backend}")


def resolve_execution_config(experiment_config: Mapping[str, Any], args: argparse.Namespace) -> ExecutionConfig:
    raw_execution = experiment_config.get("execution", {})
    if raw_execution and not isinstance(raw_execution, Mapping):
        raise TypeError("`execution` must be a mapping when provided.")

    num_gpus = int(raw_execution.get("num_gpus", experiment_config.get("num_gpus", 1)))
    if num_gpus < 1:
        raise ValueError("`execution.num_gpus` must be at least 1.")

    gpu_ids_raw = raw_execution.get("gpu_ids", experiment_config.get("gpu_ids"))
    if gpu_ids_raw is None:
        gpu_ids = tuple(range(num_gpus))
    else:
        gpu_ids = tuple(int(item) for item in gpu_ids_raw)
        if len(gpu_ids) < num_gpus:
            raise ValueError("`execution.gpu_ids` must contain at least `execution.num_gpus` entries.")
        gpu_ids = gpu_ids[:num_gpus]

    prepare_schedules_first = bool(
        raw_execution.get("prepare_schedules_first", experiment_config.get("prepare_schedules_first", True))
    )
    prepare_gpu = int(
        raw_execution.get(
            "prepare_gpu",
            raw_execution.get("schedule_build_gpu", experiment_config.get("schedule_build_gpu", gpu_ids[0])),
        )
    )
    materialize_schedules = bool(
        raw_execution.get("materialize_schedules", experiment_config.get("materialize_schedules", True))
        or args.materialize_schedules
    )
    skip_existing = bool(raw_execution.get("skip_existing", experiment_config.get("skip_existing", False)) or args.skip_existing)
    log_dir_root = Path(raw_execution.get("log_dir_root", experiment_config.get("log_dir_root", "outputs/logs")))
    schedule_cache_root = Path(
        raw_execution.get(
            "schedule_cache_root",
            experiment_config.get(
                "schedule_cache_root",
                Path("outputs/experiment_records") / str(experiment_config["name"]) / "schedules",
            ),
        )
    )

    return ExecutionConfig(
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        prepare_schedules_first=prepare_schedules_first,
        prepare_gpu=prepare_gpu,
        materialize_schedules=materialize_schedules,
        skip_existing=skip_existing,
        log_dir_root=log_dir_root,
        schedule_cache_root=schedule_cache_root,
    )


def shard_group_key(invocation: ExperimentInvocation) -> str:
    if ":nfe_" in invocation.label:
        return invocation.label.rsplit(":nfe_", 1)[0]
    return invocation.label


def shard_invocations(
    invocations: list[ExperimentInvocation],
    *,
    shard_count: int,
    shard_index: int,
) -> list[ExperimentInvocation]:
    if shard_count < 1:
        raise ValueError("--shard-count must be at least 1.")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count.")
    if shard_count == 1:
        return invocations

    group_order: dict[str, int] = {}
    selected: list[ExperimentInvocation] = []
    for invocation in invocations:
        group_key = shard_group_key(invocation)
        if group_key not in group_order:
            group_order[group_key] = len(group_order)
        if group_order[group_key] % shard_count == shard_index:
            selected.append(invocation)
    return selected


def filter_existing_invocations(invocations: list[ExperimentInvocation]) -> tuple[list[ExperimentInvocation], list[ExperimentInvocation]]:
    pending: list[ExperimentInvocation] = []
    skipped: list[ExperimentInvocation] = []
    for invocation in invocations:
        if invocation.output_dir is None:
            pending.append(invocation)
            continue
        if (resolve_repo_path(invocation.output_dir) / "run_manifest.json").exists():
            skipped.append(invocation)
            continue
        pending.append(invocation)
    return pending, skipped


def collect_prepare_steps(invocations: list[ExperimentInvocation]) -> list[PrepareStep]:
    unique_steps: dict[str, PrepareStep] = {}
    missing_external: list[str] = []

    for invocation in invocations:
        if invocation.schedule_dir is None:
            continue
        resolved_schedule_dir = resolve_repo_path(invocation.schedule_dir)
        schedule_is_current = resolved_schedule_dir.exists()
        if schedule_is_current and invocation.materializable:
            schedule_is_current = _is_current_materializable_schedule_bundle_dir(resolved_schedule_dir)
        if schedule_is_current:
            continue
        if not invocation.materializable:
            missing_external.append(f"{invocation.label}: {invocation.schedule_dir}")
            continue
        for step in invocation.prepare_steps:
            if step.key not in unique_steps:
                unique_steps[step.key] = step

    if missing_external:
        raise FileNotFoundError(
            "Missing non-materializable schedule assets:\n" + "\n".join(f"  - {item}" for item in missing_external)
        )
    return list(unique_steps.values())


def _expected_schedule_implementation_version(schedule_family: str) -> int | None:
    versions = {
        "base": BASELINE_SCHEDULE_IMPLEMENTATION_VERSION,
        "linear": BASELINE_SCHEDULE_IMPLEMENTATION_VERSION,
        "V_a": VA_SCHEDULE_IMPLEMENTATION_VERSION,
        "LCS-1": LCS_SCHEDULE_IMPLEMENTATION_VERSION,
        "LCS-2": LCS_SCHEDULE_IMPLEMENTATION_VERSION,
    }
    return versions.get(schedule_family)


def _is_current_materializable_schedule_bundle_dir(schedule_dir: Path) -> bool:
    meta_path = schedule_dir / "meta.json"
    if not meta_path.exists():
        return False
    meta = load_json(meta_path)
    schedule_family = str(meta.get("schedule_family", ""))
    expected_version = _expected_schedule_implementation_version(schedule_family)
    if expected_version is None:
        return True
    return int(meta.get("schedule_implementation_version", -1)) == expected_version


def validate_schedule_dirs(invocations: list[ExperimentInvocation]) -> None:
    missing: list[str] = []
    for invocation in invocations:
        if invocation.schedule_dir is None:
            continue
        if not resolve_repo_path(invocation.schedule_dir).exists():
            missing.append(f"{invocation.label}: {invocation.schedule_dir}")
    if missing:
        raise FileNotFoundError("Missing schedule bundles after prepare phase:\n" + "\n".join(f"  - {item}" for item in missing))


def write_schedule_cache_record(
    *,
    experiment_name: str,
    schedule_cache_root: Path,
    invocations: list[ExperimentInvocation],
) -> None:
    grouped: dict[str, dict[str, Any]] = {}
    for invocation in invocations:
        if invocation.schedule_dir is None:
            continue
        key = str(invocation.schedule_dir)
        entry = grouped.setdefault(
            key,
            {
                "schedule_dir": key,
                "source": "experiment_cache" if invocation.materializable else "external_asset",
                "exists": resolve_repo_path(invocation.schedule_dir).exists(),
                "labels": [],
            },
        )
        entry["exists"] = resolve_repo_path(invocation.schedule_dir).exists()
        entry["labels"].append(invocation.label)

    record = {
        "experiment": experiment_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schedule_cache_root": str(schedule_cache_root),
        "entries": list(grouped.values()),
    }
    dump_json(record, schedule_cache_root.parent / "schedule_cache_manifest.json")


def run_prepare_phase(
    steps: list[PrepareStep],
    *,
    runtime_config: str,
    prepare_gpu: int,
) -> None:
    if not steps:
        return
    env_overrides = {"CUDA_VISIBLE_DEVICES": str(prepare_gpu)}
    for step in steps:
        runtime_env = get_runtime_env(step.runtime_backend, runtime_config)
        print(f"[prepare] {step.key}")
        print(f"  gpu: {prepare_gpu}")
        print(f"  {command_preview(runtime_env, step.arguments)}")
        run_in_runtime_env(runtime_env, step.arguments, env_overrides=env_overrides)


def print_invocations(invocations: list[ExperimentInvocation], runtime_config: str) -> None:
    for index, invocation in enumerate(invocations, start=1):
        runtime_env = get_runtime_env(invocation.runtime_backend, runtime_config)
        print(f"[{index}] {invocation.label}")
        print(f"  env: {runtime_env.name}")
        if invocation.prepare_steps:
            for step in invocation.prepare_steps:
                print(f"  prepare: {command_preview(runtime_env, step.arguments)}")
        print(f"  run: {command_preview(runtime_env, invocation.run_arguments)}")
        if invocation.notes:
            print(f"  notes: {', '.join(invocation.notes)}")


def execute_invocations(
    invocations: list[ExperimentInvocation],
    *,
    runtime_config: str,
    materialize_schedules: bool,
) -> None:
    completed_prepare_keys: set[str] = set()
    for invocation in invocations:
        runtime_env = get_runtime_env(invocation.runtime_backend, runtime_config)
        if invocation.schedule_dir is not None and not resolve_repo_path(invocation.schedule_dir).exists():
            if invocation.materializable and materialize_schedules:
                for step in invocation.prepare_steps:
                    if step.key in completed_prepare_keys:
                        continue
                    print(f"[prepare] {invocation.label}")
                    print(f"  {command_preview(runtime_env, step.arguments)}")
                    run_in_runtime_env(runtime_env, step.arguments)
                    completed_prepare_keys.add(step.key)
                if not resolve_repo_path(invocation.schedule_dir).exists():
                    raise FileNotFoundError(
                        f"Schedule generation did not produce the expected bundle for {invocation.label}: {invocation.schedule_dir}"
                    )
            elif invocation.materializable:
                raise FileNotFoundError(
                    f"Missing materializable schedule for {invocation.label}: {invocation.schedule_dir}. "
                    "Enable config execution.materialize_schedules or pass --materialize-schedules."
                )
            else:
                raise FileNotFoundError(
                    f"Missing external schedule asset for {invocation.label}: {invocation.schedule_dir}."
                )

        print(f"[run] {invocation.label}")
        print(f"  {command_preview(runtime_env, invocation.run_arguments)}")
        run_in_runtime_env(runtime_env, invocation.run_arguments)


def build_child_command(args: argparse.Namespace, *, shard_count: int, shard_index: int, skip_existing: bool) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--experiment-config",
        args.experiment_config,
        "--manifest",
        args.manifest,
        "--runtime-config",
        args.runtime_config,
        "--models-config",
        args.models_config,
        "--clock-config",
        args.clock_config,
        "--ays-config",
        args.ays_config,
        "--outputs-root",
        args.outputs_root,
        "--metrics-root",
        args.metrics_root,
        "--dtype",
        args.dtype,
        "--execute",
        "--skip-preview",
        "--distributed-child",
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if skip_existing:
        command.append("--skip-existing")
    return command


def dispatch_multi_gpu(
    args: argparse.Namespace,
    experiment_config: Mapping[str, Any],
    *,
    execution_config: ExecutionConfig,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = resolve_repo_path(execution_config.log_dir_root / f"{experiment_config['name']}_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dispatch] experiment={experiment_config['name']} num_gpus={execution_config.num_gpus} gpu_ids={list(execution_config.gpu_ids)}")
    print(f"[dispatch] logs={log_dir}")

    children: list[tuple[subprocess.Popen[str], Any, Path, int, int]] = []
    for shard_index, gpu_id in enumerate(execution_config.gpu_ids):
        command = build_child_command(
            args,
            shard_count=execution_config.num_gpus,
            shard_index=shard_index,
            skip_existing=execution_config.skip_existing,
        )
        env = build_subprocess_env(env_overrides={"CUDA_VISIBLE_DEVICES": str(gpu_id)})
        log_path = log_dir / f"gpu{gpu_id}.shard{shard_index}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(repo_root()),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        children.append((process, handle, log_path, gpu_id, shard_index))
        print(f"[dispatch] shard={shard_index} gpu={gpu_id} log={log_path}")

    failed: list[str] = []
    for process, handle, log_path, gpu_id, shard_index in children:
        return_code = process.wait()
        handle.close()
        if return_code == 0:
            print(f"[dispatch] completed shard={shard_index} gpu={gpu_id}")
        else:
            failed.append(f"gpu={gpu_id} shard={shard_index} log={log_path}")

    if failed:
        raise RuntimeError("One or more GPU shards failed:\n" + "\n".join(f"  - {item}" for item in failed))


def main() -> None:
    args = parse_args()
    experiment_config = load_experiment_config(args.experiment_config)
    execution_config = resolve_execution_config(experiment_config, args)

    invocations = build_invocations(args, experiment_config, execution_config=execution_config)
    if execution_config.skip_existing:
        invocations, skipped = filter_existing_invocations(invocations)
        if skipped:
            print(f"[skip-existing] skipped {len(skipped)} completed invocations")

    if args.limit is not None:
        invocations = invocations[: args.limit]
    if not invocations:
        print("No pending invocations.")
        return

    if args.execute and not args.distributed_child and args.shard_count == 1 and execution_config.prepare_schedules_first:
        prepare_steps = collect_prepare_steps(invocations)
        if prepare_steps and not execution_config.materialize_schedules:
            raise FileNotFoundError(
                "This experiment requires materializable schedule assets, but execution.materialize_schedules is disabled."
            )
        run_prepare_phase(prepare_steps, runtime_config=args.runtime_config, prepare_gpu=execution_config.prepare_gpu)
        validate_schedule_dirs(invocations)
        write_schedule_cache_record(
            experiment_name=str(experiment_config["name"]),
            schedule_cache_root=execution_config.schedule_cache_root,
            invocations=invocations,
        )

    if args.execute and not args.distributed_child and args.shard_count == 1 and execution_config.num_gpus > 1:
        dispatch_multi_gpu(args, experiment_config, execution_config=execution_config)
        return

    invocations = shard_invocations(invocations, shard_count=args.shard_count, shard_index=args.shard_index)
    if not invocations:
        print("No pending invocations for this shard.")
        return

    if not args.skip_preview:
        print_invocations(invocations, args.runtime_config)
    if args.execute:
        execute_invocations(
            invocations,
            runtime_config=args.runtime_config,
            materialize_schedules=execution_config.materialize_schedules and not execution_config.prepare_schedules_first,
        )
        if not args.distributed_child and args.shard_count == 1:
            write_schedule_cache_record(
                experiment_name=str(experiment_config["name"]),
                schedule_cache_root=execution_config.schedule_cache_root,
                invocations=invocations,
            )


if __name__ == "__main__":
    main()

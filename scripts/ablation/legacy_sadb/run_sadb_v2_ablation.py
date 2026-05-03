#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import ensure_dir


BASE_CLOCK: dict[str, Any] = {
    "family": "SADB",
    "calibration_solver": "euler",
    "model_output_type": "epsilon",
    "physical_grid_size": 65,
    "physical_grid_mode": "scheduler_sigmas",
    "pilot_batch_size": 8,
    "pilot_num_batches": 8,
    "pilot_observation_microbatch": 8,
    "pilot_prompt_asset": "diffusers_ablation_prompts",
    "smoothing_window": 3,
    "epsilon": 1.0e-12,
    "cache_path": "outputs/sadb_v2_ablation/cache/sadb_profiles",
    "target_nfes": [10],
}


VARIANTS: dict[str, dict[str, Any]] = {
    "A": {
        "coordinate_domain": "sigma",
        "prior_schedule": "none",
        "prior_blend": 0.0,
        "density_temperature": 1.0,
        "q_min": 1.05,
        "q_max": 6.0,
        "q_shrinkage": 0.0,
        "defect_reduce": "rms",
        "max_dt_factor": 1.5,
        "max_neighbor_dt_ratio": 4.0,
        "sde_noise_kappa": 0.0,
    },
    "B": {
        "coordinate_domain": "lambda",
        "prior_schedule": "none",
        "prior_blend": 0.0,
        "density_temperature": 1.0,
        "q_min": 1.05,
        "q_max": 6.0,
        "q_shrinkage": 0.0,
        "defect_reduce": "rms",
        "max_dt_factor": 1.5,
        "max_neighbor_dt_ratio": 4.0,
        "sde_noise_kappa": 0.0,
    },
    "C": {
        "coordinate_domain": "lambda",
        "prior_schedule": "ays",
        "prior_blend": 0.25,
        "density_temperature": 0.6,
        "q_min": 1.05,
        "q_max": 6.0,
        "q_shrinkage": 0.0,
        "defect_reduce": "rms",
        "max_dt_factor": 1.5,
        "max_neighbor_dt_ratio": 4.0,
        "sde_noise_kappa": 0.0,
    },
    "D": {
        "coordinate_domain": "lambda",
        "prior_schedule": "ays",
        "prior_blend": 0.25,
        "density_temperature": 0.6,
        "q_prior": 3.0,
        "q_shrinkage": 0.5,
        "q_min": 1.5,
        "q_max": 4.0,
        "sde_q_prior": 2.2,
        "sde_q_shrinkage": 0.6,
        "sde_q_min": 1.3,
        "sde_q_max": 3.5,
        "defect_reduce": "rms",
        "max_dt_factor": 1.5,
        "max_neighbor_dt_ratio": 4.0,
        "sde_noise_kappa": 0.0,
    },
    "E": {
        "coordinate_domain": "lambda",
        "prior_schedule": "ays",
        "prior_blend": 0.25,
        "density_temperature": 0.6,
        "q_prior": 3.0,
        "q_shrinkage": 0.5,
        "q_min": 1.5,
        "q_max": 4.0,
        "sde_q_prior": 2.2,
        "sde_q_shrinkage": 0.6,
        "sde_q_min": 1.3,
        "sde_q_max": 3.5,
        "defect_reduce": "rms",
        "max_dt_factor": 1.5,
        "max_neighbor_dt_ratio": 1.8,
        "sde_noise_kappa": 0.0,
    },
    "F": {
        "coordinate_domain": "lambda",
        "prior_schedule": "ays",
        "prior_blend": 0.25,
        "density_temperature": 0.6,
        "q_prior": 3.0,
        "q_shrinkage": 0.5,
        "q_min": 1.5,
        "q_max": 4.0,
        "sde_q_prior": 2.2,
        "sde_q_shrinkage": 0.6,
        "sde_q_min": 1.3,
        "sde_q_max": 3.5,
        "defect_reduce": "quantile",
        "defect_quantile": 0.75,
        "max_dt_factor": 1.5,
        "max_neighbor_dt_ratio": 1.8,
        "sde_noise_kappa": 0.3,
        "sde_brownian_bridge": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and run cumulative SADB-v2 ablation variants.")
    parser.add_argument("--variants", default="A,B,C,D,E,F")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-dir", default="outputs/sadb_v2_ablation/generated_configs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_variant_configs(variant: str, output_dir: Path, *, num_gpus: int, seeds: list[int]) -> Path:
    clock = dict(BASE_CLOCK)
    clock.update(VARIANTS[variant])
    clock_path = output_dir / f"SADB_v2_variant_{variant}.yaml"
    write_yaml(clock_path, {"clock": clock})

    experiment = {
        "name": f"sd15_sdxl_sadb_v2_ablation_{variant}",
        "backend": "diffusers",
        "base_config": {
            "prompt_asset": "diffusers_ablation_prompts",
            "seeds": seeds,
            "num_gpus": int(num_gpus),
            "prepare_schedules_first": True,
            "materialize_schedules": True,
        },
        "models": ["stable_diffusion_15", "sdxl"],
        "solvers": ["dpm_solver_pp", "sde_dpm_solver_pp"],
        "schedules": ["base", "AYS", "SADB"],
        "schedule_clock_configs": {"SADB": str(clock_path)},
        "eval_nfes": [10],
        "metrics": ["clipscore", "imagereward"],
    }
    experiment_path = output_dir / f"sd15_sdxl_sadb_v2_ablation_{variant}.yaml"
    write_yaml(experiment_path, experiment)
    return experiment_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    variants = [item.strip().upper() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in variants if item not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    seeds = [int(item) for item in args.seeds.split(",") if item]
    for variant in variants:
        experiment_path = build_variant_configs(variant, output_dir, num_gpus=args.num_gpus, seeds=seeds)
        command = [
            args.python,
            "scripts/run/run_experiment_config.py",
            "--experiment-config",
            str(experiment_path),
            "--skip-preview",
        ]
        if args.execute:
            command.append("--execute")
            command.append("--skip-existing")
        command.extend(args.extra_arg)
        print("[sadb-v2-ablation]", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

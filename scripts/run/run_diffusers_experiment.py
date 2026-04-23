#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runners.diffusers_experiment import run_diffusers_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a project-owned modern diffusers experiment.")
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--model-asset", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--prompt-asset", default="diffusers_smoke_prompts")
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-csv", default="outputs/metrics/diffusers_summary.csv")
    parser.add_argument("--schedule-name", default="base")
    parser.add_argument("--schedule-dir")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diffusers_experiment(
        manifest_path=args.manifest,
        model_asset_key=args.model_asset,
        solver_name=args.solver,
        prompt_asset_or_path=args.prompt_asset,
        num_inference_steps=args.nfe,
        seed=args.seed,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        schedule_name=args.schedule_name,
        schedule_dir=args.schedule_dir,
        dtype_name=args.dtype,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runners.pndm_experiment import run_pndm_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a project-owned PNDM experiment.")
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-asset", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-csv", default="outputs/metrics/pndm_summary.csv")
    parser.add_argument("--schedule-name", default="base")
    parser.add_argument("--schedule-dir")
    parser.add_argument("--compute-fid", action="store_true", default=False)
    parser.add_argument("--discard-samples", action="store_true", default=False)
    parser.add_argument("--preview-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pndm_experiment(
        manifest_path=args.manifest,
        dataset_config_path=args.dataset_config,
        model_asset_key=args.model_asset,
        solver_name=args.solver,
        num_inference_steps=args.nfe,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        schedule_name=args.schedule_name,
        schedule_dir=args.schedule_dir,
        compute_fid_score=args.compute_fid,
        save_samples=not args.discard_samples,
        preview_samples=args.preview_samples,
    )


if __name__ == "__main__":
    main()

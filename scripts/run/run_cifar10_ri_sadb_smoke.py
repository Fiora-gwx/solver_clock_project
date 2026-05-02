#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_json, load_yaml, resolve_repo_path


SMOKE_FIELDS = (
    "dataset",
    "schedule",
    "eta",
    "beta",
    "nfe",
    "seed",
    "fid_or_proxy",
    "sample_count",
    "runtime_sec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or expand the CIFAR-10 RI-SADB smoke experiment.")
    parser.add_argument("--experiment-config", default="configs/experiments/ri_sadb_cifar10_smoke.yaml")
    parser.add_argument("--outputs-root", default="outputs/samples")
    parser.add_argument("--metrics-root", default="outputs/metrics")
    parser.add_argument("--smoke-csv", default="outputs/metrics/ri_sadb_cifar10_smoke_status.csv")
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--materialize-schedules", action="store_true", default=False)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _run_experiment_config(args: argparse.Namespace) -> float:
    command = [
        sys.executable,
        "scripts/run/run_experiment_config.py",
        "--experiment-config",
        args.experiment_config,
        "--outputs-root",
        args.outputs_root,
        "--metrics-root",
        args.metrics_root,
    ]
    if args.execute:
        command.append("--execute")
    if args.materialize_schedules:
        command.append("--materialize-schedules")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    start = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return time.perf_counter() - start


def _variant_params(schedule: str, output_dir: str) -> tuple[str, str]:
    if not schedule.startswith("RI_SADB"):
        return "", ""
    manifest_path = Path(output_dir) / "run_manifest.json"
    if not manifest_path.exists():
        return "", ""
    payload = load_json(manifest_path)
    schedule_dir = payload.get("schedule_dir")
    if not schedule_dir:
        return "", ""
    meta_path = resolve_repo_path(schedule_dir) / "meta.json"
    if not meta_path.exists():
        return "", ""
    meta = load_json(meta_path)
    return str(meta.get("eta", "")), str(meta.get("beta", ""))


def write_smoke_csv(args: argparse.Namespace, *, runtime_sec: float) -> Path:
    experiment = load_yaml(args.experiment_config)
    metrics_path = resolve_repo_path(args.metrics_root) / f"{experiment['name']}.csv"
    output_path = resolve_repo_path(args.smoke_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                eta, beta = _variant_params(str(row.get("schedule", "")), str(row.get("output_dir", "")))
                rows.append(
                    {
                        "dataset": str(row.get("dataset", "")),
                        "schedule": str(row.get("schedule", "")),
                        "eta": eta,
                        "beta": beta,
                        "nfe": str(row.get("nfe", "")),
                        "seed": str(row.get("seed", "")),
                        "fid_or_proxy": str(row.get("fid", "")),
                        "sample_count": str(row.get("num_samples", "")),
                        "runtime_sec": f"{runtime_sec:.3f}",
                    }
                )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SMOKE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    args = parse_args()
    runtime_sec = _run_experiment_config(args)
    smoke_csv = write_smoke_csv(args, runtime_sec=runtime_sec)
    print(f"[ri-sadb-smoke] wrote {smoke_csv}")


if __name__ == "__main__":
    main()

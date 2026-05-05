#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent

DEFAULT_EXISTING_DETAIL = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_fid_detail_seeds0_1_2.csv"
DEFAULT_REUSE_SOURCE = (
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_50k_reuse_seed0_schedule_seeds1_2/metrics/"
    / "gpde_pndm_cifar10_50k_reuse_seed0_schedule_seeds1_2.csv"
)
DEFAULT_DETAIL = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_reuse_seed0_schedule_detail.csv"
DEFAULT_AGGREGATE = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_reuse_seed0_schedule_aggregate.csv"
DEFAULT_ROUNDING = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_reuse_seed0_schedule_rounding.csv"
DEFAULT_TABLE = PAPER_ROOT / "tables/cifar10_pndm_euler_50k_reuse_seed0_schedule.tex"

METHODS = ("base", "D-GPDE-seed-specific", "D-GPDE-reuse-seed0")
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}

DETAIL_FIELDS = [
    "evidence_level",
    "backend",
    "dataset",
    "model",
    "solver",
    "method",
    "nfe",
    "seed",
    "num_samples",
    "metric",
    "metric_direction",
    "fid",
    "source_csv",
    "source_output_dir",
    "source_schedule_dir",
    "reused_from_seed",
    "rounded_timesteps_equal_to_seed_specific",
    "max_abs_raw_timestep_diff",
]

AGGREGATE_FIELDS = [
    "evidence_level",
    "backend",
    "dataset",
    "model",
    "solver",
    "nfe",
    "seed_count",
    "seeds",
    "num_samples",
    "metric",
    "metric_direction",
    "base_fid_mean",
    "base_fid_sem",
    "dgpde_seed_specific_fid_mean",
    "dgpde_seed_specific_fid_sem",
    "dgpde_reuse_seed0_fid_mean",
    "dgpde_reuse_seed0_fid_sem",
    "reuse_minus_seed_specific_fid_mean",
    "reuse_minus_seed_specific_fid_sem",
    "rounded_timesteps_all_equal",
]

ROUNDING_FIELDS = [
    "nfe",
    "seed",
    "reused_schedule_dir",
    "seed_specific_schedule_dir",
    "max_abs_raw_timestep_diff",
    "rounded_timesteps_equal",
    "rounded_timesteps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CIFAR-10 seed-0 schedule reuse validation.")
    parser.add_argument("--existing-detail-csv", type=Path, default=DEFAULT_EXISTING_DETAIL)
    parser.add_argument("--reuse-source-csv", type=Path, default=DEFAULT_REUSE_SOURCE)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--aggregate-csv", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--rounding-csv", type=Path, default=DEFAULT_ROUNDING)
    parser.add_argument("--table-tex", type=Path, default=DEFAULT_TABLE)
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.6f}"


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_repo_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_rounded_timesteps(schedule_dir: str) -> np.ndarray:
    timesteps = np.load(resolve_repo_path(schedule_dir) / "timesteps.npy").astype(np.float64)
    return np.round(timesteps).astype(np.int64)


def timestep_diff(reused_schedule_dir: str, seed_specific_schedule_dir: str) -> tuple[float, bool, str]:
    reused = np.load(resolve_repo_path(reused_schedule_dir) / "timesteps.npy").astype(np.float64)
    specific = np.load(resolve_repo_path(seed_specific_schedule_dir) / "timesteps.npy").astype(np.float64)
    if reused.shape != specific.shape:
        raise ValueError(
            f"Schedule shapes differ: {reused_schedule_dir} {reused.shape} vs "
            f"{seed_specific_schedule_dir} {specific.shape}"
        )
    rounded_reused = np.round(reused).astype(np.int64)
    rounded_specific = np.round(specific).astype(np.int64)
    max_abs = float(np.max(np.abs(reused - specific)))
    equal = bool(np.array_equal(rounded_reused, rounded_specific))
    rounded_text = " ".join(str(int(value)) for value in rounded_reused)
    return max_abs, equal, rounded_text


def build_detail_rows(
    *,
    existing_detail_csv: Path,
    existing_rows: list[dict[str, str]],
    reuse_source_csv: Path,
    reuse_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    existing_by_key: dict[tuple[int, int, str], dict[str, str]] = {}
    for row in existing_rows:
        if row.get("dataset") != "cifar10" or row.get("solver") != "euler":
            continue
        method = row.get("method")
        if method not in {"base", "D-GPDE"}:
            continue
        nfe = int(row["nfe"])
        seed = int(row["seed"])
        if nfe not in {10, 20} or seed not in {1, 2}:
            continue
        existing_by_key[(nfe, seed, method)] = row

    reuse_by_key: dict[tuple[int, int], dict[str, str]] = {}
    for row in reuse_rows:
        if row.get("status") != "OK":
            continue
        if row.get("dataset") != "cifar10" or row.get("solver") != "euler":
            continue
        if row.get("schedule") != "D-GPDE-reuse-seed0" or row.get("metric_name") != "fid":
            continue
        nfe = int(row["nfe"])
        seed = int(row["seed"])
        if nfe not in {10, 20} or seed not in {1, 2}:
            continue
        reuse_by_key[(nfe, seed)] = row

    expected_existing = {
        (nfe, seed, method)
        for nfe in (10, 20)
        for seed in (1, 2)
        for method in ("base", "D-GPDE")
    }
    expected_reuse = {(nfe, seed) for nfe in (10, 20) for seed in (1, 2)}
    missing_existing = sorted(expected_existing - set(existing_by_key))
    missing_reuse = sorted(expected_reuse - set(reuse_by_key))
    if missing_existing or missing_reuse:
        raise ValueError(f"Incomplete reuse validation inputs: missing_existing={missing_existing} missing_reuse={missing_reuse}")

    details: list[dict[str, str]] = []
    rounding_rows: list[dict[str, str]] = []
    for nfe in (10, 20):
        for seed in (1, 2):
            base = existing_by_key[(nfe, seed, "base")]
            specific = existing_by_key[(nfe, seed, "D-GPDE")]
            reuse = reuse_by_key[(nfe, seed)]
            max_abs, rounded_equal, rounded_text = timestep_diff(reuse["schedule_dir"], specific["source_schedule_dir"])
            rounding_rows.append(
                {
                    "nfe": str(nfe),
                    "seed": str(seed),
                    "reused_schedule_dir": reuse["schedule_dir"],
                    "seed_specific_schedule_dir": specific["source_schedule_dir"],
                    "max_abs_raw_timestep_diff": fmt(max_abs),
                    "rounded_timesteps_equal": str(rounded_equal),
                    "rounded_timesteps": rounded_text,
                }
            )

            details.append(copy_existing_detail(base, existing_detail_csv, "base"))
            details.append(copy_existing_detail(specific, existing_detail_csv, "D-GPDE-seed-specific"))
            details.append(
                {
                    "evidence_level": "actual_seed0_schedule_reuse_50k_fid",
                    "backend": reuse.get("backend", ""),
                    "dataset": reuse.get("dataset", ""),
                    "model": reuse.get("model", ""),
                    "solver": reuse.get("solver", ""),
                    "method": "D-GPDE-reuse-seed0",
                    "nfe": str(nfe),
                    "seed": str(seed),
                    "num_samples": reuse.get("num_samples", ""),
                    "metric": "FID",
                    "metric_direction": "lower_is_better",
                    "fid": fmt(float(reuse["fid"])),
                    "source_csv": rel(reuse_source_csv),
                    "source_output_dir": reuse.get("output_dir", ""),
                    "source_schedule_dir": reuse.get("schedule_dir", ""),
                    "reused_from_seed": "0",
                    "rounded_timesteps_equal_to_seed_specific": str(rounded_equal),
                    "max_abs_raw_timestep_diff": fmt(max_abs),
                }
            )

    details.sort(key=lambda item: (int(item["nfe"]), int(item["seed"]), METHOD_ORDER[item["method"]]))
    return details, rounding_rows


def copy_existing_detail(row: dict[str, str], source_csv: Path, method: str) -> dict[str, str]:
    return {
        "evidence_level": "actual_seed0_schedule_reuse_50k_fid",
        "backend": row.get("backend", ""),
        "dataset": row.get("dataset", ""),
        "model": row.get("model", ""),
        "solver": row.get("solver", ""),
        "method": method,
        "nfe": row.get("nfe", ""),
        "seed": row.get("seed", ""),
        "num_samples": row.get("num_samples", ""),
        "metric": "FID",
        "metric_direction": "lower_is_better",
        "fid": fmt(float(row["fid"])),
        "source_csv": rel(source_csv),
        "source_output_dir": row.get("source_output_dir", ""),
        "source_schedule_dir": row.get("source_schedule_dir", ""),
        "reused_from_seed": "",
        "rounded_timesteps_equal_to_seed_specific": "",
        "max_abs_raw_timestep_diff": "",
    }


def aggregate_rows(details: list[dict[str, str]], rounding_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in details:
        grouped[int(row["nfe"])].append(row)

    aggregate: list[dict[str, str]] = []
    for nfe in sorted(grouped):
        group = grouped[nfe]
        seeds = sorted({int(row["seed"]) for row in group})
        by_method = {
            method: [float(row["fid"]) for row in group if row["method"] == method]
            for method in METHODS
        }
        if any(len(values) != len(seeds) for values in by_method.values()):
            raise ValueError(f"Incomplete aggregate group for NFE {nfe}: {by_method}")
        specific_by_seed = {
            int(row["seed"]): float(row["fid"])
            for row in group
            if row["method"] == "D-GPDE-seed-specific"
        }
        reuse_by_seed = {
            int(row["seed"]): float(row["fid"])
            for row in group
            if row["method"] == "D-GPDE-reuse-seed0"
        }
        gaps = [reuse_by_seed[seed] - specific_by_seed[seed] for seed in seeds]
        first = group[0]
        rounded_equal = all(row["rounded_timesteps_equal"] == "True" for row in rounding_rows if int(row["nfe"]) == nfe)
        aggregate.append(
            {
                "evidence_level": "actual_seed0_schedule_reuse_50k_fid",
                "backend": first["backend"],
                "dataset": first["dataset"],
                "model": first["model"],
                "solver": first["solver"],
                "nfe": str(nfe),
                "seed_count": str(len(seeds)),
                "seeds": ",".join(str(seed) for seed in seeds),
                "num_samples": first["num_samples"],
                "metric": "FID",
                "metric_direction": "lower_is_better",
                "base_fid_mean": fmt(statistics.mean(by_method["base"])),
                "base_fid_sem": fmt(sem(by_method["base"])),
                "dgpde_seed_specific_fid_mean": fmt(statistics.mean(by_method["D-GPDE-seed-specific"])),
                "dgpde_seed_specific_fid_sem": fmt(sem(by_method["D-GPDE-seed-specific"])),
                "dgpde_reuse_seed0_fid_mean": fmt(statistics.mean(by_method["D-GPDE-reuse-seed0"])),
                "dgpde_reuse_seed0_fid_sem": fmt(sem(by_method["D-GPDE-reuse-seed0"])),
                "reuse_minus_seed_specific_fid_mean": fmt(statistics.mean(gaps)),
                "reuse_minus_seed_specific_fid_sem": fmt(sem(gaps)),
                "rounded_timesteps_all_equal": str(rounded_equal),
            }
        )
    return aggregate


def write_table(aggregate: list[dict[str, str]], table_path: Path) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Actual seed-0 schedule reuse for PNDM/CIFAR-10 with",
        "  \\texttt{pndm\\_model\\_ddim\\_cifar10}, Euler solver, NFE 10 and 20,",
        "  seeds 1 and 2, and 50k generated images per seed. FID is",
        "  lower-is-better. The reuse gap is seed-0-reuse FID minus seed-specific",
        "  \\method{} FID. For these rows the continuous schedules differ by",
        "  less than 0.002 timestep units and round to the same integer",
        "  PNDM Euler execution grids.}",
        "  \\label{tab:cifar10-pndm-euler-50k-reuse-seed0-schedule}",
        "  \\begin{tabular}{rrrrr}",
        "    \\toprule",
        "    NFE & base & per-seed \\method{} & reused \\method{} & reuse gap \\\\",
        "    \\midrule",
    ]
    for row in aggregate:
        lines.append(
            "    "
            f"{row['nfe']} & "
            f"{float(row['base_fid_mean']):.2f} $\\pm$ {float(row['base_fid_sem']):.2f} & "
            f"{float(row['dgpde_seed_specific_fid_mean']):.2f} $\\pm$ {float(row['dgpde_seed_specific_fid_sem']):.2f} & "
            f"{float(row['dgpde_reuse_seed0_fid_mean']):.2f} $\\pm$ {float(row['dgpde_reuse_seed0_fid_sem']):.2f} & "
            f"{float(row['reuse_minus_seed_specific_fid_mean']):.3f} $\\pm$ "
            f"{float(row['reuse_minus_seed_specific_fid_sem']):.3f} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    table_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    existing_rows = read_csv(args.existing_detail_csv)
    reuse_rows = read_csv(args.reuse_source_csv)
    details, rounding_rows = build_detail_rows(
        existing_detail_csv=args.existing_detail_csv,
        existing_rows=existing_rows,
        reuse_source_csv=args.reuse_source_csv,
        reuse_rows=reuse_rows,
    )
    aggregate = aggregate_rows(details, rounding_rows)
    write_csv(args.detail_csv, DETAIL_FIELDS, details)
    write_csv(args.aggregate_csv, AGGREGATE_FIELDS, aggregate)
    write_csv(args.rounding_csv, ROUNDING_FIELDS, rounding_rows)
    write_table(aggregate, args.table_tex)
    print(f"[aggregate-cifar-reuse] detail_rows={len(details)} detail={args.detail_csv}")
    print(f"[aggregate-cifar-reuse] aggregate_rows={len(aggregate)} aggregate={args.aggregate_csv}")
    print(f"[aggregate-cifar-reuse] rounding_rows={len(rounding_rows)} rounding={args.rounding_csv}")
    print(f"[aggregate-cifar-reuse] table={args.table_tex}")


if __name__ == "__main__":
    main()

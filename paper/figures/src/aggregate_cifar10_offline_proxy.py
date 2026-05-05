#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent

MAIN_DETAIL = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_fid_detail_seeds0_1_2.csv"
OFFLINE_SOURCE = (
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_50k_offline_proxy_baseline_seeds0_1_2/metrics/"
    / "gpde_pndm_cifar10_50k_offline_proxy_baseline_seeds0_1_2.csv"
)
OUT_DETAIL = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_offline_proxy_detail.csv"
OUT_AGGREGATE = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_offline_proxy_aggregate.csv"
OUT_TABLE = PAPER_ROOT / "tables/cifar10_pndm_euler_50k_offline_proxy.tex"

METHODS = ("base", "Karras", "D-GPDE", "offline-proxy")
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
]

AGGREGATE_FIELDS = [
    "evidence_level",
    "backend",
    "dataset",
    "model",
    "solver",
    "method",
    "nfe",
    "seed_count",
    "seeds",
    "num_samples",
    "metric",
    "metric_direction",
    "fid_mean",
    "fid_std",
    "fid_sem",
]


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


def fmt(value: float) -> str:
    return f"{value:.6f}"


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def copy_main_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "evidence_level": "offline_proxy_50k_fid",
        "backend": row.get("backend", ""),
        "dataset": row.get("dataset", ""),
        "model": row.get("model", ""),
        "solver": row.get("solver", ""),
        "method": row.get("method", ""),
        "nfe": row.get("nfe", ""),
        "seed": row.get("seed", ""),
        "num_samples": row.get("num_samples", ""),
        "metric": "FID",
        "metric_direction": "lower_is_better",
        "fid": fmt(float(row["fid"])),
        "source_csv": row.get("source_csv", ""),
        "source_output_dir": row.get("source_output_dir", ""),
        "source_schedule_dir": row.get("source_schedule_dir", ""),
    }


def load_detail_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(MAIN_DETAIL):
        if row.get("method") not in {"base", "Karras", "D-GPDE"}:
            continue
        if int(row["nfe"]) not in {10, 20} or int(row["seed"]) not in {0, 1, 2}:
            continue
        rows.append(copy_main_row(row))

    for row in read_csv(OFFLINE_SOURCE):
        if row.get("status") != "OK" or row.get("schedule") != "offline-proxy":
            continue
        rows.append(
            {
                "evidence_level": "offline_proxy_50k_fid",
                "backend": row.get("backend", ""),
                "dataset": row.get("dataset", ""),
                "model": row.get("model", ""),
                "solver": row.get("solver", ""),
                "method": "offline-proxy",
                "nfe": str(int(row["nfe"])),
                "seed": str(int(row["seed"])),
                "num_samples": row.get("num_samples", ""),
                "metric": "FID",
                "metric_direction": "lower_is_better",
                "fid": fmt(float(row["fid"])),
                "source_csv": rel(OFFLINE_SOURCE),
                "source_output_dir": row.get("output_dir", ""),
                "source_schedule_dir": row.get("schedule_dir", ""),
            }
        )

    expected = {(nfe, seed, method) for nfe in (10, 20) for seed in (0, 1, 2) for method in METHODS}
    actual = {(int(row["nfe"]), int(row["seed"]), row["method"]) for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Unexpected offline-proxy coverage. missing={missing} extra={extra}")
    rows.sort(key=lambda item: (int(item["nfe"]), int(item["seed"]), METHOD_ORDER[item["method"]]))
    return rows


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["nfe"]), row["method"])].append(row)
    output: list[dict[str, str]] = []
    for (nfe, method), group in sorted(grouped.items(), key=lambda item: (item[0][0], METHOD_ORDER[item[0][1]])):
        values = [float(row["fid"]) for row in group]
        seeds = sorted(int(row["seed"]) for row in group)
        first = group[0]
        output.append(
            {
                "evidence_level": "offline_proxy_50k_fid",
                "backend": first["backend"],
                "dataset": first["dataset"],
                "model": first["model"],
                "solver": first["solver"],
                "method": method,
                "nfe": str(nfe),
                "seed_count": str(len(seeds)),
                "seeds": ",".join(str(seed) for seed in seeds),
                "num_samples": first["num_samples"],
                "metric": "FID",
                "metric_direction": "lower_is_better",
                "fid_mean": fmt(statistics.mean(values)),
                "fid_std": fmt(std(values)),
                "fid_sem": fmt(sem(values)),
            }
        )
    return output


def write_table(aggregate_rows: list[dict[str, str]]) -> None:
    by_key = {(int(row["nfe"]), row["method"]): row for row in aggregate_rows}
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Retained lightweight offline-proxy CIFAR-10 baseline.",
        "  Rows report PNDM/CIFAR-10 Euler FID, mean $\\pm$ standard error over",
        "  seeds 0, 1, and 2 with 50k generated images per seed. The",
        "  offline-proxy schedule is a small-budget project-owned hierarchical",
        "  coordinate-search run, not the published AYS schedule. Lower FID is",
        "  better.}",
        "  \\label{tab:cifar10-pndm-euler-50k-offline-proxy}",
        "  \\begin{tabular}{rrrrr}",
        "    \\toprule",
        "    NFE & base & Karras & \\method{} & offline-proxy \\\\",
        "    \\midrule",
    ]
    for nfe in (10, 20):
        lines.append(
            "    "
            f"{nfe} & "
            f"{float(by_key[(nfe, 'base')]['fid_mean']):.2f} $\\pm$ {float(by_key[(nfe, 'base')]['fid_sem']):.2f} & "
            f"{float(by_key[(nfe, 'Karras')]['fid_mean']):.2f} $\\pm$ {float(by_key[(nfe, 'Karras')]['fid_sem']):.2f} & "
            f"{float(by_key[(nfe, 'D-GPDE')]['fid_mean']):.2f} $\\pm$ {float(by_key[(nfe, 'D-GPDE')]['fid_sem']):.2f} & "
            f"{float(by_key[(nfe, 'offline-proxy')]['fid_mean']):.2f} $\\pm$ {float(by_key[(nfe, 'offline-proxy')]['fid_sem']):.2f} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    OUT_TABLE.write_text("\n".join(lines))


def copy_schedule_metadata(rows: list[dict[str, str]]) -> None:
    destination = OUT_DETAIL.parent
    schedule_dirs = sorted({Path(row["source_schedule_dir"]) for row in rows if row["method"] == "offline-proxy"})
    for schedule_dir in schedule_dirs:
        nfe = schedule_dir.name.replace("nfe_", "nfe")
        target_prefix = destination / f"cifar10_pndm_euler_50k_offline_proxy_schedule_{nfe}"
        for filename in ("meta.json", "timesteps.npy", "time_grid.npy"):
            source = REPO_ROOT / schedule_dir / filename
            if source.exists():
                shutil.copy2(source, target_prefix.with_name(f"{target_prefix.name}_{filename}"))


def main() -> None:
    rows = load_detail_rows()
    aggregate_rows = aggregate(rows)
    write_csv(OUT_DETAIL, DETAIL_FIELDS, rows)
    write_csv(OUT_AGGREGATE, AGGREGATE_FIELDS, aggregate_rows)
    write_table(aggregate_rows)
    copy_schedule_metadata(rows)
    print(f"[offline-proxy] detail_rows={len(rows)} detail={OUT_DETAIL}")
    print(f"[offline-proxy] aggregate_rows={len(aggregate_rows)} aggregate={OUT_AGGREGATE}")
    print(f"[offline-proxy] table={OUT_TABLE}")


if __name__ == "__main__":
    main()

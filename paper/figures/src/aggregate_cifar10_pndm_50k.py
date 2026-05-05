#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent

DEFAULT_SOURCES = [
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_50k_nfe10_20_seed0/metrics/gpde_pndm_cifar10_50k_nfe10_20_seed0.csv",
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_50k_nfe10_20_seeds1_2/metrics/gpde_pndm_cifar10_50k_nfe10_20_seeds1_2.csv",
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_50k_linear_baseline_seeds0_1_2/metrics/gpde_pndm_cifar10_50k_linear_baseline_seeds0_1_2.csv",
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_50k_karras_baseline_seeds0_1_2/metrics/gpde_pndm_cifar10_50k_karras_baseline_seeds0_1_2.csv",
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_authorized_offline/metrics/offline_authorized_seed0_nfe10.csv",
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_authorized_offline/metrics/offline_authorized_seed0_nfe20.csv",
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_authorized_offline/metrics/offline_authorized_seed1_nfe10.csv",
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_authorized_offline/metrics/offline_authorized_seed1_nfe20.csv",
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_authorized_offline/metrics/offline_authorized_seed2_nfe10.csv",
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_authorized_offline/metrics/offline_authorized_seed2_nfe20.csv",
]
DEFAULT_DETAIL = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_fid_detail_seeds0_1_2.csv"
DEFAULT_AGGREGATE = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_fid_aggregate_seeds0_1_2.csv"
DEFAULT_DELTA = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_fid_delta_seeds0_1_2.csv"
DEFAULT_TABLE = PAPER_ROOT / "tables/cifar10_pndm_euler_50k_fid_seeds0_1_2.tex"
DEFAULT_FIGURE = PAPER_ROOT / "figures/cifar10_pndm_euler_50k_fid_seeds0_1_2"

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

DELTA_FIELDS = [
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
    "offline_fid_mean",
    "offline_fid_sem",
    "dgpde_fid_mean",
    "dgpde_fid_sem",
    "fid_reduction_mean",
    "fid_reduction_std",
    "fid_reduction_sem",
    "all_seed_reductions_positive",
    "dgpde_minus_offline_mean",
    "dgpde_minus_offline_std",
    "dgpde_minus_offline_sem",
    "all_seed_dgpde_less_than_offline",
]

METHODS = ("base", "linear", "Karras", "offline", "D-GPDE")
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
METHOD_DISPLAY = {"base": "base", "linear": "linear", "Karras": "Karras", "offline": "offline", "D-GPDE": "D-GPDE"}
COLORS = {
    "base": "#6F7D8C",
    "linear": "#2A9D8F",
    "Karras": "#8E5CF7",
    "offline": "#4C78A8",
    "D-GPDE": "#E76F51",
}
MARKERS = {"base": "o", "linear": "^", "Karras": "D", "offline": "v", "D-GPDE": "s"}


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PNDM/CIFAR-10 50k FID across seeds.")
    parser.add_argument("--source-csv", action="append", type=Path, default=None)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--aggregate-csv", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--delta-csv", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--table-tex", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--figure-prefix", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--solver", default="euler")
    parser.add_argument("--expected-seeds", default="0,1,2")
    parser.add_argument("--expected-nfes", default="10,20")
    return parser.parse_args()


def method_label(schedule: str) -> str | None:
    normalized = schedule.strip().lower()
    if normalized == "base":
        return "base"
    if normalized == "linear":
        return "linear"
    if normalized in {"karras", "edm", "karras_edm"}:
        return "Karras"
    if normalized in {"offline", "offline_authorized", "authorized_offline"}:
        return "offline"
    if normalized in {"gpde", "goes"}:
        return "D-GPDE"
    return None


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def read_rows(source_csvs: list[Path], *, solver: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in source_csvs:
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(f"Missing source CSV: {source}")
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("status") != "OK":
                    continue
                if row.get("metric_name") != "fid":
                    continue
                if row.get("dataset") != "cifar10":
                    continue
                if row.get("solver") != solver:
                    continue
                method = method_label(row.get("schedule", ""))
                if method is None:
                    continue
                rows.append(
                    {
                        "evidence_level": "three_seed_50k_fid",
                        "backend": row.get("backend", ""),
                        "dataset": row.get("dataset", ""),
                        "model": row.get("model", ""),
                        "solver": row.get("solver", ""),
                        "method": method,
                        "nfe": str(int(row.get("nfe", "0"))),
                        "seed": str(int(row.get("seed", "0"))),
                        "num_samples": row.get("num_samples", ""),
                        "metric": "FID",
                        "metric_direction": "lower_is_better",
                        "fid": fmt(float(row.get("fid", "nan"))),
                        "source_csv": str(source.relative_to(REPO_ROOT)),
                        "source_output_dir": row.get("output_dir", ""),
                        "source_schedule_dir": row.get("schedule_dir", ""),
                    }
                )

    rows.sort(key=lambda item: (int(item["nfe"]), int(item["seed"]), METHOD_ORDER[item["method"]]))
    if not rows:
        raise ValueError("No matching OK FID rows found.")
    return rows


def validate_complete(rows: list[dict[str, str]], *, expected_seeds: list[int], expected_nfes: list[int]) -> None:
    expected = {
        (nfe, seed, method)
        for nfe in expected_nfes
        for seed in expected_seeds
        for method in METHODS
    }
    actual = {(int(row["nfe"]), int(row["seed"]), row["method"]) for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Unexpected row coverage. missing={missing} extra={extra}")
    sample_counts = {row["num_samples"] for row in rows}
    if sample_counts != {"50000"}:
        raise ValueError(f"Expected only 50000-sample rows, got {sorted(sample_counts)}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["nfe"]), row["method"])].append(row)

    aggregate: list[dict[str, str]] = []
    for (nfe, method), group in sorted(grouped.items(), key=lambda item: (item[0][0], METHOD_ORDER[item[0][1]])):
        values = [float(row["fid"]) for row in group]
        seeds = sorted(int(row["seed"]) for row in group)
        first = group[0]
        aggregate.append(
            {
                "evidence_level": "three_seed_50k_fid",
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
    return aggregate


def delta_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keyed = {(int(row["nfe"]), int(row["seed"]), row["method"]): row for row in rows}
    nfes = sorted({int(row["nfe"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    deltas: list[dict[str, str]] = []
    for nfe in nfes:
        base_values = [float(keyed[(nfe, seed, "base")]["fid"]) for seed in seeds]
        offline_values = [float(keyed[(nfe, seed, "offline")]["fid"]) for seed in seeds]
        dgpde_values = [float(keyed[(nfe, seed, "D-GPDE")]["fid"]) for seed in seeds]
        reductions = [base - dgpde for base, dgpde in zip(base_values, dgpde_values)]
        offline_gaps = [dgpde - offline for dgpde, offline in zip(dgpde_values, offline_values)]
        first = keyed[(nfe, seeds[0], "base")]
        deltas.append(
            {
                "evidence_level": "three_seed_50k_fid",
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
                "base_fid_mean": fmt(statistics.mean(base_values)),
                "base_fid_sem": fmt(sem(base_values)),
                "offline_fid_mean": fmt(statistics.mean(offline_values)),
                "offline_fid_sem": fmt(sem(offline_values)),
                "dgpde_fid_mean": fmt(statistics.mean(dgpde_values)),
                "dgpde_fid_sem": fmt(sem(dgpde_values)),
                "fid_reduction_mean": fmt(statistics.mean(reductions)),
                "fid_reduction_std": fmt(std(reductions)),
                "fid_reduction_sem": fmt(sem(reductions)),
                "all_seed_reductions_positive": str(all(value > 0.0 for value in reductions)),
                "dgpde_minus_offline_mean": fmt(statistics.mean(offline_gaps)),
                "dgpde_minus_offline_std": fmt(std(offline_gaps)),
                "dgpde_minus_offline_sem": fmt(sem(offline_gaps)),
                "all_seed_dgpde_less_than_offline": str(all(value < 0.0 for value in offline_gaps)),
            }
        )
    return deltas


def paired_gap(
    rows: list[dict[str, str]],
    *,
    nfe: int,
    left_method: str,
    right_method: str,
) -> tuple[float, float]:
    keyed = {(int(row["seed"]), row["method"]): float(row["fid"]) for row in rows if int(row["nfe"]) == nfe}
    seeds = sorted(seed for seed, method in keyed if method == left_method)
    gaps = [keyed[(seed, left_method)] - keyed[(seed, right_method)] for seed in seeds]
    return statistics.mean(gaps), sem(gaps)


def write_table(
    rows: list[dict[str, str]],
    aggregate: list[dict[str, str]],
    deltas: list[dict[str, str]],
    table_path: Path,
) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_by_key = {(int(row["nfe"]), row["method"]): row for row in aggregate}
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{PNDM/CIFAR-10 50k FID for the Euler solver. Rows report",
        "  mean $\\pm$ standard error over three seeds (0, 1, 2), base, linear",
        "  native-coordinate, fixed Karras/EDM-style, authorized project-owned",
        "  offline, and \\method{} schedules. FID is lower-is-better and each",
        "  seed uses 50k generated images. The final two columns are paired",
        "  \\method{}-minus-baseline FID gaps; negative values favor \\method{}.}",
        "  \\label{tab:cifar10-pndm-euler-50k-fid-seeds0-1-2}",
        "  {\\scriptsize",
        "  \\setlength{\\tabcolsep}{2pt}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{rrrrrrrr}",
        "    \\toprule",
        "    NFE & base & linear & Karras & offline & \\method{} &",
        "    \\method{} $-$ offline & \\method{} $-$ Karras \\\\",
        "    \\midrule",
    ]
    for row in deltas:
        nfe = int(row["nfe"])
        linear = aggregate_by_key[(nfe, "linear")]
        karras = aggregate_by_key[(nfe, "Karras")]
        offline = aggregate_by_key[(nfe, "offline")]
        dgpde = aggregate_by_key[(nfe, "D-GPDE")]
        dgpde_minus_offline, dgpde_minus_offline_sem = paired_gap(
            rows,
            nfe=nfe,
            left_method="D-GPDE",
            right_method="offline",
        )
        dgpde_minus_karras, dgpde_minus_karras_sem = paired_gap(
            rows,
            nfe=nfe,
            left_method="D-GPDE",
            right_method="Karras",
        )
        lines.append(
            "    "
            f"{row['nfe']} & "
            f"{float(row['base_fid_mean']):.2f} $\\pm$ {float(row['base_fid_sem']):.2f} & "
            f"{float(linear['fid_mean']):.2f} $\\pm$ {float(linear['fid_sem']):.2f} & "
            f"{float(karras['fid_mean']):.2f} $\\pm$ {float(karras['fid_sem']):.2f} & "
            f"{float(offline['fid_mean']):.2f} $\\pm$ {float(offline['fid_sem']):.2f} & "
            f"{float(dgpde['fid_mean']):.2f} $\\pm$ {float(dgpde['fid_sem']):.2f} & "
            f"{dgpde_minus_offline:.3f} $\\pm$ {dgpde_minus_offline_sem:.3f} & "
            f"{dgpde_minus_karras:.3f} $\\pm$ {dgpde_minus_karras_sem:.3f} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  }",
            "  }",
            "\\end{table}",
            "",
        ]
    )
    table_path.write_text("\n".join(lines))


def plot(rows: list[dict[str, str]], aggregate: list[dict[str, str]], figure_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.9,
            "lines.markersize": 5,
        }
    )

    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    for method in METHODS:
        method_rows = [row for row in aggregate if row["method"] == method]
        xs = [int(row["nfe"]) for row in method_rows]
        ys = [float(row["fid_mean"]) for row in method_rows]
        yerr = [float(row["fid_sem"]) for row in method_rows]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=MARKERS[method],
            color=COLORS[method],
            capsize=3,
            label=METHOD_DISPLAY[method],
            zorder=3 if method == "D-GPDE" else 2,
        )
        for detail in rows:
            if detail["method"] != method:
                continue
            jitter = {"base": -0.4, "linear": -0.2, "Karras": 0.0, "offline": 0.2, "D-GPDE": 0.4}[method]
            ax.scatter(
                int(detail["nfe"]) + jitter,
                float(detail["fid"]),
                s=12,
                color=COLORS[method],
                alpha=0.32,
                linewidths=0,
            )

    ax.set_xlabel("NFE")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("PNDM/CIFAR-10 Euler, 50k samples, seeds 0/1/2", pad=8)
    ax.set_xticks(sorted({int(row["nfe"]) for row in rows}))
    ax.legend(loc="upper right")

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_prefix.with_suffix(".pdf"))
    fig.savefig(figure_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


def copy_baseline_schedules(rows: list[dict[str, str]], detail_path: Path) -> None:
    destination = detail_path.parent
    schedule_dirs = sorted(
        {
            (row["method"], Path(row["source_schedule_dir"]))
            for row in rows
            if row["method"] in {"linear", "Karras"} and row["source_schedule_dir"]
        }
    )
    for method, schedule_dir in schedule_dirs:
        nfe = schedule_dir.name.replace("nfe_", "nfe")
        method_token = "linear" if method == "linear" else "karras"
        target_prefix = destination / f"cifar10_pndm_euler_50k_{method_token}_schedule_{nfe}"
        for filename in ("meta.json", "timesteps.npy", "time_grid.npy", "sigmas.npy", "sigma_grid.npy"):
            source = REPO_ROOT / schedule_dir / filename
            if source.exists():
                shutil.copy2(source, target_prefix.with_name(f"{target_prefix.name}_{filename}"))


def main() -> None:
    args = parse_args()
    source_csvs = args.source_csv if args.source_csv else DEFAULT_SOURCES
    expected_seeds = parse_int_list(args.expected_seeds)
    expected_nfes = parse_int_list(args.expected_nfes)
    rows = read_rows(source_csvs, solver=args.solver)
    validate_complete(rows, expected_seeds=expected_seeds, expected_nfes=expected_nfes)
    aggregate = aggregate_rows(rows)
    deltas = delta_rows(rows)
    write_csv(args.detail_csv, DETAIL_FIELDS, rows)
    write_csv(args.aggregate_csv, AGGREGATE_FIELDS, aggregate)
    write_csv(args.delta_csv, DELTA_FIELDS, deltas)
    copy_baseline_schedules(rows, args.detail_csv)
    write_table(rows, aggregate, deltas, args.table_tex)
    plot(rows, aggregate, args.figure_prefix)
    print(f"[aggregate-cifar] detail_rows={len(rows)} detail={args.detail_csv}")
    print(f"[aggregate-cifar] aggregate_rows={len(aggregate)} aggregate={args.aggregate_csv}")
    print(f"[aggregate-cifar] delta_rows={len(deltas)} delta={args.delta_csv}")
    print(f"[aggregate-cifar] figure={args.figure_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent

DEFAULT_SOURCE = REPO_ROOT / "outputs/gpde_pndm_test/metrics/gpde_pndm_test.csv"
DEFAULT_RESULTS = PAPER_ROOT / "results/pilot/cifar10_pndm_euler_pilot_fid.csv"
DEFAULT_TABLE = PAPER_ROOT / "tables/cifar10_pndm_euler_pilot_fid.tex"
DEFAULT_FIGURE = PAPER_ROOT / "figures/cifar10_pndm_euler_pilot_fid"
DEFAULT_EVIDENCE_LEVEL = "pilot_5k_seed0_not_paper_grade"
DEFAULT_CAPTION_NOTE = "This is retained pilot evidence, not paper-grade 50k FID."
DEFAULT_PLOT_TITLE = "PNDM/CIFAR-10 Euler pilot, 5k samples, seed 0"

CLEAN_FIELDS = [
    "evidence_level",
    "backend",
    "dataset",
    "model",
    "solver",
    "method",
    "nfe",
    "seed",
    "seed_count",
    "num_samples",
    "metric",
    "metric_direction",
    "fid",
    "source_csv",
    "source_output_dir",
    "source_schedule_dir",
]

METHOD_ORDER = {"base": 0, "D-GPDE": 1}
COLORS = {"base": "#6F7D8C", "D-GPDE": "#E76F51"}
MARKERS = {"base": "o", "D-GPDE": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-facing pilot CIFAR-10 FID CSV, table, and figure."
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--table-tex", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--figure-prefix", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--solver", default="euler")
    parser.add_argument("--evidence-level", default=DEFAULT_EVIDENCE_LEVEL)
    parser.add_argument("--caption-note", default=DEFAULT_CAPTION_NOTE)
    parser.add_argument("--plot-title", default=DEFAULT_PLOT_TITLE)
    return parser.parse_args()


def method_label(schedule: str) -> str | None:
    normalized = schedule.strip().lower()
    if normalized == "base":
        return "base"
    if normalized in {"gpde", "goes"}:
        return "D-GPDE"
    return None


def read_clean_rows(source_csv: Path, *, solver: str, evidence_level: str) -> list[dict[str, str]]:
    source_csv = source_csv.resolve()
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {source_csv}")

    rows: list[dict[str, str]] = []
    with source_csv.open(newline="") as handle:
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
                    "evidence_level": evidence_level,
                    "backend": row.get("backend", ""),
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "solver": row.get("solver", ""),
                    "method": method,
                    "nfe": str(int(row.get("nfe", "0"))),
                    "seed": row.get("seed", ""),
                    "seed_count": "1",
                    "num_samples": row.get("num_samples", ""),
                    "metric": "FID",
                    "metric_direction": "lower_is_better",
                    "fid": f"{float(row.get('fid', 'nan')):.6f}",
                    "source_csv": str(source_csv.relative_to(REPO_ROOT)),
                    "source_output_dir": row.get("output_dir", ""),
                    "source_schedule_dir": row.get("schedule_dir", ""),
                }
            )

    rows.sort(key=lambda item: (int(item["nfe"]), METHOD_ORDER.get(item["method"], 99)))
    if not rows:
        raise ValueError(f"No OK FID rows found for solver={solver!r} in {source_csv}")
    return rows


def write_clean_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def pivot_by_nfe(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    pivot: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        pivot[int(row["nfe"])][row["method"]] = float(row["fid"])
    return dict(sorted(pivot.items()))


def pluralized(values: list[str], noun: str) -> str:
    return f"{len(values)} {noun}" + ("" if len(values) == 1 else "s")


def write_table(rows: list[dict[str, str]], table_tex: Path, *, caption_note: str) -> None:
    pivot = pivot_by_nfe(rows)
    sample_counts = sorted({row["num_samples"] for row in rows}, key=int)
    seeds = sorted({row["seed"] for row in rows}, key=int)
    sample_text = sample_counts[0] if len(sample_counts) == 1 else "/".join(sample_counts)
    seed_text = ", ".join(seeds)
    label = table_tex.stem.replace("_", "-")
    table_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{PNDM/CIFAR-10 FID for the Euler solver. Schedules are",
        f"  base and \\method{{}}, NFE is listed per row, seed count is {pluralized(seeds, 'seed')}",
        f"  ({seed_text}), and FID is lower-is-better. Each row uses {sample_text}",
        f"  generated images. {caption_note}}}",
        f"  \\label{{tab:{label}}}",
        "  \\begin{tabular}{rrrr}",
        "    \\toprule",
        "    NFE & base FID $\\downarrow$ & \\method{} FID $\\downarrow$ & FID reduction \\\\",
        "    \\midrule",
    ]
    for nfe, values in pivot.items():
        if "base" not in values or "D-GPDE" not in values:
            continue
        reduction = values["base"] - values["D-GPDE"]
        lines.append(
            f"    {nfe} & {values['base']:.2f} & {values['D-GPDE']:.2f} & {reduction:.2f} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    table_tex.write_text("\n".join(lines))


def plot_fid(rows: list[dict[str, str]], figure_prefix: Path, *, title: str) -> None:
    by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append((int(row["nfe"]), float(row["fid"])))

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

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    for method in ["base", "D-GPDE"]:
        points = sorted(by_method.get(method, []))
        if not points:
            continue
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(
            xs,
            ys,
            marker=MARKERS[method],
            color=COLORS[method],
            label=method,
            zorder=3 if method == "D-GPDE" else 2,
        )

    ax.set_xlabel("NFE")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title(title, pad=8)
    ax.set_xticks(sorted({int(row["nfe"]) for row in rows}))
    ax.legend(loc="upper right")

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_prefix.with_suffix(".pdf"))
    fig.savefig(figure_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_clean_rows(args.source_csv, solver=args.solver, evidence_level=args.evidence_level)
    write_clean_csv(rows, args.results_csv)
    write_table(rows, args.table_tex, caption_note=args.caption_note)
    plot_fid(rows, args.figure_prefix, title=args.plot_title)
    print(f"[pilot-results] rows={len(rows)} csv={args.results_csv}")
    print(f"[pilot-results] table={args.table_tex}")
    print(f"[pilot-results] figure={args.figure_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PAPER_ROOT = Path(__file__).resolve().parents[2]

CIFAR_DELTA = PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_fid_delta_seeds0_1_2.csv"
CIFAR_COSTS = [
    PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_oracle_reuse_cost_seed0.csv",
    PAPER_ROOT / "results/cifar10_50k/cifar10_pndm_euler_50k_oracle_reuse_cost_seeds1_2.csv",
]
T2I_PAIRWISE = PAPER_ROOT / "results/t2i/sd15_euler_nfe10_cfg_sweep_pairwise_summary.csv"
T2I_COST = PAPER_ROOT / "results/t2i/sd15_euler_nfe10_cfg_sweep_oracle_reuse_cost.csv"
T2I_SDXL_PAIRWISE = PAPER_ROOT / "results/t2i/sdxl_euler_nfe10_cfg_sweep_pairwise_summary.csv"
T2I_SDXL_COST = PAPER_ROOT / "results/t2i/sdxl_euler_nfe10_cfg_sweep_oracle_reuse_cost.csv"

OUTPUT_CSV = PAPER_ROOT / "results/cost/calibration_cost_summary.csv"
OUTPUT_TABLE = PAPER_ROOT / "tables/calibration_cost_summary.tex"
OUTPUT_AMORTIZATION_CSV = PAPER_ROOT / "results/cost/calibration_amortization_summary.csv"
OUTPUT_AMORTIZATION_TABLE = PAPER_ROOT / "tables/calibration_amortization_summary.tex"
OUTPUT_FIGURE = PAPER_ROOT / "figures/calibration_cost_vs_quality"


FIELDS = [
    "evidence_level",
    "setting",
    "model",
    "solver",
    "condition",
    "evaluated_items",
    "quality_metric",
    "quality_delta_vs_base",
    "quality_delta_sem",
    "quality_delta_display",
    "calibration_model_eval_equivalents",
    "evaluated_generation_model_eval_equivalents",
    "calibration_to_generation_ratio",
]

AMORTIZATION_FIELDS = [
    "setting",
    "condition",
    "evaluated_items",
    "quality_delta_display",
    "calibration_to_generation_ratio",
    "overhead_after_1_batch",
    "overhead_after_10_batches",
    "overhead_after_100_batches",
    "break_even_evaluated_batches",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fmt(value: float) -> str:
    return f"{value:.6f}"


def latex_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}".replace(",", r"{,}")


def latex_ratio(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}".replace(",", r"{,}") + r"$\times$"
    return f"{value:.2f}" + r"$\times$"


def build_cifar_rows() -> list[dict[str, str]]:
    deltas = {int(row["nfe"]): row for row in read_csv(CIFAR_DELTA)}
    cost_rows: list[dict[str, str]] = []
    for path in CIFAR_COSTS:
        cost_rows.extend(read_csv(path))

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in cost_rows:
        if row.get("status") == "OK" and row.get("backend") == "pndm":
            grouped[int(row["nfe"])].append(row)

    rows: list[dict[str, str]] = []
    for nfe in (10, 20):
        group = grouped[nfe]
        if len(group) != 3:
            raise ValueError(f"Expected three CIFAR cost rows for NFE {nfe}, got {len(group)}")
        delta = deltas[nfe]
        calibration = sum(as_float(row, "per_schedule_total_model_eval_equivalents") for row in group)
        generation = 3 * 50_000 * nfe
        quality = as_float(delta, "fid_reduction_mean")
        quality_sem = as_float(delta, "fid_reduction_sem")
        rows.append(
            {
                "evidence_level": "three_seed_50k_fid",
                "setting": "CIFAR-10",
                "model": "pndm_model_ddim_cifar10",
                "solver": "Euler",
                "condition": f"NFE {nfe}",
                "evaluated_items": "150k images",
                "quality_metric": "FID reduction",
                "quality_delta_vs_base": fmt(quality),
                "quality_delta_sem": fmt(quality_sem),
                "quality_delta_display": f"+{quality:.3f} +/- {quality_sem:.3f} FID",
                "calibration_model_eval_equivalents": fmt(calibration),
                "evaluated_generation_model_eval_equivalents": fmt(float(generation)),
                "calibration_to_generation_ratio": fmt(calibration / generation),
            }
        )
    return rows


def build_t2i_rows(
    *,
    pairwise_path: Path,
    cost_path: Path,
    guidance_scales: tuple[float, ...],
    evidence_level: str,
    setting: str,
    model: str,
) -> list[dict[str, str]]:
    pairwise = {
        float(row["guidance_scale"]): row
        for row in read_csv(pairwise_path)
        if row["comparison"] == "D-GPDE - base"
    }
    cost_rows = read_csv(cost_path)

    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in cost_rows:
        if row.get("status") == "OK" and row.get("backend") == "diffusers":
            grouped[float(row["guidance_scale"])].append(row)

    rows: list[dict[str, str]] = []
    for guidance in guidance_scales:
        group = grouped[guidance]
        if len(group) != 3:
            raise ValueError(f"Expected three {setting} cost rows for CFG {guidance:g}, got {len(group)}")
        calibration = sum(as_float(row, "per_schedule_total_model_eval_equivalents") for row in group)
        generation = len(group) * 50 * 10 * 2
        quality = as_float(pairwise[guidance], "image_reward_mean_delta")
        rows.append(
            {
                "evidence_level": evidence_level,
                "setting": setting,
                "model": model,
                "solver": "Diffusers Euler",
                "condition": f"CFG {guidance:g}",
                "evaluated_items": "150 images",
                "quality_metric": "ImageReward delta",
                "quality_delta_vs_base": fmt(quality),
                "quality_delta_sem": "",
                "quality_delta_display": f"{quality:+.3f} IR",
                "calibration_model_eval_equivalents": fmt(calibration),
                "evaluated_generation_model_eval_equivalents": fmt(float(generation)),
                "calibration_to_generation_ratio": fmt(calibration / generation),
            }
        )
    return rows


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Calibration cost accounting for current paper-facing runs.",
        r"  The quality column reports base-minus-\method{} FID for CIFAR-10 and",
        r"  \method{}-minus-base ImageReward for text-to-image, so positive values favor",
        r"  \method{}. Model-evaluation equivalents count retained calibration",
        r"  cost and the evaluated generation batch, excluding unrelated pilot",
        r"  experiments.}",
        r"  \label{tab:calibration-cost-summary}",
        r"  \begin{tabular}{lllrrr}",
        r"    \toprule",
        r"    Setting & condition & quality vs. base & cal. evals (M) & gen. evals (M) & cal./gen. \\",
        r"    \midrule",
    ]
    for row in rows:
        calibration = float(row["calibration_model_eval_equivalents"]) / 1_000_000.0
        generation = float(row["evaluated_generation_model_eval_equivalents"]) / 1_000_000.0
        ratio = float(row["calibration_to_generation_ratio"])
        lines.append(
            "    "
            + " & ".join(
                [
                    row["setting"],
                    row["condition"],
                    row["quality_delta_display"].replace("+/-", r"$\pm$"),
                    latex_number(calibration),
                    latex_number(generation, decimals=3),
                    latex_ratio(ratio),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def write_amortization(path_csv: Path, path_table: Path, rows: list[dict[str, str]]) -> None:
    records: list[dict[str, str]] = []
    for row in rows:
        ratio = float(row["calibration_to_generation_ratio"])
        records.append(
            {
                "setting": row["setting"],
                "condition": row["condition"],
                "evaluated_items": row["evaluated_items"],
                "quality_delta_display": row["quality_delta_display"],
                "calibration_to_generation_ratio": fmt(ratio),
                "overhead_after_1_batch": fmt(ratio),
                "overhead_after_10_batches": fmt(ratio / 10.0),
                "overhead_after_100_batches": fmt(ratio / 100.0),
                "break_even_evaluated_batches": str(math.ceil(ratio)),
            }
        )

    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AMORTIZATION_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Calibration amortization under reuse of the same schedule.",
        r"  Columns report calibration overhead divided by generation cost after",
        r"  reusing the schedule for $M$ evaluated-size batches. Break-even is the",
        r"  smallest integer $M$ for which calibration cost is no larger than",
        r"  generation cost.}",
        r"  \label{tab:calibration-amortization-summary}",
        r"  \small",
        r"  \begin{tabular}{lllrrrr}",
        r"    \toprule",
        r"    Setting & condition & quality vs. base & $M=1$ & $M=10$ & $M=100$ & break-even \\",
        r"    \midrule",
    ]
    for record in records:
        lines.append(
            "    "
            + " & ".join(
                [
                    record["setting"],
                    record["condition"],
                    record["quality_delta_display"].replace("+/-", r"$\pm$"),
                    latex_ratio(float(record["overhead_after_1_batch"])),
                    latex_ratio(float(record["overhead_after_10_batches"])),
                    latex_ratio(float(record["overhead_after_100_batches"])),
                    record["break_even_evaluated_batches"],
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    path_table.parent.mkdir(parents=True, exist_ok=True)
    path_table.write_text("\n".join(lines))


def write_figure(prefix: Path, rows: list[dict[str, str]]) -> None:
    labels = [f"{row['setting']} {row['condition']}" for row in rows]
    ratios = [float(row["calibration_to_generation_ratio"]) for row in rows]
    quality = [float(row["quality_delta_vs_base"]) for row in rows]
    quality_labels = [row["quality_delta_display"].replace("+/-", "+/-") for row in rows]
    colors = ["#009E73" if value > 0 else "#D55E00" for value in quality]

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    y_positions = list(range(len(rows)))
    ax.barh(y_positions, ratios, color=colors, alpha=0.88)
    ax.set_xscale("log")
    ax.set_xlabel("calibration cost / evaluated generation cost")
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    for y, ratio, label in zip(y_positions, ratios, quality_labels):
        ax.text(ratio * 1.08, y, label, va="center", fontsize=8)
    ax.set_title("Calibration cost versus observed quality change")
    fig.tight_layout()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    rows = (
        build_cifar_rows()
        + build_t2i_rows(
            pairwise_path=T2I_PAIRWISE,
            cost_path=T2I_COST,
            guidance_scales=(5.0, 7.5, 10.0),
            evidence_level="matched_sd15_cfg_sweep",
            setting="SD1.5",
            model="stable_diffusion_15",
        )
        + build_t2i_rows(
            pairwise_path=T2I_SDXL_PAIRWISE,
            cost_path=T2I_SDXL_COST,
            guidance_scales=(5.0, 7.5),
            evidence_level="matched_sdxl_cfg_sweep",
            setting="SDXL",
            model="sdxl",
        )
    )
    write_csv(OUTPUT_CSV, rows)
    write_table(OUTPUT_TABLE, rows)
    write_amortization(OUTPUT_AMORTIZATION_CSV, OUTPUT_AMORTIZATION_TABLE, rows)
    write_figure(OUTPUT_FIGURE, rows)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_TABLE}")
    print(f"Wrote {OUTPUT_AMORTIZATION_CSV}")
    print(f"Wrote {OUTPUT_AMORTIZATION_TABLE}")
    print(f"Wrote {OUTPUT_FIGURE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/gpde_diffusers_sdxl_nfe10_cfg_seed_sweep"
DEFAULT_RESULTS_DIR = PAPER_ROOT / "results/t2i"
DEFAULT_TABLE = PAPER_ROOT / "tables/sdxl_euler_nfe10_cfg_sweep_pairwise.tex"
DEFAULT_FIGURE = PAPER_ROOT / "figures/sdxl_euler_nfe10_cfg_sweep_pairwise"

EXPECTED_GUIDANCE = (5.0, 7.5)
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_SCHEDULES = ("base", "ays", "GPDE")
EXPECTED_COMPARISONS = ("AYS vs base", "GPDE vs base", "GPDE vs AYS")
EXPECTED_METRICS = ("clip_score", "image_reward")
EXPECTED_JPG_COUNT = 900
EXPECTED_MANIFEST_COUNT = 18
EVIDENCE_LEVEL = "matched_sdxl_cfg_sweep"
MODEL_LABEL = "sdxl"

SCHEDULE_LABELS = {"base": "base", "ays": "AYS", "GPDE": "D-GPDE"}
COMPARISON_LABELS = {
    "AYS vs base": "AYS - base",
    "GPDE vs base": "D-GPDE - base",
    "GPDE vs AYS": "D-GPDE - AYS",
}
COMPARISON_ORDER = {name: index for index, name in enumerate(EXPECTED_COMPARISONS)}

COLORS = {
    "AYS vs base": "#2A9D8F",
    "GPDE vs base": "#E76F51",
    "GPDE vs AYS": "#264653",
}
MARKERS = {"AYS vs base": "o", "GPDE vs base": "s", "GPDE vs AYS": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SDXL CFG-sweep text-to-image metrics.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--table-tex", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--figure-prefix", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_local_paths(text: str) -> str:
    repo = str(REPO_ROOT)
    return text.replace(repo + "/", "").replace(repo, ".")


def copy_sanitized_text(source: Path, target: Path) -> None:
    target.write_text(sanitize_local_paths(source.read_text()))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def validate_run_outputs(output_root: Path) -> dict[str, Path]:
    metrics_root = output_root / "metrics"
    paths = {
        "summary": metrics_root / "gpde_diffusers_sdxl_nfe10_cfg_seed_sweep.csv",
        "aggregate": metrics_root / "gpde_diffusers_sdxl_nfe10_cfg_seed_sweep_aggregate.csv",
        "detail": metrics_root / "gpde_diffusers_sdxl_nfe10_cfg_seed_sweep_detail.csv",
        "pairwise": metrics_root / "gpde_diffusers_sdxl_nfe10_cfg_seed_sweep_pairwise.csv",
        "oracle_cost": output_root / "paper_tables/oracle_reuse_cost.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required outputs: {missing}")

    schedule_paths = sorted(
        (
            output_root
            / "schedules/GPDE/diffusers/sdxl/euler"
        ).glob("cfg_*/seed_*/nfe_010/schedule.json")
    )
    expected_schedules = len(EXPECTED_GUIDANCE) * len(EXPECTED_SEEDS)
    if len(schedule_paths) != expected_schedules:
        raise ValueError(f"Expected {expected_schedules} D-GPDE schedules, found {len(schedule_paths)}.")

    jpg_count = sum(1 for _ in (output_root / "samples").glob("**/*.jpg"))
    manifest_count = sum(1 for _ in (output_root / "samples").glob("**/run_manifest.json"))
    if jpg_count != EXPECTED_JPG_COUNT or manifest_count != EXPECTED_MANIFEST_COUNT:
        raise ValueError(f"Unexpected sample coverage: jpg={jpg_count} manifests={manifest_count}.")
    return paths


def validate_aggregate(rows: list[dict[str, str]]) -> None:
    expected = {
        (schedule, guidance, seed)
        for schedule in EXPECTED_SCHEDULES
        for guidance in EXPECTED_GUIDANCE
        for seed in EXPECTED_SEEDS
    }
    actual = {
        (row["schedule"], float(row["guidance_scale"]), int(row["seed"]))
        for row in rows
    }
    if actual != expected:
        raise ValueError(f"Unexpected aggregate coverage. missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    if {int(row["num_images"]) for row in rows} != {50}:
        raise ValueError("Expected each aggregate row to contain 50 generated images.")


def validate_pairwise(rows: list[dict[str, str]]) -> None:
    expected = {
        (comparison, metric, guidance)
        for comparison in EXPECTED_COMPARISONS
        for metric in EXPECTED_METRICS
        for guidance in EXPECTED_GUIDANCE
    }
    actual = {
        (row["comparison"], row["metric"], float(row["guidance_scale"]))
        for row in rows
    }
    if actual != expected:
        raise ValueError(f"Unexpected pairwise coverage. missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    bad_pairs = [row for row in rows if int(row["num_pairs"]) != 150]
    if bad_pairs:
        raise ValueError(f"Expected 150 pairs per row, got {bad_pairs[:3]}")


def schedule_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[float, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["guidance_scale"]), row["schedule"])].append(row)

    output: list[dict[str, str]] = []
    for guidance in EXPECTED_GUIDANCE:
        for schedule in EXPECTED_SCHEDULES:
            group = grouped[(guidance, schedule)]
            clip_values = [as_float(row, "clip_score_mean") for row in group]
            reward_values = [as_float(row, "image_reward_mean") for row in group]
            output.append(
                {
                    "evidence_level": EVIDENCE_LEVEL,
                    "model": MODEL_LABEL,
                    "solver": "euler",
                    "schedule": SCHEDULE_LABELS[schedule],
                    "nfe": "10",
                    "guidance_scale": f"{guidance:g}",
                    "seed_count": str(len(group)),
                    "seeds": ",".join(str(seed) for seed in EXPECTED_SEEDS),
                    "prompt_asset": "diffusers_ablation_prompts",
                    "num_prompts_per_seed": "50",
                    "clip_score_mean": fmt_float(statistics.mean(clip_values)),
                    "clip_score_sem_over_seeds": fmt_float(sem(clip_values)),
                    "image_reward_mean": fmt_float(statistics.mean(reward_values)),
                    "image_reward_sem_over_seeds": fmt_float(sem(reward_values)),
                    "metric_direction": "higher_is_better",
                }
            )
    return output


def pairwise_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keyed = {
        (row["comparison"], row["metric"], float(row["guidance_scale"])): row
        for row in rows
    }
    output: list[dict[str, str]] = []
    for guidance in EXPECTED_GUIDANCE:
        for comparison in EXPECTED_COMPARISONS:
            clip = keyed[(comparison, "clip_score", guidance)]
            reward = keyed[(comparison, "image_reward", guidance)]
            output.append(
                {
                    "evidence_level": EVIDENCE_LEVEL,
                    "model": MODEL_LABEL,
                    "solver": "euler",
                    "nfe": "10",
                    "guidance_scale": f"{guidance:g}",
                    "comparison": COMPARISON_LABELS[comparison],
                    "num_pairs": clip["num_pairs"],
                    "clip_score_mean_delta": fmt_float(as_float(clip, "mean_delta")),
                    "clip_score_win_rate": fmt_float(as_float(clip, "win_rate")),
                    "clip_score_ci_low": fmt_float(as_float(clip, "mean_delta_ci_low")),
                    "clip_score_ci_high": fmt_float(as_float(clip, "mean_delta_ci_high")),
                    "image_reward_mean_delta": fmt_float(as_float(reward, "mean_delta")),
                    "image_reward_win_rate": fmt_float(as_float(reward, "win_rate")),
                    "image_reward_ci_low": fmt_float(as_float(reward, "mean_delta_ci_low")),
                    "image_reward_ci_high": fmt_float(as_float(reward, "mean_delta_ci_high")),
                    "metric_direction": "higher_is_better",
                }
            )
    return output


def format_delta(value: str, decimals: int) -> str:
    return f"{float(value):+.{decimals}f}"


def format_pct(value: str) -> str:
    return f"{100.0 * float(value):.1f}"


def latex_comparison(label: str) -> str:
    return label.replace("D-GPDE", r"\method{}")


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{SDXL text-to-image matched CFG sweep with Euler solver,",
        r"  NFE 10, seeds 0--2, and 50 prompts from",
        r"  \texttt{diffusers\_ablation\_prompts}. Rows report pairwise mean",
        r"  deltas and win rates over 150 matched prompt/seed pairs. CLIPScore",
        r"  and ImageReward are higher-is-better.}",
        r"  \label{tab:sdxl-euler-nfe10-cfg-sweep-pairwise}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    CFG & comparison & $\Delta$CLIP & CLIP win (\%) & $\Delta$IR & IR win (\%) \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                [
                    row["guidance_scale"],
                    latex_comparison(row["comparison"]),
                    format_delta(row["clip_score_mean_delta"], 2),
                    format_pct(row["clip_score_win_rate"]),
                    format_delta(row["image_reward_mean_delta"], 3),
                    format_pct(row["image_reward_win_rate"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def plot_pairwise(raw_rows: list[dict[str, str]], figure_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    by_metric: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        by_metric[row["metric"]][row["comparison"]].append(
            (float(row["guidance_scale"]), float(row["mean_delta"]))
        )

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), sharex=True)
    metric_titles = [("clip_score", "CLIPScore", r"$\Delta$ CLIPScore"), ("image_reward", "ImageReward", r"$\Delta$ ImageReward")]
    handles = []
    labels = []
    for ax, (metric, title, ylabel) in zip(axes, metric_titles):
        ax.axhline(0.0, color="#4A4A4A", linewidth=0.9, linestyle="--", zorder=1)
        for comparison in EXPECTED_COMPARISONS:
            values = sorted(by_metric[metric][comparison])
            xs = [item[0] for item in values]
            ys = [item[1] for item in values]
            line = ax.plot(
                xs,
                ys,
                marker=MARKERS[comparison],
                color=COLORS[comparison],
                linewidth=1.8,
                markersize=4.5,
                label=COMPARISON_LABELS[comparison],
                zorder=3,
            )
            if metric == "clip_score":
                handles.append(line[0])
                labels.append(COMPARISON_LABELS[comparison])
        ax.set_xlabel("CFG scale")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(EXPECTED_GUIDANCE))
        ax.set_title(title)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3)
    fig.subplots_adjust(top=0.78, bottom=0.22, wspace=0.32)
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{figure_prefix}.pdf")
    fig.savefig(f"{figure_prefix}.png", dpi=300)
    plt.close(fig)


def copy_inputs(paths: dict[str, Path], output_root: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "detail": "sdxl_euler_nfe10_cfg_sweep_detail.csv",
        "aggregate": "sdxl_euler_nfe10_cfg_sweep_aggregate_by_seed.csv",
        "pairwise": "sdxl_euler_nfe10_cfg_sweep_pairwise_raw.csv",
        "summary": "sdxl_euler_nfe10_cfg_sweep_summary_raw.csv",
        "oracle_cost": "sdxl_euler_nfe10_cfg_sweep_oracle_reuse_cost.csv",
    }
    for key, filename in copies.items():
        copy_sanitized_text(paths[key], results_dir / filename)

    schedule_dir = results_dir / "schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(
        (
            output_root
            / "schedules/GPDE/diffusers/sdxl/euler"
        ).glob("cfg_*/seed_*/nfe_010/schedule.json")
    ):
        cfg = path.parents[2].name.replace(".", "p")
        seed = path.parents[1].name
        copy_sanitized_text(path, schedule_dir / f"sdxl_euler_{cfg}_{seed}_nfe010_schedule.json")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    paths = validate_run_outputs(output_root)

    aggregate_rows = read_csv(paths["aggregate"])
    pairwise_rows = read_csv(paths["pairwise"])
    validate_aggregate(aggregate_rows)
    validate_pairwise(pairwise_rows)

    copy_inputs(paths, output_root, args.results_dir)

    schedule_rows = schedule_summary(aggregate_rows)
    pairwise_rows_clean = pairwise_summary(pairwise_rows)
    write_csv(
        args.results_dir / "sdxl_euler_nfe10_cfg_sweep_schedule_summary.csv",
        list(schedule_rows[0].keys()),
        schedule_rows,
    )
    write_csv(
        args.results_dir / "sdxl_euler_nfe10_cfg_sweep_pairwise_summary.csv",
        list(pairwise_rows_clean[0].keys()),
        pairwise_rows_clean,
    )
    write_table(args.table_tex, pairwise_rows_clean)
    plot_pairwise(pairwise_rows, args.figure_prefix)

    print(f"validated_output_root={output_root.relative_to(REPO_ROOT)}")
    print(f"wrote_results={args.results_dir.relative_to(REPO_ROOT)}")
    print(f"wrote_table={args.table_tex.relative_to(REPO_ROOT)}")
    print(f"wrote_figure={args.figure_prefix.relative_to(REPO_ROOT)}.[pdf,png]")


if __name__ == "__main__":
    main()

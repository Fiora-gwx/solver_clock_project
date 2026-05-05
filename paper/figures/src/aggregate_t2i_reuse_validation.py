#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper"
ORIGINAL_DETAIL = PAPER_ROOT / "results/t2i/sd15_euler_nfe10_cfg_sweep_detail.csv"
ORIGINAL_SUMMARY = PAPER_ROOT / "results/t2i/sd15_euler_nfe10_cfg_sweep_schedule_summary.csv"
REUSE_DETAIL_RAW = (
    REPO_ROOT
    / "outputs/gpde_diffusers_sd15_reuse_cfg7p5_seed0_schedule/metrics/"
    / "gpde_diffusers_sd15_reuse_cfg7p5_seed0_schedule_detail.csv"
)
REUSE_AGG_RAW = (
    REPO_ROOT
    / "outputs/gpde_diffusers_sd15_reuse_cfg7p5_seed0_schedule/metrics/"
    / "gpde_diffusers_sd15_reuse_cfg7p5_seed0_schedule_aggregate.csv"
)
RESULTS_DIR = PAPER_ROOT / "results/t2i"
TABLE_PATH = PAPER_ROOT / "tables/sd15_euler_nfe10_cfg7p5_seed0_reuse.tex"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rel_path(value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path.relative_to(REPO_ROOT))
    return value


def guidance_label(value: str | float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def paper_schedule(raw: str) -> str:
    if raw.upper() == "GPDE":
        return "D-GPDE"
    if raw.lower() == "ays":
        return "AYS"
    if raw == "D-GPDE-reuse-cfg7p5-seed0":
        return raw
    return raw


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing `{key}` in row: {row}")
    return float(value)


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def clean_reuse_detail(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    cleaned: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (float(item["guidance_scale"]), int(item["seed"]), int(item["prompt_index"]))):
        cleaned.append(
            {
                "evidence_level": "actual_sd15_reuse_cfg7p5_seed0_schedule",
                "model": "stable_diffusion_15",
                "solver": row["solver"],
                "schedule": "D-GPDE-reuse-cfg7p5-seed0",
                "source_schedule_guidance_scale": "7.5",
                "source_schedule_seed": 0,
                "nfe": int(row["nfe"]),
                "guidance_scale": guidance_label(row["guidance_scale"]),
                "seed": int(row["seed"]),
                "prompt_asset": "diffusers_ablation_prompts",
                "prompt_index": int(row["prompt_index"]),
                "prompt": row["prompt"],
                "image_path": rel_path(row["image_path"]),
                "schedule_dir": rel_path(row["schedule_dir"]),
                "clip_score": fmt_float(as_float(row, "clip_score")),
                "image_reward": fmt_float(as_float(row, "image_reward")),
            }
        )
    return cleaned


def reuse_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[guidance_label(row["guidance_scale"])].append(row)

    output: list[dict[str, object]] = []
    for guidance in sorted(grouped, key=float):
        group = sorted(grouped[guidance], key=lambda item: int(item["seed"]))
        clip_values = [as_float(row, "clip_score_mean") for row in group]
        reward_values = [as_float(row, "image_reward_mean") for row in group]
        output.append(
            {
                "evidence_level": "actual_sd15_reuse_cfg7p5_seed0_schedule",
                "model": "stable_diffusion_15",
                "solver": "euler",
                "schedule": "D-GPDE-reuse-cfg7p5-seed0",
                "source_schedule_guidance_scale": "7.5",
                "source_schedule_seed": 0,
                "nfe": 10,
                "guidance_scale": guidance,
                "seed_count": len(group),
                "seeds": ",".join(str(int(row["seed"])) for row in group),
                "prompt_asset": "diffusers_ablation_prompts",
                "num_prompts_per_seed": 50,
                "clip_score_mean": fmt_float(statistics.mean(clip_values)),
                "clip_score_sem_over_seeds": fmt_float(sem(clip_values)),
                "image_reward_mean": fmt_float(statistics.mean(reward_values)),
                "image_reward_sem_over_seeds": fmt_float(sem(reward_values)),
                "metric_direction": "higher_is_better",
            }
        )
    return output


def combined_summary(original: list[dict[str, str]], reuse: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in original:
        rows.append(
            {
                "evidence_level": row["evidence_level"],
                "model": row["model"],
                "solver": row["solver"],
                "schedule": paper_schedule(row["schedule"]),
                "source_schedule_guidance_scale": "",
                "source_schedule_seed": "",
                "nfe": int(row["nfe"]),
                "guidance_scale": guidance_label(row["guidance_scale"]),
                "seed_count": int(row["seed_count"]),
                "seeds": row["seeds"],
                "prompt_asset": row["prompt_asset"],
                "num_prompts_per_seed": int(row["num_prompts_per_seed"]),
                "clip_score_mean": row["clip_score_mean"],
                "clip_score_sem_over_seeds": row["clip_score_sem_over_seeds"],
                "image_reward_mean": row["image_reward_mean"],
                "image_reward_sem_over_seeds": row["image_reward_sem_over_seeds"],
                "metric_direction": row["metric_direction"],
            }
        )
    rows.extend(reuse)
    return sorted(rows, key=lambda item: (float(item["guidance_scale"]), str(item["schedule"])))


def detail_key(row: dict[str, str]) -> tuple[str, int, int]:
    return (guidance_label(row["guidance_scale"]), int(row["seed"]), int(row["prompt_index"]))


def pairwise_summary(original_rows: list[dict[str, str]], reuse_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    original_by_schedule: dict[str, dict[tuple[str, int, int], dict[str, str]]] = defaultdict(dict)
    for row in original_rows:
        original_by_schedule[paper_schedule(row["schedule"])][detail_key(row)] = row
    reuse_by_key = {detail_key(row): row for row in reuse_rows}

    output: list[dict[str, object]] = []
    for guidance in ("5", "7.5", "10"):
        keys = [key for key in sorted(reuse_by_key) if key[0] == guidance]
        for baseline in ("base", "AYS", "D-GPDE"):
            for metric in ("clip_score", "image_reward"):
                deltas: list[float] = []
                wins = 0
                for key in keys:
                    reuse_value = as_float(reuse_by_key[key], metric)
                    base_value = as_float(original_by_schedule[baseline][key], metric)
                    delta = reuse_value - base_value
                    deltas.append(delta)
                    wins += int(delta > 0.0)
                output.append(
                    {
                        "evidence_level": "actual_sd15_reuse_cfg7p5_seed0_schedule",
                        "model": "stable_diffusion_15",
                        "solver": "euler",
                        "nfe": 10,
                        "guidance_scale": guidance,
                        "comparison": f"D-GPDE-reuse-cfg7p5-seed0 - {baseline}",
                        "metric": metric,
                        "matched_pairs": len(deltas),
                        "mean_delta": fmt_float(statistics.mean(deltas)),
                        "win_rate": fmt_float(100.0 * wins / len(deltas)),
                        "metric_direction": "higher_is_better",
                    }
                )
    return output


def table_rows(pairwise: list[dict[str, object]]) -> list[dict[str, object]]:
    keyed = {(str(row["guidance_scale"]), str(row["comparison"]), str(row["metric"])): row for row in pairwise}
    rows: list[dict[str, object]] = []
    for guidance in ("5", "7.5", "10"):
        clip = keyed[(guidance, "D-GPDE-reuse-cfg7p5-seed0 - D-GPDE", "clip_score")]
        reward = keyed[(guidance, "D-GPDE-reuse-cfg7p5-seed0 - D-GPDE", "image_reward")]
        rows.append(
            {
                "guidance_scale": guidance,
                "clip_delta": float(clip["mean_delta"]),
                "clip_win_rate": float(clip["win_rate"]),
                "image_reward_delta": float(reward["mean_delta"]),
                "image_reward_win_rate": float(reward["win_rate"]),
            }
        )
    return rows


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Actual SD1.5 schedule reuse check. We reuse a single",
        r"  \method{} schedule calibrated at CFG 7.5, seed 0 for CFG 5.0, 7.5,",
        r"  and 10.0, seeds 0--2, and 50 prompts per seed. Rows report matched prompt/seed",
        r"  deltas for reuse minus the CFG- and seed-specific \method{} schedule.",
        r"  Higher is better for both metrics.}",
        r"  \label{tab:sd15-euler-nfe10-reuse-cfg7p5-seed0}",
        r"  \begin{tabular}{rrrrr}",
        r"    \toprule",
        r"    CFG & $\Delta$ CLIP & CLIP win (\%) & $\Delta$ ImageReward & IR win (\%) \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + f"{row['guidance_scale']} & {float(row['clip_delta']):+.2f} & {float(row['clip_win_rate']):.1f}"
            + f" & {float(row['image_reward_delta']):+.3f} & {float(row['image_reward_win_rate']):.1f} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    original_detail = read_csv(ORIGINAL_DETAIL)
    original_summary = read_csv(ORIGINAL_SUMMARY)
    reuse_detail_raw = read_csv(REUSE_DETAIL_RAW)
    reuse_agg_raw = read_csv(REUSE_AGG_RAW)

    if len(reuse_detail_raw) != 450:
        raise RuntimeError(f"Expected 450 reuse detail rows, got {len(reuse_detail_raw)}")
    if len(reuse_agg_raw) != 9:
        raise RuntimeError(f"Expected 9 reuse aggregate rows, got {len(reuse_agg_raw)}")

    clean_detail = clean_reuse_detail(reuse_detail_raw)
    reuse_rows = reuse_summary(reuse_agg_raw)
    pairwise_rows = pairwise_summary(original_detail, reuse_detail_raw)
    comparison_rows = combined_summary(original_summary, reuse_rows)

    write_csv(
        RESULTS_DIR / "sd15_euler_nfe10_cfg7p5_seed0_reuse_detail.csv",
        clean_detail[0].keys(),
        clean_detail,
    )
    write_csv(
        RESULTS_DIR / "sd15_euler_nfe10_cfg7p5_seed0_reuse_summary.csv",
        reuse_rows[0].keys(),
        reuse_rows,
    )
    write_csv(
        RESULTS_DIR / "sd15_euler_nfe10_cfg7p5_seed0_reuse_comparison.csv",
        comparison_rows[0].keys(),
        comparison_rows,
    )
    write_csv(
        RESULTS_DIR / "sd15_euler_nfe10_cfg7p5_seed0_reuse_pairwise_summary.csv",
        pairwise_rows[0].keys(),
        pairwise_rows,
    )
    write_table(TABLE_PATH, table_rows(pairwise_rows))


if __name__ == "__main__":
    main()

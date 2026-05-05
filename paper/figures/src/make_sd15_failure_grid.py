#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent

DETAIL_CSV = PAPER_ROOT / "results/t2i/sd15_euler_nfe10_cfg_sweep_detail.csv"
SELECTION_CSV = PAPER_ROOT / "results/failure/sd15_failure_grid_selection.csv"
FIGURE_PREFIX = PAPER_ROOT / "figures/sd15_failure_grid"

SELECTION_FIELDS = [
    "rank",
    "model",
    "solver",
    "nfe",
    "guidance_scale",
    "seed",
    "prompt_index",
    "prompt",
    "base_image_path",
    "dgpde_image_path",
    "base_clip_score",
    "dgpde_clip_score",
    "clip_score_delta_dgpde_minus_base",
    "base_image_reward",
    "dgpde_image_reward",
    "image_reward_delta_dgpde_minus_base",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def matched_key(row: dict[str, str]) -> tuple[float, int, int]:
    return (
        float(row["guidance_scale"]),
        int(row["seed"]),
        int(row["prompt_index"]),
    )


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def select_failures(rows: list[dict[str, str]], count: int = 3) -> list[dict[str, str]]:
    base = {matched_key(row): row for row in rows if row["schedule"] == "base"}
    dgpde = {matched_key(row): row for row in rows if row["schedule"] == "GPDE"}

    candidates: list[dict[str, str]] = []
    for key, dgpde_row in dgpde.items():
        base_row = base.get(key)
        if base_row is None:
            continue
        clip_delta = as_float(dgpde_row, "clip_score") - as_float(base_row, "clip_score")
        reward_delta = as_float(dgpde_row, "image_reward") - as_float(base_row, "image_reward")
        candidates.append(
            {
                "model": "stable_diffusion_15",
                "solver": dgpde_row["solver"],
                "nfe": dgpde_row["nfe"],
                "guidance_scale": f"{key[0]:g}",
                "seed": str(key[1]),
                "prompt_index": str(key[2]),
                "prompt": dgpde_row["prompt"],
                "base_image_path": base_row["image_path"],
                "dgpde_image_path": dgpde_row["image_path"],
                "base_clip_score": f"{as_float(base_row, 'clip_score'):.6f}",
                "dgpde_clip_score": f"{as_float(dgpde_row, 'clip_score'):.6f}",
                "clip_score_delta_dgpde_minus_base": f"{clip_delta:.6f}",
                "base_image_reward": f"{as_float(base_row, 'image_reward'):.6f}",
                "dgpde_image_reward": f"{as_float(dgpde_row, 'image_reward'):.6f}",
                "image_reward_delta_dgpde_minus_base": f"{reward_delta:.6f}",
            }
        )

    selected: list[dict[str, str]] = []
    seen_prompts: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: float(item["image_reward_delta_dgpde_minus_base"])):
        if candidate["prompt_index"] in seen_prompts:
            continue
        seen_prompts.add(candidate["prompt_index"])
        candidate["rank"] = str(len(selected) + 1)
        selected.append(candidate)
        if len(selected) == count:
            return selected
    raise ValueError(f"Could not select {count} unique failure cases.")


def write_selection(rows: list[dict[str, str]]) -> None:
    SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SELECTION_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_image(relative_path: str):
    path = REPO_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return mpimg.imread(path)


def render_grid(rows: list[dict[str, str]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7.5,
            "savefig.dpi": 300,
            "figure.dpi": 300,
        }
    )
    fig, axes = plt.subplots(len(rows), 2, figsize=(6.7, 1.95 * len(rows)))
    for row_index, row in enumerate(rows):
        left = axes[row_index][0]
        right = axes[row_index][1]
        left.imshow(read_image(row["base_image_path"]))
        right.imshow(read_image(row["dgpde_image_path"]))
        for ax in (left, right):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#444444")

        context = f"CFG {row['guidance_scale']}, seed {row['seed']}, prompt {row['prompt_index']}"
        left.set_title("base\n" + context, pad=3)
        right.set_title(
            "D-GPDE\n"
            + f"Delta IR {float(row['image_reward_delta_dgpde_minus_base']):+.2f}, "
            + f"Delta CLIP {float(row['clip_score_delta_dgpde_minus_base']):+.2f}",
            pad=3,
        )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.03, wspace=0.04, hspace=0.38)
    FIGURE_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PREFIX.with_suffix(".pdf"))
    fig.savefig(FIGURE_PREFIX.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    rows = read_rows(DETAIL_CSV)
    selected = select_failures(rows)
    write_selection(selected)
    render_grid(selected)
    print(f"wrote_selection={SELECTION_CSV.relative_to(REPO_ROOT)}")
    print(f"wrote_figure={FIGURE_PREFIX.relative_to(REPO_ROOT)}.[pdf,png]")


if __name__ == "__main__":
    main()

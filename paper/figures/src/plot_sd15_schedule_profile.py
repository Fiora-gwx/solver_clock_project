#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent
if str(REPO_ROOT / "third_party/diffusers/src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party/diffusers/src"))

from diffusers import EulerDiscreteScheduler  # type: ignore  # noqa: E402


NFE = 10
GUIDANCE_SCALE = "7p5"
GUIDANCE_LABEL = "7.5"
SEED = 0
RESULT_DIR = PAPER_ROOT / "results/t2i"
PROFILE_CSV = RESULT_DIR / "sd15_euler_nfe10_cfg7p5_schedule_profile_seed0.csv"
FIGURE_PREFIX = PAPER_ROOT / "figures/sd15_euler_nfe10_cfg7p5_schedule_profile_seed0"

COLORS = {"base": "#6F7D8C", "AYS": "#2A9D8F", "D-GPDE": "#E76F51"}
MARKERS = {"base": "o", "AYS": "^", "D-GPDE": "s"}


def base_sigma_grid() -> np.ndarray:
    scheduler = EulerDiscreteScheduler.from_pretrained(
        REPO_ROOT / "checkpoints/hf/runwayml--stable-diffusion-v1-5",
        subfolder="scheduler",
        local_files_only=True,
    )
    scheduler.set_timesteps(NFE, device="cpu")
    return scheduler.sigmas.detach().cpu().float().numpy().astype(np.float64)


def ays_sigma_grid() -> np.ndarray:
    return np.load(
        REPO_ROOT / "schedules/ays_like/published/stable_diffusion_15/nfe_010/sigma_grid.npy"
    ).astype(np.float64)


def dgpde_sigma_grid() -> np.ndarray:
    path = RESULT_DIR / "schedules/sd15_euler_cfg_7p5_seed_0_nfe010_schedule.json"
    with path.open() as handle:
        payload = json.load(handle)
    return np.asarray(payload["native_schedule"], dtype=np.float64)


def write_profile_csv(schedules: dict[str, np.ndarray]) -> None:
    PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "evidence_level",
        "model",
        "solver",
        "method",
        "nfe",
        "guidance_scale",
        "seed",
        "boundary_index",
        "sigma",
        "sigma_drop_to_next",
    ]
    with PROFILE_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, grid in schedules.items():
            drops = np.abs(np.diff(grid))
            for index, sigma in enumerate(grid):
                writer.writerow(
                    {
                        "evidence_level": "matched_sd15_cfg_sweep_schedule_profile",
                        "model": "stable_diffusion_15",
                        "solver": "euler",
                        "method": method,
                        "nfe": str(NFE),
                        "guidance_scale": GUIDANCE_LABEL,
                        "seed": str(SEED),
                        "boundary_index": str(index),
                        "sigma": f"{float(sigma):.8f}",
                        "sigma_drop_to_next": f"{float(drops[index]):.8f}" if index < len(drops) else "",
                    }
                )


def plot(schedules: dict[str, np.ndarray]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.6,
            "lines.markersize": 3.2,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.7, 2.35),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        constrained_layout=True,
    )
    for method, grid in schedules.items():
        boundaries = np.arange(grid.size)
        axes[0].plot(
            boundaries,
            grid,
            marker=MARKERS[method],
            color=COLORS[method],
            label=method,
        )
        axes[1].plot(
            np.arange(1, grid.size),
            np.abs(np.diff(grid)),
            marker=MARKERS[method],
            color=COLORS[method],
            label=method,
        )
    axes[0].set_title("Sigma grid")
    axes[0].set_xlabel("boundary index")
    axes[0].set_ylabel(r"$\sigma$")
    axes[0].set_xticks([0, 2, 4, 6, 8, 10])
    axes[1].set_title("Sigma drops")
    axes[1].set_xlabel("interval index")
    axes[1].set_ylabel(r"$|\Delta\sigma|$")
    axes[1].set_xticks([1, 2, 4, 6, 8, 10])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    FIGURE_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PREFIX.with_suffix(".pdf"))
    fig.savefig(FIGURE_PREFIX.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    schedules = {
        "base": base_sigma_grid(),
        "AYS": ays_sigma_grid(),
        "D-GPDE": dgpde_sigma_grid(),
    }
    expected = NFE + 1
    for method, grid in schedules.items():
        if grid.size != expected:
            raise ValueError(f"{method} grid has {grid.size} nodes, expected {expected}.")
        if np.any(np.diff(grid) > 1.0e-8):
            raise ValueError(f"{method} sigma grid is not non-increasing.")
    write_profile_csv(schedules)
    plot(schedules)
    print(f"[schedule-profile] rows={sum(grid.size for grid in schedules.values())} csv={PROFILE_CSV}")
    print(f"[schedule-profile] figure={FIGURE_PREFIX.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

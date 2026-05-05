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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters.pndm import build_scheduler, load_native_config
from src.utils.config import load_yaml
from src.utils.nfe_budget import resolve_effective_nfe_plan


NFE = 20
SEED = 0
RESULT_DIR = PAPER_ROOT / "results/cifar10_50k"
PROFILE_CSV = RESULT_DIR / "cifar10_pndm_euler_nfe20_schedule_profile_seed0.csv"
FIGURE_PREFIX = PAPER_ROOT / "figures/cifar10_pndm_euler_nfe20_schedule_profile_seed0"

COLORS = {"base": "#6F7D8C", "linear": "#2A9D8F", "Karras": "#8E5CF7", "D-GPDE": "#E76F51"}
MARKERS = {"base": "o", "linear": "^", "Karras": "D", "D-GPDE": "s"}


def base_timesteps() -> np.ndarray:
    dataset_config = load_yaml(REPO_ROOT / "configs/datasets/cifar10.yaml")
    native = load_native_config(dataset_config["native_config"])
    schedule_cfg = native["Schedule"]
    scheduler = build_scheduler(
        "euler",
        diffusion_step=schedule_cfg["diffusion_step"],
        beta_start=schedule_cfg["beta_start"],
        beta_end=schedule_cfg["beta_end"],
        beta_schedule=schedule_cfg["type"],
    )
    plan = resolve_effective_nfe_plan("euler", NFE)
    scheduler.set_timesteps(plan.solver_steps)
    return scheduler.timesteps.detach().cpu().float().numpy().astype(np.float64)


def linear_timesteps() -> np.ndarray:
    return np.load(RESULT_DIR / "cifar10_pndm_euler_50k_linear_schedule_nfe020_timesteps.npy").astype(np.float64)


def karras_timesteps() -> np.ndarray:
    return np.load(RESULT_DIR / "cifar10_pndm_euler_50k_karras_schedule_nfe020_timesteps.npy").astype(np.float64)


def dgpde_timesteps() -> np.ndarray:
    path = RESULT_DIR / "cifar10_pndm_euler_50k_schedule_nfe020_seed0.json"
    with path.open() as handle:
        payload = json.load(handle)
    return np.asarray(payload["native_schedule"], dtype=np.float64)[:-1]


def write_profile_csv(schedules: dict[str, np.ndarray]) -> None:
    PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "evidence_level",
        "backend",
        "dataset",
        "model",
        "solver",
        "method",
        "nfe",
        "seed",
        "step_index",
        "start_timestep",
        "spacing_to_next_start",
    ]
    with PROFILE_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, grid in schedules.items():
            widths = np.abs(np.diff(grid))
            for index, timestep in enumerate(grid):
                writer.writerow(
                    {
                        "evidence_level": "three_seed_50k_fid_schedule_profile",
                        "backend": "pndm",
                        "dataset": "cifar10",
                        "model": "pndm_model_ddim_cifar10",
                        "solver": "euler",
                        "method": method,
                        "nfe": str(NFE),
                        "seed": str(SEED),
                        "step_index": str(index + 1),
                        "start_timestep": f"{float(timestep):.8f}",
                        "spacing_to_next_start": f"{float(widths[index]):.8f}" if index < len(widths) else "",
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
        gridspec_kw={"width_ratios": [1.2, 1.0]},
        constrained_layout=True,
    )
    for method, grid in schedules.items():
        step_indices = np.arange(1, grid.size + 1)
        axes[0].plot(
            step_indices,
            grid,
            marker=MARKERS[method],
            color=COLORS[method],
            label=method,
        )
        interval_indices = np.arange(1, grid.size)
        axes[1].plot(
            interval_indices,
            np.abs(np.diff(grid)),
            marker=MARKERS[method],
            color=COLORS[method],
            label=method,
        )
    axes[0].set_title("Step start grid")
    axes[0].set_xlabel("solver step")
    axes[0].set_ylabel("start timestep")
    axes[0].set_xticks([1, 5, 10, 15, 20])
    axes[1].set_title("Spacing between starts")
    axes[1].set_xlabel("start-to-start interval")
    axes[1].set_ylabel(r"$|\Delta t|$")
    axes[1].set_xticks([1, 5, 10, 15, 20])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    FIGURE_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PREFIX.with_suffix(".pdf"))
    fig.savefig(FIGURE_PREFIX.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    schedules = {
        "base": base_timesteps(),
        "linear": linear_timesteps(),
        "Karras": karras_timesteps(),
        "D-GPDE": dgpde_timesteps(),
    }
    expected = NFE
    for method, grid in schedules.items():
        if grid.size != expected:
            raise ValueError(f"{method} grid has {grid.size} nodes, expected {expected}.")
    write_profile_csv(schedules)
    plot(schedules)
    print(f"[schedule-profile] rows={sum(grid.size for grid in schedules.values())} csv={PROFILE_CSV}")
    print(f"[schedule-profile] figure={FIGURE_PREFIX.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

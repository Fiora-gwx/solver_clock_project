#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PAPER_ROOT = Path(__file__).resolve().parents[2]
FIGURE_PREFIX = PAPER_ROOT / "figures/dgpde_method_overview"

STAGES = [
    {
        "title": "Pilot teacher trajectories",
        "body": "Sample teacher marginals for the fixed model, solver, NFE, and condition.",
        "color": "#E8EDF2",
        "accent": "#4A90D9",
    },
    {
        "title": "Oracle-start residuals",
        "body": "Probe candidate edges and measure solver prediction defects in a G-metric.",
        "color": "#E8F2EE",
        "accent": "#2A9D8F",
    },
    {
        "title": "Distributional risk",
        "body": "Aggregate local defect coefficients with mean, CVaR, or mixed risk.",
        "color": "#FFF4E6",
        "accent": "#D4A252",
    },
    {
        "title": "Monitor clock",
        "body": "Build the monitor density and invert equal monitor mass.",
        "color": "#F8ECE8",
        "accent": "#E76F51",
    },
    {
        "title": "Materialized schedule",
        "body": "Export explicit timesteps or sigmas with metadata and calibration cost.",
        "color": "#EEEAF7",
        "accent": "#8E5CF7",
    },
]


def add_box(ax, x: float, y: float, width: float, height: float, title: str, body: str, color: str, accent: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="#D4D7DC",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.018,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0,
            facecolor=accent,
        )
    )
    ax.text(x + 0.034, y + height - 0.045, title, ha="left", va="top", fontsize=7.8, fontweight="bold", color="#222222")
    ax.text(x + 0.034, y + height - 0.108, fill(body, 29), ha="left", va="top", fontsize=6.7, color="#30343B", linespacing=1.18)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#6B7280") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    width = 0.275
    height = 0.28
    positions = [
        (0.055, 0.58),
        (0.362, 0.58),
        (0.669, 0.58),
        (0.208, 0.23),
        (0.515, 0.23),
    ]

    for index, (stage, (x, y)) in enumerate(zip(STAGES, positions, strict=True)):
        add_box(ax, x, y, width, height, stage["title"], stage["body"], stage["color"], stage["accent"])

    for source, target in [(0, 1), (1, 2), (3, 4)]:
        sx, sy = positions[source]
        tx, ty = positions[target]
        add_arrow(ax, (sx + width + 0.01, sy + height / 2), (tx - 0.012, ty + height / 2))
    sx, sy = positions[2]
    tx, ty = positions[3]
    add_arrow(ax, (sx + width * 0.5, sy - 0.015), (tx + width * 0.5, ty + height + 0.015))

    ax.text(0.5, 0.94, "D-GPDE calibration pipeline", ha="center", va="center", fontsize=9.4, fontweight="bold", color="#1F2933")
    ax.text(
        0.5,
        0.08,
        "All stages are schedule calibration only; model weights stay fixed.",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#4B5563",
    )

    FIGURE_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PREFIX.with_suffix(".pdf"))
    fig.savefig(FIGURE_PREFIX.with_suffix(".png"))
    plt.close(fig)
    print(f"[method-overview] figure={FIGURE_PREFIX.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

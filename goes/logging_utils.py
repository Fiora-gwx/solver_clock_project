from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import repo_path, short_config_hash


def git_commit_hash() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path("."),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit_hash(),
        "time_unix": time.time(),
    }
    try:
        import torch

        metadata["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        }
    except Exception:
        metadata["torch"] = None
    metadata["numpy"] = np.__version__
    return metadata


def make_run_dir(config: dict[str, Any], command: str) -> Path:
    root = repo_path(config["output"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    solver = str(config["solver"]["name"])
    nfe = int(config["solver"]["target_nfe"])
    run_name = f"{timestamp}_goes_{command}_{solver}_{nfe}nfe_{short_config_hash(config)}"
    path = root / run_name
    suffix = 1
    while path.exists():
        path = root / f"{run_name}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    (path / "plots").mkdir()
    (path / "paper_tables").mkdir()
    return path


def dump_json(payload: Any, path: str | Path) -> Path:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return resolved


def write_csv(rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    import csv

    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return resolved


def maybe_write_plots(
    run_dir: Path,
    *,
    candidate_grid: np.ndarray | None = None,
    edge_costs: np.ndarray | None = None,
    u_schedule: np.ndarray | None = None,
    selected_edge_costs: np.ndarray | None = None,
) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return written

    if u_schedule is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.arange(len(u_schedule)), u_schedule, marker="o")
        ax.set_xlabel("schedule index")
        ax.set_ylabel("unified coordinate u")
        ax.set_title("GOES schedule")
        fig.tight_layout()
        path = run_dir / "plots" / "schedule.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    if edge_costs is not None:
        finite = np.where(np.isfinite(edge_costs), edge_costs, np.nan)
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(finite, origin="lower", aspect="auto")
        ax.set_xlabel("edge end index")
        ax.set_ylabel("edge start index")
        ax.set_title("GOES edge costs")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        path = run_dir / "plots" / "edge_cost_heatmap.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    if selected_edge_costs is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.arange(len(selected_edge_costs)), selected_edge_costs, marker="o")
        ax.set_xlabel("selected edge")
        ax.set_ylabel("edge cost")
        ax.set_title("Selected GOES edge costs")
        fig.tight_layout()
        path = run_dir / "plots" / "selected_edge_costs.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    if candidate_grid is not None and u_schedule is not None:
        del candidate_grid
    return written


def maybe_write_nfe_quality_curve(run_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return written

    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        try:
            nfe = int(row["nfe"])
            value = float(row["final_latent_mse"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(str(row.get("schedule", "unknown")), []).append((nfe, value))
    if not grouped:
        return written

    fig, ax = plt.subplots(figsize=(5, 3))
    for schedule, values in sorted(grouped.items()):
        values = sorted(values)
        ax.plot([item[0] for item in values], [item[1] for item in values], marker="o", label=schedule)
    ax.set_xlabel("NFE")
    ax.set_ylabel("held-out final latent MSE")
    ax.set_title("NFE quality curve")
    ax.legend()
    fig.tight_layout()
    path = run_dir / "plots" / "nfe_quality_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))
    return written

#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent
SOURCE_CSV = (
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_medium_offline_proxy/metrics/"
    / "gpde_pndm_cifar10_medium_offline_proxy_smoke5k.csv"
)
CONFIG_PATH = REPO_ROOT / "configs/clocks/AYS_cifar10_medium_proxy.yaml"
SCHEDULE_ROOT = (
    REPO_ROOT
    / "outputs/gpde_pndm_cifar10_medium_offline_proxy/schedules/offline_proxy_medium/"
    / "pndm/cifar10/pndm_model_ddim_cifar10/euler"
)
OUT_DETAIL = PAPER_ROOT / "results/failure/cifar10_medium_offline_proxy_smoke5k_detail.csv"
OUT_AGGREGATE = PAPER_ROOT / "results/failure/cifar10_medium_offline_proxy_smoke5k_aggregate.csv"
OUT_CONFIG = PAPER_ROOT / "results/failure/cifar10_medium_offline_proxy_config.yaml"


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
    "optimizer_config",
    "optimizer_config_sha256",
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
    "proxy_best_fid",
    "stage_iterations",
    "optimizer_config",
    "optimizer_config_sha256",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def rel(path: str | Path) -> str:
    resolved = (REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    return str(resolved.relative_to(REPO_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float) -> str:
    return f"{value:.6f}"


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def schedule_stage_metadata(nfe: int) -> tuple[str, str]:
    meta_path = SCHEDULE_ROOT / f"nfe_{nfe:03d}" / "meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    stage_results = payload.get("stage_results", {})
    active = stage_results.get(str(nfe), {})
    best_proxy = active.get("best_proxy_value")
    iterations = active.get("iterations_ran")
    return "" if best_proxy is None else fmt(float(best_proxy)), "" if iterations is None else str(int(iterations))


def clean_detail_rows() -> list[dict[str, object]]:
    OUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CONFIG.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    config_hash = sha256(CONFIG_PATH)
    rows: list[dict[str, object]] = []
    for row in read_csv(SOURCE_CSV):
        if row.get("status") != "OK":
            continue
        rows.append(
            {
                "evidence_level": "medium_offline_proxy_5k_smoke_not_paper_grade",
                "backend": row["backend"],
                "dataset": row["dataset"],
                "model": row["model_asset"],
                "solver": row["solver"],
                "method": "offline-proxy-medium",
                "nfe": int(row["nfe"]),
                "seed": int(row["seed"]),
                "num_samples": int(row["num_samples"]),
                "metric": "FID",
                "metric_direction": "lower_is_better",
                "fid": fmt(float(row["fid"])),
                "source_csv": rel(SOURCE_CSV),
                "source_output_dir": rel(row["output_dir"]),
                "source_schedule_dir": rel(row["schedule_dir"]),
                "optimizer_config": rel(OUT_CONFIG),
                "optimizer_config_sha256": config_hash,
            }
        )
    expected = {(nfe, seed) for nfe in (10, 20) for seed in (0, 1, 2)}
    actual = {(int(row["nfe"]), int(row["seed"])) for row in rows}
    if actual != expected:
        raise RuntimeError(f"Unexpected medium offline-proxy coverage: expected={sorted(expected)} actual={sorted(actual)}")
    return sorted(rows, key=lambda item: (int(item["nfe"]), int(item["seed"])))


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["nfe"])].append(row)
    output: list[dict[str, object]] = []
    for nfe in sorted(grouped):
        group = grouped[nfe]
        values = [float(row["fid"]) for row in group]
        seeds = sorted(int(row["seed"]) for row in group)
        best_proxy, iterations = schedule_stage_metadata(nfe)
        first = group[0]
        output.append(
            {
                "evidence_level": "medium_offline_proxy_5k_smoke_not_paper_grade",
                "backend": first["backend"],
                "dataset": first["dataset"],
                "model": first["model"],
                "solver": first["solver"],
                "method": first["method"],
                "nfe": nfe,
                "seed_count": len(group),
                "seeds": ",".join(str(seed) for seed in seeds),
                "num_samples": first["num_samples"],
                "metric": "FID",
                "metric_direction": "lower_is_better",
                "fid_mean": fmt(statistics.mean(values)),
                "fid_std": fmt(std(values)),
                "fid_sem": fmt(sem(values)),
                "proxy_best_fid": best_proxy,
                "stage_iterations": iterations,
                "optimizer_config": first["optimizer_config"],
                "optimizer_config_sha256": first["optimizer_config_sha256"],
            }
        )
    return output


def main() -> None:
    rows = clean_detail_rows()
    aggregate_rows = aggregate(rows)
    write_csv(OUT_DETAIL, DETAIL_FIELDS, rows)
    write_csv(OUT_AGGREGATE, AGGREGATE_FIELDS, aggregate_rows)
    print(f"[medium-offline-proxy] detail_rows={len(rows)} detail={OUT_DETAIL}")
    print(f"[medium-offline-proxy] aggregate_rows={len(aggregate_rows)} aggregate={OUT_AGGREGATE}")


if __name__ == "__main__":
    main()

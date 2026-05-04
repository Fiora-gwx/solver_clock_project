from __future__ import annotations

import csv
import sys
from pathlib import Path

from scripts.eval import pairwise_win_rates


def _write_detail_csv(path: Path) -> None:
    rows = []
    for schedule, values in {
        "base": [0.1, 0.2],
        "AYS": [0.2, 0.3],
        "GOES": [0.3, 0.25],
        "goes[rho_0]": [0.15, 0.35],
    }.items():
        for prompt_index, value in enumerate(values):
            rows.append(
                {
                    "model_asset": "hf_sd35_medium",
                    "solver": "flow_euler",
                    "nfe": "10",
                    "seed": "0",
                    "prompt_index": str(prompt_index),
                    "guidance_scale": "7.5",
                    "schedule": schedule,
                    "clip_score": str(value),
                    "image_reward": "",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pairwise_win_rates_compares_goes_and_labeled_goes_variants(tmp_path, monkeypatch) -> None:
    detail_csv = tmp_path / "detail.csv"
    output_csv = tmp_path / "pairwise.csv"
    _write_detail_csv(detail_csv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pairwise_win_rates.py",
            "--input-csv",
            str(detail_csv),
            "--output-csv",
            str(output_csv),
            "--metrics",
            "clip_score",
        ],
    )

    pairwise_win_rates.main()

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_comparison = {row["comparison"]: row for row in rows}

    assert pairwise_win_rates.schedule_family("goes[rho_0]") == "GOES[rho_0]"
    assert "GOES vs base" in by_comparison
    assert "GOES vs AYS" in by_comparison
    assert "GOES[rho_0] vs base" in by_comparison
    assert "GOES[rho_0] vs AYS" in by_comparison
    assert by_comparison["GOES vs base"]["num_pairs"] == "2"
    assert by_comparison["GOES vs base"]["win_rate"] == "1.0"
    assert by_comparison["GOES vs base"]["bootstrap_samples"] == "1000"
    assert by_comparison["GOES vs base"]["ci_level"] == "0.95"
    assert by_comparison["GOES vs base"]["win_rate_bootstrap_se"]
    assert by_comparison["GOES vs base"]["win_rate_ci_low"]
    assert by_comparison["GOES vs base"]["win_rate_ci_high"]
    assert by_comparison["GOES vs base"]["mean_delta_bootstrap_se"]
    assert by_comparison["GOES vs base"]["mean_delta_ci_low"]
    assert by_comparison["GOES vs base"]["mean_delta_ci_high"]


def test_pairwise_win_rates_allows_disabling_bootstrap_uncertainty(tmp_path, monkeypatch) -> None:
    detail_csv = tmp_path / "detail.csv"
    output_csv = tmp_path / "pairwise.csv"
    _write_detail_csv(detail_csv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pairwise_win_rates.py",
            "--input-csv",
            str(detail_csv),
            "--output-csv",
            str(output_csv),
            "--metrics",
            "clip_score",
            "--bootstrap-samples",
            "0",
        ],
    )

    pairwise_win_rates.main()

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = {item["comparison"]: item for item in rows}["GOES vs base"]
    assert row["bootstrap_samples"] == "0"
    assert row["win_rate_bootstrap_se"] == ""
    assert row["mean_delta_ci_high"] == ""


def test_pairwise_win_rates_rejects_empty_pairing_result(tmp_path, monkeypatch) -> None:
    detail_csv = tmp_path / "detail.csv"
    output_csv = tmp_path / "pairwise.csv"
    rows = [
        {
            "model_asset": "hf_sd35_medium",
            "solver": "flow_euler",
            "nfe": "10",
            "seed": "0",
            "prompt_index": "0",
            "guidance_scale": "7.5",
            "schedule": "base",
            "clip_score": "0.1",
            "image_reward": "",
        }
    ]
    with detail_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pairwise_win_rates.py",
            "--input-csv",
            str(detail_csv),
            "--output-csv",
            str(output_csv),
            "--metrics",
            "clip_score",
        ],
    )

    try:
        pairwise_win_rates.main()
    except ValueError as error:
        assert "No paired schedule comparisons found" in str(error)
    else:
        raise AssertionError("Pairwise scoring should reject empty comparison outputs.")

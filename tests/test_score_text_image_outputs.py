from __future__ import annotations

import json
import sys

from scripts.eval import score_text_image_outputs


def test_bootstrap_mean_uncertainty_is_deterministic() -> None:
    first = score_text_image_outputs.bootstrap_mean_uncertainty(
        [1.0, 2.0, 3.0, 4.0],
        samples=50,
        seed=123,
        ci_level=0.9,
    )
    second = score_text_image_outputs.bootstrap_mean_uncertainty(
        [1.0, 2.0, 3.0, 4.0],
        samples=50,
        seed=123,
        ci_level=0.9,
    )

    assert first == second
    assert first["bootstrap_se"]
    assert first["ci_low"] <= first["ci_high"]


def test_metric_summary_fields_blank_uncertainty_without_scores() -> None:
    fields = score_text_image_outputs.metric_summary_fields(
        "clip_score",
        [None, None],
        bootstrap_samples=10,
        bootstrap_seed=0,
        ci_level=0.95,
    )

    assert fields["clip_score_mean"] == ""
    assert fields["clip_score_bootstrap_se"] == ""
    assert fields["clip_score_ci_low"] == ""
    assert fields["clip_score_ci_high"] == ""


def test_score_text_image_outputs_rejects_runs_without_images(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({"output_dir": str(run_dir)}), encoding="utf-8")
    detail_csv = tmp_path / "detail.csv"
    aggregate_csv = tmp_path / "aggregate.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_text_image_outputs.py",
            "--run-dir",
            str(run_dir),
            "--metrics",
            "",
            "--output-csv",
            str(detail_csv),
            "--aggregate-csv",
            str(aggregate_csv),
        ],
    )

    try:
        score_text_image_outputs.main()
    except ValueError as error:
        assert "No scorable images found" in str(error)
    else:
        raise AssertionError("Scoring should reject run manifests that have no images.")

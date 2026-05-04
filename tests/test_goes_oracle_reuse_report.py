import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _load_report_module():
    path = Path(__file__).resolve().parents[1] / "scripts/report_goes_oracle_reuse_cost.py"
    spec = importlib.util.spec_from_file_location("report_goes_oracle_reuse_cost_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_schedule(
    root: Path,
    *,
    solver: str,
    nfe: int,
    oracle_cache_key: str,
    loaded_from_cache: bool,
) -> Path:
    schedule_dir = root / solver / f"nfe_{nfe:03d}"
    schedule_dir.mkdir(parents=True)
    breakdown = {
        "num_samples": 8,
        "cfg_multiplier": 2,
        "oracle_cost_per_sample": 1029,
        "edge_cost_per_sample": 528,
        "total_model_eval_equivalents": 8 * 2 * (1029 + 528),
    }
    schedule = {
        "method": "GOES",
        "solver": solver,
        "target_nfe": nfe,
        "model_asset": "hf_sd35_medium",
        "prompt_asset": "diffusers_smoke_prompts",
        "guidance_scale": 7.5,
        "seed": 0,
        "oracle_cache_key": oracle_cache_key,
        "schedule_hash": f"hash-{solver}-{nfe}",
        "edge_objective": 1.25,
        "calibration_cost_breakdown": breakdown,
    }
    (schedule_dir / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    run_metadata = {
        "oracle_loaded_from_cache": loaded_from_cache,
        "oracle_build_or_load_seconds": 0.25 if loaded_from_cache else 3.0,
    }
    (schedule_dir / "run_metadata.json").write_text(json.dumps(run_metadata), encoding="utf-8")
    return schedule_dir


def test_goes_oracle_reuse_report_summarizes_shared_cache_cost(tmp_path) -> None:
    module = _load_report_module()
    root = tmp_path / "schedules"
    _write_schedule(root, solver="flow_euler", nfe=5, oracle_cache_key="shared-key", loaded_from_cache=False)
    _write_schedule(root, solver="flow_heun", nfe=9, oracle_cache_key="shared-key", loaded_from_cache=True)

    rows = module.build_report([root])

    assert len(rows) == 2
    assert {row["solver"] for row in rows} == {"flow_euler", "flow_heun"}
    for row in rows:
        assert row["status"] == "OK"
        assert row["schedules_sharing_cache_count"] == 2
        assert row["solvers_sharing_cache"] == "flow_euler,flow_heun"
        assert row["nfes_sharing_cache"] == "5,9"
        assert row["per_schedule_oracle_model_eval_equivalents"] == 8 * 2 * 1029
        assert row["per_schedule_edge_model_eval_equivalents"] == 8 * 2 * 528
        assert row["separate_total_model_eval_equivalents"] == 2 * 8 * 2 * (1029 + 528)
        assert row["shared_total_model_eval_equivalents"] == 8 * 2 * 1029 + 2 * 8 * 2 * 528
        assert row["saved_model_eval_equivalents"] == 8 * 2 * 1029

    output_csv = tmp_path / "oracle_reuse_cost.csv"
    module.write_csv(output_csv, rows)
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 2
    assert written[0]["oracle_cache_key"] == "shared-key"


def test_goes_oracle_reuse_report_rejects_empty_roots(tmp_path) -> None:
    module = _load_report_module()

    with pytest.raises(ValueError, match="No GOES schedule.json files"):
        module.build_report([tmp_path])

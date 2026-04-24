from pathlib import Path
from types import SimpleNamespace

from scripts.run.compact_metrics_csv import build_row_filter, canonical_schedule_label
from scripts.run.export_defect_clock_schedule import _build_profile_meta, profile_cache_dir, schedule_family_label
from scripts.run.run_experiment_config import (
    build_invocations,
    canonical_schedule_name,
    is_materializable_schedule,
    resolve_execution_config,
)


def _build_args() -> SimpleNamespace:
    return SimpleNamespace(
        manifest="configs/assets_manifest.yaml",
        runtime_config="configs/runtime_envs.yaml",
        models_config="configs/models/modern_diffusers.yaml",
        clock_config="configs/clocks/SADB.yaml",
        ays_config="configs/clocks/AYS.yaml",
        outputs_root="outputs/samples",
        metrics_root="outputs/metrics",
        dtype="bfloat16",
        execute=False,
        materialize_schedules=False,
        limit=None,
        shard_count=1,
        shard_index=0,
        skip_preview=False,
        skip_existing=False,
        distributed_child=False,
        experiment_config="",
    )


def test_canonical_schedule_name_accepts_only_sadb_clock_alias() -> None:
    assert canonical_schedule_name("SADB") == ("SADB", "SADB")
    assert canonical_schedule_label("sadb") == "SADB"
    try:
        canonical_schedule_name("LCS-1")
    except ValueError as error:
        assert "Unsupported schedule name" in str(error)
    else:
        raise AssertionError("Old LCS aliases should not be accepted.")


def test_materializable_schedule_registry_includes_sadb() -> None:
    assert is_materializable_schedule("pndm", "SADB")
    assert is_materializable_schedule("diffusers", "SADB")
    assert not is_materializable_schedule("pndm", "LCS-1")
    assert not is_materializable_schedule("diffusers", "LCS-1")


def test_compact_metrics_filter_accepts_sadb_rows(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "solvers:\n"
        "  - euler\n"
        "schedules:\n"
        "  - SADB\n",
        encoding="utf-8",
    )
    keep_row = build_row_filter(config_path)
    assert keep_row({"solver": "euler", "schedule": "SADB"})
    assert not keep_row({"solver": "euler", "schedule": "LCS-1"})


def test_profile_cache_dir_records_step_refinement_clock() -> None:
    cache_root = Path("outputs/cache/sadb_profiles")
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        backend="pndm",
        dataset_name="cifar10",
        model_asset="model_a",
        solver="heun2",
        calibration_solver="heun2",
        physical_grid_size=65,
        pilot_batch_size=8,
        pilot_num_batches=4,
        pilot_observation_microbatch=4,
        smoothing_window=3,
        epsilon=1.0e-12,
        q_min=1.05,
        q_max=6.0,
        seed=0,
        model_output_type="epsilon",
        coordinate_domain="sigmas",
    )
    meta = _build_profile_meta(
        backend="pndm",
        model_asset="model_a",
        solver="heun2",
        calibration_solver="heun2",
        physical_grid_size=65,
        pilot_batch_size=8,
        pilot_num_batches=4,
        pilot_observation_microbatch=4,
        epsilon=1.0e-12,
        smoothing_window=3,
        q_min=1.05,
        q_max=6.0,
        model_output_type="epsilon",
        coordinate_domain="sigmas",
    )
    assert schedule_family_label() == "SADB"
    assert "SADB" in str(cache_dir)
    assert meta["schedule_family"] == "SADB"
    assert meta["estimator"] == "step_refinement"
    assert meta["calibration_solver"] == "heun2"
    assert meta["coordinate_domain"] == "sigmas"


def test_build_invocations_expands_sadb_schedules_for_pndm_and_diffusers() -> None:
    args = _build_args()
    pndm_config = {
        "name": "sadb_test_pndm",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["euler"],
        "schedules": ["SADB"],
        "nfes": [6],
    }
    execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=execution)
    assert len(pndm_invocations) == 1
    assert any("export_defect_clock_schedule.py" in step.arguments[0] for step in pndm_invocations[0].prepare_steps)
    assert all(invocation.materializable for invocation in pndm_invocations)
    assert any("SADB" in str(invocation.schedule_dir) for invocation in pndm_invocations)

    diffusers_config = {
        "name": "sadb_test_diffusers",
        "backend": "diffusers",
        "models": ["sd35_medium"],
        "solvers": ["flow_euler"],
        "schedules": ["SADB"],
        "nfes": [8],
    }
    execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=execution)
    assert len(diffusers_invocations) == 1
    assert any("export_defect_clock_schedule.py" in step.arguments[0] for step in diffusers_invocations[0].prepare_steps)
    assert any("SADB" in str(invocation.schedule_dir) for invocation in diffusers_invocations)


def test_build_invocations_rejects_custom_schedules_for_dpm_solver() -> None:
    args = _build_args()
    pndm_config = {
        "name": "dpm_custom_schedule",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["dpm_solver_lu"],
        "schedules": ["base", "SADB"],
        "nfes": [6],
    }
    execution = resolve_execution_config(pndm_config, args)
    try:
        build_invocations(args, pndm_config, execution_config=execution)
    except ValueError as error:
        assert "base-only" in str(error)
    else:
        raise AssertionError("Expected custom DPM solver schedules to be rejected.")

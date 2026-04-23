from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.run.compact_metrics_csv import build_row_filter, canonical_schedule_label
from scripts.run.export_lcs_schedule import _build_profile_meta, profile_cache_dir, schedule_family_label
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
        clock_config="configs/clocks/V_a.yaml",
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


def test_canonical_schedule_name_accepts_lcs_aliases() -> None:
    assert canonical_schedule_name("LCS-1") == ("LCS-1", "LCS_1")
    assert canonical_schedule_name("LCS_2") == ("LCS-2", "LCS_2")
    assert canonical_schedule_label("LCS_1") == "LCS-1"
    assert canonical_schedule_label("LCS-2") == "LCS-2"


def test_materializable_schedule_registry_includes_lcs() -> None:
    assert is_materializable_schedule("pndm", "LCS-1")
    assert is_materializable_schedule("pndm", "LCS-2")
    assert is_materializable_schedule("diffusers", "LCS-1")
    assert is_materializable_schedule("diffusers", "LCS-2")


def test_compact_metrics_filter_accepts_lcs_rows(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "solvers:\n"
        "  - euler\n"
        "schedules:\n"
        "  - LCS-1\n",
        encoding="utf-8",
    )
    keep_row = build_row_filter(config_path)
    assert keep_row({"solver": "euler", "schedule": "LCS-1"})
    assert not keep_row({"solver": "euler", "schedule": "V_a"})


def test_profile_cache_dir_distinguishes_lcs_orders() -> None:
    cache_root = Path("outputs/cache/lcs_profiles")
    order_one = profile_cache_dir(
        cache_root=cache_root,
        backend="pndm",
        dataset_name="cifar10",
        model_asset="model_a",
        order=1,
        estimator="material_derivative",
        pilot_solver="euler",
        physical_grid_size=65,
        pilot_batch_size=8,
        pilot_num_batches=4,
        pilot_observation_microbatch=4,
        smoothing_window=3,
        epsilon=1.0e-6,
        seed=0,
        model_output_type="epsilon",
        coordinate_domain="timesteps",
    )
    order_two = profile_cache_dir(
        cache_root=cache_root,
        backend="pndm",
        dataset_name="cifar10",
        model_asset="model_a",
        order=2,
        estimator="step_doubling",
        pilot_solver="heun2",
        physical_grid_size=65,
        pilot_batch_size=8,
        pilot_num_batches=4,
        pilot_observation_microbatch=4,
        smoothing_window=3,
        epsilon=1.0e-6,
        seed=0,
        model_output_type="epsilon",
        coordinate_domain="sigmas",
    )
    assert order_one != order_two
    assert schedule_family_label(1) == "LCS-1"
    assert schedule_family_label(2) == "LCS-2"
    meta = _build_profile_meta(
        backend="pndm",
        model_asset="model_a",
        order=2,
        estimator="step_doubling",
        pilot_solver="heun2",
        physical_grid_size=65,
        pilot_batch_size=8,
        pilot_num_batches=4,
        pilot_observation_microbatch=4,
        epsilon=1.0e-6,
        smoothing_window=3,
        model_output_type="epsilon",
        cache_version=2,
        coordinate_domain="sigmas",
    )
    assert meta["schedule_family"] == "LCS-2"
    assert meta["lcs_order"] == 2
    assert meta["estimator"] == "step_doubling"
    assert meta["coordinate_domain"] == "sigmas"


def test_build_invocations_expands_lcs_schedules_for_pndm_and_diffusers() -> None:
    args = _build_args()
    pndm_config = {
        "name": "lcs_test_pndm",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["euler"],
        "schedules": ["LCS-1", "LCS-2"],
        "nfes": [6],
    }
    execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=execution)
    assert len(pndm_invocations) == 2
    assert any("export_lcs_schedule.py" in step.arguments[0] for step in pndm_invocations[0].prepare_steps)
    assert all(invocation.materializable for invocation in pndm_invocations)
    assert any("LCS_1" in str(invocation.schedule_dir) for invocation in pndm_invocations)
    assert any("LCS_2" in str(invocation.schedule_dir) for invocation in pndm_invocations)

    diffusers_config = {
        "name": "lcs_test_diffusers",
        "backend": "diffusers",
        "models": ["sd35_medium"],
        "solvers": ["flow_euler"],
        "schedules": ["LCS-1", "LCS-2"],
        "nfes": [8],
    }
    execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=execution)
    assert len(diffusers_invocations) == 2
    assert any("export_lcs_schedule.py" in step.arguments[0] for step in diffusers_invocations[0].prepare_steps)
    assert any("LCS_1" in str(invocation.schedule_dir) for invocation in diffusers_invocations)
    assert any("LCS_2" in str(invocation.schedule_dir) for invocation in diffusers_invocations)


def test_build_invocations_rejects_custom_schedules_for_both_dpm_variants() -> None:
    args = _build_args()
    pndm_config = {
        "name": "dpm_variant_overlap",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["dpm_solver_lu", "dpm_solver_default"],
        "schedules": ["base", "LCS-1"],
        "nfes": [6],
    }
    execution = resolve_execution_config(pndm_config, args)
    try:
        build_invocations(args, pndm_config, execution_config=execution)
    except ValueError as error:
        assert "same explicit sigma-grid execution path" in str(error)
    else:
        raise AssertionError("Expected overlapping custom DPM solver schedules to be rejected.")

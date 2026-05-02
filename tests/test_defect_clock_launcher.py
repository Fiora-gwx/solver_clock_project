from pathlib import Path
from types import SimpleNamespace

from scripts.run.compact_metrics_csv import build_row_filter, canonical_schedule_label
import numpy as np

from scripts.run.export_defect_clock_schedule import (
    _build_profile_meta,
    limit_schedule_step_sizes,
    profile_cache_dir,
    schedule_family_label,
)
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
    assert canonical_schedule_name("RI_SADB") == ("RI_SADB", "RI_SADB")
    assert canonical_schedule_label("sadb") == "SADB"
    assert canonical_schedule_label("ri_sadb") == "RI_SADB"
    try:
        canonical_schedule_name("LCS-1")
    except ValueError as error:
        assert "Unsupported schedule name" in str(error)
    else:
        raise AssertionError("Old LCS aliases should not be accepted.")


def test_materializable_schedule_registry_includes_sadb() -> None:
    assert is_materializable_schedule("pndm", "SADB")
    assert is_materializable_schedule("diffusers", "SADB")
    assert is_materializable_schedule("pndm", "RI_SADB")
    assert is_materializable_schedule("diffusers", "RI_SADB")
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


def test_profile_cache_dir_records_ri_sadb_parameters() -> None:
    cache_root = Path("outputs/cache/sadb_profiles")
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        schedule_family="RI_SADB",
        backend="pndm",
        dataset_name="cifar10",
        model_asset="model_a",
        solver="euler",
        calibration_solver="euler",
        estimator="ri_sadb",
        physical_grid_size=17,
        pilot_batch_size=4,
        pilot_num_batches=1,
        pilot_observation_microbatch=2,
        smoothing_window=1,
        epsilon=1.0e-12,
        q_min=1.05,
        q_max=6.0,
        seed=0,
        model_output_type="epsilon",
        coordinate_domain="timesteps",
        target_nfe=10,
        target_steps=10,
        eta=0.25,
        beta=0.0,
        ell_scale="step",
        ri_agg="mean",
    )
    meta = _build_profile_meta(
        schedule_family="RI_SADB",
        backend="pndm",
        model_asset="model_a",
        solver="euler",
        calibration_solver="euler",
        estimator="ri_sadb",
        physical_grid_size=17,
        pilot_batch_size=4,
        pilot_num_batches=1,
        pilot_observation_microbatch=2,
        epsilon=1.0e-12,
        smoothing_window=1,
        q_min=1.05,
        q_max=6.0,
        model_output_type="epsilon",
        coordinate_domain="timesteps",
        target_nfe=10,
        target_steps=10,
        eta=0.25,
        beta=0.0,
        ell_scale="step",
        ri_agg="mean",
    )

    assert "RI_SADB" in str(cache_dir)
    assert "eta_0.25" in str(cache_dir)
    assert "beta_0" in str(cache_dir)
    assert meta["schedule_family"] == "RI_SADB"
    assert meta["eta"] == 0.25
    assert meta["beta"] == 0.0
    assert meta["ri_formula_version"] == 1


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


def test_step_limiter_does_not_collapse_to_reference_schedule() -> None:
    reference = np.linspace(10.0, 0.0, 11, dtype=np.float64)
    nodes = np.asarray([10.0, 8.7, 7.2, 6.3, 5.4, 4.5, 3.6, 2.2, 1.1, 0.35, 0.0], dtype=np.float64)

    limited, meta = limit_schedule_step_sizes(
        nodes,
        reference,
        max_dt_factor=1.5,
        max_neighbor_ratio=1.8,
    )

    assert meta["step_limiter_enabled"]
    assert np.isclose(limited[0], nodes[0])
    assert np.isclose(limited[-1], nodes[-1])
    assert not np.allclose(limited, reference)
    assert np.max(np.abs(np.diff(limited))) <= 1.5 * np.mean(np.abs(np.diff(reference))) + 1.0e-8


def test_step_limiter_preserves_nonuniform_schedule_after_timestep_snap() -> None:
    reference = np.asarray([999.0, 899.0, 799.0, 699.0, 599.0, 500.0, 400.0, 300.0, 200.0, 100.0, 0.0])
    nodes = np.asarray([999.0, 930.2, 849.4, 756.1, 648.3, 522.0, 376.5, 215.4, 80.3, 13.1, 0.0])

    limited, _ = limit_schedule_step_sizes(
        nodes,
        reference,
        max_dt_factor=1.5,
        max_neighbor_ratio=1.8,
    )
    snapped = np.rint(limited[:-1])

    assert not np.allclose(snapped, reference[:-1])


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


def test_build_invocations_expands_ri_sadb_schedule_for_pndm() -> None:
    args = _build_args()
    pndm_config = {
        "name": "ri_sadb_test_pndm",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["euler"],
        "schedules": ["RI_SADB"],
        "nfes": [10],
        "schedule_clock_configs": {
            "RI_SADB": "configs/clocks/RI_SADB_cifar10_smoke.yaml",
        },
    }
    execution = resolve_execution_config(pndm_config, args)
    invocations = build_invocations(args, pndm_config, execution_config=execution)
    assert len(invocations) == 1
    assert invocations[0].materializable
    assert any("export_defect_clock_schedule.py" in step.arguments[0] for step in invocations[0].prepare_steps)
    assert "RI_SADB" in str(invocations[0].schedule_dir)
    assert ":RI_SADB:" in invocations[0].label


def test_build_invocations_expands_diffusers_seeds_and_published_ays() -> None:
    args = _build_args()
    config = {
        "name": "diffusers_ays_seeds",
        "backend": "diffusers",
        "models": ["stable_diffusion_15"],
        "solvers": ["dpm_solver_pp"],
        "schedules": ["base", "AYS"],
        "nfes": [10],
        "seeds": [0, 1, 2],
    }
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)
    assert len(invocations) == 6
    labels = [invocation.label for invocation in invocations]
    assert any(":base:seed_002:nfe_010" in label for label in labels)
    ays_invocations = [invocation for invocation in invocations if ":ays:" in invocation.label]
    assert len(ays_invocations) == 3
    assert all("schedules/ays_like/published/stable_diffusion_15/nfe_010" in str(item.schedule_dir) for item in ays_invocations)
    assert all("seed_" in str(item.output_dir) for item in invocations)


def test_build_invocations_rejects_non_10_step_published_diffusers_ays() -> None:
    args = _build_args()
    config = {
        "name": "diffusers_bad_ays_nfe",
        "backend": "diffusers",
        "models": ["stable_diffusion_15"],
        "solvers": ["dpm_solver_pp"],
        "schedules": ["AYS"],
        "nfes": [8],
    }
    execution = resolve_execution_config(config, args)
    try:
        build_invocations(args, config, execution_config=execution)
    except ValueError as error:
        assert "10-step only" in str(error)
    else:
        raise AssertionError("Expected published diffusers AYS to reject nfe != 10.")


def test_build_invocations_expands_labeled_sadb_clock_variants() -> None:
    args = _build_args()
    config = {
        "name": "diffusers_sadb_variants",
        "backend": "diffusers",
        "models": ["stable_diffusion_15"],
        "solvers": ["dpm_solver_pp"],
        "schedules": ["SADB"],
        "nfes": [10],
        "seeds": [0],
        "schedule_clock_configs": {
            "SADB": {
                "small": "configs/clocks/SADB_diffusers_sd_small.yaml",
                "medium": "configs/clocks/SADB_diffusers_sd_medium.yaml",
            }
        },
    }
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)
    assert len(invocations) == 2
    assert {invocation.label.split(":")[3] for invocation in invocations} == {"SADB[small]", "SADB[medium]"}
    assert all(invocation.prepare_steps for invocation in invocations)
    assert any("/small/" in str(invocation.schedule_dir) for invocation in invocations)
    assert any("/medium/" in str(invocation.schedule_dir) for invocation in invocations)


def test_build_invocations_partitions_diffusers_guidance_scale_schedules() -> None:
    args = _build_args()
    config = {
        "name": "sdxl_guidance_partition",
        "backend": "diffusers",
        "models": ["sdxl"],
        "solvers": ["dpm_solver_pp"],
        "schedules": ["base", "SADB"],
        "nfes": [10],
        "seeds": [0],
        "guidance_scales": [3.0, 5.0],
        "schedule_clock_configs": {
            "SADB": "configs/clocks/SADB_diffusers_sd_medium.yaml",
        },
    }
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)
    assert len(invocations) == 4
    sadb_invocations = [invocation for invocation in invocations if ":SADB:" in invocation.label]
    assert len(sadb_invocations) == 2
    assert any("cfg_3" in str(invocation.schedule_dir) for invocation in sadb_invocations)
    assert any("cfg_5" in str(invocation.schedule_dir) for invocation in sadb_invocations)
    assert all("cfg_" in str(invocation.output_dir) for invocation in invocations)


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

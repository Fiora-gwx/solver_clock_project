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
        clock_config="configs/clocks/FP_CLOCK.yaml",
        ays_config="configs/clocks/AYS.yaml",
        outputs_root="outputs",
        metrics_root="",
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


def test_canonical_schedule_name_accepts_only_active_clock_aliases() -> None:
    assert canonical_schedule_name("LEGACY_SADB") == ("LEGACY_SADB", "LEGACY_SADB")
    assert canonical_schedule_name("FP_CLOCK") == ("FP_CLOCK", "FP_CLOCK")
    assert canonical_schedule_label("legacy_sadb") == "LEGACY_SADB"
    assert canonical_schedule_label("fp_clock") == "FP_CLOCK"
    for retired in ("SADB", "RI_SADB"):
        try:
            canonical_schedule_name(retired)
        except ValueError as error:
            assert "Unsupported schedule name" in str(error)
        else:
            raise AssertionError(f"{retired} should not be accepted in the main launcher.")
    try:
        canonical_schedule_name("LCS-1")
    except ValueError as error:
        assert "Unsupported schedule name" in str(error)
    else:
        raise AssertionError("Old LCS aliases should not be accepted.")


def test_materializable_schedule_registry_includes_active_clocks() -> None:
    assert is_materializable_schedule("pndm", "LEGACY_SADB")
    assert is_materializable_schedule("diffusers", "LEGACY_SADB")
    assert is_materializable_schedule("pndm", "FP_CLOCK")
    assert is_materializable_schedule("diffusers", "FP_CLOCK")
    assert not is_materializable_schedule("pndm", "SADB")
    assert not is_materializable_schedule("pndm", "RI_SADB")
    assert not is_materializable_schedule("pndm", "LCS-1")
    assert not is_materializable_schedule("diffusers", "LCS-1")


def test_compact_metrics_filter_accepts_fp_clock_rows(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "solvers:\n"
        "  - euler\n"
        "schedules:\n"
        "  - FP_CLOCK\n",
        encoding="utf-8",
    )
    keep_row = build_row_filter(config_path)
    assert keep_row({"solver": "euler", "schedule": "FP_CLOCK"})
    assert not keep_row({"solver": "euler", "schedule": "LCS-1"})


def test_profile_cache_dir_records_fp_clock_parameters() -> None:
    cache_root = Path("outputs/example_experiment/schedules/_profile_cache")
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        schedule_family="FP_CLOCK",
        backend="pndm",
        dataset_name="cifar10",
        model_asset="model_a",
        solver="euler",
        calibration_solver="euler",
        estimator="fp_clock",
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
    )
    meta = _build_profile_meta(
        schedule_family="FP_CLOCK",
        backend="pndm",
        model_asset="model_a",
        solver="euler",
        calibration_solver="euler",
        estimator="fp_clock",
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
    )

    assert "FP_CLOCK" in str(cache_dir)
    assert "nfe_10" in str(cache_dir)
    assert "steps_10" in str(cache_dir)
    assert meta["schedule_family"] == "FP_CLOCK"
    assert meta["target_nfe"] == 10
    assert meta["target_steps"] == 10
    assert meta["calibration_method"] == "frenet_projected_richardson_arc_pullback"


def test_profile_cache_dir_partitions_trajectory_window_fp_clock() -> None:
    cache_root = Path("outputs/example_experiment/schedules/_profile_cache")
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        schedule_family="FP_CLOCK",
        backend="diffusers",
        dataset_name=None,
        model_asset="model_a",
        solver="flow_stork4_1st",
        calibration_solver="flow_stork4_1st",
        estimator="trajectory_window",
        physical_grid_size=17,
        physical_grid_mode="official_base",
        pilot_batch_size=2,
        pilot_num_batches=1,
        pilot_observation_microbatch=1,
        smoothing_window=1,
        epsilon=1.0e-12,
        q_min=1.05,
        q_max=6.0,
        seed=0,
        prompt_tag="diffusers_smoke_prompts",
        height=512,
        width=512,
        guidance_scale=3.5,
        model_output_type="flow",
        coordinate_domain="sigmas",
        target_nfe=10,
        target_steps=10,
        multires_nfes=(16, 32, 64),
        window_size=4,
    )
    meta = _build_profile_meta(
        schedule_family="FP_CLOCK",
        backend="diffusers",
        model_asset="model_a",
        solver="flow_stork4_1st",
        calibration_solver="flow_stork4_1st",
        estimator="trajectory_window",
        physical_grid_size=17,
        physical_grid_mode="official_base",
        pilot_batch_size=2,
        pilot_num_batches=1,
        pilot_observation_microbatch=1,
        epsilon=1.0e-12,
        smoothing_window=1,
        q_min=1.05,
        q_max=6.0,
        model_output_type="flow",
        coordinate_domain="sigmas",
        target_nfe=10,
        target_steps=10,
        extra={
            "defect_estimator": "trajectory_window",
            "multires_nfes": [16, 32, 64],
            "grid_mode": "official_base",
            "window_size": 4,
            "solver_order": 4,
            "guidance_scale": 3.5,
        },
    )

    assert "trajectory_window" in str(cache_dir)
    assert "multires_16_32_64" in str(cache_dir)
    assert "window_4" in str(cache_dir)
    assert "cfg_3.5" in str(cache_dir)
    assert meta["calibration_method"] == "target_solver_official_base_trajectory_window"
    assert meta["defect_estimator"] == "trajectory_window"
    assert meta["grid_mode"] == "official_base"


def test_profile_cache_dir_records_step_refinement_clock() -> None:
    cache_root = Path("outputs/example_experiment/schedules/_profile_cache")
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        schedule_family="LEGACY_SADB",
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
        schedule_family="LEGACY_SADB",
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
    assert schedule_family_label() == "FP_CLOCK"
    assert "LEGACY_SADB" in str(cache_dir)
    assert meta["schedule_family"] == "LEGACY_SADB"
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


def test_build_invocations_expands_active_clock_schedules_for_pndm_and_diffusers() -> None:
    args = _build_args()
    pndm_config = {
        "name": "fp_clock_test_pndm",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["euler"],
        "schedules": ["FP_CLOCK"],
        "nfes": [6],
    }
    execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=execution)
    assert len(pndm_invocations) == 1
    assert any("export_defect_clock_schedule.py" in step.arguments[0] for step in pndm_invocations[0].prepare_steps)
    assert all(invocation.materializable for invocation in pndm_invocations)
    assert any("FP_CLOCK" in str(invocation.schedule_dir) for invocation in pndm_invocations)
    assert str(pndm_invocations[0].output_dir).startswith("outputs/fp_clock_test_pndm/samples/")
    assert str(pndm_invocations[0].schedule_dir).startswith("outputs/fp_clock_test_pndm/schedules/")
    assert "--profile-cache-root" in pndm_invocations[0].prepare_steps[0].arguments

    diffusers_config = {
        "name": "fp_clock_test_diffusers",
        "backend": "diffusers",
        "models": ["sd35_medium"],
        "solvers": ["flow_euler"],
        "schedules": ["FP_CLOCK"],
        "nfes": [8],
    }
    execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=execution)
    assert len(diffusers_invocations) == 1
    assert any("export_defect_clock_schedule.py" in step.arguments[0] for step in diffusers_invocations[0].prepare_steps)
    assert any("FP_CLOCK" in str(invocation.schedule_dir) for invocation in diffusers_invocations)
    assert str(diffusers_invocations[0].output_dir).startswith("outputs/fp_clock_test_diffusers/samples/")
    assert str(diffusers_invocations[0].schedule_dir).startswith("outputs/fp_clock_test_diffusers/schedules/")


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


def test_build_invocations_expands_labeled_legacy_sadb_clock_variants() -> None:
    args = _build_args()
    config = {
        "name": "diffusers_legacy_sadb_variants",
        "backend": "diffusers",
        "models": ["stable_diffusion_15"],
        "solvers": ["dpm_solver_pp"],
        "schedules": ["LEGACY_SADB"],
        "nfes": [10],
        "seeds": [0],
        "schedule_clock_configs": {
            "LEGACY_SADB": {
                "small": "configs/clocks/LEGACY_SADB.yaml",
                "medium": "configs/clocks/LEGACY_SADB.yaml",
            }
        },
    }
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)
    assert len(invocations) == 2
    assert {invocation.label.split(":")[3] for invocation in invocations} == {"LEGACY_SADB[small]", "LEGACY_SADB[medium]"}
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
        "schedules": ["base", "LEGACY_SADB"],
        "nfes": [10],
        "seeds": [0],
        "guidance_scales": [3.0, 5.0],
        "schedule_clock_configs": {
            "LEGACY_SADB": "configs/clocks/LEGACY_SADB.yaml",
        },
    }
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)
    assert len(invocations) == 4
    sadb_invocations = [invocation for invocation in invocations if ":LEGACY_SADB:" in invocation.label]
    assert len(sadb_invocations) == 2
    assert any("cfg_3" in str(invocation.schedule_dir) for invocation in sadb_invocations)
    assert any("cfg_5" in str(invocation.schedule_dir) for invocation in sadb_invocations)
    assert all("cfg_" in str(invocation.output_dir) for invocation in invocations)


def test_build_invocations_allows_trajectory_clock_schedules_for_dpm_solver() -> None:
    args = _build_args()
    pndm_config = {
        "name": "dpm_custom_schedule",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["dpm_solver_lu"],
        "schedules": ["base", "FP_CLOCK[trajectory_window]"],
        "schedule_clock_configs": {
            "FP_CLOCK": {
                "variants": {
                    "trajectory_window": "configs/clocks/FP_CLOCK_trajectory_window_smoke.yaml",
                },
            },
        },
        "nfes": [6],
    }
    execution = resolve_execution_config(pndm_config, args)
    invocations = build_invocations(args, pndm_config, execution_config=execution)

    assert len(invocations) == 2
    fp_invocation = [invocation for invocation in invocations if "FP_CLOCK[trajectory_window]" in invocation.label][0]
    assert fp_invocation.materializable
    assert "dpm_solver_lu" in str(fp_invocation.schedule_dir)

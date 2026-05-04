import csv
import json
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
    ExperimentInvocation,
    PrepareStep,
    append_failure_row,
    build_invocations,
    canonical_schedule_name,
    count_unique_prepare_steps,
    infer_batch_size,
    infer_num_samples,
    is_materializable_schedule,
    load_experiment_config,
    print_invocations,
    resolve_execution_config,
    run_goes_oracle_reuse_report,
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
    assert canonical_schedule_name("GOES") == ("GOES", "GOES")
    assert canonical_schedule_name("LEGACY_SADB") == ("LEGACY_SADB", "LEGACY_SADB")
    assert canonical_schedule_name("FP_CLOCK") == ("FP_CLOCK", "FP_CLOCK")
    assert canonical_schedule_label("goes") == "GOES"
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
    assert is_materializable_schedule("pndm", "GOES")
    assert is_materializable_schedule("diffusers", "GOES")
    assert is_materializable_schedule("pndm", "LEGACY_SADB")
    assert is_materializable_schedule("diffusers", "LEGACY_SADB")
    assert is_materializable_schedule("pndm", "FP_CLOCK")
    assert is_materializable_schedule("diffusers", "FP_CLOCK")
    assert not is_materializable_schedule("pndm", "SADB")
    assert not is_materializable_schedule("pndm", "RI_SADB")
    assert not is_materializable_schedule("pndm", "LCS-1")
    assert not is_materializable_schedule("diffusers", "LCS-1")


def test_smoke_suffix_uses_dataset_smoke_sample_counts() -> None:
    experiment = {"name": "fp_multires_pndm_all_solvers_smoke"}
    dataset = {
        "smoke_num_samples": 100,
        "full_num_samples": 50000,
        "smoke_batch_size": 8,
        "default_batch_size": 64,
    }

    assert infer_num_samples(experiment, dataset) == 100
    assert infer_batch_size(experiment, dataset) == 8


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


def test_compact_metrics_filter_accepts_goes_rows(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "solvers:\n"
        "  - euler\n"
        "schedules:\n"
        "  - GOES\n",
        encoding="utf-8",
    )
    keep_row = build_row_filter(config_path)
    assert keep_row({"solver": "euler", "schedule": "GOES"})
    assert not keep_row({"solver": "euler", "schedule": "FP_CLOCK"})


def test_launcher_accepts_pndm_kid_and_rejects_unsupported_metrics() -> None:
    args = _build_args()
    pndm_kid_config = {
        "name": "pndm_kid_metric",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["euler"],
        "schedules": ["base"],
        "nfes": [4],
        "metrics": ["kid"],
    }
    execution = resolve_execution_config(pndm_kid_config, args)
    invocations = build_invocations(args, pndm_kid_config, execution_config=execution)
    assert len(invocations) == 1
    assert "--compute-kid" in invocations[0].run_arguments

    bad_pndm_config = {**pndm_kid_config, "metrics": ["clipscore"]}
    execution = resolve_execution_config(bad_pndm_config, args)
    try:
        build_invocations(args, bad_pndm_config, execution_config=execution)
    except ValueError as error:
        assert "Unsupported metrics for backend `pndm`" in str(error)
    else:
        raise AssertionError("Unsupported PNDM metrics should be rejected during expansion.")

    diffusers_config = {
        "name": "bad_diffusers_metric",
        "backend": "diffusers",
        "models": ["sd35_medium"],
        "solvers": ["flow_euler"],
        "schedules": ["base"],
        "nfes": [4],
        "metrics": ["fid"],
    }
    execution = resolve_execution_config(diffusers_config, args)
    try:
        build_invocations(args, diffusers_config, execution_config=execution)
    except ValueError as error:
        assert "Unsupported metrics for backend `diffusers`" in str(error)
    else:
        raise AssertionError("Unsupported diffusers metrics should be rejected during expansion.")


def test_append_failure_row_records_requested_pndm_kid_metric(tmp_path) -> None:
    summary_csv = tmp_path / "summary.csv"
    invocation = ExperimentInvocation(
        label="pndm:kid_failure",
        runtime_backend="pndm",
        run_arguments=(
            "scripts/run/run_pndm_experiment.py",
            "--summary-csv",
            str(summary_csv),
            "--dataset-config",
            "configs/datasets/cifar10.yaml",
            "--model-asset",
            "pndm_model_ddim_cifar10",
            "--solver",
            "euler",
            "--schedule-name",
            "base",
            "--nfe",
            "4",
            "--seed",
            "0",
            "--num-samples",
            "8",
            "--compute-fid",
            "--compute-kid",
            "--output-dir",
            str(tmp_path / "images"),
        ),
    )

    append_failure_row(invocation, RuntimeError("kid failed"))

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["metric_name"] == "fid,kid"
    assert "kid" in rows[0]
    assert rows[0]["kid"] == ""
    assert rows[0]["status"] == "FAILED"


def test_launcher_normalizes_supported_metric_aliases() -> None:
    args = _build_args()
    config = {
        "name": "diffusers_metric_aliases",
        "backend": "diffusers",
        "models": ["sd35_medium"],
        "solvers": ["flow_euler"],
        "schedules": ["base"],
        "nfes": [4],
        "metrics": ["clip_score", "image-reward"],
    }
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)

    assert len(invocations) == 1


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


def test_profile_cache_dir_partitions_anchored_replay_fp_clock() -> None:
    cache_root = Path("outputs/example_experiment/schedules/_profile_cache")
    cache_dir = profile_cache_dir(
        cache_root=cache_root,
        schedule_family="FP_CLOCK",
        backend="pndm",
        dataset_name="cifar10",
        model_asset="model_a",
        solver="dpm_solver_lu",
        calibration_solver="dpm_solver_lu",
        estimator="anchored_replay",
        physical_grid_size=17,
        physical_grid_mode="anchored_base",
        pilot_batch_size=8,
        pilot_num_batches=4,
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
        anchor_nfe=16,
        window_size=2,
    )
    meta = _build_profile_meta(
        schedule_family="FP_CLOCK",
        backend="pndm",
        model_asset="model_a",
        solver="dpm_solver_lu",
        calibration_solver="dpm_solver_lu",
        estimator="anchored_replay",
        physical_grid_size=17,
        physical_grid_mode="anchored_base",
        pilot_batch_size=8,
        pilot_num_batches=4,
        pilot_observation_microbatch=2,
        epsilon=1.0e-12,
        smoothing_window=1,
        q_min=1.05,
        q_max=6.0,
        model_output_type="epsilon",
        coordinate_domain="timesteps",
        target_nfe=10,
        target_steps=10,
        extra={
            "defect_estimator": "anchored_replay",
            "anchor_nfe": 16,
            "grid_mode": "anchored_base",
            "window_size": 2,
            "solver_order": 2,
        },
    )

    assert "anchored_replay" in str(cache_dir)
    assert "anchor_16" in str(cache_dir)
    assert "window_2" in str(cache_dir)
    assert meta["calibration_method"] == "anchored_quarter_replay_horizontal_defect"
    assert meta["defect_estimator"] == "anchored_replay"
    assert meta["grid_mode"] == "anchored_base"


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


def test_build_invocations_base_schedule_does_not_call_goes_verifier(monkeypatch) -> None:
    args = _build_args()
    config = {
        "name": "base_only_launcher",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["euler"],
        "schedules": ["base"],
        "nfes": [4],
    }

    def fail_import(*_args, **_kwargs):
        raise AssertionError("GOES verifier should not be needed for base-only expansion.")

    monkeypatch.setattr("goes.verify.verify_goes_schedule", fail_import)
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)

    assert len(invocations) == 1
    assert invocations[0].schedule_dir is None


def test_build_invocations_expands_goes_schedules_for_pndm_and_diffusers() -> None:
    args = _build_args()
    goes_options = {
        "batch_size": 2,
        "num_batches": 1,
        "ref_nfe": 8,
        "ref_grid_size": 9,
        "candidate_grid_size": 8,
        "metric": "identity",
        "rho": 0.1,
    }
    pndm_config = {
        "name": "goes_test_pndm",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["euler"],
        "schedules": ["GOES"],
        "nfes": [6],
        "seeds": [0, 1],
        "goes": {**goes_options, "coordinate_domain": "timesteps"},
    }
    execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=execution)
    assert len(pndm_invocations) == 2
    assert all(invocation.materializable for invocation in pndm_invocations)
    assert all("GOES" in str(invocation.schedule_dir) for invocation in pndm_invocations)
    assert any("seed_0" in str(invocation.schedule_dir) for invocation in pndm_invocations)
    step = pndm_invocations[0].prepare_steps[0]
    assert step.arguments[0] == "scripts/run/export_goes_pndm_schedule.py"
    assert "--oracle-cache-dir" in step.arguments
    assert "--coordinate-domain" in step.arguments
    assert "--ref-nfe" in step.arguments

    diffusers_config = {
        "name": "goes_test_diffusers",
        "backend": "diffusers",
        "models": ["sd35_medium"],
        "solvers": ["flow_euler"],
        "schedules": ["GOES"],
        "nfes": [8],
        "guidance_scales": [3.0],
        "goes": {**goes_options, "physical_grid_mode": "scheduler_sigmas"},
    }
    execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=execution)
    assert len(diffusers_invocations) == 1
    assert diffusers_invocations[0].materializable
    assert "GOES" in str(diffusers_invocations[0].schedule_dir)
    assert "cfg_3" in str(diffusers_invocations[0].schedule_dir)
    step = diffusers_invocations[0].prepare_steps[0]
    assert step.arguments[0] == "scripts/run/export_goes_diffusers_schedule.py"
    assert "--guidance-scale" in step.arguments
    assert "--physical-grid-mode" in step.arguments
    assert "--oracle-cache-dir" in step.arguments


def test_build_invocations_applies_labeled_goes_variant_configs(tmp_path) -> None:
    args = _build_args()
    rho0_config = tmp_path / "goes_rho0.yaml"
    rho1_config = tmp_path / "goes_rho1.yaml"
    rho0_config.write_text("rho: 0.0\nmetric: identity\n", encoding="utf-8")
    rho1_config.write_text("goes:\n  rho: 1.0\n  metric: channel_whitened\n", encoding="utf-8")
    config = {
        "name": "goes_variant_test",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["euler"],
        "schedules": ["GOES[rho0]", "GOES[rho1]"],
        "nfes": [5],
        "seeds": [0],
        "goes": {
            "batch_size": 2,
            "num_batches": 1,
            "ref_nfe": 8,
            "ref_grid_size": 9,
            "candidate_grid_size": 8,
            "metric": "identity",
            "rho": 0.1,
        },
        "schedule_clock_configs": {
            "GOES": {
                "variants": {
                    "rho0": str(rho0_config),
                    "rho1": str(rho1_config),
                },
            },
        },
    }

    def arg_value(arguments: tuple[str, ...], flag: str) -> str:
        return arguments[arguments.index(flag) + 1]

    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)
    by_schedule = {invocation.label.split(":")[4]: invocation for invocation in invocations}

    assert set(by_schedule) == {"GOES[rho0]", "GOES[rho1]"}
    assert count_unique_prepare_steps(invocations) == 2
    assert "/rho0/" in str(by_schedule["GOES[rho0]"].schedule_dir)
    assert "/rho1/" in str(by_schedule["GOES[rho1]"].schedule_dir)

    rho0_args = by_schedule["GOES[rho0]"].prepare_steps[0].arguments
    rho1_args = by_schedule["GOES[rho1]"].prepare_steps[0].arguments
    assert arg_value(rho0_args, "--rho") == "0.0"
    assert arg_value(rho0_args, "--metric") == "identity"
    assert arg_value(rho1_args, "--rho") == "1.0"
    assert arg_value(rho1_args, "--metric") == "channel_whitened"


def test_goes_schedule_dirs_are_unique_across_seed_guidance_and_nfe() -> None:
    args = _build_args()
    config = {
        "name": "goes_diffusers_partitioning",
        "backend": "diffusers",
        "models": ["sd35_medium"],
        "solvers": ["flow_euler"],
        "schedules": ["GOES"],
        "nfes": [5, 9],
        "seeds": [0, 1],
        "guidance_scales": [3.0, 5.0],
        "goes": {
            "batch_size": 2,
            "num_batches": 1,
            "ref_nfe": 8,
            "ref_grid_size": 9,
            "candidate_grid_size": 8,
            "physical_grid_mode": "scheduler_sigmas",
        },
    }
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)

    schedule_dirs = [str(invocation.schedule_dir) for invocation in invocations]

    assert len(invocations) == 8
    assert len(set(schedule_dirs)) == len(schedule_dirs)
    assert all("GOES/diffusers/sd35_medium/flow_euler" in item for item in schedule_dirs)
    assert any("cfg_3" in item for item in schedule_dirs)
    assert any("cfg_5" in item for item in schedule_dirs)
    assert any("seed_0" in item for item in schedule_dirs)
    assert any("seed_1" in item for item in schedule_dirs)
    assert any("nfe_005" in item for item in schedule_dirs)
    assert any("nfe_009" in item for item in schedule_dirs)


def test_pndm_goes_schedule_dirs_are_unique_across_model_solver_seed_and_nfe() -> None:
    args = _build_args()
    config = {
        "name": "goes_pndm_partitioning",
        "backend": "pndm",
        "dataset": "cifar10",
        "model_assets": ["pndm_model_ddim_cifar10", "pndm_model_pf_cifar10"],
        "solvers": ["euler", "heun2"],
        "schedules": ["GOES"],
        "nfes": [5, 9],
        "seeds": [0, 1],
        "goes": {
            "batch_size": 2,
            "num_batches": 1,
            "ref_nfe": 8,
            "ref_grid_size": 9,
            "candidate_grid_size": 8,
            "coordinate_domain": "timesteps",
        },
    }
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)

    schedule_dirs = [str(invocation.schedule_dir) for invocation in invocations]

    assert len(invocations) == 16
    assert len(set(schedule_dirs)) == len(schedule_dirs)
    assert any("pndm_model_ddim_cifar10" in item for item in schedule_dirs)
    assert any("pndm_model_pf_cifar10" in item for item in schedule_dirs)
    assert any("/euler/" in item for item in schedule_dirs)
    assert any("/heun2/" in item for item in schedule_dirs)
    assert any("seed_0" in item for item in schedule_dirs)
    assert any("seed_1" in item for item in schedule_dirs)
    assert any("nfe_005" in item for item in schedule_dirs)
    assert any("nfe_009" in item for item in schedule_dirs)


def test_goes_collection_configs_expand_materialization_plans() -> None:
    args = _build_args()

    pndm_config = load_experiment_config("configs/experiments/goes_pndm_nfe_sweep.yaml")
    pndm_execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=pndm_execution)
    pndm_goes = [item for item in pndm_invocations if ":GOES:" in item.label]
    pndm_step_args = pndm_goes[0].prepare_steps[0].arguments

    assert pndm_execution.materialize_schedules is False
    assert len(pndm_invocations) == 60
    assert len(pndm_goes) == 30
    assert count_unique_prepare_steps(pndm_invocations) == 30
    assert pndm_step_args[0] == "scripts/run/export_goes_pndm_schedule.py"
    assert "--ref-nfe" in pndm_step_args and "256" in pndm_step_args
    assert "--candidate-grid-size" in pndm_step_args and "512" in pndm_step_args
    assert "--metric" in pndm_step_args and "channel_whitened" in pndm_step_args
    assert any("seed_0" in str(item.schedule_dir) for item in pndm_goes)
    assert any("seed_2" in str(item.schedule_dir) for item in pndm_goes)
    assert any("nfe_050" in str(item.schedule_dir) for item in pndm_goes)

    diffusers_config = load_experiment_config("configs/experiments/goes_diffusers_cfg_nfe_sweep.yaml")
    diffusers_execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=diffusers_execution)
    diffusers_goes = [item for item in diffusers_invocations if ":GOES:" in item.label]
    diffusers_step_args = diffusers_goes[0].prepare_steps[0].arguments

    assert diffusers_execution.materialize_schedules is False
    assert len(diffusers_invocations) == 240
    assert len(diffusers_goes) == 120
    assert count_unique_prepare_steps(diffusers_invocations) == 120
    assert diffusers_step_args[0] == "scripts/run/export_goes_diffusers_schedule.py"
    assert "--guidance-scale" in diffusers_step_args and "3.0" in diffusers_step_args
    assert "--physical-grid-mode" in diffusers_step_args and "scheduler_sigmas" in diffusers_step_args
    assert "--metric" in diffusers_step_args and "channel_whitened" in diffusers_step_args
    assert any("cfg_3" in str(item.schedule_dir) for item in diffusers_goes)
    assert any("cfg_10" in str(item.schedule_dir) for item in diffusers_goes)
    assert any("seed_2" in str(item.schedule_dir) for item in diffusers_goes)
    assert any("nfe_020" in str(item.schedule_dir) for item in diffusers_goes)
    assert any("nfe_050" in str(item.schedule_dir) for item in diffusers_goes)


def test_goes_solver_comparison_configs_expand_theory_covered_solvers() -> None:
    args = _build_args()

    def arg_value(arguments: tuple[str, ...], flag: str) -> str:
        return arguments[arguments.index(flag) + 1]

    pndm_config = load_experiment_config("configs/experiments/goes_pndm_solver_comparison_odd_nfe.yaml")
    pndm_execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=pndm_execution)
    pndm_goes = [item for item in pndm_invocations if ":GOES:" in item.label]
    pndm_prepare_args = [step.arguments for invocation in pndm_goes for step in invocation.prepare_steps]
    pndm_oracle_cache_dirs = {arg_value(arguments, "--oracle-cache-dir") for arguments in pndm_prepare_args}

    assert pndm_execution.materialize_schedules is False
    assert len(pndm_invocations) == 36
    assert len(pndm_goes) == 18
    assert count_unique_prepare_steps(pndm_invocations) == 18
    assert len(pndm_oracle_cache_dirs) == 1
    assert next(iter(pndm_oracle_cache_dirs)).endswith("schedules/_goes_oracle_cache/pndm")
    assert any("--solver" in args and "euler" in args for args in pndm_prepare_args)
    assert any("--solver" in args and "heun2" in args for args in pndm_prepare_args)
    assert all(
        any(f"nfe_{nfe:03d}" in str(invocation.schedule_dir) for invocation in pndm_goes)
        for nfe in (5, 9, 15)
    )

    diffusers_config = load_experiment_config(
        "configs/experiments/goes_diffusers_flow_solver_comparison_odd_nfe.yaml"
    )
    diffusers_execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=diffusers_execution)
    diffusers_goes = [item for item in diffusers_invocations if ":GOES:" in item.label]
    diffusers_prepare_args = [step.arguments for invocation in diffusers_goes for step in invocation.prepare_steps]
    diffusers_oracle_cache_dirs = {arg_value(arguments, "--oracle-cache-dir") for arguments in diffusers_prepare_args}

    assert diffusers_execution.materialize_schedules is False
    assert len(diffusers_invocations) == 72
    assert len(diffusers_goes) == 36
    assert count_unique_prepare_steps(diffusers_invocations) == 36
    assert len(diffusers_oracle_cache_dirs) == 1
    assert next(iter(diffusers_oracle_cache_dirs)).endswith("schedules/_goes_oracle_cache/diffusers")
    assert any("--solver" in args and "flow_euler" in args for args in diffusers_prepare_args)
    assert any("--solver" in args and "flow_heun" in args for args in diffusers_prepare_args)
    assert any("cfg_3" in str(invocation.schedule_dir) for invocation in diffusers_goes)
    assert any("cfg_7.5" in str(invocation.schedule_dir) for invocation in diffusers_goes)


def test_goes_ablation_configs_expand_labeled_variants() -> None:
    args = _build_args()

    def arg_value(arguments: tuple[str, ...], flag: str) -> str:
        return arguments[arguments.index(flag) + 1]

    pndm_config = load_experiment_config("configs/experiments/goes_pndm_rho_metric_ablation.yaml")
    pndm_execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=pndm_execution)
    pndm_goes = [item for item in pndm_invocations if ":GOES[" in item.label]
    pndm_by_schedule = {item.label.split(":")[4]: item for item in pndm_goes if ":seed_000:" in item.label}

    assert pndm_execution.materialize_schedules is False
    assert len(pndm_invocations) == 30
    assert len(pndm_goes) == 27
    assert count_unique_prepare_steps(pndm_invocations) == 27
    assert "GOES[rho_0]" in pndm_by_schedule
    assert "GOES[metric_identity]" in pndm_by_schedule
    assert arg_value(pndm_by_schedule["GOES[rho_0]"].prepare_steps[0].arguments, "--rho") == "0.0"
    assert arg_value(pndm_by_schedule["GOES[rho_1]"].prepare_steps[0].arguments, "--rho") == "1.0"
    assert arg_value(pndm_by_schedule["GOES[metric_identity]"].prepare_steps[0].arguments, "--metric") == "identity"
    assert arg_value(pndm_by_schedule["GOES[metric_edm_scalar]"].prepare_steps[0].arguments, "--metric") == "edm_scalar"
    assert arg_value(pndm_by_schedule["GOES[metric_edm_scalar]"].prepare_steps[0].arguments, "--sigma-data") == "0.5"
    assert "/metric_channel_whitened/" in str(pndm_by_schedule["GOES[metric_channel_whitened]"].schedule_dir)

    diffusers_config = load_experiment_config("configs/experiments/goes_diffusers_rho_metric_ablation.yaml")
    diffusers_execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=diffusers_execution)
    diffusers_goes = [item for item in diffusers_invocations if ":GOES[" in item.label]
    diffusers_by_schedule = {item.label.split(":")[4]: item for item in diffusers_goes if ":seed_000:" in item.label}

    assert diffusers_execution.materialize_schedules is False
    assert len(diffusers_invocations) == 30
    assert len(diffusers_goes) == 27
    assert count_unique_prepare_steps(diffusers_invocations) == 27
    assert "cfg_7.5" in str(diffusers_by_schedule["GOES[rho_01]"].schedule_dir)
    assert arg_value(diffusers_by_schedule["GOES[rho_005]"].prepare_steps[0].arguments, "--rho") == "0.05"
    assert arg_value(diffusers_by_schedule["GOES[metric_identity]"].prepare_steps[0].arguments, "--metric") == "identity"
    assert arg_value(diffusers_by_schedule["GOES[metric_edm_scalar]"].prepare_steps[0].arguments, "--metric") == "edm_scalar"
    assert arg_value(diffusers_by_schedule["GOES[metric_edm_scalar]"].prepare_steps[0].arguments, "--sigma-data") == "0.5"
    assert (
        arg_value(diffusers_by_schedule["GOES[metric_channel_whitened]"].prepare_steps[0].arguments, "--metric")
        == "channel_whitened"
    )


def test_goes_calibration_size_configs_expand_labeled_variants() -> None:
    args = _build_args()

    def arg_value(arguments: tuple[str, ...], flag: str) -> str:
        return arguments[arguments.index(flag) + 1]

    pndm_config = load_experiment_config("configs/experiments/goes_pndm_calibration_size_ablation.yaml")
    pndm_execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=pndm_execution)
    pndm_goes = [item for item in pndm_invocations if ":GOES[" in item.label]
    pndm_by_schedule = {item.label.split(":")[4]: item for item in pndm_goes if ":seed_000:" in item.label}

    assert pndm_execution.materialize_schedules is False
    assert len(pndm_invocations) == 21
    assert len(pndm_goes) == 18
    assert count_unique_prepare_steps(pndm_invocations) == 18
    assert arg_value(pndm_by_schedule["GOES[calibration_k4]"].prepare_steps[0].arguments, "--batch-size") == "4"
    assert arg_value(pndm_by_schedule["GOES[calibration_k4]"].prepare_steps[0].arguments, "--num-batches") == "1"
    assert arg_value(pndm_by_schedule["GOES[calibration_k64]"].prepare_steps[0].arguments, "--batch-size") == "8"
    assert arg_value(pndm_by_schedule["GOES[calibration_k64]"].prepare_steps[0].arguments, "--num-batches") == "8"
    assert arg_value(pndm_by_schedule["GOES[calibration_k128]"].prepare_steps[0].arguments, "--batch-size") == "8"
    assert arg_value(pndm_by_schedule["GOES[calibration_k128]"].prepare_steps[0].arguments, "--num-batches") == "16"
    assert "/calibration_k32/" in str(pndm_by_schedule["GOES[calibration_k32]"].schedule_dir)

    diffusers_config = load_experiment_config("configs/experiments/goes_diffusers_calibration_size_ablation.yaml")
    diffusers_execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=diffusers_execution)
    diffusers_goes = [item for item in diffusers_invocations if ":GOES[" in item.label]
    diffusers_by_schedule = {item.label.split(":")[4]: item for item in diffusers_goes if ":seed_000:" in item.label}

    assert diffusers_execution.materialize_schedules is False
    assert len(diffusers_invocations) == 21
    assert len(diffusers_goes) == 18
    assert count_unique_prepare_steps(diffusers_invocations) == 18
    assert "cfg_7.5" in str(diffusers_by_schedule["GOES[calibration_k16]"].schedule_dir)
    assert arg_value(diffusers_by_schedule["GOES[calibration_k8]"].prepare_steps[0].arguments, "--batch-size") == "4"
    assert arg_value(diffusers_by_schedule["GOES[calibration_k8]"].prepare_steps[0].arguments, "--num-batches") == "2"
    assert arg_value(diffusers_by_schedule["GOES[calibration_k64]"].prepare_steps[0].arguments, "--batch-size") == "8"
    assert arg_value(diffusers_by_schedule["GOES[calibration_k64]"].prepare_steps[0].arguments, "--num-batches") == "8"
    assert arg_value(diffusers_by_schedule["GOES[calibration_k128]"].prepare_steps[0].arguments, "--batch-size") == "8"
    assert arg_value(diffusers_by_schedule["GOES[calibration_k128]"].prepare_steps[0].arguments, "--num-batches") == "16"


def test_goes_candidate_grid_configs_expand_labeled_variants() -> None:
    args = _build_args()

    def arg_value(arguments: tuple[str, ...], flag: str) -> str:
        return arguments[arguments.index(flag) + 1]

    pndm_config = load_experiment_config("configs/experiments/goes_pndm_candidate_grid_ablation.yaml")
    pndm_execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=pndm_execution)
    pndm_goes = [item for item in pndm_invocations if ":GOES[" in item.label]
    pndm_by_schedule = {item.label.split(":")[4]: item for item in pndm_goes if ":seed_000:" in item.label}

    assert pndm_execution.materialize_schedules is False
    assert len(pndm_invocations) == 18
    assert len(pndm_goes) == 15
    assert count_unique_prepare_steps(pndm_invocations) == 15
    assert arg_value(pndm_by_schedule["GOES[candidate_m64]"].prepare_steps[0].arguments, "--candidate-grid-size") == "64"
    assert arg_value(pndm_by_schedule["GOES[candidate_m1024]"].prepare_steps[0].arguments, "--candidate-grid-size") == "1024"
    assert "/candidate_m512/" in str(pndm_by_schedule["GOES[candidate_m512]"].schedule_dir)

    diffusers_config = load_experiment_config("configs/experiments/goes_diffusers_candidate_grid_ablation.yaml")
    diffusers_execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=diffusers_execution)
    diffusers_goes = [item for item in diffusers_invocations if ":GOES[" in item.label]
    diffusers_by_schedule = {item.label.split(":")[4]: item for item in diffusers_goes if ":seed_000:" in item.label}

    assert diffusers_execution.materialize_schedules is False
    assert len(diffusers_invocations) == 18
    assert len(diffusers_goes) == 15
    assert count_unique_prepare_steps(diffusers_invocations) == 15
    assert "cfg_7.5" in str(diffusers_by_schedule["GOES[candidate_m256]"].schedule_dir)
    assert (
        arg_value(diffusers_by_schedule["GOES[candidate_m128]"].prepare_steps[0].arguments, "--candidate-grid-size")
        == "128"
    )
    assert (
        arg_value(diffusers_by_schedule["GOES[candidate_m1024]"].prepare_steps[0].arguments, "--candidate-grid-size")
        == "1024"
    )


def test_goes_oracle_convergence_configs_expand_ref_nfe_variants() -> None:
    args = _build_args()

    def arg_value(arguments: tuple[str, ...], flag: str) -> str:
        return arguments[arguments.index(flag) + 1]

    pndm_config = load_experiment_config("configs/experiments/goes_pndm_oracle_convergence.yaml")
    pndm_execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=pndm_execution)
    pndm_goes = [item for item in pndm_invocations if ":GOES[" in item.label]
    pndm_by_schedule = {item.label.split(":")[4]: item for item in pndm_goes if ":seed_000:" in item.label}

    assert pndm_execution.materialize_schedules is False
    assert len(pndm_invocations) == 15
    assert len(pndm_goes) == 12
    assert count_unique_prepare_steps(pndm_invocations) == 12
    assert arg_value(pndm_by_schedule["GOES[ref_nfe_100]"].prepare_steps[0].arguments, "--ref-nfe") == "100"
    assert arg_value(pndm_by_schedule["GOES[ref_nfe_100]"].prepare_steps[0].arguments, "--ref-grid-size") == "101"
    assert arg_value(pndm_by_schedule["GOES[ref_nfe_1000]"].prepare_steps[0].arguments, "--ref-nfe") == "1000"
    assert arg_value(pndm_by_schedule["GOES[ref_nfe_1000]"].prepare_steps[0].arguments, "--ref-grid-size") == "1001"
    assert "/ref_nfe_500/" in str(pndm_by_schedule["GOES[ref_nfe_500]"].schedule_dir)

    diffusers_config = load_experiment_config("configs/experiments/goes_diffusers_oracle_convergence.yaml")
    diffusers_execution = resolve_execution_config(diffusers_config, args)
    diffusers_invocations = build_invocations(args, diffusers_config, execution_config=diffusers_execution)
    diffusers_goes = [item for item in diffusers_invocations if ":GOES[" in item.label]
    diffusers_by_schedule = {item.label.split(":")[4]: item for item in diffusers_goes if ":seed_000:" in item.label}

    assert diffusers_execution.materialize_schedules is False
    assert len(diffusers_invocations) == 15
    assert len(diffusers_goes) == 12
    assert count_unique_prepare_steps(diffusers_invocations) == 12
    assert "cfg_7.5" in str(diffusers_by_schedule["GOES[ref_nfe_200]"].schedule_dir)
    assert arg_value(diffusers_by_schedule["GOES[ref_nfe_200]"].prepare_steps[0].arguments, "--ref-nfe") == "200"
    assert arg_value(diffusers_by_schedule["GOES[ref_nfe_200]"].prepare_steps[0].arguments, "--ref-grid-size") == "201"
    assert arg_value(diffusers_by_schedule["GOES[ref_nfe_1000]"].prepare_steps[0].arguments, "--ref-nfe") == "1000"
    assert arg_value(diffusers_by_schedule["GOES[ref_nfe_1000]"].prepare_steps[0].arguments, "--ref-grid-size") == "1001"


def test_goes_diffusers_models_without_published_schedule_config_expands_base_and_goes_only() -> None:
    args = _build_args()

    def arg_value(arguments: tuple[str, ...], flag: str) -> str:
        return arguments[arguments.index(flag) + 1]

    config = load_experiment_config("configs/experiments/goes_diffusers_models_without_published_schedules.yaml")
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)
    goes_invocations = [item for item in invocations if ":GOES:" in item.label]
    base_invocations = [item for item in invocations if ":base:" in item.label]
    model_labels = {item.label.split(":")[1] for item in invocations}
    prepare_assets = {
        arg_value(step.arguments, "--model-asset")
        for invocation in goes_invocations
        for step in invocation.prepare_steps
    }

    assert execution.materialize_schedules is False
    assert len(invocations) == 18
    assert len(base_invocations) == 9
    assert len(goes_invocations) == 9
    assert count_unique_prepare_steps(invocations) == 9
    assert model_labels == {"sd35_medium", "flux_dev", "lumina_image_2"}
    assert prepare_assets == {"hf_sd35_medium", "hf_flux_dev", "hf_lumina_image_2"}
    assert all(":ays:" not in item.label.lower() for item in invocations)
    assert all("ays_like" not in str(item.schedule_dir or "") for item in invocations)
    assert all(item.schedule_dir is None and not item.prepare_steps for item in base_invocations)
    assert all("cfg_3.5" in str(item.schedule_dir) for item in goes_invocations)
    assert all("nfe_010" in str(item.schedule_dir) for item in goes_invocations)
    assert all(item.prepare_steps[0].arguments[0] == "scripts/run/export_goes_diffusers_schedule.py" for item in goes_invocations)
    assert arg_value(goes_invocations[0].prepare_steps[0].arguments, "--metric") == "channel_whitened"
    assert arg_value(goes_invocations[0].prepare_steps[0].arguments, "--batch-size") == "2"
    assert arg_value(goes_invocations[0].prepare_steps[0].arguments, "--num-batches") == "4"


def test_launcher_writes_goes_oracle_reuse_report_after_materialization(tmp_path) -> None:
    args = _build_args()
    args.outputs_root = str(tmp_path)
    experiment_config = {
        "name": "goes_report",
        "backend": "diffusers",
        "schedules": ["GOES"],
        "execution": {
            "schedule_cache_root": str(tmp_path / "goes_report" / "schedules"),
            "continue_on_error": False,
        },
    }
    execution = resolve_execution_config(experiment_config, args)
    schedule_dir = execution.schedule_cache_root / "GOES" / "diffusers" / "model" / "flow_euler" / "nfe_004"
    schedule_dir.mkdir(parents=True)
    schedule_payload = {
        "method": "GOES",
        "solver": "flow_euler",
        "target_nfe": 4,
        "oracle_cache_key": "shared-key",
        "schedule_hash": "hash",
        "edge_objective": 0.1,
        "model_asset": "hf_sd35_medium",
        "prompt_asset": "diffusers_smoke_prompts",
        "guidance_scale": 3.5,
        "seed": 0,
        "calibration_cost_breakdown": {
            "num_samples": 2,
            "cfg_multiplier": 2,
            "oracle_cost_per_sample": 41,
            "edge_cost_per_sample": 10,
            "total_model_eval_equivalents": 2 * 2 * (41 + 10),
        },
    }
    (schedule_dir / "schedule.json").write_text(json.dumps(schedule_payload), encoding="utf-8")
    (schedule_dir / "run_metadata.json").write_text(
        json.dumps({"oracle_loaded_from_cache": False, "oracle_build_or_load_seconds": 0.1}),
        encoding="utf-8",
    )

    run_goes_oracle_reuse_report(args, experiment_config, execution_config=execution)

    report_csv = tmp_path / "goes_report" / "paper_tables" / "oracle_reuse_cost.csv"
    with report_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["oracle_cache_key"] == "shared-key"
    assert rows[0]["shared_total_model_eval_equivalents"] == str(2 * 2 * (41 + 10))


def test_preview_prints_shared_prepare_step_once(capsys) -> None:
    shared_step = PrepareStep(
        key="GOES:pndm:shared:nfe_005",
        runtime_backend="pndm",
        output_path=Path("outputs/schedules/GOES/nfe_005"),
        arguments=(
            "scripts/run/export_goes_pndm_schedule.py",
            "--nfe",
            "5",
            "--output-dir",
            "outputs/schedules/GOES/nfe_005",
        ),
    )
    invocations = [
        ExperimentInvocation(
            label="run:0",
            runtime_backend="pndm",
            run_arguments=("scripts/run/run_pndm_experiment.py", "--nfe", "5"),
            prepare_steps=(shared_step,),
        ),
        ExperimentInvocation(
            label="run:1",
            runtime_backend="pndm",
            run_arguments=("scripts/run/run_pndm_experiment.py", "--nfe", "5"),
            prepare_steps=(shared_step,),
        ),
    ]

    print_invocations(invocations, "configs/runtime_envs.yaml")
    output = capsys.readouterr().out

    assert output.count("scripts/run/export_goes_pndm_schedule.py") == 1
    assert "prepare: reuse key=GOES:pndm:shared:nfe_005" in output


def test_build_invocations_rejects_unsupported_goes_solvers_before_prepare() -> None:
    args = _build_args()
    pndm_config = {
        "name": "goes_pndm_replay_solver",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["dpm_solver_lu"],
        "schedules": ["GOES"],
        "nfes": [6],
    }
    execution = resolve_execution_config(pndm_config, args)
    pndm_invocations = build_invocations(args, pndm_config, execution_config=execution)
    assert len(pndm_invocations) == 1
    assert pndm_invocations[0].prepare_steps
    assert "--solver" in pndm_invocations[0].prepare_steps[0].arguments
    assert "dpm_solver_lu" in pndm_invocations[0].prepare_steps[0].arguments

    unsupported_pndm_config = {
        "name": "goes_bad_pndm_solver",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["pndm"],
        "schedules": ["GOES"],
        "nfes": [6],
    }
    execution = resolve_execution_config(unsupported_pndm_config, args)
    try:
        build_invocations(args, unsupported_pndm_config, execution_config=execution)
    except ValueError as error:
        assert "PRK/PLMS" in str(error)
    else:
        raise AssertionError("PNDM/PLMS GOES should be rejected until its nonuniform runner is validated.")

    flow_config = {
        "name": "goes_flow_replay_solver",
        "backend": "diffusers",
        "models": ["sd35_medium"],
        "solvers": ["flow_unipc"],
        "schedules": ["GOES"],
        "nfes": [8],
    }
    execution = resolve_execution_config(flow_config, args)
    flow_invocations = build_invocations(args, flow_config, execution_config=execution)
    assert len(flow_invocations) == 1
    assert flow_invocations[0].prepare_steps
    assert "--solver" in flow_invocations[0].prepare_steps[0].arguments
    assert "flow_unipc" in flow_invocations[0].prepare_steps[0].arguments

    vp_config = {
        "name": "goes_vp_replay_solver",
        "backend": "diffusers",
        "models": ["stable_diffusion_15"],
        "solvers": ["dpm_solver_pp"],
        "schedules": ["GOES"],
        "nfes": [10],
    }
    execution = resolve_execution_config(vp_config, args)
    vp_invocations = build_invocations(args, vp_config, execution_config=execution)
    assert len(vp_invocations) == 1
    assert vp_invocations[0].prepare_steps
    assert "--solver" in vp_invocations[0].prepare_steps[0].arguments
    assert "dpm_solver_pp" in vp_invocations[0].prepare_steps[0].arguments


def test_build_invocations_allows_diffusers_goes_history_solver_adapters() -> None:
    args = _build_args()
    vp_config = {
        "name": "goes_vp_history_solvers",
        "backend": "diffusers",
        "models": ["stable_diffusion_15"],
        "solvers": ["dpm_solver_pp", "sde_dpm_solver_pp", "unipc", "stork4_1st", "edm_dpm_solver_pp"],
        "schedules": ["GOES"],
        "nfes": [10],
        "gpde": {"defect_backend": "anchored_replay", "anchor_nfe": 16, "window_size": 4},
    }
    execution = resolve_execution_config(vp_config, args)
    vp_invocations = build_invocations(args, vp_config, execution_config=execution)
    assert len(vp_invocations) == 5
    for invocation in vp_invocations:
        arguments = invocation.prepare_steps[0].arguments
        assert "--defect-backend" in arguments
        assert "anchored_replay" in arguments
        assert "--anchor-nfe" in arguments
        assert "16" in arguments
        assert "--window-size" in arguments
        assert "4" in arguments

    flow_config = {
        "name": "goes_flow_history_solvers",
        "backend": "diffusers",
        "models": ["sd35_medium"],
        "solvers": ["flow_dpm_solver", "flow_unipc", "flow_stork4_1st"],
        "schedules": ["GOES"],
        "nfes": [8],
    }
    execution = resolve_execution_config(flow_config, args)
    assert len(build_invocations(args, flow_config, execution_config=execution)) == 3


def test_build_invocations_allows_pndm_goes_history_solver_adapters() -> None:
    args = _build_args()
    config = {
        "name": "goes_pndm_history_solvers",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": [
            "ddim",
            "deis",
            "dpm_solver_lu",
            "dpm_solver_default",
            "dpm_solver_pp",
            "unipc",
            "stork4_1st",
            "stork4_2nd",
        ],
        "schedules": ["GOES"],
        "nfes": [6],
        "gpde": {"defect_backend": "anchored_replay", "anchor_nfe": 16, "window_size": 4},
    }
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)
    assert len(invocations) == 8
    for invocation in invocations:
        arguments = invocation.prepare_steps[0].arguments
        assert "--defect-backend" in arguments
        assert "anchored_replay" in arguments
        assert "--anchor-nfe" in arguments
        assert "16" in arguments
        assert "--window-size" in arguments
        assert "4" in arguments


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


def test_build_invocations_allows_anchored_replay_clock_schedules_for_dpm_solver() -> None:
    args = _build_args()
    pndm_config = {
        "name": "dpm_custom_schedule",
        "backend": "pndm",
        "dataset": "cifar10",
        "solvers": ["dpm_solver_lu"],
        "schedules": ["base", "FP_CLOCK[anchored_replay]"],
        "schedule_clock_configs": {
            "FP_CLOCK": {
                "variants": {
                    "anchored_replay": "configs/clocks/FP_CLOCK_anchored_replay_smoke.yaml",
                },
            },
        },
        "nfes": [6],
    }
    execution = resolve_execution_config(pndm_config, args)
    invocations = build_invocations(args, pndm_config, execution_config=execution)

    assert len(invocations) == 2
    fp_invocation = [invocation for invocation in invocations if "FP_CLOCK[anchored_replay]" in invocation.label][0]
    assert fp_invocation.materializable
    assert "dpm_solver_lu" in str(fp_invocation.schedule_dir)

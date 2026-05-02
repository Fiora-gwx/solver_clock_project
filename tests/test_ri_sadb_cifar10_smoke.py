from types import SimpleNamespace

from scripts.run.run_experiment_config import build_invocations, load_experiment_config, resolve_execution_config


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


def test_ri_sadb_cifar10_smoke_expands_without_execution() -> None:
    args = _build_args()
    args.experiment_config = "configs/experiments/ri_sadb_cifar10_smoke.yaml"
    config = load_experiment_config(args.experiment_config)
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)

    schedules = {invocation.label.split(":")[4] for invocation in invocations}
    assert "base" in schedules
    assert "SADB" in schedules
    assert any(schedule.startswith("RI_SADB[eta_") for schedule in schedules)
    ri_invocations = [item for item in invocations if ":RI_SADB[" in item.label]
    assert ri_invocations
    assert all(item.prepare_steps for item in ri_invocations)
    assert all("RI_SADB" in str(item.schedule_dir) for item in ri_invocations)

from types import SimpleNamespace
import csv
import subprocess
import sys

from scripts.run.run_experiment_config import build_invocations, load_experiment_config, resolve_execution_config


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


def test_fp_clock_cifar10_smoke_expands_without_execution() -> None:
    args = _build_args()
    args.experiment_config = "configs/experiments/fp_clock_cifar10_smoke.yaml"
    config = load_experiment_config(args.experiment_config)
    execution = resolve_execution_config(config, args)
    invocations = build_invocations(args, config, execution_config=execution)

    schedules = {invocation.label.split(":")[4] for invocation in invocations}
    assert "base" in schedules
    assert "LEGACY_SADB" in schedules
    assert "FP_CLOCK" in schedules
    fp_invocations = [item for item in invocations if ":FP_CLOCK:" in item.label]
    assert fp_invocations
    assert all(item.prepare_steps for item in fp_invocations)
    assert all("FP_CLOCK" in str(item.schedule_dir) for item in fp_invocations)
    assert all(str(item.output_dir).startswith("outputs/fp_clock_cifar10_smoke/samples/") for item in invocations)
    assert all(
        "--profile-cache-root" in step.arguments
        for item in fp_invocations
        for step in item.prepare_steps
    )


def test_fp_clock_smoke_suite_writes_required_csvs_and_report(tmp_path) -> None:
    output_dir = tmp_path / "fp_clock_suite"
    command = [
        sys.executable,
        "scripts/run_fp_clock_smoke_suite.py",
        "--dataset",
        "cifar10",
        "--output-dir",
        str(output_dir),
        "--nfe",
        "10",
        "--seeds",
        "0",
        "--num-samples",
        "1",
        "--include-base",
        "--include-legacy-sadb",
        "--include-fp-clock",
        "--device",
        "cpu",
        "--solvers",
        "euler",
        "--schedule-only",
    ]
    subprocess.run(command, check=True)

    required = [
        output_dir / "results_raw.csv",
        output_dir / "results_aggregate.csv",
        output_dir / "schedule_diagnostics.csv",
        output_dir / "fp_clock_smoke_report.md",
        output_dir / "run.log",
    ]
    for path in required:
        assert path.exists()

    with (output_dir / "results_raw.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    schedules = {row["schedule"] for row in rows}
    assert {"base", "LEGACY_SADB", "FP_CLOCK"} <= schedules

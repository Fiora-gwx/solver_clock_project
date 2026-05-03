from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run.run_experiment_config import ExperimentInvocation, PrepareStep, collect_prepare_steps
from src.clock.baseline import BASELINE_SCHEDULE_IMPLEMENTATION_VERSION
from src.clock.defect_balanced import DEFECT_BALANCED_CLOCK_VERSION
from src.clock.fp_clock import FP_CLOCK_VERSION
from src.utils.config import dump_json


def _build_fp_clock_invocation(schedule_dir: Path, prepare_root: Path) -> ExperimentInvocation:
    step = PrepareStep(
        key="FP_CLOCK:pndm:test_solver",
        runtime_backend="pndm",
        arguments=("scripts/run/export_defect_clock_schedule.py",),
        output_path=prepare_root,
    )
    return ExperimentInvocation(
        label="test",
        runtime_backend="pndm",
        run_arguments=tuple(),
        prepare_steps=(step,),
        output_dir=None,
        schedule_dir=schedule_dir,
        materializable=True,
        notes=tuple(),
    )


def test_collect_prepare_steps_skips_current_baseline_bundle() -> None:
    with TemporaryDirectory() as temp_dir:
        schedule_dir = Path(temp_dir) / "schedule" / "nfe_006"
        schedule_dir.mkdir(parents=True)
        dump_json(
            {
                "schedule_family": "base",
                "schedule_implementation_version": BASELINE_SCHEDULE_IMPLEMENTATION_VERSION,
            },
            schedule_dir / "meta.json",
        )

        step = PrepareStep(
            key="base:pndm:test_solver",
            runtime_backend="pndm",
            arguments=("scripts/run/export_baseline_schedule.py",),
            output_path=schedule_dir.parent,
        )
        invocation = ExperimentInvocation(
            label="test_base",
            runtime_backend="pndm",
            run_arguments=tuple(),
            prepare_steps=(step,),
            output_dir=None,
            schedule_dir=schedule_dir,
            materializable=True,
            notes=tuple(),
        )
        assert collect_prepare_steps([invocation]) == []


def test_collect_prepare_steps_skips_current_legacy_sadb_bundle() -> None:
    with TemporaryDirectory() as temp_dir:
        schedule_dir = Path(temp_dir) / "schedule" / "nfe_006"
        schedule_dir.mkdir(parents=True)
        dump_json(
            {
                "schedule_family": "LEGACY_SADB",
                "schedule_implementation_version": DEFECT_BALANCED_CLOCK_VERSION,
            },
            schedule_dir / "meta.json",
        )

        step = PrepareStep(
            key="LEGACY_SADB:pndm:test_solver",
            runtime_backend="pndm",
            arguments=("scripts/run/export_defect_clock_schedule.py",),
            output_path=schedule_dir.parent,
        )
        invocation = ExperimentInvocation(
            label="test_legacy_sadb",
            runtime_backend="pndm",
            run_arguments=tuple(),
            prepare_steps=(step,),
            output_dir=None,
            schedule_dir=schedule_dir,
            materializable=True,
            notes=tuple(),
        )
        assert collect_prepare_steps([invocation]) == []


def test_collect_prepare_steps_skips_current_fp_clock_bundle() -> None:
    with TemporaryDirectory() as temp_dir:
        schedule_dir = Path(temp_dir) / "schedule" / "nfe_006"
        schedule_dir.mkdir(parents=True)
        dump_json(
            {
                "schedule_family": "FP_CLOCK",
                "schedule_implementation_version": FP_CLOCK_VERSION,
            },
            schedule_dir / "meta.json",
        )

        invocation = _build_fp_clock_invocation(schedule_dir, schedule_dir.parent)
        assert collect_prepare_steps([invocation]) == []

def test_collect_prepare_steps_rebuilds_stale_baseline_bundle() -> None:
    with TemporaryDirectory() as temp_dir:
        schedule_dir = Path(temp_dir) / "schedule" / "nfe_006"
        schedule_dir.mkdir(parents=True)
        dump_json(
            {
                "schedule_family": "linear",
                "schedule_implementation_version": BASELINE_SCHEDULE_IMPLEMENTATION_VERSION - 1,
            },
            schedule_dir / "meta.json",
        )

        step = PrepareStep(
            key="linear:pndm:test_solver",
            runtime_backend="pndm",
            arguments=("scripts/run/export_baseline_schedule.py",),
            output_path=schedule_dir.parent,
        )
        invocation = ExperimentInvocation(
            label="test_linear_stale",
            runtime_backend="pndm",
            run_arguments=tuple(),
            prepare_steps=(step,),
            output_dir=None,
            schedule_dir=schedule_dir,
            materializable=True,
            notes=tuple(),
        )
        steps = collect_prepare_steps([invocation])
        assert len(steps) == 1
        assert steps[0].key == "linear:pndm:test_solver"

"""Geometry-aware Predictive Defect Equalization (GPDE)."""

from .aggregation import robust_aggregate
from .coordinate import CoordinateAdapter, make_coordinate_adapter
from .edge_evaluator import EdgeCostTable, evaluate_edge_table
from .gpde import (
    GPDEProfile,
    GPDESchedule,
    default_q_for_solver,
    evaluate_gpde_profile,
    make_probe_grid,
    make_probe_steps,
    materialize_gpde_schedule,
)
from .metrics import ChannelWhitenedMetric, EDMScalarMetric, IdentityMetric, make_metric
from .mixed_defect import MixedDefectResult, mixed_normal_defect_sq
from .oracle import OracleData
from .oracle_cache import build_or_load_oracle
from .schedules import GOES_SCHEDULE_IMPLEMENTATION_VERSION
from .torch_backend import (
    TorchOracleCacheResult,
    TorchStepSolver,
    build_or_load_torch_velocity_oracle,
    build_torch_velocity_oracle,
    make_torch_step_solver,
)
from .verify import verify_goes_schedule, verify_schedule_bundle, verify_schedule_json, verify_schedule_payload

__all__ = [
    "ChannelWhitenedMetric",
    "CoordinateAdapter",
    "EDMScalarMetric",
    "EdgeCostTable",
    "GOES_SCHEDULE_IMPLEMENTATION_VERSION",
    "GPDEProfile",
    "GPDESchedule",
    "IdentityMetric",
    "MixedDefectResult",
    "OracleData",
    "TorchOracleCacheResult",
    "TorchStepSolver",
    "build_or_load_oracle",
    "build_or_load_torch_velocity_oracle",
    "build_torch_velocity_oracle",
    "evaluate_edge_table",
    "evaluate_gpde_profile",
    "default_q_for_solver",
    "make_probe_grid",
    "make_probe_steps",
    "materialize_gpde_schedule",
    "make_torch_step_solver",
    "make_coordinate_adapter",
    "make_metric",
    "mixed_normal_defect_sq",
    "robust_aggregate",
    "verify_goes_schedule",
    "verify_schedule_bundle",
    "verify_schedule_json",
    "verify_schedule_payload",
]

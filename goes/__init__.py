"""Geometry-aware Oracle Edge Scheduling (GOES)."""

from .aggregation import robust_aggregate
from .coordinate import CoordinateAdapter, make_coordinate_adapter
from .dp_minimax import MinimaxPath, solve_minimax_schedule
from .edge_evaluator import EdgeCostTable, evaluate_edge_table
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
    evaluate_torch_velocity_edge_table,
    make_torch_step_solver,
)
from .verify import verify_goes_schedule, verify_schedule_bundle, verify_schedule_json, verify_schedule_payload

__all__ = [
    "ChannelWhitenedMetric",
    "CoordinateAdapter",
    "EDMScalarMetric",
    "EdgeCostTable",
    "GOES_SCHEDULE_IMPLEMENTATION_VERSION",
    "IdentityMetric",
    "MinimaxPath",
    "MixedDefectResult",
    "OracleData",
    "TorchOracleCacheResult",
    "TorchStepSolver",
    "build_or_load_oracle",
    "build_or_load_torch_velocity_oracle",
    "build_torch_velocity_oracle",
    "evaluate_torch_velocity_edge_table",
    "evaluate_edge_table",
    "make_torch_step_solver",
    "make_coordinate_adapter",
    "make_metric",
    "mixed_normal_defect_sq",
    "robust_aggregate",
    "solve_minimax_schedule",
    "verify_goes_schedule",
    "verify_schedule_bundle",
    "verify_schedule_json",
    "verify_schedule_payload",
]

import json
import math
import subprocess
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from goes.aggregation import robust_aggregate
from goes import experiment_runner as goes_runner
from goes.config import load_config
from goes.coordinate import CoordinateAdapter
from goes.dp_minimax import brute_force_minimax, solve_minimax_schedule
from goes.edge_evaluator import evaluate_edge_table, evaluate_replay_metrics
from goes.metrics import IdentityMetric, make_metric
from goes.mixed_defect import mixed_normal_defect_sq
from goes.oracle import OracleData
from goes.oracle_cache import build_or_load_oracle, make_oracle_key
from goes.replay_refinement import refine_schedule_blackbox
from goes.repository_schedules import export_schedule_bundle
from goes.schedules import GOES_SCHEDULE_IMPLEMENTATION_VERSION
from goes.torch_backend import build_or_load_torch_velocity_oracle, evaluate_torch_velocity_edge_table
from goes.toy import make_solver, make_toy_model
from goes.verify import verify_goes_schedule, verify_schedule_bundle, verify_schedule_payload
from src.utils.config import load_yaml
from src.utils.schedule_bundle import ScheduleBundle


def _load_script_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mixed_defect_rho_one_equals_full_residual() -> None:
    metric = IdentityMetric()
    residual = np.asarray([[3.0, 4.0], [2.0, 0.0]], dtype=np.float64)
    tangent = np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)

    result = mixed_normal_defect_sq(residual, tangent, metric, 0.5, rho=1.0)

    assert np.allclose(result.values, metric.norm_sq(residual, 0.5))


def test_mixed_defect_rho_zero_equals_identity_normal_residual() -> None:
    metric = IdentityMetric()
    residual = np.asarray([[3.0, 4.0]], dtype=np.float64)
    tangent = np.asarray([[1.0, 0.0]], dtype=np.float64)

    result = mixed_normal_defect_sq(residual, tangent, metric, 0.5, rho=0.0)

    assert np.allclose(result.values, [16.0])


def test_mixed_defect_tangent_sign_flip_does_not_change_value() -> None:
    metric = IdentityMetric()
    residual = np.asarray([[1.5, -0.5]], dtype=np.float64)
    tangent = np.asarray([[2.0, 1.0]], dtype=np.float64)

    pos = mixed_normal_defect_sq(residual, tangent, metric, 0.5, rho=0.2)
    neg = mixed_normal_defect_sq(residual, -tangent, metric, 0.5, rho=0.2)

    assert np.allclose(pos.values, neg.values)


def test_mixed_defect_is_nonnegative_and_tiny_tangent_falls_back() -> None:
    metric = IdentityMetric()
    residual = np.asarray([[1.0, 2.0]], dtype=np.float64)
    tangent = np.asarray([[0.0, 0.0]], dtype=np.float64)

    result = mixed_normal_defect_sq(residual, tangent, metric, 0.5, rho=0.0, eps=1.0e-10)

    assert np.all(result.values >= 0.0)
    assert np.allclose(result.values, [5.0])
    assert result.fallback_mask.tolist() == [True]
    assert result.fallback_fraction == 1.0


def test_mixed_defect_rho_positive_detects_pure_tangential_error() -> None:
    metric = IdentityMetric()
    residual = np.asarray([[2.0, 0.0]], dtype=np.float64)
    tangent = np.asarray([[1.0, 0.0]], dtype=np.float64)

    pure_normal = mixed_normal_defect_sq(residual, tangent, metric, 0.5, rho=0.0)
    mixed = mixed_normal_defect_sq(residual, tangent, metric, 0.5, rho=0.1)

    assert np.allclose(pure_normal.values, [0.0])
    assert np.allclose(mixed.values, [0.4])


def test_metric_choices_apply_expected_weights() -> None:
    edm = make_metric({"name": "edm_scalar", "sigma_data": 0.5}, coordinate=SimpleNamespace(name="log_sigma"))
    values = np.ones((2, 2), dtype=np.float64)
    weight = 1.0 / (2.0**2 + 0.5**2)
    assert np.allclose(edm.norm_sq(values, np.log(2.0)), [2.0 * weight, 2.0 * weight])

    negative_sigma = make_metric({"name": "edm_scalar", "sigma_data": 0.5}, coordinate=SimpleNamespace(name="negative_sigma"))
    assert np.allclose(negative_sigma.norm_sq(values, -2.0), [2.0 * weight, 2.0 * weight])

    interpolated_sigma = make_metric(
        {
            "name": "edm_scalar",
            "sigma_data": 0.5,
            "u_grid": np.asarray([-4.0, 0.0], dtype=np.float64),
            "sigma_grid": np.asarray([4.0, 0.0], dtype=np.float64),
        }
    )
    assert np.allclose(interpolated_sigma.norm_sq(values, -2.0), [2.0 * weight, 2.0 * weight])

    states = np.zeros((2, 2, 2, 1, 1), dtype=np.float64)
    states[:, 0, :, 0, 0] = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    states[:, 1, :, 0, 0] = np.asarray([[0.0, 10.0], [4.0, 14.0]], dtype=np.float64)
    oracle = SimpleNamespace(states=states, u_grid=np.asarray([0.0, 1.0], dtype=np.float64))
    whitened = make_metric({"name": "channel_whitened", "eps": 0.0}, oracle=oracle)
    assert np.allclose(whitened.apply(np.ones((1, 2, 1, 1)), 0.0), np.ones((1, 2, 1, 1)))
    assert np.allclose(whitened.apply(np.ones((1, 2, 1, 1)), 1.0), 0.25 * np.ones((1, 2, 1, 1)))


def test_robust_aggregation_modes() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 100.0], dtype=np.float64)

    assert robust_aggregate(values, {"name": "mean"}) == 26.5
    assert robust_aggregate(values, {"name": "median"}) == 2.5
    assert robust_aggregate(values, {"name": "trimmed_mean", "trim_ratio": 0.25}) == 2.5
    assert robust_aggregate(values, {"name": "cvar", "alpha": 0.5}) == 51.5


def test_dp_minimax_matches_bruteforce_for_small_matrix() -> None:
    D = np.full((6, 6), np.inf, dtype=np.float64)
    for j in range(6):
        for l in range(j + 1, 6):
            D[j, l] = float((l - j) ** 2 + (j % 2))

    path = solve_minimax_schedule(D, 3)
    brute = brute_force_minimax(D, 3)

    assert path.indices[0] == 0
    assert path.indices[-1] == 5
    assert len(path.indices) == 4
    assert all(b > a for a, b in zip(path.indices[:-1], path.indices[1:]))
    assert math.isclose(path.objective, brute.objective)
    assert math.isclose(path.total_cost, brute.total_cost)


def test_dp_tie_break_keeps_primary_minimax_optimum() -> None:
    D = np.full((4, 4), np.inf, dtype=np.float64)
    D[0, 1] = 1.0
    D[1, 3] = 5.0
    D[0, 2] = 4.0
    D[2, 3] = 5.0
    D[0, 3] = 100.0
    D[1, 2] = 100.0

    path = solve_minimax_schedule(D, 2, tie_break_sum_cost=True)

    assert path.objective == 5.0
    assert path.indices == [0, 1, 3]
    assert path.total_cost == 6.0


def _goes_test_config(tmp_path: Path, **overrides):
    base = {
        "oracle": {
            "cache_dir": str(tmp_path / "oracle_cache"),
            "ref_nfe": 24,
            "ref_grid_size": 25,
            "reuse": True,
        },
        "calibration": {"num_samples": 3, "seed": 7, "split": "calibration", "guidance_scale": 1.0},
        "heldout": {"num_samples": 3, "seed": 13, "split": "heldout"},
        "candidate_grid": {"size": 8, "type": "uniform_in_u"},
        "solver": {"name": "euler", "target_nfe": 4, "mode": "one_step"},
        "output": {"root": str(tmp_path / "outputs"), "save_plots": False},
    }
    for section, value in overrides.items():
        base.setdefault(section, {}).update(value)
    return load_config(None, base)


def test_oracle_cache_key_loads_reused_data_without_rewriting(tmp_path) -> None:
    config = _goes_test_config(tmp_path)

    first = build_or_load_oracle(config)
    mtime = first.cache_path.stat().st_mtime_ns
    second = build_or_load_oracle(config)

    assert first.cache_key == second.cache_key
    assert first.oracle.metadata["oracle_cache_key"] == first.cache_key
    assert second.oracle.metadata["oracle_cache_key"] == second.cache_key
    assert second.loaded_from_cache
    assert second.cache_path.stat().st_mtime_ns == mtime
    assert np.allclose(first.oracle.states, second.oracle.states)


def test_oracle_cache_key_changes_for_model_coordinate_ref_nfe_and_seed(tmp_path) -> None:
    base = _goes_test_config(tmp_path)
    base_result = build_or_load_oracle(base)
    alternate_interpolation_key = make_oracle_key({**base_result.oracle.metadata, "interpolation": "nearest"})

    model_result = build_or_load_oracle(_goes_test_config(tmp_path, model={"name": "toy_flow_alt"}))
    coord_result = build_or_load_oracle(
        _goes_test_config(tmp_path, coordinate={"name": "sigma", "u_min": 0.1, "u_max": 1.0})
    )
    ref_result = build_or_load_oracle(_goes_test_config(tmp_path, oracle={"ref_nfe": 32}))
    seed_result = build_or_load_oracle(_goes_test_config(tmp_path, calibration={"seed": 8}))

    keys = {base_result.cache_key, model_result.cache_key, coord_result.cache_key, ref_result.cache_key, seed_result.cache_key}
    assert len(keys) == 5
    assert base_result.oracle.metadata["interpolation"] == "linear"
    assert alternate_interpolation_key != base_result.cache_key


def test_goes_config_rejects_unsupported_oracle_interpolation(tmp_path) -> None:
    with pytest.raises(ValueError, match="oracle.interpolation"):
        _goes_test_config(tmp_path, oracle={"interpolation": "cubic"})


def test_goes_config_rejects_disabling_required_outputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="output.save_schedule"):
        _goes_test_config(tmp_path, output={"save_schedule": False})
    with pytest.raises(ValueError, match="output.save_edge_table"):
        _goes_test_config(tmp_path, output={"save_edge_table": False})
    with pytest.raises(ValueError, match="output.save_images"):
        _goes_test_config(tmp_path, output={"save_images": True})


def test_goes_config_rejects_unsupported_cpu_runner_options(tmp_path) -> None:
    invalid_cases = [
        ({"model": {"state_shape": [2, 0]}}, "model.state_shape"),
        ({"oracle": {"ref_integrator": "euler"}}, "oracle.ref_integrator"),
        ({"oracle": {"ref_nfe": 0}}, "oracle.ref_nfe"),
        ({"oracle": {"ref_grid_size": 1}}, "oracle.ref_grid_size"),
        ({"calibration": {"num_samples": 0}}, "calibration.num_samples"),
        ({"heldout": {"num_samples": 0}}, "heldout.num_samples"),
        ({"calibration": {"guidance_scale": float("nan")}}, "calibration.guidance_scale"),
        ({"coordinate": {"name": "bad"}}, "coordinate.name"),
        ({"coordinate": {"direction": "sideways"}}, "coordinate.direction"),
        ({"candidate_grid": {"type": "log"}}, "candidate_grid.type"),
        ({"solver": {"name": "unsupported"}}, "solver.name"),
        ({"solver": {"mode": "unknown"}}, "solver.mode"),
        ({"metric": {"name": "unsupported"}}, "metric.name"),
        ({"metric": {"eps": 0.0}}, "metric.eps"),
        ({"metric": {"sigma_data": 0.0}}, "metric.sigma_data"),
        ({"metric": {"min_weight": 10.0, "max_weight": 1.0}}, "metric min_weight/max_weight"),
        ({"aggregation": {"name": "unsupported"}}, "aggregation.name"),
        ({"aggregation": {"name": "cvar", "alpha": 1.0}}, "aggregation.alpha"),
        ({"optimizer": {"name": "equalize"}}, "optimizer.name"),
        ({"optimizer": {"tie_tolerance": -1.0}}, "optimizer.tie_tolerance"),
        ({"mixed_defect": {"eps": 0.0}}, "mixed_defect.eps"),
        ({"replay_refinement": {"rounds": -1}}, "replay_refinement.rounds"),
        ({"replay_refinement": {"local_window": 0}}, "replay_refinement.local_window"),
        ({"replay_refinement": {"lambda_final": -0.1}}, "replay_refinement.lambda_final"),
        ({"replay_refinement": {"mu_smooth": -0.1}}, "replay_refinement.mu_smooth"),
    ]
    for overrides, expected in invalid_cases:
        with pytest.raises(ValueError, match=expected):
            _goes_test_config(tmp_path, **overrides)


def test_oracle_reused_by_two_dummy_solvers_without_rebuilding(tmp_path) -> None:
    config = _goes_test_config(tmp_path)
    oracle_result = build_or_load_oracle(config)
    model = make_toy_model(config["model"])
    metric = make_metric(config["metric"], oracle=oracle_result.oracle)
    grid = np.linspace(0.0, 1.0, 5)

    euler_table = evaluate_edge_table(make_solver("euler", model), oracle_result.oracle, grid, metric)
    heun_table = evaluate_edge_table(make_solver("heun", model), oracle_result.oracle, grid, metric)
    second = build_or_load_oracle(config)

    assert second.loaded_from_cache
    assert euler_table.edge_costs.shape == heun_table.edge_costs.shape == (5, 5)
    assert np.isfinite(euler_table.edge_costs[0, -1])
    assert np.isfinite(heun_table.edge_costs[0, -1])


def test_oracle_interpolation_shapes_and_coordinate_roundtrip(tmp_path) -> None:
    config = _goes_test_config(tmp_path)
    oracle = build_or_load_oracle(config).oracle

    state = oracle.state_at(0.5)
    tangent = oracle.tangent_at(np.asarray([0.25, 0.75]))
    coord = CoordinateAdapter(name="log_sigma", u_min=-2.0, u_max=0.0)
    native = np.asarray([0.2, 0.7], dtype=np.float64)

    assert state.shape == (3, 2)
    assert tangent.shape == (3, 2, 2)
    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(tangent))
    assert np.allclose(coord.u_to_native(coord.native_to_u(native)), native)


def _straight_oracle() -> OracleData:
    u_grid = np.asarray([0.0, 1.0], dtype=np.float64)
    states = np.asarray([[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float64)
    tangents = np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float64)
    return OracleData(
        states=states,
        tangents=tangents,
        u_grid=u_grid,
        conditions=np.asarray([0.0]),
        noise_seeds=np.asarray([0]),
        metadata={},
    )


def test_edge_evaluator_detects_tangential_bias_when_rho_positive() -> None:
    class TangentialBiasSolver:
        def single_edge_step_from_state(self, x_a, a, b, condition):
            del x_a, a, condition
            return np.asarray([[b + 0.2, 0.0]], dtype=np.float64)

    oracle = _straight_oracle()
    metric = IdentityMetric()
    grid = np.asarray([0.0, 1.0], dtype=np.float64)

    pure_normal = evaluate_edge_table(TangentialBiasSolver(), oracle, grid, metric, rho=0.0)
    mixed = evaluate_edge_table(TangentialBiasSolver(), oracle, grid, metric, rho=0.1)
    full = evaluate_edge_table(TangentialBiasSolver(), oracle, grid, metric, rho=1.0)

    assert np.allclose(pure_normal.edge_costs[0, 1], 0.0)
    assert mixed.edge_costs[0, 1] > 0.0
    assert np.allclose(full.edge_costs[0, 1], 0.04)


def test_edge_evaluator_detects_circle_normal_bias() -> None:
    u_grid = np.asarray([0.0, np.pi / 2.0], dtype=np.float64)
    states = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    tangents = np.asarray([[[0.0, 1.0], [-1.0, 0.0]]], dtype=np.float64)
    oracle = OracleData(states, tangents, u_grid, np.asarray([0.0]), np.asarray([0]), {})

    class NormalBiasSolver:
        def single_edge_step_from_state(self, x_a, a, b, condition):
            del x_a, a, condition
            return np.asarray([[0.0, 1.25]], dtype=np.float64)

    table = evaluate_edge_table(NormalBiasSolver(), oracle, u_grid, IdentityMetric(), rho=0.0)

    assert table.edge_costs[0, 1] > 0.0


def test_edge_evaluator_uses_oracle_edge_start_not_replayed_prefix_state() -> None:
    oracle = _straight_oracle()

    class RecordingSolver:
        def __init__(self):
            self.starts = []

        def single_edge_step_from_state(self, x_a, a, b, condition):
            del b, condition
            self.starts.append((float(a), x_a.copy()))
            return x_a

    solver = RecordingSolver()
    evaluate_edge_table(solver, oracle, np.asarray([0.0, 1.0]), IdentityMetric(), rho=1.0)

    assert len(solver.starts) == 1
    assert solver.starts[0][0] == 0.0
    assert np.allclose(solver.starts[0][1], oracle.state_at(0.0))


def test_replay_metrics_respects_tiny_tangent_fallback_setting() -> None:
    u_grid = np.asarray([0.0, 1.0], dtype=np.float64)
    states = np.zeros((1, 2, 1), dtype=np.float64)
    tangents = np.ones((1, 2, 1), dtype=np.float64)
    oracle = OracleData(states, tangents, u_grid, np.asarray([0.0]), np.asarray([0]), {})

    class UnitResidualSolver:
        name = "unit_residual"

        def single_edge_step_from_state(self, x_a, a, b, condition):
            del a, b, condition
            return np.asarray(x_a, dtype=np.float64) + 1.0

    with_fallback = evaluate_replay_metrics(
        UnitResidualSolver(),
        oracle,
        u_grid,
        IdentityMetric(),
        rho=0.0,
        eps=2.0,
        fallback_full_residual_on_tiny_tangent=True,
    )
    without_fallback = evaluate_replay_metrics(
        UnitResidualSolver(),
        oracle,
        u_grid,
        IdentityMetric(),
        rho=0.0,
        eps=2.0,
        fallback_full_residual_on_tiny_tangent=False,
    )

    assert with_fallback.fallback_fraction == 1.0
    assert without_fallback.fallback_fraction == 1.0
    assert np.allclose(with_fallback.endpoint_costs, [1.0])
    assert np.allclose(without_fallback.endpoint_costs, [0.5])


def test_failure_cases_reports_no_underperformance_when_goes_matches_baseline(tmp_path) -> None:
    u_grid = np.asarray([0.0, 1.0], dtype=np.float64)
    states = np.asarray(
        [
            [[0.0], [1.0]],
            [[1.0], [2.0]],
        ],
        dtype=np.float64,
    )
    tangents = np.ones_like(states)
    oracle = OracleData(states, tangents, u_grid, np.asarray([0.0, 1.0]), np.asarray([11, 12]), {})

    class IdentitySolver:
        name = "identity"

        def single_edge_step_from_state(self, x_a, a, b, condition):
            del a, b, condition
            return np.asarray(x_a, dtype=np.float64)

    config = _goes_test_config(tmp_path, solver={"target_nfe": 1})
    goes_runner._write_failure_cases(
        run_dir=tmp_path,
        config=config,
        solver=IdentitySolver(),
        heldout_oracle=oracle,
        goes_schedule=u_grid,
        uniform_schedule=u_grid,
        edge_objective=0.0,
        tiny_tangent_fallback_fraction=0.0,
    )
    failure_cases = (tmp_path / "failure_cases.csv").read_text(encoding="utf-8")

    assert "No held-out sample underperformed" in failure_cases
    assert "mse_delta_goes_minus_baseline,0.0" not in failure_cases


def test_goes_smoke_cli_produces_required_outputs(tmp_path) -> None:
    config = _goes_test_config(tmp_path)
    config_path = tmp_path / "goes_smoke.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "goes.experiment_runner", "search-schedule", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    run_dir = Path(payload["run_dir"])

    assert (run_dir / "schedule.json").exists()
    assert (run_dir / "schedule_native.json").exists()
    assert (run_dir / "edge_costs.npz").exists()
    assert (run_dir / "selected_edges.csv").exists()
    assert (run_dir / "failure_cases.csv").exists()
    assert (run_dir / "calibration_metrics.csv").exists()
    assert (run_dir / "heldout_metrics.csv").exists()
    assert (run_dir / "oracle_metadata.json").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "paper_tables" / "main_results.csv").exists()
    main_results = (run_dir / "paper_tables" / "main_results.csv").read_text(encoding="utf-8")
    assert "final_latent_mse_bootstrap_se" in main_results
    assert "final_latent_mse_ci95_low" in main_results
    assert "final_latent_mse_ci95_high" in main_results

    schedule = json.loads((run_dir / "schedule.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert schedule["method"] == "GOES"
    assert schedule["schedule_implementation_version"] == GOES_SCHEDULE_IMPLEMENTATION_VERSION
    assert len(schedule["u_schedule"]) == config["solver"]["target_nfe"] + 1
    assert all(b > a for a, b in zip(schedule["u_schedule"][:-1], schedule["u_schedule"][1:]))
    assert metadata["runtime"]["python"]
    assert "numpy" in metadata["runtime"]
    assert Path(metadata["config_resolved_path"]).exists()
    assert metadata["deterministic_seeds"]["python_random_seed"] == config["calibration"]["seed"]
    assert metadata["deterministic_seeds"]["numpy_seed"] == config["calibration"]["seed"]
    assert metadata["deterministic_seeds"]["calibration_seed"] == config["calibration"]["seed"]
    assert metadata["deterministic_seeds"]["heldout_seed"] == config["heldout"]["seed"]
    assert metadata["deterministic_seeds"]["common_random_numbers"] is True
    assert metadata["plots_written"] == []
    assert metadata["model"]["name"] == "toy_flow"
    assert metadata["model"]["identifier"].startswith("toy_flow:")
    assert metadata["model"]["checkpoint_path"] is None
    assert metadata["calibration_split_hash"] != metadata["heldout_split_hash"]
    assert metadata["calibration_initial_noise_hash"] != metadata["heldout_initial_noise_hash"]
    assert metadata["calibration_noise_seed_hash"] != metadata["heldout_noise_seed_hash"]
    assert len(metadata["calibration_noise_seeds"]) == config["calibration"]["num_samples"]
    assert len(metadata["heldout_noise_seeds"]) == config["heldout"]["num_samples"]
    assert metadata["calibration_noise_seeds"] != metadata["heldout_noise_seeds"]
    assert metadata["baselines"]["run"] == ["uniform_in_u"]
    skipped = {item["name"]: item["reason"] for item in metadata["baselines"]["skipped"]}
    assert "AYS" in skipped and skipped["AYS"]
    assert "image_metrics" in skipped and skipped["image_metrics"]
    failure_cases = (run_dir / "failure_cases.csv").read_text(encoding="utf-8")
    assert "tiny_tangent_fallback_fraction" in failure_cases
    assert "goes_replay_endpoint_mse" in failure_cases
    assert "baseline_replay_endpoint_mse" in failure_cases


def test_goes_search_writes_core_plots_when_enabled(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    config = _goes_test_config(
        tmp_path,
        oracle={"ref_nfe": 12, "ref_grid_size": 13},
        calibration={"num_samples": 2},
        heldout={"num_samples": 2},
        candidate_grid={"size": 6},
        solver={"target_nfe": 3},
        output={"save_plots": True},
    )

    result = goes_runner.search_schedule_command(config)
    run_dir = Path(result["run_dir"])
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    expected = {
        "plots/schedule.png",
        "plots/edge_cost_heatmap.png",
        "plots/selected_edge_costs.png",
    }
    assert expected.issubset({str(Path(path).relative_to(run_dir)) for path in run_dir.glob("plots/*.png")})
    assert expected.issubset({str(Path(path).relative_to(run_dir)) for path in metadata["plots_written"]})


def test_goes_empirical_solver_records_theory_coverage_warning(tmp_path) -> None:
    config = _goes_test_config(
        tmp_path,
        oracle={"ref_nfe": 10, "ref_grid_size": 11},
        calibration={"num_samples": 1},
        heldout={"num_samples": 1},
        candidate_grid={"size": 4},
        solver={"name": "empirical_noisy_euler", "target_nfe": 2},
    )
    config_path = tmp_path / "goes_empirical_solver.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "goes.experiment_runner", "search-schedule", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(json.loads(proc.stdout)["run_dir"])
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert metadata["solver"]["name"] == "empirical_noisy_euler"
    assert metadata["solver"]["deterministic_oracle_theory"] is False
    assert "empirical-only" in metadata["solver"]["coverage_note"]


def test_goes_build_oracle_search_and_evaluate_cli(tmp_path) -> None:
    config = _goes_test_config(
        tmp_path,
        oracle={"ref_nfe": 12, "ref_grid_size": 13},
        calibration={"num_samples": 2},
        heldout={"num_samples": 2},
        candidate_grid={"size": 5},
        solver={"target_nfe": 3},
    )
    config_path = tmp_path / "goes_cli.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]

    build = subprocess.run(
        [sys.executable, "-m", "goes.experiment_runner", "build-oracle", "--config", str(config_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    build_payload = json.loads(build.stdout)
    build_dir = Path(build_payload["run_dir"])
    build_metadata = json.loads((build_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert build_metadata["command"] == "build-oracle"
    assert build_metadata["run_dir"] == str(build_dir)
    assert Path(build_metadata["config_resolved_path"]).exists()
    assert Path(build_metadata["oracle_metadata_path"]).exists()
    assert Path(build_metadata["oracle_cache_metadata_path"]).exists()
    assert build_metadata["model"]["name"] == "toy_flow"
    assert build_metadata["model"]["identifier"].startswith("toy_flow:")
    assert build_metadata["model"]["checkpoint_path"] is None
    assert build_payload["oracle_cache_key"] == build_metadata["oracle_cache_key"]
    assert build_metadata["calibration_split_hash"]
    assert build_metadata["calibration_initial_noise_hash"]
    assert build_metadata["calibration_noise_seed_hash"]
    assert len(build_metadata["calibration_noise_seeds"]) == config["calibration"]["num_samples"]
    assert build_metadata["deterministic_seeds"]["python_random_seed"] == config["calibration"]["seed"]
    assert build_metadata["total_seconds"] >= build_metadata["oracle_build_or_load_seconds"]
    assert (build_dir / "oracle_metadata.json").exists()

    search = subprocess.run(
        [sys.executable, "-m", "goes.experiment_runner", "search-schedule", "--config", str(config_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    search_payload = json.loads(search.stdout)
    schedule_path = Path(search_payload["run_dir"]) / "schedule.json"
    assert schedule_path.exists()

    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "goes.experiment_runner",
            "evaluate",
            "--config",
            str(config_path),
            "--schedule",
            str(schedule_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    evaluate_payload = json.loads(evaluate.stdout)
    evaluate_dir = Path(evaluate_payload["run_dir"])
    evaluate_metadata = json.loads((evaluate_dir / "run_metadata.json").read_text(encoding="utf-8"))
    evaluate_oracle_metadata = json.loads((evaluate_dir / "oracle_metadata.json").read_text(encoding="utf-8"))
    assert evaluate_metadata["command"] == "evaluate"
    assert evaluate_metadata["run_dir"] == str(evaluate_dir)
    assert evaluate_metadata["schedule_path"] == str(schedule_path)
    assert Path(evaluate_metadata["config_resolved_path"]).exists()
    assert Path(evaluate_metadata["oracle_metadata_path"]).exists()
    assert evaluate_metadata["model"]["name"] == "toy_flow"
    assert evaluate_metadata["model"]["identifier"].startswith("toy_flow:")
    assert evaluate_metadata["model"]["checkpoint_path"] is None
    assert evaluate_metadata["deterministic_seeds"]["python_random_seed"] == config["calibration"]["seed"]
    assert evaluate_metadata["calibration_oracle_cache_key"]
    assert evaluate_metadata["heldout_oracle_cache_key"]
    assert evaluate_metadata["calibration_split_hash"] != evaluate_metadata["heldout_split_hash"]
    assert evaluate_metadata["calibration_initial_noise_hash"] != evaluate_metadata["heldout_initial_noise_hash"]
    assert evaluate_metadata["calibration_noise_seed_hash"] != evaluate_metadata["heldout_noise_seed_hash"]
    assert len(evaluate_metadata["calibration_noise_seeds"]) == config["calibration"]["num_samples"]
    assert len(evaluate_metadata["heldout_noise_seeds"]) == config["heldout"]["num_samples"]
    assert evaluate_oracle_metadata["calibration"]["condition_split_hash"] == evaluate_metadata["calibration_split_hash"]
    assert evaluate_oracle_metadata["heldout"]["condition_split_hash"] == evaluate_metadata["heldout_split_hash"]
    assert evaluate_metadata["total_seconds"] >= 0.0
    assert (evaluate_dir / "heldout_metrics.csv").exists()
    assert (evaluate_dir / "paper_tables" / "main_results.csv").exists()
    assert evaluate_payload["heldout_metrics"]["split"] == "heldout"


def test_goes_evaluate_rejects_schedule_config_mismatch(tmp_path) -> None:
    config = _goes_test_config(
        tmp_path,
        oracle={"ref_nfe": 8, "ref_grid_size": 9},
        calibration={"num_samples": 1},
        heldout={"num_samples": 1},
        candidate_grid={"size": 4},
        solver={"name": "euler", "target_nfe": 3},
    )
    config_path = tmp_path / "goes_mismatch_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    schedule_path = tmp_path / "mismatched_schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                "method": "GOES",
                "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
                "solver": "heun",
                "target_nfe": 2,
                "coordinate": "t",
                "coordinate_direction": "increasing",
                "u_schedule": [0.0, 0.5, 1.0],
                "native_schedule": [0.0, 0.5, 1.0],
                "oracle_cache_key": "abc",
                "rho": 0.1,
                "metric": {"name": "identity"},
                "aggregation": "trimmed_mean_10pct",
                "edge_objective": 0.2,
                "selected_edge_costs": [0.1, 0.2],
                "selected_indices": [0, 1, 2],
                "schedule_hash": "hash",
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "goes.experiment_runner",
            "evaluate",
            "--config",
            str(config_path),
            "--schedule",
            str(schedule_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "does not match config solver.target_nfe" in proc.stderr

    solver_mismatch_path = tmp_path / "solver_mismatched_schedule.json"
    solver_mismatch_path.write_text(
        json.dumps(
            {
                "method": "GOES",
                "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
                "solver": "heun",
                "target_nfe": 3,
                "coordinate": "t",
                "coordinate_direction": "increasing",
                "u_schedule": [0.0, 0.25, 0.75, 1.0],
                "native_schedule": [0.0, 0.25, 0.75, 1.0],
                "oracle_cache_key": "abc",
                "rho": 0.1,
                "metric": {"name": "identity"},
                "aggregation": "trimmed_mean_10pct",
                "edge_objective": 0.3,
                "selected_edge_costs": [0.1, 0.2, 0.3],
                "selected_indices": [0, 1, 2, 3],
                "schedule_hash": "hash_solver",
            }
        ),
        encoding="utf-8",
    )

    solver_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "goes.experiment_runner",
            "evaluate",
            "--config",
            str(config_path),
            "--schedule",
            str(solver_mismatch_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert solver_proc.returncode != 0
    assert "does not match config solver" in solver_proc.stderr


def test_goes_cpu_sweeps_write_required_paper_table_fields(tmp_path) -> None:
    config = _goes_test_config(
        tmp_path,
        oracle={"ref_nfe": 16, "ref_grid_size": 17},
        calibration={"num_samples": 2},
        heldout={"num_samples": 2},
        candidate_grid={"size": 6},
        solver={"target_nfe": 3},
    )
    config_path = tmp_path / "goes_sweeps.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]

    convergence = subprocess.run(
        [
            sys.executable,
            "-m",
            "goes.experiment_runner",
            "oracle-convergence",
            "--config",
            str(config_path),
            "--values",
            "8,12",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    convergence_dir = Path(json.loads(convergence.stdout)["run_dir"])
    convergence_csv = (convergence_dir / "paper_tables" / "oracle_convergence.csv").read_text(encoding="utf-8")
    convergence_metadata = json.loads((convergence_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert "final_latent_oracle_mse" in convergence_csv
    assert "oracle_build_or_load_seconds" in convergence_csv
    assert "edge_cost_correlation_to_highest_ref" in convergence_csv
    assert "edge_cost_rank_correlation_to_highest_ref" in convergence_csv
    assert convergence_metadata["command"] == "oracle-convergence"
    assert Path(convergence_metadata["config_resolved_path"]).exists()
    assert convergence_metadata["deterministic_seeds"]["python_random_seed"] == config["calibration"]["seed"]
    assert convergence_metadata["model"]["name"] == "toy_flow"

    reuse = subprocess.run(
        [
            sys.executable,
            "-m",
            "goes.experiment_runner",
            "cross-solver-reuse",
            "--config",
            str(config_path),
            "--solvers",
            "euler,heun,missing_solver",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    reuse_dir = Path(json.loads(reuse.stdout)["run_dir"])
    reuse_csv = (reuse_dir / "paper_tables" / "oracle_reuse_cost.csv").read_text(encoding="utf-8")
    reuse_metadata = json.loads((reuse_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert "shared_oracle_builds" in reuse_csv
    assert "separate_oracle_builds" in reuse_csv
    assert "edge_evaluation_seconds" in reuse_csv
    assert "oracle_reused" in reuse_csv
    assert "runnable_solver_count" in reuse_csv
    assert "estimated_shared_oracle_build_or_load_seconds" in reuse_csv
    assert "estimated_separate_oracle_build_or_load_seconds" in reuse_csv
    assert "shared_oracle_amortized_build_or_load_seconds" in reuse_csv
    assert "estimated_shared_total_solver_seconds" in reuse_csv
    assert "estimated_separate_total_solver_seconds" in reuse_csv
    assert "skip_reason" in reuse_csv
    assert "Unsupported toy solver: missing_solver" in reuse_csv
    assert reuse_metadata["command"] == "cross-solver-reuse"
    assert Path(reuse_metadata["config_resolved_path"]).exists()
    assert reuse_metadata["deterministic_seeds"]["python_random_seed"] == config["calibration"]["seed"]
    assert reuse_metadata["model"]["name"] == "toy_flow"
    assert reuse_metadata["skipped_solvers"] == [
        {"solver": "missing_solver", "reason": "Unsupported toy solver: missing_solver"}
    ]


def test_goes_cpu_ablation_clis_write_raw_paper_tables(tmp_path) -> None:
    config = _goes_test_config(
        tmp_path,
        oracle={"ref_nfe": 8, "ref_grid_size": 9},
        calibration={"num_samples": 1},
        heldout={"num_samples": 1},
        candidate_grid={"size": 4},
        solver={"target_nfe": 2},
    )
    config_path = tmp_path / "goes_ablation_clis.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]

    commands = [
        (
            ["nfe-sweep", "--values", "2,3"],
            "nfe_quality_curve.csv",
            [
                "schedule",
                "nfe",
                "final_latent_mse",
                "final_latent_mse_bootstrap_se",
                "edge_evaluation_seconds",
                "search_dp_seconds",
                "run_dir",
            ],
        ),
        (
            ["ablate-rho", "--values", "0.0,0.1"],
            "ablations.csv",
            ["ablation", "rho", "heldout_generalization_gap", "total_seconds", "schedule_hash"],
        ),
        (
            ["ablate-metric", "--values", "identity,edm_scalar"],
            "ablations.csv",
            ["ablation", "metric", "heldout_generalization_gap", "total_seconds", "schedule_hash"],
        ),
        (
            ["calibration-size-ablation", "--values", "1,2"],
            "calibration_size_ablation.csv",
            [
                "ablation",
                "calibration_samples",
                "heldout_generalization_gap",
                "schedule_l1_to_largest_calibration_size",
                "total_seconds",
                "schedule_hash",
            ],
        ),
        (
            ["candidate-grid-ablation", "--values", "3,4"],
            "candidate_grid_ablation.csv",
            [
                "ablation",
                "candidate_grid_size",
                "edge_evaluation_seconds",
                "search_dp_seconds",
                "schedule_l1_to_largest_candidate_grid",
                "total_seconds",
                "schedule_hash",
            ],
        ),
    ]
    for command, table_name, expected_columns in commands:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "goes.experiment_runner",
                command[0],
                "--config",
                str(config_path),
                *command[1:],
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        run_dir = Path(json.loads(proc.stdout)["run_dir"])
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        table = (run_dir / "paper_tables" / table_name).read_text(encoding="utf-8")
        assert metadata["command"] == command[0]
        assert Path(metadata["config_resolved_path"]).exists()
        assert metadata["deterministic_seeds"]["python_random_seed"] == config["calibration"]["seed"]
        assert metadata["model"]["name"] == "toy_flow"
        if command[0] == "nfe-sweep":
            assert metadata["plots_written"] == []
        for column in expected_columns:
            assert column in table


def test_replay_refinement_recomputes_selected_edge_costs_for_final_schedule(tmp_path, monkeypatch) -> None:
    config = _goes_test_config(
        tmp_path,
        oracle={"ref_nfe": 8, "ref_grid_size": 9},
        calibration={"num_samples": 1},
        heldout={"num_samples": 1},
        candidate_grid={"size": 4},
        solver={"target_nfe": 2},
        replay_refinement={"enabled": True, "rounds": 1},
    )

    def fake_solve_minimax(edge_costs, target_nfe, **kwargs):
        del edge_costs, kwargs
        assert target_nfe == 2
        return goes_runner.MinimaxPath(indices=[0, 1, 4], objective=999.0, total_cost=1000.0, edge_costs=[333.0, 444.0])

    def fake_refine(solver, oracle, u_schedule, candidate_grid, metric, **kwargs):
        del solver, oracle, u_schedule, metric, kwargs
        return SimpleNamespace(
            u_schedule=np.asarray([candidate_grid[0], candidate_grid[3], candidate_grid[4]], dtype=np.float64),
            history=[{"round": 0.0, "objective": 1.0, "final_mse": 1.0}],
        )

    monkeypatch.setattr(goes_runner, "solve_minimax_schedule", fake_solve_minimax)
    monkeypatch.setattr(goes_runner, "refine_schedule_blackbox", fake_refine)

    result = goes_runner._search_once(config, tmp_path / "refined_run", command="refinement_test")
    run_dir = Path(result["run_dir"])
    with np.load(run_dir / "edge_costs.npz") as payload:
        edge_costs = payload["edge_costs"]

    schedule = result["schedule"]
    expected = [float(edge_costs[0, 3]), float(edge_costs[3, 4])]
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert schedule["selected_indices"] == [0, 3, 4]
    assert np.allclose(schedule["selected_edge_costs"], expected)
    assert np.isclose(schedule["edge_objective"], max(expected))
    assert metadata["pre_refinement_schedule_indices"] == [0, 1, 4]
    assert metadata["pre_refinement_edge_objective"] == 999.0


def test_goes_nfe_sweep_writes_quality_curve_plot_when_enabled(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    config = _goes_test_config(
        tmp_path,
        oracle={"ref_nfe": 8, "ref_grid_size": 9},
        calibration={"num_samples": 1},
        heldout={"num_samples": 1},
        candidate_grid={"size": 4},
        solver={"target_nfe": 2},
        output={"save_plots": True},
    )
    config_path = tmp_path / "goes_nfe_plot.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "goes.experiment_runner",
            "nfe-sweep",
            "--config",
            str(config_path),
            "--values",
            "2,3",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(json.loads(proc.stdout)["run_dir"])
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert (run_dir / "plots" / "nfe_quality_curve.png").exists()
    assert any(path.endswith("nfe_quality_curve.png") for path in metadata["plots_written"])


def test_goes_schedule_json_exports_to_repository_schedule_bundle(tmp_path) -> None:
    schedule_json = tmp_path / "schedule.json"
    schedule_json.write_text(
        json.dumps(
            {
                "method": "GOES",
                "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
                "solver": "euler",
                "target_nfe": 2,
                "coordinate": "t",
                "coordinate_direction": "decreasing",
                "u_schedule": [0.0, 0.5, 1.0],
                "native_schedule": [999.0, 500.0, 0.0],
                "oracle_cache_key": "abc",
                "rho": 0.1,
                "metric": {"name": "identity"},
                "aggregation": "trimmed_mean_10pct",
                "edge_objective": 1.5,
                "selected_edge_costs": [1.0, 1.5],
                "selected_indices": [0, 1, 2],
                "schedule_hash": "hash",
                "model_asset": "pndm_model_ddim_cifar10",
                "seed": 123,
                "pilot_config": {"num_samples": 4, "batch_size": 2, "num_batches": 2},
                "oracle_config": {"ref_nfe": 64, "ref_grid_size": 65},
                "calibration_cost_estimate": 1024,
                "calibration_cost_unit": "model_evaluation_equivalents",
                "calibration_cost_breakdown": {"total_model_eval_equivalents": 1024},
            }
        ),
        encoding="utf-8",
    )

    output_dir = export_schedule_bundle(
        schedule_json,
        tmp_path / "bundle",
        representation="timesteps",
        backend="pndm",
        solver="euler",
    )
    bundle = ScheduleBundle.load(output_dir)

    assert np.allclose(bundle.timesteps, [999.0, 500.0])
    assert np.allclose(bundle.time_grid, [999.0, 500.0, 0.0])
    assert bundle.meta["schedule_family"] == "GOES"
    assert bundle.meta["schedule_implementation_version"] == GOES_SCHEDULE_IMPLEMENTATION_VERSION
    assert bundle.meta["oracle_cache_key"] == "abc"
    assert bundle.meta["effective_nfe"] == 2
    assert bundle.meta["model_asset"] == "pndm_model_ddim_cifar10"
    assert bundle.meta["seed"] == 123
    assert bundle.meta["pilot_config"]["num_samples"] == 4
    assert bundle.meta["oracle_config"]["ref_nfe"] == 64
    assert bundle.meta["calibration_cost_estimate"] == 1024
    assert bundle.meta["calibration_cost_unit"] == "model_evaluation_equivalents"
    assert bundle.meta["calibration_cost_breakdown"]["total_model_eval_equivalents"] == 1024


def test_goes_schedule_bundle_export_rejects_incomplete_schedule_json(tmp_path) -> None:
    schedule_json = tmp_path / "incomplete_schedule.json"
    schedule_json.write_text(
        json.dumps(
            {
                "method": "GOES",
                "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
                "solver": "euler",
                "target_nfe": 2,
                "coordinate": "t",
                "coordinate_direction": "decreasing",
                "native_schedule": [999.0, 500.0, 0.0],
                "oracle_cache_key": "abc",
                "rho": 0.1,
                "metric": {"name": "identity"},
                "aggregation": "trimmed_mean_10pct",
                "edge_objective": 1.5,
                "selected_edge_costs": [1.0, 1.5],
                "schedule_hash": "hash",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing `u_schedule`"):
        export_schedule_bundle(
            schedule_json,
            tmp_path / "bundle",
            representation="timesteps",
            backend="pndm",
            solver="euler",
        )


def test_export_goes_schedule_cli_accepts_schedule_alias(tmp_path) -> None:
    schedule_json = tmp_path / "schedule.json"
    schedule_json.write_text(
        json.dumps(
            {
                "method": "GOES",
                "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
                "solver": "euler",
                "target_nfe": 2,
                "coordinate": "t",
                "coordinate_direction": "decreasing",
                "u_schedule": [0.0, 0.5, 1.0],
                "native_schedule": [999.0, 500.0, 0.0],
                "oracle_cache_key": "abc",
                "rho": 0.1,
                "metric": {"name": "identity"},
                "aggregation": "trimmed_mean_10pct",
                "edge_objective": 1.5,
                "selected_edge_costs": [1.0, 1.5],
                "selected_indices": [0, 1, 2],
                "schedule_hash": "hash",
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "bundle_from_cli"

    subprocess.run(
        [
            sys.executable,
            "scripts/run/export_goes_schedule.py",
            "--schedule",
            str(schedule_json),
            "--output-dir",
            str(output_dir),
            "--representation",
            "timesteps",
            "--backend",
            "pndm",
            "--solver",
            "euler",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    bundle = ScheduleBundle.load(output_dir)

    assert bundle.meta["schedule_family"] == "GOES"
    assert np.allclose(bundle.time_grid, [999.0, 500.0, 0.0])


def test_verify_goes_schedule_checks_payload_and_bundle(tmp_path) -> None:
    schedule_json = tmp_path / "schedule.json"
    schedule_json.write_text(
        json.dumps(
            {
                "method": "GOES",
                "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
                "solver": "euler",
                "target_nfe": 2,
                "coordinate": "t",
                "coordinate_direction": "decreasing",
                "u_schedule": [0.0, 0.5, 1.0],
                "native_schedule": [999.0, 500.0, 0.0],
                "oracle_cache_key": "abc",
                "rho": 0.1,
                "metric": {"name": "identity"},
                "aggregation": "trimmed_mean_10pct",
                "edge_objective": 1.5,
                "selected_edge_costs": [1.0, 1.5],
                "selected_indices": [0, 1, 2],
                "schedule_hash": "hash",
            }
        ),
        encoding="utf-8",
    )
    bundle_dir = export_schedule_bundle(
        schedule_json,
        tmp_path / "bundle",
        representation="timesteps",
        backend="pndm",
        solver="euler",
    )

    result = verify_goes_schedule(schedule_json, bundle_dir=bundle_dir)

    assert result["schedule"]["target_nfe"] == 2
    assert result["bundle"]["effective_nfe"] == 2
    assert result["bundle"]["schedule_hash"] == "hash"


def test_verify_schedule_bundle_rejects_nonmonotone_arrays(tmp_path) -> None:
    bundle_dir = ScheduleBundle(
        timesteps=np.asarray([999.0, 500.0, 700.0], dtype=np.float64),
        time_grid=np.asarray([999.0, 500.0, 700.0, 0.0], dtype=np.float64),
        meta={
            "schedule_family": "GOES",
            "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
            "effective_nfe": 3,
            "oracle_cache_key": "abc",
            "schedule_hash": "hash",
            "representation": "timesteps",
        },
    ).save(tmp_path / "bad_bundle")

    with pytest.raises(ValueError, match="time_grid must be strictly monotone"):
        verify_schedule_bundle(bundle_dir, expected_nfe=3)


def test_verify_goes_schedule_rejects_nonmonotone_unified_schedule() -> None:
    payload = {
        "method": "GOES",
        "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
        "solver": "euler",
        "target_nfe": 2,
        "coordinate": "t",
        "coordinate_direction": "increasing",
        "u_schedule": [0.0, 0.7, 0.6],
        "native_schedule": [999.0, 500.0, 0.0],
        "oracle_cache_key": "abc",
        "rho": 0.1,
        "metric": {"name": "identity"},
        "aggregation": "trimmed_mean_10pct",
        "edge_objective": 1.5,
        "selected_edge_costs": [1.0, 1.5],
        "selected_indices": [0, 1, 2],
        "schedule_hash": "hash",
    }

    try:
        verify_schedule_payload(payload)
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("Non-monotone unified GOES schedules must be rejected.")


def test_verify_goes_schedule_requires_core_metadata_fields() -> None:
    payload = {
        "method": "GOES",
        "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
        "solver": "euler",
        "target_nfe": 1,
        "coordinate": "t",
        "coordinate_direction": "increasing",
        "u_schedule": [0.0, 1.0],
        "native_schedule": [0.0, 1.0],
        "oracle_cache_key": "abc",
        "rho": 0.1,
        "metric": {"name": "identity"},
        "aggregation": "trimmed_mean_10pct",
        "edge_objective": 0.5,
        "selected_edge_costs": [0.5],
        "selected_indices": [0, 1],
        "schedule_hash": "hash",
    }

    for key, expected in [
        ("solver", "`solver` must be recorded"),
        ("coordinate", "`coordinate` must be recorded"),
        ("coordinate_direction", "`coordinate_direction` must be recorded"),
        ("edge_objective", "`edge_objective` must be finite"),
    ]:
        broken = dict(payload)
        broken.pop(key)
        with pytest.raises(ValueError, match=expected):
            verify_schedule_payload(broken)


def test_verify_goes_schedule_cli(tmp_path) -> None:
    schedule_json = tmp_path / "schedule.json"
    schedule_json.write_text(
        json.dumps(
            {
                "method": "GOES",
                "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
                "solver": "euler",
                "target_nfe": 1,
                "coordinate": "t",
                "coordinate_direction": "increasing",
                "u_schedule": [0.0, 1.0],
                "native_schedule": [0.0, 1.0],
                "oracle_cache_key": "abc",
                "rho": 0.1,
                "metric": {"name": "identity"},
                "aggregation": "trimmed_mean_10pct",
                "edge_objective": 0.5,
                "selected_edge_costs": [0.5],
                "selected_indices": [0, 1],
                "schedule_hash": "hash",
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run/verify_goes_schedule.py",
            "--schedule",
            str(schedule_json),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(proc.stdout)

    assert result["schedule"]["schedule_hash"] == "hash"


def test_pndm_goes_export_payload_writes_schedule_hash(tmp_path) -> None:
    module = _load_script_module("export_goes_pndm_schedule_payload_test", "scripts/run/export_goes_pndm_schedule.py")
    payload = {
        "method": "GOES",
        "schedule_implementation_version": GOES_SCHEDULE_IMPLEMENTATION_VERSION,
        "solver": "euler",
        "target_nfe": 2,
        "coordinate": "negative_timesteps",
        "coordinate_direction": "increasing_u_native_decreasing",
        "u_schedule": [-999.0, -500.0, -0.0],
        "native_schedule": [999.0, 500.0, 0.0],
        "rho": 0.1,
        "metric": {"name": "identity"},
        "aggregation": "trimmed_mean_10pct",
        "oracle_cache_key": "abc",
        "edge_objective": 1.0,
        "selected_edge_costs": [0.8, 1.0],
        "selected_indices": [0, 1, 2],
        "schedule_hash": module.stable_hash([-999.0, -500.0, -0.0]),
    }

    module.save_schedule_outputs(
        tmp_path,
        payload=payload,
        selected_indices=payload["selected_indices"],
        selected_edge_costs=payload["selected_edge_costs"],
    )
    native_payload = json.loads((tmp_path / "schedule_native.json").read_text(encoding="utf-8"))

    assert native_payload["schedule_hash"] == payload["schedule_hash"]


def test_pndm_goes_export_metadata_records_calibration_and_oracle_config() -> None:
    module = _load_script_module("export_goes_pndm_schedule_metadata_test", "scripts/run/export_goes_pndm_schedule.py")
    args = SimpleNamespace(
        solver="euler",
        seed=123,
        batch_size=4,
        num_batches=3,
        microbatch_size=2,
        ref_nfe=64,
        ref_grid_size=65,
        oracle_cache_dir="outputs/goes/cache",
        no_reuse_oracle=False,
        candidate_grid_size=32,
        model_output_type="epsilon",
        sigma_floor=1.0e-6,
    )

    metadata = module._goes_context_metadata(
        args,
        dataset_name="cifar10",
        model_asset="pndm_model_ddim_cifar10",
        coordinate_domain="timesteps",
        model_path="checkpoints/pndm.pt",
        dataset_config_path="configs/datasets/cifar10.yaml",
    )

    assert metadata["model_asset"] == "pndm_model_ddim_cifar10"
    assert metadata["model_path"] == "checkpoints/pndm.pt"
    assert metadata["dataset"] == "cifar10"
    assert metadata["dataset_config_path"] == "configs/datasets/cifar10.yaml"
    assert metadata["guidance_scale"] is None
    assert metadata["calibration_config"]["num_samples"] == 12
    assert metadata["pilot_config"]["microbatch_size"] == 2
    assert metadata["oracle_config"]["ref_nfe"] == 64
    assert metadata["oracle_config"]["reuse"] is True
    assert metadata["candidate_grid_config"]["size"] == 32
    assert metadata["calibration_cost_unit"] == "model_evaluation_equivalents"
    assert metadata["calibration_cost_breakdown"]["cfg_multiplier"] == 1
    assert metadata["calibration_cost_breakdown"]["candidate_edges"] == 528
    assert metadata["calibration_cost_breakdown"]["solver_evals_per_edge"] == 1
    assert metadata["calibration_cost_estimate"] == 12 * ((4 * 64 + 65) + 528)

    args.solver = "heun2"
    heun_metadata = module._goes_context_metadata(
        args,
        dataset_name="cifar10",
        model_asset="pndm_model_ddim_cifar10",
        coordinate_domain="timesteps",
    )
    assert heun_metadata["calibration_cost_breakdown"]["solver_evals_per_edge"] == 2
    assert heun_metadata["calibration_cost_estimate"] == 12 * ((4 * 64 + 65) + 2 * 528)


def test_diffusers_goes_export_metadata_records_prompt_guidance_and_oracle_config() -> None:
    module = _load_script_module(
        "export_goes_diffusers_schedule_metadata_test",
        "scripts/run/export_goes_diffusers_schedule.py",
    )
    args = SimpleNamespace(
        model_asset="hf_sd35_medium",
        solver="flow_euler",
        seed=123,
        guidance_scale=3.5,
        prompt_asset="diffusers_smoke_prompts",
        height=512,
        width=512,
        dtype="bfloat16",
        physical_grid_mode="scheduler_sigmas",
        batch_size=2,
        num_batches=4,
        microbatch_size=0,
        ref_nfe=64,
        ref_grid_size=65,
        oracle_cache_dir="outputs/goes/cache",
        no_reuse_oracle=True,
        candidate_grid_size=32,
    )

    metadata = module._goes_context_metadata(args, pipeline_kind="sd3", prompt_count=8)

    assert metadata["model_asset"] == "hf_sd35_medium"
    assert metadata["model_path"] == ""
    assert metadata["guidance_scale"] == 3.5
    assert metadata["prompt_asset"] == "diffusers_smoke_prompts"
    assert metadata["prompt_path"] == ""
    assert metadata["prompt_count"] == 8
    assert metadata["calibration_config"]["num_samples"] == 8
    assert metadata["pilot_config"]["microbatch_size"] is None
    assert metadata["oracle_config"]["ref_grid_size"] == 65
    assert metadata["oracle_config"]["reuse"] is False
    assert metadata["candidate_grid_config"]["type"] == "uniform_in_negative_sigma"
    assert metadata["calibration_cost_unit"] == "model_evaluation_equivalents"
    assert metadata["calibration_cost_breakdown"]["cfg_multiplier"] == 2
    assert metadata["calibration_cost_breakdown"]["candidate_edges"] == 528
    assert metadata["calibration_cost_breakdown"]["solver_evals_per_edge"] == 1
    assert metadata["calibration_cost_estimate"] == 8 * 2 * ((4 * 64 + 65) + 528)

    args.solver = "flow_heun"
    heun_metadata = module._goes_context_metadata(args, pipeline_kind="sd3", prompt_count=8)
    assert heun_metadata["calibration_cost_breakdown"]["cfg_multiplier"] == 2
    assert heun_metadata["calibration_cost_breakdown"]["solver_evals_per_edge"] == 2
    assert heun_metadata["calibration_cost_estimate"] == 8 * 2 * ((4 * 64 + 65) + 2 * 528)

    path_metadata = module._goes_context_metadata(
        args,
        pipeline_kind="sd3",
        prompt_count=8,
        model_path="models/sd35",
        prompt_path="configs/prompts.json",
    )
    assert path_metadata["model_path"] == "models/sd35"
    assert path_metadata["prompt_path"] == "configs/prompts.json"


def test_real_goes_exporters_record_deterministic_seed_metadata() -> None:
    pndm_module = _load_script_module(
        "export_goes_pndm_schedule_seed_metadata_test",
        "scripts/run/export_goes_pndm_schedule.py",
    )
    diffusers_module = _load_script_module(
        "export_goes_diffusers_schedule_seed_metadata_test",
        "scripts/run/export_goes_diffusers_schedule.py",
    )

    pndm = pndm_module._set_deterministic_seeds(123)
    diffusers = diffusers_module._set_deterministic_seeds(456)

    assert pndm["python_random_seed"] == 123
    assert pndm["numpy_seed"] == 123
    assert pndm["torch_seed"] == 123
    assert pndm["common_random_numbers"] is True
    assert diffusers["python_random_seed"] == 456
    assert diffusers["numpy_seed"] == 456
    assert diffusers["torch_seed"] == 456
    assert diffusers["common_random_numbers"] is True


def test_real_goes_exporters_write_resolved_export_configs(tmp_path) -> None:
    pndm_module = _load_script_module(
        "export_goes_pndm_schedule_resolved_config_test",
        "scripts/run/export_goes_pndm_schedule.py",
    )
    diffusers_module = _load_script_module(
        "export_goes_diffusers_schedule_resolved_config_test",
        "scripts/run/export_goes_diffusers_schedule.py",
    )
    pndm_args = SimpleNamespace(
        manifest="configs/assets_manifest.yaml",
        dataset_config="configs/datasets/cifar10.yaml",
        model_asset="pndm_model_ddim_cifar10",
        solver="euler",
        nfe=4,
        output_dir=str(tmp_path / "pndm"),
        oracle_cache_dir="outputs/goes/cache",
        seed=7,
    )
    diffusers_args = SimpleNamespace(
        manifest="configs/assets_manifest.yaml",
        model_asset="hf_sd35_medium",
        prompt_asset="diffusers_smoke_prompts",
        solver="flow_euler",
        nfe=4,
        output_dir=str(tmp_path / "diffusers"),
        oracle_cache_dir="outputs/goes/cache",
        seed=9,
    )

    pndm_path = pndm_module._write_resolved_export_config(
        tmp_path / "pndm",
        args=pndm_args,
        context_metadata={"model_path": "models/pndm.pt", "dataset": "cifar10"},
    )
    diffusers_path = diffusers_module._write_resolved_export_config(
        tmp_path / "diffusers",
        args=diffusers_args,
        context_metadata={"model_path": "models/sd35", "prompt_path": "prompts.json"},
    )

    pndm_config = load_yaml(pndm_path)
    diffusers_config = load_yaml(diffusers_path)
    assert pndm_path.name == "config.resolved.yaml"
    assert diffusers_path.name == "config.resolved.yaml"
    assert pndm_config["method"] == "goes"
    assert pndm_config["backend"] == "pndm"
    assert pndm_config["arguments"]["seed"] == 7
    assert pndm_config["resolved_context"]["model_path"] == "models/pndm.pt"
    assert diffusers_config["backend"] == "diffusers"
    assert diffusers_config["arguments"]["prompt_asset"] == "diffusers_smoke_prompts"
    assert diffusers_config["resolved_context"]["prompt_path"] == "prompts.json"


def test_real_goes_exporters_mark_heldout_metrics_not_evaluated(tmp_path) -> None:
    pndm_module = _load_script_module(
        "export_goes_pndm_schedule_metric_rows_test",
        "scripts/run/export_goes_pndm_schedule.py",
    )
    diffusers_module = _load_script_module(
        "export_goes_diffusers_schedule_metric_rows_test",
        "scripts/run/export_goes_diffusers_schedule.py",
    )

    pndm_calibration, pndm_heldout = pndm_module._schedule_export_metric_rows(
        solver="euler",
        nfe=10,
        num_samples=8,
        final_latent_mse=0.1,
        replay_loss=0.2,
        fallback_fraction=0.0,
        schedule_dir=tmp_path / "pndm",
        oracle_cache_key="pndm-cache",
        theory_covered=True,
    )
    diffusers_calibration, diffusers_heldout = diffusers_module._schedule_export_metric_rows(
        solver="flow_euler",
        nfe=10,
        guidance_scale=7.5,
        num_samples=4,
        final_latent_mse=0.3,
        replay_loss=0.4,
        fallback_fraction=0.0,
        schedule_dir=tmp_path / "diffusers",
        oracle_cache_key="diffusers-cache",
        theory_covered=True,
    )

    assert pndm_calibration["split"] == "calibration"
    assert pndm_calibration["status"] == "OK"
    assert pndm_heldout["split"] == "heldout"
    assert pndm_heldout["status"] == "NOT_EVALUATED"
    assert pndm_heldout["num_samples"] == 0
    assert pndm_heldout["final_latent_mse"] == ""
    assert "experiment launcher" in pndm_heldout["note"]
    assert diffusers_calibration["guidance_scale"] == 7.5
    assert diffusers_heldout["guidance_scale"] == 7.5
    assert diffusers_heldout["status"] == "NOT_EVALUATED"


def test_pndm_goes_exporter_validates_numeric_arguments() -> None:
    module = _load_script_module("export_goes_pndm_schedule_validation_test", "scripts/run/export_goes_pndm_schedule.py")
    valid = SimpleNamespace(
        nfe=4,
        batch_size=2,
        num_batches=1,
        microbatch_size=0,
        ref_nfe=8,
        ref_grid_size=9,
        candidate_grid_size=8,
        rho=0.1,
        trim_ratio=0.1,
        cvar_alpha=0.8,
        sigma_floor=1.0e-6,
        sigma_data=0.5,
    )

    module._validate_args(valid)

    invalid_cases = [
        ("nfe", 0, "--nfe must be positive"),
        ("batch_size", 0, "--batch-size must be positive"),
        ("num_batches", 0, "--num-batches must be positive"),
        ("microbatch_size", -1, "--microbatch-size must be non-negative"),
        ("ref_nfe", 0, "--ref-nfe must be positive"),
        ("ref_grid_size", 1, "--ref-grid-size must be at least 2"),
        ("candidate_grid_size", 3, "--candidate-grid-size must be at least --nfe"),
        ("rho", math.inf, "--rho must be finite"),
        ("trim_ratio", 0.5, "--trim-ratio must be finite"),
        ("cvar_alpha", 1.0, "--cvar-alpha must be finite"),
        ("sigma_floor", 0.0, "--sigma-floor must be finite"),
        ("sigma_data", 0.0, "--sigma-data must be finite"),
    ]
    for field, value, match in invalid_cases:
        args = SimpleNamespace(**{**vars(valid), field: value})
        with pytest.raises(ValueError, match=match):
            module._validate_args(args)


def test_diffusers_goes_exporter_validates_numeric_arguments() -> None:
    module = _load_script_module(
        "export_goes_diffusers_schedule_validation_test",
        "scripts/run/export_goes_diffusers_schedule.py",
    )
    valid = SimpleNamespace(
        nfe=4,
        batch_size=2,
        num_batches=1,
        microbatch_size=0,
        height=512,
        width=512,
        guidance_scale=3.5,
        ref_nfe=8,
        ref_grid_size=9,
        candidate_grid_size=8,
        rho=0.1,
        trim_ratio=0.1,
        cvar_alpha=0.8,
        sigma_data=0.5,
    )

    module._validate_args(valid)

    invalid_cases = [
        ("nfe", 0, "--nfe must be positive"),
        ("batch_size", 0, "--batch-size must be positive"),
        ("num_batches", 0, "--num-batches must be positive"),
        ("microbatch_size", -1, "--microbatch-size must be non-negative"),
        ("height", 0, "--height must be positive"),
        ("width", 0, "--width must be positive"),
        ("guidance_scale", math.nan, "--guidance-scale must be finite"),
        ("ref_nfe", 0, "--ref-nfe must be positive"),
        ("ref_grid_size", 1, "--ref-grid-size must be at least 2"),
        ("candidate_grid_size", 3, "--candidate-grid-size must be at least --nfe"),
        ("rho", math.inf, "--rho must be finite"),
        ("trim_ratio", -0.1, "--trim-ratio must be finite"),
        ("cvar_alpha", 1.0, "--cvar-alpha must be finite"),
        ("sigma_data", 0.0, "--sigma-data must be finite"),
    ]
    for field, value, match in invalid_cases:
        args = SimpleNamespace(**{**vars(valid), field: value})
        with pytest.raises(ValueError, match=match):
            module._validate_args(args)


def test_goes_exporter_dry_runs_report_calibration_cost(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pndm = subprocess.run(
        [
            sys.executable,
            "scripts/run/export_goes_pndm_schedule.py",
            "--dataset-config",
            "configs/datasets/cifar10.yaml",
            "--solver",
            "euler",
            "--nfe",
            "4",
            "--output-dir",
            str(tmp_path / "pndm"),
            "--batch-size",
            "2",
            "--num-batches",
            "1",
            "--ref-nfe",
            "8",
            "--ref-grid-size",
            "9",
            "--candidate-grid-size",
            "8",
            "--metric",
            "edm_scalar",
            "--sigma-data",
            "0.5",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    pndm_payload = json.loads(pndm.stdout)
    assert pndm_payload["would_load_model"] is False
    assert pndm_payload["calibration_cost_unit"] == "model_evaluation_equivalents"
    assert pndm_payload["calibration_cost_estimate"] == pndm_payload["calibration_cost_breakdown"][
        "total_model_eval_equivalents"
    ]

    diffusers = subprocess.run(
        [
            sys.executable,
            "scripts/run/export_goes_diffusers_schedule.py",
            "--model-asset",
            "hf_sd35_medium",
            "--prompt-asset",
            "diffusers_smoke_prompts",
            "--solver",
            "flow_euler",
            "--nfe",
            "4",
            "--output-dir",
            str(tmp_path / "diffusers"),
            "--batch-size",
            "2",
            "--num-batches",
            "1",
            "--ref-nfe",
            "8",
            "--ref-grid-size",
            "9",
            "--candidate-grid-size",
            "8",
            "--metric",
            "edm_scalar",
            "--sigma-data",
            "0.5",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    diffusers_payload = json.loads(diffusers.stdout)
    assert diffusers_payload["would_load_pipeline"] is False
    assert diffusers_payload["calibration_cost_unit"] == "model_evaluation_equivalents"
    assert diffusers_payload["calibration_cost_breakdown"]["cfg_multiplier"] == 2
    assert diffusers_payload["calibration_cost_estimate"] == diffusers_payload["calibration_cost_breakdown"][
        "total_model_eval_equivalents"
    ]


def test_blackbox_replay_refinement_improves_prefix_dependent_loss() -> None:
    grid = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    states = grid.reshape(1, -1, 1)
    tangents = np.ones_like(states)
    oracle = OracleData(
        states=states,
        tangents=tangents,
        u_grid=grid,
        conditions=np.asarray([0.0]),
        noise_seeds=np.asarray([0]),
        metadata={},
    )

    class QuadraticDriftBiasSolver:
        def single_edge_step_from_state(self, x_a, a, b, condition):
            del condition
            h = float(b) - float(a)
            return np.asarray(x_a, dtype=np.float64) + h + h * h

    result = refine_schedule_blackbox(
        QuadraticDriftBiasSolver(),
        oracle,
        np.asarray([0.0, 0.8, 1.0], dtype=np.float64),
        grid,
        IdentityMetric(),
        rho=1.0,
        aggregation="mean",
        rounds=3,
        local_window=8,
    )

    assert result.u_schedule[0] == 0.0
    assert result.u_schedule[-1] == 1.0
    assert np.all(np.diff(result.u_schedule) > 0.0)
    assert abs(result.u_schedule[1] - 0.5) <= 0.1
    assert result.history[-1]["objective"] < result.history[0]["objective"]


def test_diffusers_goes_exporter_rejects_unsupported_solver_modes() -> None:
    module = _load_script_module("export_goes_diffusers_schedule_test", "scripts/run/export_goes_diffusers_schedule.py")

    assert module._validate_solver_pipeline_pair("flux", "flow_heun") == "heun2"
    assert module._validate_solver_pipeline_pair("sdxl", "euler") == "euler"

    try:
        module._validate_solver_pipeline_pair("flux", "flow_unipc")
    except ValueError as exc:
        assert "Multistep solvers require black-box replay refinement" in str(exc)
    else:
        raise AssertionError("flow_unipc should be rejected until scheduler-history replay refinement is implemented.")

    try:
        module._validate_solver_pipeline_pair("sdxl", "dpm_solver_pp")
    except ValueError as exc:
        assert "scheduler-history replay refinement" in str(exc)
    else:
        raise AssertionError("VP DPM solver should be rejected until replay refinement is implemented.")


def test_diffusers_goes_exporter_repeats_prompt_asset_without_mixing_splits(tmp_path) -> None:
    module = _load_script_module("export_goes_diffusers_schedule_prompts_test", "scripts/run/export_goes_diffusers_schedule.py")
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(["a", "b"]), encoding="utf-8")

    class FakeManifest:
        def has(self, key):
            return key == "prompt_key"

        def path(self, key):
            assert key == "prompt_key"
            return prompt_file

    assert module._load_prompt_batch(FakeManifest(), "prompt_key", 5) == ["a", "b", "a", "b", "a"]


def test_torch_velocity_oracle_cache_reuses_repository_velocity_interface(tmp_path) -> None:
    initial = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float64)

    def velocity_fn(sample, coordinate, sample_start=0, sample_stop=None):
        del sample_stop
        result = torch.zeros_like(sample)
        result[:, 0] = 1.0
        result[:, 1] = 2.0 * coordinate.to(device=sample.device, dtype=sample.dtype)
        # Depend on absolute sample index to verify microbatch slicing preserves alignment.
        absolute = torch.arange(sample_start, sample_start + sample.shape[0], device=sample.device, dtype=sample.dtype)
        result[:, 0] += 0.1 * absolute
        return result

    metadata = {
        "model_identifier": "dummy_torch_velocity",
        "ode_sampler_family": "deterministic_velocity_ode",
        "coordinate_mapping": {"coordinate": "u", "direction": "increasing", "u_min": 0.0, "u_max": 1.0},
        "cfg": {"guidance_scale": 1.0},
    }
    grid = np.linspace(0.0, 1.0, 5, dtype=np.float64)

    first = build_or_load_torch_velocity_oracle(
        cache_dir=tmp_path / "torch_cache",
        initial_sample=initial,
        velocity_fn=velocity_fn,
        u_grid=grid,
        ref_nfe=16,
        metadata=metadata,
        microbatch_size=1,
    )
    second = build_or_load_torch_velocity_oracle(
        cache_dir=tmp_path / "torch_cache",
        initial_sample=initial,
        velocity_fn=velocity_fn,
        u_grid=grid,
        ref_nfe=16,
        metadata=metadata,
        microbatch_size=1,
    )

    assert first.cache_key == second.cache_key
    assert second.loaded_from_cache
    assert np.allclose(second.oracle.tangents[:, 0, 0], [1.0, 1.1])
    assert second.oracle.metadata["model_identifier"] == "dummy_torch_velocity"
    assert first.oracle.metadata["oracle_cache_key"] == first.cache_key
    assert second.oracle.metadata["oracle_cache_key"] == second.cache_key
    assert second.oracle.metadata["interpolation"] == "linear"
    assert second.oracle.metadata["noise_seeds"] == [0, 1]

    edge_table = evaluate_torch_velocity_edge_table(
        solver_name="euler",
        velocity_fn=velocity_fn,
        oracle=second.oracle,
        candidate_grid=grid,
        metric=IdentityMetric(),
        rho=0.1,
        device="cpu",
        dtype=torch.float64,
    )
    assert edge_table.edge_costs.shape == (5, 5)
    assert np.isfinite(edge_table.edge_costs[0, -1])

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .aggregation import aggregation_label
from .config import load_config, repo_path, stable_hash, write_config
from .coordinate import make_coordinate_adapter
from .dp_minimax import MinimaxPath, solve_minimax_schedule
from .edge_evaluator import EdgeCostTable, ReplayMetrics, evaluate_edge_table, evaluate_replay_metrics
from .logging_utils import dump_json, make_run_dir, maybe_write_nfe_quality_curve, maybe_write_plots, runtime_metadata, write_csv
from .metrics import Metric, make_metric
from .oracle import OracleData
from .oracle_cache import OracleCacheResult, build_or_load_oracle
from .replay_refinement import refine_schedule_blackbox
from .schedules import save_schedule_outputs, schedule_payload
from .toy import make_solver, make_toy_model
from .verify import verify_schedule_payload


def _prepare_common(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    model = make_toy_model(config["model"])
    coordinate = make_coordinate_adapter(config["coordinate"])
    solver = make_solver(str(config["solver"]["name"]), model, mode=str(config["solver"].get("mode", "one_step")))
    return model, coordinate, solver


def _model_run_metadata(config: dict[str, Any], model_identifier: str | None = None) -> dict[str, Any]:
    model_config = dict(config.get("model", {}))
    checkpoint = model_config.get("checkpoint")
    return {
        "identifier": model_identifier or str(model_config.get("name", "")),
        "name": model_config.get("name"),
        "checkpoint": checkpoint,
        "checkpoint_path": None if checkpoint is None else str(checkpoint),
        "dtype": model_config.get("dtype"),
        "device": model_config.get("device"),
    }


def _set_deterministic_seeds(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("calibration", {}).get("seed", 0))
    numpy_seed = seed % (2**32 - 1)
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch_seed_set = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch_seed_set = True
    except Exception:
        pass
    return {
        "python_random_seed": seed,
        "numpy_seed": numpy_seed,
        "torch_seed": seed if torch_seed_set else None,
        "torch_seed_set": torch_seed_set,
        "calibration_seed": int(config.get("calibration", {}).get("seed", seed)),
        "heldout_seed": int(config.get("heldout", {}).get("seed", seed)),
        "common_random_numbers": True,
        "data_loader_worker_seed": None,
    }


def _write_top_level_run_metadata(
    config: dict[str, Any],
    run_dir: Path,
    *,
    command: str,
    deterministic_seeds: dict[str, Any],
    extra: dict[str, Any],
) -> None:
    config_resolved_path = write_config(config, run_dir / "config.resolved.yaml")
    payload = {
        "command": command,
        "run_dir": str(run_dir),
        "config_resolved_path": str(config_resolved_path),
        "runtime": runtime_metadata(),
        "deterministic_seeds": deterministic_seeds,
        "model": _model_run_metadata(config),
    }
    payload.update(extra)
    dump_json(payload, run_dir / "run_metadata.json")


def _bootstrap_mse_summary(values: np.ndarray, *, seed: int, num_bootstrap: int = 1000) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"bootstrap_se": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    if data.size == 1:
        value = float(data[0])
        return {"bootstrap_se": 0.0, "ci95_low": value, "ci95_high": value}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, data.size, size=(int(num_bootstrap), data.size))
    means = np.mean(data[indices], axis=1)
    return {
        "bootstrap_se": float(np.std(means, ddof=1)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _metrics_row(
    *,
    split: str,
    schedule_name: str,
    solver_name: str,
    target_nfe: int,
    metrics: ReplayMetrics,
    oracle: OracleData,
    schedule_hash: str,
    guidance_scale: float,
) -> dict[str, Any]:
    bootstrap = _bootstrap_mse_summary(
        metrics.final_sample_mse,
        seed=int(oracle.metadata.get("calibration", {}).get("seed", 0)),
    )
    return {
        "split": split,
        "schedule": schedule_name,
        "solver": solver_name,
        "nfe": int(target_nfe),
        "num_samples": oracle.num_samples,
        "final_latent_mse": metrics.final_mse,
        "final_latent_mse_bootstrap_se": bootstrap["bootstrap_se"],
        "final_latent_mse_ci95_low": bootstrap["ci95_low"],
        "final_latent_mse_ci95_high": bootstrap["ci95_high"],
        "replay_loss": metrics.replay_loss,
        "max_endpoint_cost": float(np.max(metrics.endpoint_costs)) if metrics.endpoint_costs.size else 0.0,
        "mean_endpoint_mse": float(np.mean(metrics.endpoint_mse)) if metrics.endpoint_mse.size else 0.0,
        "fallback_fraction": metrics.fallback_fraction,
        "guidance_scale": float(guidance_scale),
        "split_hash": oracle.metadata.get("condition_split_hash", ""),
        "initial_noise_hash": oracle.metadata.get("initial_noise_hash", ""),
        "schedule_hash": schedule_hash,
    }


def _evaluate_common_schedules(
    *,
    config: dict[str, Any],
    solver: Any,
    metric: Metric,
    calibration_oracle: OracleData,
    heldout_oracle: OracleData,
    goes_schedule: np.ndarray,
    goes_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_nfe = int(config["solver"]["target_nfe"])
    coordinate = make_coordinate_adapter(config["coordinate"])
    uniform_schedule = np.linspace(coordinate.u_min, coordinate.u_max, target_nfe + 1, dtype=np.float64)
    schedules = {
        "GOES": (goes_schedule, goes_hash),
        "uniform_in_u": (uniform_schedule, stable_hash(uniform_schedule.tolist())),
    }
    calibration_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    for name, (u_schedule, schedule_hash) in schedules.items():
        calibration_metrics = evaluate_replay_metrics(
            solver,
            calibration_oracle,
            u_schedule,
            metric,
            rho=float(config["mixed_defect"]["rho"]),
            aggregation=config["aggregation"],
            eps=float(config["mixed_defect"].get("eps", 1.0e-12)),
            fallback_full_residual_on_tiny_tangent=bool(
                config["mixed_defect"].get("fallback_full_residual_on_tiny_tangent", True)
            ),
        )
        heldout_metrics = evaluate_replay_metrics(
            solver,
            heldout_oracle,
            u_schedule,
            metric,
            rho=float(config["mixed_defect"]["rho"]),
            aggregation=config["aggregation"],
            eps=float(config["mixed_defect"].get("eps", 1.0e-12)),
            fallback_full_residual_on_tiny_tangent=bool(
                config["mixed_defect"].get("fallback_full_residual_on_tiny_tangent", True)
            ),
        )
        calibration_rows.append(
            _metrics_row(
                split="calibration",
                schedule_name=name,
                solver_name=solver.name,
                target_nfe=target_nfe,
                metrics=calibration_metrics,
                oracle=calibration_oracle,
                schedule_hash=schedule_hash,
                guidance_scale=float(config["calibration"].get("guidance_scale", 1.0)),
            )
        )
        heldout_rows.append(
            _metrics_row(
                split="heldout",
                schedule_name=name,
                solver_name=solver.name,
                target_nfe=target_nfe,
                metrics=heldout_metrics,
                oracle=heldout_oracle,
                schedule_hash=schedule_hash,
                guidance_scale=float(config["calibration"].get("guidance_scale", 1.0)),
            )
        )
    return calibration_rows, heldout_rows


def _per_sample_endpoint_mse_trace(solver: Any, oracle: OracleData, u_schedule: np.ndarray) -> np.ndarray:
    schedule = np.asarray(u_schedule, dtype=np.float64)
    current = oracle.state_at(float(schedule[0]))
    traces: list[np.ndarray] = []
    for a, b in zip(schedule[:-1], schedule[1:]):
        current = solver.single_edge_step_from_state(current, float(a), float(b), oracle.conditions)
        target = oracle.state_at(float(b))
        residual = current - target
        traces.append(np.mean(np.reshape(residual, (residual.shape[0], -1)) ** 2, axis=1))
    if not traces:
        return np.zeros((oracle.num_samples, 0), dtype=np.float64)
    return np.stack(traces, axis=1)


def _per_sample_final_mse(solver: Any, oracle: OracleData, u_schedule: np.ndarray) -> np.ndarray:
    endpoint_trace = _per_sample_endpoint_mse_trace(solver, oracle, u_schedule)
    if endpoint_trace.shape[1] == 0:
        return np.zeros((oracle.num_samples,), dtype=np.float64)
    return endpoint_trace[:, -1]


def _write_failure_cases(
    *,
    run_dir: Path,
    config: dict[str, Any],
    solver: Any,
    heldout_oracle: OracleData,
    goes_schedule: np.ndarray,
    uniform_schedule: np.ndarray,
    edge_objective: float,
    tiny_tangent_fallback_fraction: float,
) -> None:
    goes_mse = _per_sample_final_mse(solver, heldout_oracle, goes_schedule)
    baseline_mse = _per_sample_final_mse(solver, heldout_oracle, uniform_schedule)
    goes_endpoint_mse = _per_sample_endpoint_mse_trace(solver, heldout_oracle, goes_schedule)
    baseline_endpoint_mse = _per_sample_endpoint_mse_trace(solver, heldout_oracle, uniform_schedule)
    deltas = goes_mse - baseline_mse
    rows: list[dict[str, Any]] = []
    for rank, sample_index in enumerate(np.argsort(deltas)[::-1]):
        delta = float(deltas[sample_index])
        if delta <= 0.0:
            continue
        rows.append(
            {
                "rank": int(rank),
                "sample_index": int(sample_index),
                "condition": float(heldout_oracle.conditions[sample_index]),
                "seed": int(heldout_oracle.noise_seeds[sample_index]),
                "baseline": "uniform_in_u",
                "goes_final_latent_mse": float(goes_mse[sample_index]),
                "baseline_final_latent_mse": float(baseline_mse[sample_index]),
                "mse_delta_goes_minus_baseline": delta,
                "goes_replay_endpoint_mse": json.dumps(
                    [float(item) for item in goes_endpoint_mse[sample_index].tolist()]
                ),
                "baseline_replay_endpoint_mse": json.dumps(
                    [float(item) for item in baseline_endpoint_mse[sample_index].tolist()]
                ),
                "solver": solver.name,
                "nfe": int(config["solver"]["target_nfe"]),
                "guidance_scale": float(config["calibration"].get("guidance_scale", 1.0)),
                "selected_schedule": json.dumps([float(item) for item in goes_schedule]),
                "edge_objective": float(edge_objective),
                "tiny_tangent_fallback_fraction": float(tiny_tangent_fallback_fraction),
                "notes": "Toy latent validation; no image artifact is produced.",
            }
        )
        if len(rows) >= min(8, heldout_oracle.num_samples):
            break
    if not rows:
        rows.append(
            {
                "rank": 0,
                "sample_index": "",
                "condition": "",
                "seed": "",
                "baseline": "uniform_in_u",
                "goes_final_latent_mse": "",
                "baseline_final_latent_mse": "",
                "mse_delta_goes_minus_baseline": "",
                "goes_replay_endpoint_mse": "",
                "baseline_replay_endpoint_mse": "",
                "solver": solver.name,
                "nfe": int(config["solver"]["target_nfe"]),
                "guidance_scale": float(config["calibration"].get("guidance_scale", 1.0)),
                "selected_schedule": json.dumps([float(item) for item in goes_schedule]),
                "edge_objective": float(edge_objective),
                "tiny_tangent_fallback_fraction": float(tiny_tangent_fallback_fraction),
                "notes": "No held-out sample underperformed the uniform_in_u baseline by final latent MSE.",
            }
        )
    write_csv(rows, run_dir / "failure_cases.csv")


def _save_edge_table(edge_table: EdgeCostTable, run_dir: Path) -> None:
    np.savez_compressed(
        run_dir / "edge_costs.npz",
        candidate_grid=edge_table.candidate_grid,
        edge_costs=edge_table.edge_costs,
        per_sample_costs=edge_table.per_sample_costs,
        fallback_counts=edge_table.fallback_counts,
    )


def _path_from_indices(edge_costs: np.ndarray, indices: list[int]) -> MinimaxPath:
    selected = [float(edge_costs[start, end]) for start, end in zip(indices[:-1], indices[1:])]
    objective = float(np.max(selected)) if selected else 0.0
    total_cost = float(np.sum(selected)) if selected else 0.0
    return MinimaxPath(indices=indices, objective=objective, total_cost=total_cost, edge_costs=selected)


def _search_once(config: dict[str, Any], run_dir: Path | None = None, *, command: str = "search") -> dict[str, Any]:
    started = time.time()
    deterministic_seeds = _set_deterministic_seeds(config)
    model, coordinate, solver = _prepare_common(config)
    if run_dir is None:
        run_dir = make_run_dir(config, command)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "paper_tables").mkdir(exist_ok=True)

    oracle_result = build_or_load_oracle(config, split_section="calibration")
    heldout_result = build_or_load_oracle(config, split_section="heldout")
    metric = make_metric(config["metric"], oracle=oracle_result.oracle, coordinate=coordinate)
    candidate_grid = coordinate.candidate_grid(
        int(config["candidate_grid"]["size"]),
        str(config["candidate_grid"].get("type", "uniform_in_u")),
    )
    edge_started = time.time()
    edge_table = evaluate_edge_table(
        solver,
        oracle_result.oracle,
        candidate_grid,
        metric,
        rho=float(config["mixed_defect"]["rho"]),
        aggregation=config["aggregation"],
        eps=float(config["mixed_defect"].get("eps", 1.0e-12)),
        fallback_full_residual_on_tiny_tangent=bool(
            config["mixed_defect"].get("fallback_full_residual_on_tiny_tangent", True)
        ),
    )
    edge_seconds = time.time() - edge_started
    dp_started = time.time()
    path = solve_minimax_schedule(
        edge_table.edge_costs,
        int(config["solver"]["target_nfe"]),
        tie_break_sum_cost=bool(config["optimizer"].get("tie_break_sum_cost", True)),
        tie_tolerance=float(config["optimizer"].get("tie_tolerance", 1.0e-12)),
    )
    dp_seconds = time.time() - dp_started
    u_schedule = candidate_grid[path.indices]

    refinement_history: list[dict[str, float]] = []
    pre_refinement_path: MinimaxPath | None = None
    if bool(config.get("replay_refinement", {}).get("enabled", False)):
        pre_refinement_path = path
        refinement = refine_schedule_blackbox(
            solver,
            oracle_result.oracle,
            u_schedule,
            candidate_grid,
            metric,
            rho=float(config["mixed_defect"]["rho"]),
            aggregation=config["aggregation"],
            rounds=int(config["replay_refinement"].get("rounds", 3)),
            local_window=int(config["replay_refinement"].get("local_window", 8)),
            lambda_final=float(config["replay_refinement"].get("lambda_final", 0.0)),
            mu_smooth=float(config["replay_refinement"].get("mu_smooth", 0.0)),
            fallback_full_residual_on_tiny_tangent=bool(
                config["mixed_defect"].get("fallback_full_residual_on_tiny_tangent", True)
            ),
        )
        u_schedule = refinement.u_schedule
        refinement_history = refinement.history
        path = _path_from_indices(
            edge_table.edge_costs,
            [int(np.argmin(np.abs(candidate_grid - item))) for item in u_schedule],
        )

    payload = schedule_payload(
        solver_name=solver.name,
        target_nfe=int(config["solver"]["target_nfe"]),
        coordinate=coordinate,
        u_schedule=u_schedule,
        rho=float(config["mixed_defect"]["rho"]),
        metric_metadata=metric.metadata(),
        aggregation_config=config["aggregation"],
        oracle_cache_key=oracle_result.cache_key,
        path=path,
    )
    save_schedule_outputs(
        run_dir,
        payload=payload,
        selected_indices=path.indices,
        selected_edge_costs=path.edge_costs,
    )
    _save_edge_table(edge_table, run_dir)
    config_resolved_path = write_config(config, run_dir / "config.resolved.yaml")
    dump_json(oracle_result.oracle.metadata, run_dir / "oracle_metadata.json")

    calibration_rows, heldout_rows = _evaluate_common_schedules(
        config=config,
        solver=solver,
        metric=metric,
        calibration_oracle=oracle_result.oracle,
        heldout_oracle=heldout_result.oracle,
        goes_schedule=u_schedule,
        goes_hash=payload["schedule_hash"],
    )
    uniform_schedule = np.linspace(coordinate.u_min, coordinate.u_max, int(config["solver"]["target_nfe"]) + 1)
    _write_failure_cases(
        run_dir=run_dir,
        config=config,
        solver=solver,
        heldout_oracle=heldout_result.oracle,
        goes_schedule=u_schedule,
        uniform_schedule=uniform_schedule,
        edge_objective=path.objective,
        tiny_tangent_fallback_fraction=edge_table.fallback_fraction,
    )
    metric_fields = [
        "split",
        "schedule",
        "solver",
        "nfe",
        "num_samples",
        "final_latent_mse",
        "final_latent_mse_bootstrap_se",
        "final_latent_mse_ci95_low",
        "final_latent_mse_ci95_high",
        "replay_loss",
        "max_endpoint_cost",
        "mean_endpoint_mse",
        "fallback_fraction",
        "guidance_scale",
        "split_hash",
        "initial_noise_hash",
        "schedule_hash",
    ]
    write_csv(calibration_rows, run_dir / "calibration_metrics.csv", metric_fields)
    write_csv(heldout_rows, run_dir / "heldout_metrics.csv", metric_fields)
    write_csv(heldout_rows, run_dir / "paper_tables" / "main_results.csv", metric_fields)
    write_csv(
        [
            {
                "ablation": "default",
                "solver": solver.name,
                "nfe": int(config["solver"]["target_nfe"]),
                "rho": float(config["mixed_defect"]["rho"]),
                "metric": metric.metadata()["name"],
                "aggregation": aggregation_label(config["aggregation"]),
                "edge_objective": path.objective,
                "heldout_final_latent_mse": next(
                    row["final_latent_mse"] for row in heldout_rows if row["schedule"] == "GOES"
                ),
            }
        ],
        run_dir / "paper_tables" / "ablations.csv",
    )
    write_csv(
        [
            {
                "oracle_cache_key": oracle_result.cache_key,
                "solvers": solver.name,
                "shared_oracle_builds": 1,
                "separate_oracle_builds": 1,
                "oracle_loaded_from_cache": oracle_result.loaded_from_cache,
                "oracle_build_or_load_seconds": oracle_result.elapsed_seconds,
                "edge_evaluation_seconds": edge_seconds,
                "search_dp_seconds": dp_seconds,
            }
        ],
        run_dir / "paper_tables" / "oracle_reuse_cost.csv",
    )

    if bool(config.get("output", {}).get("save_plots", True)):
        plots_written = maybe_write_plots(
            run_dir,
            candidate_grid=candidate_grid,
            edge_costs=edge_table.edge_costs,
            u_schedule=u_schedule,
            selected_edge_costs=np.asarray(path.edge_costs, dtype=np.float64),
        )
    else:
        plots_written = []
    run_metadata = {
        "command": command,
        "run_dir": str(run_dir),
        "config_resolved_path": str(config_resolved_path),
        "runtime": runtime_metadata(),
        "deterministic_seeds": deterministic_seeds,
        "model_identifier": model.identifier,
        "model": _model_run_metadata(config, model.identifier),
        "solver": {"name": solver.name, **solver.compatibility_metadata()},
        "oracle_cache_key": oracle_result.cache_key,
        "oracle_cache_path": str(oracle_result.cache_path),
        "oracle_build_or_load_seconds": oracle_result.elapsed_seconds,
        "oracle_loaded_from_cache": oracle_result.loaded_from_cache,
        "heldout_oracle_cache_key": heldout_result.cache_key,
        "heldout_oracle_build_or_load_seconds": heldout_result.elapsed_seconds,
        "heldout_oracle_loaded_from_cache": heldout_result.loaded_from_cache,
        "edge_evaluation_seconds": edge_seconds,
        "dp_seconds": dp_seconds,
        "total_seconds": time.time() - started,
        "edge_table": edge_table.metadata,
        "mixed_defect": {
            "rho": float(config["mixed_defect"]["rho"]),
            "tiny_tangent_fallback_fraction": edge_table.fallback_fraction,
        },
        "baselines": {
            "run": ["uniform_in_u"],
            "skipped": [
                {
                    "name": "AYS",
                    "reason": "No trusted explicit AYS schedule is defined for the toy_flow validation model.",
                },
                {
                    "name": "image_metrics",
                    "reason": "Toy CPU validation produces latent MSE and replay losses, not images.",
                },
            ],
        },
        "calibration_split_hash": oracle_result.oracle.metadata.get("condition_split_hash"),
        "heldout_split_hash": heldout_result.oracle.metadata.get("condition_split_hash"),
        "calibration_initial_noise_hash": oracle_result.oracle.metadata.get("initial_noise_hash"),
        "heldout_initial_noise_hash": heldout_result.oracle.metadata.get("initial_noise_hash"),
        "calibration_noise_seed_hash": oracle_result.oracle.metadata.get("noise_seed_hash"),
        "heldout_noise_seed_hash": heldout_result.oracle.metadata.get("noise_seed_hash"),
        "calibration_noise_seeds": oracle_result.oracle.metadata.get("noise_seeds"),
        "heldout_noise_seeds": heldout_result.oracle.metadata.get("noise_seeds"),
        "schedule_hash": payload["schedule_hash"],
        "plots_written": plots_written,
        "refinement_history": refinement_history,
        "pre_refinement_edge_objective": None if pre_refinement_path is None else pre_refinement_path.objective,
        "pre_refinement_schedule_indices": None if pre_refinement_path is None else pre_refinement_path.indices,
    }
    dump_json(run_metadata, run_dir / "run_metadata.json")
    return {
        "run_dir": str(run_dir),
        "schedule": payload,
        "run_metadata": run_metadata,
        "heldout_rows": heldout_rows,
        "calibration_rows": calibration_rows,
    }


def build_oracle_command(config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    deterministic_seeds = _set_deterministic_seeds(config)
    run_dir = make_run_dir(config, "build_oracle")
    result = build_or_load_oracle(config, split_section="calibration")
    config_resolved_path = write_config(config, run_dir / "config.resolved.yaml")
    oracle_metadata_path = dump_json(result.oracle.metadata, run_dir / "oracle_metadata.json")
    dump_json(
        {
            "command": "build-oracle",
            "run_dir": str(run_dir),
            "config_resolved_path": str(config_resolved_path),
            "model": _model_run_metadata(config, result.oracle.metadata.get("model_identifier")),
            "oracle_cache_key": result.cache_key,
            "oracle_cache_path": str(result.cache_path),
            "oracle_metadata_path": str(oracle_metadata_path),
            "oracle_cache_metadata_path": str(result.metadata_path),
            "loaded_from_cache": result.loaded_from_cache,
            "oracle_build_or_load_seconds": result.elapsed_seconds,
            "total_seconds": time.time() - started,
            "calibration_split_hash": result.oracle.metadata.get("condition_split_hash"),
            "calibration_initial_noise_hash": result.oracle.metadata.get("initial_noise_hash"),
            "calibration_noise_seed_hash": result.oracle.metadata.get("noise_seed_hash"),
            "calibration_noise_seeds": result.oracle.metadata.get("noise_seeds"),
            "deterministic_seeds": deterministic_seeds,
            "runtime": runtime_metadata(),
        },
        run_dir / "run_metadata.json",
    )
    return {"run_dir": str(run_dir), "oracle_cache_key": result.cache_key}


def search_schedule_command(config: dict[str, Any]) -> dict[str, Any]:
    return _search_once(config, command="search_schedule")


def evaluate_command(config: dict[str, Any], schedule_path: str | Path) -> dict[str, Any]:
    started = time.time()
    deterministic_seeds = _set_deterministic_seeds(config)
    run_dir = make_run_dir(config, "evaluate")
    with repo_path(schedule_path).open("r", encoding="utf-8") as handle:
        schedule = json.load(handle)
    _, _, solver = _prepare_common(config)
    coordinate = make_coordinate_adapter(config["coordinate"])
    schedule_verification = verify_schedule_payload(schedule)
    expected_nfe = int(config["solver"]["target_nfe"])
    if int(schedule_verification["target_nfe"]) != expected_nfe:
        raise ValueError(
            f"Schedule target_nfe {schedule_verification['target_nfe']} does not match config solver.target_nfe {expected_nfe}."
        )
    schedule_solver = str(schedule.get("solver", ""))
    if schedule_solver and schedule_solver != solver.name:
        raise ValueError(f"Schedule solver {schedule_solver!r} does not match config solver {solver.name!r}.")
    schedule_coordinate = str(schedule.get("coordinate", ""))
    if schedule_coordinate and schedule_coordinate != coordinate.name:
        raise ValueError(
            f"Schedule coordinate {schedule_coordinate!r} does not match config coordinate {coordinate.name!r}."
        )
    heldout = build_or_load_oracle(config, split_section="heldout")
    calibration = build_or_load_oracle(config, split_section="calibration")
    metric = make_metric(config["metric"], oracle=calibration.oracle, coordinate=coordinate)
    u_schedule = np.asarray(schedule["u_schedule"], dtype=np.float64)
    metrics = evaluate_replay_metrics(
        solver,
        heldout.oracle,
        u_schedule,
        metric,
        rho=float(schedule.get("rho", config["mixed_defect"]["rho"])),
        aggregation=config["aggregation"],
        eps=float(config["mixed_defect"].get("eps", 1.0e-12)),
        fallback_full_residual_on_tiny_tangent=bool(
            config["mixed_defect"].get("fallback_full_residual_on_tiny_tangent", True)
        ),
    )
    row = _metrics_row(
        split="heldout",
        schedule_name=schedule.get("method", "GOES"),
        solver_name=solver.name,
        target_nfe=int(config["solver"]["target_nfe"]),
        metrics=metrics,
        oracle=heldout.oracle,
        schedule_hash=schedule.get("schedule_hash", stable_hash(u_schedule.tolist())),
        guidance_scale=float(config["calibration"].get("guidance_scale", 1.0)),
    )
    fields = list(row.keys())
    write_csv([row], run_dir / "heldout_metrics.csv", fields)
    write_csv([row], run_dir / "paper_tables" / "main_results.csv", fields)
    config_resolved_path = write_config(config, run_dir / "config.resolved.yaml")
    oracle_metadata_path = dump_json(
        {
            "calibration": calibration.oracle.metadata,
            "heldout": heldout.oracle.metadata,
        },
        run_dir / "oracle_metadata.json",
    )
    dump_json(
        {
            "command": "evaluate",
            "run_dir": str(run_dir),
            "schedule_path": str(repo_path(schedule_path)),
            "schedule_verification": schedule_verification,
            "config_resolved_path": str(config_resolved_path),
            "runtime": runtime_metadata(),
            "deterministic_seeds": deterministic_seeds,
            "model": _model_run_metadata(config, calibration.oracle.metadata.get("model_identifier")),
            "calibration_oracle_cache_key": calibration.cache_key,
            "heldout_oracle_cache_key": heldout.cache_key,
            "oracle_metadata_path": str(oracle_metadata_path),
            "calibration_split_hash": calibration.oracle.metadata.get("condition_split_hash"),
            "heldout_split_hash": heldout.oracle.metadata.get("condition_split_hash"),
            "calibration_initial_noise_hash": calibration.oracle.metadata.get("initial_noise_hash"),
            "heldout_initial_noise_hash": heldout.oracle.metadata.get("initial_noise_hash"),
            "calibration_noise_seed_hash": calibration.oracle.metadata.get("noise_seed_hash"),
            "heldout_noise_seed_hash": heldout.oracle.metadata.get("noise_seed_hash"),
            "calibration_noise_seeds": calibration.oracle.metadata.get("noise_seeds"),
            "heldout_noise_seeds": heldout.oracle.metadata.get("noise_seeds"),
            "schedule_hash": row["schedule_hash"],
            "total_seconds": time.time() - started,
        },
        run_dir / "run_metadata.json",
    )
    return {"run_dir": str(run_dir), "heldout_metrics": row}


def _sweep_values(raw: str | None, defaults: list[Any], caster: Any) -> list[Any]:
    if raw is None:
        return defaults
    return [caster(item.strip()) for item in raw.split(",") if item.strip()]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(data.shape[0], dtype=np.float64)
    sorted_data = data[order]
    start = 0
    while start < sorted_data.shape[0]:
        stop = start + 1
        while stop < sorted_data.shape[0] and sorted_data[stop] == sorted_data[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1)
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | str:
    left_ranks = _average_ranks(np.asarray(left, dtype=np.float64))
    right_ranks = _average_ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return ""
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def ablate_rho_command(config: dict[str, Any], values: list[float]) -> dict[str, Any]:
    deterministic_seeds = _set_deterministic_seeds(config)
    run_dir = make_run_dir(config, "ablate_rho")
    rows: list[dict[str, Any]] = []
    for rho in values:
        cfg = copy.deepcopy(config)
        cfg["mixed_defect"]["rho"] = float(rho)
        subdir = run_dir / f"rho_{rho:g}"
        result = _search_once(cfg, subdir, command="ablate_rho")
        goes_row = next(row for row in result["heldout_rows"] if row["schedule"] == "GOES")
        calibration_row = next(row for row in result["calibration_rows"] if row["schedule"] == "GOES")
        rows.append(
            {
                "ablation": "rho",
                "rho": float(rho),
                "metric": cfg["metric"]["name"],
                "edge_objective": result["schedule"]["edge_objective"],
                "calibration_final_latent_mse": calibration_row["final_latent_mse"],
                "heldout_final_latent_mse": goes_row["final_latent_mse"],
                "heldout_generalization_gap": float(goes_row["final_latent_mse"])
                - float(calibration_row["final_latent_mse"]),
                "total_seconds": result["run_metadata"]["total_seconds"],
                "schedule_hash": result["schedule"]["schedule_hash"],
                "run_dir": result["run_dir"],
            }
        )
    write_csv(rows, run_dir / "paper_tables" / "ablations.csv")
    _write_top_level_run_metadata(
        config,
        run_dir,
        command="ablate-rho",
        deterministic_seeds=deterministic_seeds,
        extra={"values": values},
    )
    return {"run_dir": str(run_dir), "rows": rows}


def ablate_metric_command(config: dict[str, Any], values: list[str]) -> dict[str, Any]:
    deterministic_seeds = _set_deterministic_seeds(config)
    run_dir = make_run_dir(config, "ablate_metric")
    rows: list[dict[str, Any]] = []
    for metric_name in values:
        cfg = copy.deepcopy(config)
        cfg["metric"]["name"] = metric_name
        subdir = run_dir / f"metric_{metric_name}"
        result = _search_once(cfg, subdir, command="ablate_metric")
        goes_row = next(row for row in result["heldout_rows"] if row["schedule"] == "GOES")
        calibration_row = next(row for row in result["calibration_rows"] if row["schedule"] == "GOES")
        rows.append(
            {
                "ablation": "metric",
                "rho": cfg["mixed_defect"]["rho"],
                "metric": metric_name,
                "edge_objective": result["schedule"]["edge_objective"],
                "calibration_final_latent_mse": calibration_row["final_latent_mse"],
                "heldout_final_latent_mse": goes_row["final_latent_mse"],
                "heldout_generalization_gap": float(goes_row["final_latent_mse"])
                - float(calibration_row["final_latent_mse"]),
                "total_seconds": result["run_metadata"]["total_seconds"],
                "schedule_hash": result["schedule"]["schedule_hash"],
                "run_dir": result["run_dir"],
            }
        )
    write_csv(rows, run_dir / "paper_tables" / "ablations.csv")
    _write_top_level_run_metadata(
        config,
        run_dir,
        command="ablate-metric",
        deterministic_seeds=deterministic_seeds,
        extra={"values": values},
    )
    return {"run_dir": str(run_dir), "rows": rows}


def oracle_convergence_command(config: dict[str, Any], values: list[int]) -> dict[str, Any]:
    deterministic_seeds = _set_deterministic_seeds(config)
    run_dir = make_run_dir(config, "oracle_convergence")
    rows: list[dict[str, Any]] = []
    schedules: dict[int, np.ndarray] = {}
    edge_costs: dict[int, np.ndarray] = {}
    for ref_nfe in values:
        cfg = copy.deepcopy(config)
        cfg["oracle"]["ref_nfe"] = int(ref_nfe)
        cfg["oracle"]["reuse"] = True
        subdir = run_dir / f"ref_nfe_{ref_nfe}"
        result = _search_once(cfg, subdir, command="oracle_convergence")
        schedule = np.asarray(result["schedule"]["u_schedule"], dtype=np.float64)
        schedules[int(ref_nfe)] = schedule
        with np.load(subdir / "edge_costs.npz") as payload:
            edge_costs[int(ref_nfe)] = payload["edge_costs"]
        goes_row = next(row for row in result["heldout_rows"] if row["schedule"] == "GOES")
        rows.append(
            {
                "ref_nfe": int(ref_nfe),
                "oracle_cache_key": result["schedule"]["oracle_cache_key"],
                "oracle_loaded_from_cache": result["run_metadata"]["oracle_loaded_from_cache"],
                "oracle_build_or_load_seconds": result["run_metadata"]["oracle_build_or_load_seconds"],
                "edge_objective": result["schedule"]["edge_objective"],
                "heldout_final_latent_mse": goes_row["final_latent_mse"],
                "final_latent_oracle_mse": goes_row["final_latent_mse"],
                "schedule_hash": result["schedule"]["schedule_hash"],
                "run_dir": result["run_dir"],
            }
        )
    highest = max(values)
    reference_schedule = schedules[highest]
    reference_edges = edge_costs[highest]
    for row in rows:
        ref_nfe = int(row["ref_nfe"])
        row["schedule_l1_to_highest_ref"] = float(np.mean(np.abs(schedules[ref_nfe] - reference_schedule)))
        finite = np.isfinite(edge_costs[ref_nfe]) & np.isfinite(reference_edges)
        if np.count_nonzero(finite) > 1:
            row["edge_cost_correlation_to_highest_ref"] = float(
                np.corrcoef(edge_costs[ref_nfe][finite], reference_edges[finite])[0, 1]
            )
            row["edge_cost_rank_correlation_to_highest_ref"] = _rank_correlation(
                edge_costs[ref_nfe][finite],
                reference_edges[finite],
            )
        else:
            row["edge_cost_correlation_to_highest_ref"] = ""
            row["edge_cost_rank_correlation_to_highest_ref"] = ""
    write_csv(rows, run_dir / "paper_tables" / "oracle_convergence.csv")
    _write_top_level_run_metadata(
        config,
        run_dir,
        command="oracle-convergence",
        deterministic_seeds=deterministic_seeds,
        extra={"values": values},
    )
    return {"run_dir": str(run_dir), "rows": rows}


def nfe_sweep_command(config: dict[str, Any], values: list[int]) -> dict[str, Any]:
    deterministic_seeds = _set_deterministic_seeds(config)
    run_dir = make_run_dir(config, "nfe_sweep")
    rows: list[dict[str, Any]] = []
    for nfe in values:
        cfg = copy.deepcopy(config)
        cfg["solver"]["target_nfe"] = int(nfe)
        cfg["candidate_grid"]["size"] = max(int(cfg["candidate_grid"]["size"]), int(nfe))
        subdir = run_dir / f"nfe_{nfe}"
        result = _search_once(cfg, subdir, command="nfe_sweep")
        for row in result["heldout_rows"]:
            rows.append(
                {
                    **row,
                    "total_seconds": result["run_metadata"]["total_seconds"],
                    "edge_evaluation_seconds": result["run_metadata"]["edge_evaluation_seconds"],
                    "search_dp_seconds": result["run_metadata"]["dp_seconds"],
                    "run_dir": result["run_dir"],
                }
            )
    write_csv(rows, run_dir / "paper_tables" / "nfe_quality_curve.csv")
    plots_written = (
        maybe_write_nfe_quality_curve(run_dir, rows)
        if bool(config.get("output", {}).get("save_plots", True))
        else []
    )
    _write_top_level_run_metadata(
        config,
        run_dir,
        command="nfe-sweep",
        deterministic_seeds=deterministic_seeds,
        extra={"values": values, "plots_written": plots_written},
    )
    return {"run_dir": str(run_dir), "rows": rows}


def calibration_size_ablation_command(config: dict[str, Any], values: list[int]) -> dict[str, Any]:
    deterministic_seeds = _set_deterministic_seeds(config)
    run_dir = make_run_dir(config, "calibration_size")
    rows: list[dict[str, Any]] = []
    schedules: dict[int, np.ndarray] = {}
    for size in values:
        cfg = copy.deepcopy(config)
        cfg["calibration"]["num_samples"] = int(size)
        subdir = run_dir / f"calibration_k_{size}"
        result = _search_once(cfg, subdir, command="calibration_size")
        goes_row = next(row for row in result["heldout_rows"] if row["schedule"] == "GOES")
        calibration_row = next(row for row in result["calibration_rows"] if row["schedule"] == "GOES")
        schedules[int(size)] = np.asarray(result["schedule"]["u_schedule"], dtype=np.float64)
        rows.append(
            {
                "ablation": "calibration_size",
                "calibration_samples": int(size),
                "solver": cfg["solver"]["name"],
                "nfe": int(cfg["solver"]["target_nfe"]),
                "edge_objective": result["schedule"]["edge_objective"],
                "calibration_final_latent_mse": calibration_row["final_latent_mse"],
                "heldout_final_latent_mse": goes_row["final_latent_mse"],
                "heldout_generalization_gap": float(goes_row["final_latent_mse"])
                - float(calibration_row["final_latent_mse"]),
                "total_seconds": result["run_metadata"]["total_seconds"],
                "schedule_hash": result["schedule"]["schedule_hash"],
                "run_dir": result["run_dir"],
            }
        )
    largest = max(values)
    reference_schedule = schedules[int(largest)]
    for row in rows:
        size = int(row["calibration_samples"])
        row["schedule_l1_to_largest_calibration_size"] = float(np.mean(np.abs(schedules[size] - reference_schedule)))
    write_csv(rows, run_dir / "paper_tables" / "calibration_size_ablation.csv")
    _write_top_level_run_metadata(
        config,
        run_dir,
        command="calibration-size-ablation",
        deterministic_seeds=deterministic_seeds,
        extra={"values": values},
    )
    return {"run_dir": str(run_dir), "rows": rows}


def candidate_grid_ablation_command(config: dict[str, Any], values: list[int]) -> dict[str, Any]:
    deterministic_seeds = _set_deterministic_seeds(config)
    run_dir = make_run_dir(config, "candidate_grid")
    rows: list[dict[str, Any]] = []
    schedules: dict[int, np.ndarray] = {}
    for size in values:
        cfg = copy.deepcopy(config)
        cfg["candidate_grid"]["size"] = max(int(size), int(cfg["solver"]["target_nfe"]))
        subdir = run_dir / f"candidate_grid_{size}"
        result = _search_once(cfg, subdir, command="candidate_grid")
        goes_row = next(row for row in result["heldout_rows"] if row["schedule"] == "GOES")
        actual_size = int(cfg["candidate_grid"]["size"])
        schedules[actual_size] = np.asarray(result["schedule"]["u_schedule"], dtype=np.float64)
        rows.append(
            {
                "ablation": "candidate_grid",
                "candidate_grid_size": actual_size,
                "solver": cfg["solver"]["name"],
                "nfe": int(cfg["solver"]["target_nfe"]),
                "edge_objective": result["schedule"]["edge_objective"],
                "heldout_final_latent_mse": goes_row["final_latent_mse"],
                "edge_evaluation_seconds": result["run_metadata"]["edge_evaluation_seconds"],
                "search_dp_seconds": result["run_metadata"]["dp_seconds"],
                "total_seconds": result["run_metadata"]["total_seconds"],
                "schedule_hash": result["schedule"]["schedule_hash"],
                "run_dir": result["run_dir"],
            }
        )
    largest = max(schedules)
    reference_schedule = schedules[largest]
    for row in rows:
        size = int(row["candidate_grid_size"])
        row["schedule_l1_to_largest_candidate_grid"] = float(np.mean(np.abs(schedules[size] - reference_schedule)))
    write_csv(rows, run_dir / "paper_tables" / "candidate_grid_ablation.csv")
    _write_top_level_run_metadata(
        config,
        run_dir,
        command="candidate-grid-ablation",
        deterministic_seeds=deterministic_seeds,
        extra={"values": values},
    )
    return {"run_dir": str(run_dir), "rows": rows}


def cross_solver_reuse_command(config: dict[str, Any], solvers: list[str]) -> dict[str, Any]:
    deterministic_seeds = _set_deterministic_seeds(config)
    run_dir = make_run_dir(config, "cross_solver_reuse")
    oracle_result = build_or_load_oracle(config, split_section="calibration")
    rows: list[dict[str, Any]] = []
    skipped_solvers: list[dict[str, str]] = []
    for solver_name in solvers:
        cfg = copy.deepcopy(config)
        cfg["solver"]["name"] = solver_name
        subdir = run_dir / f"solver_{solver_name}"
        try:
            result = _search_once(cfg, subdir, command="cross_solver_reuse")
        except ValueError as exc:
            if "Unsupported toy solver" not in str(exc):
                raise
            skip_reason = str(exc)
            skipped_solvers.append({"solver": solver_name, "reason": skip_reason})
            rows.append(
                {
                    "solver": solver_name,
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "oracle_cache_key": "",
                    "shared_oracle_cache_key": oracle_result.cache_key,
                    "oracle_reused": "",
                    "shared_oracle_builds": 1,
                    "separate_oracle_builds": len(solvers),
                    "shared_oracle_build_or_load_seconds": oracle_result.elapsed_seconds,
                    "solver_oracle_build_or_load_seconds": "",
                    "edge_evaluation_seconds": "",
                    "search_dp_seconds": "",
                    "total_solver_search_seconds": "",
                    "edge_objective": "",
                    "heldout_final_latent_mse": "",
                    "run_dir": str(subdir),
                }
            )
            continue
        goes_row = next(row for row in result["heldout_rows"] if row["schedule"] == "GOES")
        run_metadata = result["run_metadata"]
        rows.append(
            {
                "solver": solver_name,
                "skipped": False,
                "skip_reason": "",
                "oracle_cache_key": result["schedule"]["oracle_cache_key"],
                "shared_oracle_cache_key": oracle_result.cache_key,
                "oracle_reused": result["schedule"]["oracle_cache_key"] == oracle_result.cache_key,
                "shared_oracle_builds": 1,
                "separate_oracle_builds": len(solvers),
                "shared_oracle_build_or_load_seconds": oracle_result.elapsed_seconds,
                "solver_oracle_build_or_load_seconds": run_metadata["oracle_build_or_load_seconds"],
                "edge_evaluation_seconds": run_metadata["edge_evaluation_seconds"],
                "search_dp_seconds": run_metadata["dp_seconds"],
                "total_solver_search_seconds": run_metadata["total_seconds"],
                "edge_objective": result["schedule"]["edge_objective"],
                "heldout_final_latent_mse": goes_row["final_latent_mse"],
                "run_dir": result["run_dir"],
            }
        )
    runnable_rows = [row for row in rows if not bool(row.get("skipped", False))]
    runnable_count = len(runnable_rows)
    shared_oracle_seconds = float(oracle_result.elapsed_seconds)
    estimated_separate_oracle_seconds = shared_oracle_seconds * runnable_count
    shared_oracle_amortized_seconds = shared_oracle_seconds / runnable_count if runnable_count else 0.0
    for row in rows:
        row["runnable_solver_count"] = runnable_count
        row["estimated_shared_oracle_build_or_load_seconds"] = shared_oracle_seconds
        row["estimated_separate_oracle_build_or_load_seconds"] = estimated_separate_oracle_seconds
        if bool(row.get("skipped", False)):
            row["shared_oracle_amortized_build_or_load_seconds"] = ""
            row["estimated_shared_total_solver_seconds"] = ""
            row["estimated_separate_total_solver_seconds"] = ""
            continue
        edge_seconds = float(row["edge_evaluation_seconds"])
        dp_seconds = float(row["search_dp_seconds"])
        row["shared_oracle_amortized_build_or_load_seconds"] = shared_oracle_amortized_seconds
        row["estimated_shared_total_solver_seconds"] = shared_oracle_amortized_seconds + edge_seconds + dp_seconds
        row["estimated_separate_total_solver_seconds"] = shared_oracle_seconds + edge_seconds + dp_seconds
    write_csv(rows, run_dir / "paper_tables" / "oracle_reuse_cost.csv")
    _write_top_level_run_metadata(
        config,
        run_dir,
        command="cross-solver-reuse",
        deterministic_seeds=deterministic_seeds,
        extra={
            "solvers": solvers,
            "shared_oracle_cache_key": oracle_result.cache_key,
            "skipped_solvers": skipped_solvers,
        },
    )
    return {"run_dir": str(run_dir), "rows": rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GOES experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_arg(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", required=True, help="Path to a GOES YAML/JSON config.")

    for name in [
        "build-oracle",
        "search-schedule",
        "ablate-rho",
        "ablate-metric",
        "oracle-convergence",
        "nfe-sweep",
        "calibration-size-ablation",
        "candidate-grid-ablation",
        "cross-solver-reuse",
    ]:
        sub = subparsers.add_parser(name)
        add_config_arg(sub)
        if name == "ablate-rho":
            sub.add_argument("--values", default=None, help="Comma-separated rho values.")
        elif name == "ablate-metric":
            sub.add_argument("--values", default=None, help="Comma-separated metric names.")
        elif name == "oracle-convergence":
            sub.add_argument("--values", default=None, help="Comma-separated ref_nfe values.")
        elif name == "nfe-sweep":
            sub.add_argument("--values", default=None, help="Comma-separated target NFE values.")
        elif name == "calibration-size-ablation":
            sub.add_argument("--values", default=None, help="Comma-separated calibration sample counts.")
        elif name == "candidate-grid-ablation":
            sub.add_argument("--values", default=None, help="Comma-separated candidate grid sizes.")
        elif name == "cross-solver-reuse":
            sub.add_argument("--solvers", default=None, help="Comma-separated solver names.")

    evaluate = subparsers.add_parser("evaluate")
    add_config_arg(evaluate)
    evaluate.add_argument("--schedule", required=True, help="Path to schedule.json.")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "build-oracle":
        result = build_oracle_command(config)
    elif args.command == "search-schedule":
        result = search_schedule_command(config)
    elif args.command == "evaluate":
        result = evaluate_command(config, args.schedule)
    elif args.command == "ablate-rho":
        values = _sweep_values(args.values, [0.0, 0.05, 0.1, 0.2, 0.5, 1.0], float)
        result = ablate_rho_command(config, values)
    elif args.command == "ablate-metric":
        values = _sweep_values(args.values, ["identity", "edm_scalar", "channel_whitened"], str)
        result = ablate_metric_command(config, values)
    elif args.command == "oracle-convergence":
        values = _sweep_values(args.values, [100, 200, 500, 1000], int)
        result = oracle_convergence_command(config, values)
    elif args.command == "nfe-sweep":
        values = _sweep_values(args.values, [4, 5, 6, 8, 10, 12, 15, 20, 30, 50], int)
        result = nfe_sweep_command(config, values)
    elif args.command == "calibration-size-ablation":
        values = _sweep_values(args.values, [4, 8, 16, 32, 64, 128], int)
        result = calibration_size_ablation_command(config, values)
    elif args.command == "candidate-grid-ablation":
        values = _sweep_values(args.values, [64, 128, 256, 512, 1024], int)
        result = candidate_grid_ablation_command(config, values)
    elif args.command == "cross-solver-reuse":
        values = _sweep_values(args.solvers, ["euler", "heun", "midpoint"], str)
        result = cross_solver_reuse_command(config, values)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

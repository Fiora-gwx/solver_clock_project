from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG: dict[str, Any] = {
    "method": "gpde",
    "model": {
        "name": "toy_flow",
        "checkpoint": None,
        "dtype": "float64",
        "device": "cpu",
        "state_shape": [2],
    },
    "coordinate": {
        "name": "t",
        "direction": "increasing",
        "u_min": 0.0,
        "u_max": 1.0,
    },
    "oracle": {
        "ref_integrator": "rk4",
        "ref_nfe": 200,
        "ref_grid_size": 257,
        "interpolation": "linear",
        "cache_dir": "./outputs/goes/oracle_cache",
        "reuse": True,
    },
    "calibration": {
        "num_samples": 8,
        "seed": 123,
        "prompt_file": None,
        "split": "calibration",
        "guidance_scale": 1.0,
    },
    "heldout": {
        "num_samples": 8,
        "seed": 456,
        "split": "heldout",
    },
    "candidate_grid": {
        "size": 64,
        "type": "uniform_in_u",
    },
    "probe_grid": {
        "size": 64,
        "type": "uniform_in_u",
    },
    "probe_steps": {
        "multipliers": [1.0, 2.0, 4.0],
        "absolute": None,
    },
    "solver": {
        "name": "euler",
        "target_nfe": 10,
        "mode": "one_step",
    },
    "metric": {
        "name": "identity",
        "sigma_data": 0.5,
        "eps": 1.0e-12,
        "min_weight": 1.0e-4,
        "max_weight": 1.0e4,
    },
    "mixed_defect": {
        "rho": 0.1,
        "eps": 1.0e-12,
        "fallback_full_residual_on_tiny_tangent": True,
    },
    "aggregation": {
        "name": "cvar",
        "trim_ratio": 0.10,
        "alpha": 0.80,
    },
    "optimizer": {
        "name": "monitor_inverse_cdf",
    },
    "q_estimation": {
        "mode": "global_fit",
        "fixed_q": None,
        "min_q": 0.25,
        "max_q": 12.0,
    },
    "monitor": {
        "epsilon_a": 1.0e-12,
        "smoothing_window": 3,
        "exponent": "q_root",
    },
    "admissible_grid": {
        "enabled": False,
        "size": None,
        "type": "uniform_in_u",
    },
    "replay_refinement": {
        "enabled": False,
        "rounds": 3,
        "local_window": 8,
        "smoothmax_alpha": 10.0,
        "lambda_final": 0.0,
        "mu_smooth": 0.0,
    },
    "output": {
        "root": "./outputs/goes",
        "save_probe_profile": True,
        "save_schedule": True,
        "save_edge_table": True,
        "save_images": False,
        "save_plots": True,
    },
}


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError("PyYAML is required for GOES YAML configs.") from exc
    return yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if path is not None:
        resolved = repo_path(path)
        if resolved.suffix.lower() == ".json":
            with resolved.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        else:
            with resolved.open("r", encoding="utf-8") as handle:
                loaded = _yaml_module().safe_load(handle)
        payload = loaded or {}
    config = deep_merge(DEFAULT_CONFIG, payload)
    if overrides:
        config = deep_merge(config, overrides)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if str(config.get("method")) not in {"gpde", "goes"}:
        raise ValueError("GPDE configs must set method: gpde.")
    state_shape = tuple(int(item) for item in config["model"].get("state_shape", [2]))
    if not state_shape or any(dim < 1 for dim in state_shape):
        raise ValueError("model.state_shape must contain positive dimensions.")
    rho = float(config["mixed_defect"].get("rho", 0.1))
    if not 0.0 <= rho <= 1.0:
        raise ValueError("mixed_defect.rho must be in [0, 1].")
    eps = float(config["mixed_defect"].get("eps", 1.0e-12))
    if eps <= 0.0:
        raise ValueError("mixed_defect.eps must be positive.")
    target_nfe = int(config["solver"]["target_nfe"])
    candidate_grid = config.get("candidate_grid", {})
    probe_grid = config.get("probe_grid") or candidate_grid
    probe_size = int(probe_grid.get("size", candidate_grid.get("size", 64)))
    if target_nfe < 1:
        raise ValueError("solver.target_nfe must be positive.")
    solver_name = str(config["solver"].get("name", "euler"))
    if solver_name not in {"euler", "heun", "heun2", "midpoint", "biased_euler", "empirical_noisy_euler"}:
        raise ValueError("solver.name is not supported by the CPU GOES runner.")
    solver_mode = str(config["solver"].get("mode", "one_step"))
    if solver_mode not in {"one_step", "blackbox_multistep"}:
        raise ValueError("solver.mode must be one_step or blackbox_multistep.")
    if probe_size < target_nfe + 1:
        raise ValueError("probe_grid.size must be at least solver.target_nfe + 1.")
    if int(candidate_grid.get("size", probe_size)) < target_nfe + 1:
        raise ValueError("candidate_grid.size must be at least solver.target_nfe + 1.")
    if str(probe_grid.get("type", "uniform_in_u")) != "uniform_in_u":
        raise ValueError("probe_grid.type must be 'uniform_in_u' for the CPU GPDE runner.")
    if str(candidate_grid.get("type", "uniform_in_u")) != "uniform_in_u":
        raise ValueError("candidate_grid.type must be 'uniform_in_u' for the CPU GPDE runner.")
    u_min = float(config["coordinate"]["u_min"])
    u_max = float(config["coordinate"]["u_max"])
    coordinate_name = str(config["coordinate"].get("name", "t"))
    if coordinate_name not in {"t", "identity", "u", "sigma", "log_sigma", "logsnr"}:
        raise ValueError("coordinate.name is not supported by GOES.")
    coordinate_direction = str(config["coordinate"].get("direction", "increasing"))
    if coordinate_direction not in {"increasing", "decreasing"}:
        raise ValueError("coordinate.direction must be increasing or decreasing.")
    if not u_min < u_max:
        raise ValueError("coordinate.u_min must be smaller than coordinate.u_max in unified coordinates.")
    if str(config["oracle"].get("ref_integrator", "rk4")) != "rk4":
        raise ValueError("oracle.ref_integrator must be 'rk4' for the current GOES implementation.")
    if int(config["oracle"]["ref_nfe"]) < 1:
        raise ValueError("oracle.ref_nfe must be positive.")
    if int(config["oracle"]["ref_grid_size"]) < 2:
        raise ValueError("oracle.ref_grid_size must be at least 2.")
    for split_name in ("calibration", "heldout"):
        if int(config[split_name]["num_samples"]) < 1:
            raise ValueError(f"{split_name}.num_samples must be positive.")
    guidance_scale = float(config["calibration"].get("guidance_scale", 1.0))
    if not math.isfinite(guidance_scale):
        raise ValueError("calibration.guidance_scale must be finite.")
    metric_name = str(config["metric"].get("name", "identity"))
    if metric_name not in {"identity", "edm_scalar", "channel_whitened"}:
        raise ValueError("metric.name must be one of identity, edm_scalar, channel_whitened.")
    metric_eps = float(config["metric"].get("eps", 1.0e-12))
    if metric_eps <= 0.0:
        raise ValueError("metric.eps must be positive.")
    if float(config["metric"].get("sigma_data", 0.5)) <= 0.0:
        raise ValueError("metric.sigma_data must be positive.")
    min_weight = float(config["metric"].get("min_weight", 1.0e-4))
    max_weight = float(config["metric"].get("max_weight", 1.0e4))
    if min_weight <= 0.0 or max_weight <= 0.0 or min_weight > max_weight:
        raise ValueError("metric min_weight/max_weight must be positive and ordered.")
    aggregation_name = str(config["aggregation"].get("name", "trimmed_mean"))
    if aggregation_name not in {"mean", "median", "trimmed_mean", "trimmed_mean_10pct", "trimmed_mean_10", "cvar"}:
        raise ValueError("aggregation.name is not supported by GPDE.")
    if aggregation_name == "trimmed_mean":
        trim_ratio = float(config["aggregation"].get("trim_ratio", 0.1))
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError("aggregation.trim_ratio must satisfy 0 <= trim_ratio < 0.5.")
    if aggregation_name == "cvar":
        alpha = float(config["aggregation"].get("alpha", 0.8))
        if not 0.0 <= alpha < 1.0:
            raise ValueError("aggregation.alpha must satisfy 0 <= alpha < 1 for cvar.")
    if str(config["optimizer"].get("name", "monitor_inverse_cdf")) != "monitor_inverse_cdf":
        raise ValueError("optimizer.name must be 'monitor_inverse_cdf'.")
    if float(config["optimizer"].get("tie_tolerance", 1.0e-12)) < 0.0:
        raise ValueError("optimizer.tie_tolerance must be nonnegative.")
    q_config = config.get("q_estimation", {})
    q_mode = str(q_config.get("mode", "global_fit"))
    if q_mode not in {"global_fit", "fixed"}:
        raise ValueError("q_estimation.mode must be global_fit or fixed.")
    fixed_q = q_config.get("fixed_q")
    if q_mode == "fixed" and (fixed_q is None or float(fixed_q) <= 0.0):
        raise ValueError("q_estimation.fixed_q must be positive when mode is fixed.")
    if float(q_config.get("min_q", 0.25)) <= 0.0:
        raise ValueError("q_estimation.min_q must be positive.")
    if float(q_config.get("max_q", 12.0)) < float(q_config.get("min_q", 0.25)):
        raise ValueError("q_estimation.max_q must be >= min_q.")
    monitor = config.get("monitor", {})
    if float(monitor.get("epsilon_a", 1.0e-12)) <= 0.0:
        raise ValueError("monitor.epsilon_a must be positive.")
    if int(monitor.get("smoothing_window", 3)) < 1:
        raise ValueError("monitor.smoothing_window must be positive.")
    if str(monitor.get("exponent", "q_root")) not in {"q_root", "identity", "a", "coefficient"}:
        raise ValueError("monitor.exponent is not supported.")
    if int(config["replay_refinement"].get("rounds", 3)) < 0:
        raise ValueError("replay_refinement.rounds must be nonnegative.")
    if int(config["replay_refinement"].get("local_window", 8)) < 1:
        raise ValueError("replay_refinement.local_window must be positive.")
    if float(config["replay_refinement"].get("lambda_final", 0.0)) < 0.0:
        raise ValueError("replay_refinement.lambda_final must be nonnegative.")
    if float(config["replay_refinement"].get("mu_smooth", 0.0)) < 0.0:
        raise ValueError("replay_refinement.mu_smooth must be nonnegative.")
    interpolation = str(config["oracle"].get("interpolation", "linear"))
    if interpolation != "linear":
        raise ValueError("oracle.interpolation must be 'linear' for the current GOES implementation.")
    output = config.get("output", {})
    if bool(output.get("save_images", False)):
        raise ValueError("output.save_images is not supported by the CPU GOES runner.")
    if not bool(output.get("save_schedule", True)):
        raise ValueError("output.save_schedule must remain true because GPDE schedule files are required outputs.")
    if not bool(output.get("save_probe_profile", True)):
        raise ValueError("output.save_probe_profile must remain true because GPDE probe profiles are required outputs.")
    if not bool(output.get("save_edge_table", True)):
        raise ValueError("output.save_edge_table must remain true because GPDE edge tables are required outputs.")


def write_config(config: dict[str, Any], path: str | Path) -> Path:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        _yaml_module().safe_dump(config, handle, sort_keys=False)
    return resolved


def stable_hash(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def short_config_hash(config: dict[str, Any]) -> str:
    ignored = copy.deepcopy(config)
    ignored.get("output", {}).pop("root", None)
    return stable_hash(ignored, length=8)

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import repo_path, stable_hash
from .coordinate import make_coordinate_adapter
from .oracle import OracleData, build_universal_oracle, make_calibration_samples
from .toy import make_toy_model


@dataclass
class OracleCacheResult:
    oracle: OracleData
    cache_key: str
    cache_path: Path
    metadata_path: Path
    loaded_from_cache: bool
    elapsed_seconds: float


def make_oracle_key(metadata: dict[str, Any]) -> str:
    key_payload = {
        "model_identifier": metadata["model_identifier"],
        "ode_sampler_family": metadata["ode_sampler_family"],
        "coordinate_mapping": metadata["coordinate_mapping"],
        "ref_integrator": metadata["ref_integrator"],
        "interpolation": metadata.get("interpolation", "linear"),
        "ref_nfe": metadata["ref_nfe"],
        "ref_grid_size": metadata["ref_grid_size"],
        "ref_grid_hash": metadata["ref_grid_hash"],
        "condition_split_hash": metadata["condition_split_hash"],
        "initial_noise_hash": metadata["initial_noise_hash"],
        "noise_seed_hash": metadata["noise_seed_hash"],
        "cfg": metadata["cfg"],
        "dtype": metadata["dtype"],
        "device": metadata["device"],
    }
    return stable_hash(key_payload, length=24)


def save_oracle(cache_dir: str | Path, cache_key: str, oracle: OracleData) -> tuple[Path, Path]:
    root = repo_path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / f"{cache_key}.npz"
    metadata_path = root / f"{cache_key}.json"
    np.savez_compressed(
        data_path,
        states=oracle.states,
        tangents=oracle.tangents,
        u_grid=oracle.u_grid,
        conditions=oracle.conditions,
        noise_seeds=oracle.noise_seeds,
    )
    oracle.metadata["oracle_cache_key"] = cache_key
    metadata = dict(oracle.metadata)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return data_path, metadata_path


def load_oracle(cache_dir: str | Path, cache_key: str) -> OracleData:
    root = repo_path(cache_dir)
    data_path = root / f"{cache_key}.npz"
    metadata_path = root / f"{cache_key}.json"
    with np.load(data_path) as payload:
        states = payload["states"]
        tangents = payload["tangents"]
        u_grid = payload["u_grid"]
        conditions = payload["conditions"]
        noise_seeds = payload["noise_seeds"]
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return OracleData(
        states=states,
        tangents=tangents,
        u_grid=u_grid,
        conditions=conditions,
        noise_seeds=noise_seeds,
        metadata=metadata,
    )


def build_or_load_oracle(config: dict[str, Any], *, split_section: str = "calibration") -> OracleCacheResult:
    started = time.time()
    model = make_toy_model(config["model"])
    coordinate = make_coordinate_adapter(config["coordinate"])
    split_cfg = config[split_section]
    samples = make_calibration_samples(
        num_samples=int(split_cfg["num_samples"]),
        seed=int(split_cfg["seed"]),
        state_shape=model.state_shape,
        split=str(split_cfg.get("split", split_section)),
    )
    ref_grid_size = int(config["oracle"]["ref_grid_size"])
    ref_grid = np.linspace(coordinate.u_min, coordinate.u_max, ref_grid_size, dtype=np.float64)
    key_metadata = {
        "model_identifier": model.identifier,
        "ode_sampler_family": "toy_deterministic_flow_ode",
        "coordinate_mapping": coordinate.metadata(),
        "ref_integrator": str(config["oracle"]["ref_integrator"]),
        "interpolation": str(config["oracle"].get("interpolation", "linear")),
        "ref_nfe": int(config["oracle"]["ref_nfe"]),
        "ref_grid_size": ref_grid_size,
        "ref_grid_hash": stable_hash(ref_grid.tolist()),
        "condition_split_hash": samples.metadata["condition_split_hash"],
        "initial_noise_hash": samples.metadata["initial_noise_hash"],
        "noise_seed_hash": samples.metadata["noise_seed_hash"],
        "cfg": {"guidance_scale": float(config["calibration"].get("guidance_scale", 1.0))},
        "dtype": "float64",
        "device": "cpu",
    }
    cache_key = make_oracle_key(key_metadata)
    cache_dir = repo_path(config["oracle"]["cache_dir"])
    cache_path = cache_dir / f"{cache_key}.npz"
    metadata_path = cache_dir / f"{cache_key}.json"
    if bool(config["oracle"].get("reuse", True)) and cache_path.exists() and metadata_path.exists():
        loaded = load_oracle(cache_dir, cache_key)
        loaded.metadata["loaded_from_cache"] = True
        return OracleCacheResult(
            oracle=loaded,
            cache_key=cache_key,
            cache_path=cache_path,
            metadata_path=metadata_path,
            loaded_from_cache=True,
            elapsed_seconds=time.time() - started,
        )

    oracle = build_universal_oracle(
        model,
        samples,
        coordinate_metadata=coordinate.metadata(),
        ref_integrator=str(config["oracle"]["ref_integrator"]),
        interpolation=str(config["oracle"].get("interpolation", "linear")),
        ref_nfe=int(config["oracle"]["ref_nfe"]),
        ref_grid_size=ref_grid_size,
        guidance_scale=float(config["calibration"].get("guidance_scale", 1.0)),
    )
    oracle.metadata["loaded_from_cache"] = False
    save_oracle(cache_dir, cache_key, oracle)
    return OracleCacheResult(
        oracle=oracle,
        cache_key=cache_key,
        cache_path=cache_path,
        metadata_path=metadata_path,
        loaded_from_cache=False,
        elapsed_seconds=time.time() - started,
    )

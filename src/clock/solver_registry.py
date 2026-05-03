from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CoordinateDomain = Literal["timestep", "timesteps", "sigma", "sigmas", "lambda", "log_sigma"]


@dataclass(frozen=True)
class SolverNativeSpec:
    name: str
    family: str
    native_coordinate: CoordinateDomain
    supports_base_trajectory_recording: bool
    recommended_window_len: int
    solver_order: int
    notes: str = ""


def _normalize_solver(name: str) -> str:
    return name.lower().replace("-", "_").replace("+", "p")


_PNDM_SPECS: dict[str, SolverNativeSpec] = {
    "euler": SolverNativeSpec("euler", "pndm", "timesteps", True, 1, 1),
    "ddim": SolverNativeSpec("ddim", "pndm", "timesteps", True, 1, 1),
    "pndm": SolverNativeSpec("pndm", "pndm", "timesteps", True, 4, 4),
    "deis": SolverNativeSpec("deis", "pndm", "timesteps", True, 2, 2),
    "dpm_solver_lu": SolverNativeSpec("dpm_solver_lu", "pndm", "timesteps", True, 2, 2),
    "dpm_solver_default": SolverNativeSpec("dpm_solver_default", "pndm", "timesteps", True, 2, 2),
    "dpm_solver_pp": SolverNativeSpec("dpm_solver_pp", "pndm", "timesteps", True, 2, 2),
    "dpm_solverpp": SolverNativeSpec("dpm_solverpp", "pndm", "timesteps", True, 2, 2),
    "unipc": SolverNativeSpec("unipc", "pndm", "timesteps", True, 2, 2),
    "stork4_1st": SolverNativeSpec("stork4_1st", "pndm", "sigmas", True, 4, 4),
    "stork_4_1st": SolverNativeSpec("stork_4_1st", "pndm", "sigmas", True, 4, 4),
    "stork4_2nd": SolverNativeSpec("stork4_2nd", "pndm", "sigmas", True, 4, 4),
    "stork_4_2nd": SolverNativeSpec("stork_4_2nd", "pndm", "sigmas", True, 4, 4),
    "heun2": SolverNativeSpec(
        "heun2",
        "pndm",
        "sigmas",
        False,
        2,
        2,
        "excluded_from_even_nfe_multiresolution_fp_clock",
    ),
}


_DIFFUSERS_SPECS: dict[str, SolverNativeSpec] = {
    "flow_euler": SolverNativeSpec("flow_euler", "diffusers_flow", "sigmas", True, 1, 1),
    "flow_dpm_solver": SolverNativeSpec("flow_dpm_solver", "diffusers_flow", "sigmas", True, 2, 2),
    "flow_unipc": SolverNativeSpec("flow_unipc", "diffusers_flow", "sigmas", True, 2, 2),
    "flow_stork4_1st": SolverNativeSpec("flow_stork4_1st", "diffusers_flow", "sigmas", True, 4, 4),
    "flow_stork_4_1st": SolverNativeSpec("flow_stork_4_1st", "diffusers_flow", "sigmas", True, 4, 4),
    "flow_stork4_2nd": SolverNativeSpec("flow_stork4_2nd", "diffusers_flow", "sigmas", True, 4, 4),
    "flow_stork_4_2nd": SolverNativeSpec("flow_stork_4_2nd", "diffusers_flow", "sigmas", True, 4, 4),
    "flow_stork4_3rd": SolverNativeSpec("flow_stork4_3rd", "diffusers_flow", "sigmas", True, 4, 4),
    "flow_stork_4_3rd": SolverNativeSpec("flow_stork_4_3rd", "diffusers_flow", "sigmas", True, 4, 4),
    "flow_heun": SolverNativeSpec(
        "flow_heun",
        "diffusers_flow",
        "sigmas",
        False,
        2,
        2,
        "excluded_from_even_nfe_multiresolution_fp_clock",
    ),
    "euler": SolverNativeSpec("euler", "diffusers_vp", "sigmas", True, 1, 1),
    "dpm_solver_pp": SolverNativeSpec("dpm_solver_pp", "diffusers_vp", "sigmas", True, 2, 2),
    "dpm_solverpp": SolverNativeSpec("dpm_solverpp", "diffusers_vp", "sigmas", True, 2, 2),
    "sde_dpm_solver_pp": SolverNativeSpec("sde_dpm_solver_pp", "diffusers_vp", "sigmas", True, 2, 2),
    "sde_dpmsolverpp": SolverNativeSpec("sde_dpmsolverpp", "diffusers_vp", "sigmas", True, 2, 2),
}


def get_solver_native_spec(backend: str, solver_name: str) -> SolverNativeSpec:
    normalized = _normalize_solver(solver_name)
    backend_name = str(backend).lower().strip()
    if backend_name == "pndm":
        registry = _PNDM_SPECS
    elif backend_name == "diffusers":
        registry = _DIFFUSERS_SPECS
    else:
        raise ValueError(f"Unsupported solver registry backend: {backend}")
    if normalized not in registry:
        raise ValueError(f"No native solver spec registered for {backend}:{solver_name}.")
    return registry[normalized]

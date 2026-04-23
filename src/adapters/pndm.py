from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from functools import wraps

import numpy as np
import torch
from PIL import Image

from src.clock.calibration import ForwardNormCollector
from src.utils.nfe_budget import resolve_effective_nfe_plan
from src.utils.config import load_yaml, repo_root
from src.utils.schedule_bundle import ScheduleBundle, scheduler_accepts


def _ensure_local_imports() -> None:
    root = repo_root()
    diffusers_src = root / "third_party" / "diffusers" / "src"
    pndm_root = root / "third_party" / "STORK" / "external" / "PNDM"
    stork_root = root / "third_party" / "STORK"
    for path in (str(diffusers_src), str(pndm_root), str(stork_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_local_imports()

from diffusers import (  # type: ignore  # noqa: E402
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from STORKScheduler import STORKScheduler  # type: ignore  # noqa: E402
from model.ddim import Model as DDIMModel  # type: ignore  # noqa: E402
from model.scoresde.ddpm import DDPM as ScoreSDEDDPMModel  # type: ignore  # noqa: E402
from model.scoresde.ncsnpp import NCSNpp as NCSNppModel  # type: ignore  # noqa: E402


PNDM_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "ddim": {
        "builder": DDIMModel,
    },
    "pf": {
        "builder": ScoreSDEDDPMModel,
        "config": {
            "nonlinearity": "swish",
            "nf": 128,
            "ch_mult": [1, 2, 2, 2],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.1,
            "resamp_with_conv": True,
            "conditional": True,
            "centered": True,
            "num_channels": 3,
            "image_size": 32,
        },
    },
    "pf_deep": {
        "builder": NCSNppModel,
        "config": {
            "nonlinearity": "swish",
            "nf": 128,
            "ch_mult": [1, 2, 2, 2],
            "num_res_blocks": 8,
            "attn_resolutions": [16],
            "dropout": 0.1,
            "resamp_with_conv": True,
            "conditional": True,
            "centered": True,
            "num_channels": 3,
            "image_size": 32,
            "fir": True,
            "fir_kernel": [1, 3, 3, 1],
            "skip_rescale": True,
            "resblock_type": "biggan",
            "progressive": "none",
            "progressive_input": "none",
            "embedding_type": "positional",
            "init_scale": 0.0,
            "combine_method": "sum",
            "continuous": False,
            "fourier_scale": 16,
        },
    },
}

SUPPORTED_MODEL_OUTPUT_TYPES = {"epsilon", "v_prediction", "flow"}
SIGMA_NATIVE_PNDM_SOLVERS = {
    "heun2",
    "stork4_1st",
    "stork_4_1st",
    "stork_4_1st_noise",
    "stork4_2nd",
    "stork_4_2nd",
    "stork_4_2nd_noise",
    "stork4_3rd",
    "stork_4_3rd",
    "stork_4_3rd_noise",
}
LAMBDA_NATIVE_PNDM_SOLVERS = {
    "dpm_solver_lu",
    "dpm_solver_default",
}
STORK_PNDM_SOLVERS = {
    "stork4_1st",
    "stork_4_1st",
    "stork_4_1st_noise",
    "stork4_2nd",
    "stork_4_2nd",
    "stork_4_2nd_noise",
}
STORK_FIRST_ORDER_PNDM_SOLVERS = {
    "stork4_1st",
    "stork_4_1st",
    "stork_4_1st_noise",
}


def _load_checkpoint_state(model_path: str | Path) -> dict[str, Any]:
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type for {model_path}: {type(state)!r}")
    return state


def infer_model_family(state_dict: dict[str, Any], model_path: str | Path | None = None) -> str:
    key_set = set(state_dict.keys())
    if "temb.dense.0.weight" in key_set:
        return "ddim"
    if any(key.startswith("all_modules.") for key in key_set):
        module_indices = [
            int(parts[1])
            for key in key_set
            if (parts := key.split(".", 2)) and len(parts) >= 2 and parts[0] == "all_modules" and parts[1].isdigit()
        ]
        if module_indices and max(module_indices) >= 80:
            return "pf_deep"
        return "pf"
    source = str(model_path) if model_path is not None else "checkpoint"
    raise ValueError(f"Unable to infer a supported PNDM model family from {source}.")


def build_model(model_family: str, *, device: str, native_model_config: dict[str, Any] | None = None) -> torch.nn.Module:
    if model_family not in PNDM_MODEL_CONFIGS:
        raise ValueError(f"Unsupported PNDM model family: {model_family}")
    entry = PNDM_MODEL_CONFIGS[model_family]
    builder = entry["builder"]
    if model_family == "ddim":
        if native_model_config is None:
            raise ValueError("DDIM model construction requires the native model config.")
        model_config = copy.deepcopy(native_model_config)
    else:
        model_config = copy.deepcopy(entry["config"])
    return builder(SimpleNamespace(device=device), model_config)


class NoisePredictionModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, model_family: str) -> None:
        super().__init__()
        self.model = model
        self.model_family = model_family
        model_config = getattr(model, "config", {})
        self._in_channels = int(
            getattr(model, "in_channels", model_config.get("in_channels", model_config.get("num_channels", 3)))
        )

    @property
    def in_channels(self) -> int:
        return self._in_channels

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        output = self.model(x, timestep)
        if output.ndim == x.ndim and output.shape[1] == x.shape[1] * 2:
            return output[:, : x.shape[1]]
        return output


def normalize_solver_name(name: str) -> str:
    return name.lower().replace("-", "_").replace("+", "p")


def solver_uses_sigma_schedule(solver_name: str) -> bool:
    return normalize_solver_name(solver_name) in SIGMA_NATIVE_PNDM_SOLVERS


def solver_uses_lambda_schedule(solver_name: str) -> bool:
    return normalize_solver_name(solver_name) in LAMBDA_NATIVE_PNDM_SOLVERS


def preferred_schedule_representation(solver_name: str) -> str:
    normalized = normalize_solver_name(solver_name)
    if normalized in SIGMA_NATIVE_PNDM_SOLVERS or normalized in LAMBDA_NATIVE_PNDM_SOLVERS:
        return "sigmas"
    return "timesteps"


def preferred_calibration_domain(solver_name: str) -> str:
    normalized = normalize_solver_name(solver_name)
    if normalized in LAMBDA_NATIVE_PNDM_SOLVERS:
        return "lambda"
    if normalized in SIGMA_NATIVE_PNDM_SOLVERS:
        return "sigmas"
    return "timesteps"


def build_scheduler(
    solver_name: str,
    *,
    diffusion_step: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
):
    common = dict(
        num_train_timesteps=diffusion_step,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
    )
    solver = normalize_solver_name(solver_name)
    if solver == "euler":
        return EulerDiscreteScheduler(**common)
    if solver == "heun2":
        return HeunDiscreteScheduler(**common)
    if solver == "ddim":
        return DDIMScheduler(**common)
    if solver == "pndm":
        return PNDMScheduler(**common)
    if solver == "deis":
        return DEISMultistepScheduler(**common, solver_order=2, algorithm_type="deis")
    if solver == "dpm_solver":
        raise ValueError(
            "Legacy solver `dpm_solver` has been removed. Use `dpm_solver_lu` or `dpm_solver_default` explicitly."
        )
    if solver == "dpm_solver_lu":
        return DPMSolverMultistepScheduler(
            **common,
            solver_order=2,
            algorithm_type="dpmsolver",
            use_lu_lambdas=True,
            final_sigmas_type="sigma_min",
        )
    if solver == "dpm_solver_default":
        return DPMSolverMultistepScheduler(
            **common,
            solver_order=2,
            algorithm_type="dpmsolver",
            use_lu_lambdas=False,
            final_sigmas_type="sigma_min",
        )
    if solver in {"dpm_solver_pp", "dpm_solverpp"}:
        return DPMSolverMultistepScheduler(
            **common,
            solver_order=2,
            algorithm_type="dpmsolver++",
            use_lu_lambdas=True,
        )
    if solver in {"stork4_1st", "stork_4_1st", "stork_4_1st_noise"}:
        return STORKScheduler(
            **common,
            prediction_type="epsilon",
            solver_order=4,
            derivative_order=1,
        )
    if solver in {"stork4_2nd", "stork_4_2nd", "stork_4_2nd_noise"}:
        return STORKScheduler(
            **common,
            prediction_type="epsilon",
            solver_order=4,
            derivative_order=2,
        )
    if solver in {"stork4_3rd", "stork_4_3rd", "stork_4_3rd_noise"}:
        raise ValueError(
            "STORK `stork4_3rd` is not implemented for noise-based PNDM models in upstream STORKScheduler. "
            "Use `stork4_1st` or `stork4_2nd` for PNDM experiments."
        )
    if solver == "unipc":
        return UniPCMultistepScheduler(**common, solver_order=2)
    raise ValueError(f"Unsupported PNDM solver: {solver_name}")


def _force_zero_terminal_sigma(scheduler) -> None:
    if not getattr(scheduler, "_force_final_sigma_zero", False):
        return
    if not hasattr(scheduler, "sigmas") or scheduler.sigmas is None:
        return
    scheduler.sigmas[-1] = torch.zeros((), device=scheduler.sigmas.device, dtype=scheduler.sigmas.dtype)


def _attach_force_zero_terminal_sigma(scheduler):
    if getattr(scheduler, "_force_final_sigma_zero", False):
        return scheduler

    scheduler._force_final_sigma_zero = True
    original_set_timesteps = scheduler.set_timesteps

    @wraps(original_set_timesteps)
    def set_timesteps_with_zero(*args, **kwargs):
        result = original_set_timesteps(*args, **kwargs)
        _force_zero_terminal_sigma(scheduler)
        return result

    scheduler.set_timesteps = set_timesteps_with_zero
    return scheduler


def _scheduler_uses_zero_final_sigma(scheduler) -> bool:
    if getattr(scheduler, "_force_final_sigma_zero", False):
        return True
    final_sigmas_type = getattr(getattr(scheduler, "config", None), "final_sigmas_type", None)
    return final_sigmas_type == "zero"


def _set_scheduler_state_from_timesteps(
    scheduler,
    timesteps: np.ndarray,
    *,
    device: torch.device,
) -> None:
    if not hasattr(scheduler, "alphas_cumprod"):
        raise ValueError(
            f"Scheduler {scheduler.__class__.__name__} does not expose `alphas_cumprod`, "
            "so a custom timestep schedule cannot be injected."
        )

    custom_timesteps = np.round(np.asarray(timesteps, dtype=np.float64)).astype(np.int64)
    if custom_timesteps.ndim != 1 or len(custom_timesteps) == 0:
        raise ValueError("Custom scheduler timesteps must be a non-empty 1D array.")
    if np.any(np.diff(custom_timesteps) > 0):
        raise ValueError("Custom scheduler timesteps must be descending (duplicates are allowed).")

    alphas_cumprod = scheduler.alphas_cumprod.detach().cpu().float().numpy()
    max_timestep = len(alphas_cumprod) - 1
    if custom_timesteps[0] > max_timestep or custom_timesteps[-1] < 0:
        raise ValueError(
            f"Custom scheduler timesteps must stay within [0, {max_timestep}], got "
            f"[{int(custom_timesteps[-1])}, {int(custom_timesteps[0])}]."
        )

    base_sigmas = np.sqrt(np.maximum(1.0 - alphas_cumprod, 0.0) / np.maximum(alphas_cumprod, 1.0e-12))
    schedule_sigmas = np.interp(custom_timesteps, np.arange(len(base_sigmas)), base_sigmas)

    if _scheduler_uses_zero_final_sigma(scheduler):
        sigma_last = 0.0
    else:
        sigma_last = float(base_sigmas[0])

    scheduler.timesteps = torch.from_numpy(custom_timesteps).to(device=device, dtype=torch.int64)
    scheduler.sigmas = torch.from_numpy(
        np.concatenate([schedule_sigmas, np.asarray([sigma_last], dtype=np.float64)]).astype(np.float32)
    ).to("cpu")
    scheduler.num_inference_steps = len(custom_timesteps)

    solver_order = int(getattr(getattr(scheduler, "config", None), "solver_order", 1))
    if hasattr(scheduler, "model_outputs"):
        scheduler.model_outputs = [None] * solver_order
    if hasattr(scheduler, "timestep_list"):
        scheduler.timestep_list = [None] * solver_order
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = None
    _force_zero_terminal_sigma(scheduler)


def _set_scheduler_state_from_sigmas(
    scheduler,
    sigmas: np.ndarray,
    *,
    device: torch.device,
    timesteps: np.ndarray | None = None,
) -> None:
    custom_sigmas = np.asarray(sigmas, dtype=np.float64)
    if custom_sigmas.ndim != 1 or len(custom_sigmas) < 2:
        raise ValueError("Custom scheduler sigmas must be a 1D array with at least two entries (including terminal sigma).")
    if np.any(np.diff(custom_sigmas) >= 0):
        raise ValueError("Custom scheduler sigmas must be strictly descending.")

    if timesteps is None:
        anchor_timesteps = _interp_timesteps_for_sigmas(
            scheduler,
            custom_sigmas[:-1],
            round_output=True,
            force_log_sigma=isinstance(scheduler, DPMSolverMultistepScheduler),
        )
    else:
        anchor_timesteps = np.asarray(timesteps, dtype=np.float64)
        if anchor_timesteps.ndim != 1 or len(anchor_timesteps) != len(custom_sigmas) - 1:
            raise ValueError("Custom scheduler timesteps must have length len(sigmas) - 1.")
        if np.any(np.diff(anchor_timesteps) > 0):
            raise ValueError("Custom scheduler timesteps must be descending (duplicates are allowed).")
        if getattr(getattr(scheduler, "config", None), "beta_schedule", None) != "squaredcos_cap_v2":
            anchor_timesteps = np.round(anchor_timesteps)

    scheduler.timesteps = torch.from_numpy(anchor_timesteps.astype(np.int64)).to(device=device, dtype=torch.int64)
    scheduler.sigmas = torch.from_numpy(custom_sigmas.astype(np.float32)).to("cpu")
    scheduler.num_inference_steps = len(custom_sigmas) - 1

    solver_order = int(getattr(getattr(scheduler, "config", None), "solver_order", 1))
    if hasattr(scheduler, "model_outputs"):
        scheduler.model_outputs = [None] * solver_order
    if hasattr(scheduler, "timestep_list"):
        scheduler.timestep_list = [None] * solver_order
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = None
    _force_zero_terminal_sigma(scheduler)


def _scheduler_prefers_sigma_schedule(scheduler) -> bool:
    return isinstance(scheduler, (HeunDiscreteScheduler, STORKScheduler))


def _scheduler_uses_manual_sigma_state(scheduler) -> bool:
    return isinstance(scheduler, DPMSolverMultistepScheduler)


def _bundle_anchor_timesteps(
    scheduler,
    schedule_bundle: ScheduleBundle,
) -> np.ndarray | None:
    if schedule_bundle.timesteps is not None:
        return np.asarray(schedule_bundle.timesteps, dtype=np.float64)
    if schedule_bundle.time_grid is not None:
        return np.asarray(schedule_bundle.time_grid[:-1], dtype=np.float64)
    if schedule_bundle.sigmas is not None:
        return _interp_timesteps_for_sigmas(scheduler, np.asarray(schedule_bundle.sigmas, dtype=np.float64))
    if schedule_bundle.sigma_grid is not None:
        return _interp_timesteps_for_sigmas(scheduler, np.asarray(schedule_bundle.sigma_grid[:-1], dtype=np.float64))
    return None


def _bundle_anchor_sigmas(schedule_bundle: ScheduleBundle) -> np.ndarray | None:
    if schedule_bundle.sigmas is not None:
        return np.asarray(schedule_bundle.sigmas, dtype=np.float64)
    if schedule_bundle.sigma_grid is not None:
        return np.asarray(schedule_bundle.sigma_grid[:-1], dtype=np.float64)
    return None


def _schedule_timesteps_arg(scheduler, timesteps: np.ndarray) -> list[float] | list[int]:
    values = np.asarray(timesteps, dtype=np.float64)
    if isinstance(scheduler, STORKScheduler):
        return values.astype(np.float32).tolist()
    return np.round(values).astype(np.int64).tolist()


def _schedule_bundle_kwargs(
    scheduler,
    schedule_bundle: ScheduleBundle,
    *,
    prefer: str,
) -> dict[str, list[float] | list[int]]:
    if prefer not in {"sigmas", "timesteps"}:
        raise ValueError(f"Unsupported schedule preference: {prefer}")

    kwargs: dict[str, list[float] | list[int]] = {}
    if prefer == "sigmas":
        sigmas = _bundle_anchor_sigmas(schedule_bundle)
        if sigmas is not None and scheduler_accepts(scheduler, "sigmas"):
            kwargs["sigmas"] = sigmas.tolist()
            timesteps = _bundle_anchor_timesteps(scheduler, schedule_bundle)
            if timesteps is not None and scheduler_accepts(scheduler, "timesteps"):
                kwargs["timesteps"] = _schedule_timesteps_arg(scheduler, timesteps)
            return kwargs

    timesteps = _bundle_anchor_timesteps(scheduler, schedule_bundle)
    if timesteps is not None and scheduler_accepts(scheduler, "timesteps"):
        kwargs["timesteps"] = _schedule_timesteps_arg(scheduler, timesteps)
        if prefer == "timesteps":
            sigmas = _bundle_anchor_sigmas(schedule_bundle)
            if sigmas is not None and scheduler_accepts(scheduler, "sigmas"):
                kwargs["sigmas"] = sigmas.tolist()
        return kwargs

    if prefer == "timesteps":
        sigmas = _bundle_anchor_sigmas(schedule_bundle)
        if sigmas is not None and scheduler_accepts(scheduler, "sigmas"):
            kwargs["sigmas"] = sigmas.tolist()
            return kwargs

    supported = [name for name in ("timesteps", "sigmas") if scheduler_accepts(scheduler, name)]
    supported_str = ", ".join(supported) if supported else "none"
    raise ValueError(f"No compatible schedule field found. Scheduler supports: {supported_str}")


def _configure_scheduler_timesteps(
    scheduler,
    *,
    num_inference_steps: int,
    device: torch.device,
    schedule_bundle: ScheduleBundle | None,
) -> None:
    if schedule_bundle is None:
        scheduler.set_timesteps(num_inference_steps, device=device)
        return

    if _scheduler_uses_manual_sigma_state(scheduler):
        sigma_grid = None if schedule_bundle.sigma_grid is None else np.asarray(schedule_bundle.sigma_grid, dtype=np.float64)
        if sigma_grid is None and schedule_bundle.sigmas is not None:
            anchor_sigmas = np.asarray(schedule_bundle.sigmas, dtype=np.float64)
            terminal_sigma = float(
                schedule_bundle.meta.get(
                    "terminal_sigma",
                    0.0 if _scheduler_uses_zero_final_sigma(scheduler) else float(_base_sigmas_from_scheduler(scheduler)[0]),
                )
            )
            sigma_grid = np.concatenate([anchor_sigmas, np.asarray([terminal_sigma], dtype=np.float64)])
        if sigma_grid is not None:
            anchor_timesteps = _bundle_anchor_timesteps(scheduler, schedule_bundle)
            _set_scheduler_state_from_sigmas(scheduler, sigma_grid, device=device, timesteps=anchor_timesteps)
            return

    prefer = "sigmas" if _scheduler_prefers_sigma_schedule(scheduler) else "timesteps"
    try:
        schedule_kwargs = _schedule_bundle_kwargs(scheduler, schedule_bundle, prefer=prefer)
    except ValueError:
        fallback_timesteps = _bundle_anchor_timesteps(scheduler, schedule_bundle)
        if fallback_timesteps is not None:
            _set_scheduler_state_from_timesteps(scheduler, fallback_timesteps, device=device)
            return
        raise

    try:
        scheduler.set_timesteps(device=device, **schedule_kwargs)
        _force_zero_terminal_sigma(scheduler)
    except ValueError as error:
        fallback_timesteps = _bundle_anchor_timesteps(scheduler, schedule_bundle)
        if fallback_timesteps is None:
            raise

        error_text = str(error)
        supported_fallback = any(
            marker in error_text
            for marker in (
                "Cannot use `timesteps`",
                "Cannot set `timesteps`",
                "Cannot use `sigmas`",
                "Cannot set `sigmas`",
            )
        )
        if not supported_fallback:
            raise
        _set_scheduler_state_from_timesteps(scheduler, fallback_timesteps, device=device)


def _base_sigmas_from_scheduler(scheduler) -> np.ndarray:
    if not hasattr(scheduler, "alphas_cumprod"):
        raise ValueError(f"Scheduler {scheduler.__class__.__name__} does not expose `alphas_cumprod`.")
    alphas_cumprod = scheduler.alphas_cumprod.detach().cpu().float().numpy()
    return np.sqrt(np.maximum(1.0 - alphas_cumprod, 0.0) / np.maximum(alphas_cumprod, 1.0e-12))


def _interp_sigmas_for_timesteps(scheduler, timesteps: np.ndarray) -> np.ndarray:
    base_sigmas = _base_sigmas_from_scheduler(scheduler)
    return np.interp(
        np.asarray(timesteps, dtype=np.float64),
        np.arange(len(base_sigmas), dtype=np.float64),
        base_sigmas,
    ).astype(np.float64)


def _interp_timesteps_for_sigmas(
    scheduler,
    sigmas: np.ndarray,
    *,
    round_output: bool = False,
    force_log_sigma: bool = False,
) -> np.ndarray:
    base_sigmas = _base_sigmas_from_scheduler(scheduler)
    query = np.asarray(sigmas, dtype=np.float64)
    clipped = np.clip(query, float(base_sigmas[0]), float(base_sigmas[-1]))
    use_log_sigma = force_log_sigma or isinstance(scheduler, DPMSolverMultistepScheduler)
    if use_log_sigma:
        log_sigmas = np.log(np.maximum(base_sigmas, 1.0e-10))
        if hasattr(scheduler, "_sigma_to_t"):
            try:
                timesteps = np.asarray(scheduler._sigma_to_t(clipped, log_sigmas), dtype=np.float64)
            except TypeError:
                timesteps = np.asarray(scheduler._sigma_to_t(clipped), dtype=np.float64)
        else:
            timesteps = np.interp(
                np.log(np.maximum(clipped, 1.0e-10)),
                log_sigmas,
                np.arange(len(base_sigmas), dtype=np.float64),
            ).astype(np.float64)
    else:
        timesteps = np.interp(
            clipped,
            base_sigmas,
            np.arange(len(base_sigmas), dtype=np.float64),
        ).astype(np.float64)
    if round_output and getattr(getattr(scheduler, "config", None), "beta_schedule", None) != "squaredcos_cap_v2":
        timesteps = np.round(timesteps)
    return timesteps


def build_pndm_sigma_grid(
    scheduler,
    *,
    physical_grid_size: int,
) -> np.ndarray:
    if physical_grid_size < 2:
        raise ValueError("physical_grid_size must be at least 2.")
    sigma_max = float(_base_sigmas_from_scheduler(scheduler)[-1])
    return np.linspace(sigma_max, 0.0, physical_grid_size, dtype=np.float64)


def build_pndm_lambda_grid(
    scheduler,
    *,
    physical_grid_size: int,
) -> np.ndarray:
    if physical_grid_size < 2:
        raise ValueError("physical_grid_size must be at least 2.")
    base_sigmas = _base_sigmas_from_scheduler(scheduler)
    sigma_max = float(base_sigmas[-1])
    sigma_min = float(max(base_sigmas[0], 1.0e-10))
    return np.linspace(-np.log(sigma_max), -np.log(sigma_min), physical_grid_size, dtype=np.float64)


def _sigmas_from_lambda_grid(lambdas: np.ndarray) -> np.ndarray:
    return np.exp(-np.asarray(lambdas, dtype=np.float64))


def _collapse_repeated_values(values: np.ndarray, *, expected_length: int | None = None) -> np.ndarray:
    collapsed: list[float] = []
    for value in np.asarray(values, dtype=np.float64).tolist():
        if not collapsed or not np.isclose(collapsed[-1], value):
            collapsed.append(float(value))
    result = np.asarray(collapsed, dtype=np.float64)
    if expected_length is not None and len(result) != expected_length:
        raise RuntimeError(
            f"Expected {expected_length} unique schedule values after collapsing repeats, got {len(result)}."
        )
    return result


def _collapse_repeated_timesteps(values: np.ndarray, *, expected_length: int | None = None) -> np.ndarray:
    return _collapse_repeated_values(values, expected_length=expected_length)


def build_pndm_native_coordinate_grid(
    scheduler,
    *,
    solver_name: str,
    effective_nfe: int,
    coordinate_domain: str,
) -> np.ndarray:
    plan = resolve_effective_nfe_plan(solver_name, effective_nfe)
    scheduler.set_timesteps(plan.solver_steps, device=torch.device("cpu"))

    normalized_domain = str(coordinate_domain).lower().strip()
    if normalized_domain == "timesteps":
        anchor_timesteps = _collapse_repeated_values(
            scheduler.timesteps.detach().cpu().float().numpy(),
            expected_length=plan.solver_steps,
        )
        return np.concatenate([anchor_timesteps, np.asarray([0.0], dtype=np.float64)])

    raw_sigmas = getattr(scheduler, "sigmas", None)
    if raw_sigmas is None:
        raise ValueError(f"Scheduler {scheduler.__class__.__name__} does not expose sigma schedules.")
    sigma_values = raw_sigmas.detach().cpu().float().numpy() if hasattr(raw_sigmas, "detach") else np.asarray(raw_sigmas)
    anchor_sigmas = _collapse_repeated_values(
        np.asarray(sigma_values[:-1], dtype=np.float64),
        expected_length=plan.solver_steps,
    )
    terminal_sigma = float(np.asarray(sigma_values, dtype=np.float64)[-1])
    sigma_grid = np.concatenate([anchor_sigmas, np.asarray([terminal_sigma], dtype=np.float64)])

    if normalized_domain == "sigmas":
        return sigma_grid
    if normalized_domain == "lambda":
        return -np.log(np.clip(sigma_grid, 1.0e-10, None))
    raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")


def _resolve_custom_heun_grid(
    scheduler,
    *,
    effective_nfe: int,
    schedule_bundle: ScheduleBundle | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    plan = resolve_effective_nfe_plan("heun2", effective_nfe)
    step_methods = tuple(schedule_bundle.meta.get("step_methods", plan.step_methods)) if schedule_bundle else plan.step_methods

    if schedule_bundle is not None and schedule_bundle.time_grid is not None:
        time_grid = np.asarray(schedule_bundle.time_grid, dtype=np.float64)
        if len(time_grid) != plan.solver_steps + 1:
            raise ValueError(
                f"Custom Heun time_grid must have length {plan.solver_steps + 1}, got {len(time_grid)}."
            )
        anchor_timesteps = time_grid[:-1]
        sigma_grid = _interp_sigmas_for_timesteps(scheduler, time_grid)
        sigma_grid[-1] = 0.0
        return anchor_timesteps, time_grid, sigma_grid, step_methods

    if schedule_bundle is not None and schedule_bundle.sigma_grid is not None:
        sigma_grid = np.asarray(schedule_bundle.sigma_grid, dtype=np.float64)
        if len(sigma_grid) != plan.solver_steps + 1:
            raise ValueError(
                f"Custom Heun sigma_grid must have length {plan.solver_steps + 1}, got {len(sigma_grid)}."
            )
        time_grid = _interp_timesteps_for_sigmas(scheduler, sigma_grid)
        anchor_timesteps = time_grid[:-1]
        return anchor_timesteps, time_grid, sigma_grid, step_methods

    if schedule_bundle is not None and schedule_bundle.sigmas is not None:
        anchor_sigmas = np.asarray(schedule_bundle.sigmas, dtype=np.float64)
        if len(anchor_sigmas) != plan.solver_steps:
            raise ValueError(
                f"Custom Heun sigmas must have length {plan.solver_steps}, got {len(anchor_sigmas)}."
            )
        terminal_sigma = float(schedule_bundle.meta.get("terminal_sigma", 0.0))
        sigma_grid = np.concatenate([anchor_sigmas, np.asarray([terminal_sigma], dtype=np.float64)])
        time_grid = _interp_timesteps_for_sigmas(scheduler, sigma_grid)
        anchor_timesteps = time_grid[:-1]
        return anchor_timesteps, time_grid, sigma_grid, step_methods

    if schedule_bundle is not None and schedule_bundle.timesteps is not None:
        anchor_timesteps = np.asarray(schedule_bundle.timesteps, dtype=np.float64)
    else:
        scheduler.set_timesteps(plan.solver_steps, device=device)
        anchor_timesteps = _collapse_repeated_timesteps(
            scheduler.timesteps.detach().cpu().float().numpy(),
            expected_length=plan.solver_steps,
        )

    terminal_timestep = float(schedule_bundle.meta.get("terminal_timestep", 0.0)) if schedule_bundle else 0.0
    time_grid = np.concatenate([anchor_timesteps, np.asarray([terminal_timestep], dtype=np.float64)])
    sigma_grid = _interp_sigmas_for_timesteps(scheduler, time_grid)
    sigma_grid[-1] = 0.0
    return anchor_timesteps, time_grid, sigma_grid, step_methods


def _normalize_model_output_type(model_output_type: str) -> str:
    normalized = str(model_output_type).lower().strip()
    if normalized == "flow_prediction":
        normalized = "flow"
    if normalized not in SUPPORTED_MODEL_OUTPUT_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_OUTPUT_TYPES))
        raise ValueError(f"Unsupported model_output_type `{model_output_type}`. Expected one of: {supported}.")
    return normalized


def _torch_interp_1d(query: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    if xp.ndim != 1 or fp.ndim != 1 or xp.shape != fp.shape:
        raise ValueError("xp and fp must be 1D tensors with identical shapes.")
    if len(xp) < 2:
        raise ValueError("xp and fp must contain at least two points for interpolation.")

    query_flat = torch.clamp(query.reshape(-1).to(dtype=fp.dtype), min=float(xp[0].item()), max=float(xp[-1].item()))
    indices = torch.searchsorted(xp, query_flat, right=False)
    indices = torch.clamp(indices, min=1, max=len(xp) - 1)
    left = indices - 1
    right = indices

    x0 = xp[left]
    x1 = xp[right]
    y0 = fp[left]
    y1 = fp[right]
    weight = (query_flat - x0) / torch.clamp(x1 - x0, min=torch.finfo(fp.dtype).eps)
    interpolated = y0 + weight * (y1 - y0)
    return interpolated.reshape(query.shape)


def _beta_at_timestep_torch(
    scheduler,
    timestep_value: float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not hasattr(scheduler, "betas"):
        raise ValueError(
            f"Scheduler {scheduler.__class__.__name__} does not expose `betas`, "
            "so VP PF-ODE velocity conversion cannot be evaluated."
        )
    betas = scheduler.betas.detach().to(device=device, dtype=dtype)
    lookup = torch.arange(len(betas), device=device, dtype=dtype)
    timestep_tensor = torch.as_tensor(timestep_value, device=device, dtype=dtype)
    return _torch_interp_1d(timestep_tensor, lookup, betas)


def _sigma_at_timestep_torch(
    scheduler,
    timestep_value: float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not hasattr(scheduler, "alphas_cumprod"):
        raise ValueError(
            f"Scheduler {scheduler.__class__.__name__} does not expose `alphas_cumprod`, "
            "so VP sigma interpolation cannot be evaluated."
        )
    alphas_cumprod = scheduler.alphas_cumprod.detach().to(device=device, dtype=dtype)
    base_sigmas = torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=0.0) / torch.clamp(alphas_cumprod, min=1.0e-12))
    lookup = torch.arange(len(base_sigmas), device=device, dtype=dtype)
    timestep_tensor = torch.as_tensor(timestep_value, device=device, dtype=dtype)
    return _torch_interp_1d(timestep_tensor, lookup, base_sigmas)


def _timestep_at_sigma_torch(
    scheduler,
    sigma_value: float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    force_log_sigma: bool = False,
) -> torch.Tensor:
    if not hasattr(scheduler, "alphas_cumprod"):
        raise ValueError(
            f"Scheduler {scheduler.__class__.__name__} does not expose `alphas_cumprod`, "
            "so VP timestep interpolation cannot be evaluated."
        )
    alphas_cumprod = scheduler.alphas_cumprod.detach().to(device=device, dtype=dtype)
    base_sigmas = torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=0.0) / torch.clamp(alphas_cumprod, min=1.0e-12))
    sigma_tensor = torch.as_tensor(sigma_value, device=device, dtype=dtype)
    sigma_tensor = torch.clamp(sigma_tensor, min=float(base_sigmas[0].item()), max=float(base_sigmas[-1].item()))
    lookup = torch.arange(len(base_sigmas), device=device, dtype=dtype)
    use_log_sigma = force_log_sigma or isinstance(scheduler, DPMSolverMultistepScheduler)
    if use_log_sigma:
        return _torch_interp_1d(
            torch.log(torch.clamp(sigma_tensor, min=1.0e-10)),
            torch.log(torch.clamp(base_sigmas, min=1.0e-10)),
            lookup,
        )
    return _torch_interp_1d(sigma_tensor, base_sigmas, lookup)


def _beta_at_timestep(
    scheduler,
    timestep_value: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _beta_at_timestep_torch(scheduler, timestep_value, device=device, dtype=dtype)


def _evaluate_velocity_with_tensors(
    model: torch.nn.Module,
    scheduler,
    sample: torch.Tensor,
    timestep_tensor: torch.Tensor,
    sigma_tensor: torch.Tensor,
    *,
    model_output_type: str = "epsilon",
    sigma_floor: float = 1.0e-6,
) -> torch.Tensor:
    device = sample.device
    model_timestep = timestep_tensor.to(device=device, dtype=torch.float32).reshape(()).expand(sample.shape[0])
    sigma_value = sigma_tensor.to(device=device, dtype=sample.dtype)
    model_input = sample / torch.sqrt(sigma_value.square() + 1.0)
    raw_model_output = model(model_input, model_timestep)
    normalized_output_type = _normalize_model_output_type(model_output_type)
    if normalized_output_type == "flow":
        return raw_model_output

    alpha_t = 1.0 / torch.sqrt(1.0 + sigma_value.square())
    sigma_t = sigma_value * alpha_t
    if normalized_output_type == "epsilon":
        epsilon_prediction = raw_model_output
    else:
        epsilon_prediction = alpha_t * raw_model_output + sigma_t * model_input

    beta_t = _beta_at_timestep_torch(scheduler, timestep_tensor, device=device, dtype=sample.dtype)
    sigma_denom = torch.clamp(sigma_value, min=float(sigma_floor))
    return 0.5 * beta_t * (epsilon_prediction / sigma_denom - sample)


def _evaluate_velocity(
    model: torch.nn.Module,
    scheduler,
    sample: torch.Tensor,
    timestep_value: float,
    sigma_value: float,
    *,
    model_output_type: str = "epsilon",
    sigma_floor: float = 1.0e-6,
) -> torch.Tensor:
    device = sample.device
    timestep_tensor = torch.as_tensor(float(timestep_value), device=device, dtype=sample.dtype)
    sigma_tensor = torch.as_tensor(float(sigma_value), device=device, dtype=sample.dtype)
    return _evaluate_velocity_with_tensors(
        model,
        scheduler,
        sample,
        timestep_tensor,
        sigma_tensor,
        model_output_type=model_output_type,
        sigma_floor=sigma_floor,
    )


def _evaluate_sigma_derivative_with_tensors(
    model: torch.nn.Module,
    sample: torch.Tensor,
    timestep_tensor: torch.Tensor,
    sigma_tensor: torch.Tensor,
    *,
    model_output_type: str = "epsilon",
) -> torch.Tensor:
    device = sample.device
    model_timestep = timestep_tensor.to(device=device, dtype=torch.float32).reshape(()).expand(sample.shape[0])
    sigma_value = sigma_tensor.to(device=device, dtype=sample.dtype)
    model_input = sample / torch.sqrt(sigma_value.square() + 1.0)
    raw_model_output = model(model_input, model_timestep)

    normalized_output_type = _normalize_model_output_type(model_output_type)
    if normalized_output_type in {"flow", "epsilon"}:
        return raw_model_output

    alpha_t = 1.0 / torch.sqrt(1.0 + sigma_value.square())
    sigma_t = sigma_value * alpha_t
    return alpha_t * raw_model_output + sigma_t * model_input


def _evaluate_sigma_derivative(
    model: torch.nn.Module,
    sample: torch.Tensor,
    timestep_value: float,
    sigma_value: float,
    *,
    model_output_type: str = "epsilon",
) -> torch.Tensor:
    device = sample.device
    timestep_tensor = torch.as_tensor(float(timestep_value), device=device, dtype=sample.dtype)
    sigma_tensor = torch.as_tensor(float(sigma_value), device=device, dtype=sample.dtype)
    return _evaluate_sigma_derivative_with_tensors(
        model,
        sample,
        timestep_tensor,
        sigma_tensor,
        model_output_type=model_output_type,
    )


def _evaluate_lambda_derivative(
    model: torch.nn.Module,
    scheduler,
    sample: torch.Tensor,
    lambda_value: float,
    *,
    model_output_type: str = "epsilon",
) -> torch.Tensor:
    device = sample.device
    lambda_tensor = torch.as_tensor(float(lambda_value), device=device, dtype=sample.dtype)
    sigma_tensor = torch.exp(-lambda_tensor)
    timestep_tensor = _timestep_at_sigma_torch(
        scheduler,
        sigma_tensor,
        device=device,
        dtype=sample.dtype,
        force_log_sigma=True,
    )
    sigma_derivative = _evaluate_sigma_derivative_with_tensors(
        model,
        sample,
        timestep_tensor,
        sigma_tensor,
        model_output_type=model_output_type,
    )
    return -sigma_tensor * sigma_derivative


def _evaluate_sigma_derivative_microbatched(
    model: torch.nn.Module,
    sample: torch.Tensor,
    timestep_value: float,
    sigma_value: float,
    *,
    microbatch_size: int | None,
    model_output_type: str = "epsilon",
) -> torch.Tensor:
    if microbatch_size is None or microbatch_size <= 0 or microbatch_size >= sample.shape[0]:
        return _evaluate_sigma_derivative(
            model,
            sample,
            timestep_value,
            sigma_value,
            model_output_type=model_output_type,
        )

    chunks: list[torch.Tensor] = []
    for start in range(0, sample.shape[0], microbatch_size):
        stop = min(start + microbatch_size, sample.shape[0])
        chunks.append(
            _evaluate_sigma_derivative(
                model,
                sample[start:stop],
                timestep_value,
                sigma_value,
                model_output_type=model_output_type,
            )
        )
    return torch.cat(chunks, dim=0)


def _evaluate_lambda_derivative_microbatched(
    model: torch.nn.Module,
    scheduler,
    sample: torch.Tensor,
    lambda_value: float,
    *,
    microbatch_size: int | None,
    model_output_type: str = "epsilon",
) -> torch.Tensor:
    if microbatch_size is None or microbatch_size <= 0 or microbatch_size >= sample.shape[0]:
        return _evaluate_lambda_derivative(
            model,
            scheduler,
            sample,
            lambda_value,
            model_output_type=model_output_type,
        )

    chunks: list[torch.Tensor] = []
    for start in range(0, sample.shape[0], microbatch_size):
        stop = min(start + microbatch_size, sample.shape[0])
        chunks.append(
            _evaluate_lambda_derivative(
                model,
                scheduler,
                sample[start:stop],
                lambda_value,
                model_output_type=model_output_type,
            )
        )
    return torch.cat(chunks, dim=0)


def _evaluate_velocity_microbatched(
    model: torch.nn.Module,
    scheduler,
    sample: torch.Tensor,
    timestep_value: float,
    sigma_value: float,
    *,
    microbatch_size: int | None,
    model_output_type: str = "epsilon",
    sigma_floor: float = 1.0e-6,
) -> torch.Tensor:
    if microbatch_size is None or microbatch_size <= 0 or microbatch_size >= sample.shape[0]:
        return _evaluate_velocity(
            model,
            scheduler,
            sample,
            timestep_value,
            sigma_value,
            model_output_type=model_output_type,
            sigma_floor=sigma_floor,
        )

    chunks: list[torch.Tensor] = []
    for start in range(0, sample.shape[0], microbatch_size):
        stop = min(start + microbatch_size, sample.shape[0])
        chunks.append(
            _evaluate_velocity(
                model,
                scheduler,
                sample[start:stop],
                timestep_value,
                sigma_value,
                model_output_type=model_output_type,
                sigma_floor=sigma_floor,
            )
        )
    return torch.cat(chunks, dim=0)


def build_velocity_oracle(
    model: torch.nn.Module,
    scheduler,
    *,
    model_output_type: str = "epsilon",
    sigma_floor: float = 1.0e-6,
):
    def oracle(sample: torch.Tensor, timestep_tensor: torch.Tensor) -> torch.Tensor:
        sigma_tensor = _sigma_at_timestep_torch(
            scheduler,
            timestep_tensor,
            device=sample.device,
            dtype=sample.dtype,
        )
        return _evaluate_velocity_with_tensors(
            model,
            scheduler,
            sample,
            timestep_tensor,
            sigma_tensor,
            model_output_type=model_output_type,
            sigma_floor=sigma_floor,
        )

    return oracle


def build_sigma_derivative_oracle(
    model: torch.nn.Module,
    scheduler,
    *,
    model_output_type: str = "epsilon",
):
    def oracle(sample: torch.Tensor, sigma_tensor: torch.Tensor) -> torch.Tensor:
        sigma_value = sigma_tensor.to(device=sample.device, dtype=sample.dtype)
        timestep_tensor = _timestep_at_sigma_torch(
            scheduler,
            sigma_value,
            device=sample.device,
            dtype=sample.dtype,
        )
        return _evaluate_sigma_derivative_with_tensors(
            model,
            sample,
            timestep_tensor,
            sigma_value,
            model_output_type=model_output_type,
        )

    return oracle


def build_lambda_velocity_oracle(
    model: torch.nn.Module,
    scheduler,
    *,
    model_output_type: str = "epsilon",
):
    def oracle(sample: torch.Tensor, lambda_tensor: torch.Tensor) -> torch.Tensor:
        lambda_value = lambda_tensor.to(device=sample.device, dtype=sample.dtype)
        sigma_tensor = torch.exp(-lambda_value)
        timestep_tensor = _timestep_at_sigma_torch(
            scheduler,
            sigma_tensor,
            device=sample.device,
            dtype=sample.dtype,
            force_log_sigma=True,
        )
        sigma_derivative = _evaluate_sigma_derivative_with_tensors(
            model,
            sample,
            timestep_tensor,
            sigma_tensor,
            model_output_type=model_output_type,
        )
        return -sigma_tensor * sigma_derivative

    return oracle


def collect_shared_clock_velocity_norms(
    *,
    model: torch.nn.Module,
    scheduler,
    physical_grid: np.ndarray,
    pilot_solver: str,
    image_size: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    observation_microbatch: int | None = None,
    model_output_type: str = "epsilon",
    sigma_floor: float = 1.0e-6,
    coordinate_domain: str = "timesteps",
    quantity: str = "velocity",
) -> np.ndarray:
    normalized_pilot = normalize_solver_name(pilot_solver)
    if normalized_pilot not in {"euler", "heun2"} and normalized_pilot not in STORK_PNDM_SOLVERS:
        raise ValueError(f"Unsupported PNDM shared-clock pilot solver: {pilot_solver}")
    normalized_quantity = str(quantity).lower().strip()
    if normalized_quantity in {"dtv", "dtv_norm", "material_derivative_norm"}:
        normalized_quantity = "material_derivative"
    if normalized_quantity not in {"velocity", "material_derivative"}:
        raise ValueError(f"Unsupported shared-clock quantity: {quantity}")
    use_material_derivative = normalized_quantity == "material_derivative"

    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must contain at least two points.")

    normalized_domain = str(coordinate_domain).lower().strip()
    if normalized_domain not in {"timesteps", "sigmas", "lambda"}:
        raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")

    if normalized_domain == "timesteps":
        time_grid = grid
        sigma_grid = _interp_sigmas_for_timesteps(scheduler, grid)
        sigma_grid[-1] = 0.0
        lambda_grid = None
    elif normalized_domain == "sigmas":
        sigma_grid = grid
        time_grid = _interp_timesteps_for_sigmas(scheduler, sigma_grid)
        lambda_grid = None
    else:
        lambda_grid = grid
        sigma_grid = _sigmas_from_lambda_grid(lambda_grid)
        time_grid = _interp_timesteps_for_sigmas(scheduler, sigma_grid, force_log_sigma=True)

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    batches: list[np.ndarray] = []
    use_native_stork_pilot = isinstance(scheduler, STORKScheduler) and normalized_domain in {"timesteps", "sigmas"}

    if use_material_derivative:
        from src.clock.lcs import material_derivative_jvp, per_sample_l2_norm

        if normalized_domain == "timesteps":
            velocity_fn = build_velocity_oracle(
                model,
                scheduler,
                model_output_type=model_output_type,
                sigma_floor=sigma_floor,
            )
        elif normalized_domain == "lambda":
            velocity_fn = build_lambda_velocity_oracle(
                model,
                scheduler,
                model_output_type=model_output_type,
            )
        else:
            velocity_fn = build_sigma_derivative_oracle(
                model,
                scheduler,
                model_output_type=model_output_type,
            )

        def evaluate_material_derivative(sample_tensor: torch.Tensor, coordinate_value: float) -> torch.Tensor:
            if (
                observation_microbatch is None
                or observation_microbatch <= 0
                or observation_microbatch >= sample_tensor.shape[0]
            ):
                return material_derivative_jvp(velocity_fn, sample_tensor, coordinate_value)
            chunks: list[torch.Tensor] = []
            for start in range(0, sample_tensor.shape[0], observation_microbatch):
                stop = min(start + observation_microbatch, sample_tensor.shape[0])
                chunks.append(material_derivative_jvp(velocity_fn, sample_tensor[start:stop], coordinate_value))
            return torch.cat(chunks, dim=0)

    context = torch.inference_mode(False) if use_material_derivative else torch.inference_mode()
    with context:
        for _ in range(num_batches):
            if use_native_stork_pilot:
                scheduler.set_timesteps(
                    num_inference_steps=len(grid) - 1,
                    device=device,
                    timesteps=np.asarray(time_grid[:-1], dtype=np.float32).tolist(),
                    sigmas=np.asarray(sigma_grid[:-1], dtype=np.float32).tolist(),
                )
            sample = torch.randn(
                (batch_size, model.in_channels, image_size, image_size),
                generator=generator,
                device=device,
            ) * float(sigma_grid[0])
            batch_norms: list[np.ndarray] = []
            for index in range(len(grid)):
                sigma_value = float(sigma_grid[index])
                timestep_value = float(time_grid[index])
                if use_material_derivative:
                    if normalized_domain == "timesteps":
                        coordinate_value = timestep_value
                    elif normalized_domain == "lambda":
                        coordinate_value = float(lambda_grid[index])
                    else:
                        coordinate_value = sigma_value
                    material_derivative = evaluate_material_derivative(sample, coordinate_value)
                    batch_norms.append(per_sample_l2_norm(material_derivative).cpu().numpy())
                else:
                    if normalized_domain == "timesteps":
                        velocity = _evaluate_velocity_microbatched(
                            model,
                            scheduler,
                            sample,
                            timestep_value,
                            sigma_value,
                            microbatch_size=observation_microbatch,
                            model_output_type=model_output_type,
                            sigma_floor=sigma_floor,
                        )
                    elif normalized_domain == "lambda":
                        velocity = _evaluate_lambda_derivative_microbatched(
                            model,
                            scheduler,
                            sample,
                            float(lambda_grid[index]),
                            microbatch_size=observation_microbatch,
                            model_output_type=model_output_type,
                        )
                    else:
                        velocity = _evaluate_sigma_derivative_microbatched(
                            model,
                            sample,
                            timestep_value,
                            sigma_value,
                            microbatch_size=observation_microbatch,
                            model_output_type=model_output_type,
                        )
                    batch_norms.append(velocity.float().reshape(batch_size, -1).norm(dim=1).cpu().numpy())
                if index == len(grid) - 1:
                    continue
                with torch.inference_mode():
                    if use_native_stork_pilot:
                        scheduler_timestep = scheduler.timesteps[index]
                        model_timestep = scheduler_timestep
                        if not isinstance(model_timestep, torch.Tensor):
                            model_timestep = torch.tensor([model_timestep], device=device)
                        if model_timestep.ndim == 0:
                            model_timestep = model_timestep[None]
                        if model_timestep.numel() == 1:
                            model_timestep = model_timestep.expand(batch_size)
                        model_input = sample
                        if hasattr(scheduler, "scale_model_input"):
                            model_input = scheduler.scale_model_input(sample, scheduler_timestep)
                        model_output = model(model_input, model_timestep)
                        step_output = scheduler.step(model_output, scheduler_timestep, sample)
                        sample = step_output.prev_sample
                        continue
                    if use_material_derivative:
                        if normalized_domain == "timesteps":
                            velocity = _evaluate_velocity_microbatched(
                                model,
                                scheduler,
                                sample,
                                timestep_value,
                                sigma_value,
                                microbatch_size=observation_microbatch,
                                model_output_type=model_output_type,
                                sigma_floor=sigma_floor,
                            )
                        elif normalized_domain == "lambda":
                            velocity = _evaluate_lambda_derivative_microbatched(
                                model,
                                scheduler,
                                sample,
                                float(lambda_grid[index]),
                                microbatch_size=observation_microbatch,
                                model_output_type=model_output_type,
                            )
                        else:
                            velocity = _evaluate_sigma_derivative_microbatched(
                                model,
                                sample,
                                timestep_value,
                                sigma_value,
                                microbatch_size=observation_microbatch,
                                model_output_type=model_output_type,
                            )
                    if normalized_domain == "timesteps":
                        dt = float(time_grid[index + 1] - timestep_value)
                    elif normalized_domain == "lambda":
                        dt = float(lambda_grid[index + 1] - float(lambda_grid[index]))
                    else:
                        dt = float(sigma_grid[index + 1] - sigma_value)
                    if normalized_pilot == "euler" or normalized_pilot in STORK_FIRST_ORDER_PNDM_SOLVERS:
                        sample = sample + velocity * dt
                        continue
                    predicted = sample + velocity * dt
                    next_timestep_value = float(time_grid[index + 1])
                    next_sigma_value = float(sigma_grid[index + 1])
                    if normalized_domain == "timesteps":
                        next_velocity = _evaluate_velocity_microbatched(
                            model,
                            scheduler,
                            predicted,
                            next_timestep_value,
                            next_sigma_value,
                            microbatch_size=observation_microbatch,
                            model_output_type=model_output_type,
                            sigma_floor=sigma_floor,
                        )
                    elif normalized_domain == "lambda":
                        next_velocity = _evaluate_lambda_derivative_microbatched(
                            model,
                            scheduler,
                            predicted,
                            float(lambda_grid[index + 1]),
                            microbatch_size=observation_microbatch,
                            model_output_type=model_output_type,
                        )
                    else:
                        next_velocity = _evaluate_sigma_derivative_microbatched(
                            model,
                            predicted,
                            next_timestep_value,
                            next_sigma_value,
                            microbatch_size=observation_microbatch,
                            model_output_type=model_output_type,
                        )
                    sample = sample + 0.5 * (velocity + next_velocity) * dt
            batches.append(np.stack(batch_norms, axis=1))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return np.concatenate(batches, axis=0)


def _run_budgeted_heun(
    *,
    model: torch.nn.Module,
    scheduler,
    batch_size: int,
    effective_nfe: int,
    height: int,
    width: int,
    generator: torch.Generator,
    schedule_bundle: ScheduleBundle | None,
    model_output_type: str = "epsilon",
    sigma_floor: float = 1.0e-6,
) -> torch.Tensor:
    device = next(model.parameters()).device
    plan = resolve_effective_nfe_plan("heun2", effective_nfe)
    anchor_timesteps, time_grid, sigma_grid, step_methods = _resolve_custom_heun_grid(
        scheduler,
        effective_nfe=effective_nfe,
        schedule_bundle=schedule_bundle,
        device=device,
    )
    if len(step_methods) != plan.solver_steps:
        raise ValueError(f"Expected {plan.solver_steps} step methods for Heun execution, got {len(step_methods)}.")

    init_sigma = float(sigma_grid[0])
    image = torch.randn(
        (batch_size, model.in_channels, height, width),
        generator=generator,
        device=device,
    ) * init_sigma

    for index, method in enumerate(step_methods):
        sigma_value = float(sigma_grid[index])
        sigma_next = float(sigma_grid[index + 1])
        timestep_value = float(time_grid[index])
        next_timestep_value = float(time_grid[index + 1])
        dt = sigma_next - sigma_value
        derivative = _evaluate_sigma_derivative(
            model,
            image,
            timestep_value,
            sigma_value,
            model_output_type=model_output_type,
        )
        if method == "euler":
            image = image + derivative * dt
            continue
        if method != "heun2":
            raise ValueError(f"Unsupported custom Heun step method: {method}")
        predicted = image + derivative * dt
        next_derivative = _evaluate_sigma_derivative(
            model,
            predicted,
            next_timestep_value,
            sigma_next,
            model_output_type=model_output_type,
        )
        image = image + 0.5 * (derivative + next_derivative) * dt

    return ((image.clamp(-1, 1) + 1) / 2).cpu()


class PndmGenerationPipeline:
    def __init__(self, model: torch.nn.Module, scheduler) -> None:
        self.model = model
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int,
        num_inference_steps: int,
        height: int,
        width: int,
        generator: torch.Generator,
        schedule_bundle: ScheduleBundle | None = None,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        solver_steps = num_inference_steps
        if isinstance(self.scheduler, HeunDiscreteScheduler):
            solver_steps = resolve_effective_nfe_plan("heun2", num_inference_steps).solver_steps
        latents = torch.randn(
            (batch_size, self.model.in_channels, height, width),
            generator=generator,
            device=device,
        )
        _configure_scheduler_timesteps(
            self.scheduler,
            num_inference_steps=solver_steps,
            device=device,
            schedule_bundle=schedule_bundle,
        )

        init_noise_sigma = getattr(self.scheduler, "init_noise_sigma", None)
        if init_noise_sigma is not None:
            latents = latents * torch.as_tensor(init_noise_sigma, device=device, dtype=latents.dtype)

        image = latents
        for timestep in self.scheduler.timesteps:
            model_timestep = timestep
            if not isinstance(model_timestep, torch.Tensor):
                model_timestep = torch.tensor([model_timestep], device=device)
            if model_timestep.ndim == 0:
                model_timestep = model_timestep[None]
            if model_timestep.numel() == 1:
                model_timestep = model_timestep.expand(batch_size)
            model_input = image
            if hasattr(self.scheduler, "scale_model_input"):
                model_input = self.scheduler.scale_model_input(image, timestep)
            model_output = self.model(model_input, model_timestep)
            step_output = self.scheduler.step(model_output, timestep, image)
            image = step_output.prev_sample

        return ((image.clamp(-1, 1) + 1) / 2).cpu()


def load_native_config(native_config_path: str | Path) -> dict[str, Any]:
    return load_yaml(native_config_path)


def load_training_dataset(native_dataset_config: dict[str, Any], dataset_root: str | Path | None = None):
    if dataset_root is not None and native_dataset_config.get("dataset") == "CIFAR10":
        from torchvision import transforms  # type: ignore
        from torchvision.datasets import CIFAR10  # type: ignore

        resolved_root = Path(dataset_root)
        if resolved_root.name == "cifar-10-batches-py":
            resolved_root = resolved_root.parent
        if not resolved_root.exists():
            raise FileNotFoundError(f"Configured CIFAR-10 dataset root does not exist: {resolved_root}")

        image_size = int(native_dataset_config["image_size"])
        if native_dataset_config.get("random_flip", False):
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            )
        return CIFAR10(str(resolved_root), train=True, download=False, transform=transform)

    from dataset import get_dataset  # type: ignore  # noqa: E402

    dataset, _ = get_dataset(SimpleNamespace(), native_dataset_config)
    return dataset


def load_model(native_config_path: str | Path, model_path: str | Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    native_config = load_native_config(native_config_path)
    state_dict = _load_checkpoint_state(model_path)
    model_family = infer_model_family(state_dict, model_path=model_path)
    model = build_model(model_family, device=device, native_model_config=native_config.get("Model"))
    model.load_state_dict(state_dict, strict=True)
    wrapped = NoisePredictionModel(model, model_family).to(device)
    wrapped.eval()
    return wrapped, native_config


def save_images(images: torch.Tensor, output_dir: str | Path, start_index: int) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(images):
        array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
        Image.fromarray(array).save(output_path / f"{start_index + offset:06d}.png")


def run_generation(
    *,
    model: torch.nn.Module,
    scheduler,
    image_size: int,
    num_samples: int,
    batch_size: int,
    num_inference_steps: int,
    seed: int,
    output_dir: str | Path,
    schedule_bundle: ScheduleBundle | None = None,
) -> Path:
    pipeline = PndmGenerationPipeline(model, scheduler)
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    generated = 0
    while generated < num_samples:
        current = min(batch_size, num_samples - generated)
        images = pipeline(
            batch_size=current,
            num_inference_steps=num_inference_steps,
            height=image_size,
            width=image_size,
            generator=generator,
            schedule_bundle=schedule_bundle,
        )
        save_images(images, output_dir, generated)
        generated += current
    return Path(output_dir)


def collect_calibration_records(
    *,
    model: torch.nn.Module,
    scheduler,
    image_size: int,
    num_inference_steps: int,
    seed: int,
    num_samples: int = 1,
    norm_type: str = "l2",
    normalize_by_dim: bool = False,
) -> tuple[list, torch.Tensor]:
    pipeline = PndmGenerationPipeline(model, scheduler)
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    plan = resolve_effective_nfe_plan("heun2" if isinstance(scheduler, HeunDiscreteScheduler) else "euler", num_inference_steps)
    scheduler.set_timesteps(plan.solver_steps, device=next(model.parameters()).device)
    domain_values = scheduler.timesteps.detach().cpu().float()
    if isinstance(scheduler, HeunDiscreteScheduler):
        domain_values = torch.from_numpy(
            _collapse_repeated_timesteps(domain_values.numpy(), expected_length=plan.solver_steps).astype(np.float32)
        )
    with ForwardNormCollector(model, norm_type=norm_type, normalize_by_dim=normalize_by_dim) as collector:
        pipeline(
            batch_size=num_samples,
            num_inference_steps=num_inference_steps,
            height=image_size,
            width=image_size,
            generator=generator,
            schedule_bundle=None,
        )
    return collector.records, domain_values

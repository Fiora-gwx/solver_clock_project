from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from functools import wraps
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image

from src.clock.defect_balanced import (
    StepRefinementStats,
    _microbatch_map,
    _refined_step,
    build_velocity_stepper,
    collect_step_refinement_stats,
    collect_velocity_curvature_stats,
    estimate_refinement_order_and_defect,
    per_sample_l2_norm,
)
from src.clock.fp_clock import (
    FPTrajectoryStats,
    collect_anchored_replay_stats,
    collect_fp_clock_stats,
    concatenate_fp_clock_stats,
)
from src.clock.solver_registry import get_solver_native_spec
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
    del solver_name
    return False


def preferred_schedule_representation(solver_name: str) -> str:
    normalized = normalize_solver_name(solver_name)
    if normalized in SIGMA_NATIVE_PNDM_SOLVERS:
        return "sigmas"
    return "timesteps"


def preferred_calibration_domain(solver_name: str) -> str:
    normalized = normalize_solver_name(solver_name)
    if normalized in SIGMA_NATIVE_PNDM_SOLVERS:
        return "sigmas"
    return "timesteps"


def _attach_unipc_device_sigmas(scheduler):
    if not isinstance(scheduler, UniPCMultistepScheduler) or getattr(scheduler, "_solver_clock_unipc_sigmas_patch", False):
        return scheduler

    original_set_timesteps = scheduler.set_timesteps

    @wraps(original_set_timesteps)
    def set_timesteps_with_device_sigmas(*args, **kwargs):
        result = original_set_timesteps(*args, **kwargs)
        target_device = kwargs.get("device", None)
        if target_device is None and len(args) >= 2:
            target_device = args[1]
        _move_unipc_sigmas_to_device(scheduler, target_device)
        return result

    scheduler.set_timesteps = set_timesteps_with_device_sigmas
    scheduler._solver_clock_unipc_sigmas_patch = True
    return scheduler


def _move_unipc_sigmas_to_device(scheduler, device: torch.device | str | None) -> None:
    if not isinstance(scheduler, UniPCMultistepScheduler) or device is None:
        return
    if isinstance(getattr(scheduler, "sigmas", None), torch.Tensor):
        scheduler.sigmas = scheduler.sigmas.to(device=device)
    solver_p = getattr(scheduler, "solver_p", None)
    if solver_p is not None and isinstance(getattr(solver_p, "sigmas", None), torch.Tensor):
        solver_p.sigmas = solver_p.sigmas.to(device=device)


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
        return _attach_unipc_device_sigmas(UniPCMultistepScheduler(**common, solver_order=2))
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
    _move_unipc_sigmas_to_device(scheduler, device)


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
    _move_unipc_sigmas_to_device(scheduler, device)


def _pndm_prk_plms_from_anchor_timesteps(
    scheduler,
    custom_timesteps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    descending = np.round(np.asarray(custom_timesteps, dtype=np.float64)).astype(np.int64)
    if descending.ndim != 1 or len(descending) == 0:
        raise ValueError("Custom PNDM timesteps must be a non-empty 1D array.")
    if np.any(np.diff(descending) >= 0):
        raise ValueError("Custom PNDM timesteps must be strictly descending.")

    ascending = descending[::-1].copy()
    scheduler_order = int(getattr(scheduler, "pndm_order", 4))
    scheduler_order = max(1, min(scheduler_order, len(ascending)))
    if getattr(getattr(scheduler, "config", None), "skip_prk_steps", False):
        prk_timesteps = np.asarray([], dtype=np.int64)
        if len(ascending) >= 2:
            plms_source = np.concatenate([ascending[:-1], ascending[-2:-1], ascending[-1:]])
        else:
            plms_source = ascending
        return ascending, prk_timesteps, plms_source[::-1].copy().astype(np.int64)

    warmup_start = len(ascending) - scheduler_order
    warmup = ascending[warmup_start:]
    raw: list[float] = []
    for offset, value in enumerate(warmup):
        index = warmup_start + offset
        if index + 1 < len(ascending):
            half_step = 0.5 * float(ascending[index + 1] - ascending[index])
        elif len(ascending) > 1:
            half_step = 0.5 * float(ascending[-1] - ascending[-2])
        else:
            half_step = 0.5 * float(getattr(scheduler.config, "num_train_timesteps", 1000))
        raw.extend([float(value), float(value) + half_step])
    prk_source = np.round(np.asarray(raw[:-1], dtype=np.float64)).astype(np.int64)
    prk_timesteps = prk_source.repeat(2)[1:-1][::-1].copy()
    plms_timesteps = ascending[: -(scheduler_order - 1)][::-1].copy() if scheduler_order > 1 else ascending[::-1].copy()
    return ascending, prk_timesteps.astype(np.int64), plms_timesteps.astype(np.int64)


def _set_pndm_state_from_timesteps(
    scheduler,
    timesteps: np.ndarray,
    *,
    device: torch.device,
) -> None:
    ascending, prk_timesteps, plms_timesteps = _pndm_prk_plms_from_anchor_timesteps(scheduler, timesteps)
    scheduler._timesteps = ascending
    scheduler.prk_timesteps = prk_timesteps
    scheduler.plms_timesteps = plms_timesteps
    scheduler.timesteps = torch.from_numpy(np.concatenate([prk_timesteps, plms_timesteps]).astype(np.int64)).to(device)
    scheduler.num_inference_steps = len(ascending)
    scheduler.ets = []
    scheduler.counter = 0
    scheduler.cur_model_output = 0


def _set_scheduler_state_from_timesteps_compatible(
    scheduler,
    timesteps: np.ndarray,
    *,
    device: torch.device,
) -> None:
    if isinstance(scheduler, PNDMScheduler):
        _set_pndm_state_from_timesteps(scheduler, timesteps, device=device)
        return
    _set_scheduler_state_from_timesteps(scheduler, timesteps, device=device)


def _scheduler_prefers_sigma_schedule(scheduler) -> bool:
    return isinstance(scheduler, HeunDiscreteScheduler) or _stork_uses_flow_prediction(scheduler)


def _scheduler_uses_manual_sigma_state(scheduler) -> bool:
    del scheduler
    return False


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


def _terminal_timestep_coordinate(scheduler) -> float:
    timesteps = getattr(scheduler, "timesteps", None)
    if isinstance(timesteps, torch.Tensor) and timesteps.numel() > 0:
        if isinstance(scheduler, (DDIMScheduler, PNDMScheduler)):
            step_count = max(int(getattr(scheduler, "num_inference_steps", int(timesteps.numel()))), 1)
            step = int(getattr(scheduler.config, "num_train_timesteps", 1000)) // step_count
            return float(timesteps.detach().cpu().float().reshape(-1)[-1].item()) - float(step)
        raw_sigmas = getattr(scheduler, "sigmas", None)
        if raw_sigmas is not None:
            sigmas = raw_sigmas.detach().cpu().float().numpy() if isinstance(raw_sigmas, torch.Tensor) else np.asarray(raw_sigmas)
            if len(sigmas) == int(timesteps.numel()) + 1:
                return float(_interp_timesteps_for_sigmas(scheduler, np.asarray([float(sigmas[-1])], dtype=np.float64))[0])
    return 0.0


def _stork_uses_flow_prediction(scheduler) -> bool:
    return isinstance(scheduler, STORKScheduler) and getattr(scheduler, "prediction_type", None) == "flow_prediction"


def _stork_flow_anchor_sigmas(schedule_bundle: ScheduleBundle) -> np.ndarray | None:
    sigmas = None
    if schedule_bundle.sigma_grid is not None:
        sigmas = np.asarray(schedule_bundle.sigma_grid, dtype=np.float64)
    elif schedule_bundle.sigmas is not None:
        sigmas = np.asarray(schedule_bundle.sigmas, dtype=np.float64)
    if sigmas is None:
        return None
    if len(sigmas) > 0 and abs(float(sigmas[-1])) < 1.0e-12:
        sigmas = sigmas[:-1]
    return sigmas.astype(np.float32)


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

    if _stork_uses_flow_prediction(scheduler):
        sigmas = _stork_flow_anchor_sigmas(schedule_bundle)
        if sigmas is None:
            raise ValueError("STORK flow schedules require full schedule sigmas or sigma_grid.")
        return {"sigmas": sigmas.tolist()}

    if isinstance(scheduler, STORKScheduler):
        kwargs: dict[str, list[float] | list[int]] = {}
        timesteps = _bundle_anchor_timesteps(scheduler, schedule_bundle)
        sigmas = _bundle_anchor_sigmas(schedule_bundle)
        if timesteps is not None and scheduler_accepts(scheduler, "timesteps"):
            kwargs["timesteps"] = _schedule_timesteps_arg(scheduler, timesteps)
        if sigmas is not None and scheduler_accepts(scheduler, "sigmas"):
            kwargs["sigmas"] = sigmas.tolist()
        if kwargs:
            return kwargs

    if prefer == "sigmas":
        sigmas = _bundle_anchor_sigmas(schedule_bundle)
        if sigmas is not None and scheduler_accepts(scheduler, "sigmas"):
            return {"sigmas": sigmas.tolist()}

    timesteps = _bundle_anchor_timesteps(scheduler, schedule_bundle)
    if timesteps is not None and scheduler_accepts(scheduler, "timesteps"):
        return {"timesteps": _schedule_timesteps_arg(scheduler, timesteps)}

    if prefer == "timesteps":
        sigmas = _bundle_anchor_sigmas(schedule_bundle)
        if sigmas is not None and scheduler_accepts(scheduler, "sigmas"):
            return {"sigmas": sigmas.tolist()}

    supported = [name for name in ("timesteps", "sigmas") if scheduler_accepts(scheduler, name)]
    supported_str = ", ".join(supported) if supported else "none"
    raise ValueError(f"No compatible schedule field found. Scheduler supports: {supported_str}")


def _apply_stork_runtime_options(scheduler, schedule_bundle: ScheduleBundle | None) -> None:
    if schedule_bundle is None or not isinstance(scheduler, STORKScheduler):
        return
    meta = schedule_bundle.meta or {}
    scheduler.adaptive_s = bool(meta.get("adaptive_s_enabled", meta.get("adaptive_s", False)))
    if "adaptive_s_max" in meta:
        scheduler.adaptive_s_max = int(meta["adaptive_s_max"])
    if "adaptive_s_reference" in meta:
        scheduler.adaptive_s_reference = str(meta["adaptive_s_reference"])


def _configure_scheduler_timesteps(
    scheduler,
    *,
    num_inference_steps: int,
    device: torch.device,
    schedule_bundle: ScheduleBundle | None,
) -> None:
    if schedule_bundle is None:
        scheduler.set_timesteps(num_inference_steps, device=device)
        _force_zero_terminal_sigma(scheduler)
        _move_unipc_sigmas_to_device(scheduler, device)
        return

    _apply_stork_runtime_options(scheduler, schedule_bundle)

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
            _set_scheduler_state_from_timesteps_compatible(scheduler, fallback_timesteps, device=device)
            return
        raise

    try:
        scheduler.set_timesteps(device=device, **schedule_kwargs)
        _force_zero_terminal_sigma(scheduler)
        _move_unipc_sigmas_to_device(scheduler, device)
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
        _set_scheduler_state_from_timesteps_compatible(scheduler, fallback_timesteps, device=device)


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
        )
        return _collapse_repeated_values(
            np.concatenate([anchor_timesteps, np.asarray([_terminal_timestep_coordinate(scheduler)], dtype=np.float64)])
        )

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
        return _collapse_repeated_values(sigma_grid)
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
    """Return the sigma-domain coordinate velocity V = dx / d sigma."""
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


def _evaluate_scheduler_model_output(
    model: torch.nn.Module,
    scheduler,
    sample: torch.Tensor,
    scheduler_timestep,
) -> torch.Tensor:
    device = sample.device
    model_timestep = scheduler_timestep
    if not isinstance(model_timestep, torch.Tensor):
        model_timestep = torch.tensor([model_timestep], device=device)
    if model_timestep.ndim == 0:
        model_timestep = model_timestep[None]
    if model_timestep.numel() == 1:
        model_timestep = model_timestep.expand(sample.shape[0])
    model_input = sample
    if hasattr(scheduler, "scale_model_input"):
        model_input = scheduler.scale_model_input(sample, scheduler_timestep)
    return model(model_input, model_timestep)


def _alpha_cumprod_at_timestep_torch(
    scheduler,
    timestep_value: float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    terminal_if_nonpositive: bool = False,
) -> torch.Tensor:
    alphas = scheduler.alphas_cumprod.detach().to(device=device, dtype=dtype)
    query = torch.as_tensor(timestep_value, device=device, dtype=dtype)
    terminal = torch.as_tensor(float(scheduler.final_alpha_cumprod), device=device, dtype=dtype)
    clipped = torch.clamp(query, min=0.0, max=float(len(alphas) - 1))
    lookup = torch.arange(len(alphas), device=device, dtype=dtype)
    interpolated = _torch_interp_1d(clipped, lookup, alphas)
    if terminal_if_nonpositive:
        interpolated = torch.where(query <= 0.0, terminal, interpolated)
    return interpolated


def _ddim_step_between_timesteps(
    model: torch.nn.Module,
    scheduler,
    sample: torch.Tensor,
    timestep_start: float,
    timestep_end: float,
    *,
    model_output_type: str = "epsilon",
) -> torch.Tensor:
    device = sample.device
    dtype = sample.dtype
    start_tensor = torch.as_tensor(float(timestep_start), device=device, dtype=dtype)
    end_tensor = torch.as_tensor(float(timestep_end), device=device, dtype=dtype)
    model_timestep = start_tensor.to(dtype=torch.float32).reshape(()).expand(sample.shape[0])
    model_input = sample
    if hasattr(scheduler, "scale_model_input"):
        model_input = scheduler.scale_model_input(sample, start_tensor)
    raw_model_output = model(model_input, model_timestep)

    alpha_start = _alpha_cumprod_at_timestep_torch(
        scheduler,
        start_tensor,
        device=device,
        dtype=dtype,
        terminal_if_nonpositive=False,
    )
    alpha_end = _alpha_cumprod_at_timestep_torch(
        scheduler,
        end_tensor,
        device=device,
        dtype=dtype,
        terminal_if_nonpositive=True,
    )
    beta_start = torch.clamp(1.0 - alpha_start, min=0.0)
    normalized_output_type = _normalize_model_output_type(model_output_type)
    if normalized_output_type == "epsilon":
        pred_epsilon = raw_model_output
        pred_original = (sample - beta_start.sqrt() * pred_epsilon) / torch.clamp(alpha_start.sqrt(), min=1.0e-12)
    elif normalized_output_type == "v_prediction":
        pred_original = alpha_start.sqrt() * sample - beta_start.sqrt() * raw_model_output
        pred_epsilon = alpha_start.sqrt() * raw_model_output + beta_start.sqrt() * sample
    else:
        raise ValueError("DDIM PNDM calibration supports epsilon or v_prediction outputs.")

    if getattr(scheduler.config, "thresholding", False):
        pred_original = scheduler._threshold_sample(pred_original)
    elif getattr(scheduler.config, "clip_sample", False):
        pred_original = pred_original.clamp(
            -float(getattr(scheduler.config, "clip_sample_range", 1.0)),
            float(getattr(scheduler.config, "clip_sample_range", 1.0)),
        )
    beta_end = torch.clamp(1.0 - alpha_end, min=0.0)
    return alpha_end.sqrt() * pred_original + beta_end.sqrt() * pred_epsilon


def _build_ddim_stepper(
    *,
    model: torch.nn.Module,
    scheduler,
    model_output_type: str,
):
    def step(
        sample: torch.Tensor,
        coordinate_start: float,
        coordinate_end: float,
        _sample_start: int,
        _sample_stop: int | None,
    ) -> torch.Tensor:
        return _ddim_step_between_timesteps(
            model,
            scheduler,
            sample,
            float(coordinate_start),
            float(coordinate_end),
            model_output_type=model_output_type,
        )

    return step


def _build_native_scheduler_stepper(
    *,
    model: torch.nn.Module,
    scheduler,
    time_from_coordinate: Callable[[float], float],
    sigma_from_coordinate: Callable[[float], float],
    coordinate_domain: str,
) -> Callable[[torch.Tensor, float, float], torch.Tensor]:
    if isinstance(scheduler, DPMSolverMultistepScheduler):
        raise ValueError("Solver-aware defect calibration is disabled for PNDM DPMSolver custom schedules.")

    def step(sample: torch.Tensor, coordinate_start: float, coordinate_end: float) -> torch.Tensor:
        device = sample.device
        timestep_start = float(time_from_coordinate(float(coordinate_start)))
        timestep_end = float(time_from_coordinate(float(coordinate_end)))
        sigma_start = float(sigma_from_coordinate(float(coordinate_start)))
        sigma_end = float(sigma_from_coordinate(float(coordinate_end)))

        kwargs: dict[str, object] = {"num_inference_steps": 2, "device": device}
        if _stork_uses_flow_prediction(scheduler):
            kwargs["sigmas"] = [sigma_start, sigma_end]
        elif isinstance(scheduler, STORKScheduler) and coordinate_domain == "sigmas" and scheduler_accepts(scheduler, "sigmas"):
            kwargs["sigmas"] = [sigma_start, sigma_end]
            if scheduler_accepts(scheduler, "timesteps"):
                kwargs["timesteps"] = _schedule_timesteps_arg(
                    scheduler,
                    np.asarray([timestep_start, timestep_end], dtype=np.float64),
                )
        elif coordinate_domain == "sigmas" and scheduler_accepts(scheduler, "sigmas"):
            kwargs["sigmas"] = [sigma_start, sigma_end]
        elif scheduler_accepts(scheduler, "timesteps"):
            kwargs["timesteps"] = _schedule_timesteps_arg(
                scheduler,
                np.asarray([timestep_start, timestep_end], dtype=np.float64),
            )
        else:
            raise ValueError(f"Scheduler {scheduler.__class__.__name__} does not accept custom refinement nodes.")

        scheduler.set_timesteps(**kwargs)
        _force_zero_terminal_sigma(scheduler)
        scheduler_timestep = scheduler.timesteps[0]
        model_output = _evaluate_scheduler_model_output(model, scheduler, sample, scheduler_timestep)
        step_output = scheduler.step(model_output, scheduler_timestep, sample)
        return step_output.prev_sample

    return step


def _build_stateful_stork_stepper(
    *,
    model: torch.nn.Module,
    scheduler,
) -> Callable[[torch.Tensor, float, float], torch.Tensor]:
    def step(sample: torch.Tensor, _coordinate_start: float, _coordinate_end: float) -> torch.Tensor:
        if scheduler._step_index is None:
            scheduler._step_index = 0
        scheduler_timestep = scheduler.timesteps[scheduler._step_index]
        model_output = _evaluate_scheduler_model_output(model, scheduler, sample, scheduler_timestep)
        step_output = scheduler.step(model_output, scheduler_timestep, sample)
        return step_output.prev_sample

    return step


def _reset_scheduler_history(scheduler) -> None:
    solver_order = int(getattr(getattr(scheduler, "config", None), "solver_order", 1))
    for name, value in {
        "model_outputs": [None] * solver_order,
        "timestep_list": [None] * solver_order,
        "lower_order_nums": 0,
        "last_sample": None,
        "ets": [],
        "counter": 0,
        "cur_model_output": 0,
        "cur_sample": None,
        "_step_index": None,
        "_begin_index": None,
        "noise_predictions": [],
        "velocity_predictions": [],
    }.items():
        if hasattr(scheduler, name):
            setattr(scheduler, name, value)


_REPLAY_HISTORY_NAMES = (
    "model_outputs",
    "timestep_list",
    "lower_order_nums",
    "last_sample",
    "ets",
    "counter",
    "cur_model_output",
    "cur_sample",
    "noise_predictions",
    "velocity_predictions",
    "adaptive_s_requested_per_step",
    "adaptive_s_used_per_step",
    "adaptive_s_requested_max",
    "adaptive_s_used_max",
)


def _clone_scheduler_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_scheduler_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_scheduler_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_scheduler_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _snapshot_scheduler_history(scheduler) -> dict[str, object]:
    return {
        name: _clone_scheduler_value(getattr(scheduler, name))
        for name in _REPLAY_HISTORY_NAMES
        if hasattr(scheduler, name)
    }


def _restore_scheduler_replay_history(scheduler, snapshot: dict[str, object]) -> None:
    for name, value in snapshot.items():
        if hasattr(scheduler, name):
            setattr(scheduler, name, _clone_scheduler_value(value))
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = None
    if hasattr(scheduler, "is_scale_input_called"):
        scheduler.is_scale_input_called = False


def _configured_scheduler_coordinate_nodes(
    scheduler,
    *,
    coordinate_domain: str,
) -> np.ndarray:
    normalized_domain = str(coordinate_domain).lower().strip()
    if normalized_domain == "timestep":
        normalized_domain = "timesteps"
    if normalized_domain == "sigma":
        normalized_domain = "sigmas"
    if normalized_domain == "timesteps":
        timesteps = scheduler.timesteps.detach().cpu().float().numpy()
        return np.concatenate([timesteps.astype(np.float64), np.asarray([_terminal_timestep_coordinate(scheduler)], dtype=np.float64)])
    if normalized_domain == "sigmas":
        raw_sigmas = getattr(scheduler, "sigmas", None)
        if raw_sigmas is None:
            raise ValueError(f"Scheduler {scheduler.__class__.__name__} does not expose sigma schedules.")
        values = raw_sigmas.detach().cpu().float().numpy() if hasattr(raw_sigmas, "detach") else np.asarray(raw_sigmas)
        return np.asarray(values, dtype=np.float64)
    raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")


def _collapse_adjacent_trajectory_nodes(
    coordinate_nodes: np.ndarray,
    states: list[torch.Tensor],
    *,
    eps: float,
) -> tuple[np.ndarray, torch.Tensor]:
    nodes = np.asarray(coordinate_nodes, dtype=np.float64)
    if len(nodes) != len(states):
        raise ValueError("coordinate node count must match recorded trajectory state count.")
    collapsed_nodes: list[float] = []
    collapsed_states: list[torch.Tensor] = []
    for node, state in zip(nodes.tolist(), states):
        if collapsed_nodes and abs(float(node) - collapsed_nodes[-1]) <= float(eps):
            collapsed_nodes[-1] = float(node)
            collapsed_states[-1] = state
            continue
        collapsed_nodes.append(float(node))
        collapsed_states.append(state)
    if len(collapsed_nodes) < 2:
        raise RuntimeError("Recorded trajectory collapsed to fewer than two coordinate nodes.")
    return np.asarray(collapsed_nodes, dtype=np.float64), torch.stack(collapsed_states, dim=0)


def _trajectory_init_sigma(scheduler) -> float:
    init_noise_sigma = getattr(scheduler, "init_noise_sigma", None)
    if init_noise_sigma is not None:
        if hasattr(init_noise_sigma, "detach"):
            return float(init_noise_sigma.detach().cpu().float().reshape(()).item())
        return float(init_noise_sigma)
    raw_sigmas = getattr(scheduler, "sigmas", None)
    if raw_sigmas is not None:
        values = raw_sigmas.detach().cpu().float().numpy() if hasattr(raw_sigmas, "detach") else np.asarray(raw_sigmas)
        return float(np.asarray(values, dtype=np.float64)[0])
    return 1.0


def _refined_window_nodes(window_nodes: np.ndarray, factor: int) -> np.ndarray:
    nodes = np.asarray(window_nodes, dtype=np.float64)
    if nodes.ndim != 1 or len(nodes) < 2:
        raise ValueError("window_nodes must be a 1D array with at least two nodes.")
    if int(factor) <= 0:
        raise ValueError("factor must be positive.")
    refined = [float(nodes[0])]
    for index in range(len(nodes) - 1):
        refined.extend(np.linspace(float(nodes[index]), float(nodes[index + 1]), int(factor) + 1)[1:].tolist())
    return np.asarray(refined, dtype=np.float64)


def _local_scheduler_step_count_from_timesteps(time_grid: np.ndarray, scheduler) -> int:
    values = np.asarray(time_grid, dtype=np.float64)
    gaps = np.abs(np.diff(values))
    gaps = gaps[gaps > 1.0e-8]
    if len(gaps) == 0:
        return max(len(values) - 1, 1)
    train_steps = float(getattr(getattr(scheduler, "config", None), "num_train_timesteps", 1000))
    return max(int(round(train_steps / float(np.median(gaps)))), len(values) - 1, 1)


def _set_stork_replay_state(
    scheduler,
    *,
    time_grid: np.ndarray,
    sigma_grid: np.ndarray,
    device: torch.device,
) -> None:
    start_timesteps = np.asarray(time_grid[:-1], dtype=np.float64)
    if np.any(np.diff(start_timesteps) >= 0):
        raise ValueError("STORK replay timesteps must be strictly descending.")
    if np.any(np.diff(sigma_grid) >= 0):
        raise ValueError("STORK replay sigmas must be strictly descending.")
    total_steps = float(getattr(scheduler.config, "num_train_timesteps", 1000))
    normalized_grid = np.asarray(time_grid, dtype=np.float64) / total_steps
    dt_list = normalized_grid[:-1] - normalized_grid[1:]
    scheduler.num_inference_steps = len(start_timesteps)
    scheduler._timesteps = start_timesteps.astype(np.float32)
    scheduler.timesteps = torch.from_numpy(start_timesteps.astype(np.float32)).to(dtype=scheduler.dtype, device=device)
    scheduler.sigmas = torch.from_numpy(np.asarray(sigma_grid, dtype=np.float32)).to(dtype=scheduler.dtype, device=device)
    scheduler.dt_list = torch.from_numpy(dt_list.astype(np.float32)).to(dtype=scheduler.dtype, device=device)
    scheduler.dt = float(dt_list[0]) if len(dt_list) else 0.0
    scheduler.base_dt = 1.0 / max(int(scheduler.num_inference_steps), 1)
    scheduler.dt_max = float(scheduler.dt_list.max().item()) if len(scheduler.dt_list) else 0.0
    scheduler.dt_min = float(scheduler.dt_list.min().item()) if len(scheduler.dt_list) else 0.0
    scheduler._step_index = None
    scheduler._begin_index = None


def _set_replay_scheduler_nodes(
    scheduler,
    nodes: np.ndarray,
    *,
    coordinate_domain: str,
    device: torch.device,
) -> None:
    grid = np.asarray(nodes, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("Replay grid must contain at least two nodes.")
    if np.any(np.diff(grid) >= 0):
        raise ValueError("Replay grid must be strictly descending.")

    normalized_domain = str(coordinate_domain).lower().strip()
    if normalized_domain == "timestep":
        normalized_domain = "timesteps"
    if normalized_domain == "sigma":
        normalized_domain = "sigmas"

    if normalized_domain == "timesteps":
        time_grid = grid
        sigma_grid = _interp_sigmas_for_timesteps(scheduler, time_grid)
    elif normalized_domain == "sigmas":
        sigma_grid = grid
        time_grid = _interp_timesteps_for_sigmas(
            scheduler,
            sigma_grid,
            round_output=not isinstance(scheduler, STORKScheduler),
            force_log_sigma=isinstance(scheduler, DPMSolverMultistepScheduler),
        )
    else:
        raise ValueError(f"Unsupported replay coordinate domain: {coordinate_domain}")

    start_timesteps = np.asarray(time_grid[:-1], dtype=np.float64)
    step_count = len(start_timesteps)
    if isinstance(scheduler, STORKScheduler):
        _set_stork_replay_state(scheduler, time_grid=time_grid, sigma_grid=sigma_grid, device=device)
        return

    if isinstance(scheduler, PNDMScheduler):
        scheduler.timesteps = torch.from_numpy(np.round(start_timesteps).astype(np.int64)).to(device=device)
        scheduler.prk_timesteps = np.round(start_timesteps).astype(np.int64)
        scheduler.plms_timesteps = np.round(start_timesteps).astype(np.int64)
        scheduler.num_inference_steps = _local_scheduler_step_count_from_timesteps(time_grid, scheduler)
        return

    if isinstance(scheduler, (DDIMScheduler,)):
        scheduler.timesteps = torch.from_numpy(np.round(start_timesteps).astype(np.int64)).to(device=device)
        scheduler.num_inference_steps = _local_scheduler_step_count_from_timesteps(time_grid, scheduler)
        return

    scheduler.timesteps = torch.from_numpy(start_timesteps.astype(np.float32)).to(device=device)
    scheduler.sigmas = torch.from_numpy(np.asarray(sigma_grid, dtype=np.float32)).to("cpu")
    scheduler.num_inference_steps = step_count
    if hasattr(scheduler, "model_outputs"):
        solver_order = int(getattr(getattr(scheduler, "config", None), "solver_order", 1))
        scheduler.model_outputs = [None] * solver_order
    if hasattr(scheduler, "timestep_list"):
        solver_order = int(getattr(getattr(scheduler, "config", None), "solver_order", 1))
        scheduler.timestep_list = [None] * solver_order
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = None
    _move_unipc_sigmas_to_device(scheduler, device)


def _run_pndm_base_trajectory(
    *,
    model: torch.nn.Module,
    scheduler,
    solver: str,
    effective_nfe: int,
    initial_sample: torch.Tensor,
    coordinate_domain: str,
    eps: float,
) -> tuple[np.ndarray, torch.Tensor]:
    device = initial_sample.device
    plan = resolve_effective_nfe_plan(solver, int(effective_nfe))
    scheduler.set_timesteps(plan.solver_steps, device=device)
    _force_zero_terminal_sigma(scheduler)
    _reset_scheduler_history(scheduler)
    coordinate_nodes = _configured_scheduler_coordinate_nodes(
        scheduler,
        coordinate_domain=coordinate_domain,
    )
    states = [initial_sample.detach().clone()]
    sample = initial_sample.detach().clone()
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            model_timestep = timestep
            if not isinstance(model_timestep, torch.Tensor):
                model_timestep = torch.tensor([model_timestep], device=device)
            if model_timestep.ndim == 0:
                model_timestep = model_timestep[None]
            if model_timestep.numel() == 1:
                model_timestep = model_timestep.expand(sample.shape[0])
            model_input = sample
            if hasattr(scheduler, "scale_model_input"):
                model_input = scheduler.scale_model_input(sample, timestep)
            model_output = model(model_input, model_timestep)
            step_output = scheduler.step(model_output, timestep, sample)
            sample = step_output.prev_sample
            states.append(sample.detach().clone())
    return _collapse_adjacent_trajectory_nodes(coordinate_nodes, states, eps=eps)


def _run_pndm_anchor_trajectory(
    *,
    model: torch.nn.Module,
    scheduler,
    solver: str,
    effective_nfe: int,
    initial_sample: torch.Tensor,
    coordinate_domain: str,
    eps: float,
) -> tuple[np.ndarray, torch.Tensor, list[dict[str, object]]]:
    device = initial_sample.device
    plan = resolve_effective_nfe_plan(solver, int(effective_nfe))
    scheduler.set_timesteps(plan.solver_steps, device=device)
    _force_zero_terminal_sigma(scheduler)
    _move_unipc_sigmas_to_device(scheduler, device)
    _reset_scheduler_history(scheduler)
    coordinate_nodes = _configured_scheduler_coordinate_nodes(scheduler, coordinate_domain=coordinate_domain)
    states = [initial_sample.detach().clone()]
    history = [_snapshot_scheduler_history(scheduler)]
    sample = initial_sample.detach().clone()
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            model_output = _evaluate_scheduler_model_output(model, scheduler, sample, timestep)
            step_output = scheduler.step(model_output, timestep, sample)
            sample = step_output.prev_sample
            states.append(sample.detach().clone())
            history.append(_snapshot_scheduler_history(scheduler))
    grid, state_tensor = _collapse_adjacent_trajectory_nodes(coordinate_nodes, states, eps=eps)
    if len(grid) != len(history):
        # Repeated scheduler nodes are collapsed only for state geometry; keep the
        # matching latest history snapshot for each remaining node.
        collapsed_history: list[dict[str, object]] = []
        collapsed_nodes: list[float] = []
        for node, snapshot in zip(np.asarray(coordinate_nodes, dtype=np.float64).tolist(), history):
            if collapsed_nodes and abs(float(node) - collapsed_nodes[-1]) <= float(eps):
                collapsed_nodes[-1] = float(node)
                collapsed_history[-1] = snapshot
            else:
                collapsed_nodes.append(float(node))
                collapsed_history.append(snapshot)
        history = collapsed_history
    return grid, state_tensor, history


def _replay_pndm_window_endpoint(
    *,
    model: torch.nn.Module,
    scheduler,
    coordinate_nodes: np.ndarray,
    coordinate_domain: str,
    anchor_sample: torch.Tensor,
    anchor_history: dict[str, object],
) -> torch.Tensor:
    device = anchor_sample.device
    _set_replay_scheduler_nodes(
        scheduler,
        coordinate_nodes,
        coordinate_domain=coordinate_domain,
        device=device,
    )
    _restore_scheduler_replay_history(scheduler, anchor_history)
    sample = anchor_sample.detach().clone()
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            model_output = _evaluate_scheduler_model_output(model, scheduler, sample, timestep)
            try:
                step_output = scheduler.step(model_output, timestep, sample)
            except (IndexError, ValueError) as error:
                if not isinstance(scheduler, PNDMScheduler):
                    raise
                _reset_scheduler_history(scheduler)
                _set_replay_scheduler_nodes(
                    scheduler,
                    coordinate_nodes,
                    coordinate_domain=coordinate_domain,
                    device=device,
                )
                sample = anchor_sample.detach().clone()
                model_output = _evaluate_scheduler_model_output(model, scheduler, sample, scheduler.timesteps[0])
                step_output = scheduler.step(model_output, scheduler.timesteps[0], sample)
                remaining_timesteps = scheduler.timesteps[1:]
                for retry_timestep in remaining_timesteps:
                    model_output = _evaluate_scheduler_model_output(
                        model,
                        scheduler,
                        step_output.prev_sample,
                        retry_timestep,
                    )
                    step_output = scheduler.step(model_output, retry_timestep, step_output.prev_sample)
                return step_output.prev_sample.detach().clone()
            sample = step_output.prev_sample
    return sample.detach().clone()


def _anchored_replay_cost_per_sample(interval_count: int, window_size: int) -> int:
    return int(interval_count) * (4 + 7 * int(window_size))


def _build_velocity_replay_components(
    *,
    model: torch.nn.Module,
    scheduler,
    coordinate_domain: str,
    model_output_type: str,
    eps: float,
):
    if coordinate_domain == "timesteps":
        velocity_fn = build_velocity_oracle(
            model,
            scheduler,
            model_output_type=model_output_type,
            sigma_floor=eps,
        )
    elif coordinate_domain == "sigmas":
        velocity_fn = build_sigma_derivative_oracle(
            model,
            scheduler,
            model_output_type=model_output_type,
        )
    else:
        raise ValueError(f"Unsupported velocity replay coordinate domain: {coordinate_domain}")
    return velocity_fn, build_velocity_stepper(velocity_fn, "euler")


def _collect_velocity_quarter_anchor_batch(
    *,
    initial_sample: torch.Tensor,
    physical_grid: np.ndarray,
    step_fn,
    window_size: int,
    observation_microbatch: int | None,
    q_min: float,
    q_max: float,
    eps: float,
) -> tuple[FPTrajectoryStats, object]:
    grid = np.asarray(physical_grid, dtype=np.float64)
    current = initial_sample.detach()
    reference_states: list[torch.Tensor] = [current.detach().clone()]
    replay_endpoints: dict[int, list[torch.Tensor]] = {1: [], 2: [], 4: []}
    interval_count = len(grid) - 1

    with torch.inference_mode():
        for index in range(interval_count):
            start = float(grid[index])
            interval_end = float(grid[index + 1])
            stop_index = min(index + int(window_size), interval_count)
            window_end = float(grid[stop_index])

            for factor in (1, 2, 4):
                endpoint = _microbatch_map(
                    current,
                    microbatch_size=observation_microbatch,
                    fn=lambda batch, batch_start, batch_stop, s=start, e=window_end, f=factor: _refined_step(
                        step_fn,
                        batch,
                        s,
                        e,
                        f,
                        batch_start,
                        batch_stop,
                    ),
                )
                replay_endpoints[factor].append(endpoint.detach())

            current = _microbatch_map(
                current,
                microbatch_size=observation_microbatch,
                fn=lambda batch, batch_start, batch_stop, s=start, e=interval_end: _refined_step(
                    step_fn,
                    batch,
                    s,
                    e,
                    4,
                    batch_start,
                    batch_stop,
                ),
            ).detach()
            reference_states.append(current.detach().clone())

    return collect_anchored_replay_stats(
        physical_grid=grid,
        reference_states=torch.stack(reference_states, dim=0),
        replay_1x_endpoints=torch.stack(replay_endpoints[1], dim=0),
        replay_2x_endpoints=torch.stack(replay_endpoints[2], dim=0),
        replay_4x_endpoints=torch.stack(replay_endpoints[4], dim=0),
        window_size=int(window_size),
        q_min=q_min,
        q_max=q_max,
        eps=eps,
    )


def _run_scheduler_quarter_reference_on_grid(
    *,
    model: torch.nn.Module,
    scheduler,
    initial_sample: torch.Tensor,
    coarse_grid: np.ndarray,
    coordinate_domain: str,
    refinement_factor: int,
    eps: float,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    device = initial_sample.device
    fine_grid = _refined_window_nodes(np.asarray(coarse_grid, dtype=np.float64), int(refinement_factor))
    _set_replay_scheduler_nodes(scheduler, fine_grid, coordinate_domain=coordinate_domain, device=device)
    _reset_scheduler_history(scheduler)
    expected_steps = len(fine_grid) - 1
    if len(scheduler.timesteps) != expected_steps:
        raise ValueError(
            f"History-aware anchored replay requires one scheduler step per refined interval; "
            f"got {len(scheduler.timesteps)} scheduler steps for {expected_steps} intervals."
        )

    states: list[torch.Tensor] = [initial_sample.detach().clone()]
    history: list[dict[str, object]] = [_snapshot_scheduler_history(scheduler)]
    sample = initial_sample.detach().clone()
    with torch.inference_mode():
        for step_index, timestep in enumerate(scheduler.timesteps):
            model_output = _evaluate_scheduler_model_output(model, scheduler, sample, timestep)
            step_output = scheduler.step(model_output, timestep, sample)
            sample = step_output.prev_sample
            if (step_index + 1) % int(refinement_factor) == 0:
                states.append(sample.detach().clone())
                history.append(_snapshot_scheduler_history(scheduler))

    if len(states) != len(coarse_grid):
        raise RuntimeError(
            f"Quarter reference recorded {len(states)} coarse states for {len(coarse_grid)} coarse nodes."
        )
    if len(history) != len(coarse_grid):
        raise RuntimeError("Quarter reference history count does not match coarse grid.")
    return torch.stack(states, dim=0), history


def _collect_scheduler_history_quarter_anchor_batch(
    *,
    model: torch.nn.Module,
    scheduler,
    initial_sample: torch.Tensor,
    physical_grid: np.ndarray,
    coordinate_domain: str,
    window_size: int,
    q_min: float,
    q_max: float,
    eps: float,
) -> tuple[FPTrajectoryStats, object]:
    grid = np.asarray(physical_grid, dtype=np.float64)
    reference_states, reference_history = _run_scheduler_quarter_reference_on_grid(
        model=model,
        scheduler=scheduler,
        initial_sample=initial_sample,
        coarse_grid=grid,
        coordinate_domain=coordinate_domain,
        refinement_factor=4,
        eps=eps,
    )
    interval_count = len(grid) - 1
    replay_endpoints: dict[int, list[torch.Tensor]] = {1: [], 2: [], 4: []}
    with torch.inference_mode():
        for start in range(interval_count):
            stop = min(start + int(window_size), interval_count)
            window_nodes = grid[start : stop + 1]
            anchor_sample = reference_states[start]
            anchor_history = reference_history[start]
            for factor in (1, 2, 4):
                replay_nodes = _refined_window_nodes(window_nodes, int(factor))
                endpoint = _replay_pndm_window_endpoint(
                    model=model,
                    scheduler=scheduler,
                    coordinate_nodes=replay_nodes,
                    coordinate_domain=coordinate_domain,
                    anchor_sample=anchor_sample,
                    anchor_history=anchor_history,
                )
                replay_endpoints[factor].append(endpoint.detach())

    return collect_anchored_replay_stats(
        physical_grid=grid,
        reference_states=reference_states,
        replay_1x_endpoints=torch.stack(replay_endpoints[1], dim=0),
        replay_2x_endpoints=torch.stack(replay_endpoints[2], dim=0),
        replay_4x_endpoints=torch.stack(replay_endpoints[4], dim=0),
        window_size=int(window_size),
        q_min=q_min,
        q_max=q_max,
        eps=eps,
    )


def collect_anchored_replay_calibration_stats(
    *,
    model: torch.nn.Module,
    scheduler,
    solver: str,
    image_size: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    anchor_nfe: int = 16,
    window_size: int | None = None,
    observation_microbatch: int | None = None,
    coordinate_domain: str | None = None,
    model_output_type: str = "epsilon",
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, FPTrajectoryStats, dict[str, object]]:
    normalized_solver = normalize_solver_name(solver)
    spec = get_solver_native_spec("pndm", solver)
    if not spec.supports_base_trajectory_recording:
        raise ValueError(f"PNDM solver `{solver}` does not support anchored replay FP calibration: {spec.notes}")
    active_domain = str(coordinate_domain or spec.native_coordinate).lower().strip()
    if active_domain == "timestep":
        active_domain = "timesteps"
    if active_domain == "sigma":
        active_domain = "sigmas"
    if active_domain not in {"timesteps", "sigmas"}:
        raise ValueError(f"Unsupported PNDM anchored replay coordinate domain: {active_domain}")
    active_window = int(window_size or spec.recommended_window_len)
    if active_window < int(spec.solver_order):
        raise ValueError("window_size must be at least the solver order/history length.")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    batches: list[FPTrajectoryStats] = []
    details: list[dict[str, object]] = []
    grid_reference: np.ndarray | None = None
    cost_per_sample: int | None = None

    with torch.inference_mode():
        for _ in range(num_batches):
            scheduler.set_timesteps(resolve_effective_nfe_plan(solver, int(anchor_nfe)).solver_steps, device=device)
            _force_zero_terminal_sigma(scheduler)
            _move_unipc_sigmas_to_device(scheduler, device)
            physical_grid = build_pndm_native_coordinate_grid(
                scheduler,
                solver_name=solver,
                effective_nfe=int(anchor_nfe),
                coordinate_domain=active_domain,
            )
            if grid_reference is None:
                grid_reference = physical_grid
            elif not np.allclose(grid_reference, physical_grid, rtol=0.0, atol=max(float(eps), 1.0e-8)):
                raise RuntimeError("Velocity anchored replay grids changed across calibration batches.")
            init_sigma = _trajectory_init_sigma(scheduler)
            initial_sample = torch.randn(
                (batch_size, model.in_channels, image_size, image_size),
                generator=generator,
                device=device,
            ) * init_sigma

            micro = observation_microbatch if observation_microbatch and observation_microbatch > 0 else batch_size
            micro = min(int(micro), batch_size)
            for start in range(0, batch_size, micro):
                stop = min(start + micro, batch_size)
                sample_slice = initial_sample[start:stop]
                interval_count = len(physical_grid) - 1
                cost_per_sample = _anchored_replay_cost_per_sample(interval_count, active_window)
                if normalized_solver == "euler":
                    _velocity_fn, step_fn = _build_velocity_replay_components(
                        model=model,
                        scheduler=scheduler,
                        coordinate_domain=active_domain,
                        model_output_type=model_output_type,
                        eps=eps,
                    )
                    stats, replay_details = _collect_velocity_quarter_anchor_batch(
                        initial_sample=sample_slice,
                        physical_grid=physical_grid,
                        step_fn=step_fn,
                        window_size=active_window,
                        observation_microbatch=observation_microbatch,
                        q_min=q_min,
                        q_max=q_max,
                        eps=eps,
                    )
                elif normalized_solver == "ddim":
                    if active_domain != "timesteps":
                        raise ValueError("DDIM anchored replay requires timestep-domain coordinates.")
                    step_fn = _build_ddim_stepper(
                        model=model,
                        scheduler=scheduler,
                        model_output_type=model_output_type,
                    )
                    stats, replay_details = _collect_velocity_quarter_anchor_batch(
                        initial_sample=sample_slice,
                        physical_grid=physical_grid,
                        step_fn=step_fn,
                        window_size=1,
                        observation_microbatch=observation_microbatch,
                        q_min=q_min,
                        q_max=q_max,
                        eps=eps,
                    )
                elif normalized_solver == "pndm":
                    raise ValueError(
                        "PNDM/PLMS anchored_replay still needs a custom nonuniform PRK/PLMS production runner; "
                        "the current path does not export a valid FP_CLOCK schedule for pndm."
                    )
                else:
                    stats, replay_details = _collect_scheduler_history_quarter_anchor_batch(
                        model=model,
                        scheduler=scheduler,
                        initial_sample=sample_slice,
                        physical_grid=physical_grid,
                        coordinate_domain=active_domain,
                        window_size=active_window,
                        q_min=q_min,
                        q_max=q_max,
                        eps=eps,
                    )
                batches.append(stats)
                details.append(
                    {
                        "window_size": int(replay_details.window_size),
                        "mean_window_residual_perp_norm": float(np.mean(replay_details.window_residual_perp_norm)),
                        "mean_window_delta_s": float(np.mean(replay_details.window_delta_s)),
                        "mean_window_effective_order": float(np.mean(replay_details.window_effective_order)),
                    }
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if grid_reference is None:
        raise RuntimeError("No PNDM anchored replay calibration batches were collected.")
    stats = concatenate_fp_clock_stats(batches)
    detail_meta = {
        "anchor_nfe": int(anchor_nfe),
        "window_size": int(active_window),
        "window_len": int(active_window),
        "solver_order": int(spec.solver_order),
        "coordinate_domain": active_domain,
        "native_coordinate": active_domain,
        "replay_backend": (
            "velocity_quarter_anchor"
            if normalized_solver == "euler"
            else "ddim_continuous_quarter_anchor"
            if normalized_solver == "ddim"
            else "scheduler_history_quarter_anchor"
        ),
        "reference_path": (
            "quarter_refined_velocity"
            if normalized_solver == "euler"
            else "quarter_refined_ddim"
            if normalized_solver == "ddim"
            else "quarter_refined_target_scheduler"
        ),
        "q_estimator": "full_l2_replay_ratio",
        "residual_metric": "frenet_normal_replay_residual",
        "multistep_history_aware": bool(normalized_solver not in {"euler", "ddim"}),
        "anchored_replay_batch_summaries": details,
        "calibration_cost_per_sample": int(cost_per_sample or 0),
    }
    return grid_reference, stats, detail_meta


def collect_stork_refinement_stats_stateful(
    *,
    model: torch.nn.Module,
    scheduler,
    physical_grid: np.ndarray,
    image_size: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    model_output_type: str = "epsilon",
    coordinate_domain: str = "sigmas",
    observation_microbatch: int | None = None,
    warmup_steps: int = 1,
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> StepRefinementStats:
    del model_output_type
    if str(coordinate_domain).lower().strip() != "sigmas":
        raise ValueError("Stateful STORK refinement stats require sigma-domain physical grids.")

    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must contain at least two sigma nodes.")
    if np.any(np.diff(grid) >= 0):
        raise ValueError("Stateful STORK sigma grids must be strictly descending.")

    n_intervals = len(grid) - 1
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)

    def make_refined_grid(factor: int) -> np.ndarray:
        if factor < 1:
            raise ValueError("refinement factor must be positive.")
        points = [float(grid[0])]
        for index in range(n_intervals):
            subnodes = np.linspace(float(grid[index]), float(grid[index + 1]), factor + 1, dtype=np.float64)[1:]
            points.extend(subnodes.tolist())
        return np.asarray(points, dtype=np.float64)

    def run_trajectory(sigma_grid: np.ndarray, initial_sample: torch.Tensor) -> list[torch.Tensor]:
        scheduler.set_timesteps(device=device, sigmas=sigma_grid.astype(np.float32).tolist())
        scheduler.noise_predictions = []
        scheduler.velocity_predictions = []
        scheduler._step_index = None
        scheduler._begin_index = None

        states = [initial_sample.detach().clone()]
        sample = initial_sample.detach().clone()
        stepper = _build_stateful_stork_stepper(model=model, scheduler=scheduler)
        for index in range(len(scheduler.timesteps)):
            sample = stepper(sample, float(sigma_grid[index]), float(sigma_grid[index + 1]))
            states.append(sample.detach().clone())
        return states

    def trajectory_errors(initial_sample: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        grid_1x = grid
        grid_2x = make_refined_grid(2)
        grid_4x = make_refined_grid(4)

        traj_1x = run_trajectory(grid_1x, initial_sample)
        traj_2x = run_trajectory(grid_2x, initial_sample)
        traj_4x = run_trajectory(grid_4x, initial_sample)

        full_errors = []
        half_errors = []
        for index in range(n_intervals):
            full_errors.append(per_sample_l2_norm(traj_1x[index + 1] - traj_2x[2 * (index + 1)]).cpu().numpy())
            half_errors.append(per_sample_l2_norm(traj_2x[2 * (index + 1)] - traj_4x[4 * (index + 1)]).cpu().numpy())

        full_arr = np.stack(full_errors, axis=1)
        half_arr = np.stack(half_errors, axis=1)

        warmup_count = max(int(warmup_steps), 0)
        if warmup_count > 0 and n_intervals > warmup_count:
            full_arr[:, :warmup_count] = full_arr[:, warmup_count : warmup_count + 1]
            half_arr[:, :warmup_count] = half_arr[:, warmup_count : warmup_count + 1]
        return full_arr, half_arr

    all_full_errors = []
    all_half_errors = []

    with torch.inference_mode():
        for _ in range(num_batches):
            initial_sample = torch.randn(
                (batch_size, model.in_channels, image_size, image_size),
                generator=generator,
                device=device,
            ) * float(grid[0])

            if observation_microbatch is None or observation_microbatch <= 0 or observation_microbatch >= batch_size:
                full_arr, half_arr = trajectory_errors(initial_sample)
                all_full_errors.append(full_arr)
                all_half_errors.append(half_arr)
                continue

            batch_full = []
            batch_half = []
            for start in range(0, batch_size, observation_microbatch):
                stop = min(start + observation_microbatch, batch_size)
                full_arr, half_arr = trajectory_errors(initial_sample[start:stop])
                batch_full.append(full_arr)
                batch_half.append(half_arr)
            all_full_errors.append(np.concatenate(batch_full, axis=0))
            all_half_errors.append(np.concatenate(batch_half, axis=0))

    full_error = np.concatenate(all_full_errors, axis=0)
    half_error = np.concatenate(all_half_errors, axis=0)
    effective_order, defect_strength = estimate_refinement_order_and_defect(
        full_step_error=full_error,
        half_step_error=half_error,
        step_sizes=np.abs(np.diff(grid)),
        q_min=q_min,
        q_max=q_max,
        eps=eps,
    )
    return StepRefinementStats(
        full_step_error=full_error,
        half_step_error=half_error,
        effective_order=effective_order,
        defect_strength=defect_strength,
    )


def collect_solver_refinement_stats(
    *,
    model: torch.nn.Module,
    scheduler,
    physical_grid: np.ndarray,
    solver: str,
    image_size: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    observation_microbatch: int | None = None,
    model_output_type: str = "epsilon",
    sigma_floor: float = 1.0e-6,
    coordinate_domain: str = "timesteps",
    warmup_steps: int = 1,
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> StepRefinementStats:
    normalized_solver = normalize_solver_name(solver)
    if normalized_solver in {"dpm_solver_lu", "dpm_solver_default", "dpm_solver_pp", "dpm_solverpp"}:
        raise ValueError("Solver-aware defect calibration is disabled for PNDM DPMSolver custom schedules.")
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must contain at least two points.")

    normalized_domain = str(coordinate_domain).lower().strip()
    if normalized_domain not in {"timesteps", "sigmas"}:
        raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")

    if normalized_solver in STORK_PNDM_SOLVERS:
        if normalized_domain != "sigmas":
            raise ValueError("Stateful STORK defect calibration requires sigma-domain physical grids.")
        return collect_stork_refinement_stats_stateful(
            model=model,
            scheduler=scheduler,
            physical_grid=grid,
            image_size=image_size,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=seed,
            model_output_type=model_output_type,
            coordinate_domain=normalized_domain,
            observation_microbatch=observation_microbatch,
            warmup_steps=warmup_steps,
            q_min=q_min,
            q_max=q_max,
            eps=eps,
        )

    if normalized_domain == "timesteps":
        time_grid = grid
        sigma_grid = _interp_sigmas_for_timesteps(scheduler, grid)
        sigma_grid[-1] = 0.0
        time_from_coordinate = lambda value: float(value)
        sigma_from_coordinate = lambda value: float(
            _interp_sigmas_for_timesteps(scheduler, np.asarray([value], dtype=np.float64))[0]
        )
    else:
        sigma_grid = grid
        time_grid = _interp_timesteps_for_sigmas(scheduler, sigma_grid)
        time_from_coordinate = lambda value: float(
            _interp_timesteps_for_sigmas(scheduler, np.asarray([value], dtype=np.float64))[0]
        )
        sigma_from_coordinate = lambda value: float(value)

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    batches: list[StepRefinementStats] = []
    if normalized_solver == "euler":
        if normalized_domain == "timesteps":
            velocity_fn = build_velocity_oracle(
                model,
                scheduler,
                model_output_type=model_output_type,
                sigma_floor=sigma_floor,
            )
        else:
            velocity_fn = build_sigma_derivative_oracle(
                model,
                scheduler,
                model_output_type=model_output_type,
            )
        step_fn = build_velocity_stepper(velocity_fn, "euler")
    elif normalized_solver == "heun2":
        if normalized_domain == "timesteps":
            velocity_fn = build_velocity_oracle(
                model,
                scheduler,
                model_output_type=model_output_type,
                sigma_floor=sigma_floor,
            )
        else:
            velocity_fn = build_sigma_derivative_oracle(
                model,
                scheduler,
                model_output_type=model_output_type,
            )
        step_fn = build_velocity_stepper(velocity_fn, "heun2")
    else:
        step_fn = _build_native_scheduler_stepper(
            model=model,
            scheduler=scheduler,
            time_from_coordinate=time_from_coordinate,
            sigma_from_coordinate=sigma_from_coordinate,
            coordinate_domain=normalized_domain,
        )

    with torch.inference_mode():
        for _ in range(num_batches):
            sample = torch.randn(
                (batch_size, model.in_channels, image_size, image_size),
                generator=generator,
                device=device,
            ) * float(sigma_grid[0])
            batches.append(
                collect_step_refinement_stats(
                    initial_sample=sample,
                    physical_grid=grid,
                    step_fn=step_fn,
                    observation_microbatch=observation_microbatch,
                    q_min=q_min,
                    q_max=q_max,
                    eps=eps,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return StepRefinementStats(
        full_step_error=np.concatenate([item.full_step_error for item in batches], axis=0),
        half_step_error=np.concatenate([item.half_step_error for item in batches], axis=0),
        effective_order=np.concatenate([item.effective_order for item in batches], axis=0),
        defect_strength=np.concatenate([item.defect_strength for item in batches], axis=0),
    )


def collect_fp_clock_calibration_stats(
    *,
    model: torch.nn.Module,
    scheduler,
    physical_grid: np.ndarray,
    solver: str,
    image_size: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    observation_microbatch: int | None = None,
    model_output_type: str = "epsilon",
    sigma_floor: float = 1.0e-6,
    coordinate_domain: str = "timesteps",
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> FPTrajectoryStats:
    normalized_solver = normalize_solver_name(solver)
    if normalized_solver in {"dpm_solver_lu", "dpm_solver_default", "dpm_solver_pp", "dpm_solverpp"}:
        raise ValueError("FP_CLOCK calibration is disabled for PNDM DPMSolver custom schedules.")
    if normalized_solver in STORK_PNDM_SOLVERS:
        raise ValueError("FP_CLOCK vector calibration is not implemented for stateful PNDM STORK solvers.")

    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must contain at least two points.")

    normalized_domain = str(coordinate_domain).lower().strip()
    if normalized_domain not in {"timesteps", "sigmas"}:
        raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")

    if normalized_domain == "timesteps":
        sigma_grid = _interp_sigmas_for_timesteps(scheduler, grid)
        sigma_grid[-1] = 0.0
        time_from_coordinate = lambda value: float(value)
        sigma_from_coordinate = lambda value: float(
            _interp_sigmas_for_timesteps(scheduler, np.asarray([value], dtype=np.float64))[0]
        )
        velocity_fn = build_velocity_oracle(
            model,
            scheduler,
            model_output_type=model_output_type,
            sigma_floor=sigma_floor,
        )
    else:
        sigma_grid = grid
        time_from_coordinate = lambda value: float(
            _interp_timesteps_for_sigmas(scheduler, np.asarray([value], dtype=np.float64))[0]
        )
        sigma_from_coordinate = lambda value: float(value)
        velocity_fn = build_sigma_derivative_oracle(
            model,
            scheduler,
            model_output_type=model_output_type,
        )

    if normalized_solver == "euler":
        step_fn = build_velocity_stepper(velocity_fn, "euler")
    elif normalized_solver == "heun2":
        step_fn = build_velocity_stepper(velocity_fn, "heun2")
    else:
        step_fn = _build_native_scheduler_stepper(
            model=model,
            scheduler=scheduler,
            time_from_coordinate=time_from_coordinate,
            sigma_from_coordinate=sigma_from_coordinate,
            coordinate_domain=normalized_domain,
        )

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    batches: list[FPTrajectoryStats] = []
    with torch.inference_mode():
        for _ in range(num_batches):
            sample = torch.randn(
                (batch_size, model.in_channels, image_size, image_size),
                generator=generator,
                device=device,
            ) * float(sigma_grid[0])
            batches.append(
                collect_fp_clock_stats(
                    initial_sample=sample,
                    physical_grid=grid,
                    velocity_fn=velocity_fn,
                    step_fn=step_fn,
                    observation_microbatch=observation_microbatch,
                    q_min=q_min,
                    q_max=q_max,
                    eps=eps,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return concatenate_fp_clock_stats(batches)


def collect_velocity_curvature_calibration_stats(
    *,
    model: torch.nn.Module,
    scheduler,
    physical_grid: np.ndarray,
    image_size: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    observation_microbatch: int | None = None,
    model_output_type: str = "epsilon",
    sigma_floor: float = 1.0e-6,
    coordinate_domain: str = "timesteps",
    pilot_solver: str = "heun2",
    pilot_pieces: int = 4,
    q_const: float = 3.0,
    eps: float = 1.0e-12,
    defect_clip_quantile: float | None = None,
) -> StepRefinementStats:
    grid = np.asarray(physical_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("physical_grid must contain at least two points.")

    normalized_domain = str(coordinate_domain).lower().strip()
    if normalized_domain not in {"timesteps", "sigmas"}:
        raise ValueError(f"Unsupported PNDM coordinate domain: {coordinate_domain}")

    if normalized_domain == "timesteps":
        sigma_grid = _interp_sigmas_for_timesteps(scheduler, grid)
        sigma_grid[-1] = 0.0
        velocity_fn = build_velocity_oracle(
            model,
            scheduler,
            model_output_type=model_output_type,
            sigma_floor=sigma_floor,
        )
    else:
        sigma_grid = grid
        velocity_fn = build_sigma_derivative_oracle(
            model,
            scheduler,
            model_output_type=model_output_type,
        )

    step_fn = build_velocity_stepper(velocity_fn, pilot_solver)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    batches: list[StepRefinementStats] = []
    with torch.inference_mode():
        for _ in range(num_batches):
            sample = torch.randn(
                (batch_size, model.in_channels, image_size, image_size),
                generator=generator,
                device=device,
            ) * float(sigma_grid[0])
            batches.append(
                collect_velocity_curvature_stats(
                    initial_sample=sample,
                    physical_grid=grid,
                    velocity_fn=velocity_fn,
                    pilot_step_fn=step_fn,
                    pilot_pieces=pilot_pieces,
                    observation_microbatch=observation_microbatch,
                    q_const=q_const,
                    eps=eps,
                    defect_clip_quantile=defect_clip_quantile,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return StepRefinementStats(
        full_step_error=np.concatenate([item.full_step_error for item in batches], axis=0),
        half_step_error=np.concatenate([item.half_step_error for item in batches], axis=0),
        effective_order=np.concatenate([item.effective_order for item in batches], axis=0),
        defect_strength=np.concatenate([item.defect_strength for item in batches], axis=0),
    )


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


def _run_custom_ddim(
    *,
    model: torch.nn.Module,
    scheduler,
    latents: torch.Tensor,
    schedule_bundle: ScheduleBundle,
) -> torch.Tensor:
    if schedule_bundle.time_grid is None:
        raise ValueError("Custom DDIM execution requires a full time_grid.")
    time_grid = np.asarray(schedule_bundle.time_grid, dtype=np.float64)
    if time_grid.ndim != 1 or len(time_grid) < 2:
        raise ValueError("Custom DDIM time_grid must contain at least two nodes.")
    if np.any(np.diff(time_grid) >= 0.0):
        raise ValueError("Custom DDIM time_grid must be strictly descending.")
    model_output_type = str(
        schedule_bundle.meta.get(
            "clock_model_output_type",
            schedule_bundle.meta.get("model_output_type", getattr(scheduler.config, "prediction_type", "epsilon")),
        )
    )
    scheduler.num_inference_steps = len(time_grid) - 1
    scheduler.timesteps = torch.from_numpy(time_grid[:-1].astype(np.float32)).to(device=latents.device)
    image = latents
    for index in range(len(time_grid) - 1):
        image = _ddim_step_between_timesteps(
            model,
            scheduler,
            image,
            float(time_grid[index]),
            float(time_grid[index + 1]),
            model_output_type=model_output_type,
        )
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

        if isinstance(self.scheduler, DDIMScheduler) and schedule_bundle is not None:
            return _run_custom_ddim(
                model=self.model,
                scheduler=self.scheduler,
                latents=latents,
                schedule_bundle=schedule_bundle,
            )

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

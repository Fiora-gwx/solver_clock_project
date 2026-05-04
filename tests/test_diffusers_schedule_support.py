import inspect

import numpy as np
import torch

from src.adapters.diffusers import (
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    _attach_flow_native_sigmas,
)


def test_flow_euler_custom_sigmas_are_native_not_shifted() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    _attach_flow_native_sigmas(scheduler)

    native_sigmas = np.asarray([0.9, 0.4, 0.1], dtype=np.float32)
    scheduler.set_timesteps(sigmas=native_sigmas, device=torch.device("cpu"))

    assert np.allclose(scheduler.sigmas.detach().cpu().numpy(), np.asarray([0.9, 0.4, 0.1, 0.0], dtype=np.float32))
    assert np.allclose(scheduler.timesteps.detach().cpu().numpy(), np.asarray([900.0, 400.0, 100.0], dtype=np.float32))


def test_flow_dpm_wrapper_exposes_and_applies_custom_sigmas() -> None:
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
    )
    _attach_flow_native_sigmas(scheduler)

    assert "sigmas" in inspect.signature(scheduler.set_timesteps).parameters
    native_sigmas = np.asarray([0.9, 0.4, 0.1], dtype=np.float32)
    scheduler.set_timesteps(sigmas=native_sigmas, device=torch.device("cpu"))

    assert scheduler.num_inference_steps == 3
    assert np.allclose(scheduler.sigmas.detach().cpu().numpy(), np.asarray([0.9, 0.4, 0.1, 0.0], dtype=np.float32))
    assert np.allclose(scheduler.timesteps.detach().cpu().numpy(), np.asarray([900, 400, 100], dtype=np.int64))


def test_flow_unipc_custom_sigmas_stay_on_requested_device() -> None:
    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
    )
    _attach_flow_native_sigmas(scheduler)

    native_sigmas = np.asarray([0.9, 0.4, 0.1], dtype=np.float32)
    scheduler.set_timesteps(sigmas=native_sigmas, device=torch.device("cpu"))

    assert scheduler.sigmas.device.type == "cpu"
    assert np.allclose(scheduler.sigmas.detach().cpu().numpy(), np.asarray([0.9, 0.4, 0.1, 0.0], dtype=np.float32))
    assert np.allclose(scheduler.timesteps.detach().cpu().numpy(), np.asarray([900, 400, 100], dtype=np.int64))

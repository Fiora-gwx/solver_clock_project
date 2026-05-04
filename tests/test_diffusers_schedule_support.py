import inspect

import numpy as np
import torch

from src.adapters.diffusers import (
    DPMSolverMultistepScheduler,
    EDMDPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    STORKScheduler,
    UniPCMultistepScheduler,
    _attach_flow_native_sigmas,
    build_pipeline_kwargs,
    replace_scheduler,
)
from src.utils.schedule_bundle import ScheduleBundle


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


class StableDiffusionPipeline:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(
        self,
        prompt,
        num_inference_steps,
        generator,
        output_type,
        height=None,
        width=None,
        guidance_scale=None,
        timesteps=None,
        sigmas=None,
    ):
        raise RuntimeError("This dummy pipeline should only be inspected.")


def test_replace_scheduler_registers_vp_history_solver_adapters() -> None:
    base_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        prediction_type="epsilon",
    )

    for solver_name, expected_type in (
        ("unipc", UniPCMultistepScheduler),
        ("stork4_1st", STORKScheduler),
        ("edm_euler", EDMEulerScheduler),
        ("edm_dpm_solver_pp", EDMDPMSolverMultistepScheduler),
    ):
        pipeline = StableDiffusionPipeline(base_scheduler)
        replace_scheduler(pipeline, solver_name)
        assert isinstance(pipeline.scheduler, expected_type)
        if solver_name.startswith("edm_"):
            assert "sigmas" in inspect.signature(pipeline.scheduler.set_timesteps).parameters

        pipeline.scheduler.set_timesteps(
            sigmas=np.asarray([1.0, 0.5, 0.1], dtype=np.float32),
            device=torch.device("cpu"),
        )
        assert len(pipeline.scheduler.timesteps) == 3


def test_pipeline_kwargs_choose_scheduler_compatible_custom_schedule_field() -> None:
    bundle = ScheduleBundle(
        timesteps=np.asarray([900, 500, 100], dtype=np.float64),
        sigmas=np.asarray([1.0, 0.5, 0.1], dtype=np.float64),
        sigma_grid=np.asarray([1.0, 0.5, 0.1, 0.0], dtype=np.float64),
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    dpm_pipeline = StableDiffusionPipeline(
        DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            algorithm_type="dpmsolver++",
            prediction_type="epsilon",
        )
    )

    dpm_kwargs = build_pipeline_kwargs(
        dpm_pipeline,
        prompt="test",
        num_inference_steps=3,
        schedule_bundle=bundle,
        height=64,
        width=64,
        guidance_scale=7.5,
        generator=generator,
    )
    assert "timesteps" in dpm_kwargs
    assert "sigmas" not in dpm_kwargs

    unipc_pipeline = StableDiffusionPipeline(dpm_pipeline.scheduler)
    replace_scheduler(unipc_pipeline, "unipc")
    unipc_kwargs = build_pipeline_kwargs(
        unipc_pipeline,
        prompt="test",
        num_inference_steps=3,
        schedule_bundle=bundle,
        height=64,
        width=64,
        guidance_scale=7.5,
        generator=generator,
    )
    assert "sigmas" in unipc_kwargs
    assert "timesteps" not in unipc_kwargs
    assert len(unipc_kwargs["sigmas"]) == 4

    euler_pipeline = StableDiffusionPipeline(
        EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )
    )
    euler_kwargs = build_pipeline_kwargs(
        euler_pipeline,
        prompt="test",
        num_inference_steps=3,
        schedule_bundle=bundle,
        height=64,
        width=64,
        guidance_scale=7.5,
        generator=generator,
    )
    assert "sigmas" in euler_kwargs
    assert "timesteps" not in euler_kwargs
    assert len(euler_kwargs["sigmas"]) == 4

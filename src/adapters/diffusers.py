from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from src.clock.calibration import ForwardNormCollector
from src.clock.fp_clock import FPTrajectoryStats, collect_trajectory_window_stats, concatenate_fp_clock_stats
from src.clock.solver_registry import get_solver_native_spec
from src.utils.config import repo_root
from src.utils.nfe_budget import resolve_effective_nfe_plan
from src.utils.schedule_bundle import ScheduleBundle


def _ensure_local_diffusers() -> None:
    diffusers_src = repo_root() / "third_party" / "diffusers" / "src"
    stork_root = repo_root() / "third_party" / "STORK"
    for path in (str(diffusers_src), str(stork_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_local_diffusers()

from diffusers import (  # type: ignore  # noqa: E402
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
    UniPCMultistepScheduler,
)
from STORKScheduler import STORKScheduler  # type: ignore  # noqa: E402


@dataclass(frozen=True)
class DiffusersDefectBatch:
    initial_latents: torch.Tensor
    sigma_max: float
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


def normalize_solver_name(name: str) -> str:
    return name.lower().replace("-", "_")


def load_pipeline(model_path: str | Path, *, device: str, dtype_name: str = "bfloat16"):
    pipeline = DiffusionPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype_from_name(dtype_name),
        local_files_only=True,
    )
    pipeline.to(device)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    return pipeline


def get_pipeline_device(pipeline) -> torch.device:
    module = find_denoiser_module(pipeline)
    return next(module.parameters()).device


def find_denoiser_module(pipeline) -> torch.nn.Module:
    for attribute in ("transformer", "unet"):
        if hasattr(pipeline, attribute):
            return getattr(pipeline, attribute)
    raise AttributeError(f"Could not find a denoiser module on pipeline {pipeline.__class__.__name__}")


def _signature_parameters(pipeline) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(pipeline.__call__).parameters)


def _extract_scheduler_sigmas(scheduler, target_steps: int) -> list[float] | None:
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        return None
    values = sigmas.detach().cpu().float().numpy().tolist() if isinstance(sigmas, torch.Tensor) else list(sigmas)
    if len(values) == target_steps + 1:
        values = values[:-1]
    if len(values) > target_steps:
        values = values[:target_steps]
    return values if len(values) == target_steps else None


def compute_dynamic_mu(pipeline, *, height: int, width: int) -> float:
    config = pipeline.scheduler.config
    vae_scale_factor = getattr(pipeline, "vae_scale_factor", 8)
    image_seq_len = (height // vae_scale_factor) * (width // vae_scale_factor)
    base_seq_len = getattr(config, "base_image_seq_len", 256)
    max_seq_len = getattr(config, "max_image_seq_len", 4096)
    base_shift = getattr(config, "base_shift", 0.5)
    max_shift = getattr(config, "max_shift", 1.15)
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def _pipeline_kind(pipeline) -> str:
    name = pipeline.__class__.__name__.lower()
    if name == "ifpipeline":
        return "deepfloyd_if"
    if "flux" in name:
        return "flux"
    if "stablediffusion3" in name:
        return "sd3"
    if "stablediffusionxl" in name:
        return "sdxl"
    if "stablediffusion" in name:
        return "stable_diffusion"
    if "lumina2" in name:
        return "lumina2"
    raise ValueError(f"Unsupported diffusers pipeline for defect calibration: {pipeline.__class__.__name__}")


def _default_max_sequence_length(pipeline) -> int:
    parameters = _signature_parameters(pipeline)
    default = parameters.get("max_sequence_length", inspect.Parameter("max_sequence_length", inspect.Parameter.KEYWORD_ONLY, default=256)).default
    return int(default) if isinstance(default, int) else 256


def _scheduler_mu_kwargs(pipeline, *, height: int, width: int) -> dict[str, float]:
    if getattr(pipeline.scheduler.config, "use_dynamic_shifting", False):
        return {"mu": compute_dynamic_mu(pipeline, height=height, width=width)}
    return {}


def _slice_batch_tensor(tensor: torch.Tensor | None, batch_size: int, batch_start: int = 0) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 0:
        return tensor
    if tensor.shape[0] == batch_size and batch_start == 0:
        return tensor
    batch_stop = int(batch_start) + int(batch_size)
    if tensor.shape[0] >= batch_stop:
        return tensor[int(batch_start) : batch_stop]
    if tensor.shape[0] == 1:
        expand_shape = (batch_size, *tensor.shape[1:])
        return tensor.expand(*expand_shape)
    raise ValueError(f"Cannot adapt tensor with leading dimension {tensor.shape[0]} to batch size {batch_size}.")


def build_defect_sigma_grid(
    pipeline,
    *,
    physical_grid_size: int,
    height: int,
    width: int,
    physical_grid_mode: str = "scheduler_sigmas",
) -> np.ndarray:
    if physical_grid_size < 2:
        raise ValueError("physical_grid_size must be at least 2.")
    device = get_pipeline_device(pipeline)
    scheduler_kwargs = _scheduler_mu_kwargs(pipeline, height=height, width=width)
    pipeline.scheduler.set_timesteps(physical_grid_size - 1, device=device, **scheduler_kwargs)
    raw_sigmas = getattr(pipeline.scheduler, "sigmas", None)
    if raw_sigmas is None:
        raise RuntimeError("The selected diffusers scheduler does not expose a sigma sequence for defect calibration.")
    sigma_tensor = raw_sigmas.detach().float().cpu() if isinstance(raw_sigmas, torch.Tensor) else torch.tensor(raw_sigmas, dtype=torch.float32)
    scheduler_sigmas = sigma_tensor.numpy().astype(np.float64)
    positive_sigmas = scheduler_sigmas[scheduler_sigmas > 0.0]
    if len(positive_sigmas) == 0:
        raise RuntimeError("The selected diffusers scheduler produced no positive sigma values for defect calibration.")
    sigma_max = float(positive_sigmas[0])
    sigma_min = float(positive_sigmas[-1])
    mode = str(physical_grid_mode).lower()
    if mode == "linear_sigma":
        sigma_values = np.linspace(sigma_max, 0.0, physical_grid_size, dtype=np.float64)
    elif mode == "scheduler_sigmas":
        if len(scheduler_sigmas) >= physical_grid_size:
            sigma_values = scheduler_sigmas[:physical_grid_size].astype(np.float64)
        else:
            sigma_values = np.concatenate([positive_sigmas, np.asarray([0.0], dtype=np.float64)])
        if len(sigma_values) != physical_grid_size:
            x_old = np.linspace(0.0, 1.0, len(sigma_values), dtype=np.float64)
            x_new = np.linspace(0.0, 1.0, physical_grid_size, dtype=np.float64)
            sigma_values = np.interp(x_new, x_old, sigma_values)
        sigma_values[-1] = 0.0
    elif mode == "log_sigma":
        sigma_values = np.concatenate(
            [
                np.exp(np.linspace(math.log(sigma_max), math.log(max(sigma_min, 1.0e-12)), physical_grid_size - 1)),
                np.asarray([0.0], dtype=np.float64),
            ]
        )
    elif mode == "karras_sigma":
        rho = 7.0
        ramp = np.linspace(0.0, 1.0, physical_grid_size - 1, dtype=np.float64)
        min_inv_rho = max(sigma_min, 1.0e-12) ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        sigma_values = np.concatenate(
            [
                (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho,
                np.asarray([0.0], dtype=np.float64),
            ]
        )
    else:
        raise ValueError(
            "physical_grid_mode must be one of: linear_sigma, scheduler_sigmas, log_sigma, karras_sigma."
        )
    if len(sigma_values) != physical_grid_size:
        raise RuntimeError(f"Expected physical grid length {physical_grid_size}, got {len(sigma_values)}.")
    if np.any(np.diff(sigma_values) > 1.0e-8):
        raise RuntimeError("Defect sigma grid must be non-increasing.")
    sigma_values[-1] = 0.0
    return sigma_values.astype(np.float64)


def _prepare_flux_defect_batch(
    pipeline,
    *,
    prompt: str | list[str],
    batch_size: int,
    generator: torch.Generator,
    height: int,
    width: int,
    guidance_scale: float,
) -> DiffusersDefectBatch:
    device = get_pipeline_device(pipeline)
    max_sequence_length = _default_max_sequence_length(pipeline)
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    latents, latent_image_ids = pipeline.prepare_latents(
        batch_size,
        pipeline.transformer.config.in_channels // 4,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    scheduler_kwargs = _scheduler_mu_kwargs(pipeline, height=height, width=width)
    sigma_grid = build_defect_sigma_grid(pipeline, physical_grid_size=3, height=height, width=width)
    sigma_max = float(sigma_grid[0])
    guidance = None
    if getattr(pipeline.transformer.config, "guidance_embeds", False):
        guidance = torch.full((batch_size,), guidance_scale, device=device, dtype=torch.float32)

    def velocity_fn(
        current_latents: torch.Tensor,
        sigma: torch.Tensor,
        batch_start: int = 0,
        batch_stop: int | None = None,
    ) -> torch.Tensor:
        del scheduler_kwargs
        batch = current_latents.shape[0]
        timestep = (sigma.reshape(()).to(device=current_latents.device, dtype=current_latents.dtype) * float(pipeline.scheduler.config.num_train_timesteps))
        timestep = timestep.expand(current_latents.shape[0]) / float(pipeline.scheduler.config.num_train_timesteps)
        return pipeline.transformer(
            hidden_states=current_latents,
            timestep=timestep,
            guidance=_slice_batch_tensor(guidance, batch, batch_start),
            pooled_projections=_slice_batch_tensor(pooled_prompt_embeds, batch, batch_start),
            encoder_hidden_states=_slice_batch_tensor(prompt_embeds, batch, batch_start),
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return DiffusersDefectBatch(initial_latents=latents.detach(), sigma_max=sigma_max, velocity_fn=velocity_fn)


def _prepare_sd3_defect_batch(
    pipeline,
    *,
    prompt: str | list[str],
    batch_size: int,
    generator: torch.Generator,
    height: int,
    width: int,
    guidance_scale: float,
) -> DiffusersDefectBatch:
    device = get_pipeline_device(pipeline)
    max_sequence_length = _default_max_sequence_length(pipeline)
    do_cfg = guidance_scale > 1.0
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        do_classifier_free_guidance=do_cfg,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        device=device,
        clip_skip=None,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    latents = pipeline.prepare_latents(
        batch_size,
        pipeline.transformer.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    sigma_max = float(build_defect_sigma_grid(pipeline, physical_grid_size=3, height=height, width=width)[0])

    def velocity_fn(
        current_latents: torch.Tensor,
        sigma: torch.Tensor,
        batch_start: int = 0,
        batch_stop: int | None = None,
    ) -> torch.Tensor:
        batch = current_latents.shape[0]
        timestep_scalar = sigma.reshape(()).to(device=current_latents.device, dtype=current_latents.dtype)
        timestep_scalar = timestep_scalar * float(pipeline.scheduler.config.num_train_timesteps)
        if do_cfg:
            latent_input = torch.cat([current_latents, current_latents], dim=0)
            timestep = timestep_scalar.expand(latent_input.shape[0])
            noise_pred = pipeline.transformer(
                hidden_states=latent_input,
                timestep=timestep,
                encoder_hidden_states=torch.cat(
                    [
                        _slice_batch_tensor(negative_prompt_embeds, batch, batch_start),
                        _slice_batch_tensor(prompt_embeds, batch, batch_start),
                    ],
                    dim=0,
                ),
                pooled_projections=torch.cat(
                    [
                        _slice_batch_tensor(negative_pooled_prompt_embeds, batch, batch_start),
                        _slice_batch_tensor(pooled_prompt_embeds, batch, batch_start),
                    ],
                    dim=0,
                ),
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        timestep = timestep_scalar.expand(current_latents.shape[0])
        return pipeline.transformer(
            hidden_states=current_latents,
            timestep=timestep,
            encoder_hidden_states=_slice_batch_tensor(prompt_embeds, batch, batch_start),
            pooled_projections=_slice_batch_tensor(pooled_prompt_embeds, batch, batch_start),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return DiffusersDefectBatch(initial_latents=latents.detach(), sigma_max=sigma_max, velocity_fn=velocity_fn)


def _prepare_lumina2_defect_batch(
    pipeline,
    *,
    prompt: str | list[str],
    batch_size: int,
    generator: torch.Generator,
    height: int,
    width: int,
    guidance_scale: float,
) -> DiffusersDefectBatch:
    device = get_pipeline_device(pipeline)
    max_sequence_length = _default_max_sequence_length(pipeline)
    do_cfg = guidance_scale > 1.0
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipeline.encode_prompt(
        prompt,
        do_cfg,
        negative_prompt=None,
        num_images_per_prompt=1,
        device=device,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        max_sequence_length=max_sequence_length,
        system_prompt=None,
    )
    latents = pipeline.prepare_latents(
        batch_size,
        pipeline.transformer.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    sigma_max = float(build_defect_sigma_grid(pipeline, physical_grid_size=3, height=height, width=width)[0])

    def velocity_fn(
        current_latents: torch.Tensor,
        sigma: torch.Tensor,
        batch_start: int = 0,
        batch_stop: int | None = None,
    ) -> torch.Tensor:
        batch = current_latents.shape[0]
        current_timestep = 1.0 - sigma.reshape(()).to(device=current_latents.device, dtype=current_latents.dtype)
        current_timestep = current_timestep.expand(current_latents.shape[0])
        noise_pred_cond = pipeline.transformer(
            hidden_states=current_latents,
            timestep=current_timestep,
            encoder_hidden_states=_slice_batch_tensor(prompt_embeds, batch, batch_start),
            encoder_attention_mask=_slice_batch_tensor(prompt_attention_mask, batch, batch_start),
            return_dict=False,
            attention_kwargs=None,
        )[0]
        if not do_cfg:
            return -noise_pred_cond

        noise_pred_uncond = pipeline.transformer(
            hidden_states=current_latents,
            timestep=current_timestep,
            encoder_hidden_states=_slice_batch_tensor(negative_prompt_embeds, batch, batch_start),
            encoder_attention_mask=_slice_batch_tensor(negative_prompt_attention_mask, batch, batch_start),
            return_dict=False,
            attention_kwargs=None,
        )[0]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_pred = noise_pred * (cond_norm / torch.clamp(noise_norm, min=1.0e-12))
        return -noise_pred

    return DiffusersDefectBatch(initial_latents=latents.detach(), sigma_max=sigma_max, velocity_fn=velocity_fn)


def _scheduler_sigma_values(scheduler) -> torch.Tensor:
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        raise RuntimeError("The selected diffusers scheduler does not expose sigmas for defect calibration.")
    return sigmas.detach().float().cpu() if isinstance(sigmas, torch.Tensor) else torch.tensor(sigmas, dtype=torch.float32)


def _vp_timestep_from_sigma(scheduler, sigma: torch.Tensor) -> float:
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError(f"Scheduler {scheduler.__class__.__name__} does not expose alphas_cumprod.")
    alphas = scheduler.alphas_cumprod.detach().float().cpu().numpy()
    train_sigmas = np.sqrt(np.clip(1.0 - alphas, 0.0, None) / np.clip(alphas, 1.0e-12, None))
    sigma_value = max(float(sigma.detach().float().reshape(()).cpu().item()), 1.0e-10)
    return float(
        np.interp(
            np.log(sigma_value),
            np.log(np.clip(train_sigmas, 1.0e-10, None)),
            np.arange(len(train_sigmas), dtype=np.float64),
        )
    )


def _prepare_sd_defect_latents(
    pipeline,
    *,
    batch_size: int,
    generator: torch.Generator,
    height: int,
    width: int,
) -> tuple[torch.Tensor, float]:
    device = get_pipeline_device(pipeline)
    dtype = find_denoiser_module(pipeline).dtype
    sigma_values = _scheduler_sigma_values(pipeline.scheduler)
    sigma_max = float(sigma_values[0].item())
    init_noise_sigma = float(getattr(pipeline.scheduler, "init_noise_sigma", sigma_max))
    vae_scale_factor = getattr(pipeline, "vae_scale_factor", 8)
    shape = (
        batch_size,
        int(pipeline.unet.config.in_channels),
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype) * init_noise_sigma
    return latents, sigma_max


def _prepare_stable_diffusion_defect_batch(
    pipeline,
    *,
    prompt: str | list[str],
    batch_size: int,
    generator: torch.Generator,
    height: int,
    width: int,
    guidance_scale: float,
) -> DiffusersDefectBatch:
    device = get_pipeline_device(pipeline)
    do_cfg = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        device,
        1,
        do_cfg,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
    )
    positive_prompt_embeds = prompt_embeds
    negative_prompt_embeds = negative_prompt_embeds
    latents, sigma_max = _prepare_sd_defect_latents(
        pipeline,
        batch_size=batch_size,
        generator=generator,
        height=height,
        width=width,
    )
    timestep_cond = None
    if getattr(pipeline.unet.config, "time_cond_proj_dim", None) is not None:
        guidance = torch.tensor(guidance_scale - 1.0, device=device, dtype=latents.dtype).repeat(batch_size)
        timestep_cond = pipeline.get_guidance_scale_embedding(
            guidance,
            embedding_dim=pipeline.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    def velocity_fn(
        current_latents: torch.Tensor,
        sigma: torch.Tensor,
        batch_start: int = 0,
        batch_stop: int | None = None,
    ) -> torch.Tensor:
        batch = current_latents.shape[0]
        timestep_value = _vp_timestep_from_sigma(pipeline.scheduler, sigma)
        latent_model_input = torch.cat([current_latents] * 2) if do_cfg else current_latents
        timestep = torch.full((latent_model_input.shape[0],), timestep_value, device=current_latents.device, dtype=torch.float32)
        sigma_value = sigma.reshape(()).to(device=current_latents.device, dtype=current_latents.dtype)
        latent_model_input = latent_model_input / torch.sqrt(sigma_value.square() + 1.0)
        if do_cfg:
            active_prompt_embeds = torch.cat(
                [
                    _slice_batch_tensor(negative_prompt_embeds, batch, batch_start),
                    _slice_batch_tensor(positive_prompt_embeds, batch, batch_start),
                ],
                dim=0,
            )
        else:
            active_prompt_embeds = _slice_batch_tensor(positive_prompt_embeds, batch, batch_start)
        active_timestep_cond = _slice_batch_tensor(timestep_cond, batch, batch_start)
        if do_cfg and active_timestep_cond is not None:
            active_timestep_cond = torch.cat([active_timestep_cond, active_timestep_cond], dim=0)
        noise_pred = pipeline.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=active_prompt_embeds,
            timestep_cond=active_timestep_cond,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    return DiffusersDefectBatch(initial_latents=latents.detach(), sigma_max=sigma_max, velocity_fn=velocity_fn)


def _prepare_sdxl_defect_batch(
    pipeline,
    *,
    prompt: str | list[str],
    batch_size: int,
    generator: torch.Generator,
    height: int,
    width: int,
    guidance_scale: float,
) -> DiffusersDefectBatch:
    device = get_pipeline_device(pipeline)
    do_cfg = guidance_scale > 1.0
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
    )
    positive_prompt_embeds = prompt_embeds
    positive_add_text_embeds = pooled_prompt_embeds
    negative_prompt_embeds = negative_prompt_embeds
    negative_add_text_embeds = negative_pooled_prompt_embeds
    if pipeline.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim
    positive_add_time_ids = pipeline._get_add_time_ids(
        (height, width),
        (0, 0),
        (height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    negative_add_time_ids = positive_add_time_ids
    positive_prompt_embeds = positive_prompt_embeds.to(device)
    negative_prompt_embeds = negative_prompt_embeds.to(device)
    positive_add_text_embeds = positive_add_text_embeds.to(device)
    negative_add_text_embeds = negative_add_text_embeds.to(device)
    positive_add_time_ids = positive_add_time_ids.to(device).repeat(batch_size, 1)
    negative_add_time_ids = negative_add_time_ids.to(device).repeat(batch_size, 1)
    latents, sigma_max = _prepare_sd_defect_latents(
        pipeline,
        batch_size=batch_size,
        generator=generator,
        height=height,
        width=width,
    )
    timestep_cond = None
    if getattr(pipeline.unet.config, "time_cond_proj_dim", None) is not None:
        guidance = torch.tensor(guidance_scale - 1.0, device=device, dtype=latents.dtype).repeat(batch_size)
        timestep_cond = pipeline.get_guidance_scale_embedding(
            guidance,
            embedding_dim=pipeline.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    def velocity_fn(
        current_latents: torch.Tensor,
        sigma: torch.Tensor,
        batch_start: int = 0,
        batch_stop: int | None = None,
    ) -> torch.Tensor:
        batch = current_latents.shape[0]
        timestep_value = _vp_timestep_from_sigma(pipeline.scheduler, sigma)
        latent_model_input = torch.cat([current_latents] * 2) if do_cfg else current_latents
        timestep = torch.full((latent_model_input.shape[0],), timestep_value, device=current_latents.device, dtype=torch.float32)
        sigma_value = sigma.reshape(()).to(device=current_latents.device, dtype=current_latents.dtype)
        latent_model_input = latent_model_input / torch.sqrt(sigma_value.square() + 1.0)
        if do_cfg:
            active_prompt_embeds = torch.cat(
                [
                    _slice_batch_tensor(negative_prompt_embeds, batch, batch_start),
                    _slice_batch_tensor(positive_prompt_embeds, batch, batch_start),
                ],
                dim=0,
            )
        else:
            active_prompt_embeds = _slice_batch_tensor(positive_prompt_embeds, batch, batch_start)
        active_add_text_embeds = (
            torch.cat(
                [
                    _slice_batch_tensor(negative_add_text_embeds, batch, batch_start),
                    _slice_batch_tensor(positive_add_text_embeds, batch, batch_start),
                ],
                dim=0,
            )
            if do_cfg
            else _slice_batch_tensor(positive_add_text_embeds, batch, batch_start)
        )
        active_add_time_ids = (
            torch.cat(
                [
                    _slice_batch_tensor(negative_add_time_ids, batch, batch_start),
                    _slice_batch_tensor(positive_add_time_ids, batch, batch_start),
                ],
                dim=0,
            )
            if do_cfg
            else _slice_batch_tensor(positive_add_time_ids, batch, batch_start)
        )
        active_timestep_cond = _slice_batch_tensor(timestep_cond, batch, batch_start)
        if do_cfg and active_timestep_cond is not None:
            active_timestep_cond = torch.cat([active_timestep_cond, active_timestep_cond], dim=0)
        noise_pred = pipeline.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=active_prompt_embeds,
            timestep_cond=active_timestep_cond,
            cross_attention_kwargs=None,
            added_cond_kwargs={
                "text_embeds": active_add_text_embeds,
                "time_ids": active_add_time_ids,
            },
            return_dict=False,
        )[0]
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    return DiffusersDefectBatch(initial_latents=latents.detach(), sigma_max=sigma_max, velocity_fn=velocity_fn)


def _prepare_deepfloyd_if_defect_batch(
    pipeline,
    *,
    prompt: str | list[str],
    batch_size: int,
    generator: torch.Generator,
    height: int,
    width: int,
    guidance_scale: float,
) -> DiffusersDefectBatch:
    device = get_pipeline_device(pipeline)
    do_cfg = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        do_classifier_free_guidance=do_cfg,
        num_images_per_prompt=1,
        device=device,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        clean_caption=True,
    )
    positive_prompt_embeds = prompt_embeds.to(device)
    negative_prompt_embeds = None if negative_prompt_embeds is None else negative_prompt_embeds.to(device)
    latents = pipeline.prepare_intermediate_images(
        batch_size,
        pipeline.unet.config.in_channels,
        height,
        width,
        positive_prompt_embeds.dtype,
        device,
        generator,
    )
    sigma_max = float(build_defect_sigma_grid(pipeline, physical_grid_size=3, height=height, width=width)[0])

    def velocity_fn(
        current_latents: torch.Tensor,
        sigma: torch.Tensor,
        batch_start: int = 0,
        batch_stop: int | None = None,
    ) -> torch.Tensor:
        batch = current_latents.shape[0]
        timestep_value = _vp_timestep_from_sigma(pipeline.scheduler, sigma)
        latent_model_input = torch.cat([current_latents] * 2) if do_cfg else current_latents
        timestep = torch.full(
            (latent_model_input.shape[0],),
            timestep_value,
            device=current_latents.device,
            dtype=torch.float32,
        )
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timestep)
        if do_cfg:
            if negative_prompt_embeds is None:
                raise RuntimeError("DeepFloyd IF CFG requested but negative prompt embeddings are unavailable.")
            active_prompt_embeds = torch.cat(
                [
                    _slice_batch_tensor(negative_prompt_embeds, batch, batch_start),
                    _slice_batch_tensor(positive_prompt_embeds, batch, batch_start),
                ],
                dim=0,
            )
        else:
            active_prompt_embeds = _slice_batch_tensor(positive_prompt_embeds, batch, batch_start)
        noise_pred = pipeline.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=active_prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(current_latents.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(current_latents.shape[1], dim=1)
            guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            return torch.cat([guided, predicted_variance], dim=1)
        noise_pred, _ = noise_pred.split(current_latents.shape[1], dim=1)
        return noise_pred

    return DiffusersDefectBatch(initial_latents=latents.detach(), sigma_max=sigma_max, velocity_fn=velocity_fn)


def prepare_defect_batch(
    pipeline,
    *,
    prompt: str | list[str],
    batch_size: int,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
) -> DiffusersDefectBatch:
    device = get_pipeline_device(pipeline)
    generator = torch.Generator(device=device).manual_seed(seed)
    kind = _pipeline_kind(pipeline)
    if kind == "flux":
        return _prepare_flux_defect_batch(
            pipeline,
            prompt=prompt,
            batch_size=batch_size,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
    if kind == "sd3":
        return _prepare_sd3_defect_batch(
            pipeline,
            prompt=prompt,
            batch_size=batch_size,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
    if kind == "lumina2":
        return _prepare_lumina2_defect_batch(
            pipeline,
            prompt=prompt,
            batch_size=batch_size,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
    if kind == "stable_diffusion":
        return _prepare_stable_diffusion_defect_batch(
            pipeline,
            prompt=prompt,
            batch_size=batch_size,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
    if kind == "sdxl":
        return _prepare_sdxl_defect_batch(
            pipeline,
            prompt=prompt,
            batch_size=batch_size,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
    if kind == "deepfloyd_if":
        return _prepare_deepfloyd_if_defect_batch(
            pipeline,
            prompt=prompt,
            batch_size=batch_size,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
    raise ValueError(f"Unsupported diffusers pipeline for defect calibration: {pipeline.__class__.__name__}")


def _reset_scheduler_history(scheduler) -> None:
    solver_order = int(getattr(getattr(scheduler, "config", None), "solver_order", 1))
    for name, value in {
        "model_outputs": [None] * solver_order,
        "timestep_list": [None] * solver_order,
        "lower_order_nums": 0,
        "last_sample": None,
        "_step_index": None,
        "_begin_index": None,
        "noise_predictions": [],
        "velocity_predictions": [],
    }.items():
        if hasattr(scheduler, name):
            setattr(scheduler, name, value)


def _configured_coordinate_nodes(
    scheduler,
    *,
    coordinate_domain: str,
) -> np.ndarray:
    normalized = str(coordinate_domain).lower().strip()
    if normalized == "sigma":
        normalized = "sigmas"
    if normalized == "timestep":
        normalized = "timesteps"
    if normalized == "sigmas":
        sigmas = _scheduler_sigma_values(scheduler).numpy().astype(np.float64)
        step_count = len(getattr(scheduler, "timesteps", []))
        if len(sigmas) > step_count + 1:
            sigmas = sigmas[: step_count + 1]
        if len(sigmas) == step_count:
            sigmas = np.concatenate([sigmas, np.asarray([0.0], dtype=np.float64)])
        return sigmas
    if normalized == "timesteps":
        timesteps = scheduler.timesteps.detach().cpu().float().numpy()
        return np.concatenate([timesteps.astype(np.float64), np.asarray([0.0], dtype=np.float64)])
    raise ValueError(f"Unsupported diffusers trajectory coordinate domain: {coordinate_domain}")


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
        raise RuntimeError("Recorded diffusers trajectory collapsed to fewer than two coordinate nodes.")
    return np.asarray(collapsed_nodes, dtype=np.float64), torch.stack(collapsed_states, dim=0)


def _run_diffusers_base_trajectory(
    *,
    pipeline,
    defect_batch: DiffusersDefectBatch,
    solver: str,
    effective_nfe: int,
    initial_sample: torch.Tensor,
    coordinate_domain: str,
    batch_start: int,
    batch_stop: int,
    height: int,
    width: int,
    generator_seed: int,
    eps: float,
) -> tuple[np.ndarray, torch.Tensor]:
    device = initial_sample.device
    plan = resolve_effective_nfe_plan(solver, int(effective_nfe))
    scheduler_kwargs = _scheduler_mu_kwargs(pipeline, height=height, width=width)
    pipeline.scheduler.set_timesteps(plan.solver_steps, device=device, **scheduler_kwargs)
    _reset_scheduler_history(pipeline.scheduler)
    coordinate_nodes = _configured_coordinate_nodes(
        pipeline.scheduler,
        coordinate_domain=coordinate_domain,
    )
    sigmas = _scheduler_sigma_values(pipeline.scheduler).to(device=device, dtype=initial_sample.dtype)
    states = [initial_sample.detach().clone()]
    sample = initial_sample.detach().clone()
    step_parameters = set(inspect.signature(pipeline.scheduler.step).parameters.keys())
    step_generator = torch.Generator(device=device).manual_seed(int(generator_seed))
    with torch.inference_mode():
        for index, timestep in enumerate(pipeline.scheduler.timesteps):
            sigma = sigmas[min(index, len(sigmas) - 1)].to(device=device, dtype=sample.dtype)
            model_output = defect_batch.velocity_fn(sample, sigma, batch_start, batch_stop)
            kwargs: dict[str, Any] = {}
            if "generator" in step_parameters:
                kwargs["generator"] = step_generator
            step_output = pipeline.scheduler.step(model_output, timestep, sample, **kwargs)
            sample = step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output[0]
            states.append(sample.detach().clone())
    return _collapse_adjacent_trajectory_nodes(coordinate_nodes, states, eps=eps)


def collect_diffusers_trajectory_window_stats(
    *,
    pipeline,
    prompt_pool: Sequence[str],
    batch_size: int,
    num_batches: int,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    solver: str,
    multires_nfes: Sequence[int] = (16, 32, 64),
    window_size: int | None = None,
    observation_microbatch: int | None = None,
    coordinate_domain: str | None = None,
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, FPTrajectoryStats, dict[str, object]]:
    spec = get_solver_native_spec("diffusers", solver)
    if not spec.supports_base_trajectory_recording:
        raise ValueError(f"Diffusers solver `{solver}` does not support trajectory-window FP calibration: {spec.notes}")
    nfes = tuple(int(value) for value in multires_nfes)
    if len(nfes) != 3 or nfes[1] != 2 * nfes[0] or nfes[2] != 2 * nfes[1]:
        raise ValueError("trajectory-window FP calibration expects multires_nfes=[N,2N,4N].")
    active_domain = str(coordinate_domain or spec.native_coordinate).lower().strip()
    if active_domain == "sigma":
        active_domain = "sigmas"
    if active_domain == "timestep":
        active_domain = "timesteps"
    if active_domain not in {"sigmas", "timesteps"}:
        raise ValueError(f"Unsupported diffusers trajectory coordinate domain: {active_domain}")
    active_window = int(window_size or spec.recommended_window_len)
    if active_window < int(spec.solver_order):
        raise ValueError("window_size must be at least the solver order/history length.")

    batches: list[FPTrajectoryStats] = []
    details: list[dict[str, object]] = []
    coarse_grid_reference: np.ndarray | None = None
    prompt_values = [str(prompt) for prompt in prompt_pool]
    if not prompt_values:
        raise ValueError("prompt_pool must be non-empty.")

    with torch.inference_mode():
        for batch_index in range(num_batches):
            prompt_batch = [
                prompt_values[(batch_index * batch_size + item_index) % len(prompt_values)]
                for item_index in range(batch_size)
            ]
            defect_batch = prepare_defect_batch(
                pipeline,
                prompt=prompt_batch,
                batch_size=batch_size,
                seed=seed + batch_index,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
            )
            micro = observation_microbatch if observation_microbatch and observation_microbatch > 0 else batch_size
            micro = min(int(micro), batch_size)
            for start in range(0, batch_size, micro):
                stop = min(start + micro, batch_size)
                sample_slice = defect_batch.initial_latents[start:stop]
                trajectories = [
                    _run_diffusers_base_trajectory(
                        pipeline=pipeline,
                        defect_batch=defect_batch,
                        solver=solver,
                        effective_nfe=nfe,
                        initial_sample=sample_slice,
                        coordinate_domain=active_domain,
                        batch_start=start,
                        batch_stop=stop,
                        height=height,
                        width=width,
                        generator_seed=seed + 1009 * batch_index + nfe,
                        eps=eps,
                    )
                    for nfe in nfes
                ]
                stats, window_details = collect_trajectory_window_stats(
                    coarse_grid=trajectories[0][0],
                    coarse_states=trajectories[0][1],
                    mid_grid=trajectories[1][0],
                    mid_states=trajectories[1][1],
                    fine_grid=trajectories[2][0],
                    fine_states=trajectories[2][1],
                    window_size=active_window,
                    q_min=q_min,
                    q_max=q_max,
                    eps=eps,
                )
                if coarse_grid_reference is None:
                    coarse_grid_reference = trajectories[0][0]
                elif not np.allclose(coarse_grid_reference, trajectories[0][0], rtol=0.0, atol=max(float(eps), 1.0e-8)):
                    raise RuntimeError("Official-base coarse grids changed across diffusers trajectory-window batches.")
                batches.append(stats)
                details.append(
                    {
                        "window_size": int(window_details.window_size),
                        "mean_window_residual_perp_norm": float(np.mean(window_details.window_residual_perp_norm)),
                        "mean_window_delta_s": float(np.mean(window_details.window_delta_s)),
                        "mean_window_effective_order": float(np.mean(window_details.window_effective_order)),
                    }
                )
            device = get_pipeline_device(pipeline)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if coarse_grid_reference is None:
        raise RuntimeError("No diffusers trajectory-window calibration batches were collected.")
    stats = concatenate_fp_clock_stats(batches)
    detail_meta = {
        "window_size": active_window,
        "solver_order": int(spec.solver_order),
        "coordinate_domain": active_domain,
        "multires_nfes": list(nfes),
        "trajectory_window_batch_summaries": details,
    }
    return coarse_grid_reference, stats, detail_meta


def replace_scheduler(pipeline, solver_name: str):
    solver = normalize_solver_name(solver_name)
    if solver in {"base", "default", "flow_euler"}:
        return pipeline

    shift = getattr(pipeline.scheduler.config, "shift", 1.0)
    if solver == "euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif solver == "flow_heun":
        pipeline.scheduler = FlowMatchHeunDiscreteScheduler.from_config(pipeline.scheduler.config, shift=shift)
    elif solver == "flow_dpm_solver":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            use_flow_sigmas=True,
            prediction_type="flow_prediction",
            flow_shift=shift,
        )
    elif solver == "flow_unipc":
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config,
            use_flow_sigmas=True,
            prediction_type="flow_prediction",
            flow_shift=shift,
        )
    elif solver in {"dpm_solver_pp", "dpm_solverpp"}:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
        )
    elif solver in {"sde_dpm_solver_pp", "sde_dpmsolverpp"}:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            solver_order=2,
        )
    elif solver in {"flow_stork4_1st", "flow_stork_4_1st"}:
        pipeline.scheduler = STORKScheduler.from_config(
            pipeline.scheduler.config,
            prediction_type="flow_prediction",
            solver_order=4,
            derivative_order=1,
            shift=shift,
        )
    elif solver in {"flow_stork4_2nd", "flow_stork_4_2nd"}:
        pipeline.scheduler = STORKScheduler.from_config(
            pipeline.scheduler.config,
            prediction_type="flow_prediction",
            solver_order=4,
            derivative_order=2,
            shift=shift,
        )
    elif solver in {"flow_stork4_3rd", "flow_stork_4_3rd"}:
        pipeline.scheduler = STORKScheduler.from_config(
            pipeline.scheduler.config,
            prediction_type="flow_prediction",
            solver_order=4,
            derivative_order=3,
            shift=shift,
        )
    else:
        raise ValueError(f"Unsupported diffusers solver: {solver_name}")
    return pipeline


def _stork_uses_flow_prediction(scheduler) -> bool:
    return isinstance(scheduler, STORKScheduler) and getattr(scheduler, "prediction_type", None) == "flow_prediction"


def _stork_flow_anchor_sigmas(schedule_bundle: ScheduleBundle) -> list[float] | None:
    sigmas = None
    if schedule_bundle.sigma_grid is not None:
        sigmas = schedule_bundle.sigma_grid
    elif schedule_bundle.sigmas is not None:
        sigmas = schedule_bundle.sigmas
    if sigmas is None:
        return None
    values = torch.as_tensor(sigmas, dtype=torch.float32).detach().cpu().numpy()
    if len(values) > 0 and abs(float(values[-1])) < 1.0e-12:
        values = values[:-1]
    return values.tolist()


def _snap_descending_timesteps(values: np.ndarray, *, num_train_timesteps: int) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or len(raw) == 0:
        raise ValueError("Custom scheduler timesteps must be a non-empty 1D array.")
    max_timestep = int(num_train_timesteps) - 1
    if len(raw) > max_timestep:
        raise ValueError(
            f"Cannot make {len(raw)} strictly decreasing timesteps within [1, {max_timestep}]."
        )
    snapped = np.rint(raw).astype(np.int64)
    previous = max_timestep + 1
    for index in range(len(snapped)):
        lower = len(snapped) - index
        upper = previous - 1
        snapped[index] = int(np.clip(snapped[index], lower, upper))
        previous = int(snapped[index])
    if np.any(np.diff(snapped) >= 0):
        raise ValueError(f"Snapped timesteps must be strictly descending, got {snapped.tolist()}.")
    return snapped


def build_pipeline_kwargs(
    pipeline,
    *,
    prompt: str | list[str],
    num_inference_steps: int,
    schedule_bundle: ScheduleBundle | None,
    height: int,
    width: int,
    guidance_scale: float,
    generator: torch.Generator,
    output_type: str = "pil",
) -> dict[str, Any]:
    parameters = _signature_parameters(pipeline)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "output_type": output_type,
    }
    if "height" in parameters:
        kwargs["height"] = height
    if "width" in parameters:
        kwargs["width"] = width
    if "guidance_scale" in parameters:
        kwargs["guidance_scale"] = guidance_scale
    if "true_cfg_scale" in parameters and "guidance_scale" not in parameters:
        kwargs["true_cfg_scale"] = guidance_scale
    if "cfg_trunc_ratio" in parameters:
        kwargs["cfg_trunc_ratio"] = 0.25
    if "cfg_normalization" in parameters:
        kwargs["cfg_normalization"] = True
    if "max_sequence_length" in parameters:
        default = parameters["max_sequence_length"].default
        kwargs["max_sequence_length"] = int(default) if isinstance(default, int) else 256
    if "mu" in parameters and getattr(pipeline.scheduler.config, "use_dynamic_shifting", False):
        kwargs["mu"] = compute_dynamic_mu(pipeline, height=height, width=width)
    if schedule_bundle is not None:
        if _stork_uses_flow_prediction(pipeline.scheduler) and "sigmas" in parameters:
            sigmas = _stork_flow_anchor_sigmas(schedule_bundle)
            if sigmas is not None:
                kwargs["sigmas"] = sigmas
                kwargs["num_inference_steps"] = len(sigmas)
        elif "timesteps" in parameters and schedule_bundle.timesteps is not None:
            num_train_timesteps = int(getattr(pipeline.scheduler.config, "num_train_timesteps", 1000))
            kwargs["timesteps"] = _snap_descending_timesteps(
                schedule_bundle.timesteps,
                num_train_timesteps=num_train_timesteps,
            ).tolist()
        elif "sigmas" in parameters and schedule_bundle.sigmas is not None:
            kwargs["sigmas"] = schedule_bundle.sigmas.tolist()
    return kwargs


def run_generation(
    *,
    pipeline,
    prompts: list[str],
    num_inference_steps: int,
    seed: int,
    output_dir: str | Path,
    schedule_bundle: ScheduleBundle | None = None,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 3.5,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts):
        generator = torch.Generator(device=get_pipeline_device(pipeline)).manual_seed(seed + index)
        kwargs = build_pipeline_kwargs(
            pipeline,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            schedule_bundle=schedule_bundle,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = pipeline(**kwargs).images[0]
        image.save(output_path / f"{index:06d}.jpg")
    return output_path


def collect_calibration_records(
    *,
    pipeline,
    prompt: str | list[str],
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    norm_type: str = "l2",
    normalize_by_dim: bool = False,
) -> tuple[list, torch.Tensor]:
    scheduler_kwargs = {}
    if getattr(pipeline.scheduler.config, "use_dynamic_shifting", False):
        scheduler_kwargs["mu"] = compute_dynamic_mu(pipeline, height=height, width=width)
    pipeline.scheduler.set_timesteps(num_inference_steps, device=get_pipeline_device(pipeline), **scheduler_kwargs)
    raw_sigmas = getattr(pipeline.scheduler, "sigmas", None)
    if raw_sigmas is None:
        raise RuntimeError("The selected diffusers scheduler does not expose a sigma sequence for calibration.")
    sigma_tensor = raw_sigmas.detach().cpu().float() if isinstance(raw_sigmas, torch.Tensor) else torch.tensor(raw_sigmas, dtype=torch.float32)
    domain_values = sigma_tensor[:-1]

    generator = torch.Generator(device=get_pipeline_device(pipeline)).manual_seed(seed)
    denoiser = find_denoiser_module(pipeline)
    with ForwardNormCollector(denoiser, norm_type=norm_type, normalize_by_dim=normalize_by_dim) as collector:
        kwargs = build_pipeline_kwargs(
            pipeline,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            schedule_bundle=None,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        pipeline(**kwargs)
    if len(collector.records) != len(domain_values):
        raise RuntimeError(
            f"Calibration record count ({len(collector.records)}) does not match scheduler domain length ({len(domain_values)})."
        )
    return collector.records, domain_values

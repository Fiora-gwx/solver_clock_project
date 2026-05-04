from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import wraps
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from src.clock.calibration import ForwardNormCollector
from src.clock.fp_clock import collect_anchored_replay_stats, concatenate_fp_clock_stats
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
    FlowMatchEulerDiscreteScheduler,
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
    kwargs = {
        "torch_dtype": torch_dtype_from_name(dtype_name),
        "local_files_only": True,
    }
    try:
        pipeline = DiffusionPipeline.from_pretrained(str(model_path), **kwargs)
    except AttributeError as error:
        if "all_tied_weights_keys" not in str(error):
            raise
        pipeline = DiffusionPipeline.from_pretrained(
            str(model_path),
            **kwargs,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
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
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timestep)
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
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timestep)
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
        timestep = torch.as_tensor(timestep_value, device=current_latents.device, dtype=torch.float32)
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
            if pipeline.scheduler.config.variance_type in ["learned", "learned_range"]:
                return torch.cat([guided, predicted_variance], dim=1)
            return guided
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


_REPLAY_HISTORY_NAMES = (
    "model_outputs",
    "timestep_list",
    "lower_order_nums",
    "last_sample",
    "noise_predictions",
    "velocity_predictions",
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
    if hasattr(scheduler, "is_scale_input_called"):
        scheduler.is_scale_input_called = False


def _collapse_repeated_values(values: np.ndarray, *, eps: float) -> np.ndarray:
    collapsed: list[float] = []
    for value in np.asarray(values, dtype=np.float64).tolist():
        if not collapsed or abs(float(value) - collapsed[-1]) > float(eps):
            collapsed.append(float(value))
    return np.asarray(collapsed, dtype=np.float64)


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


def _diffusers_sigmas_for_timesteps(scheduler, timesteps: np.ndarray) -> np.ndarray:
    if not hasattr(scheduler, "alphas_cumprod"):
        total_steps = float(getattr(getattr(scheduler, "config", None), "num_train_timesteps", 1000))
        return np.asarray(timesteps, dtype=np.float64) / total_steps
    alphas = scheduler.alphas_cumprod.detach().float().cpu().numpy()
    train_sigmas = np.sqrt(np.clip(1.0 - alphas, 0.0, None) / np.clip(alphas, 1.0e-12, None))
    query = np.clip(np.asarray(timesteps, dtype=np.float64), 0.0, float(len(train_sigmas) - 1))
    return np.interp(query, np.arange(len(train_sigmas), dtype=np.float64), train_sigmas).astype(np.float64)


def _diffusers_timesteps_for_sigmas(scheduler, sigmas: np.ndarray) -> np.ndarray:
    sigma_values = np.asarray(sigmas, dtype=np.float64)
    if _uses_flow_sigmas(scheduler):
        total_steps = float(getattr(getattr(scheduler, "config", None), "num_train_timesteps", 1000))
        return sigma_values * total_steps
    if hasattr(scheduler, "alphas_cumprod"):
        train_sigmas = np.sqrt(
            np.clip(1.0 - scheduler.alphas_cumprod.detach().float().cpu().numpy(), 0.0, None)
            / np.clip(scheduler.alphas_cumprod.detach().float().cpu().numpy(), 1.0e-12, None)
        )
        log_sigmas = np.log(np.clip(train_sigmas, 1.0e-10, None))
        if hasattr(scheduler, "_sigma_to_t"):
            return np.asarray(
                [
                    float(
                        np.asarray(
                            scheduler._sigma_to_t(
                                np.asarray([max(float(sigma), 1.0e-10)], dtype=np.float64),
                                log_sigmas,
                            )
                        ).reshape(-1)[0]
                    )
                    for sigma in sigma_values
                ],
                dtype=np.float64,
            )
        return np.interp(
            np.log(np.clip(sigma_values, 1.0e-10, None)),
            log_sigmas,
            np.arange(len(train_sigmas), dtype=np.float64),
        ).astype(np.float64)
    raise RuntimeError(f"Scheduler {scheduler.__class__.__name__} cannot map sigmas to timesteps.")


def _set_begin_index(scheduler, begin_index: int) -> None:
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(int(begin_index))
    elif hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = int(begin_index)
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None


def _set_replay_scheduler_nodes(
    scheduler,
    nodes: np.ndarray,
    *,
    coordinate_domain: str,
    device: torch.device,
    context_nodes: np.ndarray | None = None,
) -> int:
    replay_nodes = np.asarray(nodes, dtype=np.float64)
    if replay_nodes.ndim != 1 or len(replay_nodes) < 2:
        raise ValueError("Replay grid must contain at least two nodes.")
    if np.any(np.diff(replay_nodes) >= 0):
        raise ValueError("Replay grid must be strictly descending.")

    if context_nodes is None:
        combined = replay_nodes
        begin_index = 0
    else:
        context = np.asarray(context_nodes, dtype=np.float64)
        if context.ndim != 1 or len(context) < 1:
            raise ValueError("context_nodes must be a non-empty 1D array.")
        if len(context) > 1 and np.any(np.diff(context) >= 0):
            raise ValueError("Replay context grid must be strictly descending.")
        if abs(float(context[-1]) - float(replay_nodes[0])) > 1.0e-6:
            raise ValueError("Replay context must end at the replay anchor node.")
        combined = np.concatenate([context[:-1], replay_nodes])
        begin_index = len(context) - 1

    normalized_domain = str(coordinate_domain).lower().strip()
    if normalized_domain == "sigma":
        normalized_domain = "sigmas"
    if normalized_domain == "timestep":
        normalized_domain = "timesteps"

    if normalized_domain == "sigmas":
        sigma_grid = combined
        time_grid = _diffusers_timesteps_for_sigmas(scheduler, sigma_grid)
    elif normalized_domain == "timesteps":
        time_grid = combined
        sigma_grid = _diffusers_sigmas_for_timesteps(scheduler, time_grid)
    else:
        raise ValueError(f"Unsupported diffusers replay coordinate domain: {coordinate_domain}")

    scheduler.timesteps = torch.from_numpy(np.asarray(time_grid[:-1], dtype=np.float32)).to(device=device)
    scheduler.sigmas = torch.from_numpy(np.asarray(sigma_grid, dtype=np.float32)).to("cpu")
    scheduler.num_inference_steps = len(time_grid) - 1
    _reset_scheduler_history(scheduler)
    _set_begin_index(scheduler, begin_index)
    _move_unipc_sigmas_to_device(scheduler, device)
    return begin_index


def _scheduler_step_kwargs(scheduler, sample: torch.Tensor) -> dict[str, object]:
    parameters = set(inspect.signature(scheduler.step).parameters.keys())
    kwargs: dict[str, object] = {}
    if (
        "variance_noise" in parameters
        and str(getattr(getattr(scheduler, "config", None), "algorithm_type", "")).startswith("sde-")
    ):
        kwargs["variance_noise"] = torch.zeros_like(sample, dtype=torch.float32)
    return kwargs


def _scheduler_sigma_at(scheduler, index: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        raise RuntimeError("The selected diffusers scheduler does not expose sigmas for anchored replay.")
    sigma = sigmas[int(index)]
    return torch.as_tensor(sigma, device=device, dtype=dtype)


def _evaluate_scheduler_model_output(
    defect_batch: DiffusersDefectBatch,
    scheduler,
    sample: torch.Tensor,
    scheduler_timestep,
    sigma: torch.Tensor,
    batch_start: int,
    batch_stop: int | None,
) -> torch.Tensor:
    del scheduler, scheduler_timestep
    return defect_batch.velocity_fn(sample, sigma, batch_start, batch_stop)


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


def build_diffusers_native_coordinate_grid(
    pipeline,
    *,
    solver_name: str,
    effective_nfe: int,
    coordinate_domain: str,
    height: int,
    width: int,
    eps: float = 1.0e-12,
) -> np.ndarray:
    device = get_pipeline_device(pipeline)
    plan = resolve_effective_nfe_plan(solver_name, int(effective_nfe))
    pipeline.scheduler.set_timesteps(
        plan.solver_steps,
        device=device,
        **_scheduler_mu_kwargs(pipeline, height=height, width=width),
    )
    _reset_scheduler_history(pipeline.scheduler)
    nodes = _configured_coordinate_nodes(pipeline.scheduler, coordinate_domain=coordinate_domain)
    return _collapse_repeated_values(nodes, eps=eps)


def _run_scheduler_quarter_reference_on_grid(
    *,
    scheduler,
    defect_batch: DiffusersDefectBatch,
    initial_sample: torch.Tensor,
    coarse_grid: np.ndarray,
    coordinate_domain: str,
    refinement_factor: int,
    solver_order: int,
    batch_start: int,
    batch_stop: int | None,
    eps: float,
) -> tuple[torch.Tensor, list[dict[str, object]], np.ndarray]:
    device = initial_sample.device
    fine_grid = _refined_window_nodes(np.asarray(coarse_grid, dtype=np.float64), int(refinement_factor))
    _set_replay_scheduler_nodes(
        scheduler,
        fine_grid,
        coordinate_domain=coordinate_domain,
        device=device,
    )
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
            sigma = _scheduler_sigma_at(scheduler, step_index, device=sample.device, dtype=sample.dtype)
            model_output = _evaluate_scheduler_model_output(
                defect_batch,
                scheduler,
                sample,
                timestep,
                sigma,
                batch_start,
                batch_stop,
            )
            step_output = scheduler.step(
                model_output,
                timestep,
                sample,
                **_scheduler_step_kwargs(scheduler, sample),
            )
            sample = step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output[0]
            if (step_index + 1) % int(refinement_factor) == 0:
                states.append(sample.detach().clone())
                history.append(_snapshot_scheduler_history(scheduler))

    if len(states) != len(coarse_grid):
        raise RuntimeError(
            f"Quarter reference recorded {len(states)} coarse states for {len(coarse_grid)} coarse nodes."
        )
    if len(history) != len(coarse_grid):
        raise RuntimeError("Quarter reference history count does not match coarse grid.")
    return torch.stack(states, dim=0), history, fine_grid


def _replay_diffusers_window_endpoint(
    *,
    scheduler,
    defect_batch: DiffusersDefectBatch,
    coordinate_nodes: np.ndarray,
    coordinate_domain: str,
    context_nodes: np.ndarray,
    anchor_sample: torch.Tensor,
    anchor_history: dict[str, object],
    batch_start: int,
    batch_stop: int | None,
) -> torch.Tensor:
    device = anchor_sample.device
    begin_index = _set_replay_scheduler_nodes(
        scheduler,
        coordinate_nodes,
        coordinate_domain=coordinate_domain,
        device=device,
        context_nodes=context_nodes,
    )
    _restore_scheduler_replay_history(scheduler, anchor_history)
    _set_begin_index(scheduler, begin_index)

    sample = anchor_sample.detach().clone()
    step_count = len(coordinate_nodes) - 1
    with torch.inference_mode():
        for local_index in range(step_count):
            scheduler_index = begin_index + local_index
            timestep = scheduler.timesteps[scheduler_index]
            sigma = _scheduler_sigma_at(scheduler, scheduler_index, device=sample.device, dtype=sample.dtype)
            model_output = _evaluate_scheduler_model_output(
                defect_batch,
                scheduler,
                sample,
                timestep,
                sigma,
                batch_start,
                batch_stop,
            )
            step_output = scheduler.step(
                model_output,
                timestep,
                sample,
                **_scheduler_step_kwargs(scheduler, sample),
            )
            sample = step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output[0]
    return sample.detach().clone()


def _collect_scheduler_history_quarter_anchor_batch(
    *,
    scheduler,
    defect_batch: DiffusersDefectBatch,
    initial_sample: torch.Tensor,
    physical_grid: np.ndarray,
    coordinate_domain: str,
    window_size: int,
    solver_order: int,
    batch_start: int,
    batch_stop: int | None,
    q_min: float,
    q_max: float,
    eps: float,
) -> tuple[object, object]:
    grid = np.asarray(physical_grid, dtype=np.float64)
    reference_states, reference_history, fine_grid = _run_scheduler_quarter_reference_on_grid(
        scheduler=scheduler,
        defect_batch=defect_batch,
        initial_sample=initial_sample,
        coarse_grid=grid,
        coordinate_domain=coordinate_domain,
        refinement_factor=4,
        solver_order=solver_order,
        batch_start=batch_start,
        batch_stop=batch_stop,
        eps=eps,
    )
    interval_count = len(grid) - 1
    replay_endpoints: dict[int, list[torch.Tensor]] = {1: [], 2: [], 4: []}
    context_span = max(int(solver_order), 1)
    with torch.inference_mode():
        for start in range(interval_count):
            stop = min(start + int(window_size), interval_count)
            window_nodes = grid[start : stop + 1]
            fine_anchor = start * 4
            context_start = max(0, fine_anchor - (context_span - 1))
            context_nodes = fine_grid[context_start : fine_anchor + 1]
            anchor_sample = reference_states[start]
            anchor_history = reference_history[start]
            for factor in (1, 2, 4):
                replay_nodes = _refined_window_nodes(window_nodes, int(factor))
                endpoint = _replay_diffusers_window_endpoint(
                    scheduler=scheduler,
                    defect_batch=defect_batch,
                    coordinate_nodes=replay_nodes,
                    coordinate_domain=coordinate_domain,
                    context_nodes=context_nodes,
                    anchor_sample=anchor_sample,
                    anchor_history=anchor_history,
                    batch_start=batch_start,
                    batch_stop=batch_stop,
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
    pipeline,
    solver: str,
    prompt_pool: Sequence[str],
    batch_size: int,
    num_batches: int,
    seed: int,
    anchor_nfe: int,
    height: int,
    width: int,
    guidance_scale: float,
    window_size: int | None = None,
    observation_microbatch: int | None = None,
    coordinate_domain: str | None = None,
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, object, dict[str, object]]:
    normalized_solver = normalize_solver_name(solver)
    spec = get_solver_native_spec("diffusers", normalized_solver)
    if not spec.supports_base_trajectory_recording:
        raise ValueError(f"Diffusers solver `{solver}` does not support anchored replay FP calibration: {spec.notes}")
    active_domain = str(coordinate_domain or spec.native_coordinate).lower().strip()
    if active_domain == "lambda":
        active_domain = str(spec.native_coordinate)
    active_domain = {"sigma": "sigmas", "timestep": "timesteps"}.get(active_domain, active_domain)
    if active_domain not in {"sigmas", "timesteps"}:
        raise ValueError(f"Unsupported diffusers anchored replay coordinate domain: {active_domain}")
    active_window = int(window_size or spec.recommended_window_len)
    if active_window < int(spec.solver_order):
        raise ValueError("window_size must be at least the solver order/history length.")

    device = get_pipeline_device(pipeline)
    batches: list[object] = []
    details: list[dict[str, object]] = []
    grid_reference: np.ndarray | None = None
    cost_per_sample: int | None = None

    with torch.inference_mode():
        for batch_index in range(int(num_batches)):
            prompt_batch = [
                str(prompt_pool[(batch_index * int(batch_size) + item_index) % len(prompt_pool)])
                for item_index in range(int(batch_size))
            ]
            physical_grid = build_diffusers_native_coordinate_grid(
                pipeline,
                solver_name=normalized_solver,
                effective_nfe=int(anchor_nfe),
                coordinate_domain=active_domain,
                height=height,
                width=width,
                eps=eps,
            )
            if grid_reference is None:
                grid_reference = physical_grid
            elif not np.allclose(grid_reference, physical_grid, rtol=0.0, atol=max(float(eps), 1.0e-8)):
                raise RuntimeError("Diffusers anchored replay grids changed across calibration batches.")

            defect_batch = prepare_defect_batch(
                pipeline,
                prompt=prompt_batch,
                batch_size=int(batch_size),
                seed=int(seed) + batch_index,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
            )
            micro = observation_microbatch if observation_microbatch and observation_microbatch > 0 else batch_size
            micro = min(int(micro), int(batch_size))
            interval_count = len(physical_grid) - 1
            cost_per_sample = int(interval_count) * (4 + 7 * int(active_window))
            for start in range(0, int(batch_size), micro):
                stop = min(start + micro, int(batch_size))
                stats, replay_details = _collect_scheduler_history_quarter_anchor_batch(
                    scheduler=pipeline.scheduler,
                    defect_batch=defect_batch,
                    initial_sample=defect_batch.initial_latents[start:stop],
                    physical_grid=physical_grid,
                    coordinate_domain=active_domain,
                    window_size=active_window,
                    solver_order=int(spec.solver_order),
                    batch_start=start,
                    batch_stop=stop,
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
        raise RuntimeError("No diffusers anchored replay calibration batches were collected.")
    detail_meta = {
        "anchor_nfe": int(anchor_nfe),
        "calibration_nfes": [int(anchor_nfe)],
        "window_size": int(active_window),
        "window_len": int(active_window),
        "solver_order": int(spec.solver_order),
        "coordinate_domain": active_domain,
        "native_coordinate": active_domain,
        "target_solver": normalized_solver,
        "replay_backend": "scheduler_history_quarter_anchor",
        "reference_path": "quarter_refined_target_scheduler",
        "q_estimator": "full_l2_replay_ratio",
        "residual_metric": "frenet_normal_replay_residual",
        "multistep_history_aware": True,
        "sde_variance_noise": (
            "zero_for_defect_calibration"
            if str(getattr(getattr(pipeline.scheduler, "config", None), "algorithm_type", "")).startswith("sde-")
            else "none"
        ),
        "anchored_replay_batch_summaries": details,
        "calibration_cost_per_sample": int(cost_per_sample or 0),
    }
    return grid_reference, concatenate_fp_clock_stats(batches), detail_meta


def replace_scheduler(pipeline, solver_name: str):
    solver = normalize_solver_name(solver_name)
    if solver in {"base", "default", "flow_euler"}:
        _attach_flow_native_sigmas(pipeline.scheduler)
        return pipeline

    shift = getattr(pipeline.scheduler.config, "shift", 1.0)
    deepfloyd_if = _pipeline_kind(pipeline) == "deepfloyd_if"
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
    if deepfloyd_if and hasattr(pipeline.scheduler, "register_to_config"):
        pipeline.scheduler.register_to_config(variance_type="fixed_small")
    _attach_flow_native_sigmas(pipeline.scheduler)
    _attach_unipc_device_sigmas(pipeline.scheduler)
    return pipeline


def _stork_uses_flow_prediction(scheduler) -> bool:
    return isinstance(scheduler, STORKScheduler) and getattr(scheduler, "prediction_type", None) == "flow_prediction"


def _move_unipc_sigmas_to_device(scheduler, device: torch.device | str | None) -> None:
    if not isinstance(scheduler, UniPCMultistepScheduler) or device is None:
        return
    if isinstance(getattr(scheduler, "sigmas", None), torch.Tensor):
        scheduler.sigmas = scheduler.sigmas.to(device=device)
    solver_p = getattr(scheduler, "solver_p", None)
    if solver_p is not None and isinstance(getattr(solver_p, "sigmas", None), torch.Tensor):
        solver_p.sigmas = solver_p.sigmas.to(device=device)


def _uses_flow_sigmas(scheduler) -> bool:
    return bool(getattr(getattr(scheduler, "config", None), "use_flow_sigmas", False)) or isinstance(
        scheduler, FlowMatchEulerDiscreteScheduler
    )


def _set_flow_native_sigmas_state(
    scheduler,
    sigmas: Sequence[float] | np.ndarray,
    *,
    device: torch.device | str | None,
) -> None:
    native = np.asarray(sigmas, dtype=np.float32)
    if native.ndim != 1 or len(native) == 0:
        raise ValueError("Custom flow sigmas must be a non-empty 1D schedule.")
    if len(native) > 1 and abs(float(native[-1])) < 1.0e-12:
        native = native[:-1]
    if np.any(~np.isfinite(native)):
        raise ValueError("Custom flow sigmas contain NaN or Inf.")
    if np.any(np.diff(native) >= 0.0):
        raise ValueError("Custom flow sigmas must be strictly descending.")

    num_train_timesteps = float(getattr(scheduler.config, "num_train_timesteps", 1000))
    timesteps = (native * num_train_timesteps).astype(np.float32)
    final_sigmas_type = str(getattr(scheduler.config, "final_sigmas_type", "zero"))
    if final_sigmas_type == "sigma_min":
        sigma_last = float(native[-1])
    elif final_sigmas_type == "zero":
        sigma_last = 0.0
    else:
        raise ValueError(f"Unsupported final_sigmas_type for custom flow sigmas: {final_sigmas_type}")
    full_sigmas = np.concatenate([native, np.asarray([sigma_last], dtype=np.float32)]).astype(np.float32)

    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        timestep_tensor = torch.from_numpy(timesteps).to(device=device, dtype=torch.float32)
        sigma_tensor = torch.from_numpy(full_sigmas).to(device=device, dtype=torch.float32)
    elif isinstance(scheduler, UniPCMultistepScheduler):
        timestep_tensor = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        sigma_tensor = torch.from_numpy(full_sigmas).to(device=device, dtype=torch.float32)
    else:
        timestep_tensor = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        sigma_tensor = torch.from_numpy(full_sigmas).to("cpu")

    scheduler.timesteps = timestep_tensor
    scheduler.sigmas = sigma_tensor
    scheduler.num_inference_steps = len(timesteps)
    _reset_scheduler_history(scheduler)
    if isinstance(scheduler, STORKScheduler):
        scheduler.dt_list = torch.as_tensor(
            (scheduler.sigmas[:-1] - scheduler.sigmas[1:]).detach().cpu().numpy(),
            dtype=getattr(scheduler, "dtype", torch.float32),
            device=device,
        )


def _attach_flow_native_sigmas(scheduler):
    if not _uses_flow_sigmas(scheduler) or isinstance(scheduler, STORKScheduler):
        return scheduler
    if getattr(scheduler, "_solver_clock_flow_native_sigmas_patch", False):
        return scheduler

    original_set_timesteps = scheduler.set_timesteps

    def set_timesteps_with_native_sigmas(
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: Sequence[float] | np.ndarray | None = None,
        mu: float | None = None,
        timesteps: Sequence[float] | None = None,
    ):
        if sigmas is None:
            if isinstance(scheduler, DPMSolverMultistepScheduler):
                return original_set_timesteps(
                    num_inference_steps=num_inference_steps,
                    device=device,
                    mu=mu,
                    timesteps=timesteps,
                )
            return original_set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                timesteps=timesteps,
            )
        if timesteps is not None:
            raise ValueError("Custom Solver Clock flow schedules pass native sigmas; timesteps must be omitted.")
        if num_inference_steps is not None and len(sigmas) != int(num_inference_steps):
            raise ValueError("Custom flow sigmas length must match num_inference_steps when both are provided.")
        _set_flow_native_sigmas_state(scheduler, sigmas, device=device)
        return None

    scheduler.set_timesteps = set_timesteps_with_native_sigmas
    scheduler._solver_clock_flow_native_sigmas_patch = True
    return scheduler


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
            kwargs["sigmas"] = np.asarray(schedule_bundle.sigmas, dtype=np.float32)
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

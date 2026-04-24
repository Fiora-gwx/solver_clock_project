from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from src.clock.calibration import ForwardNormCollector
from src.utils.config import repo_root
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
    if "flux" in name:
        return "flux"
    if "stablediffusion3" in name:
        return "sd3"
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


def _slice_batch_tensor(tensor: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 0:
        return tensor
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
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
) -> np.ndarray:
    if physical_grid_size < 2:
        raise ValueError("physical_grid_size must be at least 2.")
    device = get_pipeline_device(pipeline)
    scheduler_kwargs = _scheduler_mu_kwargs(pipeline, height=height, width=width)
    pipeline.scheduler.set_timesteps(physical_grid_size - 1, device=device, **scheduler_kwargs)
    raw_sigmas = getattr(pipeline.scheduler, "sigmas", None)
    if raw_sigmas is None:
        raise RuntimeError("The selected diffusers scheduler does not expose a sigma sequence for defect calibration.")
    sigma_tensor = raw_sigmas.detach().float() if isinstance(raw_sigmas, torch.Tensor) else torch.tensor(raw_sigmas, dtype=torch.float32)
    sigma_max = float(sigma_tensor[0].item())
    return np.linspace(sigma_max, 0.0, physical_grid_size, dtype=np.float64)


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

    def velocity_fn(current_latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        del scheduler_kwargs
        batch = current_latents.shape[0]
        timestep = (sigma.reshape(()).to(device=current_latents.device, dtype=current_latents.dtype) * float(pipeline.scheduler.config.num_train_timesteps))
        timestep = timestep.expand(current_latents.shape[0]) / float(pipeline.scheduler.config.num_train_timesteps)
        return pipeline.transformer(
            hidden_states=current_latents,
            timestep=timestep,
            guidance=_slice_batch_tensor(guidance, batch),
            pooled_projections=_slice_batch_tensor(pooled_prompt_embeds, batch),
            encoder_hidden_states=_slice_batch_tensor(prompt_embeds, batch),
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

    def velocity_fn(current_latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
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
                    [_slice_batch_tensor(negative_prompt_embeds, batch), _slice_batch_tensor(prompt_embeds, batch)],
                    dim=0,
                ),
                pooled_projections=torch.cat(
                    [
                        _slice_batch_tensor(negative_pooled_prompt_embeds, batch),
                        _slice_batch_tensor(pooled_prompt_embeds, batch),
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
            encoder_hidden_states=_slice_batch_tensor(prompt_embeds, batch),
            pooled_projections=_slice_batch_tensor(pooled_prompt_embeds, batch),
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

    def velocity_fn(current_latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        batch = current_latents.shape[0]
        current_timestep = 1.0 - sigma.reshape(()).to(device=current_latents.device, dtype=current_latents.dtype)
        current_timestep = current_timestep.expand(current_latents.shape[0])
        noise_pred_cond = pipeline.transformer(
            hidden_states=current_latents,
            timestep=current_timestep,
            encoder_hidden_states=_slice_batch_tensor(prompt_embeds, batch),
            encoder_attention_mask=_slice_batch_tensor(prompt_attention_mask, batch),
            return_dict=False,
            attention_kwargs=None,
        )[0]
        if not do_cfg:
            return -noise_pred_cond

        noise_pred_uncond = pipeline.transformer(
            hidden_states=current_latents,
            timestep=current_timestep,
            encoder_hidden_states=_slice_batch_tensor(negative_prompt_embeds, batch),
            encoder_attention_mask=_slice_batch_tensor(negative_prompt_attention_mask, batch),
            return_dict=False,
            attention_kwargs=None,
        )[0]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_pred = noise_pred * (cond_norm / torch.clamp(noise_norm, min=1.0e-12))
        return -noise_pred

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
    raise ValueError(f"Unsupported diffusers pipeline for defect calibration: {pipeline.__class__.__name__}")


def replace_scheduler(pipeline, solver_name: str):
    solver = normalize_solver_name(solver_name)
    if solver in {"base", "default", "flow_euler"}:
        return pipeline

    shift = getattr(pipeline.scheduler.config, "shift", 1.0)
    if solver == "flow_heun":
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
        if "sigmas" in parameters and schedule_bundle.sigmas is not None:
            kwargs["sigmas"] = schedule_bundle.sigmas.tolist()
        elif "timesteps" in parameters and schedule_bundle.timesteps is not None:
            kwargs["timesteps"] = schedule_bundle.timesteps.tolist()
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

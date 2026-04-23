#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters.pndm import (  # noqa: E402
    _evaluate_sigma_derivative,
    _resolve_custom_heun_grid,
    build_scheduler,
    load_model,
    load_native_config,
)
from src.utils.assets import AssetManifest  # noqa: E402
from src.utils.config import dump_json, load_yaml, resolve_repo_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare custom Heun integration against vendor native Heun.")
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-asset", required=True)
    parser.add_argument("--effective-nfe", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _norm(value: torch.Tensor) -> float:
    return float(value.detach().float().reshape(value.shape[0], -1).norm(dim=1).mean().item())


def _clone_detached(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _time_like(timestep: float, batch_size: int, *, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), float(timestep), device=device, dtype=torch.float32)


def run_custom_budgeted_heun_trace(
    *,
    model: torch.nn.Module,
    scheduler,
    initial_latents: torch.Tensor,
    effective_nfe: int,
) -> dict[str, Any]:
    anchor_timesteps, time_grid, sigma_grid, step_methods = _resolve_custom_heun_grid(
        scheduler,
        effective_nfe=effective_nfe,
        schedule_bundle=None,
        device=initial_latents.device,
    )
    latents = _clone_detached(initial_latents)
    step_logs: list[dict[str, Any]] = []
    interval_latents = [_clone_detached(latents)]

    for index, method in enumerate(step_methods):
        sigma_value = float(sigma_grid[index])
        sigma_next = float(sigma_grid[index + 1])
        timestep_value = float(time_grid[index])
        next_timestep_value = float(time_grid[index + 1])
        dt = sigma_next - sigma_value

        derivative = _evaluate_sigma_derivative(
            model,
            latents,
            timestep_value,
            sigma_value,
            model_output_type="epsilon",
        )
        predicted = latents + derivative * dt
        if method in {"euler", "euler_tail"}:
            next_derivative = None
            update = derivative * dt
            latents = latents + update
        else:
            next_derivative = _evaluate_sigma_derivative(
                model,
                predicted,
                next_timestep_value,
                sigma_next,
                model_output_type="epsilon",
            )
            update = 0.5 * (derivative + next_derivative) * dt
            latents = latents + update

        step_logs.append(
            {
                "index": index,
                "method": method,
                "time_start": timestep_value,
                "time_end": next_timestep_value,
                "sigma_start": sigma_value,
                "sigma_end": sigma_next,
                "dt": dt,
                "derivative_norm": _norm(derivative),
                "predicted_norm": _norm(predicted),
                "next_derivative_norm": None if next_derivative is None else _norm(next_derivative),
                "update_norm": _norm(update),
                "latent_norm_after": _norm(latents),
            }
        )
        interval_latents.append(_clone_detached(latents))

    return {
        "time_grid": [float(x) for x in time_grid.tolist()],
        "sigma_grid": [float(x) for x in sigma_grid.tolist()],
        "step_methods": [str(x) for x in step_methods],
        "step_logs": step_logs,
        "final_latents": latents,
        "interval_latents": interval_latents,
        "anchor_timesteps": [float(x) for x in anchor_timesteps.tolist()],
    }


def run_vendor_native_heun_trace(
    *,
    model: torch.nn.Module,
    scheduler,
    initial_latents: torch.Tensor,
    solver_steps: int,
) -> dict[str, Any]:
    scheduler.set_timesteps(solver_steps, device=initial_latents.device)
    latents = _clone_detached(initial_latents)
    raw_step_logs: list[dict[str, Any]] = []
    interval_logs: list[dict[str, Any]] = []
    interval_latents = [_clone_detached(latents)]
    pending: dict[str, Any] | None = None

    for index, timestep in enumerate(scheduler.timesteps):
        timestep_value = float(timestep.item())
        state_first_order = bool(scheduler.state_in_first_order)
        sigma_index = 0 if scheduler.step_index is None else int(scheduler.step_index)
        if scheduler.step_index is None:
            sigma_value = float(scheduler.sigmas[0].item())
            sigma_next = float(scheduler.sigmas[1].item())
        elif state_first_order:
            sigma_value = float(scheduler.sigmas[scheduler.step_index].item())
            sigma_next = float(scheduler.sigmas[scheduler.step_index + 1].item())
        else:
            sigma_value = float(scheduler.sigmas[scheduler.step_index - 1].item())
            sigma_next = float(scheduler.sigmas[scheduler.step_index].item())

        model_input = scheduler.scale_model_input(latents, timestep)
        model_output = model(model_input, _time_like(timestep_value, latents.shape[0], device=latents.device))
        prediction_sample = latents if not state_first_order else latents
        derivative = (latents - (latents - sigma_next * model_output if not state_first_order else latents - sigma_value * model_output)) / (
            sigma_next if not state_first_order else sigma_value
        )
        dt = sigma_next - sigma_value
        predicted = latents + derivative * dt

        result = scheduler.step(model_output, timestep, latents)
        new_latents = _clone_detached(result.prev_sample)

        raw_step_logs.append(
            {
                "index": index,
                "mode": "first_order" if state_first_order else "second_order",
                "scheduler_step_index_before": sigma_index,
                "timestep": timestep_value,
                "sigma": sigma_value,
                "sigma_next": sigma_next,
                "dt": dt,
                "derivative_norm": _norm(derivative),
                "predicted_norm": _norm(predicted),
                "prev_sample_norm": _norm(new_latents),
            }
        )

        if state_first_order:
            pending = {
                "index": len(interval_logs),
                "time_start": timestep_value,
                "sigma_start": sigma_value,
                "dt": dt,
                "derivative_norm": _norm(derivative),
                "predicted_norm": _norm(predicted),
            }
        else:
            if pending is None:
                raise RuntimeError("Vendor Heun entered second-order stage without a pending first-order stage.")
            interval_logs.append(
                {
                    "index": pending["index"],
                    "method": "heun2",
                    "time_start": pending["time_start"],
                    "time_end": timestep_value,
                    "sigma_start": pending["sigma_start"],
                    "sigma_end": sigma_next,
                    "dt": pending["dt"],
                    "derivative_norm": pending["derivative_norm"],
                    "predicted_norm": pending["predicted_norm"],
                    "next_derivative_norm": _norm(derivative),
                    "update_norm": _norm(new_latents - interval_latents[-1]),
                    "latent_norm_after": _norm(new_latents),
                }
            )
            interval_latents.append(_clone_detached(new_latents))
            pending = None

        latents = new_latents

    if pending is not None:
        interval_logs.append(
            {
                "index": pending["index"],
                "method": "euler_tail",
                "time_start": pending["time_start"],
                "time_end": float(scheduler.timesteps[-1].item()),
                "sigma_start": pending["sigma_start"],
                "sigma_end": float(scheduler.sigmas[-1].item()),
                "dt": float(scheduler.sigmas[-1].item()) - pending["sigma_start"],
                "derivative_norm": pending["derivative_norm"],
                "predicted_norm": pending["predicted_norm"],
                "next_derivative_norm": None,
                "update_norm": _norm(latents - interval_latents[-1]),
                "latent_norm_after": _norm(latents),
            }
        )
        interval_latents.append(_clone_detached(latents))

    return {
        "raw_timesteps": [float(x) for x in scheduler.timesteps.detach().cpu().numpy().tolist()],
        "raw_sigmas": [float(x) for x in scheduler.sigmas.detach().cpu().numpy().tolist()],
        "raw_step_logs": raw_step_logs,
        "interval_logs": interval_logs,
        "final_latents": latents,
        "interval_latents": interval_latents,
    }


def summarize_latent_diffs(custom: dict[str, Any], vendor: dict[str, Any]) -> dict[str, Any]:
    final_l2 = float(torch.norm((custom["final_latents"] - vendor["final_latents"]).reshape(-1)).item())
    pair_count = min(len(custom["interval_latents"]), len(vendor["interval_latents"]))
    interval_l2 = [
        float(torch.norm((custom["interval_latents"][i] - vendor["interval_latents"][i]).reshape(-1)).item())
        for i in range(pair_count)
    ]
    return {
        "final_latent_l2": final_l2,
        "paired_interval_latent_l2": interval_l2,
        "custom_interval_count": len(custom["interval_latents"]) - 1,
        "vendor_interval_count": len(vendor["interval_latents"]) - 1,
        "vendor_has_euler_tail": any(item["method"] == "euler_tail" for item in vendor["interval_logs"]),
    }


def main() -> None:
    args = parse_args()
    manifest = AssetManifest(args.manifest)
    dataset_config = load_yaml(args.dataset_config)
    native_config = load_native_config(dataset_config["native_config"])
    model, _ = load_model(dataset_config["native_config"], manifest.path(args.model_asset), device="cuda")
    schedule_cfg = native_config["Schedule"]
    scheduler = build_scheduler(
        "heun2",
        diffusion_step=schedule_cfg["diffusion_step"],
        beta_start=schedule_cfg["beta_start"],
        beta_end=schedule_cfg["beta_end"],
        beta_schedule=schedule_cfg["type"],
    )

    plan_solver_steps = args.effective_nfe // 2
    init_sigma = float(getattr(scheduler, "init_noise_sigma"))
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(args.seed)
    initial_latents = torch.randn(
        (args.batch_size, model.in_channels, int(dataset_config["image_size"]), int(dataset_config["image_size"])),
        generator=generator,
        device=next(model.parameters()).device,
    ) * init_sigma

    custom = run_custom_budgeted_heun_trace(
        model=model,
        scheduler=build_scheduler(
            "heun2",
            diffusion_step=schedule_cfg["diffusion_step"],
            beta_start=schedule_cfg["beta_start"],
            beta_end=schedule_cfg["beta_end"],
            beta_schedule=schedule_cfg["type"],
        ),
        initial_latents=initial_latents,
        effective_nfe=args.effective_nfe,
    )
    vendor = run_vendor_native_heun_trace(
        model=model,
        scheduler=build_scheduler(
            "heun2",
            diffusion_step=schedule_cfg["diffusion_step"],
            beta_start=schedule_cfg["beta_start"],
            beta_end=schedule_cfg["beta_end"],
            beta_schedule=schedule_cfg["type"],
        ),
        initial_latents=initial_latents,
        solver_steps=plan_solver_steps,
    )
    summary = summarize_latent_diffs(custom, vendor)

    payload = {
        "dataset_config": str(args.dataset_config),
        "model_asset": args.model_asset,
        "effective_nfe": int(args.effective_nfe),
        "solver_steps_from_budget_plan": int(plan_solver_steps),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "sample_index": int(args.sample_index),
        "summary": summary,
        "custom_budgeted_heun": {
            "time_grid": custom["time_grid"],
            "sigma_grid": custom["sigma_grid"],
            "step_methods": custom["step_methods"],
            "step_logs": custom["step_logs"],
        },
        "vendor_native_heun": {
            "raw_timesteps": vendor["raw_timesteps"],
            "raw_sigmas": vendor["raw_sigmas"],
            "raw_step_logs": vendor["raw_step_logs"],
            "interval_logs": vendor["interval_logs"],
        },
    }
    dump_json(payload, resolve_repo_path(args.output_json))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

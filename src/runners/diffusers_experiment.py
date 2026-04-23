from __future__ import annotations

from pathlib import Path
from typing import Any

from src.adapters.diffusers import load_pipeline, replace_scheduler, run_generation
from src.utils.assets import AssetManifest
from src.utils.config import load_json, load_yaml
from src.utils.nfe_budget import resolve_effective_nfe_plan
from src.utils.results import append_result_row, write_run_manifest
from src.utils.schedule_bundle import ScheduleBundle


def run_diffusers_experiment(
    *,
    manifest_path: str,
    model_asset_key: str,
    solver_name: str,
    prompt_asset_or_path: str,
    num_inference_steps: int,
    seed: int,
    output_dir: str,
    summary_csv: str,
    schedule_name: str = "base",
    schedule_dir: str | None = None,
    dtype_name: str = "bfloat16",
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 3.5,
) -> dict[str, Any]:
    manifest = AssetManifest(manifest_path)
    prompt_path = manifest.path(prompt_asset_or_path) if manifest.has(prompt_asset_or_path) else prompt_asset_or_path
    prompts = load_json(prompt_path)
    model_path = manifest.path(model_asset_key)
    pipeline = load_pipeline(model_path, device="cuda", dtype_name=dtype_name)
    replace_scheduler(pipeline, solver_name)
    bundle = ScheduleBundle.load(schedule_dir) if schedule_dir else None
    execution_plan = resolve_effective_nfe_plan(
        solver_name,
        int(bundle.meta.get("effective_nfe", num_inference_steps)) if bundle else num_inference_steps,
    )
    solver_steps = int(bundle.meta.get("solver_steps", execution_plan.solver_steps)) if bundle else execution_plan.solver_steps
    step_methods = list(bundle.meta.get("step_methods", execution_plan.step_methods)) if bundle else list(execution_plan.step_methods)
    execution_backend = str(bundle.meta.get("execution_backend", execution_plan.execution_backend)) if bundle else execution_plan.execution_backend
    image_dir = run_generation(
        pipeline=pipeline,
        prompts=prompts,
        num_inference_steps=solver_steps,
        seed=seed,
        output_dir=output_dir,
        schedule_bundle=bundle,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
    )
    result = {
        "backend": "diffusers",
        "dataset": "",
        "model_asset": model_asset_key,
        "solver": solver_name,
        "schedule": schedule_name,
        "nfe": num_inference_steps,
        "num_samples": len(prompts),
        "fid": None,
        "effective_nfe": num_inference_steps,
        "solver_steps": solver_steps,
        "step_methods": step_methods,
        "execution_backend": execution_backend,
        "prompt_count": len(prompts),
        "dtype": dtype_name,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "schedule_dir": schedule_dir or "",
        "output_dir": str(image_dir),
    }
    write_run_manifest(Path(output_dir) / "run_manifest.json", result)
    append_result_row(summary_csv, result)
    return result

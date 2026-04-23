from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.adapters.pndm import build_scheduler, load_model, run_generation
from src.utils.assets import AssetManifest
from src.utils.config import load_yaml, resolve_repo_path
from src.utils.fid import compute_fid
from src.utils.nfe_budget import resolve_effective_nfe_plan
from src.utils.results import append_result_row, write_run_manifest
from src.utils.schedule_bundle import ScheduleBundle


def clear_preview_images(destination_dir: str | Path) -> None:
    preview_dir = Path(destination_dir) / "preview"
    if not preview_dir.exists():
        return
    for existing in preview_dir.glob("*.png"):
        existing.unlink()
    try:
        next(preview_dir.iterdir())
    except StopIteration:
        preview_dir.rmdir()


def persist_preview_images(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    max_images: int,
) -> tuple[int, str]:
    if max_images <= 0:
        return 0, ""

    source_path = Path(source_dir)
    preview_dir = Path(destination_dir) / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for existing in preview_dir.glob("*.png"):
        existing.unlink()

    preview_count = 0
    for image_path in sorted(source_path.glob("*.png"))[:max_images]:
        shutil.copy2(image_path, preview_dir / image_path.name)
        preview_count += 1
    if preview_count == 0:
        return 0, ""
    return preview_count, str(preview_dir)


def run_pndm_experiment(
    *,
    manifest_path: str,
    dataset_config_path: str,
    model_asset_key: str,
    solver_name: str,
    num_inference_steps: int,
    num_samples: int,
    batch_size: int,
    seed: int,
    output_dir: str,
    summary_csv: str,
    schedule_name: str = "base",
    schedule_dir: str | None = None,
    compute_fid_score: bool = False,
    save_samples: bool = True,
    preview_samples: int = 0,
) -> dict[str, Any]:
    if not save_samples and not compute_fid_score:
        raise ValueError("save_samples=False is only supported when compute_fid_score=True.")
    if preview_samples < 0:
        raise ValueError("preview_samples must be non-negative.")

    manifest = AssetManifest(manifest_path)
    dataset_config = load_yaml(dataset_config_path)
    native_config_path = resolve_repo_path(dataset_config["native_config"])
    model_path = manifest.path(model_asset_key)
    model, native_config = load_model(native_config_path, model_path, device="cuda")
    schedule_cfg = native_config["Schedule"]
    scheduler = build_scheduler(
        solver_name,
        diffusion_step=schedule_cfg["diffusion_step"],
        beta_start=schedule_cfg["beta_start"],
        beta_end=schedule_cfg["beta_end"],
        beta_schedule=schedule_cfg["type"],
    )
    bundle = ScheduleBundle.load(schedule_dir) if schedule_dir else None
    execution_plan = resolve_effective_nfe_plan(
        solver_name,
        int(bundle.meta.get("effective_nfe", num_inference_steps)) if bundle else num_inference_steps,
    )
    solver_steps = int(bundle.meta.get("solver_steps", execution_plan.solver_steps)) if bundle else execution_plan.solver_steps
    step_methods = list(bundle.meta.get("step_methods", execution_plan.step_methods)) if bundle else list(execution_plan.step_methods)
    execution_backend = str(bundle.meta.get("execution_backend", execution_plan.execution_backend)) if bundle else execution_plan.execution_backend
    persisted_output_dir = Path(output_dir)
    persisted_output_dir.mkdir(parents=True, exist_ok=True)
    clear_preview_images(persisted_output_dir)
    image_dir = persisted_output_dir
    transient_samples_dir = ""
    preview_samples_persisted = 0
    preview_dir = ""
    if save_samples:
        image_dir = run_generation(
            model=model,
            scheduler=scheduler,
            image_size=dataset_config["image_size"],
            num_samples=num_samples,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_dir=output_dir,
            schedule_bundle=bundle,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="fid_samples_", dir=str(persisted_output_dir.parent)) as temporary_dir:
            transient_samples_dir = temporary_dir
            image_dir = run_generation(
                model=model,
                scheduler=scheduler,
                image_size=dataset_config["image_size"],
                num_samples=num_samples,
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                seed=seed,
                output_dir=temporary_dir,
                schedule_bundle=bundle,
            )
            fid_value = None
            if compute_fid_score:
                fid_asset_key = dataset_config["default_fid_asset"]
                fid_value = compute_fid(image_dir, manifest.path(fid_asset_key))
            preview_samples_persisted, preview_dir = persist_preview_images(
                image_dir,
                persisted_output_dir,
                max_images=preview_samples,
            )
            result = {
                "backend": "pndm",
                "dataset": dataset_config["name"],
                "model_asset": model_asset_key,
                "solver": solver_name,
                "schedule": schedule_name,
                "nfe": num_inference_steps,
                "effective_nfe": num_inference_steps,
                "solver_steps": solver_steps,
                "step_methods": step_methods,
                "execution_backend": execution_backend,
                "num_samples": num_samples,
                "fid": fid_value,
                "schedule_dir": schedule_dir or "",
                "output_dir": str(persisted_output_dir),
                "samples_persisted": False,
            }
            manifest_payload = dict(result)
            manifest_payload["transient_samples_dir"] = transient_samples_dir
            manifest_payload["preview_samples_requested"] = preview_samples
            manifest_payload["preview_samples_persisted"] = preview_samples_persisted
            manifest_payload["preview_dir"] = preview_dir
            write_run_manifest(persisted_output_dir / "run_manifest.json", manifest_payload)
            append_result_row(summary_csv, result)
            return result

    fid_value: float | None = None
    if compute_fid_score:
        fid_asset_key = dataset_config["default_fid_asset"]
        fid_value = compute_fid(image_dir, manifest.path(fid_asset_key))

    result = {
        "backend": "pndm",
        "dataset": dataset_config["name"],
        "model_asset": model_asset_key,
        "solver": solver_name,
        "schedule": schedule_name,
        "nfe": num_inference_steps,
        "effective_nfe": num_inference_steps,
        "solver_steps": solver_steps,
        "step_methods": step_methods,
        "execution_backend": execution_backend,
        "num_samples": num_samples,
        "fid": fid_value,
        "schedule_dir": schedule_dir or "",
        "output_dir": str(persisted_output_dir),
        "samples_persisted": True,
    }
    manifest_payload = dict(result)
    manifest_payload["preview_samples_requested"] = preview_samples
    manifest_payload["preview_samples_persisted"] = preview_samples_persisted
    manifest_payload["preview_dir"] = preview_dir
    write_run_manifest(persisted_output_dir / "run_manifest.json", manifest_payload)
    append_result_row(summary_csv, result)
    return result

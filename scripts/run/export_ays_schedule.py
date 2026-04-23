#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters.pndm import (
    build_scheduler,
    load_model,
    load_native_config,
    load_training_dataset,
    run_generation,
)
from src.clock.ays import AysConfig, build_sigma_lookup, hierarchical_optimize_schedule, schedule_for_nfe
from src.utils.assets import AssetManifest
from src.utils.config import dump_json, ensure_dir, load_yaml
from src.utils.fid import compute_fid
from src.utils.nfe_budget import resolve_effective_nfe_plan
from src.utils.schedule_bundle import ScheduleBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AYS schedule bundles for project-owned PNDM experiments.")
    parser.add_argument("--backend", choices=["pndm"], default="pndm")
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-asset")
    parser.add_argument("--solver", default="euler")
    parser.add_argument("--target-nfes", default="6,8,10,12,16,20,24,32")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ays-config", default="configs/clocks/AYS.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--consumer-solvers", default="")
    return parser.parse_args()


def parse_target_nfes(raw: str) -> list[int]:
    values = sorted({int(item) for item in raw.split(",") if item})
    if not values:
        raise ValueError("At least one target NFE is required.")
    return values


def parse_solver_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_reference_loader(dataset, *, subset_size: int, batch_size: int, seed: int) -> DataLoader:
    subset_size = max(1, min(int(subset_size), len(dataset)))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=max(1, min(batch_size, subset_size)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )


def cycle_batches(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            yield images.float() * 2.0 - 1.0


def make_proxy_evaluator(
    *,
    args: argparse.Namespace,
    config: AysConfig,
    manifest: AssetManifest,
    dataset_config: dict,
    native_config: dict,
    model: torch.nn.Module,
):
    if not config.early_stop.enabled:
        return None

    if config.early_stop.metric.lower() != "fid":
        raise ValueError(f"Unsupported AYS early-stop metric: {config.early_stop.metric}")

    fid_asset_key = dataset_config.get("default_fid_asset")
    if not fid_asset_key:
        raise ValueError("FID-based AYS early stopping requires `default_fid_asset` in the dataset config.")

    reference_stats = manifest.path(fid_asset_key)
    image_size = int(dataset_config["image_size"])
    proxy_num_samples = int(config.early_stop.proxy_num_samples)
    proxy_batch_size = config.early_stop.batch_size or int(
        dataset_config.get("smoke_batch_size", dataset_config.get("default_batch_size", 128))
    )
    proxy_seed = int(args.seed + config.early_stop.seed_offset)
    schedule_cfg = native_config["Schedule"]

    def evaluate(schedule: torch.Tensor | Path | list | tuple | object, iteration: int) -> float:
        del iteration
        schedule_array = schedule if isinstance(schedule, np.ndarray) else np.asarray(schedule, dtype=np.int64)
        bundle = ScheduleBundle(timesteps=schedule_array[:0:-1].copy())
        scheduler = build_scheduler(
            args.solver,
            diffusion_step=schedule_cfg["diffusion_step"],
            beta_start=schedule_cfg["beta_start"],
            beta_end=schedule_cfg["beta_end"],
            beta_schedule=schedule_cfg["type"],
        )
        with TemporaryDirectory(prefix="ays_proxy_") as proxy_dir:
            run_generation(
                model=model,
                scheduler=scheduler,
                image_size=image_size,
                num_samples=proxy_num_samples,
                batch_size=proxy_batch_size,
                num_inference_steps=len(schedule_array) - 1,
                seed=proxy_seed,
                output_dir=proxy_dir,
                schedule_bundle=bundle,
            )
            return compute_fid(proxy_dir, reference_stats)

    return evaluate


def stage_metadata(hierarchical_result) -> dict[str, dict[str, float | int | bool | None]]:
    payload: dict[str, dict[str, float | int | bool | None]] = {}
    for stage_nfe, result in hierarchical_result.stage_results.items():
        payload[str(stage_nfe)] = {
            "iterations_ran": result.iterations,
            "converged": result.converged,
            "stopped_early": result.stopped_early,
            "best_proxy_value": result.best_proxy_value,
        }
    return payload


def make_progress_reporter(output_root: Path):
    ensure_dir(output_root)
    progress_path = output_root / "_progress.json"
    state: dict[str, object] = {
        "status": "starting",
        "started_at_monotonic": perf_counter(),
    }

    def write_state() -> None:
        payload = dict(state)
        started_at = float(payload.pop("started_at_monotonic"))
        payload["elapsed_seconds"] = round(perf_counter() - started_at, 2)
        dump_json(payload, progress_path)

    def report(event: str, payload: dict[str, object]) -> None:
        state["last_event"] = event
        state.update(payload)
        if event == "stage_start":
            print(
                f"[AYS][stage {payload['stage_steps']}] start "
                f"(active_points={payload['active_points']}, max_iterations={payload['max_iterations']}, "
                f"proxy_metric={'yes' if payload['uses_proxy_metric'] else 'no'})",
                flush=True,
            )
            state["status"] = f"stage_{payload['stage_steps']}_running"
        elif event == "proxy_start":
            print(
                f"[AYS][stage {payload['stage_steps']}] iteration {payload['iteration']}: "
                "running proxy metric evaluation",
                flush=True,
            )
        elif event == "proxy_end":
            print(
                f"[AYS][stage {payload['stage_steps']}] iteration {payload['iteration']}: "
                f"proxy={float(payload['proxy_value']):.4f}, "
                f"best={float(payload['best_proxy_value']):.4f}, "
                f"patience={payload['patience_counter']}/{payload['patience']}",
                flush=True,
            )
        elif event == "iteration_end":
            changed_label = "changed" if payload["changed"] else "no-change"
            proxy_value = payload.get("best_proxy_value")
            proxy_text = "" if proxy_value is None else f", best_proxy={float(proxy_value):.4f}"
            print(
                f"[AYS][stage {payload['stage_steps']}] iteration {payload['iteration']}/{payload['max_iterations']}: "
                f"{changed_label}{proxy_text}",
                flush=True,
            )
        elif event == "stage_stop":
            print(
                f"[AYS][stage {payload['stage_steps']}] stop at iteration {payload['iteration']} "
                f"({payload['reason']})",
                flush=True,
            )
        elif event == "stage_end":
            print(
                f"[AYS][stage {payload['stage_steps']}] done "
                f"(iterations={payload['iterations_ran']}, converged={payload['converged']}, "
                f"stopped_early={payload['stopped_early']}, best_proxy={payload['best_proxy_value']})",
                flush=True,
            )
            state["status"] = f"stage_{payload['stage_steps']}_done"
        elif event == "bundle_saved":
            print(
                f"[AYS] saved nfe={payload['nfe']} from {payload['schedule_source']} -> {payload['output_dir']}",
                flush=True,
            )
        elif event == "summary":
            print(
                f"[AYS] startup: target_nfes={payload['target_nfes']}, initial_steps={payload['initial_steps']}, "
                f"subdivision_rounds={payload['subdivision_rounds']}, data_samples={payload['data_samples']}, "
                f"batch_size={payload['batch_size']}",
                flush=True,
            )
        write_state()

    return report


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    consumer_solvers = parse_solver_list(args.consumer_solvers)
    manifest = AssetManifest(args.manifest)
    dataset_config = load_yaml(args.dataset_config)
    model_asset = args.model_asset or dataset_config["default_model_asset"]
    native_config = load_native_config(dataset_config["native_config"])

    optimizer_payload = load_yaml(args.ays_config).get("optimizer", {})
    optimizer_config = replace(AysConfig.from_dict(optimizer_payload), seed=args.seed)

    dataset_root = dataset_config.get("dataset_root")
    dataset = load_training_dataset(native_config["Dataset"], dataset_root=dataset_root)
    loader = build_reference_loader(
        dataset,
        subset_size=optimizer_config.data_samples,
        batch_size=optimizer_config.batch_size,
        seed=args.seed,
    )
    batch_iterator = cycle_batches(loader)

    def batch_provider(batch_size: int) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        total = 0
        while total < batch_size:
            batch = next(batch_iterator)
            batches.append(batch)
            total += int(batch.shape[0])
        return torch.cat(batches, dim=0)[:batch_size]

    model, _ = load_model(dataset_config["native_config"], manifest.path(model_asset), device=args.device)
    schedule_cfg = native_config["Schedule"]
    scheduler = build_scheduler(
        args.solver,
        diffusion_step=schedule_cfg["diffusion_step"],
        beta_start=schedule_cfg["beta_start"],
        beta_end=schedule_cfg["beta_end"],
        beta_schedule=schedule_cfg["type"],
    )
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError("AYS export requires a scheduler with `alphas_cumprod`.")

    device = torch.device(args.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    sigma_lookup = build_sigma_lookup(scheduler.alphas_cumprod)
    target_nfes = parse_target_nfes(args.target_nfes)
    report_progress = make_progress_reporter(output_root)
    report_progress(
        "summary",
        {
            "target_nfes": target_nfes,
            "initial_steps": optimizer_config.initial_steps,
            "subdivision_rounds": optimizer_config.subdivision_rounds,
            "data_samples": optimizer_config.data_samples,
            "batch_size": optimizer_config.batch_size,
            "dataset": dataset_config["name"],
            "solver": args.solver,
            "model_asset": model_asset,
            "proxy_metric": optimizer_config.early_stop.metric,
            "proxy_num_samples": optimizer_config.early_stop.proxy_num_samples,
        },
    )

    proxy_evaluator = make_proxy_evaluator(
        args=args,
        config=optimizer_config,
        manifest=manifest,
        dataset_config=dataset_config,
        native_config=native_config,
        model=model,
    )

    hierarchical_result = hierarchical_optimize_schedule(
        model=model,
        num_train_timesteps=int(schedule_cfg["diffusion_step"]),
        alphas_cumprod=alphas_cumprod,
        sigma_lookup=sigma_lookup,
        batch_provider=batch_provider,
        config=optimizer_config,
        device=device,
        proxy_evaluator=proxy_evaluator,
        progress_callback=report_progress,
    )
    serialized_stage_meta = stage_metadata(hierarchical_result)

    for nfe in target_nfes:
        consumer_solver = (consumer_solvers or [args.solver])[0]
        consumer_plan = resolve_effective_nfe_plan(consumer_solver, int(nfe))
        schedule, source = schedule_for_nfe(
            target_nfe=consumer_plan.solver_steps,
            hierarchical_result=hierarchical_result,
            sigma_lookup=sigma_lookup,
        )
        terminal_timestep = 0.0
        time_grid = np.concatenate([schedule[:0:-1].copy(), np.asarray([terminal_timestep], dtype=np.float64)])
        bundle = ScheduleBundle(
            timesteps=schedule[:0:-1].copy(),
            time_grid=time_grid,
            meta={
                "schedule_family": "ays",
                "backend": "pndm",
                "dataset": dataset_config["name"],
                "model_asset": model_asset,
                "solver": args.solver,
                "reference_solver": args.solver,
                "consumer_solvers": consumer_solvers or [args.solver],
                "shared_across_solvers": bool(consumer_solvers and any(item != args.solver for item in consumer_solvers)),
                "nfe": int(nfe),
                **consumer_plan.to_meta(),
                "terminal_timestep": terminal_timestep,
                "optimization_method": "paper_hierarchical_coordinate_search",
                "candidate_count": optimizer_config.candidate_count,
                "data_samples": optimizer_config.data_samples,
                "batch_size": optimizer_config.batch_size,
                "sigma_data": optimizer_config.sigma_data,
                "initialization": "time_uniform",
                "initial_steps": optimizer_config.initial_steps,
                "subdivision_rounds": optimizer_config.subdivision_rounds,
                "reference_nfe": optimizer_config.reference_steps,
                "schedule_source": source,
                "stage_results": serialized_stage_meta,
                "early_stop": asdict(optimizer_config.early_stop),
            },
        )
        saved_dir = bundle.save(output_root / f"nfe_{int(nfe):03d}")
        report_progress(
            "bundle_saved",
            {
                "nfe": int(nfe),
                "schedule_source": source,
                "output_dir": str(saved_dir),
            },
        )


if __name__ == "__main__":
    main()

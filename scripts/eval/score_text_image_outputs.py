#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
import torch

from src.utils.assets import AssetManifest
from src.utils.config import load_json, resolve_repo_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    manifest: dict[str, Any]


class ClipScoreScorer:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        local_files_only: bool,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, *, prompts: list[str], image_paths: list[Path], batch_size: int) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(image_paths), batch_size):
            stop = min(start + batch_size, len(image_paths))
            images = [Image.open(path).convert("RGB") for path in image_paths[start:stop]]
            inputs = self.processor(
                text=prompts[start:stop],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs)
            image_embeds = torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
            text_embeds = torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
            batch_scores = torch.clamp((image_embeds * text_embeds).sum(dim=-1), min=0.0) * 100.0
            scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
        return scores


class ImageRewardScorer:
    def __init__(self, *, model_name: str, device: str) -> None:
        try:
            import ImageReward as RM
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ImageReward scoring requires the `ImageReward` package in this environment."
            ) from exc
        self.model = RM.load(model_name, device=device)

    def score(self, *, prompts: list[str], image_paths: list[Path]) -> list[float]:
        scores: list[float] = []
        for prompt, image_path in zip(prompts, image_paths):
            value = self.model.score(prompt, [str(image_path)])
            if isinstance(value, (list, tuple)):
                value = value[0]
            scores.append(float(value))
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score text-to-image run outputs with CLIPScore and ImageReward.")
    parser.add_argument("--manifest", default="configs/assets_manifest.yaml")
    parser.add_argument("--outputs-root", action="append", default=[])
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--prompt-asset", default=None, help="Fallback prompt asset/path when run_manifest lacks prompt_asset.")
    parser.add_argument("--metrics", default="clipscore,imagereward")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--image-reward-model", default="ImageReward-v1.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--allow-missing-metrics", action="store_true", default=False)
    return parser.parse_args()


def selected_metrics(raw: str) -> set[str]:
    aliases = {
        "clip": "clipscore",
        "clip_score": "clipscore",
        "clipscore": "clipscore",
        "image_reward": "imagereward",
        "imagereward": "imagereward",
    }
    metrics = set()
    for item in raw.split(","):
        normalized = item.strip().lower().replace("-", "_")
        if not normalized:
            continue
        if normalized not in aliases:
            raise ValueError(f"Unsupported metric `{item}`.")
        metrics.add(aliases[normalized])
    return metrics


def discover_runs(outputs_roots: Iterable[str], run_dirs: Iterable[str]) -> list[RunRecord]:
    manifest_paths: list[Path] = []
    for raw in run_dirs:
        manifest_paths.append(resolve_repo_path(raw) / "run_manifest.json")
    for raw in outputs_roots:
        root = resolve_repo_path(raw)
        manifest_paths.extend(sorted(root.rglob("run_manifest.json")))
    records: list[RunRecord] = []
    seen: set[Path] = set()
    for manifest_path in manifest_paths:
        resolved = manifest_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        records.append(RunRecord(run_dir=manifest_path.parent, manifest=load_json(manifest_path)))
    return records


def load_prompts(manifest: AssetManifest, run_manifest: dict[str, Any], fallback: str | None) -> list[str]:
    prompt_asset = str(run_manifest.get("prompt_asset") or fallback or "")
    if not prompt_asset:
        raise ValueError(f"Missing prompt_asset for run at {run_manifest.get('output_dir', '')}.")
    prompt_path = manifest.path(prompt_asset) if manifest.has(prompt_asset) else prompt_asset
    prompts = load_json(prompt_path)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt file must contain a non-empty JSON list: {prompt_path}")
    return [str(prompt) for prompt in prompts]


def run_images(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metrics = selected_metrics(args.metrics)
    records = discover_runs(args.outputs_root, args.run_dir)
    if not records:
        raise ValueError("No run_manifest.json files found.")

    clip_scorer = None
    image_reward_scorer = None
    if "clipscore" in metrics:
        clip_scorer = ClipScoreScorer(
            model_name=args.clip_model,
            device=args.device,
            local_files_only=args.local_files_only,
        )
    if "imagereward" in metrics:
        try:
            image_reward_scorer = ImageRewardScorer(model_name=args.image_reward_model, device=args.device)
        except RuntimeError:
            if not args.allow_missing_metrics:
                raise

    asset_manifest = AssetManifest(args.manifest)
    detail_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for record in records:
        images = run_images(record.run_dir)
        if not images:
            continue
        prompts_source = load_prompts(asset_manifest, record.manifest, args.prompt_asset)
        prompts = [prompts_source[index % len(prompts_source)] for index in range(len(images))]

        clip_scores = [None] * len(images)
        image_reward_scores = [None] * len(images)
        if clip_scorer is not None:
            clip_scores = clip_scorer.score(prompts=prompts, image_paths=images, batch_size=args.batch_size)
        if image_reward_scorer is not None:
            image_reward_scores = image_reward_scorer.score(prompts=prompts, image_paths=images)

        base = {
            "backend": record.manifest.get("backend", ""),
            "model_asset": record.manifest.get("model_asset", ""),
            "solver": record.manifest.get("solver", ""),
            "schedule": record.manifest.get("schedule", ""),
            "nfe": record.manifest.get("nfe", ""),
            "seed": record.manifest.get("seed", ""),
            "output_dir": str(record.run_dir),
        }
        for index, image_path in enumerate(images):
            detail_rows.append(
                {
                    **base,
                    "prompt_index": index,
                    "prompt": prompts[index],
                    "image_path": str(image_path),
                    "clip_score": "" if clip_scores[index] is None else clip_scores[index],
                    "image_reward": "" if image_reward_scores[index] is None else image_reward_scores[index],
                }
            )
        aggregate_rows.append(
            {
                **base,
                "num_images": len(images),
                "clip_score_mean": "" if mean(clip_scores) is None else mean(clip_scores),
                "image_reward_mean": "" if mean(image_reward_scores) is None else mean(image_reward_scores),
            }
        )

    write_csv(
        args.output_csv,
        detail_rows,
        [
            "backend",
            "model_asset",
            "solver",
            "schedule",
            "nfe",
            "seed",
            "prompt_index",
            "prompt",
            "image_path",
            "output_dir",
            "clip_score",
            "image_reward",
        ],
    )
    write_csv(
        args.aggregate_csv,
        aggregate_rows,
        [
            "backend",
            "model_asset",
            "solver",
            "schedule",
            "nfe",
            "seed",
            "num_images",
            "output_dir",
            "clip_score_mean",
            "image_reward_mean",
        ],
    )


if __name__ == "__main__":
    main()

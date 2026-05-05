#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parent
PROMPT_ASSET = "diffusers_ablation_prompts"
SOURCE_PATH = REPO_ROOT / "data/pndm/prompts/modern_diffusers_ablation_prompts.json"
RESULT_DIR = PAPER_ROOT / "results/t2i"
PROMPT_CSV = RESULT_DIR / "diffusers_ablation_prompts.csv"
MANIFEST_JSON = RESULT_DIR / "diffusers_ablation_prompts_manifest.json"


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    prompts = json.loads(SOURCE_PATH.read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt asset must be a non-empty JSON list: {SOURCE_PATH}")
    prompt_list = [str(prompt) for prompt in prompts]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with PROMPT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_asset", "prompt_index", "prompt"])
        writer.writeheader()
        for index, prompt in enumerate(prompt_list):
            writer.writerow(
                {
                    "prompt_asset": PROMPT_ASSET,
                    "prompt_index": str(index),
                    "prompt": prompt,
                }
            )

    manifest = {
        "schema_version": 1,
        "prompt_asset": PROMPT_ASSET,
        "source_path": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "paper_prompt_csv": str(PROMPT_CSV.relative_to(REPO_ROOT)),
        "sha256": sha256_bytes(SOURCE_PATH),
        "prompt_count": len(prompt_list),
        "used_by": [
            "configs/experiments/gpde_diffusers_sd15_nfe10_cfg_seed_sweep.yaml",
            "paper/results/t2i/sd15_euler_nfe10_cfg_sweep_detail.csv",
        ],
        "license_status": "project_local_provenance_and_release_license_pending",
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[prompt-manifest] prompts={len(prompt_list)} csv={PROMPT_CSV}")
    print(f"[prompt-manifest] manifest={MANIFEST_JSON}")


if __name__ == "__main__":
    main()

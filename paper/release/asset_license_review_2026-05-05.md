# Asset License Review

Date: 2026-05-05

This is a provenance review for paper release packaging. It is not a final legal
approval.

## AYS Numeric Schedules

Reviewed sources:

- Official AYS project page: `https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/`
- Official AYS quickstart: `https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html`
- Diffusers scheduler constants:
  `https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/schedulers/scheduling_utils.py`
- Diffusers license:
  `https://raw.githubusercontent.com/huggingface/diffusers/main/LICENSE`

Findings:

- The AYS quickstart publishes the 10-step Stable Diffusion 1.5, SDXL,
  DeepFloyd-IF, and Stable Video Diffusion schedule values.
- The fetched AYS project and quickstart pages do not show an explicit license
  or redistribution statement for copying the numeric tables into this archive.
- Diffusers embeds AYS schedule constants in `AysSchedules` inside
  `scheduling_utils.py`, whose source header and repository license are
  Apache-2.0.
- The local `configs/reference_schedules/ays_published_10step.yaml` was
  materialized from the AYS quickstart values. The SD1.5 schedule-profile CSV
  in `paper/results/t2i/` contains AYS numeric schedule values for auditability.
- Author decision on 2026-05-05: the anonymous artifact package should not
  distribute raw AYS numeric schedule files or raw AYS schedule-profile source
  rows. It also omits authorized-offline numeric schedule bundles from the
  CIFAR project-owned offline baseline. AYS remains a cited baseline and
  optional external loader.

Release recommendation:

- Keep AYS metric results and citations in the paper.
- Do not include standalone raw AYS numeric schedule bundles, raw AYS
  schedule-profile source rows, or authorized-offline numeric schedule bundles
  in the anonymous archive.
- Keep AYS metric results, citations, hashes, rendered figures, and optional
  external loader paths in the paper artifacts.

## Prompt Asset

Reviewed local sources:

- `data/pndm/prompts/modern_diffusers_ablation_prompts.json`
- `paper/results/t2i/diffusers_ablation_prompts_manifest.json`
- `paper/results/t2i/diffusers_ablation_prompts.csv`

Findings:

- The paper-facing manifest records a 50-prompt local asset and SHA-256 hashes.
- No external prompt source URL or upstream license is recorded in the current
  manifest.
- The prompts should therefore be treated as project-local text assets until the
  authors confirm authorship and choose a license.

Release recommendation:

- If authors confirm the prompts are original project assets, include them under
  the chosen paper/data license.
- If that decision is not made before an anonymous release, omit prompt text and
  release only prompt asset names, prompt indices, hashes, and metric rows.

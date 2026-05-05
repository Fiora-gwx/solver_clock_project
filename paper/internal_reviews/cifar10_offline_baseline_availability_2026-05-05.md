# CIFAR-10 Offline Baseline Availability Note

Date: 2026-05-05

Scope: whether the current paper can add a strong published AYS/offline
schedule baseline for the retained PNDM/CIFAR-10 Euler NFE 10/20 50k FID
comparison without running another project-owned offline optimizer.

## Source Check

Reviewed sources:

- Official AYS quickstart:
  `https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html`
- Diffusers `AysSchedules` source:
  `https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/schedulers/scheduling_utils.py`
- Local published schedule inventory:
  `configs/reference_schedules/ays_published_10step.yaml`
- Local materialized published schedule bundles:
  `schedules/ays_like/published/`

Findings:

- The official AYS quickstart lists 10-step optimized schedules for Stable
  Diffusion 1.5, SDXL, DeepFloyd-IF Stage-1, and Stable Video Diffusion. It
  does not list a CIFAR-10 schedule.
- Diffusers `AysSchedules` exposes Stable Diffusion, SDXL, and Stable Video
  Diffusion constants. It does not expose a CIFAR-10 schedule.
- The local published AYS inventory and bundles mirror the text-to-image/video
  schedules above: `stable_diffusion_15`, `sdxl`, `deepfloyd_if_stage1`, and
  `stable_video_diffusion`. No `cifar10` published AYS bundle is present.

## Local Offline Attempts

Retained project-owned offline-proxy evidence:

- Lightweight 50k-evaluated proxy baseline:
  `outputs/gpde_pndm_cifar10_50k_offline_proxy_baseline_seeds0_1_2/`
- Medium-budget 5k diagnostic:
  `outputs/gpde_pndm_cifar10_medium_offline_proxy/`
- Interrupted default-budget feasibility attempt:
  `outputs/gpde_pndm_cifar10_default_offline_proxy/`

Current interpretation:

- These attempts are useful negative evidence and feasibility diagnostics.
- They are not published AYS schedules.
- They are not strong offline optimized CIFAR baselines because the lightweight
  and medium attempts produce poor FID, and the default-budget attempt was too
  slow and did not produce a paper-grade evaluated schedule.
- The exporter now saves completed hierarchy stages under `_stage_bundles/`, so
  future long default-budget attempts should preserve completed intermediate
  schedules if a later refinement stage is interrupted.

## Decision

The paper should not claim a published AYS/offline CIFAR comparison for the
current PNDM/CIFAR-10 Euler benchmark. The supported wording remains:

- D-GPDE improves over base and native-coordinate linear schedules in the
  retained PNDM/CIFAR-10 Euler 50k FID setting.
- The Karras-style comparison is mixed.
- Project-owned offline-proxy attempts are negative or incomplete and belong in
  failure analysis.

The only paths to close this baseline blocker are:

1. Find a citable, redistributable published CIFAR schedule for the exact model,
   solver, NFE, and evaluation setting.
2. Run a stronger project-owned offline optimizer to completion, then evaluate
   it with matched seeds and 50k FID.
3. Keep the main empirical claim scoped away from offline-schedule superiority.

# Server Assets

This file records the experiment assets that are intentionally used by the repo-owned experiment pipeline.

## PNDM assets

- CIFAR-10 model checkpoints
  - `checkpoints/pndm/models/ddim_cifar10.ckpt`
  - `checkpoints/pndm/models/pf_cifar10.ckpt`
  - `checkpoints/pndm/models/pf_deep_cifar10.ckpt`
- LSUN checkpoints
  - `checkpoints/pndm/models/ddim_lsun_bedroom.ckpt`
  - `checkpoints/pndm/models/ddim_lsun_church.ckpt`
- FID stats
  - `checkpoints/pndm/fids/fid_cifar10_train.npz`
  - `checkpoints/pndm/fids/fid_bedroom_train.npz`
  - `checkpoints/pndm/fids/fid_church_train.npz`

## Published AYS assets

- Source inventory
  - `configs/reference_schedules/ays_published_10step.yaml`
- Materialized published 10-step bundles
  - `schedules/ays_like/published/stable_diffusion_15/nfe_010`
  - `schedules/ays_like/published/sdxl/nfe_010`
  - `schedules/ays_like/published/deepfloyd_if_stage1/nfe_010`
  - `schedules/ays_like/published/stable_video_diffusion/nfe_010`

## Modern diffusers checkpoints

- `checkpoints/hf/stabilityai--stable-diffusion-3.5-medium`
- `checkpoints/hf/black-forest-labs--FLUX.1-dev`
- `checkpoints/hf/Alpha-VLLM--Lumina-Image-2.0`

## Out of first-pass scope

- Video checkpoints and benchmarks beyond the published Stable Video Diffusion AYS table asset
- Sana as a required first-pass backend

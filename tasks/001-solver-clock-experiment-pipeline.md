# Task 001 - Solver Clock Experiment Pipeline

<!--
说明：
- 文件命名：tasks/NNN-slug.md
- 例如：tasks/015-billing-page-filter.md
- 字段名保持英文，说明与内容可中英混排
- 本文件是单个任务的唯一主记录文件
- 本仓库采用 main-only 执行模式
-->

## Metadata

- Task ID: 001
- Title: Solver clock experiment pipeline
- Slug: solver-clock-experiment-pipeline
- Status: in_progress
- Type: feat
- Priority: high
- Owner: codex
- Reviewer: pending
- Created At: 2026-04-11 22:00
- Updated At: 2026-04-13 11:06
- Execution Branch: main
- Workspace: current repository working tree
- Related Issues: N/A
- Related Commits: N/A
- Dependencies: Project_init.md, align your steps.pdf, 14840_STORK_Faster_Diffusion_a.pdf
- Follow-up Tasks: AYS discrete asset import, Sana metric integration, video benchmarks

---

## Goal

Implement the first runnable experiment pipeline for the continuous clock method using only currently available assets:
CIFAR-10, LSUN-Bedroom, LSUN-Church, and modern diffusers checkpoints.

---

## Background / Context

The repository already contains third-party backends (`STORK`, `diffusers`) and local checkpoints/FID stats, but the
project-owned experiment layer is still empty. We need repo-native manifests, schedule export utilities, backend
adapters, and run scripts so experiments can be executed reproducibly without hard-coded machine paths.

---

## Scope

- Build repo-owned asset manifests and experiment configs.
- Implement unified schedule export/load helpers.
- Implement `V-a` clock construction from calibration traces.
- Implement runnable PNDM and diffusers adapters.
- Add LSUN-Church native config support.

---

## Non-goals

- Do not implement video benchmarks in this task.
- Do not implement the AYS optimizer itself.
- Do not integrate Sana as a required first-pass experiment target.

---

## Approach

Create a thin project layer under `src/`, `configs/`, `scripts/`, and `docs/` that wraps the existing third-party
code. Keep all checkpoint and stat paths in manifest/config files. Use forward hooks on denoiser modules to collect
calibration proxies, then export `V-a` schedules as reusable bundles (`timesteps.npy`, `sigmas.npy`, `meta.json`).

---

## Execution Plan

1. Add task/asset/config/docs scaffolding and register the current assets.
2. Implement schedule bundle I/O, clock construction, and calibration collectors.
3. Implement PNDM and diffusers adapters plus run/export scripts.
4. Run syntax/import verification and a few non-destructive checks.

---

## Test Plan

- Unit tests via direct script/module execution for schedule export/import.
- Syntax verification for all new Python files.
- Non-destructive checks on asset manifest resolution and local checkpoint discovery.

---

## Risks

- Local Python environment may differ from the active conda env for YAML/FID dependencies.
- Modern diffusers schedulers have mixed `timesteps` vs `sigmas` support.
- AYS schedule assets are still absent, so first-pass code must fail explicitly rather than silently fallback.

---

## Open Questions

- None for the first-pass implementation scope.

---

## Plan Self-Review

### Gaps

- Practical quality metrics for modern diffusers are only scaffolded, not fully benchmarked in this task.

### Feasibility

- Feasible within the existing repo because the third-party backends and local assets already exist.

### Risks Review

- The main technical risk is schedule injection differences across scheduler classes, especially in modern diffusers.

### Should This Task Be Split?

- No. The first-pass pipeline is coherent and should land as one infrastructure task.

---

## Approval-Ready Summary

The implementation will create the repo-owned experiment layer required to export schedules, run calibration, inject
custom schedules into PNDM/diffusers backends, and organize assets/configs around the currently available datasets and
checkpoints.

---

## Progress Log

- 2026-04-11 22:00 — Task created with status: in_progress.
- 2026-04-11 22:00 — Confirmed available assets for CIFAR-10, LSUN-Bedroom, LSUN-Church, and modern diffusers checkpoints.
- 2026-04-11 22:18 — Added repo-owned manifests/configs, unified schedule bundle helpers, clock code, backend adapters, and native LSUN-Church config support.
- 2026-04-11 22:34 — Added runtime env registry, environment doctor, config-driven experiment launcher, and script bootstrap so commands do not depend on the caller's active shell env.
- 2026-04-11 22:41 — Validated dry-run expansion for the legacy smoke slice, `nfe_transfer_small`, and `modern_diffusers_practical`.
- 2026-04-11 22:45 — Validated non-destructive PNDM base schedule export to `/tmp/solver_clock_baseline_test` with expected `timesteps.npy` and `meta.json`.
- 2026-04-11 22:47 — Task execution blocked by runtime env issues: `sc-pndm` and `sc-diff` report `torch.cuda.is_available() == False`, and `sc-diff` additionally fails local `DiffusionPipeline` import.
- 2026-04-11 23:04 — Repaired `sc-diff` to `torch 2.5.1+cu121 / torchvision 0.20.1+cu121 / torchaudio 2.5.1+cu121`, removed stale `xformers`, and validated `DiffusionPipeline` import plus CUDA availability.
- 2026-04-11 23:04 — Repaired `sc-pndm` to `torch 2.5.1+cu121 / torchvision 0.20.1+cu121 / torchaudio 2.5.1+cu121`, removed stale `xformers`, and validated CUDA plus backend scheduler/model imports.
- 2026-04-11 23:04 — Re-ran `scripts/run/doctor_runtime_envs.py`; `pndm`, `diffusers`, and `sana` all report status `ok`.
- 2026-04-11 23:07 — Re-validated direct script execution without manual `PYTHONPATH` by exporting a non-destructive PNDM linear schedule bundle to `/tmp/solver_clock_linear_test`.
- 2026-04-11 23:10 — Corrected the legacy CIFAR-10 PNDM smoke model to `ddim_cifar10`, fixed missing `scale_model_input` handling in the PNDM adapter, and completed the original smoke slice for CIFAR-10, LSUN-Bedroom, and LSUN-Church.
- 2026-04-11 23:13 — Exported a CIFAR-10 `V_a` clock sweep for `ddim_cifar10 / euler` and completed a real `V_a` sampling run at `6 NFE`.
- 2026-04-11 23:15 — Completed `sd35_medium / flow_euler / base / 8 NFE` and `sd35_medium / flow_euler / V_a / 8 NFE`, including modern diffusers `V_a` schedule export.
- 2026-04-12 22:02 — Refocused experiment control around YAML-native keys (`num_gpus`, `metrics`, `eval_nfes`) instead of nested execution-only settings.
- 2026-04-12 22:02 — Moved materializable schedule caching to per-experiment records under `outputs/experiment_records/<experiment>/schedules` and added `schedule_cache_manifest.json` generation.
- 2026-04-12 22:02 — Re-verified that `stork4_*` and `flow_stork4_*` solvers remain wired in both experiment configs and local adapters after the config/cache refactor.
- 2026-04-12 22:07 — Real multi-GPU smoke verification completed across four shards after removing unsupported noise-based `stork4_3rd` from the PNDM solver matrix.
- 2026-04-12 22:07 — Real `nfe_transfer_small` execution confirmed the new prepare-then-sample flow: `V_a` cache materialized under `outputs/experiment_records/nfe_transfer_small/schedules` and the run consumed that cached bundle.
- 2026-04-19 10:40 — Imported the published 10-step AYS tables as external assets under `configs/reference_schedules/ays_published_10step.yaml` and `schedules/ays_like/published/...`.
- 2026-04-19 10:40 — Retired the repo-owned PNDM AYS reproduction from active launcher paths; experiment configs now compare AYS only where published assets exist.

---

## Decisions

- Treat `V-a` as the only primary continuous-clock implementation in this task.
- Keep Sana compatibility as future-facing only; do not block first-pass delivery on Sana integration.
- Keep AYS support at the asset-loader level only.

---

## Working Notes

### Files Touched

- tasks/001-solver-clock-experiment-pipeline.md
- configs/assets_manifest.yaml
- configs/runtime_envs.yaml
- configs/datasets/*.yaml
- configs/models/modern_diffusers.yaml
- configs/solvers/*.yaml
- configs/experiments/*.yaml
- docs/RUNNING_EXPERIMENTS.md
- src/utils/*.py
- src/clock/*.py
- src/adapters/*.py
- src/runners/*.py
- scripts/run/*.py
- scripts/eval/compute_fid.py
- third_party/STORK/external/PNDM/config/ddim_church.yml

### Notes

- `checkpoints/pndm` now contains the required CIFAR/Bedroom/Church FID stats and checkpoint files.
- `third_party/STORK/external/PNDM/config` lacks a native Church config and needs one.
- The current project-owned PNDM adapter is validated most thoroughly on `DDIM`-structure checkpoints. `PF` and `PF-deep` CIFAR checkpoints are now wired into the adapter and configs, but still need broader execution coverage.
- Materializable schedules now default to experiment-local caches under `outputs/experiment_records/<experiment>/schedules` instead of the shared top-level `schedules/` tree.
- Upstream `STORKScheduler` supports third-derivative `STORK4` for flow-matching, but not for noise-based PNDM sampling; PNDM experiment configs should therefore stay on `stork4_1st` / `stork4_2nd`.
- Published AYS assets are now the only supported comparison target. Repo-owned PNDM AYS reproduction is no longer active in launcher-managed experiments.

---

## Scope Updates

- None

---

## Blockers

- None

---

## Review Summary

### Baseline

- N/A

### Completed

- Repo-owned experiment layer for assets, configs, schedules, clocks, adapters, runners, and environment-aware launch tooling.

### Validated

- `scripts/run/doctor_runtime_envs.py`
- `scripts/run/doctor_runtime_envs.py` after environment repair
- `scripts/run/run_experiment_config.py --experiment-config configs/experiments/cifar10_partial.yaml --limit 2`
- `scripts/run/run_experiment_config.py --experiment-config configs/experiments/nfe_transfer_small.yaml --limit 2`
- `scripts/run/run_experiment_config.py --experiment-config configs/experiments/modern_diffusers_practical.yaml --limit 2`
- `scripts/run/run_experiment_config.py --experiment-config configs/experiments/cifar10_partial.yaml --outputs-root /tmp/solver_clock_verify_samples --metrics-root /tmp/solver_clock_verify_metrics --execute --limit 4 --skip-existing`
- `scripts/run/run_experiment_config.py --experiment-config configs/experiments/nfe_transfer_small.yaml --outputs-root /tmp/solver_clock_verify_samples --metrics-root /tmp/solver_clock_verify_metrics --execute --limit 1 --skip-existing`
- `scripts/run/export_ays_schedule.py --backend pndm --manifest configs/assets_manifest.yaml --dataset-config configs/datasets/cifar10.yaml --model-asset pndm_model_ddim_cifar10 --solver euler --seed 0 --ays-config configs/clocks/AYS_smoke.yaml --target-nfes 6,8 --output-root /tmp/ays_paper_smoke_v2`
- `scripts/run/export_baseline_schedule.py --backend pndm --dataset-config configs/datasets/cifar10.yaml --mode base --solver euler --nfe 8 --output-dir /tmp/solver_clock_baseline_test`
- `sc-diff`: `torch.cuda.is_available()`, `diffusers.DiffusionPipeline` import, `pip check`
- `sc-pndm`: `torch.cuda.is_available()`, `diffusers.EulerDiscreteScheduler` + `model.ddim.Model` imports, `pip check`
- Syntax check via `py_compile` for new launcher and entrypoint files.
- Legacy smoke slice: CIFAR-10, LSUN-Bedroom, LSUN-Church each wrote `100` images plus `run_manifest.json`
- `nfe_transfer_small`: CIFAR-10 `V_a / euler / 6 NFE` wrote `100` images plus `run_manifest.json` and exported `6..32 NFE` schedule bundles
- `modern_diffusers_practical`: `sd35_medium / flow_euler / base / 8 NFE` and `sd35_medium / flow_euler / V_a / 8 NFE` each wrote `8` images plus `run_manifest.json`

### Uncertain

- Full large-matrix execution remains untested, especially `AYS` comparisons and non-DDIM CIFAR checkpoints (`PF`, `PF-deep`).

---

## Review Issues

| # | 问题描述 | 风险等级 | 影响范围（涉及哪些文件/模块） | 推荐修复方式 |
|---|---|---|---|---|
| 1 | Pending review | low | N/A | Fill after implementation review |

---

## Fix Log

- None

---

## Completion Summary

The repo-side implementation is in place, including environment-aware launch and diagnostics. The previously broken
`sc-pndm` and `sc-diff` CUDA environments were repaired and now pass runtime checks.

---

## Completion Stats

- Execution Branch: main
- Commits: N/A
- Files Changed: repo scaffolding, configs, docs, runtime launcher, doctor script, backend adapters, and runners
- Tests Run: doctor script before/after repair, dry-run launcher checks, non-destructive PNDM schedule export, syntax compilation, direct CUDA/import checks in `sc-pndm` and `sc-diff`
- Tests Run: doctor script before/after repair, dry-run launcher checks, non-destructive schedule export checks, syntax compilation, direct CUDA/import checks in `sc-pndm` and `sc-diff`, full PNDM smoke runs, CIFAR-10 `V_a` run, SD3.5 `base/V_a` runs
- Review Issues Fixed: 0
- Remaining Known Limitations: only the published 10-step AYS assets are supported; `PF`/`PF-deep` CIFAR checkpoints need broader runtime validation across the full matrix; PNDM noise-based STORK remains limited to `stork4_1st` and `stork4_2nd`

---

## Final Commit

- N/A

---

## Final Handoff

- Archive Path: tasks/_archive/001-solver-clock-experiment-pipeline.md
- Suggested Final Commit Message: feat: add solver-clock experiment pipeline scaffolding
- Status: in_progress

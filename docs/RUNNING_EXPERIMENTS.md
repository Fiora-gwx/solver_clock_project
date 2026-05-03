# Running Experiments

## Runtime Mapping

The project uses fixed backend-to-conda mappings from `configs/runtime_envs.yaml`:

- `pndm` -> `sc-pndm`
- `diffusers` -> `sc-diff`
- `sana` -> `sc-sana`

Do not rely on whichever conda env happens to be active in the shell. Use the launcher and doctor scripts below.

## Config-First Execution

Experiment behavior should live in the YAML, not in ad-hoc shell orchestration. The launcher reads keys directly from
`base_config`, including:

- `num_gpus`: how many GPUs to shard sampling across
- `metrics`: metric list such as `[fid]`
- `eval_nfes`: NFE grid to run
- `prepare_schedules_first`: precompute reusable schedule bundles before sampling
- `schedule_cache_root` (optional): override the default per-experiment cache location
- `save_samples`: when `false`, keep only metrics/manifests and discard generated sample images after metric computation
- `schedule_clock_configs`: override materializable clock configs for `FP_CLOCK` or the retained `LEGACY_SADB` comparison

If `schedule_cache_root` is omitted, materialized bundles are stored under the experiment directory:

- `outputs/<experiment_name>/schedules`

The launcher always checks whether a requested materializable bundle already exists in that cache. If it exists, it is reused; if it does not, it is built first and recorded in
`outputs/<experiment_name>/schedule_cache_manifest.json`. Samples, metrics, logs, and generated clock profile caches are also kept under `outputs/<experiment_name>/`.

Current STORK scope:

- PNDM / noise-based runs: `stork4_1st`, `stork4_2nd`
- Diffusers / flow-matching runs: `flow_stork4_1st`, `flow_stork4_2nd`, `flow_stork4_3rd`

## 1. Probe Runtime Environments

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/doctor_runtime_envs.py
```

This checks:

- python executable per backend
- `torch / torchvision / transformers / diffusers` versions
- `torch.cuda.is_available()`
- backend-specific import sanity

## 2. Expand An Experiment Config

Dry-run only:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/fp_clock_cifar10_smoke.yaml
```

This prints the exact commands that will run, and which conda env each command will use.

## 3. Execute With Auto-Materialized Schedules

Example: run the CIFAR-10 partial sweep and auto-generate missing reusable bundles before dispatching sampling.

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/fp_clock_cifar10_smoke.yaml \
  --execute \
  --materialize-schedules
```

Notes:

- `base` does not require a schedule bundle.
- `linear`, `FP_CLOCK`, and the retained `LEGACY_SADB` comparison are materializable schedules. They are checked in the per-experiment cache first, then generated only if missing.
- Old V-a/LCS/RI schedules are removed from the launcher; `FP_CLOCK` is the main target-solver residual clock, and `LEGACY_SADB` is retained for ablations.
- PNDM DPMSolver runs are base-only; the lambda-domain custom clock path is disabled.
- Modern diffusers `FP_CLOCK` is enabled for `flow_euler` in the practical config; flow DPM/STORK entries remain base-only until their native custom-step refinement path is validated.
- `AYS` is treated as an external asset only. Use the published bundles recorded in
  `configs/reference_schedules/ays_published_10step.yaml` and `schedules/ays_like/published/...`.
- Retired SADB/V-series/RI configs live under `configs/ablation/`; they are not active launcher families.

## 4. Useful Slices

CIFAR-10 partial sweep:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/fp_clock_cifar10_smoke.yaml \
  --limit 3
```

CIFAR-10 FP_CLOCK mainline:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/cifar10_mainline_fp_clock.yaml
```

CIFAR-10 FP_CLOCK smoke:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/fp_clock_cifar10_smoke.yaml \
  --execute \
  --materialize-schedules
```

Modern diffusers practical:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/modern_diffusers_practical.yaml
```

## Outputs

- samples: `outputs/<experiment_name>/samples/...`
- metrics: `outputs/<experiment_name>/metrics/<experiment_name>.csv`
- per-run manifest: `run_manifest.json` inside each output directory
- schedules: `outputs/<experiment_name>/schedules`
- schedule cache record: `outputs/<experiment_name>/schedule_cache_manifest.json`
- prepare and dispatch logs: `outputs/<experiment_name>/logs`

## Current Execution Policy

- All task implementation happens directly on `main`
- Only one write-capable task stage should run at a time
- Read-only analysis can run in parallel

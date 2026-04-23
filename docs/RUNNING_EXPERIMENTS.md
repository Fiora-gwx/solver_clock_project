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
- `clock_variants`: expand one experiment YAML across multiple `V_a` clock configs
- `schedule_clock_configs`: override the default materializable clock config per family (`V_a`, `LCS-1`, `LCS-2`)

If `schedule_cache_root` is omitted, materialized bundles are stored under:

- `outputs/experiment_records/<experiment_name>/schedules`

The launcher always checks whether a requested materializable bundle already exists in that cache. If it exists, it is reused; if it does not, it is built first and recorded in
`outputs/experiment_records/<experiment_name>/schedule_cache_manifest.json`. After that, all configured GPUs are used
for sampling.

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
  --experiment-config configs/experiments/cifar10_partial.yaml
```

This prints the exact commands that will run, and which conda env each command will use.

## 3. Execute With Auto-Materialized Schedules

Example: run the CIFAR-10 partial sweep and auto-generate missing reusable bundles before dispatching sampling.

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/cifar10_partial.yaml \
  --execute \
  --materialize-schedules
```

Notes:

- `base` does not require a schedule bundle.
- `linear`, `V_a`, `LCS-1`, and `LCS-2` are materializable schedules. They are checked in the per-experiment cache first, then generated only if missing.
- `V_a` remains the historical proxy baseline.
- `LCS-1` and `LCS-2` are the primary theory-backed schedule families. They only replace schedule nodes and do not modify the online solver itself.
- `AYS` is treated as an external asset only. Use the published bundles recorded in
  `configs/reference_schedules/ays_published_10step.yaml` and `schedules/ays_like/published/...`.
- The `V_b` / `A_a` / `A_b` families are still treated as external assets.

## 4. Useful Slices

CIFAR-10 partial sweep:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/cifar10_partial.yaml \
  --limit 3
```

Small-model schedule study:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/schedule_vs_clock_small.yaml
```

CIFAR-10 mainline:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/cifar10_mainline.yaml
```

CIFAR-10 partial:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/cifar10_partial.yaml \
  --execute \
  --materialize-schedules
```

Modern diffusers practical:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/modern_diffusers_practical.yaml
```

LCS confirmation on PNDM:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/lcs_confirmation_pndm.yaml
```

LCS confirmation on diffusers:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/lcs_confirmation_diffusers.yaml
```

V_a ablation stage A:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/cifar10_va_ablation_selection.yaml \
  --execute \
  --materialize-schedules \
  --skip-existing
```

Compose the phase-B `A-winners-combined` clock config from the stage-A metrics:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/compose_va_winner_config.py \
  --selection-config configs/experiments/cifar10_va_ablation_selection.yaml \
  --metrics-csv outputs/metrics/cifar10_va_ablation_selection.csv \
  --output configs/clocks/V_a_combo_awinners.yaml
```

V_a ablation stage B:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/cifar10_va_ablation_confirmation.yaml \
  --execute \
  --materialize-schedules \
  --skip-existing
```

## Outputs

- samples: `outputs/samples/<experiment_name>/...`
- metrics: `outputs/metrics/<experiment_name>.csv`
- per-run manifest: `run_manifest.json` inside each output directory
- schedule cache: `outputs/experiment_records/<experiment_name>/schedules`
- schedule cache record: `outputs/experiment_records/<experiment_name>/schedule_cache_manifest.json`

## Current Execution Policy

- All task implementation happens directly on `main`
- Only one write-capable task stage should run at a time
- Read-only analysis can run in parallel

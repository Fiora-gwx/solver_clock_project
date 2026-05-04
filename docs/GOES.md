# GOES Usage Notes

GOES (Geometry-aware Oracle Edge Scheduling) is a training-free schedule
optimizer. It builds a high-accuracy deterministic oracle trajectory for a
calibration split, evaluates solver-specific edge defects from oracle states,
and exports a repository `ScheduleBundle`.

## CPU Smoke

These commands use the toy ODE backend and do not load diffusion checkpoints.

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python -m goes.experiment_runner build-oracle \
  --config configs/goes/smoke.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python -m goes.experiment_runner search-schedule \
  --config configs/goes/smoke.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python -m goes.experiment_runner evaluate \
  --config configs/goes/smoke.yaml \
  --schedule outputs/goes/<run_dir>/schedule.json

/home/gwx/miniconda3/envs/sc-diff/bin/python -m goes.experiment_runner oracle-convergence \
  --config configs/goes/smoke.yaml --values 8,16

/home/gwx/miniconda3/envs/sc-diff/bin/python -m goes.experiment_runner cross-solver-reuse \
  --config configs/goes/smoke.yaml --solvers euler,heun,midpoint
```

The smoke runner writes `schedule.json`, `edge_costs.npz`,
`selected_edges.csv`, calibration and held-out CSVs, and `paper_tables/`.
Toy results are latent replay errors only; they are not image-quality evidence.

## Real Schedule Materialization

The main launcher can preview GOES schedules without loading models:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_smoke.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_smoke.yaml
```

Wider, non-smoke collection configs are available for data collection once GPU
time is explicitly allocated:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_nfe_sweep.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_cfg_nfe_sweep.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_solver_comparison_odd_nfe.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_flow_solver_comparison_odd_nfe.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_rho_metric_ablation.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_rho_metric_ablation.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_calibration_size_ablation.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_calibration_size_ablation.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_candidate_grid_ablation.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_candidate_grid_ablation.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_oracle_convergence.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_oracle_convergence.yaml

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_models_without_published_schedules.yaml
```

To actually build schedules and run generation, pass `--execute` and enable
schedule materialization in the config or with `--materialize-schedules`. This
loads checkpoints and should be treated as a GPU experiment.

`goes_diffusers_models_without_published_schedules.yaml` intentionally compares
only `base` and `GOES` for SD3.5 Medium, FLUX.1-dev, and Lumina-Image-2.0,
because this repository does not declare published AYS schedule assets for
those model keys.

GOES ablation configs use `GOES[label]` schedule selectors with
`schedule_clock_configs.GOES.variants`. Variant YAMLs in
`configs/goes/variants/` override the top-level `goes:` exporter options for
that labeled schedule, so rho and replay-metric sweeps get distinct schedule
directories and prepare commands. The metric variants cover `identity`,
`edm_scalar`, and `channel_whitened`. The candidate-grid variants cover
M={64,128,256,512,1024}; the calibration-size variants cover
K={4,8,16,32,64,128}. The oracle-convergence variants cover reference NFE
values {100,200,500,1000}.

## Direct Exporters

PNDM deterministic velocity ODE schedules:

```bash
/home/gwx/miniconda3/envs/sc-pndm/bin/python scripts/run/export_goes_pndm_schedule.py \
  --dataset-config configs/datasets/cifar10.yaml \
  --model-asset pndm_model_ddim_cifar10 \
  --solver euler \
  --nfe 4 \
  --output-dir outputs/goes_pndm_smoke/schedules/GOES/pndm/cifar10/pndm_model_ddim_cifar10/euler/nfe_004
```

Diffusers flow schedules:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/export_goes_diffusers_schedule.py \
  --model-asset hf_sd35_medium \
  --prompt-asset diffusers_smoke_prompts \
  --solver flow_euler \
  --nfe 4 \
  --output-dir outputs/goes_diffusers_smoke/schedules/GOES/diffusers/sd35_medium/flow_euler/cfg_3.5/nfe_004
```

Add `--dry-run` to either exporter to validate arguments without loading a
model or pipeline. Dry-runs also print deterministic seed metadata and the
resolved model/prompt paths that will be recorded by real materialization.
Non-dry-run materialization writes `config.resolved.yaml` next to
`schedule.json`, `oracle_metadata.json`, and `run_metadata.json`.

## Schedule Verification

After materializing a GOES schedule, verify the standalone schedule payload and
the exported `ScheduleBundle` before generation:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/verify_goes_schedule.py \
  --schedule outputs/goes_pndm_smoke/schedules/GOES/pndm/cifar10/pndm_model_ddim_cifar10/euler/nfe_004/schedule.json \
  --bundle-dir outputs/goes_pndm_smoke/schedules/GOES/pndm/cifar10/pndm_model_ddim_cifar10/euler/nfe_004
```

The verifier checks the GOES method/version, number of function evaluations,
strictly increasing unified schedule, selected edge count, oracle cache key,
schedule hash, bundle metadata, and schedule array lengths.

The main experiment launcher uses the same checks when deciding whether an
existing GOES materialized schedule directory is current. A directory with only
`meta.json`, stale version metadata, missing `schedule.json`, or inconsistent
bundle arrays is treated as stale and will be rebuilt when schedule
materialization is enabled.

## Cost Accounting

GOES PNDM and diffusers exporters write `calibration_cost_estimate`,
`calibration_cost_unit`, and `calibration_cost_breakdown` into `schedule.json`
and bundle `meta.json`. The estimate is in model-evaluation equivalents and
includes RK4 oracle drift calls, tangent evaluations, and one-step edge replay
calls. For diffusers classifier-free guidance, the estimate multiplies by two
when guidance scale differs from `1.0`. The estimate excludes production image
generation.

After materializing multiple GOES schedules, summarize solver and NFE reuse of
the same universal oracle cache into a paper-table CSV:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/report_goes_oracle_reuse_cost.py \
  outputs/goes_diffusers_cfg_nfe_sweep/schedules/GOES \
  --output-csv outputs/goes_diffusers_cfg_nfe_sweep/paper_tables/oracle_reuse_cost.csv
```

The report groups rows by `oracle_cache_key` and compares model-evaluation
equivalents for separate-oracle materialization against a shared universal
oracle plus per-solver edge evaluation. It fails if no GOES schedules with cost
metadata are found.

When `run_experiment_config.py --execute` finishes a GOES experiment in a
single parent process, it runs the same report automatically if a materialized
GOES schedule root exists.

## Theory Coverage

The deterministic-oracle interpretation is strict for one-step deterministic
velocity ODE solvers where the wrapped velocity function is the target ODE.
The current repository exporters cover:

- PNDM backend: `euler` and `heun2` deterministic velocity steps.
- Diffusers flow pipelines: `flow_euler` and `flow_heun` velocity-style
  proxies for flow models.

The diffusers VP/SD empirical `euler` path can export schedules, but it is not
strictly solver-independent probability-flow theory because the adapter exposes
scheduler model outputs rather than a clean ODE drift.

DPM, UniPC, SDE, and other scheduler-history or multistep modes are rejected by
the GOES diffusers exporter until scheduler-history replay refinement is wired
through the real pipeline path.

## Reporting

Do not report GOES as improving quality from toy smoke runs. Paper claims need
real generation outputs with matched prompt/seed pairs and the requested image
metrics, including CLIPScore, ImageReward, and pairwise win rates when
available. Record model, solver, number of function evaluations (NFE),
guidance scale, prompt asset, seed count, schedule directory, and GOES oracle
cache key in every table.

PNDM KID is available only when the reference `.npz` contains real-image
feature activations (`features`, `activations`, `pool_3`, or `real_features`).
FID-only stats with just `mu`/`sigma` are rejected for KID rather than used as a
proxy.

The text-image scoring scripts fail if no scorable images are found or if
pairwise scoring cannot form any matched schedule comparisons. Empty scoring
CSVs are not valid experiment evidence. Aggregate CLIPScore/ImageReward rows
and pairwise win-rate summaries include deterministic bootstrap standard errors
and confidence intervals by default; pass `--bootstrap-samples 0` only for a raw
point-estimate diagnostic.

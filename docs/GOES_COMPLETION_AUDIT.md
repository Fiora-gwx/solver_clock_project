# GOES Completion Audit

This audit tracks the current implementation against `goal.md`. It is not a
claim that the goal is complete.

## Objective

Implement and validate GOES (Geometry-aware Oracle Edge Scheduling): a
training-free schedule optimizer that builds a solver-independent deterministic
oracle, evaluates solver-specific edge defects, solves a discrete min-max
schedule problem, exports schedule bundles, and produces paper-usable metrics,
ablations, and tables if the method works.

## Prompt-to-Artifact Checklist

| `goal.md` requirement | Current evidence | Status |
| --- | --- | --- |
| Training-free optimizer; do not change model weights | GOES code only materializes schedules and metadata; no training loop or weight write path is introduced | Covered by implementation scope |
| `goes/` modules or equivalent functionality | `goes/config.py`, `coordinate.py`, `oracle.py`, `oracle_cache.py`, `interpolation.py`, `metrics.py`, `mixed_defect.py`, `edge_evaluator.py`, `dp_minimax.py`, `replay_refinement.py`, `schedules.py`, `logging_utils.py`, `experiment_runner.py` | Covered |
| Config schema with oracle/calibration/candidate/solver/metric/defect/aggregation/output sections | `configs/goes/default.yaml`, `configs/goes/euler_10nfe.yaml`, `configs/goes/smoke.yaml`; loaded through `goes.config.load_config`; validation rejects unsupported CPU-runner options for model shape, coordinate, solver name/mode, oracle integrator/interpolation, candidate grid type, metric parameters, aggregation, optimizer, replay-refinement bounds, image saving, and missing required outputs | Covered for CPU GOES configs |
| CLI commands: `build-oracle`, `search-schedule`, `evaluate`, ablations | `goes.experiment_runner` implements these subcommands; `tests/test_goes_core.py` covers `build-oracle -> search-schedule -> evaluate`; sweep tests cover NFE, rho, metric, calibration size, candidate grid, oracle convergence, and cross-solver reuse | Covered for CPU toy backend |
| Required run files: schedule, edge table, selected edges, calibration/held-out metrics, oracle/run metadata, paper tables | CPU smoke integration test asserts required files exist; `evaluate` writes a combined calibration/held-out `oracle_metadata.json`; `maybe_write_plots` writes plots when matplotlib is available | Covered for CPU toy backend |
| Output config behavior | `output.save_plots: false` is respected by `_search_once` and `nfe-sweep`; smoke/sweep tests assert `plots_written` is empty under no-plot configs, and `sc-diff` verifies `schedule.png`, `edge_cost_heatmap.png`, `selected_edge_costs.png`, and `nfe_quality_curve.png` when matplotlib is available; `output.save_schedule: false` and `output.save_edge_table: false` are rejected because schedule and edge tables are required GOES outputs | Covered for CPU toy backend |
| Schedule JSON fields | `goes/schedules.py`; verifier and tests check method/version/solver/NFE/coordinate/u/native schedule/hash/edge objective/edge fields | Covered |
| Mixed defect formula and tiny tangent fallback | `goes/mixed_defect.py`; exact formula tests cover `rho=0`, `rho=1`, sign flip, non-negativity, fallback, tangential error detection; edge-table, replay-metric, and replay-refinement paths consistently honor `fallback_full_residual_on_tiny_tangent` | Covered |
| Metric choices: `identity`, `edm_scalar`, `channel_whitened` | `goes/metrics.py`; unit tests cover EDM scalar weighting, negative-sigma/interpolated sigma mapping, and channel-whitened weighting; real PNDM/diffusers GOES exporters accept all three metric names and `--sigma-data` | Covered |
| Robust aggregation: mean/median/trimmed mean/CVaR | `goes/aggregation.py`; unit tests cover all required modes | Covered |
| DP min-max recurrence and tie-breaking | `goes/dp_minimax.py`; tests compare against brute force and check tie-break keeps the primary optimum | Covered |
| Coordinate adapter and native conversion | `goes/coordinate.py`; tests cover coordinate round-trip and schedule verifier checks monotone unified schedules | Covered for implemented coordinate families |
| Universal oracle cache and reuse across solvers | `goes/oracle_cache.py`; cache key tests cover model/coordinate/ref-NFE/seed/interpolation changes and dummy two-solver reuse; unsupported oracle interpolation configs are rejected instead of silently reusing linear caches | Covered for CPU toy backend |
| Theory coverage marking | CPU empirical-only solver path records `deterministic_oracle_theory: false` and an empirical-only coverage note in `run_metadata.json` | Covered for CPU toy backend and exporter guardrails |
| Real-backend schedule exporters | `scripts/run/export_goes_pndm_schedule.py`, `scripts/run/export_goes_diffusers_schedule.py`; dry-run and unit tests cover argument expansion, numeric argument validation, deterministic seed metadata, model/prompt path metadata, resolved export config writing, cost accounting, and explicit held-out `NOT_EVALUATED` metric rows without loading models | Partially covered; real GPU materialization not run |
| Repository launcher integration | `scripts/run/run_experiment_config.py`; preview commands expand PNDM and diffusers GOES prepare/run steps, labeled `GOES[...]` variants merge variant YAML overrides into exporter arguments, shared prepare steps are shown once with reuse markers for duplicate run invocations, and executed single-process GOES experiments run the oracle-reuse cost report after materialization/scoring when a GOES schedule root exists | Covered for preview/post-run wiring; real execution not run |
| Schedule verification before generation/evaluation | `goes/verify.py`, `scripts/run/verify_goes_schedule.py`; verifier rejects missing solver, coordinate, coordinate direction, edge objective, non-monotone unified schedules, and non-monotone ScheduleBundle timestep/sigma arrays; staleness tests verify incomplete/stale GOES schedule dirs rebuild/fail validation; `evaluate` verifies `schedule.json` and rejects solver/NFE/config mismatches before oracle construction | Covered |
| Baseline policy and skip reasons | CPU run metadata records `uniform_in_u` and skip reasons for unavailable AYS/image metrics; smoke test asserts these entries | Covered for CPU toy backend; real experiment baselines still need run evidence |
| Reproducibility logging and split separation | `run_metadata.json` records run directory, runtime, resolved config path, deterministic Python/NumPy/PyTorch seed setup, model identifier/name/checkpoint path, calibration/held-out split hashes, initial-noise hashes, noise-seed hashes, and noise seed lists; `build-oracle` records both run-local and cache oracle metadata paths; oracle metadata consistently records `oracle_cache_key` for newly built and cache-loaded oracles; top-level sweep/ablation commands also write `config.resolved.yaml`, deterministic seed metadata, and model metadata; CLI tests assert resolved config path, seed/model metadata plus calibration/held-out hash separation, and torch oracle tests assert seed metadata survives cache reuse; real exporter unit tests verify deterministic seed, model path, prompt path, dataset config path metadata, and `config.resolved.yaml` writing | Covered for CPU toy backend; real backend metadata is dry-run/unit verified but not execution verified |
| Paper-grade experiment tables and ablations | CPU runner writes raw CSVs for main results, NFE sweep, rho/metric/calibration/candidate-grid ablations, oracle convergence, and oracle reuse; main/NFE rows include deterministic bootstrap standard error and 95% CI for final latent MSE; oracle convergence includes edge-cost rank correlation; ablation tables include runtime, held-out generalization gap, schedule stability, and candidate-grid edge/DP timings where applicable; cross-solver reuse records skip reasons for unavailable solvers and amortized shared-vs-separate oracle cost fields; table outputs and top-level reproducibility metadata are covered by CLI regression tests | Implemented for CPU toy backend; real paper data not collected |
| Failure analysis diagnostics | `failure_cases.csv` records held-out underperformance rows, selected schedule, edge objective, GOES/baseline replay endpoint MSE traces, and tiny tangent fallback fraction; regression tests verify the no-underperformance case writes an explicit summary row instead of a false failure | Covered for CPU toy backend; image failure artifacts require real generation |
| At least one main benchmark end-to-end | No real model schedule materialization, generation, or scoring has been run | Missing; requires explicit GPU experiment approval |
| Image metrics: FID/KID/CLIPScore/ImageReward/pairwise | Scoring scripts exist for PNDM FID, PNDM KID when the reference `.npz` contains feature activations, diffusers CLIPScore/ImageReward, and paired win rates; text-image scoring rejects runs with no scorable images, and pairwise scoring rejects empty matched-comparison outputs. Aggregate CLIPScore/ImageReward rows and pairwise summaries include deterministic bootstrap standard errors and confidence intervals. Pairwise regression tests verify `GOES` and labeled `GOES[...]` schedules are compared against base and AYS when matched detail rows exist. No GOES image outputs exist in this workspace | Missing real scored outputs; KID requires reference feature activations, not FID-only `mu/sigma` stats |
| Multi-seed/NFE/CFG/model sweeps | Non-executed configs now exist for PNDM NFE sweeps, diffusers CFG/NFE sweeps, odd-NFE solver comparisons covering PNDM `euler/heun2` plus diffusers `flow_euler/flow_heun`, PNDM/diffusers rho-metric ablations, PNDM/diffusers calibration-size ablations, PNDM/diffusers candidate-grid ablations, PNDM/diffusers oracle-convergence sweeps using labeled GOES variants, and diffusers models without published AYS assets; launcher preview and regression tests expand GOES materialization and generation commands | Missing execution for paper-grade real sweeps |
| Multistep real black-box replay refinement | CPU `goes/replay_refinement.py` exists and is tested; runner recomputes selected edge costs/objective for the final refined schedule; real diffusers/PNDM scheduler-history modes are rejected or empirical-only | Partial; real pipeline replay refinement is not implemented |

## Implemented Evidence

| Requirement | Evidence |
| --- | --- |
| Core GOES modules | `goes/config.py`, `coordinate.py`, `oracle.py`, `oracle_cache.py`, `metrics.py`, `mixed_defect.py`, `edge_evaluator.py`, `dp_minimax.py`, `replay_refinement.py`, `schedules.py`, `experiment_runner.py`, `torch_backend.py` |
| Mixed normal defect with nonzero default `rho` | `goes/mixed_defect.py`; default `rho: 0.1` in `configs/goes/default.yaml` |
| Robust aggregation | `goes/aggregation.py`; default `trimmed_mean` with `trim_ratio: 0.10`; mean/median/trimmed mean/CVaR unit test coverage in `tests/test_goes_core.py` |
| Discrete min-max DP with brute-force tests | `goes/dp_minimax.py`; `tests/test_goes_core.py` |
| Universal oracle cache | `goes/oracle_cache.py` and `goes/torch_backend.py`; cache keys include model, ODE family, coordinate mapping, reference integrator, interpolation, reference grid hash, split/noise hashes, CFG metadata, dtype, and device |
| Edge evaluator starts from `x_star(a)` | `goes/edge_evaluator.py`; regression test in `tests/test_goes_core.py` |
| Metrics | `identity`, `edm_scalar`, and `channel_whitened` in `goes/metrics.py`; scalar sigma and channel-whitened weighting are unit tested |
| Black-box replay refinement fallback | `goes/replay_refinement.py`; CPU regression tests cover prefix-dependent replay improvement and final-schedule edge-cost recomputation after refinement |
| CPU smoke runner | `python -m goes.experiment_runner search-schedule --config configs/goes/smoke.yaml` |
| CPU oracle/evaluation CLIs | `build-oracle`, `search-schedule`, and `evaluate` are covered by an end-to-end CLI regression test in `tests/test_goes_core.py`; the evaluate path writes held-out metrics, paper table rows, run metadata, and combined calibration/held-out oracle metadata |
| CPU ablation/sweep commands | `oracle-convergence`, `nfe-sweep`, `ablate-rho`, `ablate-metric`, `calibration-size-ablation`, `candidate-grid-ablation`, `cross-solver-reuse` in `goes/experiment_runner.py` |
| Required toy output files | Smoke tests check `schedule.json`, `schedule_native.json`, `edge_costs.npz`, `selected_edges.csv`, `failure_cases.csv`, calibration/held-out CSVs, metadata, and `paper_tables/` |
| ScheduleBundle export | `goes/repository_schedules.py`, `scripts/run/export_goes_schedule.py`; exporter reuses full GOES payload verification before materializing a bundle, rejects incomplete schedule JSON, and CLI accepts `--schedule-json` and `--schedule` |
| Schedule metadata completeness | PNDM/diffusers exporters and generic bundle export preserve seed, guidance, pilot/calibration config, oracle config, candidate grid config, model/prompt identifiers, and cache metadata where applicable |
| Real-backend exporter validation | PNDM/diffusers exporters reject invalid NFE, batch counts, microbatch size, reference grids, candidate grids, defect mixing parameters, robust aggregation parameters, and backend-specific numeric parameters before loading models |
| Real-backend exporter metric files | PNDM/diffusers exporters write calibration replay metrics separately from held-out status rows; `paper_tables/main_results.csv` is marked `NOT_EVALUATED` by schedule export so calibration replay metrics are not presented as held-out image quality |
| Real-backend oracle reuse cost report | `scripts/report_goes_oracle_reuse_cost.py` reads materialized GOES schedule directories, groups by `oracle_cache_key`, and writes shared-oracle vs separate-oracle model-evaluation-equivalent cost rows; tests cover shared-cache aggregation and empty-root rejection |
| Schedule verification before generation/evaluation | `goes/verify.py` and `scripts/run/verify_goes_schedule.py` check GOES payloads and optional `ScheduleBundle` directories, including strict monotonicity of materialized timestep/sigma arrays; `evaluate` reuses the payload verifier and checks config compatibility before oracle construction |
| Repository launcher integration | `GOES` schedule support in `scripts/run/run_experiment_config.py`; labeled `GOES[...]` variants merge `schedule_clock_configs.GOES.variants` YAML overrides into exporter arguments; preview output de-duplicates shared prepare commands while preserving reuse markers |
| Real experiment collection configs | `configs/experiments/goes_pndm_nfe_sweep.yaml`, `configs/experiments/goes_diffusers_cfg_nfe_sweep.yaml`, `configs/experiments/goes_pndm_solver_comparison_odd_nfe.yaml`, `configs/experiments/goes_diffusers_flow_solver_comparison_odd_nfe.yaml`, `configs/experiments/goes_pndm_rho_metric_ablation.yaml`, `configs/experiments/goes_diffusers_rho_metric_ablation.yaml`, `configs/experiments/goes_pndm_calibration_size_ablation.yaml`, `configs/experiments/goes_diffusers_calibration_size_ablation.yaml`, `configs/experiments/goes_pndm_candidate_grid_ablation.yaml`, `configs/experiments/goes_diffusers_candidate_grid_ablation.yaml`, `configs/experiments/goes_pndm_oracle_convergence.yaml`, `configs/experiments/goes_diffusers_oracle_convergence.yaml`, and `configs/experiments/goes_diffusers_models_without_published_schedules.yaml`; launcher tests validate invocation counts, GOES prepare-step counts, seed/guidance/NFE path partitioning, theory-covered solver coverage, shared oracle cache dirs across real solver-comparison prepare commands, labeled variant overrides, base-vs-GOES-only coverage for SD3.5/FLUX/Lumina models without published AYS assets, and exporter arguments without loading models; main NFE sweeps cover N={4,5,6,8,10,12,15,20,30,50}; rho-metric ablations cover `identity`, `edm_scalar`, and `channel_whitened`; calibration-size ablations cover K={4,8,16,32,64,128}; candidate-grid ablations cover M={64,128,256,512,1024}; oracle-convergence sweeps cover ref-NFE={100,200,500,1000} |
| GOES pairwise scoring support | `scripts/eval/pairwise_win_rates.py`; `tests/test_eval_pairwise.py` verifies `GOES` and lowercase labeled `goes[...]` detail rows normalize to GOES labels and produce paired win-rate comparisons versus base and AYS |
| Scoring guardrails | `scripts/eval/score_text_image_outputs.py` rejects discovered runs with no image files before writing detail/aggregate CSVs; `scripts/eval/pairwise_win_rates.py` rejects empty pairwise outputs when matched schedule pairs or numeric metrics are absent |
| Metric validation guardrails | `scripts/run/run_experiment_config.py`; launcher tests accept PNDM `kid`, reject unsupported backend metrics such as PNDM `clipscore` and diffusers `fid` instead of silently skipping them, and accept `clip_score`/`image-reward` aliases |
| Launcher guardrails | PNDM GOES rejects non-`euler/heun2`; diffusers GOES rejects unsupported flow/VP solver combinations before prepare execution |
| Launcher stale schedule detection | GOES materialized schedule dirs are treated as current only if `verify_goes_schedule.py`-equivalent checks pass for `schedule.json` and the bundle |
| Schedule path partitioning | Launcher tests verify GOES diffusers schedule directories are unique across guidance scale, seed, and NFE, and PNDM schedule directories are unique across model asset, solver, seed, and NFE |
| Real-backend GOES exporters | `scripts/run/export_goes_pndm_schedule.py` and `scripts/run/export_goes_diffusers_schedule.py` |
| Schedule staleness versioning | `src/clock/goes.py`; GOES bundles write `schedule_implementation_version` |
| Usage and theory notes | `docs/GOES.md` |

## Verified Commands

These checks have passed in the current workspace:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python -m pytest \
  tests/test_goes_core.py -q
# 51 passed, 2 warnings

/home/gwx/miniconda3/envs/sc-diff/bin/python -m pytest \
  tests/test_goes_core.py tests/test_eval_pairwise.py \
  tests/test_score_text_image_outputs.py tests/test_goes_oracle_reuse_report.py \
  tests/test_defect_clock_launcher.py tests/test_schedule_prepare_staleness.py \
  tests/test_kid_metric.py tests/test_results_csv.py -q
# 106 passed, 2 warnings

/home/gwx/miniconda3/envs/sc-diff/bin/python -m pytest \
  tests/test_goes_oracle_reuse_report.py -q
# 2 passed

/home/gwx/miniconda3/envs/sc-pndm/bin/python -m pytest \
  tests/test_goes_core.py -q
# 46 passed, 2 skipped

/home/gwx/miniconda3/envs/sc-diff/bin/python -m pytest \
  tests/test_defect_clock_launcher.py tests/test_schedule_prepare_staleness.py -q
# 32 passed

/home/gwx/miniconda3/envs/sc-diff/bin/python -m pytest \
  tests/test_eval_pairwise.py -q
# 2 passed

/home/gwx/miniconda3/envs/sc-diff/bin/python -m pytest \
  tests/test_eval_pairwise.py tests/test_score_text_image_outputs.py -q
# 3 passed

/home/gwx/miniconda3/envs/sc-diff/bin/python -m pytest \
  tests/test_eval_pairwise.py tests/test_score_text_image_outputs.py \
  tests/test_defect_clock_launcher.py tests/test_schedule_prepare_staleness.py -q
# 41 passed

/home/gwx/miniconda3/envs/sc-diff/bin/python -m pytest \
  tests/test_pndm_schedule_support.py tests/test_diffusers_schedule_support.py -q
# 25 passed, warnings only

/home/gwx/miniconda3/envs/sc-pndm/bin/python -m pytest \
  tests/test_pndm_schedule_support.py -q
# 22 passed, warnings only

/home/gwx/miniconda3/envs/sc-diff/bin/python -m py_compile \
  goes/*.py scripts/run/export_goes_schedule.py \
  scripts/run/export_goes_pndm_schedule.py \
  scripts/run/export_goes_diffusers_schedule.py \
  scripts/run/verify_goes_schedule.py scripts/report_goes_oracle_reuse_cost.py \
  scripts/run/run_experiment_config.py scripts/run/run_pndm_experiment.py \
  src/runners/pndm_experiment.py src/utils/fid.py src/utils/results.py src/clock/goes.py
# passed

/home/gwx/miniconda3/envs/sc-pndm/bin/python -m py_compile \
  goes/*.py scripts/run/export_goes_schedule.py \
  scripts/run/export_goes_pndm_schedule.py \
  scripts/run/export_goes_diffusers_schedule.py \
  scripts/run/verify_goes_schedule.py scripts/report_goes_oracle_reuse_cost.py \
  scripts/run/run_experiment_config.py scripts/run/run_pndm_experiment.py \
  src/runners/pndm_experiment.py src/utils/fid.py src/utils/results.py src/clock/goes.py
# passed

git diff --check
# passed

find . -type d -name __pycache__ -print
# no output
```

Launcher previews also expand GOES prepare and run commands:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_smoke.yaml --limit 4

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_smoke.yaml --limit 4

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_nfe_sweep.yaml --limit 40

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_cfg_nfe_sweep.yaml --limit 32

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_solver_comparison_odd_nfe.yaml --limit 18

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_flow_solver_comparison_odd_nfe.yaml --limit 18

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_rho_metric_ablation.yaml --limit 18

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_rho_metric_ablation.yaml --limit 18

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_calibration_size_ablation.yaml --limit 21

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_calibration_size_ablation.yaml --limit 21

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_candidate_grid_ablation.yaml --limit 18

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_candidate_grid_ablation.yaml --limit 18

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_oracle_convergence.yaml --limit 15

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_oracle_convergence.yaml --limit 15

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_diffusers_models_without_published_schedules.yaml --limit 18
```

CPU `build-oracle`, `search-schedule`, and `evaluate` entry points are covered
by `tests/test_goes_core.py::test_goes_build_oracle_search_and_evaluate_cli`.

Direct exporter dry-runs and unit tests have validated argument expansion,
numeric argument validation, and calibration-cost reporting without loading
models or pipelines:

```bash
/home/gwx/miniconda3/envs/sc-pndm/bin/python scripts/run/export_goes_pndm_schedule.py \
  --dataset-config configs/datasets/cifar10.yaml --model-asset pndm_model_ddim_cifar10 \
  --solver euler --nfe 4 --output-dir outputs/goes_pndm_smoke/schedules/GOES/pndm/cifar10/pndm_model_ddim_cifar10/euler/nfe_004 \
  --oracle-cache-dir outputs/goes_pndm_smoke/schedules/_goes_oracle_cache/pndm \
  --batch-size 2 --num-batches 1 --microbatch-size 1 --ref-nfe 8 --ref-grid-size 9 \
  --candidate-grid-size 8 --metric identity --rho 0.1 --aggregation trimmed_mean \
  --trim-ratio 0.1 --coordinate-domain timesteps --dry-run

/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/export_goes_diffusers_schedule.py \
  --model-asset hf_sd35_medium --prompt-asset diffusers_smoke_prompts --solver flow_euler \
  --seed 0 --dtype bfloat16 --height 512 --width 512 --guidance-scale 3.5 --nfe 4 \
  --output-dir outputs/goes_diffusers_smoke/schedules/GOES/diffusers/sd35_medium/flow_euler/cfg_3.5/nfe_004 \
  --oracle-cache-dir outputs/goes_diffusers_smoke/schedules/_goes_oracle_cache/diffusers \
  --batch-size 2 --num-batches 1 --microbatch-size 1 --ref-nfe 8 --ref-grid-size 9 \
  --candidate-grid-size 8 --metric identity --rho 0.1 --aggregation trimmed_mean \
  --trim-ratio 0.1 --physical-grid-mode scheduler_sigmas --dry-run
```

## Incomplete Requirements

The following `goal.md` requirements are not complete because they require real
model execution or broader experiment runs:

| Requirement | Status |
| --- | --- |
| At least one real main benchmark end-to-end | Not run. Only toy CPU smoke and exporter dry-runs are verified. |
| Real PNDM/diffusers GOES schedule materialization | Not run without `--dry-run`; would load checkpoints and run GPU calibration. |
| Image generation for GOES/base/AYS comparisons | Not run. |
| FID/KID/CLIPScore/ImageReward/pairwise win rates | Not run for GOES outputs. PNDM KID is implemented only for reference stats files that include feature activations; existing FID-only `mu/sigma` stats are insufficient for KID. |
| Multi-seed, NFE, CFG, calibration-size, metric, rho, candidate-grid, and oracle-convergence sweeps on real models | Not run. Non-executed launcher configs now cover NFE/CFG/solver, rho-metric ablation, calibration-size ablation, candidate-grid ablation, and oracle-convergence grids. |
| Models without published offline schedules | Non-executed diffusers config covers SD3.5 Medium, FLUX.1-dev, and Lumina-Image-2.0 with `base` vs `GOES` only; generation and scoring are not run. |
| Paper-ready quality claims | Not established. Toy latent replay results are not image-quality evidence. |
| DPM/UniPC/SDE real scheduler-history replay refinement | Not implemented for real diffusers/PNDM multistep pipeline paths. Unsupported modes are rejected or marked empirical-only. |

## Theory Coverage

Current theory-covered exporter paths:

- PNDM backend: deterministic velocity `euler` and `heun2`.
- Diffusers flow pipelines: velocity-style `flow_euler` and `flow_heun` proxies.

Current empirical-only or unsupported paths:

- Diffusers VP/SD empirical `euler` is not strict solver-independent ODE theory.
- Diffusers DPM, UniPC, SDE, and scheduler-history modes are rejected until
  real black-box replay refinement is wired through the pipeline path.

## Next Required Evidence

To complete `goal.md`, run at least one real benchmark end-to-end with explicit
model, solver, number of function evaluations (NFE), seed count, prompt or
dataset split, guidance scale, schedule directories, and metrics. A minimal
first GPU gate is:

```bash
/home/gwx/miniconda3/envs/sc-diff/bin/python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_smoke.yaml \
  --materialize-schedules --execute
```

This should only be run when GPU calibration/generation is explicitly allowed.

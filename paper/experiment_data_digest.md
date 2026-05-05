# D-GPDE Experiment Data Digest

Date: 2026-05-05

Purpose: one-file writing reference for experiments already run and copied into
paper-facing artifacts. This file summarizes aggregate results and points to
the detailed CSVs, schedules, figures, and raw output roots needed for later
paper writing. It intentionally avoids raw generated-image directories and raw
AYS numeric schedule redistribution.

## Claim Boundary

- Main positive evidence: PNDM/CIFAR-10 Euler, 50k samples per seed, seeds
  0/1/2, NFE 10 and 20. D-GPDE improves over base, native-linear, and the
  authorized project-owned offline baseline at both NFEs.
- Strong fixed baseline caveat: Karras is worse than D-GPDE at NFE 10 but
  slightly better at NFE 20.
- Text-to-image evidence: SD1.5 is mixed/negative for D-GPDE under CLIPScore
  and ImageReward; SDXL is condition-dependent and positive against AYS only
  at CFG 7.5.
- Reuse evidence: CIFAR seed-0 schedule reuse has zero observed FID gap because
  rounded PNDM Euler timesteps match. SD1.5 CFG reuse remains worse than base
  and AYS, so deployment-efficiency claims remain unsupported.
- AYS policy: do not distribute raw AYS numeric schedules. Use citations,
  hashes, aggregate metric rows, rendered figures, and optional external loader
  paths.

## Source File Index

| Category | Paper-facing files | Raw/source output roots |
| --- | --- | --- |
| Smoke gate | `paper/results/smoke/` | `outputs/goes_pndm_smoke/` |
| CIFAR pilot | `paper/results/pilot/cifar10_pndm_euler_pilot_fid.csv` | `outputs/gpde_pndm_test/` |
| CIFAR 50k main | `paper/results/cifar10_50k/cifar10_pndm_euler_50k_fid_*seeds0_1_2.csv` | `outputs/gpde_pndm_cifar10_50k_nfe10_20_seed0/`, `outputs/gpde_pndm_cifar10_50k_nfe10_20_seeds1_2/` |
| CIFAR linear baseline | included in CIFAR aggregate/detail CSVs | `outputs/gpde_pndm_cifar10_50k_linear_baseline_seeds0_1_2/` |
| CIFAR Karras baseline | included in CIFAR aggregate/detail CSVs; copied schedule metadata in `paper/results/cifar10_50k/` | `outputs/gpde_pndm_cifar10_50k_karras_baseline_seeds0_1_2/` |
| CIFAR authorized offline baseline | included as `offline` in CIFAR aggregate/detail CSVs | `outputs/gpde_pndm_cifar10_authorized_offline/` |
| CIFAR reuse | `paper/results/cifar10_50k/cifar10_pndm_euler_50k_reuse_seed0_schedule_*.csv` | `outputs/gpde_pndm_cifar10_50k_reuse_seed0_schedule_seeds1_2/` |
| CIFAR failed offline proxies | `paper/results/cifar10_50k/*offline_proxy*.csv`, `paper/results/failure/cifar10_*offline_proxy*.csv` | `outputs/gpde_pndm_cifar10_50k_offline_proxy_baseline_seeds0_1_2/`, `outputs/gpde_pndm_cifar10_medium_offline_proxy/`, `outputs/gpde_pndm_cifar10_default_offline_proxy/` |
| PNDM ablation failures | `paper/results/failure/failure_cases_summary.csv` | `outputs/gpde_pndm_rho_metric_ablation_nfe5_10_15/`, `outputs/gpde_pndm_aggregation_ablation_nfe5_10_15/` |
| SD1.5 sweep | `paper/results/t2i/sd15_euler_nfe10_cfg_sweep_*.csv` | `outputs/gpde_diffusers_sd15_nfe10_cfg_seed_sweep/` |
| SDXL sweep | `paper/results/t2i/sdxl_euler_nfe10_cfg_sweep_*.csv` | `outputs/gpde_diffusers_sdxl_nfe10_cfg_seed_sweep/` |
| SD1.5 reuse | `paper/results/t2i/sd15_euler_nfe10_cfg7p5_seed0_reuse_*.csv` | `outputs/gpde_diffusers_sd15_reuse_cfg7p5_seed0_schedule/` |
| Prompt asset | `paper/results/t2i/diffusers_ablation_prompts_manifest.json`, `paper/results/t2i/diffusers_ablation_prompts.csv` | `data/pndm/prompts/modern_diffusers_ablation_prompts.json` |
| Calibration cost | `paper/results/cost/calibration_cost_summary.csv`, `paper/results/cost/calibration_amortization_summary.csv` | oracle-reuse CSVs under each experiment output root |

## Smoke Gate

PNDM/CIFAR-10 smoke, Euler, NFE 4, seed 0, 32 samples. This is an implementation
gate only.

| Schedule | FID |
| --- | ---: |
| base | 295.7675 |
| D-GPDE/GOES | 258.8215 |

Cost row: one D-GPDE schedule, 154 model-evaluation equivalents.

Sources:

- `paper/results/smoke/goes_pndm_smoke.csv`
- `paper/results/smoke/goes_pndm_smoke_oracle_reuse_cost.csv`

## CIFAR-10 Pilot FID

PNDM/CIFAR-10 Euler, seed 0, 5k samples. These rows are pilot evidence, not
paper-grade 50k FID.

| NFE | base FID | D-GPDE FID | base - D-GPDE |
| ---: | ---: | ---: | ---: |
| 4 | 118.1095 | 62.3659 | +55.7436 |
| 5 | 73.3332 | 44.9730 | +28.3602 |
| 6 | 51.2552 | 36.3022 | +14.9530 |
| 8 | 33.2490 | 27.1044 | +6.1446 |
| 10 | 25.3902 | 22.3758 | +3.0143 |
| 12 | 21.6665 | 19.5378 | +2.1287 |
| 15 | 18.5618 | 17.5397 | +1.0221 |

Source: `paper/results/pilot/cifar10_pndm_euler_pilot_fid.csv`

## CIFAR-10 50k Main Result

PNDM/CIFAR-10 Euler, 50k generated samples per seed, seeds 0/1/2, FID
lower-is-better.

| NFE | method | FID mean | FID SEM |
| ---: | --- | ---: | ---: |
| 10 | base | 20.5658 | 0.0750 |
| 10 | native-linear | 20.5658 | 0.0750 |
| 10 | Karras | 25.8415 | 0.1037 |
| 10 | authorized offline | 19.1979 | 0.0851 |
| 10 | D-GPDE | 17.5393 | 0.0755 |
| 20 | base | 11.1734 | 0.0744 |
| 20 | native-linear | 11.2039 | 0.0736 |
| 20 | Karras | 10.6371 | 0.0874 |
| 20 | authorized offline | 10.8838 | 0.0587 |
| 20 | D-GPDE | 10.7092 | 0.0709 |

Derived deltas:

| NFE | D-GPDE FID reduction vs base | SEM | all seed reductions positive | D-GPDE - authorized offline | SEM | all seeds lower than offline |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 10 | +3.0265 | 0.0194 | true | -1.6585 | 0.0170 | true |
| 20 | +0.4642 | 0.0046 | true | -0.1746 | 0.0587 | true |

Writing takeaways:

- Use this as the main controlled positive result.
- D-GPDE beats base/native-linear/authorized offline at both NFE 10 and 20.
- Karras is mixed: worse at NFE 10, slightly better than D-GPDE at NFE 20.

Sources:

- `paper/results/cifar10_50k/cifar10_pndm_euler_50k_fid_aggregate_seeds0_1_2.csv`
- `paper/results/cifar10_50k/cifar10_pndm_euler_50k_fid_delta_seeds0_1_2.csv`
- `paper/results/cifar10_50k/cifar10_pndm_euler_50k_fid_detail_seeds0_1_2.csv`
- figure: `paper/figures/cifar10_pndm_euler_50k_fid_seeds0_1_2.pdf`

## CIFAR-10 Schedule Reuse

Actual 50k generation check: reuse the seed-0 D-GPDE schedule for seeds 1 and
2. FID lower-is-better.

| NFE | seeds | base FID mean | seed-specific D-GPDE FID | reused seed-0 D-GPDE FID | reuse - seed-specific | rounded timesteps all equal |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 10 | 1,2 | 20.4917 | 17.4703 | 17.4703 | 0.0000 | true |
| 20 | 1,2 | 11.0997 | 10.6394 | 10.6394 | 0.0000 | true |

Rounding audit:

- NFE 10 rounded timesteps: `999 821 707 614 525 433 273 179 114 57`
- NFE 20 rounded timesteps: `999 896 821 760 707 659 614 569 525 481 433 377 273 218 179 145 114 85 57 30`
- Maximum raw timestep differences are at most 0.001422, then round to the
  same PNDM Euler execution grids.

Writing takeaway: this supports only a narrow same-grid CIFAR reuse statement.
It does not support general reuse across models, CFG values, or prompt
conditions.

Sources:

- `paper/results/cifar10_50k/cifar10_pndm_euler_50k_reuse_seed0_schedule_aggregate.csv`
- `paper/results/cifar10_50k/cifar10_pndm_euler_50k_reuse_seed0_schedule_rounding.csv`

## CIFAR-10 Offline Baselines and Failure Runs

### Authorized Project-Owned Offline Baseline

Included as `offline` in the main CIFAR 50k aggregate. The default standalone
hierarchical exporter saved `_stage_bundles/nfe_010` and `_stage_bundles/nfe_020`;
the later 40-step refinement was stopped because the paper evaluates only NFE
10 and 20.

| NFE | authorized offline FID mean | SEM | D-GPDE FID mean | D-GPDE - offline |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 19.1979 | 0.0851 | 17.5393 | -1.6585 |
| 20 | 10.8838 | 0.0587 | 10.7092 | -0.1746 |

Release policy: do not include authorized-offline numeric schedule bundles in
the anonymous package.

### Lightweight Offline-Proxy Failure, 50k FID

| NFE | base | Karras | D-GPDE | offline-proxy |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 20.5658 | 25.8415 | 17.5393 | 41.3734 |
| 20 | 11.1734 | 10.6371 | 10.7092 | 22.3641 |

Source: `paper/results/cifar10_50k/cifar10_pndm_euler_50k_offline_proxy_aggregate.csv`

### Medium Offline-Proxy Smoke Failure, 5k FID

Not paper-grade because it uses 5k samples.

| NFE | method | FID mean | SEM | proxy best FID | stage iterations |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10 | offline-proxy-medium | 26.7632 | 0.0633 | 35.9734 | 4 |
| 20 | offline-proxy-medium | 18.3223 | 0.0317 | | 6 |

Source: `paper/results/failure/cifar10_medium_offline_proxy_smoke5k_aggregate.csv`

### Default Offline-Proxy Interrupted Diagnostic

Default config diagnostic, not paper-grade:

- candidate count: 11
- data samples: 8192
- proxy samples: 2048
- target NFEs: 10, 20
- stage-10 proxy values: 31.999643; 38.732228; 56.749100; 75.824936
- stage-10 best proxy FID: 31.999643
- stop reason: early stop
- completed schedule bundle: false
- elapsed seconds: 3771.65

Source: `paper/results/failure/cifar10_default_offline_proxy_interrupted_summary.csv`

## PNDM Ablation and Failure Cases

Summary rows retained in failure analysis:

| Case | Setting | Comparison | Metric delta | Takeaway |
| --- | --- | --- | ---: | --- |
| rho=0 geometry | PNDM/CIFAR-10 Euler, NFE 10, seed 0, 5k | D-GPDE rho=0 - base | +30.922 FID | Removing residual mix makes FID much worse than base. |
| EDM scalar metric | PNDM/CIFAR-10 Euler, NFE 10, seed 0, 5k | D-GPDE EDM scalar - base | +4.871 FID | Metric variant loses the base-schedule gain. |
| Mean aggregation | PNDM/CIFAR-10 Euler, NFE 15, seed 0, 5k | mean - CVaR | +0.071 FID | Mean aggregation remains positive but trails CVaR. |
| SD1.5 CFG 5 | SD1.5 Euler, NFE 10, 3 seeds, 50 prompts/seed | D-GPDE - base | -0.133 IR | Matched text-to-image calibration lowers ImageReward. |
| SD1.5 CFG 7.5 | SD1.5 Euler, NFE 10, 3 seeds, 50 prompts/seed | D-GPDE - base | -0.084 IR | Matched text-to-image calibration lowers ImageReward. |
| SD1.5 CFG 10 | SD1.5 Euler, NFE 10, 3 seeds, 50 prompts/seed | D-GPDE - base | -0.087 IR | Matched text-to-image calibration lowers ImageReward. |

Source: `paper/results/failure/failure_cases_summary.csv`

## SD1.5 Text-to-Image Sweep

Stable Diffusion 1.5, Diffusers Euler, NFE 10, CFG 5.0/7.5/10.0, seeds 0/1/2,
50 prompts per seed, schedules base/AYS/D-GPDE. Metrics are higher-is-better.

### Aggregate Metrics by Schedule

| CFG | schedule | CLIPScore mean | CLIPScore SEM | ImageReward mean | ImageReward SEM |
| ---: | --- | ---: | ---: | ---: | ---: |
| 5 | base | 29.4595 | 0.2963 | 0.0828 | 0.0115 |
| 5 | AYS | 29.6442 | 0.2074 | 0.1887 | 0.0808 |
| 5 | D-GPDE | 29.2713 | 0.1573 | -0.0499 | 0.0401 |
| 7.5 | base | 29.7309 | 0.1309 | 0.2013 | 0.0494 |
| 7.5 | AYS | 29.6498 | 0.2082 | 0.2366 | 0.0709 |
| 7.5 | D-GPDE | 29.5668 | 0.1146 | 0.1171 | 0.0404 |
| 10 | base | 29.6737 | 0.2324 | 0.2387 | 0.0360 |
| 10 | AYS | 29.5647 | 0.0537 | 0.2741 | 0.0785 |
| 10 | D-GPDE | 29.5425 | 0.1708 | 0.1516 | 0.0140 |

### Pairwise Deltas

| CFG | comparison | Delta CLIPScore | CLIP win rate | Delta ImageReward | IR win rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 5 | AYS - base | +0.1847 | 53.3% | +0.1058 | 66.0% |
| 5 | D-GPDE - base | -0.1882 | 42.7% | -0.1328 | 33.3% |
| 5 | D-GPDE - AYS | -0.3729 | 41.3% | -0.2386 | 26.7% |
| 7.5 | AYS - base | -0.0811 | 51.3% | +0.0352 | 57.3% |
| 7.5 | D-GPDE - base | -0.1641 | 45.3% | -0.0843 | 46.0% |
| 7.5 | D-GPDE - AYS | -0.0830 | 47.3% | -0.1195 | 38.7% |
| 10 | AYS - base | -0.1090 | 46.0% | +0.0354 | 54.0% |
| 10 | D-GPDE - base | -0.1312 | 41.3% | -0.0871 | 40.7% |
| 10 | D-GPDE - AYS | -0.0222 | 48.0% | -0.1225 | 37.3% |

Writing takeaway: SD1.5 is a negative/mixed case for D-GPDE under these
automated metrics. Do not use it as a positive text-to-image quality result.

Sources:

- `paper/results/t2i/sd15_euler_nfe10_cfg_sweep_schedule_summary.csv`
- `paper/results/t2i/sd15_euler_nfe10_cfg_sweep_pairwise_summary.csv`
- `paper/results/t2i/sd15_euler_nfe10_cfg_sweep_detail.csv`
- figure: `paper/figures/sd15_euler_nfe10_cfg_sweep_pairwise.pdf`
- schedule profile: `paper/figures/sd15_euler_nfe10_cfg7p5_schedule_profile_seed0.pdf`

## SDXL Text-to-Image Sweep

SDXL, Diffusers Euler, NFE 10, CFG 5.0/7.5, seeds 0/1/2, 50 prompts per seed,
schedules base/AYS/D-GPDE. Metrics are higher-is-better.

Run coverage:

- 6 D-GPDE schedules
- 18 generation manifests
- 900 JPEGs
- CLIPScore/ImageReward detail and aggregate CSVs
- pairwise win-rate rows
- oracle-reuse cost rows

Calibration setting:

- batch size 1
- 4 calibration batches
- reference NFE 64
- reference grid 65
- candidate grid 128

### Aggregate Metrics by Schedule

| CFG | schedule | CLIPScore mean | CLIPScore SEM | ImageReward mean | ImageReward SEM |
| ---: | --- | ---: | ---: | ---: | ---: |
| 5 | base | 31.0018 | 0.0976 | 0.8220 | 0.0453 |
| 5 | AYS | 31.0256 | 0.1613 | 1.0192 | 0.0384 |
| 5 | D-GPDE | 30.8351 | 0.0912 | 0.9006 | 0.0336 |
| 7.5 | base | 31.0200 | 0.2072 | 1.0072 | 0.0718 |
| 7.5 | AYS | 30.6910 | 0.2614 | 0.9708 | 0.0355 |
| 7.5 | D-GPDE | 31.1077 | 0.1428 | 1.1292 | 0.0394 |

### Pairwise Deltas

| CFG | comparison | Delta CLIPScore | CLIP win rate | Delta ImageReward | IR win rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 5 | AYS - base | +0.0237 | 46.0% | +0.1973 | 66.0% |
| 5 | D-GPDE - base | -0.1668 | 43.3% | +0.0786 | 52.7% |
| 5 | D-GPDE - AYS | -0.1905 | 48.0% | -0.1187 | 35.3% |
| 7.5 | AYS - base | -0.3290 | 44.7% | -0.0364 | 49.3% |
| 7.5 | D-GPDE - base | +0.0877 | 46.7% | +0.1219 | 54.7% |
| 7.5 | D-GPDE - AYS | +0.4167 | 58.0% | +0.1584 | 52.7% |

Writing takeaway: SDXL is condition-dependent. At CFG 5.0, D-GPDE improves
ImageReward over base but trails AYS. At CFG 7.5, D-GPDE has positive mean
CLIPScore and ImageReward deltas against both base and AYS.

Sources:

- `paper/results/t2i/sdxl_euler_nfe10_cfg_sweep_schedule_summary.csv`
- `paper/results/t2i/sdxl_euler_nfe10_cfg_sweep_pairwise_summary.csv`
- `paper/results/t2i/sdxl_euler_nfe10_cfg_sweep_detail.csv`
- figure: `paper/figures/sdxl_euler_nfe10_cfg_sweep_pairwise.pdf`

## SD1.5 Schedule Reuse

Actual generation reuse check: reuse the D-GPDE schedule calibrated at SD1.5
CFG 7.5, seed 0, across CFG 5.0/7.5/10.0 and seeds 0/1/2.

### Reuse Aggregate

| CFG | reused schedule CLIPScore | CLIPScore SEM | reused schedule ImageReward | ImageReward SEM |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 29.1698 | 0.2298 | -0.0436 | 0.0606 |
| 7.5 | 29.5188 | 0.1090 | 0.1197 | 0.0611 |
| 10 | 29.5384 | 0.1384 | 0.1852 | 0.0241 |

### Reuse Pairwise Highlights

| CFG | comparison | metric | mean delta | win rate |
| ---: | --- | --- | ---: | ---: |
| 5 | reuse - base | CLIPScore | -0.2898 | 44.7% |
| 5 | reuse - base | ImageReward | -0.1265 | 34.7% |
| 5 | reuse - AYS | CLIPScore | -0.4745 | 40.7% |
| 5 | reuse - AYS | ImageReward | -0.2323 | 28.7% |
| 5 | reuse - D-GPDE | CLIPScore | -0.1016 | 46.7% |
| 5 | reuse - D-GPDE | ImageReward | +0.0063 | 50.7% |
| 7.5 | reuse - base | CLIPScore | -0.2121 | 44.7% |
| 7.5 | reuse - base | ImageReward | -0.0816 | 44.0% |
| 7.5 | reuse - AYS | CLIPScore | -0.1310 | 48.7% |
| 7.5 | reuse - AYS | ImageReward | -0.1169 | 40.0% |
| 7.5 | reuse - D-GPDE | CLIPScore | -0.0480 | 28.0% |
| 7.5 | reuse - D-GPDE | ImageReward | +0.0027 | 31.3% |
| 10 | reuse - base | CLIPScore | -0.1353 | 43.3% |
| 10 | reuse - base | ImageReward | -0.0535 | 43.3% |
| 10 | reuse - AYS | CLIPScore | -0.0263 | 47.3% |
| 10 | reuse - AYS | ImageReward | -0.0889 | 41.3% |
| 10 | reuse - D-GPDE | CLIPScore | -0.0041 | 48.7% |
| 10 | reuse - D-GPDE | ImageReward | +0.0336 | 53.3% |

Writing takeaway: reuse stays close to seed/CFG-specific D-GPDE on ImageReward,
but remains worse than base and AYS. This is reuse-risk evidence, not
deployment-efficiency evidence.

Sources:

- `paper/results/t2i/sd15_euler_nfe10_cfg7p5_seed0_reuse_summary.csv`
- `paper/results/t2i/sd15_euler_nfe10_cfg7p5_seed0_reuse_pairwise_summary.csv`
- `paper/results/t2i/sd15_euler_nfe10_cfg7p5_seed0_reuse_detail.csv`

## Calibration Cost and Amortization

Model-evaluation-equivalent accounting. Positive quality deltas favor D-GPDE.

| Setting | condition | quality vs base | calibration evals | generation evals | cal/gen |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | NFE 10 | +3.026 +/- 0.019 FID | 12,730,464 | 1,500,000 | 8.487x |
| CIFAR-10 | NFE 20 | +0.464 +/- 0.005 FID | 12,730,464 | 3,000,000 | 4.243x |
| SD1.5 | CFG 5 | -0.133 IR | 25,460,928 | 3,000 | 8486.976x |
| SD1.5 | CFG 7.5 | -0.084 IR | 25,460,928 | 3,000 | 8486.976x |
| SD1.5 | CFG 10 | -0.087 IR | 25,460,928 | 3,000 | 8486.976x |
| SDXL | CFG 5 | +0.079 IR | 205,848 | 3,000 | 68.616x |
| SDXL | CFG 7.5 | +0.122 IR | 205,848 | 3,000 | 68.616x |

Break-even evaluated batches:

| Setting | condition | M=1 overhead | M=10 overhead | M=100 overhead | break-even batches |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | NFE 10 | 8.487x | 0.849x | 0.085x | 9 |
| CIFAR-10 | NFE 20 | 4.243x | 0.424x | 0.042x | 5 |
| SD1.5 | CFG 5 | 8486.976x | 848.698x | 84.870x | 8487 |
| SD1.5 | CFG 7.5 | 8486.976x | 848.698x | 84.870x | 8487 |
| SD1.5 | CFG 10 | 8486.976x | 848.698x | 84.870x | 8487 |
| SDXL | CFG 5 | 68.616x | 6.862x | 0.686x | 69 |
| SDXL | CFG 7.5 | 68.616x | 6.862x | 0.686x | 69 |

Writing takeaway: cost is explicit and often high. CIFAR has narrow same-grid
reuse evidence. SD1.5 has negative quality under reuse. SDXL has lower cost
because of the lightweight calibration run, but no positive reused-schedule
validation yet.

Sources:

- `paper/results/cost/calibration_cost_summary.csv`
- `paper/results/cost/calibration_amortization_summary.csv`
- figure: `paper/figures/calibration_cost_vs_quality.pdf`

## Prompt Asset

Prompt asset: `diffusers_ablation_prompts`

- source path: `data/pndm/prompts/modern_diffusers_ablation_prompts.json`
- paper prompt CSV: `paper/results/t2i/diffusers_ablation_prompts.csv`
- SHA-256: `ce18968ad8140ed23e6741d3f1ab0b00b64a2c9b9b346ff5a82a468a8688b9aa`
- prompt count: 50
- license status: project-local provenance; release license pending

Source: `paper/results/t2i/diffusers_ablation_prompts_manifest.json`

## Figures and Tables Available for Writing

Figures:

- `paper/figures/cifar10_pndm_euler_50k_fid_seeds0_1_2.pdf`
- `paper/figures/cifar10_pndm_euler_nfe20_schedule_profile_seed0.pdf`
- `paper/figures/cifar10_pndm_euler_pilot_fid.pdf`
- `paper/figures/sd15_euler_nfe10_cfg_sweep_pairwise.pdf`
- `paper/figures/sd15_euler_nfe10_cfg7p5_schedule_profile_seed0.pdf`
- `paper/figures/sdxl_euler_nfe10_cfg_sweep_pairwise.pdf`
- `paper/figures/sd15_failure_grid.pdf`
- `paper/figures/calibration_cost_vs_quality.pdf`
- `paper/figures/dgpde_method_overview.pdf`

Tables:

- `paper/tables/smoke_gate.tex`
- `paper/tables/cifar10_pndm_euler_pilot_fid.tex`
- `paper/tables/cifar10_pndm_euler_50k_fid_seeds0_1_2.tex`
- `paper/tables/cifar10_pndm_euler_50k_reuse_seed0_schedule.tex`
- `paper/tables/cifar10_pndm_euler_50k_offline_proxy.tex`
- `paper/tables/sd15_euler_nfe10_cfg_sweep_pairwise.tex`
- `paper/tables/sdxl_euler_nfe10_cfg_sweep_pairwise.tex`
- `paper/tables/sd15_euler_nfe10_cfg7p5_seed0_reuse.tex`
- `paper/tables/calibration_cost_summary.tex`
- `paper/tables/calibration_amortization_summary.tex`
- `paper/tables/failure_cases_summary.tex`

## Writing-Ready Summary Sentences

- In the three-seed PNDM/CIFAR-10 Euler 50k benchmark, D-GPDE lowers FID from
  20.5658 to 17.5393 at NFE 10 and from 11.1734 to 10.7092 at NFE 20.
- The authorized project-owned offline baseline is stronger than base but still
  trails D-GPDE at both NFE 10 and 20.
- Karras is a strong fixed baseline: it is worse than D-GPDE at NFE 10 but
  slightly better at NFE 20.
- SD1.5 is the main text-to-image negative case: D-GPDE has negative
  ImageReward deltas against base at CFG 5, 7.5, and 10.
- SDXL is condition-dependent: D-GPDE trails AYS at CFG 5.0 but has positive
  mean CLIPScore and ImageReward deltas against both base and AYS at CFG 7.5.
- Cost accounting shows that schedule calibration must be amortized; current
  evidence supports cost-aware reporting, not deployment-efficiency claims.


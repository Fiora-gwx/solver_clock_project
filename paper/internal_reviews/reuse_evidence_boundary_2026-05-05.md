# Reuse Evidence Boundary Note

Date: 2026-05-05

Scope: whether retained actual reuse runs support deployment or amortization
claims for D-GPDE schedules.

## Evidence Reviewed

Paper-facing reuse artifacts:

- CIFAR-10 aggregate:
  `paper/results/cifar10_50k/cifar10_pndm_euler_50k_reuse_seed0_schedule_aggregate.csv`
- CIFAR-10 rounded-grid check:
  `paper/results/cifar10_50k/cifar10_pndm_euler_50k_reuse_seed0_schedule_rounding.csv`
- SD1.5 reuse summary:
  `paper/results/t2i/sd15_euler_nfe10_cfg7p5_seed0_reuse_summary.csv`
- SD1.5 reuse pairwise summary:
  `paper/results/t2i/sd15_euler_nfe10_cfg7p5_seed0_reuse_pairwise_summary.csv`
- Paper tables:
  `paper/tables/cifar10_pndm_euler_50k_reuse_seed0_schedule.tex`
  and `paper/tables/sd15_euler_nfe10_cfg7p5_seed0_reuse.tex`

## CIFAR-10 Finding

The CIFAR-10 reuse check reuses the seed-0 D-GPDE schedule for seeds 1 and 2
at NFE 10 and 20, with 50k generated images per seed.

Result:

- NFE 10: reused D-GPDE FID equals seed-specific D-GPDE FID within the retained
  aggregate, with reuse gap `0.000 +/- 0.000`.
- NFE 20: reused D-GPDE FID equals seed-specific D-GPDE FID within the retained
  aggregate, with reuse gap `0.000 +/- 0.000`.
- The rounded-grid CSV shows that all reused and seed-specific continuous
  schedules differ by at most `0.001422` raw timestep units and round to the
  same integer PNDM Euler execution grids.

Interpretation:

- This is actual generation evidence, not just arithmetic amortization.
- It supports only a narrow same-backend reuse statement for this PNDM/CIFAR
  setting.
- It does not show robustness across model, solver, NFE family, guidance scale,
  prompt distribution, or perceptual metric.

## SD1.5 Finding

The SD1.5 reuse check reuses one D-GPDE schedule calibrated at CFG 7.5, seed 0
across CFG 5.0, 7.5, and 10.0, seeds 0--2, with 50 prompts per seed.

Against CFG- and seed-specific D-GPDE:

- CFG 5.0: CLIPScore delta `-0.101595`, ImageReward delta `+0.006291`.
- CFG 7.5: CLIPScore delta `-0.047990`, ImageReward delta `+0.002678`.
- CFG 10.0: CLIPScore delta `-0.004092`, ImageReward delta `+0.033573`.

Against base and AYS:

- Reused D-GPDE trails base on both CLIPScore and ImageReward at CFG 5.0, 7.5,
  and 10.0.
- Reused D-GPDE trails AYS on both CLIPScore and ImageReward at CFG 5.0, 7.5,
  and 10.0.

Interpretation:

- The reused schedule stays close to condition-specific D-GPDE on ImageReward
  and slightly below it on CLIPScore.
- Because both reused and condition-specific D-GPDE remain worse than base or
  AYS in this retained SD1.5 run, the reuse result cannot support a
  deployment-efficiency claim.

## Decision

Supported reuse wording:

- "A narrow PNDM/CIFAR-10 same-backend reuse check shows no FID change because
  the seed-specific schedules round to the same integer execution grids."
- "An SD1.5 CFG-shift reuse check stays close to D-GPDE but remains worse than
  base and AYS in the retained metrics."

Unsupported wording:

- "D-GPDE schedules amortize calibration cost in general."
- "D-GPDE reuse improves text-to-image quality."
- "One calibrated D-GPDE schedule transfers robustly across conditions."

The practical deployment claim remains blocked until reused schedules improve
quality against relevant baselines across nontrivial batches or conditions.

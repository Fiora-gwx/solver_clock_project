# Text-to-Image CFG Sweeps

SD1.5 source output root:

```bash
outputs/gpde_diffusers_sd15_nfe10_cfg_seed_sweep/
```

SDXL source output root:

```bash
outputs/gpde_diffusers_sdxl_nfe10_cfg_seed_sweep/
```

Run command:

```bash
export PYTHONDONTWRITEBYTECODE=1
PY=/path/to/sc-diff/bin/python
$PY scripts/run/run_experiment_config.py --experiment-config configs/experiments/gpde_diffusers_sd15_nfe10_cfg_seed_sweep.yaml --materialize-schedules --execute
$PY scripts/run/run_experiment_config.py --experiment-config configs/experiments/gpde_diffusers_sdxl_nfe10_cfg_seed_sweep.yaml --materialize-schedules --execute
```

Aggregation command:

```bash
$PY paper/figures/src/aggregate_t2i_sd15_cfg_sweep.py
$PY paper/figures/src/aggregate_t2i_sdxl_cfg_sweep.py
$PY paper/figures/src/plot_sd15_schedule_profile.py
$PY paper/figures/src/export_t2i_prompt_manifest.py
```

Coverage: Stable Diffusion 1.5, Diffusers Euler solver, NFE 10, guidance scales
5.0, 7.5, and 10.0, seeds 0, 1, and 2, 50 prompts from
`diffusers_ablation_prompts`, schedules base, AYS, and D-GPDE/GPDE. The SDXL
run covers guidance scales 5.0 and 7.5 with the same seeds, prompts, solver,
NFE, schedules, and metrics. Metrics are CLIPScore and ImageReward, both
higher-is-better.

The retained result is mixed: D-GPDE trails SD1.5 automated text-to-image
metrics, improves SDXL ImageReward over base at both guidance scales, and beats
AYS on mean SDXL CLIPScore/ImageReward deltas at CFG 7.5.

`sd15_euler_nfe10_cfg7p5_schedule_profile_seed0.csv` is the local source table
for the representative base/AYS/D-GPDE sigma-grid diagnostic figure at CFG 7.5
and seed 0. The anonymous artifact package omits this CSV because it contains
raw AYS numeric schedule values; AYS is retained there through citations,
metric rows, rendered figures, hashes, and optional external loader paths.

`diffusers_ablation_prompts_manifest.json` records the prompt-asset SHA-256 hash
and prompt count. `diffusers_ablation_prompts.csv` stores the 50 prompt texts by
prompt index for matched-pair auditability.

`sd15_euler_nfe10_cfg7p5_seed0_reuse_*.csv` records an actual schedule-reuse
check. The run loads the D-GPDE schedule calibrated at CFG 7.5, seed 0, then
generates CFG 5.0, 7.5, and 10.0 outputs for seeds 0, 1, and 2 with the same
50-prompt asset. The comparison remains mixed or negative, so it is reuse
evidence, not deployment-efficiency evidence.

`sdxl_euler_nfe10_cfg_sweep_*.csv` and `schedules/sdxl_*.json` record the
minimal SDXL validation. The D-GPDE SDXL schedules use the lighter calibration
configuration in `configs/experiments/gpde_diffusers_sdxl_nfe10_cfg_seed_sweep.yaml`:
batch size 1, four calibration batches, 64 reference steps, a 65-node
reference grid, and a 128-node candidate grid.

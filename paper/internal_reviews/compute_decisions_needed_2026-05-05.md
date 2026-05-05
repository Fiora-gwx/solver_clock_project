# Compute Decisions Needed

Date: 2026-05-05

Scope: remaining experiment blockers that cannot be closed by local cleanup,
source review, or existing-result aggregation.

## Decision 1: Strong CIFAR Offline Baseline

Goal blocker addressed:

- Strong offline optimized CIFAR schedule evidence for the retained
  PNDM/CIFAR-10 Euler NFE 10/20 benchmark.

Current evidence:

- Published AYS inventory check found no CIFAR-10 AYS bundle in the official
  quickstart, Diffusers `AysSchedules`, or local published schedules.
- Lightweight and medium project-owned offline-proxy attempts produced poor FID.
- The default-budget project-owned attempt became usable after the exporter
  saved completed NFE 10 and 20 stage bundles.
- The exporter now saves completed intermediate stages under `_stage_bundles/`,
  which reduces but does not remove the risk of another interrupted run.

Author decision:

- Approved on 2026-05-05. The stronger project-owned offline optimizer run
  retained completed stage bundles and evaluated NFE 10 and 20 with matched
  50k FID before promotion to paper evidence.

Compute authorization options:

| Option | Expected output | Risk |
| --- | --- | --- |
| Do not run more offline compute. | Keep the main CIFAR claim scoped to base/native-linear and report Karras/offline-proxy as mixed/failure evidence. | The offline-baseline blocker remains open, but the claim is disciplined. |
| Run a stronger project-owned offline optimizer. | Completed schedule bundles for NFE 10/20, matched 50k FID rows, aggregate table, and failure record if it degrades. | Prior default-budget run was slow and the 10-step proxy degraded after the first iteration. |
| Provide an external CIFAR schedule source. | Imported schedule bundle plus source/license note and matched 50k FID rows. | Must match or justify model, solver, NFE, metric, and redistribution terms. |

Command skeleton if the long run is authorized:

```bash
export PYTHONDONTWRITEBYTECODE=1
PY=/path/to/sc-diff/bin/python
$PY scripts/run/export_ays_schedule.py \
  --backend pndm \
  --dataset-config configs/datasets/cifar10.yaml \
  --model-asset pndm_model_ddim_cifar10 \
  --solver euler \
  --target-nfes 10,20 \
  --output-root outputs/gpde_pndm_cifar10_authorized_offline/schedules/offline_authorized/pndm/cifar10/pndm_model_ddim_cifar10/euler \
  --ays-config configs/clocks/AYS.yaml \
  --seed 0 \
  --device cuda
```

Outcome on 2026-05-05: the run saved `_stage_bundles/nfe_010` and
`_stage_bundles/nfe_020`; the later 40-step refinement was stopped because the
paper evaluates only NFE 10 and 20. Matched 50k FID over seeds 0, 1, and 2
gave authorized-offline FID `19.197856 +/- 0.085130` at NFE 10 and
`10.883802 +/- 0.058717` at NFE 20. D-GPDE remains lower at both NFEs:
`17.539309 +/- 0.075482` and `10.709165 +/- 0.070886`. The anonymous release
policy omits the numeric offline schedule bundles.

## Decision 2: Broader Reuse Validation

Goal blocker addressed:

- Broad positive schedule reuse evidence for deployment or amortization claims.

Current evidence:

- CIFAR seed-0 schedule reuse is actual 50k generation evidence, but it is a
  same-grid case: all compared schedules round to the same PNDM Euler timesteps.
- SD1.5 CFG-shift reuse stays close to D-GPDE, but reused D-GPDE remains worse
  than base and AYS on retained CLIPScore and ImageReward metrics.

Compute authorization options:

| Option | Expected output | Risk |
| --- | --- | --- |
| Do not run more reuse compute. | Keep the paper as cost-aware reporting with no deployment-efficiency claim. | Deployment claim remains unsupported. |
| Run a larger same-model reuse batch where D-GPDE already improves quality. | Matched quality and reuse-gap table over additional seeds or held-out batches. | May only confirm same-grid or same-condition behavior. |
| Run a nontrivial reuse transfer setting. | Matched base/AYS/D-GPDE/reused-D-GPDE quality table under shifted prompts, CFG, or model conditions. | Existing SD1.5 transfer evidence is negative against base and AYS. |

Minimum acceptance criteria for a positive reuse claim:

- The reused schedule must be evaluated with actual generation, not arithmetic
  amortization only.
- It must be compared against relevant base and offline/fixed baselines.
- It must improve or match quality while reducing effective calibration cost.
- Metadata must prove model, solver, NFE, seed, prompt set or dataset split,
  guidance scale, schedule source, and cache identity.

## Decision 3: Minimal SDXL Text-to-Image Validation

Goal blocker addressed:

- Minimal SDXL validation for the approved SD1.5+SDXL text-to-image route.

Author decision:

- Approved on 2026-05-05. The run kept the validation matrix small: SDXL,
  Diffusers Euler, NFE 10, CFG 5.0 and 7.5, seeds 0, 1, and 2, 50 prompts,
  and base/AYS/D-GPDE schedules.

Run note:

- The first SDXL attempt inherited the SD1.5 calibration grid
  (256 reference steps and a 512-node candidate grid) and was interrupted after
  more than 20 minutes without writing a schedule. The completed run uses the
  lighter SDXL calibration config in
  `configs/experiments/gpde_diffusers_sdxl_nfe10_cfg_seed_sweep.yaml`:
  batch size 1, four calibration batches, 64 reference steps, a 65-node
  reference grid, and a 128-node candidate grid.

Outcome:

- The run wrote 6 D-GPDE schedules, 18 generation manifests, 900 JPEGs, 18
  OK summary rows, CLIPScore/ImageReward detail and aggregate CSVs, pairwise
  win-rate rows, and oracle-reuse cost rows under
  `outputs/gpde_diffusers_sdxl_nfe10_cfg_seed_sweep/`.
- Paper-facing artifacts were generated with
  `paper/figures/src/aggregate_t2i_sdxl_cfg_sweep.py`.
- SDXL remains condition-dependent: at CFG 5.0 D-GPDE improves ImageReward
  over base but trails AYS; at CFG 7.5 D-GPDE improves mean CLIPScore and
  ImageReward deltas against both base and AYS. This supports text-to-image
  coverage and boundary analysis, not a broad quality-improvement claim.

## Decision 4: Stop Experiment Work And Submit Bounded Draft

If authors choose not to authorize more compute, the paper can only make the
bounded current claims:

- PNDM/CIFAR-10 Euler D-GPDE improves over base and native-coordinate linear at
  NFE 10/20 in the retained three-seed 50k FID run.
- The authorized project-owned offline baseline is lower FID than base but
  higher FID than D-GPDE at NFE 10 and 20.
- Karras is mixed: worse than D-GPDE at NFE 10 and slightly better at NFE 20.
- Lightweight and medium project-owned offline-proxy attempts remain negative.
- SD1.5 text-to-image and reuse results are mixed or negative against base/AYS.
- SDXL text-to-image is condition-dependent, with positive mean deltas at
  CFG 7.5 and a weaker CFG 5.0 comparison against AYS.
- Calibration cost is reported, but deployment efficiency remains unsupported.

# Default Offline-Proxy Feasibility Note

Date: 2026-05-05

Scope: interrupted default-budget project-owned offline-proxy attempt for
PNDM/CIFAR-10 Euler schedules.

## Run

Command family:

```bash
PY=/path/to/sc-diff/bin/python
$PY scripts/run/export_ays_schedule.py \
  --backend pndm \
  --dataset-config configs/datasets/cifar10.yaml \
  --model-asset pndm_model_ddim_cifar10 \
  --solver euler \
  --target-nfes 10,20 \
  --output-root outputs/gpde_pndm_cifar10_default_offline_proxy/schedules/offline_proxy_default/pndm/cifar10/pndm_model_ddim_cifar10/euler \
  --ays-config configs/clocks/AYS.yaml \
  --seed 0 \
  --device cuda
```

The optimizer used `configs/clocks/AYS.yaml`: 8192 training images, 11
candidates per coordinate, a 2048-sample FID proxy, 10 initial steps, and two
subdivision rounds.

## Observed Behavior

The 10-step stage early-stopped after four iterations. Proxy FID values were
31.9996, 38.7322, 56.7491, and 75.8249, so the best proxy remained the first
iteration. The run then entered the 20-step refinement stage, completed one
changed iteration, and continued slowly. It was interrupted after 3771.65
seconds because the default configuration would still run a 40-step reference
stage before saving any target schedule bundles.

## Decision

This interrupted run does not produce a paper-grade offline baseline. It does
record that the repository default AYS-style CIFAR optimizer is not a practical
short-path fix for the current baseline blocker: the 10-step proxy degraded
under default data and candidate settings, and the full default hierarchy did
not save a schedule within the attempted window.

The cleaned one-row summary lives at
`paper/results/failure/cifar10_default_offline_proxy_interrupted_summary.csv`.

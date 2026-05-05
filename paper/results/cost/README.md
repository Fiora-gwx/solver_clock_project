# Calibration Cost Results

This directory stores paper-facing calibration cost accounting generated from
retained CIFAR-10 and SD1.5 oracle reuse CSVs.

- `calibration_cost_summary.csv`: combined cost-quality rows for the three-seed
  CIFAR-10 50k FID run and the matched SD1.5 CFG sweep.

Regenerate the CSV, table, and figure with:

```bash
export PYTHONDONTWRITEBYTECODE=1
PY=/path/to/sc-diff/bin/python
$PY paper/figures/src/aggregate_calibration_cost.py
```

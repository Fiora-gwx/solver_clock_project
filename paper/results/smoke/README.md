# Smoke Result Sources

This directory stores paper-facing copies of the real PNDM/CIFAR-10 smoke gate.
The run validates schedule export, generation, scoring, and metadata plumbing.
It is not paper-grade evidence for image-quality claims.

- `goes_pndm_smoke.csv`: two OK FID rows, base and D-GPDE/GOES.
- `goes_pndm_smoke_oracle_reuse_cost.csv`: calibration cost accounting.
- `goes_pndm_smoke_schedule.json`: exported schedule metadata for the D-GPDE
  smoke run.

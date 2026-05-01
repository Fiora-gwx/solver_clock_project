# AGENTS.md

This file is the operating manual for Codex/AI agents working in this repository. Read it before making any code, experiment, or manuscript change.

---

## Project goal

This project studies **training-data-free, inference-time adaptive sampling schedule calibration** for diffusion and flow-based generative models.

The working method is **SADB**: a defect-balanced clock method that estimates where a sampler is locally difficult using small pilot calibration runs, then exports a non-uniform timestep/sigma schedule for low-NFE generation.

The core comparison targets are the **base uniform schedule** and **AYS schedules** (as a strong offline optimization baseline). SADB's intended positioning is:

- SADB is training-data-free and runs at inference time, requiring no offline precomputation.
- AYS is a principled offline method that can produce optimized schedules for a given model/solver/NFE combination, but requires re-running whenever those conditions change.
- SADB's advantage is adaptability at inference time with low pilot cost, and better robustness across NFE settings — not that AYS fundamentally cannot handle different conditions.

The main NeurIPS-style contribution:

> Sampling schedules are not solver-invariant. We study low-cost solver- and condition-aware schedule calibration at inference time, compare it against offline-optimized schedules, and show when adaptive calibration can approach or improve fixed schedules, especially under varied NFE, guidance scale, or model settings.

---

## Non-negotiable writing rules

### Scientific honesty

- Never hide negative results. Analyze them.
- Never describe a result as significant without adequate seeds or confidence evidence.
- Do not use "clearly", "dramatically", "universally", "SOTA", or "best" unless directly supported.
- Distinguish: observed result / plausible explanation / hypothesis requiring future validation.
- Always state the model, solver, NFE, seed count, prompt set, metric, and guidance scale.
- If a result is noisy, say so. If a method only works in a subset of settings, state the subset.

### Writing style

Use concise, technical, NeurIPS-style prose.

- Present tense for method and general claims; past tense for completed experiments.
- No marketing language. No long paragraphs. Each paragraph should have one core point.
- Define all acronyms on first use: "number of function evaluations (NFE)", "classifier-free guidance (CFG)".
- Use "NeurIPS", not "NIPS".
- Use "sampling schedule", "sigma grid", "timestep grid", and "noise schedule" consistently.
- Avoid vague verbs ("boosts", "greatly improves"). Prefer "improves", "reduces", "matches", "approaches", "degrades".

### Claims allowed by default

- SADB is training-data-free.
- SADB runs at inference time with lightweight pilot calibration.
- SADB is more robust to NFE variation than schedules optimized for a single NFE target.
- SADB can adapt to CFG scale, prompt distribution, and model without re-running offline optimization.
- SADB can approach offline-optimized schedules in certain settings.
- SADB improves over the base schedule in certain low-NFE settings.

### Claims forbidden unless explicitly proven

- "SADB outperforms AYS."
- "SADB is universally better than the base schedule."
- "SADB is optimal."
- "AYS cannot handle [condition]." (AYS can; it just requires recomputation.)
- "SADB provides theoretical guarantees."
- "SADB is model-agnostic" unless all required adapters and experiments support it.
- "SADB is faster than AYS" unless calibration cost is accounted for correctly.

### Framing AYS accurately

AYS is a principled offline schedule optimization method. It can be applied to different models, solvers, and NFE values, but each combination requires its own optimization run, and published assets cover only selected settings. Do not say AYS "cannot adapt" — say "re-optimizing AYS for each deployment condition adds offline cost." Our difference is inference-time execution and NFE robustness, not AYS's fundamental inability.

---

## Paper structure

Target: standard NeurIPS conference paper. Use the official NeurIPS LaTeX template if present.

### Title candidates

- "Inference-Time Schedule Calibration for Diffusion Sampling"
- "Training-Data-Free Adaptive Schedule Calibration for Diffusion Models"
- "Defect-Balanced Clock Calibration for Low-Step Diffusion Sampling"
- "When Fixed Schedules Fall Short: Inference-Time Adaptive Clocks for Diffusion Sampling"

Avoid titles implying universal superiority.

### Abstract (four moves)

1. **Problem**: low-NFE diffusion sampling is sensitive to the sampling schedule.
2. **Gap**: offline-optimized schedules require recomputation per model/solver/NFE/condition; inference-time adaptation is understudied.
3. **Method**: SADB estimates a defect profile via lightweight pilot calibration and exports a non-uniform schedule.
4. **Findings**: actual results — approaching offline schedules, improving over base in certain settings, adapting to CFG, or working on models without published schedules.

### Introduction (paragraph plan)

1. Diffusion models are high quality; low-step sampling is schedule-sensitive.
2. Prior work focused on solvers; schedule optimization (e.g., AYS) showed schedules themselves matter.
3. Offline-optimized schedules require re-running for new conditions; inference-time adaptation is the open problem.
4. SADB: lightweight, training-data-free inference-time calibration.
5. Key insight: schedule calibration must be solver- and condition-aware.
6. Contributions.

### Contributions

1. SADB: a training-data-free inference-time schedule calibration framework.
2. CFG-aware calibration for modern text-to-image pipelines (SD1.5/SDXL).
3. Empirical comparison against base and offline-optimized schedules; quality-cost analysis.
4. Robustness study across NFE and guidance scale.
5. Failure mode analysis: solver-schedule mismatch, sigma grid choices, timestep snapping.

Only keep contributions supported by completed experiments.

### Related Work

Organize by theme: fast diffusion sampling and solvers / sampling schedule optimization / adaptive or learned timestep schedules / CFG and text-to-image evaluation / preference-based metrics (if used).

### Method (subsections)

1. Background: diffusion sampling schedules and low-NFE discretization error.
2. SADB overview: pilot trajectory → local defect estimation → profile smoothing → schedule materialization.
3. CFG-aware calibration.
4. Schedule materialization: sigma grid, timestep snapping, monotonicity, metadata.
5. Cost accounting: real model-evaluation-equivalent cost including CFG doubling.

### Experiments

1. **Baseline comparison** — SD1.5, SDXL; DPM-Solver++ and SDE-DPM-Solver++; base vs offline-optimized vs SADB; 10 NFE; ≥3 seeds; CLIPScore, ImageReward, pairwise win rate.
2. **Pilot cost ablation** — SADB-small/medium/large; quality vs calibration cost curve.
3. **NFE robustness** — compare SADB and fixed offline schedules at varying NFE.
4. **Guidance adaptation** — SDXL; guidance scales 3.0/5.0/7.5/10.0.
5. **Models without published offline schedules** — SD3.5, FLUX, Lumina if available; base vs SADB only.
6. **Failure analysis** (appendix) — solver mismatch, sigma grid choices, prompt-ensemble vs single-prompt, timestep snapping sanity checks.

### Tables and figures

- Table 1: SD1.5/SDXL — base / offline-optimized / SADB, CLIPScore and ImageReward.
- Table 2: Pairwise win rates.
- Figure 1: Calibration cost vs quality.
- Figure 2: NFE robustness or guidance adaptation curve.
- Table 3 / appendix: models without published schedules.
- Appendix: schedule visualizations, defect profiles, ablations.

Every table must specify model, solver, NFE, seed count, prompt set, metric direction, and guidance scale.

### Limitations (required, honest)

- SADB is not guaranteed to match or beat offline-optimized schedules.
- Quality depends on prompt representativeness in pilot calibration.
- Solver-schedule mismatch can degrade results.
- Metrics (CLIPScore, ImageReward) are imperfect proxies for human preference.
- Pilot calibration adds overhead; cost accounting must be fair.
- Current experiments may not establish superiority on all models or solvers.

---

## Code rules

### General

- Keep changes minimal and targeted. Do not mix refactors with experiment changes.
- Do not commit model weights, generated images, large caches, or environment files.
- Keep scripts deterministic when seeds are provided.
- Prefer small, focused commits with clear messages.

### Mandatory checks before expensive experiments

1. `python -m py_compile` passes for changed files.
2. Dry-run expansion counts are sensible.
3. Schedule paths are unique across seed/model/solver/guidance scale/schedule variant.
4. Base, offline, and SADB smoke generation each produce the expected number of images.
5. Scoring scripts produce detail and aggregate CSVs without NaN.
6. Pairwise win-rate scripts find matched prompt/seed pairs.
7. SADB schedule metadata records the correct model, solver, guidance scale, NFE, cache version, and pilot config.

### SADB cache and metadata

Bump the cache version whenever SADB semantics change. Metadata must include: model, production solver, calibration solver, schedule family, NFE, seed, guidance scale, prompt asset, pilot batch size, pilot batches, microbatch size, physical grid size/mode, schedule representation, timestep snapping mode, cache version, calibration cost estimate.

Never reuse a SADB cache across different guidance scales.

### CFG rules

Calibration and inference must use the same guidance scale unless explicitly testing transfer.

For SD/SDXL: latent, timestep, prompt embeddings, pooled embeddings, time IDs, and timestep conditioning must have matching leading batch dimensions. Microbatching must preserve prompt-to-latent alignment; condition embeddings must be sliced by absolute indices, not local batch size. A smoke check should verify manual calibration UNet calls match pipeline denoising calls for a fixed latent, timestep, prompt, and guidance scale.

### Sigma and timestep rules

- Do not assume a linear sigma grid is appropriate.
- Prefer scheduler-native, log-sigma, or Karras-style grids for calibration ablations.
- Ensure exported timesteps are explicit and monotone.
- Use round/nearest-neighbor snapping with monotonic repair; record snap error in metadata.
- Verify a custom base-equivalent timestep bundle reproduces base scheduler behavior before trusting comparisons.

### Metrics

Use CLIPScore (higher is better), ImageReward (higher is better), and pairwise win rate. Always include paired comparisons when possible. For pairwise comparisons, match on model/solver/NFE/seed/prompt index/guidance scale.

### Experiment output paths

Must encode: experiment name, model, solver, schedule, NFE, seed, guidance scale. Aggregate CSVs must preserve guidance scale and schedule directory.

### High-priority tests

1. CFG batch-dimension test for SD1.5 and SDXL.
2. Microbatch prompt-alignment test.
3. Custom base-equivalent schedule test.
4. Timestep snapping monotonicity test.
5. Schedule path resolution test (base, offline, SADB).
6. SADB cache partition by guidance scale test.
7. Scoring aggregation and pairwise matching test.
8. Physical grid mode test (linear / scheduler-native / log/Karras sigma).

### Ambiguous results checklist

1. Check cache version and schedule metadata.
2. Check calibration guidance equals inference guidance.
3. Check prompt coverage in pilot.
4. Check microbatch prompt-to-latent alignment.
5. Check physical grid mode and timestep snapping.
6. Compare sensitivity across solvers (Euler, Heun, DPM++, SDE-DPM++).
7. Report the failure mode; do not suppress it.

---

## Codex task style

**For code changes:** identify exact files and functions, make the smallest correct change, add focused tests, run cheapest validation first, report what was changed / tested / untested. Do not run expensive GPU experiments unless explicitly requested.

**For paper text:** start from current experimental evidence, avoid overclaiming, use NeurIPS-style concise writing, include model/solver/NFE/metric details, present negative or mixed results as analysis.

---

## Current research priority

1. Does SADB improve over the base schedule on SD1.5/SDXL at 10 NFE?
2. Does SADB approach offline-optimized schedules at lower calibration cost?
3. Is SADB more robust than fixed schedules when NFE or CFG changes?
4. Does SADB work on models without published offline schedules?
5. Which failure modes explain weak or negative results?

All code and writing should serve answering these questions clearly.
# NeurIPS 2026 Project Goal: SADB / D-GPDE Schedule Calibration

This repository is a research project targeting a NeurIPS 2026 submission. The paper studies training-data-free, inference-time adaptive sampling schedule calibration for diffusion and flow-based generative models. The working method is SADB / D-GPDE, as specified in `method.md`.

The paper must use `paper/` as the single source of truth for manuscript content, figures, tables, appendices, checklist material, and final polished artifacts. The NeurIPS 2026 style files already present in `paper/` must be used for all LaTeX work.

## 1. Core Objective

Develop a complete NeurIPS-style paper showing when inference-time defect-balanced schedule calibration improves or approaches strong fixed schedules in low-number-of-function-evaluations (low-NFE) diffusion sampling.

The central claim must remain scientifically bounded:

- SADB / D-GPDE is training-data-free.
- SADB / D-GPDE calibrates the sampling schedule at inference time using lightweight pilot runs.
- SADB / D-GPDE is solver- and condition-aware.
- SADB / D-GPDE can improve over the base schedule in selected low-NFE settings.
- SADB / D-GPDE can approach offline-optimized schedules in selected settings.
- SADB / D-GPDE may be more robust than fixed schedules when NFE, guidance scale, model, or prompt distribution changes.

Do not claim universal superiority. Do not claim that AYS cannot adapt. AYS is an offline schedule optimization method that can be rerun for each model, solver, NFE, and condition; SADB / D-GPDE differs by running calibration at inference time.

## 2. Required Agent Collaboration

For major project phases, use at least three specialized agents. The coordinator integrates their outputs and enforces scientific consistency.

### Experiment Agent

Responsibilities:

- Run CIFAR-10, SD1.5, SDXL, and available model experiments.
- Expand experiment YAMLs when needed.
- Export SADB / D-GPDE schedules with complete metadata.
- Run base, AYS / offline, and SADB / D-GPDE comparisons.
- Produce detail CSVs, aggregate CSVs, score summaries, pairwise win rates, and schedule metadata checks.
- Use large final evaluation settings when compute permits, including CIFAR-10 FID with 50k generated samples for paper-level claims.

### Code / Debug Agent

Responsibilities:

- Fix experiment code when failures occur.
- Add focused tests for CFG alignment, schedule path resolution, timestep snapping, sigma-grid materialization, scoring aggregation, pairwise matching, and cache partitioning.
- Run cheap validation before expensive jobs:
  1. `python -m py_compile` for changed Python files.
  2. Dry-run expansion count checks.
  3. Base / AYS / SADB smoke generation.
  4. Scoring scripts without NaN.
  5. Pairwise matching sanity checks.

### Paper Writing Agent

Responsibilities:

- Write and revise all manuscript content under `paper/`.
- Use `method.md` as the authoritative method description.
- Use `diffusion_schedule_papers_summary.md` and verified literature as Related Work and experiment-design references.
- Emphasize NeurIPS-style writing: concise technical prose, bounded claims, explicit metrics, reproducibility details, and no marketing language.
- Use the `ml-paper-writing` and `paper-writing` skills for all paper drafting, revision, LaTeX, and NeurIPS checklist work.

### Figure / Visualization Agent

Responsibilities:

- Use the `academic-plotting` skill for all paper figures.
- Use matplotlib / seaborn for data-driven plots.
- Use OpenAI image generation, preferably `gpt-image-2` when available, for method overview diagrams or conceptual visual figures.
- Save figure source data, plotting scripts, and rendered assets under `paper/`.
- Ensure figures are publication-quality, readable in grayscale when possible, and consistent with NeurIPS formatting.

Additional agents may be used for literature review, internal academic review, prompt-set construction, or artifact cleanup when useful.

## 3. Skill Usage Requirements

Use skills whenever the task matches an available skill. In particular:

- Use `ml-paper-writing` for NeurIPS paper structure, citations, related work, and reproducibility checklist guidance.
- Use `paper-writing` for any `.tex` writing, section revision, prose polishing, or final manuscript editing.
- Use `academic-plotting` for experiment figures, ablation plots, calibration-cost curves, schedule visualizations, and method diagrams.
- Use `humanizer` only as a final prose pass when the draft already satisfies technical and scientific constraints.
- Use `academic-paper-reviewer` or `ara-rigor-reviewer` for internal review before submission if time permits.

Do not fabricate citations. Every citation must be verified from a real paper source or marked as a placeholder requiring human verification.

## 4. Required Experiment Program

The experiment plan should be shaped by repository resources, `method.md`, and schedule-optimization references summarized in `diffusion_schedule_papers_summary.md`, especially AYS and score-optimal diffusion schedules.

### 4.1 CIFAR-10 Schedule Calibration

Purpose:

- Establish controlled low-NFE behavior with FID.
- Compare base schedules, AYS / offline schedules, and SADB / D-GPDE.
- Study solver coverage and schedule materialization failures in a cheaper setting before text-to-image runs.

Required settings:

- Final paper FID uses 50k generated images.
- Pilot / debugging FID may use 5k or 10k images, but cannot support final claims alone.
- Evaluate NFE values such as 5, 10, and 15, with additional NFE values if compute permits.
- Include seed count in every table. Use more than one seed for claims about statistical reliability.
- Report mean and uncertainty when multiple seeds are used.

Required ablations:

- Base vs SADB / D-GPDE.
- AYS / offline schedule vs SADB / D-GPDE when available.
- CVaR aggregation vs mean aggregation.
- Tangential / transverse residual weighting (`rho`) and metric variants.
- Physical grid mode: scheduler-native, log-sigma, Karras-style, or other implemented modes.
- Solver compatibility: Euler, Heun, PNDM-compatible solvers, DPM-Solver++, UniPC, EDM-style solvers, and any other supported solver whose adaptation passes smoke tests.

### 4.2 Text-to-Image Experiments

Purpose:

- Evaluate whether SADB / D-GPDE improves low-NFE generation quality under classifier-free guidance (CFG).
- Compare base, AYS / offline, and SADB / D-GPDE schedules under modern diffusers pipelines.

Required settings:

- Use SD1.5 and SDXL at minimum if compute and dependencies allow.
- Use NFE 10 as a primary setting, with NFE 5 and 15 for robustness when feasible.
- Use guidance scales such as 3.0, 5.0, 7.5, and 10.0 for CFG adaptation.
- Use expanded pilot and evaluation prompt sets. Prompt sets must be saved and versioned.
- Use at least three seeds for paper-level text-to-image claims when feasible.

Required metrics:

- CLIPScore, higher is better.
- ImageReward, higher is better.
- Pairwise win rate matched by model, solver, NFE, seed, prompt index, and guidance scale.

Required comparisons:

- Base schedule.
- AYS / published or project-owned offline schedules where available.
- SADB / D-GPDE calibrated with matching guidance scale.
- SADB / D-GPDE pilot-cost variants.
- Schedule transfer or mismatch cases only as ablations or failure analysis.

### 4.3 Solver and Schedule Robustness

Purpose:

- Demonstrate that schedule calibration is not solver-invariant.
- Identify where SADB / D-GPDE helps, matches, or degrades.

Required checks:

- Verify custom base-equivalent schedules reproduce base scheduler behavior before using custom timestep bundles.
- Verify exported timesteps / sigmas are explicit and monotone.
- Record snapping mode, snap error, sigma-grid mode, solver name, NFE, seed, model, guidance scale, prompt asset, cache version, pilot config, and calibration cost in metadata.
- Do not reuse SADB / D-GPDE cache across different guidance scales.
- Remove unstable solvers from main tables only after recording their failure mode in appendix or experiment notes.

### 4.4 Pilot Cost and Deployment Cost

Purpose:

- Account for calibration overhead fairly.
- Compare quality against real model-evaluation-equivalent cost.

Required outputs:

- Calibration cost vs quality curve.
- Pilot batch size / pilot batch count ablation.
- CFG-aware cost accounting, including the effective doubled UNet calls when applicable.
- Schedule reuse analysis across NFE values if the cache supports it.

## 5. Result Selection and Scientific Honesty

Main-paper tables and figures should emphasize clean, interpretable, and well-supported results. Weak, unstable, or visually bad outputs do not need to appear in the main paper unless they are central to a claim.

However, do not delete negative evidence from the project record. Negative or mixed results must be preserved in at least one of:

- appendix tables,
- failure-analysis sections,
- experiment logs,
- archived CSVs,
- schedule metadata,
- internal notes under `paper/` or `outputs/`.

Acceptable phrasing:

- "SADB improves over the base schedule in this setting."
- "SADB approaches the offline schedule under this model / solver / NFE."
- "This solver shows schedule mismatch, which degrades SADB under the current adapter."
- "The result is noisy and requires more seeds."

Forbidden phrasing unless directly proven:

- "SADB universally outperforms AYS."
- "SADB is optimal."
- "AYS cannot adapt."
- "The result is significant" without adequate seeds or confidence evidence.

## 6. Paper Artifacts

All paper content must live under `paper/`.

Required structure:

- `paper/neurips_2026.tex`: main paper entry point or imported template base.
- `paper/sections/`: abstract, introduction, method, experiments, related work, limitations, conclusion.
- `paper/figures/`: rendered PDF / PNG figures.
- `paper/figures/src/`: plotting scripts and diagram-generation prompts.
- `paper/tables/`: generated LaTeX tables and table source CSVs.
- `paper/results/`: cleaned aggregate CSVs used by the paper.
- `paper/appendix/`: extra experiments, failure analysis, prompt lists, schedule visualizations, reproducibility details.
- `paper/references.bib`: verified bibliography only.
- `paper/project_context.md`: locked paper framing, claims, experiment-to-claim mapping, and open questions.

Every table and figure caption must specify model, solver, schedule, NFE, seed count, prompt set, metric direction, and guidance scale when applicable.

## 7. Figure Plan

Required paper figures:

- Method overview: pilot trajectories -> defect estimation -> profile aggregation -> schedule materialization -> generation.
- Schedule visualization: base, AYS / offline, and SADB / D-GPDE sigma or timestep grids.
- Calibration cost vs quality curve.
- NFE robustness curve.
- CFG guidance adaptation curve.
- Main text-to-image qualitative grid with matched prompts, seeds, model, solver, NFE, and guidance scale.

Data plots must be reproducible from CSVs. Conceptual diagrams generated with image models must store their prompts and generation metadata.

## 8. Paper Narrative

The paper should follow a NeurIPS-style structure:

1. Low-NFE diffusion sampling is schedule-sensitive.
2. Offline schedule optimization, including AYS, shows that schedules matter but requires optimization for each deployment condition.
3. SADB / D-GPDE estimates a local defect profile using training-data-free pilot calibration at inference time.
4. The method materializes a solver- and condition-aware schedule.
5. Experiments evaluate when this helps, when it approaches offline schedules, and when solver or CFG mismatch causes failures.

The method section must be derived from `method.md`. The related-work section must use verified references and the project summary in `diffusion_schedule_papers_summary.md`.

## 9. Final Submission Checklist

Before treating the paper as submission-ready:

- Run final CIFAR-10 FID with 50k samples for claims that rely on FID.
- Run final text-to-image metrics with saved prompt assets and matched seed / prompt pairs.
- Regenerate all tables and figures from scripts.
- Verify no NaN appears in detail or aggregate metrics.
- Verify pairwise win-rate matching keys.
- Verify SADB / D-GPDE metadata includes model, solver, NFE, seed, guidance scale, prompt asset, pilot config, schedule representation, timestep snapping mode, cache version, and calibration cost.
- Run LaTeX compilation from `paper/`.
- Complete the NeurIPS checklist.
- Run an internal review pass with `academic-paper-reviewer` or equivalent.
- Confirm that main-paper claims are exactly supported by retained evidence.

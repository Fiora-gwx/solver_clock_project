
# NeurIPS 2026 Project Goal: D-GPDE Schedule Calibration

This repository is a research project targeting a NeurIPS 2026 submission. The project studies **training-data-free, inference-time schedule calibration** for low-NFE deterministic diffusion / flow ODE sampling.

The method name is **D-GPDE**: **Distributional Geometric Prediction Defect Equalization**. `method.md` is the authoritative mathematical specification.

**Naming rule:**

- **D-GPDE** is the paper method and the only method name to use in the manuscript.
- **GOES** is the implementation / code-folder / experiment-runner naming used in this repository. In prose, write “the GOES implementation of D-GPDE” only when referring to code.
- **`gpde` in configs, paths, or experiment names refers to D-GPDE.**
- **SADB is deprecated.** Do not use SADB as a method name, paper title term, figure label, table label, or main experimental label. If old files mention SADB, treat them as legacy naming and migrate paper-facing language to D-GPDE.

The paper must be scientifically bounded: the goal is not to claim universal superiority over all fixed or offline schedules. The goal is to demonstrate, with theory and experiments, when inference-time defect-balanced schedule calibration improves, matches, or fails relative to strong baselines such as native schedules, EDM / Karras-style schedules, AYS / offline optimized schedules, and other schedule-optimization references summarized in `diffusion_schedule_papers_summary.md`.

All manuscript content, final figures, final tables, cleaned results, appendix material, checklist material, and final artifacts must live under `paper/`. The existing NeurIPS 2026 style files in `paper/` must be used without modifying conference formatting.

---

## 0. Non-Negotiable Operating Rules

1. **Use D-GPDE as the method name.** GOES is implementation naming only. SADB is deprecated.
2. **Do not fabricate results, citations, seeds, metrics, or visual comparisons.**
3. **Do not delete negative evidence.** Bad or unstable results may be omitted from the main paper, but must remain in appendix tables, logs, archived CSVs, failure analysis, or experiment notes.
4. **Do not claim universal optimality.** D-GPDE is a schedule calibration method, not a proof of globally optimal generation quality.
5. **Do not present teacher-alignment as image-quality evidence.** Teacher-distance improvements and generation-quality improvements must be reported separately.
6. **Do not present empirical-only solver paths as theoretically guaranteed.** Deterministic single-step ODE settings are theory-supported; multistep, stochastic, ancestral, or scheduler-history paths require empirical framing unless corresponding theory and implementation are added.
7. **Every main-paper claim must map to retained evidence.** If evidence is weak, noisy, or missing, downgrade the claim.
8. **Every table and figure caption must state model, solver, schedule, NFE, seed count, prompt set or dataset split, metric direction, guidance scale when applicable, and whether the result is calibration, held-out, or final evaluation.**

---

## 1. Core Objective

Develop a complete NeurIPS-style paper showing that **D-GPDE** provides a practical, training-free way to adapt low-NFE sampling schedules to model, solver, NFE, and condition through inference-time defect balancing.

The central bounded claims are:

- D-GPDE is **training-data-free**: it does not update model weights.
- D-GPDE uses **pilot calibration / teacher-oracle information** to estimate local solver prediction defects.
- D-GPDE materializes a **solver-aware and condition-aware schedule**.
- D-GPDE can improve over native or heuristic schedules in selected low-NFE deterministic ODE settings.
- D-GPDE can approach offline optimized schedules, such as AYS-style schedules, in selected settings.
- D-GPDE may be more robust than fixed schedules when NFE, guidance scale, model, or prompt distribution changes.
- D-GPDE has limitations: few-step regimes may violate local asymptotics; multistep and stochastic solvers may require empirical replay or additional theory; teacher-alignment may not always improve perceptual quality.

Forbidden claims unless directly proven by retained evidence:

- D-GPDE universally outperforms AYS.
- D-GPDE is globally optimal.
- AYS cannot adapt.
- Teacher-distance improvement guarantees FID, ImageReward, CLIPScore, or human preference improvement.
- A result is statistically significant without adequate seeds, uncertainty estimates, or matched-pair evidence.

---

## 2. Agent Collaboration Protocol

For every major phase, use at least three specialized agents. The Main Agent coordinates them, resolves conflicts, and enforces claim discipline.

### 2.1 Main / Coordinator Agent

Responsibilities:

- Read `goal.md`, `method.md`, `diffusion_schedule_papers_summary.md`, `docs/GOES_COMPLETION_AUDIT.md`, and the current `paper/` state before starting.
- Maintain a project state file under `paper/project_context.md` recording current claims, evidence, missing runs, risks, and next actions.
- Enforce the naming rule: D-GPDE is the method; GOES is implementation; SADB is deprecated.
- Assign tasks to specialized agents and merge their outputs.
- Stop any claim from entering the paper unless the claim is backed by retained evidence.
- Keep theory-supported and empirical-only settings clearly separated.
- Keep all final paper artifacts under `paper/`.

### 2.2 Experiment Agent

Responsibilities:

- Run CIFAR-10, SD1.5, SDXL, and other available model experiments according to the staged plan below.
- Expand experiment YAMLs only when the additional run supports a claim in the claim-to-evidence matrix.
- Use repository GOES scripts/configs as the implementation of D-GPDE.
- Run base, AYS / offline, and D-GPDE comparisons when available.
- Export D-GPDE schedules with complete metadata.
- Produce detail CSVs, aggregate CSVs, score summaries, schedule metadata checks, calibration-cost reports, and pairwise win rates.
- Use paper-grade evaluation settings for final claims, including CIFAR-10 FID with 50k generated samples when FID is used as a main-paper claim.
- Preserve failed and unstable runs with explicit failure modes.

### 2.3 Code / Debug Agent

Responsibilities:

- Fix experiment code when failures occur.
- Add focused tests for CFG partitioning, schedule path resolution, timestep snapping, sigma-grid materialization, scoring aggregation, pairwise matching, schedule verification, cache partitioning, and metadata completeness.
- Run cheap validation before expensive jobs:
  1. `python -m py_compile` for changed Python files.
  2. Targeted unit tests for changed modules.
  3. Dry-run expansion count checks.
  4. Base / AYS / D-GPDE smoke generation when applicable.
  5. Scoring scripts with no NaN and non-empty outputs.
  6. Pairwise matching sanity checks.
  7. Schedule monotonicity and staleness checks.
- Never patch code only to hide bad results.
- Do not rename stable code folders just for paper terminology; instead map GOES implementation artifacts to D-GPDE paper-facing labels.

### 2.4 Paper Writing Agent

Responsibilities:

- Write and revise all manuscript content under `paper/`.
- Use `method.md` as the authoritative method description.
- Use `diffusion_schedule_papers_summary.md` and verified literature as Related Work and experiment-design references.
- Use **D-GPDE** consistently as the method name.
- Avoid **SADB** in paper-facing text except when explicitly noting that it is deprecated legacy naming, if needed.
- Mention **GOES** only when describing repository implementation, scripts, or experiment artifacts.
- Emphasize NeurIPS-style writing: concise technical prose, bounded claims, explicit assumptions, reproducibility details, and no marketing language.
- Use `ml-paper-writing`, `ml-writing`, `paper-writing`, or equivalent writing skills whenever available.
- Keep the main paper self-contained within the NeurIPS page limit.
- Complete the NeurIPS checklist honestly.

### 2.5 Figure / Visualization Agent

Responsibilities:

- Use `academic-plotting` or equivalent plotting skills for all paper plots.
- Use reproducible plotting scripts for all data-driven figures.
- Use OpenAI image generation, preferably `gpt-image-2` when available, only for conceptual method overview diagrams or visual schematics, not for fabricated experimental evidence.
- Figure labels must use **D-GPDE**, not SADB. GOES may appear only in implementation notes or file paths.
- Save conceptual diagram prompts, generation metadata, and rendered assets under `paper/figures/src/` and `paper/figures/`.
- Save all data-driven plotting scripts and source CSVs.
- Ensure plots are publication-quality, legible in grayscale when possible, and consistent with NeurIPS formatting.

### 2.6 Literature / Rigor Review Agent

This agent is mandatory before any submission-ready claim.

Responsibilities:

- Verify all citations and bibliography entries.
- Check whether each cited paper is used accurately.
- Review theory assumptions and identify overclaims.
- Review experiment design for unfair comparisons, leakage, cherry-picking, missing baselines, missing uncertainty, and mismatched compute cost.
- Check that method naming is consistent: D-GPDE as method, GOES as implementation, SADB deprecated.
- Produce a `claim -> evidence -> risk -> action` review table under `paper/internal_reviews/`.
- Use `academic-paper-reviewer`, `ara-rigor-reviewer`, or equivalent review skills when available.

---

## 3. Skill Usage Requirements

Use available skills whenever the task matches them.

- Use `ml-paper-writing` / `ml-writing` for NeurIPS framing, abstract, introduction, related work, contribution statements, and reproducibility discussion.
- Use `paper-writing` for `.tex` drafting, section revision, prose polishing, captions, appendix organization, and checklist work.
- Use `academic-plotting` for experiment figures, ablation plots, calibration-cost curves, schedule visualizations, and figure style cleanup.
- Use `academic-paper-reviewer` or `ara-rigor-reviewer` for internal review before submission.
- Use `humanizer` only after technical correctness and claim discipline are already satisfied.
- Use code/debug/test skills whenever experiments fail or outputs look inconsistent.
- Use image-generation tools only for conceptual figures; never use generated images as experimental samples unless explicitly labeled as conceptual art.

---

## 4. Theory Scope and Claim Boundaries

### 4.1 Theory-supported setting

The main theory covers:

- deterministic ODE samplers;
- oracle-start local prediction defect;
- high-accuracy teacher oracle;
- fixed model, solver, NFE, and condition;
- local power-law defect model;
- distributional risk aggregation over teacher marginal samples;
- G-metric and rho-mixed residuals;
- monitor density proportional to the q-th root of the defect coefficient;
- equal-monitor-mass schedule construction;
- asymptotic local minimax optimality for distributional local risk under stated regularity assumptions.

### 4.2 Empirical-only or limited-theory extensions

The following settings must be described as empirical unless additional theory and implementation are added:

- multistep solvers without oracle-consistent history injection;
- DPM-Solver++, UniPC, PNDM scheduler-history paths, or other history-dependent solver states;
- stochastic SDE samplers or ancestral samplers;
- classifier-free guidance regions where effective dynamics are highly nonlinear;
- very low NFE regimes where local asymptotics may fail;
- black-box replay refinement;
- text-to-image perceptual quality improvements.

### 4.3 Main theory to place in the paper

The main paper should include:

1. deterministic ODE problem setup;
2. teacher oracle and oracle-start defect;
3. G-metric and rho-mixed residual;
4. distributional local power-law defect;
5. risk aggregation with mean / CVaR / mixed risk;
6. correct monitor density `omega(u) = (a_bar(u) + epsilon)^(1/q)`;
7. inverse-CDF / equal-monitor-mass schedule construction;
8. one main theorem: asymptotic minimax optimality for distributional local risk;
9. a clear paragraph on theory boundaries.

### 4.4 Theory to place in the appendix

The appendix should include:

- full theorem proofs;
- coordinate covariance of the monitor one-form;
- inverse-CDF stability under monitor estimation error;
- q estimation and local q(u) extension;
- propagated defect analysis;
- multistep augmented-state defect discussion;
- why replay defect is not the same as oracle-start local defect;
- relation to AYS KLUB and Score-Optimal Diffusion Schedules;
- additional assumptions and failure modes.

---

## 5. Claim-to-Evidence Matrix

The paper must maintain this matrix in `paper/project_context.md` and update it after every experiment batch.

| Claim | Required evidence | Main / appendix | Fallback if evidence fails |
| --- | --- | --- | --- |
| D-GPDE improves low-NFE deterministic ODE sampling in controlled settings | CIFAR-10, 50k FID for final claims, multiple seeds when feasible, base / offline / D-GPDE comparisons, NFE 10 and 20 at minimum | Main | Reframe as teacher-alignment or schedule-calibration diagnostic if FID does not improve |
| D-GPDE is condition-aware under CFG | SD1.5 or SDXL, matched prompts/seeds, at least two guidance scales, base / offline / D-GPDE comparisons, CLIPScore/ImageReward/pairwise win rate | Main or appendix | Treat as mixed result; show where CFG mismatch hurts |
| D-GPDE approaches offline schedules in selected settings | AYS / published / project-owned offline schedule comparison with same model, solver, NFE, seed, prompt set, and metric | Main only if strong; otherwise appendix | State that offline schedules remain stronger in that setting |
| Defect monitor explains schedule allocation | schedule plots, per-edge defect distributions, monitor mass equalization diagnostics, teacher-alignment metrics | Main diagnostic figure | Keep as interpretability evidence, not quality evidence |
| Calibration overhead is acceptable | effective model-evaluation-equivalent cost vs quality curve, pilot-size ablation, CFG-aware cost accounting | Main | Claim only quality change, not deployment efficiency |
| Method generalizes across solvers | Euler / Heun main; other solvers in appendix if implemented and stable | Appendix unless strong | Mark unsupported or unstable solvers as empirical-only or failure cases |
| Negative and failure cases are understood | failure-case CSVs, bad examples, solver mismatch notes, schedule metadata, appendix discussion | Appendix | Do not remove failure evidence |

---

## 6. Required Experiment Program

Experiments must be staged. Do not jump directly to large sweeps before passing cheaper gates.

### 6.1 Stage 0: repository and paper inventory

Before running expensive experiments:

- Read `docs/GOES_COMPLETION_AUDIT.md` and identify what is implemented, dry-run only, or missing.
- Interpret GOES code/config outputs as the implementation of D-GPDE.
- Verify that `paper/neurips_2026.tex` is no longer just the template before treating the manuscript as a draft.
- Create or update `paper/project_context.md` with:
  - current method name and framing;
  - current supported solver scope;
  - claim-to-evidence matrix;
  - experiment status;
  - open risks;
  - next commands;
  - naming map: D-GPDE = method, GOES = implementation, SADB = deprecated.

### 6.2 Stage 1: cheap correctness gates

Run these before any expensive job:

- `python -m py_compile` for changed Python files.
- Relevant unit tests for GOES implementation, schedule verification, launcher expansion, scoring, and pairwise matching.
- Launcher preview for intended experiment configs.
- Schedule export dry-runs where supported.
- Smoke generation / scoring for a tiny subset when model loading is allowed.
- Check for NaN, empty CSVs, missing image files, bad schedule monotonicity, stale schedule dirs, and mismatched metadata.

A minimal first real GPU gate should run one end-to-end benchmark before broader sweeps. Prefer the existing smoke config if available:

```bash
python scripts/run/run_experiment_config.py \
  --experiment-config configs/experiments/goes_pndm_smoke.yaml \
  --materialize-schedules --execute
````

Only proceed to paper-grade sweeps after this gate produces valid schedules, images or samples, metrics, and metadata.

### 6.3 Stage 2: CIFAR-10 controlled benchmark

Purpose:

* Establish controlled low-NFE behavior with FID.
* Validate schedule materialization, solver compatibility, and evaluation pipeline before text-to-image runs.
* Provide the strongest quantitative paper anchor if results support it.

Required settings:

* Final paper FID uses 50k generated images.
* Pilot / debugging FID may use 5k or 10k images, but cannot support final claims alone.
* Evaluate NFE values including 10 and 20. Add 5, 15, 30, and 50 if compute permits.
* Use more than one seed when making reliability claims.
* Report mean and uncertainty when multiple seeds are used.

Required comparisons:

* Native / base schedule.
* EDM / Karras-style or other strong heuristic schedule when applicable.
* AYS / offline optimized schedule when available.
* D-GPDE, implemented through the GOES code path.

Required ablations:

* CVaR aggregation vs mean aggregation.
* rho weighting, e.g. 0, 0.05, 0.1, 0.3, 1.0.
* metric variants: identity, EDM scalar, channel-wise whitening when implemented.
* physical grid mode: scheduler-native, log-sigma, Karras-style, or implemented alternatives.
* calibration size.
* candidate grid size.
* oracle convergence / teacher reference NFE.

### 6.4 Stage 3: text-to-image benchmark

Purpose:

* Evaluate low-NFE generation quality under classifier-free guidance.
* Test condition-aware schedule calibration.
* Compare base, offline, and D-GPDE schedules under modern diffusers pipelines.

Required settings:

* Use SD1.5 and SDXL if dependencies and compute allow. If compute is limited, use one as the main benchmark and the other as appendix robustness.
* Use NFE=10 as the primary setting. Add NFE=5 and NFE=15 when feasible.
* Use guidance scales such as 5.0, 7.5, and 10.0; include 3.0 if testing weak guidance.
* Use saved and versioned prompt sets.
* Use matched prompt, seed, model, solver, NFE, and guidance scale across schedules.
* Use at least three seeds for paper-level claims when feasible.

Required metrics:

* CLIPScore, higher is better.
* ImageReward, higher is better.
* Pairwise win rate matched by model, solver, NFE, seed, prompt index, and guidance scale.
* Optional: HPS, aesthetic score, human preference if setup is available.

Required comparisons:

* Base schedule.
* AYS / published or project-owned offline schedule where available.
* D-GPDE calibrated with matching guidance scale.
* D-GPDE pilot-cost variants.
* Schedule transfer / mismatch cases only as ablations or failure analysis.

### 6.5 Stage 4: calibration cost and deployment cost

Purpose:

* Account for calibration overhead fairly.
* Compare against real model-evaluation-equivalent cost.

Required outputs:

* Calibration cost vs quality curve.
* Pilot batch size / pilot batch count ablation.
* CFG-aware cost accounting, including doubled UNet calls where applicable.
* Schedule reuse analysis across NFE, guidance scale, solver, or prompt distribution only if metadata proves valid reuse.

### 6.6 Stage 5: solver and schedule robustness

Purpose:

* Demonstrate that schedule calibration is not solver-invariant.
* Identify where D-GPDE helps, matches, or degrades.

Required checks:

* Verify custom base-equivalent schedules reproduce base scheduler behavior before using custom timestep bundles.
* Verify exported timesteps / sigmas are explicit and monotone.
* Record snapping mode, snap error, sigma-grid mode, solver name, NFE, seed, model, guidance scale, prompt asset, cache version, pilot config, and calibration cost in metadata.
* Do not reuse D-GPDE calibration cache across different guidance scales unless the cache key explicitly supports and validates that reuse.
* Remove unstable solvers from main tables only after recording their failure mode in appendix or experiment notes.

### 6.7 Stage 6: failure analysis

Required failure records:

* schedule mismatch cases;
* solver adapter failures;
* teacher-alignment improves but generation quality degrades;
* generation quality improves but teacher-alignment does not;
* unstable sigma / timestep snapping;
* CFG scale mismatch;
* NaN or empty metric outputs;
* visually bad qualitative samples;
* cases where offline AYS remains clearly stronger.

Failure cases should be summarized in appendix, not hidden.

---

## 7. Baseline Policy

At minimum, compare against:

* base / native schedule;
* uniform in the sampler's native coordinate when meaningful;
* uniform in sigma or log-SNR when meaningful;
* EDM / Karras-style schedule when applicable;
* AYS / published offline optimized schedules when available;
* project-owned offline optimized schedule if AYS is unavailable but offline optimization is implemented.

For models without published AYS schedules, compare base vs D-GPDE, and clearly state that offline optimized schedule comparison is unavailable.

All baseline schedules must use the same model, solver, NFE, seed, prompt set or dataset split, guidance scale, precision, and resolution unless a difference is explicitly justified.

---

## 8. Metrics and Reporting

Always separate calibration / teacher-alignment metrics from generation-quality metrics.

### 8.1 Teacher-alignment metrics

Report when available:

* endpoint latent MSE to teacher;
* held-out teacher MSE;
* LPIPS between student and teacher decoded samples when meaningful;
* per-step oracle-start defect distribution;
* selected edge costs;
* monitor mass distribution;
* schedule stability across calibration seeds;
* calibration vs held-out generalization gap.

### 8.2 Generation-quality metrics

Report when available:

* FID, lower is better;
* KID, lower is better, only if reference feature activations support it;
* CLIPScore, higher is better;
* ImageReward, higher is better;
* HPS / aesthetic score if implemented;
* pairwise win rate with matched keys;
* human preference only if protocol is controlled and recorded.

### 8.3 Uncertainty

When multiple seeds or matched comparisons are available:

* report mean;
* report standard error or confidence intervals;
* use bootstrap for pairwise or aggregate metrics when implemented;
* avoid significance language unless evidence supports it.

---

## 9. Paper Artifact Structure

All final paper artifacts must live under `paper/`.

Required structure:

* `paper/neurips_2026.tex`: main paper entry point using the NeurIPS 2026 style.
* `paper/sections/`: abstract, introduction, method, theory, experiments, related work, limitations, conclusion.
* `paper/figures/`: rendered PDF / PNG figures.
* `paper/figures/src/`: plotting scripts, source CSVs, conceptual diagram prompts, and image-generation metadata.
* `paper/tables/`: generated LaTeX tables and source CSVs.
* `paper/results/`: cleaned aggregate CSVs used by the paper.
* `paper/appendix/`: extra experiments, failure analysis, prompt lists, schedule visualizations, reproducibility details, proofs.
* `paper/references.bib`: verified bibliography only.
* `paper/project_context.md`: locked paper framing, claim-to-evidence matrix, open questions, and experiment status.
* `paper/internal_reviews/`: rigor reviews and pre-submission audits.

---

## 10. Main Paper Plan

The main paper should fit the NeurIPS page limit and remain self-contained.

Recommended structure:

1. **Introduction**

   * Low-NFE diffusion / flow sampling is schedule-sensitive.
   * Offline schedule optimization shows schedules matter but may require per-condition optimization.
   * D-GPDE provides inference-time defect-balanced schedule calibration.
   * Contributions must be bounded and evidence-backed.

2. **Related Work**

   * Diffusion and flow sampling solvers.
   * Hand-designed schedules.
   * AYS and KLUB-based schedule optimization.
   * Score-Optimal Diffusion Schedules.
   * Adaptive / learned / offline schedule methods.

3. **Method**

   * Deterministic ODE setup.
   * Teacher oracle.
   * Oracle-start defect.
   * G-metric and rho-mixed residual.
   * Distributional aggregation.
   * Monitor construction and schedule materialization.

4. **Theory**

   * Local power-law model.
   * Correct q-th-root monitor density.
   * Equal-monitor-mass theorem.
   * Asymptotic minimax local-risk theorem.
   * Theory boundaries.

5. **Experiments**

   * CIFAR-10 controlled benchmark.
   * Text-to-image benchmark.
   * Calibration cost vs quality.
   * Key ablations.
   * Failure or limitation example if space permits.

6. **Limitations**

   * Deterministic ODE theory scope.
   * Few-step asymptotic gap.
   * Multistep and stochastic solver limitations.
   * Calibration cost.
   * Teacher-alignment vs perceptual quality mismatch.

7. **Conclusion**

   * Summarize bounded contribution and future work.

---

## 11. Appendix Plan

The appendix should contain:

* full proofs;
* detailed algorithms;
* GOES implementation details for D-GPDE;
* schedule verification and metadata schema;
* full CIFAR-10 tables;
* full text-to-image tables;
* all solver robustness experiments;
* all rho / metric / aggregation / calibration-size / candidate-grid / oracle-convergence ablations;
* prompt lists and prompt-set hashes;
* qualitative grids, including failure cases;
* schedule visualizations for all models and solvers;
* exact commands and environment notes;
* additional internal review notes if useful.

Do not move essential evidence for a main-paper claim exclusively to the appendix. The main paper must stand alone.

---

## 12. Required Figure Plan

Main-paper figures should be few, strong, and reproducible.

### Figure 1: Method overview

Show:

pilot trajectories -> teacher oracle -> local defect estimation -> risk aggregation -> monitor / DP schedule construction -> materialized schedule -> generation.

This can use `gpt-image-2` for a conceptual draft, but the final figure must be clean, technical, and not misleading. Label the method as **D-GPDE**.

### Figure 2: Schedule visualization

Plot base, offline / AYS, and D-GPDE schedules for a representative model / solver / NFE / CFG.

Recommended axes:

* step index or normalized step on x-axis;
* sigma, timestep, or log-SNR on y-axis;
* include exact model, solver, NFE, guidance scale in caption.

### Figure 3: Main controlled benchmark

CIFAR-10 FID vs NFE, or a compact FID table if the curve is less clear.

### Figure 4: Calibration cost vs quality

Plot effective model-evaluation-equivalent cost against FID, ImageReward, or teacher MSE.

### Figure 5: Text-to-image qualitative grid

Matched prompts and seeds. Include base, offline / AYS if available, and D-GPDE. Captions must state model, solver, NFE, CFG, prompt set, and seed matching.

### Appendix figures

* rho ablation;
* CVaR vs mean;
* metric variants;
* q estimation stability;
* oracle convergence;
* calibration-size sensitivity;
* candidate-grid sensitivity;
* edge-cost heatmaps;
* failure cases;
* all schedule visualizations.

---

## 13. Result Selection and Scientific Honesty

Main paper results should emphasize clean, interpretable, well-supported findings. Weak or visually bad outputs do not need to appear in the main paper unless central to a claim.

However:

* every negative or mixed result must remain in the project record;
* appendix or failure analysis must mention important degradations;
* paper claims must reflect the full retained evidence, not only the best subset.

Acceptable phrasing:

* “D-GPDE improves over the base schedule in this setting.”
* “D-GPDE approaches the offline schedule under this model / solver / NFE.”
* “This solver shows schedule mismatch, which degrades D-GPDE under the current adapter.”
* “The result is noisy and requires more seeds.”
* “Teacher-alignment improves, but image-quality metrics do not improve in this setting.”

Forbidden phrasing unless directly proven:

* “D-GPDE universally outperforms AYS.”
* “D-GPDE is optimal.”
* “AYS cannot adapt.”
* “The result is significant” without adequate statistical evidence.

---

## 14. Final Submission Checklist

Before treating the paper as submission-ready:

* Replace the NeurIPS template text in `paper/neurips_2026.tex` with the actual manuscript.
* Verify all claims in `paper/project_context.md` map to retained evidence.
* Verify that D-GPDE is used as the method name throughout paper-facing artifacts.
* Verify that GOES appears only as implementation naming when needed.
* Verify that SADB does not appear as a current method name.
* Run final CIFAR-10 FID with 50k samples for any FID-based main-paper claim.
* Run final text-to-image metrics with saved prompt assets and matched seed / prompt pairs.
* Regenerate all tables and figures from scripts.
* Verify no NaN appears in detail or aggregate metrics.
* Verify no scoring output is empty.
* Verify pairwise win-rate matching keys.
* Verify D-GPDE / GOES metadata includes model, solver, NFE, seed, guidance scale, prompt asset or dataset split, pilot config, schedule representation, timestep snapping mode, cache version, and calibration cost.
* Verify negative and failure cases are preserved.
* Verify theory-supported and empirical-only settings are separated.
* Verify all citations in `paper/references.bib` correspond to real papers and are used accurately.
* Compile LaTeX from `paper/`.
* Complete the NeurIPS checklist honestly.
* Run an internal rigor review with `academic-paper-reviewer`, `ara-rigor-reviewer`, or equivalent.
* Confirm that main-paper claims are exactly supported by evidence.

---

## 15. First Execution Priority for the Main Agent

Start with this order:

1. Read `method.md`, `diffusion_schedule_papers_summary.md`, `docs/GOES_COMPLETION_AUDIT.md`, and current `paper/`.
2. Create / update `paper/project_context.md`.
3. Record the naming map: **D-GPDE = method**, **GOES = implementation**, **gpde config names = D-GPDE**, **SADB = deprecated**.
4. Run cheap tests and dry-runs.
5. Run one real end-to-end smoke benchmark.
6. Inspect schedules, metadata, generated outputs, and metrics.
7. Only then expand to CIFAR-10 paper-grade runs.
8. Then run text-to-image benchmarks.
9. Then run ablations and robustness sweeps.
10. Then write the main paper and appendix from the retained evidence.

The guiding principle is:

**Do not maximize the number of experiments. Maximize the clarity of the claim-evidence-theory loop.**



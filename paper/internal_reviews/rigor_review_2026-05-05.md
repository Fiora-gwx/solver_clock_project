# Internal Rigor Review

Date: 2026-05-05

Scope: read-only review of the current D-GPDE NeurIPS draft and retained paper
artifacts under `paper/`. This review used the academic-paper-reviewer
multi-reviewer protocol and the ARA rigor-reviewer six-dimension rubric as a
submission-readiness diagnostic. It is not a claim that the paper is ready.

Post-review update: after this review was written, `paper/checklist.tex` was
filled with honest current-state answers and the template instruction block was
removed. Several answers remain `No` because code/data release, compute
reporting, broader impacts, asset licenses, and new-asset documentation are
still incomplete. The bounded CIFAR 50k table was moved into the main
experiments section, a first calibration cost summary table/figure was added
from retained oracle-reuse CSVs, the reproducibility appendix was expanded with
environment, hardware, commands, and output paths, and the theory proof appendix
now contains the distributional-risk, coordinate-covariance, local minimax, and
inverse-CDF stability arguments. A broader-impact and asset-audit appendix and
the first concrete retained failure table were also added, along with an
evaluated-batch calibration amortization table. Strong CIFAR baselines,
qualitative failure grids, reuse validation across conditions, release
packaging, asset-license completion, and a final claim/citation audit remained
open at that point; later updates below address some of those items.

Second post-review update: a selected SD1.5 qualitative failure grid and
selection CSV were added under `paper/figures/` and `paper/results/failure/`.
Draft release metadata was added under `paper/release/`, and local absolute
paths were sanitized from paper-facing result artifacts. The current status is
summarized in `paper/internal_reviews/final_claim_audit_2026-05-05.md`.

Third post-review update: a native-coordinate linear PNDM/CIFAR-10 50k
baseline was run for seeds 0, 1, and 2 at NFE 10 and 20, and the main CIFAR
table/figure now includes base, linear, and D-GPDE schedules. This partially
addresses the original baseline finding, but stronger EDM/Karras/offline CIFAR
baselines remain absent.

Reviewers:

- Reviewer 1: methodology and experiment rigor.
- Reviewer 2: method, theory, citation, and naming rigor.
- Reviewer 3: devil's advocate, claim discipline, negative evidence, and
  checklist readiness.

Read order:

1. `goal.md`
2. `method.md`
3. `diffusion_schedule_papers_summary.md`
4. `paper/neurips_2026.tex`
5. `paper/project_context.md`
6. `paper/sections/*.tex`
7. `paper/appendix/*.tex`
8. `paper/tables/*.tex`
9. `paper/results/*/README.md`
10. `paper/internal_reviews/*.md`
11. `paper/references.bib`
12. `paper/checklist.tex`

## Overall Assessment

Recommendation: Reject / major revision in the current form.

The project has real evidence and unusually explicit claim discipline. The
strongest positive result is a three-seed, 50k-sample CIFAR-10 base-vs-D-GPDE
comparison for one PNDM/Euler setting at NFE 10 and 20. The strongest negative
result is the matched SD1.5 Euler NFE-10 CFG sweep, where D-GPDE trails base and
AYS on ImageReward and does not improve CLIPScore. These results support a
bounded schedule-calibration story, but the draft remains a scaffold: the theory
is still a target theorem without proof, the core positive result sits in the
appendix, calibration cost is recorded but not analyzed, failure analysis is
mostly a plan, and the NeurIPS checklist initially still used template
placeholders.

Strongest counter-argument: D-GPDE is motivated as model-, solver-, budget-, and
condition-aware inference-time calibration, but the only positive paper-grade
quality evidence is one base-only CIFAR setting. The one matched offline/AYS
comparison is a text-to-image negative result. A reviewer can therefore read the
method as a high-cost teacher-defect schedule fitter whose benefits have not yet
been shown to survive strong schedule baselines, perceptual metrics, or broad
conditions.

## ARA Rubric Scores

| Dimension | Score | Rationale |
| --- | ---: | --- |
| D1 Evidence relevance | 3/5 | The retained CIFAR and SD1.5 evidence is relevant to the stated bounded claims, but several paper-plan claims still point to missing figures, ablations, or cost analysis. |
| D2 Falsifiability quality | 3/5 | `paper/project_context.md` contains risks and fallback wording, but the manuscript itself does not yet state concrete falsification thresholds for theory, cost, or transfer claims. |
| D3 Scope calibration | 3/5 | The prose is mostly cautious and preserves negative evidence, but the title and motivation still imply broader schedule calibration than the positive evidence supports. |
| D4 Argument coherence | 3/5 | The problem-method-evidence loop is visible, but status/scaffold language interrupts the paper narrative and the strongest empirical result is outside the main experiments section. |
| D5 Exploration integrity | 4/5 | Negative SD1.5 evidence and stale-run risks are retained. The remaining gap is that several failure categories are described as planned records instead of completed tables. |
| D6 Methodological rigor | 2/5 | The paper-grade CIFAR result has multiple seeds and FID, but strong CIFAR baselines, calibration cost analysis, proof details, and full reproducibility are still missing. |

Mean score: 3.0/5.

## Severity-Ranked Findings

| ID | Severity | Target | Finding | Action |
| --- | --- | --- | --- | --- |
| F01 | Critical | `paper/checklist.tex:3`, `paper/checklist.tex:34` | At review time, the checklist still contained the NeurIPS instruction block and placeholder answers. This was a desk-readiness failure. | Remove the instruction block and complete every answer honestly after the current claim scope is frozen. |
| F02 | Critical | `paper/sections/theory.tex:3`, `paper/sections/theory.tex:16`, `paper/sections/theory.tex:23` | The theory section says it "will cover" the setting, labels the theorem as a "Target theorem," and says the final submission must include proofs. | Add the full assumptions, proof, coordinate-covariance argument, and stability analysis, or demote theorem claims to a future-work target. |
| F03 | Critical | `paper/sections/abstract.tex:7`, `paper/sections/conclusion.tex:6` | The paper still describes itself as a manuscript scaffold and pipeline validation record. | Replace status-report language with final, evidence-backed claims before submission. |
| F04 | Major | `paper/sections/experiments.tex:9`, `paper/sections/experiments.tex:16` | The main experiments section shows only the smoke gate directly; the strongest positive 50k CIFAR evidence is deferred to the appendix. | Move the bounded CIFAR 50k table or figure into the main experiments section if it supports a main claim. |
| F05 | Major | `paper/project_context.md:123`, `goal.md:293` | The controlled CIFAR claim is base-only. Strong CIFAR schedule baselines such as EDM/Karras/offline schedules are missing. | Add baselines or scope the main empirical claim exactly to native/base schedule comparison in the PNDM/CIFAR Euler setting. |
| F06 | Major | `paper/appendix/t2i_sd15_cfg_sweep.tex:25`, `paper/tables/sd15_euler_nfe10_cfg_sweep_pairwise.tex:13` | The matched SD1.5 AYS comparison is negative for D-GPDE on ImageReward and non-positive on CLIPScore. | Treat this as a central limitation and avoid any positive CFG quality claim. |
| F07 | Major | `paper/sections/limitations.tex:9`, `paper/results/cifar10_50k/*oracle_reuse_cost*.csv`, `paper/results/t2i/*oracle_reuse_cost.csv` | Calibration cost is recorded, but the manuscript has no cost-vs-quality table or amortization analysis. | Add a cost table or figure and state when reuse is needed for practical value. |
| F08 | Major | `paper/appendix/failure_analysis.tex:1` | Failure analysis is mostly a plan; it does not yet include concrete rho, metric, adapter, empty-output, or AYS-stronger tables. | Add concrete failure and ablation records from retained outputs. |
| F09 | Major | `paper/appendix/reproducibility.tex:3` | Reproducibility notes cover only the smoke command and do not document the 50k CIFAR run, SD1.5 sweep, aggregation commands, environment, assets, or hardware. | Expand reproducibility notes before answering checklist reproducibility items as yes. |
| F10 | Major | `paper/sections/method.tex:24` | The method section jumps from local defects to the monitor density without defining the risk functional, `\bar{a}`, aggregation choice, or inverse-CDF construction in a self-contained way. | Add a compact algorithm block or equations for local coefficient estimation, aggregation, and schedule construction. |
| F11 | Major | `paper/sections/theory.tex:19`, `method.md:571` | The theorem uses an epsilon-regularized monitor. The spec treats epsilon as a numerical floor, which may change the objective. | State the theorem for the unregularized monitor or define and prove the regularized objective explicitly. |
| F12 | Minor | `method.md:305`, `paper/sections/theory.tex:12` | The method spec allows `q>0`, while the paper theorem assumes `q>1`. | Align the theorem assumption with the spec or justify the stronger condition. |
| F13 | Minor | `paper/internal_reviews/current_gap_audit.md:15` | The earlier gap audit is stale: it says text-to-image metrics and citations are missing, although both now exist. | Mark the old audit as superseded by this review. |
| F14 | Minor | `paper/tables/sd15_euler_nfe10_cfg_sweep_pairwise.tex:3` | The text-to-image table omits confidence intervals that exist in the CSV. | Add intervals to the table or explain that the figure/table reports point estimates only. |

## Claim Evidence Risk Action Matrix

| Claim | Current Evidence | Risk | Action |
| --- | --- | --- | --- |
| D-GPDE calibrates schedules without updating model weights. | Method prose and GOES implementation framing keep model weights fixed. | Low if phrased as schedule calibration only. | Keep this claim; include calibration-cost accounting. |
| D-GPDE estimates distributional oracle-start local defects. | `method.md` and `paper/sections/method.tex` define teacher trajectories and oracle-start residuals. | Medium because teacher accuracy and aggregation details are compressed in the draft. | Make the method section self-contained and state teacher assumptions. |
| Equal-monitor schedules are asymptotically minimax for local distributional risk. | `paper/sections/theory.tex` contains a target theorem. | High because proof, assumptions, covariance, and stability appendix are absent. | Complete the proof package or remove theorem-level language. |
| D-GPDE improves controlled low-NFE deterministic sampling. | Three-seed 50k CIFAR FID improves over base at NFE 10 and 20. | Medium because it covers one model/backend/solver and no strong CIFAR schedule baseline. | State exactly: PNDM/CIFAR-10 Euler, base schedule comparison, NFE 10/20, seeds 0-2. |
| D-GPDE is condition-aware under CFG. | Matched SD1.5 CFG sweep covers CFG 5.0, 7.5, and 10.0 with base, AYS, and D-GPDE. | High because the result is mixed to negative for D-GPDE. | Use as failure analysis, not as a positive CFG quality claim. |
| D-GPDE approaches offline schedules in selected settings. | The SD1.5 run includes AYS, but D-GPDE trails AYS on ImageReward. | High. | Do not claim approach-to-AYS from current evidence. |
| Defect monitor explains schedule allocation. | Schedule metadata exists under `paper/results/*`, but no main diagnostic plot is present. | Medium. | Add schedule and monitor-mass visualizations before using this as interpretability evidence. |
| Calibration overhead is acceptable. | Oracle reuse cost CSVs exist for CIFAR and SD1.5. | High because costs are large and the SD1.5 quality result is negative. | Add a cost-vs-quality or amortization analysis and avoid blanket deployment claims. |
| Negative evidence is retained. | SD1.5 negative sweep is included; project context notes ablation failures. | Medium because failure appendix lacks concrete tables. | Add retained failure tables and summarize them explicitly. |

## Required Next Actions Before Submission Readiness

1. Finish or demote the theory theorem.
2. Move the bounded CIFAR 50k result into the main experiments section.
3. Add strong CIFAR schedule baselines or scope the claim to native/base only.
4. Add calibration cost and amortization analysis.
5. Convert failure analysis from a plan into concrete tables and diagnostics.
6. Expand reproducibility notes for the real CIFAR and SD1.5 runs.
7. Complete the NeurIPS checklist honestly.
8. Run a final claim-to-citation audit after all prose and claims stabilize.

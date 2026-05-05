# Final Claim Audit

Date: 2026-05-05

Scope: current `paper/` manuscript after the proof appendix, main CIFAR table
with native-coordinate linear and fixed Karras/EDM-style baselines, narrow
CIFAR actual reuse validation, retained failed lightweight, medium-budget, and
default-budget CIFAR offline-proxy diagnostics, an authorized project-owned
CIFAR offline baseline, CIFAR and SD1.5 schedule-profile diagnostics, matched
SDXL text-to-image validation, calibration-cost analysis, failure tables,
selected SD1.5 qualitative failure grid, release metadata, and path
sanitization pass. This audit applies the
ARA rigor-review dimensions manually because this repository is not organized
as an ARA package.

Recommendation: major revision, not submission-ready.

## Verification Snapshot

- `paper/neurips_2026.pdf` compiles to 29 pages.
- Strict LaTeX scan reports no undefined citations, undefined references,
  rerun warnings, fatal errors, or overfull boxes.
- `git diff --check -- . ':!goal.md'` passes.
- The AYS-style exporter stage-bundle path has a focused unit test and a light
  smoke export that reloads `_stage_bundles/nfe_010`,
  `_stage_bundles/nfe_020`, and final `nfe_010`.
- Style tripwire scan over `paper/sections` and `paper/appendix` reports no
  stale scaffold or overclaim terms.
- Result scans report no NaN/null/inf CSV rows and no empty files under
  `paper/results`.
- Paper-facing result, appendix, release, and project-context paths contain no
  local workspace absolute path.
- The local draft release archive must be rebuilt after the SDXL artifact
  additions; the earlier 154-file SHA-256 snapshot is stale.

## Supported Claims

| Claim | Current support | Audit decision |
| --- | --- | --- |
| D-GPDE is training-data-free schedule calibration. | Method, introduction, and implementation framing keep model weights fixed. | Supported if stated as schedule calibration only. |
| D-GPDE estimates distributional oracle-start defects. | Method/theory sections and proof appendix define teacher trajectories, risk aggregation, monitor density, and equal-mass construction. | Supported under stated teacher-oracle and deterministic-ODE assumptions. |
| Equal-monitor schedules are asymptotically minimax for local distributional risk. | Section `sec:theory` and Appendix `app:theory-proofs` contain assumptions, reduction, covariance, minimax, and inverse-CDF stability arguments. | Supported as a local asymptotic theorem only. |
| D-GPDE improves low-NFE deterministic sampling. | PNDM/CIFAR-10 Euler, 50k samples per seed, NFE 10 and 20, seeds 0/1/2, with base, native-coordinate linear, fixed Karras/EDM-style, authorized project-owned offline, failed offline-proxy diagnostics, and D-GPDE schedules. | Supported against base, native-linear, and authorized offline for the scoped CIFAR setting. The Karras comparison is mixed: D-GPDE wins at NFE 10 and trails slightly at NFE 20. The failed offline-proxy attempts remain failure evidence only. |
| D-GPDE is condition-aware under CFG. | SD1.5 Euler NFE-10 matched CFG 5/7.5/10 sweep is negative for D-GPDE. SDXL Euler NFE-10 CFG 5/7.5 is condition-dependent: D-GPDE trails AYS at CFG 5.0 and improves mean CLIPScore/ImageReward deltas over AYS at CFG 7.5. | Supported only as boundary evidence, not as a broad quality-improvement claim. |
| D-GPDE approaches offline schedules. | Matched text-to-image AYS comparison is mixed: negative on SD1.5 and SDXL CFG 5.0, positive on SDXL CFG 7.5 mean deltas. | Unsupported as a general positive claim. |
| SD1.5 prompt matching is auditable. | `paper/results/t2i/diffusers_ablation_prompts_manifest.json` records the 50-prompt asset, SHA-256 hash, and source path; `diffusers_ablation_prompts.csv` records prompt indices and text. | Supported for current paper-facing prompt provenance; release license remains pending. |
| Calibration overhead is acceptable. | Cost and amortization tables quantify high overhead; same-backend CIFAR seed-0 schedule reuse has an actual 50k generation check with zero observed FID gap; SD1.5 quality deltas are negative; SDXL uses lighter calibration and lower cost ratios. | Supported only as cost-aware reporting. Deployment-efficiency claims remain unsupported. |
| Defect monitor reallocates schedule steps. | The PNDM/CIFAR NFE-20 and SD1.5 CFG-7.5 schedule-profile figures and CSVs show D-GPDE allocation against retained baselines; D-GPDE schedule JSONs retain equal-monitor mass records. | Supported only as a schedule diagnostic, not as image-quality evidence. |
| Negative evidence is retained. | Failure summary table, offline-proxy table, medium offline-proxy smoke rows, default offline-proxy feasibility row, SD1.5 failure grid, project context, and appendix preserve concrete negative cases. | Supported for current PNDM ablations, failed offline-proxy diagnostics, and SD1.5 sweep; solver-adapter failure records remain incomplete. |

## Severity Findings

| ID | Severity | Finding | Required action |
| --- | --- | --- | --- |
| A01 | Major | The positive quality evidence is one controlled PNDM/CIFAR-10 Euler comparison. The authorized project-owned offline baseline is now included and D-GPDE beats it at NFE 10 and 20. A Karras/EDM-style fixed baseline remains stronger than D-GPDE at NFE 20. | Keep every empirical claim explicitly scoped to this backend, solver, NFE set, and baseline set; do not claim superiority over all fixed or offline schedules. |
| A02 | Major | Calibration reuse now has a narrow same-backend CIFAR check and an SD1.5 CFG-shift reuse check. The SD1.5 reused schedule remains mixed or negative against base and AYS, and the SDXL rows have no reused-schedule quality check. | Keep practical deployment claims out of the main paper unless reused schedules also improve quality. |
| A03 | Major | Existing-asset source provenance now includes a dated AYS/prompt review, PNDM source URLs, and local checkpoint/stat hashes. The AYS numeric schedule policy now omits raw AYS numeric schedule files and source rows from the anonymous package. Author decisions remain open for prompt text redistribution and paper-produced artifact licenses. | Keep checklist asset answers conservative until authors approve the remaining release terms. |
| A04 | Major | `paper/release/` now includes draft metadata, an author decision sheet, a local release file list, and a deterministic local draft archive, but there is no external anonymized archive or author-approved license. | Keep open-access checklist as `No` until an external release package exists. |
| A05 | Minor | The SD1.5 qualitative failure grid embeds selected generated samples in the PDF. The release policy now covers this, but a final safety/license review is still needed before public distribution. | Review selected rendered samples before release. |
| A06 | Minor | The historical rigor review remains intentionally stale in its original findings, although its post-review notes mark resolved items. | Treat this final audit as the current status document. |

## Current Submission Blockers

- Broader actual reuse validation across batches/conditions.
- Final external anonymous release package and author-approved licenses.
- Author-approved release terms for local prompt text and paper-produced
  artifacts.
- Optional additional solver-adapter failure records if the paper discusses
  those adapters beyond scoped limitations.

## Bottom Line

The current manuscript is much more coherent than the first reviewed draft: the
main theorem has proofs, the main positive result is in the experiments section,
cost is quantified, negative text-to-image evidence is explicit, and release
metadata exists. A narrow CIFAR schedule-reuse check now prevents the cost table
from relying only on arithmetic amortization, but it is same-backend evidence
only. The added authorized offline and Karras baselines sharpen the claim:
D-GPDE beats the project-owned offline baseline in this CIFAR setting but is
not uniformly better than strong fixed schedules. The manuscript is still not
submission-ready because the strongest scientific story remains narrow and
several reuse and release obligations are open.

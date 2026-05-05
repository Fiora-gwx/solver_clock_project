# Author Release Decisions Needed

Date: 2026-05-05

This file turns the remaining release blockers into author decisions. It is not
legal advice and does not grant permissions.

## Blocking Decisions

| Decision | Recommended default | Consequence if unresolved |
| --- | --- | --- |
| Paper-authored text, tables, figures, and cleaned CSV license. | Choose one explicit artifact license, such as CC BY 4.0 for paper artifacts. | Keep the NeurIPS checklist open-access answers as `No`; do not publish the archive externally. |
| Paper-authored scripts under `paper/figures/src/`. | Choose an explicit code license, such as Apache-2.0 or MIT. | Include scripts only in a private reviewer package, or omit scripts from public artifacts. |
| Project-generated schedule metadata. | Include D-GPDE, base, native-linear, and Karras metadata after license approval; omit raw AYS and authorized-offline numeric schedule bundles. | Keep only aggregate tables and figures in the release. |
| AYS numeric schedule bundles. | Author decision on 2026-05-05: omit standalone raw AYS numeric bundles and raw AYS schedule-profile source rows; keep AYS as a citation and optional external baseline loader. | Keep AYS metric comparisons in the paper, but release only citations, hashes, derived aggregate rows, and rendered figures. |
| Local prompt text. | Include prompt text only if authors confirm project ownership and license it. | Release prompt asset names, indices, hashes, and metric rows, but omit prompt text. |
| Raw generated text-to-image sample JPEGs. | Omit raw JPEG directories; keep only the selected rendered failure-grid figure. | No raw sample distribution until safety and license review finish. |
| External anonymous hosting. | Choose one host and URL workflow before changing checklist answers: OpenReview supplement, anonymous OSF/Zenodo record, or an anonymized repository. | The local zip remains a draft package, not an open-access release. |
| Existing dependencies. | Do not bundle CIFAR-10 raw data, model weights, PNDM checkpoints, SD1.5/SDXL weights, or metric model weights. | Bundle only if each dependency passes a separate redistribution review. |

## Files To Update After Decisions

- `paper/release/license_status.md`
- `paper/release/anonymized_artifact_manifest.yaml`
- `paper/release/generated_sample_policy.md`
- `paper/release/README.md`
- `paper/checklist.tex`
- `paper/project_context.md`
- `paper/internal_reviews/current_gap_audit.md`

## Decision Output Needed

Authors should provide these release fields before the local package is treated
as an external anonymous artifact:

| Field | Required value |
| --- | --- |
| Paper artifact license | One license for authored text, tables, figures, cleaned CSVs, and selected rendered figures. |
| Paper script license | One license for scripts under `paper/figures/src/` and release builder scripts under `paper/release/`. |
| Schedule metadata policy | Include project-generated non-AYS schedule metadata, include only aggregate CSVs/tables, or exclude schedule metadata. |
| AYS numeric policy | Resolved on 2026-05-05: omit raw AYS numeric schedule files from the anonymous package; keep AYS references and optional external loader paths only. |
| Prompt text policy | Include prompt text under an author-approved license, or release only prompt indices, asset names, and hashes. |
| Generated sample policy | Confirm that the selected failure-grid figure may remain in the paper package, or remove rendered generated samples. |
| External host | OpenReview supplement, anonymous OSF/Zenodo record, anonymized repository, or another approved anonymous URL workflow. |

## Post-Decision Workflow

After authors provide the fields above:

1. Update `license_status.md`, `anonymized_artifact_manifest.yaml`,
   `generated_sample_policy.md`, and `README.md`.
2. Update the NeurIPS checklist only for decisions that are actually resolved.
3. Run `PYTHONDONTWRITEBYTECODE=1 python paper/release/build_draft_release_archive.py`.
4. Run `sha256sum -c paper/release/dist/dgpde_neurips2026_draft_artifacts.sha256`
   and `unzip -t paper/release/dist/dgpde_neurips2026_draft_artifacts.zip`.
5. Record the external archive URL, file count, SHA-256, and remaining release
   limits in `paper/project_context.md` and `paper/internal_reviews/current_gap_audit.md`.

## Current Local Package

The deterministic local archive builder is ready under `paper/release/`. It
creates a local draft artifact zip and SHA-256 sidecar, but the package remains
non-public until authors choose licenses and an external hosting path.

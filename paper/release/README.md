# Release Metadata

This directory records the release-facing metadata for the current paper
artifacts. It is not an open-access code release by itself. The NeurIPS
checklist should continue to mark open access as unavailable until an
anonymized archive or repository is actually prepared and reviewed.

Included files:

- `anonymized_artifact_manifest.yaml`: repository-relative manifest for paper
  artifacts and external dependencies used by the manuscript.
- `build_draft_release_filelist.py`: local script that builds a draft
  repository-relative release allowlist and exclusion list.
- `build_draft_release_archive.py`: local script that rebuilds the draft
  allowlist, creates a deterministic zip under `release/dist/`, and writes a
  SHA-256 sidecar.
- `draft_release_filelist.txt`: generated draft file list for an anonymous
  paper-artifact package, excluding build products and internal notes.
- `draft_release_exclusions.txt`: generated list of excluded files and reasons.
- `generated_sample_policy.md`: policy for generated image samples and prompt
  records.
- `license_status.md`: current license status and unresolved author decisions.
- `asset_license_review_2026-05-05.md`: dated source-provenance review for the
  AYS numeric schedules and local prompt asset.
- `author_release_decisions_needed.md`: concrete author decisions that block an
  external anonymous release and checklist changes.

Release constraints:

- Paper-produced tables, figures, cleaned CSVs, scripts, and TeX files live
  under `paper/`.
- Paper-facing paths should be repository-relative; local absolute paths should
  be removed with `paper/figures/src/sanitize_paper_artifacts.py` before any
  archive is created.
- Build the draft file list with
  `PYTHONDONTWRITEBYTECODE=1 python paper/release/build_draft_release_filelist.py`
  to avoid Python bytecode caches in the paper tree.
- Build a local draft archive with
  `PYTHONDONTWRITEBYTECODE=1 python paper/release/build_draft_release_archive.py`.
  Files under `paper/release/dist/` are generated artifacts and stay outside
  the release allowlist.
- Existing models, checkpoints, datasets, metrics, and published schedules are
  dependencies, not newly released assets, unless a later package explicitly
  includes them under their own terms.
- The anonymous package omits raw AYS numeric schedule files, raw AYS
  schedule-profile source rows, and authorized-offline numeric schedule
  bundles. AYS remains present through citations, hashes, metric rows, rendered
  figures, and optional external loader paths.

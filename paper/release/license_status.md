# License Status

No final release license has been assigned for the paper-produced artifacts in
this workspace. A license decision should be made by the paper authors before
any anonymous archive or public repository is distributed.

Current dependency status:

- CIFAR-10: UCI records CC BY 4.0 for the dataset page; cite the original
  technical report and do not bundle raw data without final review.
- PNDM CIFAR checkpoint: the official PNDM repository is Apache-2.0 and its
  README links the Google Drive folder used for checkpoints and FID statistics.
  The local CIFAR checkpoint and FID-stat hashes are recorded in
  `anonymized_artifact_manifest.yaml`. Do not bundle checkpoint files in the
  anonymous archive without final redistribution review.
- Stable Diffusion 1.5: the cached model card lists CreativeML OpenRAIL-M; do
  not bundle model weights.
- SDXL: the cached model card and upstream terms need final author review; do
  not bundle model weights.
- AYS 10-step numeric schedule: a 2026-05-05 web check found that the official
  AYS quickstart lists the schedule values used here, but the fetched page does
  not state release terms for redistributing the numeric table. Hugging Face
  Diffusers also embeds AYS schedule constants in an Apache-2.0 source file.
  Author decision on 2026-05-05: do not distribute raw AYS numeric schedule
  files in the anonymous package. Keep AYS metric comparisons, citations,
  hashes, rendered figures, and optional external baseline loader paths.
- Diffusers: Apache-2.0.
- ImageReward: Apache-2.0.
- Project prompt set: the paper-facing manifest records the 50-prompt asset and
  raw source hash; `anonymized_artifact_manifest.yaml` also records hashes for
  the raw prompt JSON, paper prompt manifest, and paper prompt CSV. No external
  source is recorded in the current manifest, so this is treated as a
  project-local prompt asset. Redistribution still needs an author license
  decision; otherwise release only prompt asset names and prompt indices.

Recommended author decisions before release:

- choose a license for paper-authored text, tables, figures, and cleaned CSVs;
- choose a license for paper-authored scripts under `paper/figures/src/`;
- decide whether project-generated non-AYS schedule metadata can be bundled;
- decide whether prompt text can be redistributed or should remain referenced
  only by prompt asset name and index.

The dated source audit for these decisions is
`asset_license_review_2026-05-05.md`. The concrete author decision checklist is
`author_release_decisions_needed.md`.

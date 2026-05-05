# Generated Sample Policy

The current paper package includes a selected qualitative failure grid rendered
from retained SD1.5 generated samples. It does not redistribute the raw
SD1.5 or SDXL generated JPEG directories. The package retains aggregate
metrics, matched prompt indices, prompt text, seeds, schedule metadata, and
source-output paths sufficient to audit the reported automated metrics on the
local run outputs.

If generated samples are later included in a supplemental archive, the package
must include:

- model, solver, schedule, NFE, guidance scale, seed, prompt index, and prompt
  text for every image;
- the base-model license and usage restrictions, especially the Stable
  Diffusion 1.5 and SDXL model terms;
- a clear statement that images are synthetic samples, not dataset examples or
  human-preference evidence;
- a safety review for unsafe, personal, copyrighted, or policy-sensitive
  content before public distribution;
- a decision on whether failed or low-quality samples are included as failure
  evidence or omitted for safety/licensing reasons.

Until that review is complete, raw generated JPEGs under `outputs/` should not
be bundled into the paper release beyond the selected rendered figure already
used as qualitative failure evidence.

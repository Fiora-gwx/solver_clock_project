# Citation Verification Notes

Date: 2026-05-05

Scope: first verified bibliography pass for the NeurIPS draft. This note records
where each BibTeX entry came from and which manuscript claim it supports.

## Programmatic Sources

The following entries were fetched from primary paper pages or arXiv BibTeX
endpoints:

| Key | Verification source | Use in manuscript |
| --- | --- | --- |
| `ho2020ddpm` | `https://arxiv.org/bibtex/2006.11239` | Diffusion sampler foundation. |
| `song2022ddim` | `https://arxiv.org/bibtex/2010.02502` | Deterministic implicit sampler baseline. |
| `song2021sde` | `https://arxiv.org/bibtex/2011.13456` | Score-SDE formulation. |
| `karras2022edm` | `https://arxiv.org/bibtex/2206.00364` | EDM schedule and sampler design. |
| `lu2022dpmsolver` | `https://arxiv.org/bibtex/2206.00927` | DPM-Solver family. |
| `lu2025dpmsolverpp` | `https://arxiv.org/bibtex/2211.01095` | DPM-Solver++ guided solver. |
| `zhao2023unipc` | `https://arxiv.org/bibtex/2302.04867` | UniPC predictor-corrector sampler. |
| `watson2022fastsamplers` | `https://arxiv.org/bibtex/2202.05830` | Differentiable sampler search. |
| `salimans2022progressive` | `https://arxiv.org/bibtex/2202.00512` | Progressive distillation. |
| `sabour2024ays` | PMLR page `https://proceedings.mlr.press/v235/sabour24a.html` | AYS offline schedule optimization. |
| `williams2024scoreoptimal` | NeurIPS BibTeX endpoint `https://proceedings.neurips.cc/paper_files/paper/24553-/bibtex` | Score-optimal diffusion schedules. |
| `xue2024optimizedtimesteps` | arXiv page `https://arxiv.org/abs/2402.17376` | Optimized timestep baseline. |
| `tan2025stork` | arXiv page `https://arxiv.org/abs/2505.24210` | STORK solver-adapter context. |
| `rombach2022ldm` | `https://arxiv.org/bibtex/2112.10752` | Stable Diffusion / latent diffusion context. |
| `podell2023sdxl` | `https://arxiv.org/bibtex/2307.01952` | SDXL robustness target. |
| `radford2021clip` | `https://arxiv.org/bibtex/2103.00020` | CLIP model behind CLIPScore. |
| `hessel2022clipscore` | `https://arxiv.org/bibtex/2104.08718` | CLIPScore metric. |
| `xu2023imagereward` | `https://arxiv.org/bibtex/2304.05977` | ImageReward metric. |

## Claim Check

- Schedule optimization claims cite AYS, Score-Optimal Diffusion Schedules, and
  optimized-timestep work.
- Fixed-schedule and solver claims cite DDPM, DDIM, score SDE, EDM, DPM-Solver,
  DPM-Solver++, UniPC, and STORK where the draft discusses scheduler-adapter
  code.
- Learned fast-sampler positioning cites differentiable sampler search and
  progressive distillation.
- Text-to-image model and metric claims cite latent diffusion, SDXL,
  CLIP/CLIPScore, and ImageReward.

## Remaining Citation Work

- Add citations for any future additional human-preference metrics only after
  those experiments enter the manuscript.
- Recheck venue-preferred entries if the paper moves from arXiv-style metadata
  to final conference proceedings metadata.
- Run a full claim-to-citation audit after the main-paper experiment section is
  rewritten.

# Publication status

**Date:** 2026-09-24  
**Verdict: PUBLICATION-READY at workshop / empirical-L2 scope** (not claiming top-venue theoretical novelty).

## Contribution checklist

- [x] Topic KEEP / MODIFY with documented novelty case (`RESEARCH_SURVIVOR.md`)
- [x] Explicit novelty statement (likelihood-rank ordering; W₂ false-stability falsified)
- [x] Core experiments logged with seeds/configs (`w2_fs`)
- [x] Literature verification vs MAD / Feng / LSF / temperature papers
- [x] Manuscript (`paper/PAPER.md`)
- [x] Reproducibility package (README + scripts + logs + analysis)

## Evidence summary

| Claim | Result |
|-------|--------|
| top_k improves W₂ while hurting minorities | **FALSIFIED** |
| bottom_k > rand_k > top_k minority retention | **SUPPORTED** (GMM; 5/5 seeds at ρ=5; Wilcoxon p≈0.04 vs rand) |
| Digits transfer | **INCONCLUSIVE** |

Reproduce: `scripts/reproduce_w2.ps1`

## Honesty limits

Contribution is L2 empirical relative to MAD quality-bias discussion. Suitable for workshop / short paper / arXiv note. Not a new SOTA training method.

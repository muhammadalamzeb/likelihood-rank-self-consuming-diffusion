# Publication status

**Date:** 2026-09-24  
**Verdict: FIRST-AUTHOR SUBMISSION-READY** for arXiv cs.LG / workshop empirical note  
(not claiming top-venue theoretical novelty or ImageNet SOTA).

**Sole / first author:** Muhammad Alamzeb (`paper/SUBMIT.md`)

## Contribution checklist

- [x] Topic KEEP / MODIFY with documented novelty case (`RESEARCH_SURVIVOR.md`)
- [x] Explicit novelty statement (likelihood-rank ordering; W₂ false-stability falsified)
- [x] Core experiments logged with seeds/configs (`w2_fs`)
- [x] Transfer: 8D GMM + Digits-PCA (supported); Digits CVAE inconclusive
- [x] Literature verification vs MAD / Feng / LSF / temperature papers
- [x] Manuscript with author name (`paper/PAPER.md`, `paper/main.tex`)
- [x] Bibliography with full author lists (`paper/refs.bib`)
- [x] Reproducibility package (README + scripts + logs + analysis)
- [x] Submission checklist (`paper/SUBMIT.md`)
- [x] Compiled PDF (`paper/main.pdf`) + arXiv zip (`paper/arxiv_source.zip`)
- [ ] **Live arXiv upload** (requires your arXiv account / endorsement)

## Evidence summary

| Claim | Result |
|-------|--------|
| top_k improves W₂ while hurting minorities | **FALSIFIED** |
| bottom_k > rand_k > top_k minority retention | **SUPPORTED** (2D GMM 5/5 seeds; 8D GMM 3/3; Digits-PCA 3/3; Wilcoxon p≈0.04) |
| Digits CVAE + classifier | **INCONCLUSIVE** (superseded) |
| α / k_frac sensitivity | Scoped: strongest at α≤0.5; see ablation tables |

Reproduce: `scripts/reproduce_w2.ps1`

## Honesty limits

Contribution is L2 empirical relative to MAD quality-bias discussion. Suitable for workshop / short paper / arXiv note. Not a new SOTA training method.

# Publication status

**Date:** 2026-09-25  
**Verdict: FIRST-AUTHOR SUBMISSION-READY** (post review-revision) for arXiv cs.LG / workshop empirical note.

**Sole / first author:** Muhammad Alamzeb (`paper/SUBMIT.md`)

## Evidence summary (honest)

| Claim | Result |
|-------|--------|
| Primary \(n\) | **8** cells: seeds 0–2 × ρ∈{5,10} + seeds 3–4 × ρ=5 only |
| bottom > rand > top retention | **Supported** by means, \(d_z{\approx}1\), CIs, 5/5 order at ρ=5; Wilcoxon \(p{\approx}0.042\), Holm \(p{\approx}0.085\) |
| top improves W₂ while minorities die | **Not observed** (gen-1 \(\Delta W_2>0\) on 8/8; Wilcoxon \(p{\approx}0.014\)) |
| 8D / Digits-PCA | **Directional** (3/3 order; mean±std reported; not powered tests) |
| Digits CVAE | **INCONCLUSIVE** (unused) |

Reproduce: `python scripts/reproduce_w2.py`

## Remaining human steps

1. Host public repo (GitHub/Zenodo) and paste URL into paper appendix.
2. Upload `paper/arxiv_source.zip` to arXiv.

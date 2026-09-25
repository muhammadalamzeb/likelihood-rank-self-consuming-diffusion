# Likelihood-Ranked Synthetic Selection in Self-Consuming Diffusion: Minority Extinction Ordering on Imbalanced Mixtures

**Author:** Muhammad Alamzeb (sole / first author)  
**Contact:** shayankhanmahar@gmail.com  
**Venue:** arXiv cs.LG (named author OK); anonymize + anonymous code mirror for double-blind  
**Code:** https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion  
**Status:** Submission-ready (`paper/SUBMIT.md`)

## Abstract

Self-consuming training loops can induce *model collapse*, including loss of rare modes. We study unlabeled selection of synthetic samples by a denoising likelihood proxy in self-consuming DDPM training on imbalanced mixtures. Define \(r=m^{(1)}/m^{(0)}\). On eight seed–imbalance cells, results are **stratified by** \(\rho\) (not a pooled test as the headline). At \(\rho{=}5\) (\(n{=}5\)), generation-1 mass obeys bottom>rand>top on all seeds; paired contrasts vs random-\(k\) have large effects (top−rand: \(d_z{=}{-}1.96\), bootstrap 95% CI \([{-}0.37,{-}0.15]\); bottom−rand: \(d_z{=}{+}1.65\), CI \([0.16,0.43]\)). Exact Wilcoxon \(p{=}0.0625\) for both (minimum two-sided exact \(p\) at \(n{=}5\) when all signs agree). At \(\rho{=}10\) (\(n{=}3\)), full order holds \(0/3\) and contrasts are near null. Top-\(k\) does not improve sliced \(W_2\) at \(g{=}1\) (\(\Delta W_2>0\) on \(8/8\); exact \(p{=}0.0078\); Holm across five gens \(p{=}0.039\)). Directional transfers (8D GMM, Digits-PCA; 3 seeds) match the \(\rho{=}5\) order.

## 1 Introduction

Self-consuming training can cause model collapse [Shumailov et al., 2023; Alemohammad et al., 2023; Gerstgrasser et al., 2024]. Mitigations include accumulating real data [Gerstgrasser et al., 2024] and filtering [Feng et al., 2024; Cai et al., 2025]. We isolate an interventional top-\(k\) / random-\(k\) / bottom-\(k\) contrast under matched budgets with mode-exact metrics.

**What is new.** Controlled score-rank intervention—not a new SOTA training algorithm.

## 2 Related work

MAD / collapse [Shumailov et al., 2023; Alemohammad et al., 2023]; accumulation [Gerstgrasser et al., 2024]; verification [Feng et al., 2024]; LSF [Cai et al., 2025]; temperature sampling [Xu et al., 2025; Li et al., 2026].

## 3 Method

**Policies.** top/bottom/rand-\(k\): size-\(k\) pool then mix with \(\alpha\) real. **mix**: no rank filter (≠ rand-\(k\)). **replace**: pure synth.

**Metrics.** \(r=m^{(1)}/m^{(0)}\) (may exceed 1). Sliced \(W_2\): 24 projections, ≤400/400 points. Tests: **exact** Wilcoxon signed-rank (sign enumeration; midranks for ties).

## 4 Experiments

### 4.1 Design

| \(\rho\setminus\) seed | 0 | 1 | 2 | 3 | 4 |
|--|--|--|--|--|--|
| 5 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | ✓ | ✓ | ✓ | — | — |

### 4.2 Retention — stratified primary analysis

| \(\rho\) | \(n\) | bottom | rand | top | Full order | top<rand / bot>rand |
|--|--|--|--|--|--|--|
| 5 | 5 | 0.737 | 0.456 | 0.183 | **5/5** | 5/5 / 5/5 |
| 10 | 3 | 0.296 | 0.199 | 0.154 | **0/3** | 2/3 / 1/3 |

| \(\rho\) | Contrast | Mean \(\Delta\) | \(d_z\) | 95% CI | Exact \(p\) |
|--|--|--|--|--|--|
| 5 | top−rand | −0.273 | −1.96 | [−0.37, −0.15] | 0.0625 |
| 5 | bottom−rand | +0.281 | +1.65 | [0.16, 0.43] | 0.0625 |
| 10 | top−rand | −0.045 | −0.28 | [−0.19, 0.13] | 0.75 |
| 10 | bottom−rand | +0.097 | +0.38 | [−0.08, 0.39] | 1.00 |

Pooled \(n{=}8\) exact \(p{=}0.039\) is **secondary** (mixes heterogeneous strata). At \(\rho{=}5\), \(p{=}0.0625\) is the discrete minimum when all signs agree—not evidence of a null effect given \(|d_z|>1.6\).

![Retention bars](figures/retention_bars.png)

![Minority vs generation](figures/minority_vs_generation_rho5.png)

### 4.3 Sliced \(W_2\)

| \(g\) | Mean \(\Delta\) | Std | \(d_z\) | 95% CI | \(\#>0\) | Exact \(p\) (Holm) |
|--|--|--|--|--|--|--|
| 1 | +0.113 | 0.099 | 1.14 | [0.05, 0.18] | 8/8 | 0.0078 (0.039) |
| 2 | +0.061 | 0.060 | 1.00 | [0.02, 0.10] | 6/8 | 0.039 (0.094) |
| 3 | +0.039 | 0.035 | 1.11 | [0.02, 0.06] | 7/8 | 0.023 (0.094) |
| 4 | +0.027 | 0.032 | 0.83 | [0.01, 0.05] | 7/8 | 0.023 (0.094) |
| 5 | +0.024 | 0.033 | 0.73 | [0.01, 0.05] | 7/8 | 0.023 (0.094) |

Holm notes serial dependence across generations (conservative).

### 4.4 α ablation (illustrative; \(n{=}2\))

| \(\alpha\) | bottom | rand | top | Order holds |
|--|--|--|--|--|
| 0.25 | 0.25 | 0.17 | 0.05 | 1/2 |
| 0.50 | 0.62 | 0.49 | 0.24 | 2/2 |
| 0.75 | 1.34 | 0.77 | 0.94 | 0/2 |

### 4.5 Transfers (directional; \(n{=}3\))

8D: 0.55±0.20 / 0.27±0.06 / 0.08±0.03 (3/3). Digits-PCA: 1.49±0.11 / 1.11±0.08 / 0.68±0.17 (3/3).

### 4.6 \(k_{\mathrm{frac}}\) (illustrative; \(n{=}2\))

| \(k_{\mathrm{frac}}\) | bottom | rand | top | Order holds |
|--|--|--|--|--|
| 0.25 | 0.62 | 0.49 | 0.24 | 1/2 |
| 0.50 | 0.62 | 0.49 | 0.24 | 1/2 |
| 0.75 | 0.48 | 0.61 | 0.22 | 0/2 |

0.25≡0.5 when \(n_{\mathrm{syn}}\) binds; top remains worst on means; at 0.75 bottom vs random flips.

## 5 Analysis

Top-\(k\) retains high-likelihood (majority) synthetics. Order is clear at \(\rho{=}5\) and near noise at \(\rho{=}10\). \(W_2\) poorly tracks minority survival.

## 6 Limitations

Toy scale; \(\rho{=}5\) exact \(p\) floored at 0.0625 by \(n{=}5\); alpha / \(k_{\mathrm{frac}}\) illustrative (\(n{=}2\)); transfers unpowered; L2 empirical vs MAD/LSF/Feng.

**Note on double-blind review.** Named GitHub reveals identity—use an anonymous mirror for DB venues; fine for arXiv.

## 7 Conclusion

Stable minority-retention order at moderate imbalance (\(\rho{=}5\)); not seed-wise reliable at \(\rho{=}10\). Top-\(k\) does not improve sliced \(W_2\) at \(g{=}1\). Preferring “most likely” synthetics is not a free lunch when rare modes remain measurable.

## References

- Alemohammad et al. (2023). Self-Consuming Generative Models Go MAD. arXiv:2307.01850.
- Cai et al. (2025). Latent Space Filtering. arXiv:2511.12742.
- Feng et al. (2024). Beyond Model Collapse. arXiv:2406.07515.
- Gerstgrasser et al. (2024). Is Model Collapse Inevitable? arXiv:2404.01413.
- Li et al. (2026). Temperature Sampling and Variance-Corrective Time Shifting. arXiv:2607.10853.
- Shumailov et al. (2023). The Curse of Recursion. arXiv:2305.17493.
- Xu et al. (2025). Temporal Score Rescaling. arXiv:2510.01184.

## Appendix A Reproducibility

```bash
python scripts/reproduce_w2.py
python experiments/scripts/make_submission_stats.py
```

Code: https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion

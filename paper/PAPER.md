# Likelihood-Ranked Synthetic Selection in Self-Consuming Diffusion: Minority Extinction Ordering at Moderate Imbalance

**Author:** Muhammad Alamzeb  
**Contact:** shayankhanmahar@gmail.com  
**DOI:** https://doi.org/10.5281/zenodo.22956890  
**Code:** https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion

## Abstract

Self-consuming training loops can induce *model collapse*. We study unlabeled likelihood ranking of synthetics in self-consuming DDPM training, emphasizing **moderate** imbalance. Define \(r=m^{(1)}/m^{(0)}\). Results are **stratified by** \(\rho\). At \(\rho{=}5\) (\(n{=}5\)), bottom>rand>top on all seeds with large paired effects vs random-\(k\) (\(d_z{=}{-}1.96\) / \({+}1.65\)); we treat this as **directional** because exact Wilcoxon \(p{=}0.0625\) is the discrete minimum at \(n{=}5\) and cannot reach \(p{<}0.05\). At \(\rho{=}10\) (\(n{=}3\)), order \(0/3\), contrasts near null (minority-mass floor). Strongest powered claim: top-\(k\) does **not** improve sliced \(W_2\) at \(g{=}1\) (\(8/8\) positive; exact \(p{=}0.0078\); Holm \(p{=}0.039\)). Transfers (8D, Digits-PCA; \(n{=}3\)) match \(\rho{=}5\); Digits-CVAE was inconclusive. Code: https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion.

## 1 Introduction

Self-consuming training can cause model collapse [Shumailov et al., 2023; Alemohammad et al., 2023; Gerstgrasser et al., 2024]. Mitigations include accumulating real data [Gerstgrasser et al., 2024] and filtering [Feng et al., 2024; Cai et al., 2025]. We isolate an interventional top-\(k\) / random-\(k\) / bottom-\(k\) contrast under matched budgets with mode-exact metrics.

**What is new.** Controlled score-rank intervention plus falsification of “top-\(k\) looks better on \(W_2\)”—not a new SOTA training algorithm.

**Contributions.** (1) Matched-budget DDPM protocol with exact mode masses. (2) Directional likelihood-rank ordering of minority retention at \(\rho{=}5\), stratified by \(\rho\). (3) Powered evidence that top-\(k\) does not improve sliced \(W_2\) at \(g{=}1\). (4) Directional 8D / Digits-PCA checks (Digits-CVAE inconclusive).

## 2 Related work

MAD / collapse [Shumailov et al., 2023; Alemohammad et al., 2023]; accumulation [Gerstgrasser et al., 2024]; verification [Feng et al., 2024]; LSF [Cai et al., 2025]; siloed / biased-reference selection accelerating collapse [Qiao et al., 2026]; temperature sampling [Xu et al., 2025; Li et al., 2026].

**Contrast to Qiao et al. (2026).** They study selection against a *biased local real-data* verifier in data silos and propose collaborative Wasserstein proxy references. We study ranking by the *generator’s own* denoising-likelihood proxy, with a matched-budget top/rand/bottom contrast and minority-mass retention as the primary diversity metric, plus an explicit \(W_2\) “false stability” falsification.

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

Pooled \(n{=}8\) exact \(p{=}0.039\) is **secondary** (mixes heterogeneous strata). At \(\rho{=}5\), \(p{=}0.0625\) is the discrete minimum when all signs agree—the test cannot reach \(p{<}0.05\); we report the ordering as **directional** (unanimous signs, \(|d_z|>1.6\), CIs excluding zero). At \(\rho{=}10\), minorities are near floor at \(g{=}0\), so rankable minority synthetics are scarce and contrasts are near null (floor / saturation hypothesis).

### 4.3 Sliced \(W_2\) (primary powered claim)

| \(g\) | Mean \(\Delta\) | Std | \(d_z\) | 95% CI | \(\#>0\) | Exact \(p\) (Holm) |
|--|--|--|--|--|--|--|
| 1 | +0.113 | 0.099 | 1.14 | [0.05, 0.18] | 8/8 | 0.0078 (0.039) |
| 2 | +0.061 | 0.060 | 1.00 | [0.02, 0.10] | 6/8 | 0.039 (0.094) |
| 3 | +0.039 | 0.035 | 1.11 | [0.02, 0.06] | 7/8 | 0.023 (0.094) |
| 4 | +0.027 | 0.032 | 0.83 | [0.01, 0.05] | 7/8 | 0.023 (0.094) |
| 5 | +0.024 | 0.033 | 0.73 | [0.01, 0.05] | 7/8 | 0.023 (0.094) |

Holm notes serial dependence across generations (conservative). Evidence against top-\(k\) improving \(W_2\) is strongest at \(g{=}1\).

### 4.4 α ablation (illustrative; \(n{=}2\))

| \(\alpha\) | bottom | rand | top | Order holds |
|--|--|--|--|--|
| 0.25 | 0.25 | 0.17 | 0.05 | 1/2 |
| 0.50 | 0.62 | 0.49 | 0.24 | **1/2** |
| 0.75 | 1.34 | 0.77 | 0.94 | 0/2 |

The \(\alpha{=}0.5\) row is the **same cell** as \(k_{\mathrm{frac}}{=}0.5\) below (identical means). Seed 0 breaks full order (bottom \(m^{(1)}\) slightly < rand); seed 1 holds.

### 4.5 Transfers (directional; \(n{=}3\))

Min exact Wilcoxon \(p{=}0.25\) when all signs agree—unpowered for \(\alpha{=}0.05\). 8D: order 3/3; top−rand \(d_z{=}{-}2.56\), bottom−rand \(d_z{=}{+}1.28\). Digits-PCA: 3/3; \(d_z{=}{-}1.72\) / \(+2.67\). Digits-CVAE: **inconclusive** (no stable bottom>rand>top under the same budget protocol); Digits-PCA is the supported image-adjacent check.

### 4.6 \(k_{\mathrm{frac}}\) (illustrative; \(n{=}2\))

| \(k_{\mathrm{frac}}\) | bottom | rand | top | Order holds |
|--|--|--|--|--|
| 0.25 | 0.62 | 0.49 | 0.24 | 1/2 |
| 0.50 | 0.62 | 0.49 | 0.24 | 1/2 |
| 0.75 | 0.48 | 0.61 | 0.22 | 0/2 |

0.25≡0.5 when \(n_{\mathrm{syn}}\) binds; \(k_{\mathrm{frac}}{=}0.5\) duplicates α=0.5.

## 5 Analysis

Top-\(k\) retains high-likelihood (majority) synthetics. Order is directionally clear at \(\rho{=}5\) and near noise at \(\rho{=}10\) (floor on rankable minority mass). \(W_2\) poorly tracks minority survival.

## 6 Limitations

Toy scale; \(\rho{=}5\) retention is directional (\(p\) floored at 0.0625 by \(n{=}5\)); powered claim is \(W_2\) at \(g{=}1\); alpha / \(k_{\mathrm{frac}}\) illustrative (\(n{=}2\)); transfers unpowered; Digits-CVAE inconclusive.

**Exploratory search and \(p$-values.** The reported claim is the survivor of an earlier exploratory search over several small stacks (archived under `docs/` and `experiments/` in the public repo), not a single pre-registered test. Retention \(p\)-values should be read as descriptive under that history; we mitigate overclaiming by stratifying on \(\rho\), demoting pooled tests, labeling \(\rho{=}5\) retention as directional, and treating the \(W_2\) \(g{=}1\) result as the primary powered claim.

**Note on double-blind review.** Named GitHub reveals identity—use an anonymous mirror for DB venues; fine for arXiv.

## 7 Conclusion

Directional minority-retention order at moderate imbalance (\(\rho{=}5\)); not seed-wise reliable at \(\rho{=}10\) (floor). Top-\(k\) does not improve sliced \(W_2\) at \(g{=}1\). Preferring “most likely” synthetics is not a free lunch when rare modes remain measurable.

## References

- Alemohammad et al. (2023). Self-Consuming Generative Models Go MAD. arXiv:2307.01850.
- Cai et al. (2025). Latent Space Filtering. arXiv:2511.12742.
- Feng et al. (2024). Beyond Model Collapse. arXiv:2406.07515.
- Gerstgrasser et al. (2024). Is Model Collapse Inevitable? arXiv:2404.01413.
- Li et al. (2026). Temperature Sampling and Variance-Corrective Time Shifting. arXiv:2607.10853.
- Qiao et al. (2026). When Sample Selection Bias Precipitates Model Collapse. arXiv:2606.13732 (ICML 2026).
- Shumailov et al. (2023). The Curse of Recursion. arXiv:2305.17493.
- Xu et al. (2025). Temporal Score Rescaling. arXiv:2510.01184.

## Appendix A Reproducibility

```bash
python scripts/reproduce_w2.py
python experiments/scripts/make_submission_stats.py
```

Code: https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion

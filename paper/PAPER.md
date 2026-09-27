# Likelihood-Ranked Synthetic Selection in Self-Consuming Diffusion: An Imbalance-Dependent Boundary for Minority Retention

**Author:** Muhammad Alamzeb  
**Contact:** shayankhanmahar@gmail.com  
**DOI:** https://doi.org/10.5281/zenodo.22956890  
**Code:** https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion

## Abstract

Self-consuming training loops can induce *model collapse*. We ask when unlabeled likelihood ranking of synthetics harms minority modes in self-consuming DDPM training on imbalanced mixtures. The answer is **imbalance-dependent**: a moderate-imbalance boundary for the full bottom>rand>top ordering, not a universal law. Define \(r=m^{(1)}/m^{(0)}\). At \(\rho{=}5\) (\(n{=}5\)), order holds 5/5; bootstrap 95% CIs for mean paired \(\Delta\) vs random-\(k\) exclude zero (top−rand \([{-}0.37,{-}0.15]\); bottom−rand \([0.16,0.43]\)), while exact Wilcoxon \(p{=}0.0625\) is the discrete floor at \(n{=}5\) (directional). At \(\rho{=}7\): full order \(2/5\); at \(\rho{=}10\) (\(n{=}8\)): \(3/8\). Top−rand CIs still exclude zero at \(\rho{=}7\) and \(10\); bottom−rand does not at \(\rho{=}10\). Strongest powered claim: top-\(k\) does **not** improve sliced \(W_2\) at \(g{=}1\) (18/18 positive; CI excludes zero; exact \(p{\approx}7.6{\times}10^{-6}\); Holm \(p{\approx}3.8{\times}10^{-5}\)). Transfers: 8D GMM, Digits-PCA; Digits-CVAE inconclusive; Digits pixel-space DDPM as a small image check. Code: https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion.

## 1 Introduction

Self-consuming training can cause model collapse [Shumailov et al., 2023; Alemohammad et al., 2023; Gerstgrasser et al., 2024]. Mitigations include accumulating real data [Gerstgrasser et al., 2024] and filtering [Feng et al., 2024; Cai et al., 2025]. We isolate an interventional top-\(k\) / random-\(k\) / bottom-\(k\) contrast under matched budgets with mode-exact metrics.

**What is new.** An *imbalance-dependent boundary* for minority retention under likelihood ranking—not a universal ordering law—plus falsification of “top-\(k\) looks better on \(W_2\)”.

**Contributions.** (1) Matched-budget DDPM protocol with exact mode masses. (2) Stratified boundary: full order at \(\rho{=}5\), breakdown at \(\rho{=}7\) and \(\rho{=}10\). (3) Powered \(W_2\) negative result for top-\(k\) at \(g{=}1\). (4) Directional transfers including a Digits pixel-space DDPM check.

## 2 Related work

MAD / collapse [Shumailov et al., 2023; Alemohammad et al., 2023]; accumulation [Gerstgrasser et al., 2024]; verification [Feng et al., 2024]; LSF [Cai et al., 2025]; siloed / biased-reference selection [Qiao et al., 2026]; temperature sampling [Xu et al., 2025; Li et al., 2026].

**Contrast to Qiao et al. (2026).** They study selection against a *biased local real-data* verifier in data silos. We study ranking by the *generator’s own* denoising-likelihood proxy, with matched-budget top/rand/bottom contrasts, minority-mass retention, and an explicit \(W_2\) falsification.

## 3 Method

**Policies.** top/bottom/rand-\(k\): size-\(k\) pool then mix with \(\alpha\) real. **mix**: no rank filter (≠ rand-\(k\)). **replace**: pure synth.

**Metrics.** \(r=m^{(1)}/m^{(0)}\) (may exceed 1). Sliced \(W_2\): 24 projections, ≤400/400 points. Inference: bootstrap 95% CIs of mean paired \(\Delta\) + **exact** Wilcoxon (sign enumeration).

## 4 Experiments

### 4.1 Design (\(n{=}18\) cells)

| \(\rho\setminus\) seed | 0–4 | 5–7 |
|--|--|--|
| 5 | ✓ | — |
| 7 | ✓ | — |
| 10 | ✓ | ✓ |

### 4.2 Retention — stratified (lead with CIs)

| \(\rho\) | \(n\) | bottom | rand | top | Full order | top<rand / bot>rand |
|--|--|--|--|--|--|--|
| 5 | 5 | 0.737 | 0.456 | 0.183 | **5/5** | 5/5 / 5/5 |
| 7 | 5 | 0.437 | 0.266 | 0.145 | **2/5** | 4/5 / 3/5 |
| 10 | 8 | 0.329 | 0.247 | 0.164 | **3/8** | 7/8 / 4/8 |

| \(\rho\) | Contrast | Mean \(\Delta\) | \(d_z\) | 95% CI | Exact \(p\) |
|--|--|--|--|--|--|
| 5 | top−rand | −0.273 | −1.96 | [−0.37, −0.15] | 0.0625 |
| 5 | bottom−rand | +0.281 | +1.65 | [0.16, 0.43] | 0.0625 |
| 7 | top−rand | −0.121 | −1.09 | [−0.21, −0.04] | 0.125 |
| 7 | bottom−rand | +0.171 | +0.92 | [0.03, 0.32] | 0.250 |
| 10 | top−rand | −0.083 | −0.79 | [−0.14, −0.01] | 0.078 |
| 10 | bottom−rand | +0.082 | +0.44 | [−0.03, 0.21] | 0.578 |

\(\rho{=}5\) full order is **directional** (\(p\) floored at 0.0625). Full order breaks down at \(\rho{=}7\) and \(\rho{=}10\). Top−rand CIs exclude zero at all three \(\rho\); bottom−rand CI includes zero at \(\rho{=}10\).

### 4.3 Sliced \(W_2\) (primary powered claim; \(n{=}18\))

| \(g\) | Mean \(\Delta\) | \(d_z\) | 95% CI | \(\#>0\) | Exact \(p\) (Holm) |
|--|--|--|--|--|--|
| 1 | +0.075 | 1.00 | [0.044, 0.112] | 18/18 | \(7.6{\times}10^{-6}\) (\(3.8{\times}10^{-5}\)) |
| 2 | +0.038 | 0.82 | [0.019, 0.061] | 16/18 | 0.0007 (0.0027) |
| 3 | +0.028 | 0.90 | [0.014, 0.042] | 16/18 | 0.0013 (0.0039) |
| 4 | +0.015 | 0.47 | [0.000, 0.030] | 14/18 | 0.010 (0.010) |
| 5 | +0.016 | 0.65 | [0.007, 0.029] | 15/18 | 0.0019 (0.0039) |

### 4.4 α / \(k_{\mathrm{frac}}\) (illustrative; \(n{=}2\))

Unchanged from prior tables; default rows share one experimental cell.

### 4.5 Transfers (directional; \(n{=}3\))

Min exact Wilcoxon \(p{=}0.25\) when all signs agree. 8D GMM and Digits-PCA: order 3/3. Digits pixel-space DDPM (\(8{\times}8\)): full order **1/3** (bottom>rand on 3/3; top vs rand inconsistent)—mixed image check, not a validation of the full ordering. Digits-CVAE: inconclusive.

## 5 Analysis

Full bottom>rand>top is a moderate-imbalance phenomenon. Top remains worse than random on average at higher \(\rho\), but bottom’s advantage fades as minority mass floors. \(W_2\) poorly tracks minority survival.

## 6 Limitations

Toy / Digits scale; workshop-scope empirical note. Retention at \(\rho{=}5\) directional; lead with CIs. Exploratory-search history disclosed (public `docs/`). Digits-CVAE inconclusive.

## 7 Conclusion

Imbalance-dependent minority retention under likelihood-ranked selection: directional full order at \(\rho{=}5\), unstable at \(\rho{=}7\), unreliable at \(\rho{=}10\). Top-\(k\) does not improve sliced \(W_2\) at \(g{=}1\). Not a universal law.

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
python experiments/scripts/run_w2_fs_digits_ddpm.py
```

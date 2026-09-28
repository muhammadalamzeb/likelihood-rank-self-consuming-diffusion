# Likelihood-Ranked Selection in Self-Consuming Diffusion: A \(W_2\) Negative Result and a Narrow Minority-Retention Regime

**Author:** Muhammad Alamzeb  
**Contact:** shayankhanmahar@gmail.com  
**DOI:** https://doi.org/10.5281/zenodo.22956890  
**Code:** https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion

## Abstract

Self-consuming training loops can induce *model collapse*. We study unlabeled likelihood ranking of synthetics in self-consuming DDPM training on imbalanced mixtures.

**Primary claim:** top-\(k\) does **not** improve sliced \(W_2\) vs random-\(k\) at generation 1 (\(\Delta W_2>0\) on \(23/23\) cells; bootstrap 95% CI excludes zero; Wilcoxon \(p{=}2.9{\times}10^{-5}\), Holm \(p{=}1.4{\times}10^{-4}\)).

**Secondary, post-hoc / narrow-regime observation:** after an original “false stability” hypothesis was falsified, full bottom>rand>top minority retention appears only at \(\rho{=}5\) (\(n{=}10\), \(10/10\); exact \(p{=}0.002\)), failing for most seeds at \(\rho{=}7\) (\(2/5\)) and \(\rho{=}10\) (\(3/8\)). Treat as exploratory (HARKing risk), not a pre-registered law.

Workshop-scale note—**not** ImageNet-scale diffusion.

## 1 Introduction

Self-consuming training can cause model collapse [Shumailov et al., 2023; Alemohammad et al., 2023; Gerstgrasser et al., 2024]. Mitigations include accumulating real data [Gerstgrasser et al., 2024] and filtering [Feng et al., 2024; Cai et al., 2025]. We isolate an interventional top-\(k\) / random-\(k\) / bottom-\(k\) contrast under matched budgets.

**What is new.** Powered falsification of “top-\(k\) looks better on \(W_2\),” plus a *narrow-regime* retention observation—not a universal ordering law or image-scale result.

## 2 Related work

MAD / collapse [Shumailov et al., 2023; Alemohammad et al., 2023]; accumulation [Gerstgrasser et al., 2024]; verification [Feng et al., 2024]; LSF [Cai et al., 2025]; siloed selection [Qiao et al., 2026]; temperature sampling [Xu et al., 2025; Li et al., 2026].

**Contrast to Qiao et al. (2026).** Biased *local real-data* verifier vs ranking by the *generator’s own* likelihood proxy; we add matched top/rand/bottom contrasts, minority-mass retention, and an explicit \(W_2\) falsification.

## 3 Method

Policies: top/bottom/rand-\(k\) then mix with \(\alpha\) real. Metrics: \(r=m^{(1)}/m^{(0)}\); sliced \(W_2\) (24 projections). Inference: bootstrap 95% CIs + Wilcoxon (exact for \(n{\le}16\)).

## 4 Experiments

### 4.1 Design (\(n{=}23\) cells)

\(\rho{=}5\) seeds \(0\)–\(9\); \(\rho{=}7\) seeds \(0\)–\(4\); \(\rho{=}10\) seeds \(0\)–\(7\).

### 4.2 Sliced \(W_2\) (primary)

| \(g\) | Mean \(\Delta\) | \(d_z\) | 95% CI | \(\#>0\) | \(p\) (Holm) |
|--|--|--|--|--|--|
| 1 | +0.089 | 1.16 | [0.060, 0.121] | 23/23 | \(2.9{\times}10^{-5}\) (\(1.4{\times}10^{-4}\)) |
| 2 | +0.046 | 1.04 | [0.029, 0.065] | 21/23 | 0.00015 (0.00061) |
| 3 | +0.034 | 1.14 | [0.022, 0.045] | 21/23 | 0.00022 (0.00066) |
| 4 | +0.021 | 0.62 | [0.007, 0.034] | 19/23 | 0.0013 (0.0027) |
| 5 | +0.020 | 0.73 | [0.010, 0.031] | 19/23 | 0.0015 (0.0027) |

### 4.3 Retention (secondary; narrow regime)

| \(\rho\) | \(n\) | bottom | rand | top | Full order |
|--|--|--|--|--|--|
| 5 | 10 | 0.651 | 0.366 | 0.141 | **10/10** |
| 7 | 5 | 0.437 | 0.266 | 0.145 | **2/5** |
| 10 | 8 | 0.329 | 0.247 | 0.164 | **3/8** |

At \(\rho{=}5\): top−rand mean \(\Delta{=}{-}0.225\), CI \([{-}0.30,{-}0.15]\), exact \(p{=}0.002\); bottom−rand \({+}0.285\), CI \([0.21,0.37]\), exact \(p{=}0.002\) (Holm \(0.0039\) within the pair).

### 4.4 Transfers

8D GMM / Digits-PCA: order 3/3 (directional, \(n{=}3\)). Digits pixel DDPM: full order 1/3 (mixed). Digits-CVAE: inconclusive. CIFAR-10 tiny-UNet (\(\rho{=}5\), \(n{=}3\)): mean \(\bar r\) bottom \(>\) rand \(>\) top, but seed-wise full order \(0/3\) (preliminary/negative). **No ImageNet-scale diffusion result.**

## 5 Limitations

**Scale.** Toy GMMs / Digits / CIFAR tiny-UNet only; CIFAR is appendix preliminary; no ImageNet-scale diffusion result.

**Researcher degrees of freedom / post-hoc revision.** Long exploratory search across many topics precedes this note (`docs/`). Within this topic, the original hypothesis was top-\(k\) *improves* \(W_2\) while secretly killing minorities (“false stability”); the pilot **falsified** that dissociation, and the claim was revised to the present stratified retention story plus an explicit \(W_2\) negative. Treat bottom>rand>top as **post-hoc / exploratory** even at \(n{=}10\), \(p{=}0.002\). The \(W_2\) negative is more trustworthy: it killed the original hypothesis and held on \(n{=}23\) cells (23/23 at \(g{=}1\)).

## 6 Conclusion

Top-\(k\) does not improve sliced \(W_2\) at \(g{=}1\). Minority retention ordering appears only in a narrow moderate-imbalance regime—not a general law.

## References

- Alemohammad et al. (2023). arXiv:2307.01850.
- Cai et al. (2025). arXiv:2511.12742.
- Feng et al. (2024). arXiv:2406.07515.
- Gerstgrasser et al. (2024). arXiv:2404.01413.
- Li et al. (2026). arXiv:2607.10853.
- Qiao et al. (2026). arXiv:2606.13732.
- Shumailov et al. (2023). arXiv:2305.17493.
- Xu et al. (2025). arXiv:2510.01184.

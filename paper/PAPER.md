# Likelihood-Ranked Selection in Self-Consuming Diffusion: A \(W_2\) Negative Result and a Narrow Minority-Retention Regime

**Author:** Muhammad Alamzeb  
**Contact:** shayankhanmahar@gmail.com  
**DOI:** https://doi.org/10.5281/zenodo.22956890  
**Code:** https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion  
**Release:** [v1.7.0](https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion/releases/tag/v1.7.0)

## Abstract

Self-consuming training loops can induce *model collapse*. We study unlabeled likelihood ranking of synthetics in self-consuming DDPM training on imbalanced mixtures.

**Primary claim:** top-\(k\) does **not** improve sliced \(W_2\) vs random-\(k\) at generation 1 (\(\Delta W_2>0\) on \(38/38\) cells; bootstrap 95% CI excludes zero).

**Secondary, post-hoc / narrow-regime observation:** after an original “false stability” hypothesis was falsified, full bottom>rand>top minority retention is strongest at \(\rho{=}5\) (exploratory \(n{=}10\), \(10/10\), exact \(p{=}0.002\); held-out seeds \(10\)–\(14\): \(4/5\)). Weaker at \(\rho{=}7\) (\(6/10\)) and \(\rho{=}10\) (\(4/13\)). Treat as exploratory (HARKing risk), not a pre-registered law.

Workshop-scale note—**not** ImageNet-scale diffusion.

## 1 Introduction

Self-consuming training can cause model collapse [Shumailov et al., 2023; Alemohammad et al., 2023; Gerstgrasser et al., 2024]. Mitigations include accumulating real data [Gerstgrasser et al., 2024] and filtering [Feng et al., 2024; Cai et al., 2025]. We isolate an interventional top-\(k\) / random-\(k\) / bottom-\(k\) contrast under matched budgets.

**What is new.** Powered falsification of “top-\(k\) looks better on \(W_2\),” plus a *narrow-regime* retention observation—not a universal ordering law or image-scale result.

## 2 Related work

MAD / collapse [Shumailov et al., 2023; Alemohammad et al., 2023]; accumulation [Gerstgrasser et al., 2024]; verification [Feng et al., 2024]; LSF [Cai et al., 2025]; siloed selection [Qiao et al., 2026]; temperature sampling [Xu et al., 2025; Li et al., 2026].

**Contrast to Qiao et al. (2026).** Biased *local real-data* verifier vs ranking by the *generator’s own* likelihood proxy; we add matched top/rand/bottom contrasts, minority-mass retention, and an explicit \(W_2\) falsification.

## 3 Method

Policies: top/bottom/rand-\(k\) then mix with \(\alpha\) real; also unranked **mix** (matched \(\alpha\), no rank filter). Metrics: \(r=m^{(1)}/m^{(0)}\); sliced \(W_2\) (24 projections). Inference: bootstrap 95% CIs + Wilcoxon (exact for \(n{\le}16\)).

## 4 Experiments

### 4.1 Design (\(n{=}38\) cells)

\(K{=}4\), \(G{=}5\), \(\alpha{=}0.5\), \(k_{\mathrm{frac}}{=}0.5\). Seeds: \(\rho{=}5\) uses \(0\)–\(14\) (\(0\)–\(9\) exploratory; \(10\)–\(14\) held-out); \(\rho{=}7\) uses \(0\)–\(9\); \(\rho{=}10\) uses \(0\)–\(12\).

### 4.2 Sliced \(W_2\) (primary)

| \(g\) | Mean \(\Delta\) | \(d_z\) | 95% CI | \(\#>0\) | \(p\) (Holm) |
|--|--|--|--|--|--|
| 1 | +0.087 | 1.16 | [0.064, 0.112] | 38/38 | \(<10^{-6}\) |
| 2 | +0.043 | 0.96 | [0.029, 0.057] | 32/38 | \(5{\times}10^{-6}\) |
| 3 | +0.038 | 0.98 | [0.026, 0.051] | 36/38 | \(1{\times}10^{-6}\) |
| 4 | +0.022 | 0.49 | [0.008, 0.035] | 29/38 | 0.00029 |
| 5 | +0.027 | 0.91 | [0.018, 0.037] | 31/37 | \(6{\times}10^{-6}\) |

Pooled cells share \(\rho\) strata and are not independent Bernoulli trials; primary inference is the paired \(\Delta\) CI / Wilcoxon, not a binomial over cells.

### 4.3 Retention (secondary; stratified)

| \(\rho\) | \(n\) | bottom | rand | top | Full order |
|--|--|--|--|--|--|
| 5 | 15 | 0.604 | 0.350 | 0.128 | **14/15** |
| 7 | 10 | 0.372 | 0.248 | 0.113 | **6/10** |
| 10 | 13 | 0.248 | 0.183 | 0.120 | **4/13** |

At \(\rho{=}5\) (all \(n{=}15\)): top−rand mean \(\Delta{=}{-}0.222\), CI \([{-}0.28,{-}0.16]\), \(p{=}0.0001\); bottom−rand \({+}0.254\), CI \([0.19,0.32]\), \(p{=}0.0001\). Held-out \(10\)–\(14\): direction replicates (CIs exclude zero; full order \(4/5\)) but exact Wilcoxon is floored at \(n{=}5\).

### 4.4 Bottom-\(k\) quality, \(\rho\) sweep, and baselines

Mean sliced \(W_2\) at \(g{=}1\) (\(\rho{=}5\), \(n{=}15\)): bottom \(0.51\) < rand \(0.64\) < top \(0.80\).

Continuous \(\rho\) sweep (seeds \(0\)–\(4\)): full-order rate \(5/5\) at \(\rho\in\{3,4,5\}\), then \(3/5\) (\(\rho{=}6\)), \(2/5\) (\(\rho{=}7,8\)), \(1/5\) (\(\rho{=}10\)).

Baselines at \(\rho{=}5\) (seeds \(0\)–\(4\)): accumulate \(r{=}0.41\); mix \(0.37\); oracle bottom/top \(0.71/0.17\) ≈ proxy \(0.74/0.18\); learned real-only verifier bottom/top \(0.74/0.17\).

Precision/recall (\(k{=}5\)): top has highest precision (\(0.96\)) but worst \(W_2\); bottom trades precision (\(0.88\)) for better \(W_2\) and \(r\).

### 4.5 Proxy check and transfers

Denoising-MSE proxy vs exact GMM \(-\log p\) (\(\rho{=}5\), \(n{=}5\)): mean Spearman \(0.44\); top-\(k\) Jaccard \(0.64\). Oracle/verifier policies match proxy.

8D GMM / Digits-PCA: order 3/3. Digits pixel DDPM: 1/3. CIFAR tiny-UNet: full order \(0/3\); seed-0 FID at \(g{=}1\): bottom \(332\) vs top \(477\) / rand \(489\). **No ImageNet-scale result.**

## 5 Limitations

**Scale.** Toy GMMs / Digits / CIFAR tiny-UNet only; CIFAR FID is \(n{=}1\); no ImageNet / LT-benchmark.

**Researcher degrees of freedom / post-hoc revision.** Long exploratory search precedes this note. Treat bottom>rand>top as **post-hoc / exploratory**. Mitigation: \(W_2\) primary; stratified \(\rho\); held-out seeds; continuous \(\rho\) sweep; bootstrap CIs.

**Other.** Verifier is a real-only DDPM (not a Cai-style latent probe); baselines \(n{=}5\) at \(\rho{=}5\).

## 6 Conclusion

Top-\(k\) does not improve sliced \(W_2\) at \(g{=}1\). Minority retention ordering appears mainly in a moderate-imbalance band—not a general law.

## References

- Alemohammad et al. (2023). arXiv:2307.01850.
- Cai et al. (2025). arXiv:2511.12742.
- Feng et al. (2024). arXiv:2406.07515.
- Gerstgrasser et al. (2024). arXiv:2404.01413.
- Li et al. (2026). arXiv:2607.10853.
- Qiao et al. (2026). arXiv:2606.13732.
- Shumailov et al. (2023). arXiv:2305.17493.
- Xu et al. (2025). arXiv:2510.01184.

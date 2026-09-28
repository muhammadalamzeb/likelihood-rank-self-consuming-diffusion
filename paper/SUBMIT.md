# Publication status

**Author:** Muhammad Alamzeb · **Contact:** shayankhanmahar@gmail.com  
**Release:** https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion/releases/tag/v1.6.0  
**Zenodo DOI:** https://doi.org/10.5281/zenodo.22956890 → **New version** with v1.6.0 PDF still needed.

## Checklist

- [x] Power: ρ=5 \(n=15\) (held-out 10–14); ρ=7 \(n=10\); ρ=10 \(n=13\); \(W_2\) \(n=38\)
- [x] HARKing / held-out disclosure in manuscript
- [x] Qiao differentiation table; notation table; mechanism paragraph
- [x] Digits pixel/CVAE moved to appendix as preliminary/negative
- [x] CIFAR-10 tiny-UNet appendix check (ρ=5, n=3; full order 0/3; mean \(\bar r\) bottom>rand>top)
- [x] Sync PAPER.md with powered stats (stale markdown caused n=5 review)
- [x] Proxy vs exact GMM density (Spearman mean 0.44; Jaccard 0.64)
- [x] Bottom-k \(W_2\) cost + unranked mix baseline (ρ=5 seeds 0–4)
- [x] Continuous ρ sweep (seeds 0–4): full order 5/5 for ρ≤5, then decays to 1/5 at ρ=10
- [x] Accumulate + exact-density oracle baselines at ρ=5
- [ ] Zenodo New version upload
- [ ] arXiv after endorsement `Z4GDY9`
- [ ] FID / CIFAR-10-LT / ImageNet (out of CPU scope)

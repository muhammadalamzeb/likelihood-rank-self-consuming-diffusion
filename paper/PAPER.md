# Likelihood-Ranked Synthetic Selection in Self-Consuming Diffusion: Minority Extinction Ordering on Imbalanced Mixtures

## Abstract

Self-consuming training loops—retraining generative models on their own outputs—are known to induce *model collapse*, including loss of distributional tails. Prior work notes that bias toward “high-quality” synthetic samples worsens the quality–diversity tradeoff, and that task-level *verification* can stabilize labeled synthetic data. We study the most common *unlabeled* selection score for diffusion models: a denoising ELBO / likelihood proxy. On imbalanced 2D Gaussian mixtures with a tiny DDPM, under matched real-data mixing and matched synthetic budget, we find a reproducible **likelihood-rank ordering** of minority-mode survival:

\[
\text{bottom-}k \;\;>\;\; \text{random-}k \;\;>\;\; \text{top-}k
\]

in first-generation minority-mass retention (means \(0.46 / 0.29 / 0.16\)). Likelihood top-\(k\) also increases majority-mode concentration relative to random-\(k\). We **falsify** a natural “false stability” hypothesis: top-\(k\) does *not* improve sliced Wasserstein-2 distance to real data while minorities die—it typically worsens both. A Digits CVAE transfer was inconclusive due to classifier-based mass estimates under severe sample degeneration. We release configs, seeds, logs, and plotting code.

## 1 Introduction

As synthetic data proliferates, generative models are increasingly trained on mixtures of real and model-generated samples. Iterated self-consumption can cause *model collapse*: progressive loss of diversity and forgetting of rare modes [Shumailov et al., 2023; Alemohammad et al., 2023; Gerstgrasser et al., 2024]. Mitigations include accumulating real data [Gerstgrasser et al., 2024], injecting fresh real samples, and *filtering* synthetic data [Feng et al., 2024; Cai et al., 2025].

A practical question remains underspecified: when the only available score is the generator’s own likelihood (or denoising loss), how should one select synthetic samples? Feng et al. [2024] show that *task verifiers* for labeled synthetic answers can prevent collapse. Cai et al. [2025] filter “unrealistic” diffusion samples via latent probes and report aggregate FID/precision/recall. Alemohammad et al. [2023] observe that sampling bias toward high-quality generations drives a quality–diversity tradeoff. What is missing is a **matched-budget, mode-exact** causal contrast among likelihood ranks.

**Contributions.**

1. A controlled self-consuming DDPM protocol on imbalanced GMMs with exact mode masses.
2. Evidence for a **likelihood-rank ordering** of minority retention under matched \(k\) and matched real mix ratio.
3. An experimental **falsification** of W₂-based false stability for likelihood top-\(k\).

## 2 Related work

**Model collapse.** Shumailov et al. [2023] document recursive training collapse and tail forgetting. Gerstgrasser et al. [2024] show accumulating real+synthetic data can avoid collapse. Dohmatob et al. and Bertrand et al. provide theoretical rates. Alemohammad et al. [2023] (*MAD*) highlight quality-biased sampling.

**Verification and filtering.** Feng et al. [2024] argue verification of synthesized *labels* prevents collapse. Cai et al. [2025] (*LSF*) filter synthetic images using latent-space realism scores. Neither isolates unlabeled ELBO top-\(k\) vs random-\(k\) with mode-grounded metrics on continuous mixtures.

**Temperature / diversity knobs.** Inference-time temperature methods [Xu et al., 2025; Li et al., 2026] reshape sampling diversity; we instead intervene on *training-set selection* inside self-consuming loops.

## 3 Method

### 3.1 Self-consuming loop

Let \(f^{(g)}\) be a DDPM trained on \(\mathcal{D}^{(g)}\). Sample synthetic set \(\widehat{\mathcal{D}}^{(g+1)}\). Build the next train set by mixing a fixed real buffer \(\mathcal{D}_{\mathrm{real}}\) (fraction \(\alpha\)) with selected synthetics.

### 3.2 Likelihood proxy and policies

For each synthetic \(x\), define score \(s(x)\) as mean denoising MSE over a grid of times \(t\) (ELBO proxy; **lower** = higher likelihood). Policies at matched synthetic count:

- **top-\(k\)**: keep lowest \(s(x)\)
- **random-\(k\)**: uniform subset
- **bottom-\(k\)**: keep highest \(s(x)\)
- **mix**: random synthetics without rank filter (baseline)
- **replace**: pure synthetic replacement (collapse reference)

### 3.3 Metrics

Nearest-mean mode assignment on known GMM centers. Report minority-mean mass (modes \(1{\ldots}K-1\)), majority mass (boosted mode \(0\)), and sliced W₂ to held-out real samples.

## 4 Experiments

**Setup.** \(K{=}4\) modes on a circle, imbalance \(\rho\in\{5,10\}\), tiny MLP denoiser, \(G{=}5\) generations, seeds \(\{0,1,2\}\), \(\alpha{=}0.5\). Config: `experiments/configs/w2_fs.json`. Logs: `experiments/logs/w2_fs.jsonl`.

**Primary result (retention).** Mean minority-mass ratio \(m^{(1)}_{\min}/m^{(0)}_{\min}\):

| Policy | Mean retention | Std |
|--------|----------------|-----|
| bottom_k | 0.457 | 0.218 |
| mix | 0.298 | 0.096 |
| rand_k | 0.289 | 0.140 |
| top_k | 0.161 | 0.085 |

**Ordering.** At \(\rho{=}5\), generation 1, all three seeds satisfy \(\mathrm{bottom}>\mathrm{rand}>\mathrm{top}\) on minority mass; top-\(k\) also raises majority mass vs random-\(k\) (Table `paper/table_primary_rho5_g1.csv`).

**Falsified claim.** Top-\(k\) does **not** improve sliced W₂ vs random-\(k\) while minorities die (mean \(\Delta\)W₂ \(>0\) for generations 1–5). The original “false stability via better W₂” hypothesis is rejected.

**Digits transfer.** A CVAE self-consuming loop on sklearn Digits (`w2_fs_digits`) showed weak entropy trends in the same direction but unreliable class-mass estimates under collapsed samples; we treat transfer as **inconclusive**.

![Retention bars](figures/retention_bars.png)

![Minority vs generation](figures/minority_vs_generation_rho5.png)

## 5 Analysis

Likelihood top-\(k\) preferentially retains synthetic points the current model already explains well—typically majority-mode samples—reducing the inflow of minority examples into \(\mathcal{D}^{(g+1)}\). Bottom-\(k\) does the opposite relative to random. This is consistent with MAD’s quality-bias narrative but makes the **score-rank** interventional and mode-exact. Aggregate W₂ is a poor monitor for this failure: it moves with top-\(k\) in the *wrong* direction for a “quality win.”

## 6 Limitations

- Toy 2D GMMs and tiny networks; not ImageNet-scale.
- Digits transfer inconclusive.
- Effect strongest at moderate imbalance and early generations; at \(\rho{=}10\) minorities are already near floor at \(g{=}0\).
- Novelty is **L2 empirical**: related to MAD/LSF/Feng; we do not claim a new named training algorithm that beats LSF on FID.

## 7 Conclusion

Unlabeled likelihood ranking of synthetic data in self-consuming diffusion induces a stable **minority-retention order** (bottom-\(k\) > random-\(k\) > top-\(k\)) under matched budgets on imbalanced mixtures, while failing to deliver W₂ improvements. Preferring “most likely” synthetics is not a free lunch for minority survival.

## References

- Alemohammad, S., et al. (2023). Self-Consuming Generative Models Go MAD. arXiv:2307.01850.
- Cai, Z., et al. (2025). Stabilizing Self-Consuming Diffusion Models with Latent Space Filtering. arXiv:2511.12742.
- Feng, Y., et al. (2024). Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification. arXiv:2406.07515.
- Gerstgrasser, M., et al. (2024). Is Model Collapse Inevitable? arXiv:2404.01413.
- Li, P., et al. (2026). Diversify Diffusion with Temperature Sampling and Variance-Corrective Time Shifting. arXiv:2607.10853.
- Shumailov, I., et al. (2023). The Curse of Recursion: Training on Generated Data Makes Models Forget. arXiv:2305.17493.
- Xu, Y., et al. (2025). Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models. arXiv:2510.01184.

## Appendix A Reproducibility

See repository `README.md`. Principal command:

```bash
python experiments/scripts/run_w2_false_stability.py --out experiments --seeds 0 1 2 --rhos 5.0 10.0
python experiments/scripts/analyze_w2_fs.py
python experiments/scripts/make_w2_figures.py
```

All numerical tables in this paper are produced from `experiments/logs/w2_fs.jsonl` via the analysis scripts (no hand-edited metrics).

## Appendix B Experiment provenance

- Experiment IDs: `w2_fs` (GMM DDPM), `w2_fs_digits` (Digits CVAE; inconclusive).
- Frozen historical stacks v0–v8 were **not** reused as novelty claims.
- Constraint Set v4; discovery record: `docs/PATH2_DISCOVERY.md`; survivor: `docs/RESEARCH_SURVIVOR.md`.

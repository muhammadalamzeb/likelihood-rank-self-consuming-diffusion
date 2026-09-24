# RESEARCH SURVIVOR — W2-1

**Status:** MODIFY → KEEP (empirical pilot falsified original W₂-dissociation; revised claim)  
**Date:** 2026-09-23  
**Constraint set:** v4  
**Discovery ref:** `PATH2_DISCOVERY.md` Wave 2  
**Pilot:** `experiments/logs/w2_fs.jsonl` (quick) — top‑k accelerates minority death vs rand‑k but **does not** improve sliced W₂ (often worsens). Original “better W₂ + worse minority” **falsified**.

---

## Final research question (revised)

Under matched real-mix ratio and matched synthetic budget, does **likelihood top‑k** selection in self-consuming diffusion cause **faster minority-mode extinction** than **random‑k**, while increasing **majority-mode concentration** (a misleading “confidence” signal)?

## Hypothesis (revised)

With \(\pi_{\mathrm{top}}\) vs \(\pi_{\mathrm{rand}}\) at matched \(k\) and same \(\alpha_{\mathrm{real}}\),

\[
m_{\mathrm{min}}(\hat p_G^{\mathrm{top}}) < m_{\mathrm{min}}(\hat p_G^{\mathrm{rand}})
\quad\text{and}\quad
m_{\mathrm{maj}}(\hat p_G^{\mathrm{top}}) > m_{\mathrm{maj}}(\hat p_G^{\mathrm{rand}})
\]

for generations \(G \ge G^\star\). Optional: sliced W₂ may worsen for top‑k (honest reporting — not hidden).

## Scientific object

**Selection policy** \(\pi\) over synthetic candidates in a self-consuming loop, especially **unlabeled likelihood verification**.

## Closest prior work

1. Shumailov et al., *Curse of Recursion* (arXiv:2305.17493) — collapse; tails forgotten.  
2. Gerstgrasser et al. (arXiv:2404.01413) — accumulating real+synth can prevent collapse.  
3. Alemohammad et al., *Self-Consuming Generative Models Go MAD* (arXiv:2307.01850) — quality-biased sampling and quality–diversity tradeoff.  
4. Feng et al., *Beyond Model Collapse* (arXiv:2406.07515) — **verification** of synthesized **labels** prevents collapse.  
5. Cai et al., *Latent Space Filtering* (arXiv:2511.12742) — filter unrealistic synth via latent probes; report FID/precision/recall.

## Novelty argument

| Prior | Establishes | Does **not** establish |
|-------|-------------|-------------------------|
| Shumailov / MAD | Collapse & quality bias hurt diversity | Exact **top‑k likelihood vs random‑k** dissociation with mode-exact masses under **stable/better W₂** |
| Feng | Task verifiers help **labeled** synth | Transfer to **unlabeled likelihood** as verifier for density models |
| LSF | Realism filtering helps aggregates; recall tracked | Likelihood top‑k; **false stability** (better W₂ + worse minority) vs matched random |

A 2026 reviewer can grant: a **controlled, falsifiable false-stability law** for the most naive unlabeled verifier.

## Competing explanations

1. **C1:** Any filtering that reduces sample count causes minority death (dose effect). → Control: matched \(k\).  
2. **C2:** Top‑k only remaps overall collapse already present in replace-all. → Control: compare to random‑k and to mix-α without selection.  
3. **C3:** Likelihood correlates with majority so result is trivial. → Still valuable if practitioners use this verifier; quantify effect size vs random.

## Decisive intervention

Swap only \(\pi\): top‑k vs random‑k (same \(k\), same model class, same real buffer). Measure minority mass and W₂ each generation.

## Falsification condition

**KILL hypothesis** if across ≥3 seeds, top‑k does **not** show faster minority decay than random‑k, **or** top‑k’s aggregate metric is never better/stable when minority dies (no dissociation).

## Minimum experiment

- Data: 2D GMM, \(K=4\), imbalance \(\rho\in\{5,10\}\), known means.  
- Model: tiny MLP DDPM (reuse stack-v6 patterns).  
- Loop: \(G=5\) generations; each gen train \(E\) epochs; sample \(N_{\mathrm{synth}}\); apply π; mix with fixed real buffer.  
- Policies: `replace`, `mix_alpha`, `top_k_lik`, `rand_k`, `bottom_k_lik` (optional contrast).  
- Metrics: mode assignment by nearest true mean; \(m_{\min}\); empirical W₂ (or sliced) to held-out real; MMD_rbf.

## Full experiment matrix

| Factor | Levels |
|--------|--------|
| seed | 0,1,2 |
| ρ | 5, 10 |
| policy | replace, mix_α=0.5, top_k, rand_k |
| G | 0…5 |
| k | = 0.5 × N_synth (matched) |

## Compute estimate

CPU, <2–4 hours total for matrix; no GPU required.

## Dataset plan

Fully synthetic GMM; no download. Optional transfer: sklearn Digits subset (secondary; not required for core claim).

## Expected paper contribution

**Empirical scientific property:** Unlabeled likelihood verification in self-consuming diffusion can induce **false stability** — aggregate metrics that mislead relative to minority-mode survival — demonstrated with a decisive matched-\(k\) intervention.

## Implementation gate

Novelty case documented → proceed to code under `experiments/scripts/run_w2_false_stability.py` without further owner approval.

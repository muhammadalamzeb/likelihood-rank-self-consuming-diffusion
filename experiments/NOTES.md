# Observation notes (Strategy B)

**FROZEN 2026-09-23** for Path 2 intake. Outcomes below are archival; do not rewrite into new novelty claims. Do not delete companion logs/scripts.

Status: **no topic lock**. Path 1 terminal; Path 3 recon↔ATE terminal (0/3). See `docs/PATH2_READINESS.md`.

## Stack v0 — CIFAR conditional DDPM (`Ketansomewhere/cifar10_conditional_diffusion1`)

| Logs / grids | Probe |
|--------------|-------|
| `p1p2_seed{0,1}.*` | P2 steps 10/20/50 |
| `p6_seed{0,1}.*`, `p6_wrongclass_*`, `p6_pairs_*` | P6 fixed-noise × class |

| ID | Decision | Why |
|----|----------|-----|
| FC1–FC3 | **KILL** | NFE cliff / twin asymmetry / class difficulty — exclusions 31–33 |
| FC4–FC5 | **KILL** | Label vs noise split — exclusions 34–35 |

## Stack v1 — FM CIFAR (`FrankCCCCC/cfm-cifar10-32`)

| Logs / grids | `fm_v1_seed{0,1}.*`, `fm_v1_steps{5,10,20}_seed{0,1}.png` |

| ID | Decision | Why |
|----|----------|-----|
| FM1 | **KILL** | Euler step cliff = same object as FC1; exclusion **37** |

## Stack v2 — Wine Conditional VAE (train-from-scratch)

| Logs | `tab_v2_seed{0,1}.jsonl` |
| Imbalance | class2 ≈5–6 vs 41/50 after 80% drop |

| ID | Observation | Decision | Why |
|----|-------------|----------|-----|
| T1 | Rare-class centroid L2 inconclusive (raw-feature scale confound) | **no claim** | Metric flawed; rare-mode issues owned by CTGAN/CTAB-GAN+ |
| T2 | Support leakage ≈ 0 | **no failure** | — |
| T3 | Corr sign flips ~4–16 pairs/class (repro seeds 0–1) | **KILL** | CTGAN/TVAE correlation distortion; TabSynDex / IJACSA 2024 analyses |

## Stack v3 — Digits Conditional VAE (β sweep)

| Logs | `digits_v3_seed{0,1}.jsonl` |

| ID | Observation | Decision | Why |
|----|-------------|----------|-----|
| **D1** | β↑ → diversity↓, class-acc↑ (repro) | **KILL** | Named β-VAE tradeoff (Higgins et al.) |

Exclusions through **39**. Publication blocked — `docs/PUBLICATION_STATUS.md`.

## Stack v4 — SBM tiny GraphVAE

| Logs | `graph_v4_seed{0,1}.jsonl` |
| Script | `scripts/run_probe_graph_v4.py` |

| ID | Observation | Decision | Why |
|----|-------------|----------|-----|
| G1-sparse | `p_in`≤0.35 → empty graphs (repro) | **KILL** | GraphVAE independent-edge decoder / sparse collapse |
| G1-strong | `p_in`≥0.55 → two-clique collapse, Hamming≈0 | **KILL** | Known GraphVAE community/structure failure |
| G3 | ER dens≤0.35 empty; 0.5 OK | **KILL** | Same object as G1-sparse |
| G4 | Edge-flip ↓ modularity | **KILL** | Trivial measure |

No topic lock.

## Stack v5 — Digits CVAE (erase / τ / few-shot)

| Logs | `digits_v5_seed{0,1}.jsonl` |
| Note | Fashion-MNIST download aborted; same probes on offline digits |

| ID | Observation | Decision | Why |
|----|-------------|----------|-----|
| F1 | erase↑ → darker samples; acc flat | **KILL** | Aug ablation |
| F2 | τ↑ → diversity↑ (repro) | **KILL** | VAE prior-temperature neighborhood |
| F3 | N=20/class → acc collapse | **KILL** | Few-shot GenAI / data-scale |

No topic lock.

## Stack v6 — 2D GMM tiny DDPM

| Logs | `gmm_v6_seed{0,1}.jsonl` |

| ID | Observation | Decision | Why |
|----|-------------|----------|-----|
| M1/M2 | Full mode coverage across δ, K | **no claim** | — |
| M3 | ρ≥10 → uncovered modes, entropy↓ (repro) | **KILL** | Named multimodal coverage / imbalance |

No topic lock. V3-21 meta Path-3 also **KILL** (AI-scientist failure lit).

## Stack v7 — synthetic spectrogram CVAE

| Logs | `spec_v7_seed{0,1}.jsonl` |

| ID | Observation | Decision | Why |
|----|-------------|----------|-----|
| S1 | SNR↓ → samp_std↓; acc flat | **KILL** | Noisy-train / data-quality |
| S2 | f0 jitter↑ → acc cliff (repro) | **KILL** | Attribute noise; excl.36-adj |
| S3 | warp flat | **no claim** | — |

No topic lock. Path-1 EV low under current v3 bar.

## Stack v8 — SEM tabular CVAE → ATE

| Logs | `sem_v8_seed{0,1}.jsonl` |

| ID | Observation | Decision | Why |
|----|-------------|----------|-----|
| C1–C3 | Synth ATE badly biased vs τ; recon MSE not a safeguard (repro) | **KILL** | Amad et al. 2026 causal-synth pitfall; excl.38-adj |

**Path-1 terminal recorded:** `docs/TERMINAL_FINDING_V3.md`.

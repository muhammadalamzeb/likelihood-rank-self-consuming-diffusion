# Path 2 / Ownership-wave discovery (Constraint Set v4)

**Date:** 2026-09-23  
**Mode:** Full ownership mandate — discover → kill-test → select → implement → paper  
**Bar:** Constraint Set **v4** (`CONSTRAINTS.md`). Historical Path-2 NO KEEP under **v3** remains valid as prior; this wave uses relaxed L2 rules but **does not** reopen excl. 1–39 as novelty.  
**Frozen stacks v0–v8:** preserved; code patterns reusable; killed claims not reusable.

---

## Historical note (v3 Path 2)

Prior autonomous Path 2 ended **NO KEEP** (`PATH2_DISCOVERY` historical outcome; P2-1…P2-5 KILL). That search is **not** restarted; Wave 2 searches **outside** those named neighborhoods (CIB-Med, UniGen-MOE, Doob tilt, Han CFM residuals, CoBind).

---

## Stage 1 — Broad discovery (mechanism tensions)

Inspected / queried (bibliography of papers actually opened or API-abstract-read):

| ID | Paper | Relevance |
|----|-------|-----------|
| A1 | Shumailov et al. 2305.17493 — Curse of Recursion | Model collapse; tails forgotten |
| A2 | Gerstgrasser et al. 2404.01413 — Is Model Collapse Inevitable? | Accumulation of real+synth |
| A3 | Alemohammad et al. 2307.01850 — Self-Consuming Go MAD | Quality-biased sampling ↔ quality–diversity |
| A4 | Feng et al. 2406.07515 — Beyond Model Collapse / Verification | Verifiers prevent collapse (mostly **labeled** synth) |
| A5 | Cai et al. 2511.12742 — Latent Space Filtering | Filter “unrealistic” synth in diffusion loops; FID/P/R |
| A6 | Li et al. 2607.10853 — Diversify Diffusion / temperature | Rare-mode lift via tempered sampling |
| A7 | Xu et al. 2510.01184 — Temporal Score Rescaling | Temperature without mode drop (local) |
| A8 | Yao et al. CVPR 2025 — Reconstruction vs Generation | Tokenizer recon≠gen |
| A9 | TJS 2607.06114 — Endpoint decodability / early exit | Aggregate quality vs NFE |
| A10 | Ning / Anti-Exposure Bias / NAG 2025–26 | Exposure bias / noise shift owned |
| A11 | Long-tail diffusion 2507.09052, 2402.10821 | Train-time long-tail methods |
| A12 | Generation Phases FM 2510.24830 | Temporal phases owned |

**Open tension retained for Stage 3:** Feng/LSF argue **selection/filtering of synthetic data stabilizes** self-consuming loops (aggregate metrics). Alemohammad notes **quality bias** hurts diversity. Missing: a **decisive, mode-exact** demonstration that a *natural unlabeled verifier* (generator likelihood top‑k) can **stabilize aggregate distance while accelerating minority-mode extinction** relative to size-matched random retention — i.e. **false stability**.

---

## Stage 2 — Aggressive exclusion (immediate)

| Eliminated | Why |
|------------|-----|
| NFE / early-exit rare-mode harm | excl. **31**; TJS owns aggregate early-exit |
| CFG diversity under imbalance alone | CFG quality–diversity known; long-tail train papers |
| Temperature lifts rare modes | **Owned** by A6/A7 |
| CFM≠NLL residual→regret | Path-2 P2-4 FUTURE_WORK |
| Off-target editing Assoc→drift | P2-1 NAMED |
| MoE-DiT deadlock U-curve | P2-2 NAMED |
| VAE recon vs gen dissociation | A8 NAMED |
| Exposure bias / noise shift fixes | A10 NAMED |
| Recon↔ATE | Path-3 TERMINAL |
| β-VAE / tabular corr / GraphVAE empty | excl. 38–39 / stacks |
| “Filter unrealistic synth helps” as method | A5 NAMED (LSF) |
| “Verification prevents collapse” as method | A4 NAMED (Feng) |

---

## Stage 3 — Candidates (≤10)

### W2-1 — Likelihood-top-k selection induces false stability (KEEP candidate)

| Field | Content |
|-------|---------|
| **Object** | Unlabeled verifier \(V(x)=\) generator likelihood / denoising ELBO proxy; selection policy π on synthetic batch |
| **Claim** | In self-consuming diffusion on imbalanced mixtures, **likelihood top‑k** retention can improve or stabilize aggregate W₂/MMD vs real while **minority mode mass** declines **faster** than under **size-matched random** retention of the same synthetic count |
| **Why it matters** | Practitioners follow “keep high-quality synthetic” advice (Feng/LSF spirit) using the **easiest** unlabeled score (model likelihood); may declare collapse “solved” by aggregate metrics while rare modes vanish |
| **Closest** | A1–A5; MAD quality bias note; LSF recall |
| **They establish** | Collapse exists; accumulation/verification/LSF help aggregates; quality bias ↔ diversity qualitatively |
| **Exact difference** | **Mode-exact false-stability** under **likelihood top‑k vs random‑k** matched size on continuous self-consuming diffusion — not labeled verification, not OLE/probe LSF, not “diversity drops under collapse” |
| **Strongest rejection** | “MAD already said quality bias hurts diversity; LSF already reports recall.” |
| **Kill experiment** | If top‑k does **not** accelerate minority death vs random‑k at matched n_synth, or if literature already reports this exact contrast |
| **Min. experiment** | 2D imbalanced GMM + tiny DDPM; generations; policies; mode masses |
| **Compute/data** | CPU; synthetic GMM |
| **Novelty** | **L2** (empirical law + practical warning) |
| **Not ablation/framework/FW** | Intervening variable is **selection policy**; claim is dissociation aggregate vs minority |

### W2-2 — Cross-architecture collapse asymmetry (FM teacher → DDPM student)

| Field | Content |
|-------|---------|
| **Object** | Teacher inductive bias in self-consuming transfer |
| **Claim** | Minority extinction rate depends on teacher–student architecture pair beyond capacity |
| **Closest** | Collapse lit (same-family); distillation mode-seeking |
| **Rejection** | Confounded by capacity/optimization |
| **Verdict lean** | Weak without huge controls → likely KILL |

### W2-3 — Early-exit γ preserves FID but not rare-mode recall

| Field | Content |
|-------|---------|
| **Closest** | TJS; excl. 31 |
| **Verdict** | **KILL** NAMED/excl |

### W2-4 — CFG×imbalance: w* quality ≠ w* min-mode coverage

| Field | Content |
|-------|---------|
| **Closest** | CFG; long-tail diffusion |
| **Verdict** | **KILL** as L1 restatement of quality–diversity |

### W2-5 — Temperature γ recovers minorities under CFG

| Field | Content |
|-------|---------|
| **Closest** | A6/A7 |
| **Verdict** | **KILL** NAMED |

### W2-6 — Soft class-embedding vs hard labels for rare modes

| Field | Content |
|-------|---------|
| **Closest** | Label smoothing; long-tail diffusion training |
| **Verdict** | **KILL** ABLATION / crowded |

### W2-7 — CFM residual predicts alignment hacking

| Field | Content |
|-------|---------|
| **Verdict** | **KILL** Path-2 P2-4 |

### W2-8 — VAE rFID vs gFID class-wise dissociation

| Field | Content |
|-------|---------|
| **Verdict** | **KILL** A8 |

### W2-9 — Exposure-bias magnitude predicts rare-mode loss

| Field | Content |
|-------|---------|
| **Verdict** | **KILL** A10 |

### W2-10 — Feng verification proxy fails transfer to unlabeled density loops

| Field | Content |
|-------|---------|
| **Object** | Same as W2-1 framed as negative transfer of Feng slogan |
| **Note** | Merges into W2-1 as discussion framing; not a separate KEEP |

---

## Stage 4 — Adversarial kill-test

| ID | Verdict | Reason |
|----|---------|--------|
| W2-3…W2-9 | **KILL** | Named / exclusion / Path-2 / crowded |
| W2-2 | **KILL** | Confounded architecture–capacity; weak L2 |
| W2-10 | Merge | Absorbed into W2-1 |
| **W2-1** | **KEEP (provisional)** | Survives as distinct **false-stability** claim pending experimental test; novelty vs MAD/LSF/Feng argued in `RESEARCH_SURVIVOR.md` |

**Stopping rule:** One provisional survivor → stop discovery → document → implement → **experimental kill possible**.

---

## Outcome of discovery phase

**PROVISIONAL KEEP — W2-1** (likelihood-top-k false stability in self-consuming diffusion)

Next: `docs/RESEARCH_SURVIVOR.md` → implementation → experiments. If experiments falsify, apply Automatic Pivot Rule (MODIFY only if defensible, else KILL and rediscover).

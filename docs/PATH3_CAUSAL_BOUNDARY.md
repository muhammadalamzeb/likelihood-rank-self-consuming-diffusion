# Path 3 — Causal-synthesis empirical boundary (TERMINAL)

**Status:** **TERMINAL** — 2026-09-23  
**Survivors:** **0 / 3** (P3-C1, P3-C2, P3-C3 all **KILL**)  
**Publication viability:** **not claimed; not pursued further on this question.**

**Freeze rules (owner 2026-09-23):**
- Do **not** run additional experiments on reconstruction↔ATE.
- Do **not** repackage stacks v0–v8 into a paper on this question.
- Claims and kill reasons below are **frozen** — do not rewrite to manufacture novelty.

Working question (historical): Can repeated causal-synthesis failures support a precise empirical boundary where finite reconstruction fidelity ceases to track causal estimand validity?

---

## Adversarial literature map (2025–2026 core)

| Paper | What it already establishes |
|-------|-----------------------------|
| **Amad et al., “Generative Synthetic Data for Causal Inference: Pitfalls, Remedies, and Opportunities”** (arXiv:2604.23904, 2026) | (i) TSTR / privacy distance / **predictive or reconstruction fidelity do not imply ATE preservation**. (ii) **Prop. 3.1**: synthetic ATE error decomposes into covariate-law error + treatment-effect **contrast** error. (iii) **Thm. 3.1**: joint row reconstruction dilutes outcome/contrast loss by ~\(1/(d+1)\). (iv) Prediction loss weights contrast by **overlap** \(\pi(1-\pi)\), so imbalance/limited overlap underlearns interventions. (v) Hybrid covariate + separate \(A,Y\) mechanisms as remedy; stress-tests over overlap, \(d\), \(N\), effect complexity. |
| **STEAM** — “Improving the Generation and Evaluation of Synthetic Data for Downstream Medical Causal Inference” (NeurIPS 2025; arXiv:2510.18768) | Desiderata: preserve \(P_X\), treatment mechanism, outcome mechanism. Metrics **beyond** recon/TSTR: \(P_{\alpha,X}\), \(R_{\beta,X}\), \(\mathrm{JSD}_\pi\), **UPEHE**. Mechanism-aware generators beat joint CTGAN / **TVAE** / TabDDPM / flows on causal metrics. |
| Related / supporting | CTGAN/TVAE/TabDDPM as joint baselines; GReaT LLM tabular synth; surveys noting causal-parameter preservation was under-studied until these works; positivity/trimming lit (orthogonal). |

**Implication for our archive:** Stack **v8** (SEM + joint tabular CVAE → biased ATE with finite recon MSE; capacity↑ helps recon more than ATE) is a **toy instance** of Amad Thm. 3.1 / STEAM’s “joint vs mechanism” gap — **not** a new boundary. Stacks **v0–v7** are mostly **off-topic** for this working question (image/graph/audio GenAI kill-log); they cannot be honestly reorganized into a causal-fidelity law without cherry-picking v8.

---

## Candidate negative-result claims (≤3) — FROZEN VERDICTS

### P3-C1 — “Finite reconstruction MSE does not guarantee ATE preservation under joint generative synthesis”

| Field | Content |
|-------|---------|
| **Falsifiable claim** | For joint-row synthesizers (CVAE/GAN/diffusion tabular), there exist regimes with reconstruction (or TSTR) error below a practical threshold while \(\|\widehat{\mathrm{ATE}}_{\mathrm{synth}}-\mathrm{ATE}^\star\|\) remains large. |
| **Lit already establishes** | Amad Prop. 3.1 + Thm. 3.1 + experiments; STEAM shows TVAE/CTGAN/TabDDPM fail mechanism metrics while being standard generative models. |
| **Missing piece** | **None material.** v8 is a miniature replication. |
| **Strongest rejection** | “Amad 2026 already proved and demonstrated this; you rediscovered Thm. 3.1 on a 4-D SEM.” |
| **Min. distinguishing experiment** | Would need a claim Amad does **not** make (they already cover recon/TSTR ≠ ATE). |
| **Result that KILLs** | Finding Amad/STEAM already state the claim → **immediate kill.** |
| **Level** | L0–L1 relative to 2026 lit |
| **Verdict** | **KILL** |

---

### P3-C2 — “There is a critical reconstruction (or ELBO) threshold \(R^\star\) beyond which ATE error collapses / below which it explodes”

| Field | Content |
|-------|---------|
| **Falsifiable claim** | \(\exists R^\star\) such that for joint synthesizers, recon-MSE \(< R^\star\) implies ATE error \(<\varepsilon\), and recon-MSE \(> R^\star\) implies ATE error \(>\delta\), with a sharp transition. |
| **Lit already establishes** | Amad: failure is **structural** (contrast under-weighting + overlap weighting), **not** a scalar recon threshold. Better joint recon can still miss \(Q(1,W)-Q(0,W)\). STEAM: need mechanism metrics, not a recon cutoff. Our v8 **C2**: ↑\(z\)-dim ↓ recon MSE while ATE error stays large — **anti-evidence** for a protective \(R^\star\). |
| **Missing piece** | A universal recon→ATE phase boundary would **contradict** Amad’s theory unless heavily scoped; no gap that favors this claim. |
| **Strongest rejection** | “You propose a threshold law that Amad’s decomposition says cannot exist without controlling the contrast term; your own C2 falsifies it.” |
| **Min. distinguishing experiment** | Sweep recon via capacity/early-stop; test whether ATE error is a monotone function of recon with a detectable knee **after** conditioning on overlap and contrast error — if knee vanishes once contrast error is controlled, claim dies. |
| **Result that KILLs** | No knee, or knee explained entirely by Amad’s overlap/contrast terms; or recon↓ with ATE error flat/↑ (already seen in v8). |
| **Level** | Would-be L2–L3 **if** true; empirically/theoretically unsupported |
| **Verdict** | **KILL** |

---

### P3-C3 — “Reconstruction-ranked model selection systematically selects worse causal fidelity than contrast-aware selection (regret of the wrong proxy)”

| Field | Content |
|-------|---------|
| **Falsifiable claim** | Across a fixed candidate set of joint synthesizers, argmin recon-MSE (or TSTR) has strictly higher expected ATE/CATE error than argmin of a contrast-aware score (e.g. held-out factual \(Y\) loss on rare \(A\), or UPEHE/JSDπ), with quantified selection regret. |
| **Lit already establishes** | Amad Fig. 1 / text: TSTR & DCR “do not fully determine causal usefulness.” STEAM: evaluate with JSDπ + UPEHE precisely because resemblance/predictive utility are inadequate; mechanism-aware models win on those metrics. |
| **Missing piece** | A **formal selection-regret** theorem + multi-model bakeoff could be incremental engineering/eval — but the **scientific content** (“wrong proxy”) is already the explicit message of Amad + STEAM. |
| **Strongest rejection** | “STEAM’s evaluation section *is* this paper; Amad already warned against proxy-based causal validation. Quantifying regret is a workshop note, not a new boundary.” |
| **Min. distinguishing experiment** | ≥5 synthesizer configs × ≥3 SEMs; compare selection by recon vs UPEHE/contrast proxy vs oracle ATE error; report regret. Survives only if regret pattern is **new** (e.g. non-monotone, architecture-specific) beyond “proxies disagree.” |
| **Result that KILLs** | Regret pattern fully predicted by Amad Thm. 3.1 / STEAM tables; or identical ranking once standard causal metrics are used. |
| **Level** | L1 (measure / eval cookbook) under v3 bar |
| **Verdict** | **KILL** (fails “not merely benchmark / cosmetic”; no new intervening-variable law) |

---

## Summary verdicts

| ID | Claim sketch | Verdict |
|----|--------------|---------|
| P3-C1 | Recon ≠ ATE for joint synth | **KILL** (Amad + STEAM) |
| P3-C2 | Critical recon threshold \(R^\star\) | **KILL** (contradicts Amad; v8 C2 anti-evidence) |
| P3-C3 | Proxy model-selection regret | **KILL** (eval message already published) |

**Surviving Path-3 candidates for this working question: 0.**

---

## Path 3 status (TERMINAL)

**Exhausted** for “empirical boundary: reconstruction fidelity vs causal estimand validity.”

Next project mode: **Path 2 — external direction intake** (`EXTERNAL_DIRECTION_INTAKE.md`). Do not reopen recon↔ATE without a claim that escapes Amad Thm. 3.1 / STEAM.

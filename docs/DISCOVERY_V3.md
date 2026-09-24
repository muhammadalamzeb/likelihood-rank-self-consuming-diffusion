# Discovery under Constraint Set v3

**Started:** 2026-09-23  
**Mode:** New search under owner-approved **v3** relaxation (see `CONSTRAINTS.md`)  
**Not:** Reboot of v1 text neighborhoods; not a reinterpretation of `TERMINAL_FINDING_V2.md` as “GenAI is exhausted”

**Status update (2026-09-23):** Autonomous **Path 2** funnel completed — **NO KEEP**. See [`PATH2_DISCOVERY.md`](PATH2_DISCOVERY.md). This file remains the archive of the earlier v3 candidate wave.

**Rules this round:** ≤5 candidates → adversarial kill-test → **no implementation / no paper** unless KEEP or strong MODIFY.

Exclusions **1–39** remain closed unless explicitly flagged.

---

## Candidate V3-1 — Cross-paradigm class-difficulty ranking transfer

| Field | Content |
|-------|---------|
| **Scientific object** | Per-class *generative difficulty ranking* as a property of the data distribution vs of the generative paradigm |
| **Falsifiable claim** | On CIFAR-10, Spearman ρ of per-class quality ranks (e.g. class-acc / cFID) between a conditional **DDPM** and a conditional **FM** model is significantly higher than ρ between two independent seeds of the *same* paradigm; i.e. ranking is paradigm-stable beyond seed noise |
| **Closest 2025–2026 papers** | Pfrommer et al. Schedule Deviation (class-varying path inconsistency); DC-FM (path/time difficulty, not class rank transfer); C2OT (conditional FM OT mismatch); Patch Forcing (difficulty heterogeneity) |
| **Exact difference** | Those papers characterize difficulty *within one training/sampling object*. This claims a **cross-paradigm transfer law** for class ranks |
| **Strongest reviewer rejection** | “Organics are harder for everyone—expected from data geometry; measuring Spearman ρ is a benchmark, not a contribution.” |
| **Proposed kill-test** | Search for DDPM↔FM (or diffusion↔flow) **class-rank correlation** studies; check whether exclusion **33** already covers the object |
| **Minimum viable experiment** | Public CIFAR cond. DDPM + FM ckpts; ≥2 seeds; fixed NFE; per-class classifier acc + optional cFID; Spearman + permutation test |
| **Compute / data** | CPU/Colab T4; public CIFAR; existing HF ckpts; ~hours |
| **Novelty level** | L2 (empirical law) if transfer holds with a clear practical implication; else L1 |
| **Why not ablation / framework / future-work** | Not tuning a method; not completing a named paper’s checklist; intervening comparison is *paradigm identity* |

### Kill-test verdict — **KILL**

**Reason:** Under v3, L2 must not be “merely a benchmark.” This candidate’s deliverable *is* a comparative ranking correlation—measure-only / benchmark-shaped. Closest literature already owns **class-varying difficulty** inside diffusion/FM (Pfrommer; Patch Forcing; DC-FM’s difficulty notion). Adjacent to exclusion **33** without a new intervening mechanism.  
**No exclusion reopen.**

---

## Candidate V3-2 — Inference-time UNet/DiT block ablation → class identity vs appearance

| Field | Content |
|-------|---------|
| **Scientific object** | Causal role of residual blocks in class-conditional generation (identity vs texture/background) |
| **Falsifiable claim** | Zeroing early vs late blocks at inference differentially destroys class identity vs high-frequency appearance; block-importance order is class-dependent |
| **Closest 2025–2026 papers** | Stable Flow (vital layers, training-free editing); TexTailor / Unraveling MMDiT Blocks (block remove/disable/enhance); Localizing Knowledge in DiTs |
| **Exact difference** | Would apply the same causal ablation lens to **small class-conditional CIFAR UNets** rather than FLUX/SD3 editing |
| **Strongest reviewer rejection** | “Stable Flow / TexTailor already did block ablation causally; CIFAR UNet is a toy replication.” |
| **Proposed kill-test** | Confirm whether block-role hierarchy is already established for UNet and DiT generative models |
| **Minimum viable experiment** | Ablate residual stages on `Ketansomewhere/cifar10_conditional_diffusion1`; measure class-acc vs LPIPS-to-full |
| **Compute / data** | Colab/CPU; public ckpt |
| **Novelty level** | L2 at best |
| **Why not ablation…** | *Is* primarily an ablation study of known mechanisms |

### Kill-test verdict — **KILL**

**Reason:** Named object—training-free **layer/block importance** for diffusion generators is owned (Stable Flow 2024; TexTailor / MMDiT block analysis 2026; Localizing Knowledge in DiTs). Auto-KILL: existing named object + cosmetic scale-down.  
**No exclusion reopen.**

---

## Candidate V3-3 — Codebook utilization entropy λ at **fixed** compression

| Field | Content |
|-------|---------|
| **Scientific object** | Utilization / entropy of a VQ codebook as a causal knob for *generation* quality when compression is held fixed |
| **Falsifiable claim** | Sweeping entropy-regularization weight λ at fixed K and spatial compression yields **non-monotonic** downstream gFID (intermediate utilization beats both collapse and “max utilization”) |
| **Closest 2025–2026 papers** | CRT / “When Worse is Better” (compression–generation; exclusion **20**); FVQ / 100% utilization training; VAEVQ; GSQ entropy-loss ablations; “Taming the Entropy Cliff” (VCQ) |
| **Exact difference** | CRT varies compression / codebook size; this holds compression fixed and intervenes only on utilization pressure |
| **Strongest reviewer rejection** | “GSQ Table 2 already swept entropy loss weights; CRT already shows intermediate optima; Entropy Cliff owns discrete entropy pathology.” |
| **Proposed kill-test** | Check entropy-weight ablations reporting reconstruction/generation metrics |
| **Minimum viable experiment** | Small VQGAN on CIFAR/ImageNet-32; λ∈{0,0.01,0.1,0.5}; train tiny AR/LDM stage-2; gFID |
| **Compute / data** | Prefer Colab; ImageNet-32 or CIFAR; days on T4 if stage-2 included |
| **Novelty level** | Aimed L2/L3; likely L1 after kill |
| **Why not ablation…** | Would be a causal λ intervention—but only novel if literature lacks the non-monotonic *generation* claim at fixed compression |

### Kill-test verdict — **KILL**

**Reason:** GSQ (2024) already ablates entropy-loss weights and shows **non-monotonic / harmful** effects on tokenizer metrics as weight increases; CRT shows intermediate λ optima; “Taming the Entropy Cliff” (2026) owns entropy-structure interventions for AR visual generation. Adjacent to exclusion **20** without a cleanly new object.  
**No exclusion reopen** (do not reopen 20).

---

## Candidate V3-4 — Modality lock-in time in joint audio–video diffusion

| Field | Content |
|-------|---------|
| **Scientific object** | **Modality lock-in time**: the trajectory time after which zeroing cross-modal attention no longer changes video (resp. audio) semantics |
| **Falsifiable claim** | In a joint AV generator, video and audio have **different** lock-in times; cutting cross-modal attention after \(t_v^\star\) but before \(t_a^\star\) harms audio more than video (or the reverse), reproducibly across seeds |
| **Closest 2025–2026 papers** | MDP / SWITCH / PCI (mid-trajectory **prompt** switches); Progressive Guidance; LTX-2 modality-CFG; UniAVGen MA-CFG; Dieleman spectral-AR view of diffusion; exclusion **35** (mid-trajectory SWITCH) |
| **Exact difference** | Intervening variable is **which modality stream** is cut, not which text prompt; object is asymmetric lock-in, not sync error (excl. **21**) or CFG scale recipes |
| **Strongest reviewer rejection** | “Coarse-to-fine is known; SWITCH already did trajectory interventions; LTX-2 already exposes per-modality guidance—this is an ablation of cross-attention schedules.” |
| **Proposed kill-test** | Confirm whether any 2025–2026 paper reports **per-modality lock-in / CIS-style curves** for joint AV |
| **Minimum viable experiment** | Prefer a **small proxy** first (synthetic AV or a lightweight public V2A model): along reverse trajectory, zero cross-modal attn after fraction α∈{0.2…0.8}; score video-only and audio-only metrics; estimate lock-in α |
| **Compute / data** | Full LTX-2-class models exceed “smallest credible”; proxy may fit Colab; full joint AV likely needs ≥1×A100 |
| **Novelty level** | L3 if lock-in asymmetry is real and not implied by existing CFG papers; else L2/L1 |
| **Why not ablation / framework / future-work** | Causal intervention defines a new measurable object (modality lock-in), not a new guider product; not completing one paper’s listed future work if framed as CIS-style **modality** curves |

### Kill-test verdict — **MODIFY** (not KEEP)

**Reason:** Trajectory conditioning interventions and modality-aware CFG are heavily occupied (SWITCH/MDP/PCI; LTX-2; UniAVGen). A bare “cut cross-attention over time” experiment will be read as cosmetic ablation.  

**Required reshape before any KEEP:**  
1. Define a **CIS-style lock-in curve per modality** with a fixed evaluation protocol.  
2. Show it is **not reducible** to “apply SWITCH twice.”  
3. Deliver one **falsifiable predictive claim** (e.g. lock-in gap \(t_a^\star-t_v^\star\) predicts sync failure under modality-CFG mismatch).  
4. Explicitly argue **no reopen of exclusion 35**—or, if the kill requires it, **flag reopen of 35** with written justification (owner approval).  

**Status:** strong **MODIFY** only—**no experiment plan / no implementation** until the claim is rewritten and re-kill-tested.

---

## Candidate V3-5 — Visual clutter as intervening variable for image→audio

| Field | Content |
|-------|---------|
| **Scientific object** | Visual scene complexity / object count as a causal driver of image-conditioned audio errors |
| **Falsifiable claim** | With text held fixed, increasing object count / clutter in the conditioning image monotonically increases wrong-event audio rate and CLAP mismatch |
| **Closest 2025–2026 papers** | Sounding that Object (multi-object I2A, object-aware generation); Common Cause, Not Cross-Attention (AV visual shortcuts, interventions); Art2Mus / I2M systems |
| **Exact difference** | Would isolate **clutter count** as a scalar intervening variable on a frozen public I2A model |
| **Strongest reviewer rejection** | “Sounding that Object already targets multi-object scenes; Common Cause already intervenes on visual nuisances—this is a subset.” |
| **Proposed kill-test** | Check multi-object / shortcut papers for clutter-style interventions |
| **Minimum viable experiment** | Public I2A model; control images with 1 vs N objects; audio event accuracy / FAD |
| **Compute / data** | Colab-possible for short clips; public AudioCaps-style data |
| **Novelty level** | L2 attempt |
| **Why not ablation…** | Intended as causal clutter law—but literature already treats multi-object and visual shortcuts |

### Kill-test verdict — **KILL**

**Reason:** Named/occupied—object-aware multi-source I2A (Sounding that Object, ICML 2025) and causal visual-shortcut interventions in AV generation (Common Cause, 2026) own the scientific neighborhood. Auto-KILL: named object / future-work-shaped measurement.  
**No exclusion reopen.**

---

## Round summary

| ID | Short title | Verdict |
|----|-------------|---------|
| V3-1 | Cross-paradigm class-rank transfer | **KILL** (measure/benchmark under v3 L2 bar; excl. 33-adj) |
| V3-2 | Block ablation identity vs appearance | **KILL** (Stable Flow / TexTailor / DiT localization) |
| V3-3 | VQ utilization λ @ fixed compression | **KILL** (GSQ / CRT / Entropy Cliff; excl. 20-adj) |
| V3-4 | AV modality lock-in time | **MODIFY** (reshape + re-kill; no implement yet) |
| V3-5 | Visual clutter → I2A error | **KILL** (Sounding that Object; Common Cause) |

**KEEP survivors this round: none.**  
**Implementation: none.**  
**Paper drafting: none.**

### Next step (process only)

1. Rewrite **V3-4** into a sharper falsifiable object (modality lock-in curve + predictive claim), **or** drop it.  
2. Generate a **new ≤5** wave targeting other v3 priority niches (causal representation interventions; multimodal with a *new* intervening variable not covered by AV-shortcut / modality-CFG papers)—still kill-test before coding.

### Additional exclusion notes (do not add unless KEEP fails into a stable neighborhood)

If V3-4-style trajectory modality cuts keep recurring without a new object, consider adding exclusion **40** (mid-trajectory **cross-modal** attention cutting / modality lock-in diagnostics) after owner review—**not added yet**.

---

## Round 2 (2026-09-23) — V3-4 rewrite + new ≤5

### V3-4R — Modality CIS / lock-in curves (rewrite of V3-4)

| Field | Content |
|-------|---------|
| **Scientific object** | Per-modality Concept Insertion Success (CIS) curves: probability that injecting / cutting a modality stream at timestep \(t\) still affects the final AV output |
| **Falsifiable claim** | Audio and video have different \((\tau_{50},\tau_{70})\) lock-in statistics; the gap \(\Delta\tau=\tau^{(a)}_{50}-\tau^{(v)}_{50}\) predicts sync/semantic failure under mismatched modality-CFG |
| **Closest papers** | **PCI** (ICLR 2026) — owns CIS for **text concepts**; LTX-2 / UniAVGen modality-CFG; MDP/SWITCH; ProgForcing async \(t_v,t_a\) |
| **Exact difference** | Replace text-concept PCI with **modality-stream** PCI |
| **Strongest rejection** | “PCI applied to a new conditioning channel; auto-KILL as framework application / same phenomenon, new modality.” |
| **Kill-test** | Does any paper already report CIS-style curves for AV modalities? |

#### Verdict — **KILL**

PCI already defines the CIS object and trajectory intervention protocol. Porting it from text concepts to AV modalities is **framework application** + **same phenomenon on another channel** (auto-KILL classes 2 and 5), and sits next to exclusion **35**. Predictive link to modality-CFG failure is future-work-shaped on top of LTX-2.  
**Drop V3-4 line.** Consider owner-approved exclusion **40** later if this neighborhood keeps recurring. **No reopen of 35 without explicit flag** (not requested).

---

### Candidate V3-6 — Continuous conditioning informativeness → memorization

| Field | Content |
|-------|---------|
| **Object** | Memorization ratio as a function of continuous label–image mutual information \(I(Y;X)\) |
| **Claim** | EMM / replica rate is non-monotonic (or sharply thresholded) in \(I(Y;X)\) between pure random and true labels |
| **Closest** | DiffMemorize (random ≈ true; unique labels trigger); BIRD info-theoretic mem/gen phase boundary |
| **Difference** | Continuous MI sweep vs binary random/true |
| **Strongest rejection** | “DiffMemorize already falsified ‘informative labels cause mem’; BIRD owns info phase laws; this is future-work completion.” |
| **Novelty** | L2 attempt |

#### Verdict — **KILL** (NAMED / FUTURE_WORK of DiffMemorize + BIRD)

---

### Candidate V3-7 — Class-embedding scale α ≠ CFG scale

| Field | Content |
|-------|---------|
| **Object** | Inference-time multiplier on class embeddings as a distinct control from CFG |
| **Claim** | α-optimal for class-acc differs from CFG-optimal for FID; the two knobs are not interchangeable |
| **Closest** | CFG (Ho & Salimans); guidance design-space papers; Diffusers CFG start/stop |
| **Difference** | Embedding gain vs score-space guidance |
| **Strongest rejection** | “Untrained embedding scaling is a brittle hack; CFG already is the calibrated knob; cosmetic ablation.” |
| **Novelty** | L1–L2 |

#### Verdict — **KILL** (cosmetic ablation / CFG neighborhood)

---

### Candidate V3-8 — Batch class-entropy \(H\) threshold for mode copying

| Field | Content |
|-------|---------|
| **Object** | Critical training-batch class entropy below which majority→minority mode copying begins |
| **Claim** | There exists \(H^\star\) such that for \(H<H^\star\), minority samples are nearest-neighbor copies of majority modes at fixed architecture |
| **Closest** | CBDM; DiffROP; CCUA / long-tailed diffusion; Capacity Manipulation; CORAL |
| **Difference** | Scalar \(H\) law vs proposing another balancer method |
| **Strongest rejection** | “Long-tail diffusion literature already documents collapse under imbalance; measuring \(H^\star\) is a benchmark of known failure.” |

#### Verdict — **KILL** (NAMED / MEASURE in long-tail diffusion neighborhood)

---

### Candidate V3-9 — Early-trajectory roughness predicts seed failure (class-cond CIFAR)

| Field | Content |
|-------|---------|
| **Object** | Training-free predictor of bad seeds from early score/velocity residual |
| **Claim** | Early \(\|v_{t+\Delta}-v_t\|\) (or score residual) ranks final class-acc / twin quality with Spearman ρ≫0 |
| **Closest** | **Probe-Select** (CVPR 2026); **Diffusion Probe**; **ABSS** core-token attention seed selection; InitNO; SharpEuler |
| **Difference** | Class-conditional CIFAR / residual roughness vs T2I attention probes |
| **Strongest rejection** | “Early quality assessment for diffusion seeds is a named 2026 object; CIFAR residual is a toy port.” |

#### Verdict — **KILL** (NAMED — Probe-Select / Diffusion Probe / ABSS). Adjacent to excl. **32** without a new mechanism.

---

### Candidate V3-10 — REPA improves FID but harms tail-class diversity (negative constraint)

| Field | Content |
|-------|---------|
| **Object** | Side-effect of representation alignment strength λ on **per-class / tail** generative diversity |
| **Claim** | ∃ λ where aggregate FID↓ but tail-class within-class diversity↓ or class-acc gap↑ — constraining “REPA is uniformly beneficial” |
| **Closest** | REPA (λ ablations are **aggregate** FID/IS only); CORAL (long-tail latent entanglement — different method) |
| **Difference** | Negative-result / constraint on REPA’s **distributional** effects, not a new aligner |
| **Strongest rejection** | “Ablation of REPA; CORAL already studies long-tail latents; expected side-effect.” |
| **Why allowed under v3** | Negative results OK if they **falsify/constrain** a meaningful claim |
| **Risk** | Still reads as REPA ablation unless the constraint is sharp and non-obvious |

#### Verdict — **KILL** (weak)

Under adversarial reading this is primarily an **ablation of a known mechanism** (REPA λ) without a new law that is clearly distinct from CORAL’s long-tail latent story. Fails v3’s “not merely cosmetic ablation” bar unless redesigned as a **comparative causal** study with a named interaction object (then risks auto-KILL class 6: REPA+CORAL combo). **No KEEP.**

---

## Round 2 summary

| ID | Verdict |
|----|---------|
| V3-4R | **KILL** (PCI framework port) |
| V3-6 | **KILL** (DiffMemorize / BIRD) |
| V3-7 | **KILL** (CFG ablation) |
| V3-8 | **KILL** (long-tail diffusion) |
| V3-9 | **KILL** (Probe-Select / Diffusion Probe / ABSS) |
| V3-10 | **KILL** (REPA ablation / CORAL-adj) |

**KEEP survivors after rounds 1–2: none.**  
**Implementation: none. Manuscript: none.**

### Process next (not implementation)

Round 3 should **leave** these saturated neighborhoods:

- trajectory CIS / seed early-quality probes  
- modality-CFG / AV shortcuts / clutter  
- tokenizer entropy / CRT  
- class-imbalance diffusion methods  
- CFG / embedding-scale knobs  
- DiffMemorize-style conditioning mem  

Prefer instead (still ≤5, kill before code): e.g. **audio-only** generative mechanisms with a new intervening variable; **controlled synthetic→real transfer** of a causal claim; **3D/view** interventions outside excl. 23’s opacity object; or a **negative result that falsifies a published scaling/transfer claim** with a decisive, non-ablation design.

---

## Round 3 (2026-09-23) — leave R1–R2 saturated zones

Avoided on purpose: trajectory CIS / seed probes; AV CFG/shortcuts/clutter; visual tokenizer/CRT; long-tail diffusion methods; CFG embedding knobs; DiffMemorize-style mem.

### V3-11 — T2A timing-control intensity vs acoustic fidelity

| Field | Content |
|-------|---------|
| **Object** | Timing-control strength (attention fusion / # timed events) as intervening variable for alignment–fidelity tradeoff |
| **Claim** | Stronger timing control improves Kendall-τ / timestamp accuracy but worsens FAD beyond a critical setting |
| **Closest** | FreeAudio (α/β ablations already state alignment↑ ⇒ fidelity↓); ControlAudio; PicoAudio; T2A-Feedback |
| **Difference** | Would elevate FreeAudio’s hyperparameter note into a “law” |
| **Strongest rejection** | “FreeAudio already reported this tradeoff.” |
| **Novelty** | L1 |

#### Verdict — **KILL** (NAMED — FreeAudio)

---

### V3-12 — Audio VAE bitrate / target-KL → non-monotonic generation

| Field | Content |
|-------|---------|
| **Object** | Audio latent compression rate as causal knob for downstream T2A quality |
| **Claim** | Intermediate bitrate maximizes FAD/CLAP; too low and too high both hurt generation |
| **Closest** | **Taming Audio VAEs via Target-KL** (2026) — owns rate–distortion + non-monotonic T2A; SALAD-VAE; GenAE; AudioLDM compression notes; visual CRT (excl. **20**) |
| **Difference** | None material — same scientific object in audio |
| **Strongest rejection** | “Already the 2026 Target-KL paper; CRT clone in audio.” |
| **Novelty** | L0–L1 |

#### Verdict — **KILL** (NAMED). Treat as **excl. 20 modality transplant** — do not reopen 20.

---

### V3-13 — Negative: physics “local FM ⇒ size transfer” fails for images

| Field | Content |
|-------|---------|
| **Object** | Cross-domain transferability of the published inductive bias “FM learns local laws ⇒ extrapolates in size” |
| **Claim** | Patch-local image FM trained at resolution \(r\) does **not** transfer to \(2r\) under the conditions where lattice FM does — falsifying indiscriminate citation of physics size-transfer claims |
| **Closest** | Flow Matching at Scale (physics); “When Do Local Score Models Extrapolate Across Size?” (diagnostic theory); FlowDCN / FiT arbitrary-resolution image FM; excl. **24** |
| **Difference** | Explicit **negative** on cross-domain slogan transfer |
| **Strongest rejection** | “Strawman — physics papers don’t claim images are Ising models; FlowDCN already studies image resolution extrapolation.” |
| **Novelty** | L2 negative-result attempt |

#### Verdict — **KILL**

Fails “constrains a meaningful claim”: the physics claim is domain-scoped; image resolution extrapolation is already a named architecture/methods neighborhood (FlowDCN, FiT, PPFlow). Negative result is against a caricature.

---

### V3-14 — Molecular generation: sampling top-p as intervening validity–novelty knob

| Field | Content |
|-------|---------|
| **Object** | Nucleus/top-p (or temperature) as causal control of validity vs scaffold novelty |
| **Claim** | Intermediate top-p maximizes a joint validity×scaffold-novelty score; extremes fail differently |
| **Closest** | MolHIT top-p ablations; MOSES; “How evaluation choices distort…”; SAFE fragmentation tradeoffs |
| **Difference** | Would restate known sampling tradeoffs as a “law” |
| **Strongest rejection** | “Standard decoding ablation on MOSES/MolHIT.” |

#### Verdict — **KILL** (ABLATION / NAMED)

---

### V3-15 — Novel-view baseline / trajectory length consistency cliff

| Field | Content |
|-------|---------|
| **Object** | Camera baseline angle or trajectory arc length as intervening variable for cross-view consistency collapse |
| **Claim** | Consistency metrics exhibit a sharp cliff past baseline \(b^\star\) for fixed monocular generative NVS models |
| **Closest** | TrajectoryCrafter (limits on large-range trajectories); ReCamMaster; CausNVS (sequence-length generalization); Memory-V2V FOV retrieval; excl. **23** (opacity decoupling — different object) |
| **Difference** | Scalar baseline cliff vs proposing another camera-control method |
| **Strongest rejection** | “Every 2025 NVS paper already reports degradation with wider baselines / longer trajectories.” |

#### Verdict — **KILL** (MEASURE / known limitation neighborhood). Not excl. 23 reopen.

---

## Round 3 summary

| ID | Verdict |
|----|---------|
| V3-11 | **KILL** (FreeAudio timing–fidelity) |
| V3-12 | **KILL** (Target-KL Audio VAE; CRT-audio) |
| V3-13 | **KILL** (strawman negative; FlowDCN owns image res. extrap.) |
| V3-14 | **KILL** (MolHIT/MOSES sampling ablations) |
| V3-15 | **KILL** (NVS trajectory/baseline limitations) |

**KEEP after rounds 1–3: none.**  
**Implementation: none. Manuscript: none.**

### Proposed new exclusions (for owner approval — not auto-applied)

These neighborhoods recur under v3; closing them prevents wasted rounds:

| # | Neighborhood |
|---|--------------|
| **40** | Mid-trajectory modality CIS / cross-modal attention cutting (PCI ports) |
| **41** | Early-trajectory seed/quality probes for diffusion (Probe-Select class) |
| **42** | Audio latent bitrate / target-KL compression–generation sweeps (CRT-audio) |
| **43** | T2A timing-control hyperparameter ↔ fidelity tradeoffs (FreeAudio class) |
| **44** | Camera baseline / trajectory-length consistency cliffs in generative NVS |

### Process note

Three v3 top-down-style waves (≈15 candidates) produced **0 KEEP**. That does **not** revive “GenAI is exhausted”; it means **these opportunity classes** are occupied. Round 4 should pick a **narrower causal object** with a pre-registered difference sentence that cannot be rewritten as “we ablated λ in paper X,” ideally grounded in a **small runnable stack** (Strategy B) in a modality where we can observe first—e.g. tiny symbolic music / spectrogram toy, or sklearn-scale graph generation—rather than another literature-first sweep of SOTA T2A/NVS/mol papers.

---

## Round 4 (2026-09-23) — kill “tempting Strategy-B” neighborhoods *before* coding

Goal: close the most seductive small-stack claim shapes so Round 5 observation digs do not rediscover named papers. Still ≤5; **no probes run this round**.

### V3-16 — Symbolic music: decoding temperature / nucleus ↔ structure–diversity

| Field | Content |
|-------|---------|
| **Object** | Sampling temperature / top-p as intervening structure vs diversity knob |
| **Claim** | Intermediate decoding maximizes a joint long-range structure × novelty score |
| **Closest** | CAST skeleton→texture; MusicLayout; FIGARO; SMER / MMM controllability; Diff-Symbo guidance notes |
| **Difference** | Would restate known control–diversity tensions as a “law” |
| **Strongest rejection** | “Decoding / guidance ablation on a crowded symbolic-music stack.” |
| **Novelty** | L1 |

#### Verdict — **KILL** (ABLATION / NAMED)

---

### V3-17 — Time-series diffusion: forecast horizon / terminal denoising step cliff

| Field | Content |
|-------|---------|
| **Object** | Horizon \(H\) or reverse-step index as intervening forecast quality variable |
| **Claim** | Quality is non-monotonic in reverse steps / collapses past \(H^\star\) |
| **Closest** | **When Denoising Hurts** (2026) — owns terminal-step drift + early stop; Stage-Diff; Dynamical Diffusion; GPD |
| **Difference** | None material |
| **Strongest rejection** | “Named 2026 TS diffusion finding; NFE-cliff cousin (excl. **31**-adj).” |
| **Novelty** | L0–L1 |

#### Verdict — **KILL** (NAMED)

---

### V3-18 — Point-cloud generation: test-time point count ↔ fidelity–variability

| Field | Content |
|-------|---------|
| **Object** | Inference resolution \(N_{\mathrm{pts}}\) as intervening fidelity–diversity knob |
| **Claim** | Scaling \(N\) beyond train resolution improves fidelity while reducing variability (CFG-analogue) |
| **Closest** | **PointInfinity** (CVPR 2024) — owns this exact tradeoff + CFG comparison; EMERGE (2026) resolution-agnostic PC diffusion |
| **Difference** | None |
| **Strongest rejection** | “PointInfinity’s central empirical claim.” |
| **Novelty** | L0 |

#### Verdict — **KILL** (NAMED)

---

### V3-19 — Palette / histogram entropy ↔ adherence–quality

| Field | Content |
|-------|---------|
| **Object** | Palette-histogram entropy / distance as intervening color-control strength |
| **Claim** | Intermediate entropy maximizes joint palette adherence and perceptual quality |
| **Closest** | **Palette-Adapter** (entropy + palette-to-histogram distance); Color Alignment in Diffusion; inference-time colour preservation (2026) |
| **Difference** | Would republish Palette-Adapter’s control axes as a discovery |
| **Strongest rejection** | “Those scalars are the method’s interface, already ablated.” |
| **Novelty** | L0–L1 |

#### Verdict — **KILL** (NAMED)

---

### V3-20 — Char-level / small-vocab masked diffusion: tokens-unmasked-per-step

| Field | Content |
|-------|---------|
| **Object** | Parallel unmask width \(k\) (or schedule shape) as intervening quality–speed variable |
| **Claim** | Intermediate \(k\) / non-linear schedule is uniquely optimal for tiny-vocab MDM |
| **Closest** | Optimal Inference Schedules for MDMs; Error Bounds & Optimal Schedules; EB-Sampler; P2 path planning; MD4/GenMD4; nanoDLM / tiny-diffusion educational stacks |
| **Difference** | Toy re-measurement of a theory-owned schedule object |
| **Strongest rejection** | “Schedule theory + multiple 2025 samplers already own computation–accuracy.” |
| **Novelty** | L1 at best; also leans **text** neighborhood exhausted under v1 |

#### Verdict — **KILL** (NAMED / MEASURE)

---

## Round 4 summary

| ID | Verdict |
|----|---------|
| V3-16 | **KILL** (symbolic music decoding / structure control) |
| V3-17 | **KILL** (TS diffusion terminal step / horizon — When Denoising Hurts) |
| V3-18 | **KILL** (PointInfinity point-count fidelity–variability) |
| V3-19 | **KILL** (Palette-Adapter entropy / distance) |
| V3-20 | **KILL** (MDM unmask schedule theory) |

**KEEP after rounds 1–4: none (~20 candidates).**  
**Implementation: none. Manuscript: none.**

### Round 5 directive (Strategy B stack v4 — observation first)

Do **not** lit-invent another claim. Freeze a **new runnable stack** outside exclusions **1–39** and proposed **40–44**, run **observation-only** probes (≥2 seeds), cluster failures, **then** kill-test.

| Piece | Choice (stack v4) |
|-------|-------------------|
| Task | Class-conditional **small graph** generation (non-molecular) |
| Data | Synthetic **Stochastic Block Models** (controllable; public code) + optional transfer check on a tiny real graph (e.g. karate / football) only if a claim survives |
| Model | Tiny GraphVAE or simple discrete edge-diffusion (CPU/Colab) |
| Probe themes (observation only — no novelty claim until kill-test) | (G1) train assortativity / community strength sweep; (G2) train graph-size mismatch at sample time; (G3) edge-density mismatch; (G4) community-label noise; watch validity, degree MMD, assortativity error, mode collapse |
| Anti-promote | Edge-independent triangle bounds restatements (Chanpuriya); molecular validity–novelty (excl. mol / V3-14); MAG/DiGress architecture races |

If stack v4 observations only recover named graph-GenAI limitations → **KILL cluster**, pick stack v5, or owner chooses `PUBLICATION_STATUS` paths 2–4.

---

## Round 5 (2026-09-23) — Strategy B stack v4 observations → kill

**Stack:** SBM (`n=20`, 2 blocks) → upper-tri adjacency → tiny MLP GraphVAE (BCE+KL).  
**Logs:** `experiments/logs/graph_v4_seed{0,1}.jsonl` (quick: 40 epochs, 200 graphs, 32 samples).  
**Script:** `experiments/scripts/run_probe_graph_v4.py`

### Observed clusters (repro seeds 0–1)

| ID | Observation | Verdict |
|----|-------------|---------|
| **G1-sparse** | Weak/moderate SBM (`p_in`≤0.35) → **empty-graph collapse** (gen density 0, 20 components) | **KILL** |
| **G1-strong** | Strong SBM (`p_in`≥0.55) → near-**two-clique mode collapse** (planted modularity ≈0.5, pairwise Hamming ≈0–0.05, density overshoot) | **KILL** |
| **G3** | ER density ≤0.35 → empty collapse; `p=0.5` roughly matches density | **KILL** (same sparse-decoder object as G1-sparse) |
| **G4** | Post-hoc edge-flip noise monotonically destroys planted modularity | **KILL** (trivial corruption measure; not a generative law) |

### Candidate claims considered (then killed)

| Claim sketch | Why killed |
|--------------|------------|
| “Community strength is a causal knob for GraphVAE fidelity–diversity” | Restates known adjacency-Bernoulli GraphVAE limits (Simonovsky & Komodakis GraphVAE; GRL book §9; MAG notes VAEs weak on structure; kernel-regularized GVAE exists *because* independent edges miss global structure) |
| “There is a critical train density below which GraphVAE emits empty graphs” | Engineering failure of independent-edge decoder / posterior collapse neighborhood — not a new intervening-variable law |
| “Edge noise ↔ modularity cliff” | Not generative; trivial |

**KEEP after Round 5 stack v4: none.**  
**Proposed exclusion 45** (pending owner): adjacency-MLP GraphVAE density / community mode-collapse on SBM as novelty.

### Next

Stack **v5** must leave graph-VAE Bernoulli-decoder failure modes. Prefer a stack where the *decoder inductive bias is not the already-known bug* (e.g. tiny equivariant / degree-guided graph diffusion only if a **new** intervening variable appears in logs; or a non-graph modality not closed by rounds 1–4). Still: observe → cluster → kill → no paper.

---

## Round 6 (2026-09-23) — Strategy B stack v5 observations → kill

**Attempted:** Fashion-MNIST CVAE — **aborted** (torchvision download stalled ~8+ min; process killed).  
**Stack used:** sklearn Digits CVAE (offline), probes ≠ β (excl. **39**).  
**Logs:** `experiments/logs/digits_v5_seed{0,1}.jsonl`  
**Script:** `experiments/scripts/run_probe_digits_v5.py`

### Observed clusters (repro seeds 0–1)

| ID | Observation | Verdict |
|----|-------------|---------|
| **F1** | ↑ random-erase → ↓ sample mean intensity; class-acc stays ~1.0; diversity flat | **KILL** (aug ablation / no law) |
| **F2** | ↑ prior scale τ → ↑ pairwise L2 diversity (≈0.17→0.78); class-acc stays ≈1 until τ=2.5 | **KILL** (VAE prior/temperature / truncated-prior neighborhood) |
| **F3** | per-class cap=20 → class-acc collapse (0.30–0.55); ≥50 → acc≈1, diversity↓ with more data | **KILL** (few-shot GenAI / data-scale; D2C-class) |

**KEEP after stack v5: none.**  
**Proposed exclusion 46:** VAE prior-scale τ diversity knob; **47:** few-shot / per-class-N GenAI accuracy cliffs (pending owner).

Publication still blocked. Next: stack **v6** outside F1–F3 and prior exclusions — or owner paths 2–4 in `PUBLICATION_STATUS.md`.

---

## Round 7 (2026-09-23) — stack v6 GMM-DDPM + meta-path kill

### Stack v6 — synthetic 2D GMM + tiny MLP DDPM (offline)

**Logs:** `experiments/logs/gmm_v6_seed{0,1}.jsonl`  
**Script:** `experiments/scripts/run_probe_gmm_v6.py`

| ID | Observation | Verdict |
|----|-------------|---------|
| **M1** | Mode separation δ ∈ [0.4,2.5]: full coverage both seeds | **no claim** (no failure) |
| **M2** | K ∈ {2..8}: full coverage | **no claim** |
| **M3** | Imbalance ρ↑ → assign-entropy↓; ρ≥10 → uncovered modes (repro) | **KILL** |

**M3 kill reason:** Classic multimodal coverage failure under imbalance — GMM diffusion theory / mode-coverage literature (e.g. guidance-on-GMM analyses; GMP mode-coverage work; CBDM-adjacent long-tail story for discrete labels). Not a new intervening-variable law.

### V3-21 — Meta / negative-results paper on this kill-log (Path 3)

| Field | Content |
|-------|---------|
| **Object** | Constrained solo GenAI topic search with adversarial kill-tests as a publishable negative result |
| **Claim** | Under explicit novelty+exclusion rules, ~20 lit candidates + stacks v0–v6 yield 0 KEEP; kill-log is a reusable artifact |
| **Closest** | Why LLMs Aren’t Scientists Yet (2026); Can AI agents conduct open-ended AI research? (shadow evals); AutoResearchEval / ARFT 45 failure patterns (2026); Agents4Science negative-result accepts |
| **Difference** | Human+rules kill-log vs LLM-agent failure taxonomies |
| **Strongest rejection** | “Process negative results / agent-research failure is already a 2025–2026 named neighborhood; without a new measurable protocol claim this is a lab notebook.” |
| **Novelty** | L1–L2 borderline |

#### Verdict — **KILL** (NAMED / MEASURE) under current v3 bar  
Would need Path **2** (explicit L1/process-paper allowance) or a sharper protocol contribution (e.g. falsifiable stop-rule with external validation) to reopen.

**KEEP after Round 7: none.**  
**Proposed exclusion 48:** GMM/mode-imbalance coverage cliffs for toy diffusion as novelty.

### Status

Publication **blocked**. Honest options remain `PUBLICATION_STATUS` paths **1** (continue dig), **2** (relax), **3** (meta — needs reframing beyond V3-21), **4** (stop).

---

## Round 8 (2026-09-23) — Strategy B stack v7

**Stack:** Synthetic class-conditional spectrogram patches (harmonic templates) → tiny CVAE. Offline.  
**Logs:** `experiments/logs/spec_v7_seed{0,1}.jsonl`  
**Script:** `experiments/scripts/run_probe_spec_v7.py`

| ID | Observation | Verdict |
|----|-------------|---------|
| **S1** | Train SNR↓ → sample std↓; class-acc stays 1.0 | **KILL** (noisy-train / data-quality; denoising-AE neighborhood; auto-KILL class 5 on synthetic) |
| **S2** | f0 jitter↑ → class-acc collapse at high jitter (0.86 / 0.725 @ 40 Hz, repro) | **KILL** (attribute/label-noise robustness; excl. **36**-adj) |
| **S3** | Time-warp ∈ [0,0.5]: flat metrics | **no claim** |

**KEEP: none.** Proposed excl. **49:** train-SNR / attribute-jitter sweeps on toy audio-like GenAI.

### Path-1 EV note

Lit rounds 1–4 + stacks v0–v7 + V3-21: **0 KEEP**. Further Path-1 stacks under current v3 bar are expected to rediscover named neighborhoods. Continuing requires either a qualitatively new stack class or an explicit Path **2** constraint change.

---

## Round 9 (2026-09-23) — stack v8 SEM–ATE + Path-1 terminal

**Stack:** Synthetic SEM `[X,A,Y]` → unconditional tabular CVAE → OLS ATE on samples.  
**Logs:** `experiments/logs/sem_v8_seed{0,1}.jsonl`  
**Script:** `experiments/scripts/run_probe_sem_v8.py`

| ID | Observation | Verdict |
|----|-------------|---------|
| **C1** | True τ↑ → synth ATE lags; large `ate_abs_err` despite finite recon MSE (repro) | **KILL** |
| **C2** | ↑ z_dim ↓ recon MSE but ATE error stays large | **KILL** |
| **C3** | ↑ N does not reliably fix ATE error | **KILL** |

**Kill reason:** Named pitfall — joint generative objectives preserve observational fit while distorting treatment-effect contrast (**Generative Synthetic Data for Causal Inference**, 2026); excl. **38**-adjacent tabular synth neighborhood.

**KEEP: none.** Proposed excl. **50:** joint-row tabular synth ATE distortion sweeps.

### Path-1 closed for practical purposes

See [`docs/TERMINAL_FINDING_V3.md`](TERMINAL_FINDING_V3.md). Further `proceed` under Path 1 without Path **2** override is not expected to unlock a venue paper.

---

## Path 3 fork — recon fidelity ↔ causal estimand (2026-09-23)

Working question adversarial-checked in [`PATH3_CAUSAL_BOUNDARY.md`](PATH3_CAUSAL_BOUNDARY.md).

| ID | Verdict |
|----|---------|
| P3-C1 | **KILL** (Amad 2026 Prop.3.1 / Thm.3.1) |
| P3-C2 | **KILL** (no \(R^\star\); contradicts Amad; v8 C2 anti-evidence) |
| P3-C3 | **KILL** (STEAM + Amad eval message) |

**Survivors: 0. Path-3 (this question) exhausted.** Do not draft; do not add experiments for these claims.

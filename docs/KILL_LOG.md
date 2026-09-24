# Kill log — adversarial novelty archive

Factual archive of candidates rejected during the topic-search phase.  
**Not** a literature review paper. Paper IDs are those used during kill-tests; verify before citing.

Kill reason codes: `NAMED` | `FRAMEWORK` | `ABLATION` | `FUTURE_WORK` | `CLEANER_SYNTH` | `COMBO` | `CLONE` | `CROWDED` | `EXCLUDED_NBHD`

---

## Round 0 — Initial candidates

| ID | Topic (short) | Decision | Reason | Closest anchors (examples) |
|----|---------------|----------|--------|----------------------------|
| C1 | PEFT/UQ drift + recalibration | KILL | NAMED / CROWDED | UQ4CT, TAD, semantic entropy, SAR |
| C1-lite | Diagnostic-only remnant | ARCHIVED | weak | — |
| C13 | Confidence–consistency selective gen | KILL | NAMED / CROWDED | selective generation, disagreement signals |
| C3 | Related selective/UQ backup | KILL | NAMED / CROWDED | — |
| N1 | Associated-hallucination + minimal evidence | KILL | FUTURE_WORK / FRAMEWORK | AH taxonomy; RAG verification |
| N2 | Related hallucination/evidence | KILL | CROWDED | — |
| N3 | Small-RAG conflict policy | KILL | NAMED | FRANQ and RAG-conflict line |

---

## Reboot landscape — D-series

| ID | Topic (short) | Decision | Reason |
|----|---------------|----------|--------|
| D1 | Constrained/structured decoding eval | KILL | ABLATION / FRAMEWORK (DCCD neighborhood) |
| D3 | Speculative / decoding quality tradeoff | KILL | NAMED / CROWDED |
| D4 | Synthetic curriculum / diversity predictor | KILL | CROWDED |
| D5 | Related constrained-decoding | KILL | ABLATION |
| D6 | Classical SE→LLM / Colab-as-novelty class | KILL | FRAMEWORK / EXCLUDED_NBHD |

---

## Reboot #2 — M-series (mechanism)

| ID | Topic (short) | Decision | Reason |
|----|---------------|----------|--------|
| M1 | MDM / remasking / SER neighborhood | KILL | NAMED / EXCLUDED_NBHD |
| M2 | Lazy→rich LoRA / related PEFT geometry | KILL | NAMED / COMBO |
| M4 | TinyLoRA / feature learning rate | KILL | NAMED |
| M5 | Related remasking independence | KILL | FUTURE_WORK / NAMED |

Hard exclusion after this round: entire **MDM / SER / remasking / planner–sampler** neighborhood.

---

## Reboot #3 — R3-series

| ID | Topic (short) | Decision | Reason |
|----|---------------|----------|--------|
| R3-1 … R3-5 | Interactive state / KV / equilibrium / related | KILL | NAMED / COMBO / CROWDED |

Saturated clusters noted: interactive state integrity, gen–rec asymmetry, tokenization-as-binding, multitask gradient geometry, retokenization path-patching.

---

## Reboot #4 — R4-series

| ID | Topic (short) | Decision | Reason |
|----|---------------|----------|--------|
| R4-1 … R4-4 | Follow-on mechanism dualisms | KILL | NAMED / COMBO |

---

## Reboot #5 — Morphology / phonology

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R5-1 | Frequency × complexity morphological productivity | KILL | CLEANER_SYNTH / NAMED | Petty productivity; SIGTYP 2025; Spanish morphome; PNAS analogy |
| R5-2 | Forecasted-trigger allomorphy | KILL | CLONE | arXiv:2609.04708 |
| R5-3 … R5-5 | Not warranted / not tested | ABANDONED | — | — |

---

## Reboot #6 — Unlearning / quantization

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R6-1 | Skill unlearning vs fact unlearning | KILL | NAMED | Skill Unlearning (2503.21730); Tool Unlearning; FLU; PISCES; CLUE |
| R6-2 | Computation vs representation collapse (PTQ) | KILL | CLONE / COMBO | arXiv:2604.19884 |
| R6-3 … R6-5 | Not tested | ABANDONED | — | — |

---

## Reboot #7 — Interference

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R7-1 | Dual-procedure / typed interference | KILL | NAMED | Circuit-Interference Law (2603.06923); Huang capacity–interference (2605.29548); Ortho-LoRA; LoRI; OSRM |
| R7-2 | CoT commitment as hardness law | KILL | CLONE | Decorative→load-bearing CoT (2609.25366) |

---

## Reboot #8 — Prefill / curriculum

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R8-1 | Prefill vs decode regimes | KILL | NAMED | Notes-at-prefill (2606.17107); SPEED; serving P/D literature |
| R8-2 | Training order → which circuit | KILL | NAMED | Curricula→circuits (2607.04846); Order Is The Message; multi-hop curriculum |

---

## Reboot #9 — Recency

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R9-2 | Exposure-recency vs content circuits | KILL | NAMED / CLONE | Fresh in Memory (2509.14223, ICLR 2026); PTC/TAS |
| R9-1,3,4 | Not tested after R9-2 death | ABANDONED | — | — |

---

## Reboot #10 — Attribution / consolidation

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| R10-1 | Loss/attribution mediates capability consolidation | KILL | NAMED / COMBO | Capability Provenance; TokenUnlearn; IF-GUIDE; Huang; PRISM; PiKE |
| R10-2 | Seen/unseen = recency axis | SKIPPED | already claimed by Fresh in Memory | — |

---

## Reboots #11–#14 — Scout waves (immediate kills)

All scouts below died on contact with titled 2025–2026 owners (representative):

| Scout | Decision | Anchors |
|-------|----------|---------|
| Context stickiness | KILL | 2604.23371 |
| Schema ↔ Binding ICL | KILL | 2512.17325 |
| Procedural hallucination 2A/2B | KILL | 2602.19239; Legible Failures |
| Context vs memory authority | KILL | 2609.00753 |
| Variable binding / rebinding | KILL | Binding-ID line; 2505.20896; rebinding |
| Entropy / trajectory UQ | KILL | Entropy-Lens; HEAD ENTROPY; Truth as Trajectory |
| Teacher-force vs free-run | KILL | exposure bias / drift literature |
| Tokenisation fertility / bias | KILL | Causal tokenisation bias; BPE morphology papers |
| Manifold vs discrete logic | KILL | counting manifolds; geometric discrete logic |
| CLM vs MDM mechanism shift | KILL | 2601.14758 (+ MDM exclusion) |
| Gradient-Causal Gap | KILL | 2602.01442 |
| Represented ≠ computed | KILL | 2605.22488 |
| Rank collapse / Post-Norm | KILL | 2608.09417 |
| Attention sinks necessity | KILL | ACL 2026 short / 2603.11487 |
| Copy suppression | KILL | classic negative-head work |
| Self-repair / CoAx | KILL | Hydra; GIM; 2607.01940 |
| Superposition interference | KILL | 2602.04718; SAE recovery |
| Protoreasoning / Dyck CoT | KILL | 2608.04980 |

---

## Survivors

**None.**

## Hard exclusion neighborhoods (accumulated)

Do not re-propose topics inside:

1. UQ / hallucination detection / selective generation  
2. RAG-conflict policies  
3. Decoding-only / speculative tradeoffs (as novelty)  
4. Synthetic-collapse predictors (measure-only)  
5. Classical SE→LLM transfer; Colab-as-novelty  
6. MDM / SER / remasking / planner–sampler  
7. Interactive state integrity / judge-Goodhart cluster  
8. Gen–rec asymmetry; tokenization-as-binding; Ortho-LoRA / multitask geometry  
9. Morphology frequency×complexity productivity designs  
10. Phonological forecast-allomorphy  
11. Skill vs knowledge unlearning; PTQ two-failure-modes  
12. Circuit-Interference Law; capacity–interference dualisms  
13. Prefill–decode notes; curriculum→circuit; Fresh-in-Memory recency  
14. ICL schema/binding; context stickiness; authority; binding/rebinding  
15. Trajectory entropy UQ; sinks; self-repair; superposition-interference method papers  

---

---

## Constraint Set v2 — Kill round 1 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| V2-1 | Timestep × data-complexity curriculum interaction | KILL | COMBO / CROWDED | SpeeD; Data Warmup (+REPA orthogonality); CLTS; Denoising Task Difficulty Curriculum (ICLR 2025); Patch Forcing (CVPR 2026) |
| V2-2 | Video physics vs temporal-compositionality dissociation | KILL | MEASURE_ONLY / NAMED axes | VideoPhy (SA⊥PC by design); TC-Bench; VBench consistency↔dynamic tradeoff; VBench-2.0 Physics axis |
| V2-3 | Controllable T2I conflict-regime map | KILL | NAMED / FUTURE_WORK | Uni-ControlNet App.B priority ranking; SemanticControl; SmartControl; Cross-ControlNet |
| V2-5 | Symbolic music structural-drift law | KILL | NAMED / CLONE | ACG feature-drift; CAST structural error vs length; SMDIM late drift; Yin-Yang / MusicLayout |
| V2-6 | Small-compute visual-tokenizer compression flip | KILL | NAMED | CRT / “When Worse is Better” (NeurIPS 2025) owns small-model + compute scaling tradeoff |

### v2 round 2 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| V2-9 | AV generative tempo/event desync (non-lip) | KILL | CROWDED / NAMED | JavisDiT; SyncDPO; Syncphony; MTV; SynthSync structural AV failures |
| V2-10 | Text→image→video cascade error amplification | KILL | NAMED / FUTURE_WORK | EVS encapsulated T2I↔T2V; MLLM mid-gen correction; GENMAC; cascade surveys |
| V2-11 | 3DGS appearance–geometry decoupling under sparse views | KILL | NAMED | Geometry Gaussians; ICO-GS; RT-Splatting |
| V2-12 | Flow-matching train–test schedule/trajectory mismatch | KILL | NAMED | FlowSteer / PeRFlow trajectory mismatch; few-step FM scheduler literature |
| V2-13 | Multilingual T2I script/layout binding failures | KILL | NAMED / MEASURE | MultiTextEdit; IMTBench; UniTranslator; Qwen-Image-2.0 multilingual typography bottlenecks |

### v2 round 3 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| V2-14 | Synthetic TS/tabular help-vs-hurt forecasting regimes | KILL | NAMED | *Does Synthetic Data Help?* (arXiv:2605.06032) owns architecture-conditional help/hurt; DynLMC; ReGeN; Chronos mix surveys |
| V2-15 | Document/UI layout reading-order & hierarchy binding | KILL | CROWDED | RT-DocLayout; parser structural refinement; DesignCoder; MLS hierarchy IR |
| V2-16 | Code↔UI structural executability vs visual fidelity law | KILL | NAMED | Figma2Code fidelity/responsiveness/maintainability; WidgetGen; RubSE; UI2Code^N |
| V2-17 | Continual generative mode forgetting | KILL | NAMED / CROWDED | CLoG; Continual Consistency Diffusion (IKC/UKC/LKC); generative distillation |
| V2-18 | PDE surrogate conservation-law violation object | KILL | NAMED | ProbConserv; PCFM; PMFM; SNAP-FM; structure-preserving GNN solvers |

### Strategy B — observation clusters (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| FC1 | Low-NFE quality cliff (steps 10→50) | KILL | MEASURE / known | DDIM NFE–quality tradeoff; few-step covariance / aDDIM literature |
| FC2 | Within-class twin asymmetry at low NFE | KILL | MEASURE / NAMED-adj | Independent noise draws + few-step mean-collapse; Pfrommer et al. Schedule Deviation (class-varying path inconsistency); CADS/CFG diversity literature |
| FC3 | Organic classes degrade earlier than vehicles | KILL | NAMED-adj | Schedule Deviation varies by class (Pfrommer Fig.5); Patch Forcing difficulty heterogeneity |
| FC4 | Fixed-noise × class-label swap: label dominates coarse identity | KILL | MEASURE / known | Standard class-conditional diagnostic; InitNO / seed–condition split literature |
| FC5 | Similar-class swaps (cat↔dog, auto↔truck): layout from noise, semantics from label | KILL | NAMED-adj | MDP path switch; SWITCH concept blend; Progressive Guidance cross-class features |
| FM1 | FM Euler steps 5→20 quality cliff (uncond CIFAR CFM) | KILL | MEASURE / known | Same NFE–quality object as FC1; ODE solver hierarchy papers (Euler/Heun/RK4/DOPRI); SharpEuler; Flow Map Matching |
| T3 | Wine CVAE pairwise corr sign flips under class-conditional synth | KILL | MEASURE / NAMED | CTGAN/TVAE correlation distortion; TabSynDex; IJACSA 2024 TVAE/CTGAN analysis; CTAB-GAN+ rare-mode literature |
| D1 | Digits CVAE: ↑β → within-class diversity↓, class-acc↑ (seeds 0–1) | KILL | NAMED | Classic β-VAE capacity / diversity–fidelity tradeoff (Higgins et al.); conditional posterior-collapse literature |

### Constraint Set v3 — discovery round 1 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| V3-1 | Cross-paradigm class-difficulty rank transfer (DDPM↔FM) | KILL | MEASURE / benchmark under v3 L2 bar | Pfrommer Schedule Deviation; DC-FM; Patch Forcing; excl. 33-adj |
| V3-2 | Inference block ablation → class identity vs appearance | KILL | NAMED | Stable Flow; TexTailor / MMDiT blocks; Localizing Knowledge in DiTs |
| V3-3 | VQ utilization entropy λ @ fixed compression → non-monotonic gFID | KILL | NAMED / excl.20-adj | GSQ entropy ablations; CRT; Taming the Entropy Cliff; FVQ 100% util |
| V3-4 | Joint AV modality lock-in time (cut cross-modal attn over t) | MODIFY | CROWDED / SWITCH–CFG-adj | MDP/SWITCH/PCI; LTX-2 modality-CFG; UniAVGen MA-CFG — needs sharper predictive object |
| V3-5 | Visual clutter intervening variable for image→audio | KILL | NAMED | Sounding that Object; Common Cause (AV visual shortcuts) |

### Constraint Set v3 — discovery round 2 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| V3-4R | Modality CIS / AV lock-in curves (PCI port) | KILL | FRAMEWORK / same-phenomenon | PCI (ICLR 2026); LTX-2; SWITCH; excl. 35-adj |
| V3-6 | Continuous label MI → diffusion memorization | KILL | NAMED / FUTURE_WORK | DiffMemorize; BIRD phase boundary |
| V3-7 | Class-embedding scale α vs CFG | KILL | ABLATION | CFG design space; Diffusers guidance |
| V3-8 | Batch class-entropy H* for mode copying | KILL | NAMED / MEASURE | CBDM; DiffROP; CCUA; Capacity Manipulation; CORAL |
| V3-9 | Early trajectory roughness → seed failure | KILL | NAMED | Probe-Select; Diffusion Probe; ABSS |
| V3-10 | REPA λ helps FID but harms tail diversity | KILL | ABLATION / CORAL-adj | REPA λ tables; CORAL long-tail latents |

### Constraint Set v3 — discovery round 3 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| V3-11 | T2A timing-control ↔ fidelity tradeoff | KILL | NAMED | FreeAudio α/β ablations |
| V3-12 | Audio VAE bitrate non-monotonic T2A | KILL | NAMED / CRT-audio | Target-KL Audio VAEs (2026); SALAD-VAE; excl. 20 transplant |
| V3-13 | Physics FM size-transfer slogan fails on images | KILL | STRAWMAN / NAMED | Local-score size-extrapolation theory; FlowDCN/FiT |
| V3-14 | Molecular top-p validity–novelty knob | KILL | ABLATION | MolHIT; MOSES; eval-distortion papers |
| V3-15 | NVS baseline/trajectory-length consistency cliff | KILL | MEASURE / known | TrajectoryCrafter; CausNVS; ReCamMaster |

**v3 round-3 KEEP survivors: None.**  
**Proposed exclusions 40–44** listed in `DISCOVERY_V3.md` (pending owner approval before writing into the hard list).

### Constraint Set v3 — discovery round 4 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| V3-16 | Symbolic music temp/top-p ↔ structure–diversity | KILL | ABLATION / NAMED | CAST; MusicLayout; FIGARO; SMER |
| V3-17 | TS diffusion horizon / terminal denoising cliff | KILL | NAMED | When Denoising Hurts (2026); Stage-Diff; excl. 31-adj |
| V3-18 | Point-cloud \(N_{\mathrm{pts}}\) fidelity–variability | KILL | NAMED | PointInfinity; EMERGE |
| V3-19 | Palette histogram entropy ↔ adherence–quality | KILL | NAMED | Palette-Adapter; Color Alignment |
| V3-20 | MDM tokens-unmasked-per-step / schedule | KILL | NAMED / MEASURE | MDM schedule theory; EB-Sampler; P2; MD4 |

**v3 round-4 KEEP survivors: None.**  
Next: Strategy B **stack v4** (SBM graph GenAI) — observe then kill (`DISCOVERY_V3.md`, `STRATEGY_B.md`).

### Constraint Set v3 — Strategy B stack v4 observations (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| G1-sparse | SBM GraphVAE empty collapse at low \(p_{\mathrm{in}}\) | KILL | NAMED / known decoder limit | GraphVAE; kernel-GVAE; MAG vs VAE |
| G1-strong | Strong SBM → two-clique mode collapse (Hamming≈0) | KILL | NAMED / mode collapse | GraphVAE Bernoulli decoder; community VGAE collapse lit |
| G3 | ER density cliff → empty graphs | KILL | same as G1-sparse | — |
| G4 | Edge-flip ↔ modularity drop | KILL | MEASURE / trivial | — |

**Stack v4 KEEP survivors: None.** Proposed excl. **45** in `DISCOVERY_V3.md`.

### Constraint Set v3 — Strategy B stack v5 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| F1 | Digits CVAE train random-erase sweep | KILL | ABLATION / no law | Data-aug for generative models |
| F2 | Prior scale τ ↔ sample diversity | KILL | NAMED | Resampled/truncated VAE priors; VAE guidance diversity–fidelity |
| F3 | Per-class N ↔ class-acc cliff | KILL | NAMED / MEASURE | Few-shot GenAI (e.g. D2C); data-scale |

Fashion-MNIST attempt aborted (download stall). **Stack v5 KEEP: none.** Proposed excl. **46–47**.

### Constraint Set v3 — stack v6 + V3-21 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| M1 | GMM mode separation δ | no claim | no failure in quick regime | — |
| M2 | GMM mode count K | no claim | full coverage | — |
| M3 | GMM mode-weight imbalance ρ | KILL | NAMED mode coverage | GMM diffusion theory; GMP; CBDM-adj |
| V3-21 | Meta kill-log / Path-3 process paper | KILL | NAMED / MEASURE | AI Scientist failures; AutoResearchEval/ARFT; shadow evals |

**KEEP: none.** Proposed excl. **48**.

### Constraint Set v3 — stack v7 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| S1 | Synth-spec CVAE train SNR | KILL | data-quality / denoise-AE; class 5 | SALAD-VAE degradations; noisy-train GenAI |
| S2 | Class f0-jitter ↔ acc cliff | KILL | attribute noise; excl.36-adj | Label-noise diffusion; pitch-aug lit |
| S3 | Time-warp sweep | no claim | flat | — |

**Stack v7 KEEP: none.** Proposed excl. **49**.

### Constraint Set v3 — stack v8 (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| C1 | SEM τ vs synth ATE error | KILL | NAMED | Generative Synthetic Data for Causal Inference (2026); excl.38-adj |
| C2 | z_dim ↑ recon ↓ but ATE still wrong | KILL | same | — |
| C3 | N ↑ does not fix ATE | KILL | same | — |

**Stack v8 KEEP: none.** Proposed excl. **50**. Path-1 terminal: `TERMINAL_FINDING_V3.md`.

### Path 3 — causal recon↔ATE boundary (2026-09-23)

| ID | Claim | Decision | Reason | Anchors |
|----|-------|----------|--------|---------|
| P3-C1 | Finite recon ≠ ATE for joint synth | **KILL** | NAMED | Amad 2026 Prop.3.1 / Thm.3.1 |
| P3-C2 | Critical recon threshold \(R^\star\) | **KILL** | contradicts Amad; v8 C2 anti-evidence | Amad; STEAM |
| P3-C3 | Recon-ranked selection regret | **KILL** | MEASURE / STEAM+Amad eval message | STEAM UPEHE/JSDπ |

**Path-3 (this fork) survivors: None.** Dossier marked **TERMINAL** 0/3 — `PATH3_CAUSAL_BOUNDARY.md`. No further recon↔ATE work.

### Path 2 — autonomous external funnel (2026-09-23)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| P2-1 | Off-target drift Assoc→D_k law (general T2I) | **KILL** | NAMED | CIB-Med; ALE-Edit; ATDEdit |
| P2-2 | MoE-DiT selective deadlock U-curve as necessary law | **KILL** | NAMED | UniGen-MOE 2605.19378; ProMoE |
| P2-3 | Critical Doob plug-in k / damping for reward hacking | **KILL** | NAMED / ABLATION | Are-we-really-tilting 2606.02884; ArtifactReward |
| P2-4 | CFM–NLL residual → alignment regret | **KILL** | FUTURE_WORK | Han et al. 2608.28010 |
| P2-5 | Concept-rectification stage as binding intervening var | **KILL** | ABLATION / NAMED | CoBind; Rectify-Then-Diffuse |

**Path-2 survivors: None.** Dossier: `PATH2_DISCOVERY.md`. Intake closed. No `PATH2_SURVIVOR.md`.

### Ownership Wave-2 (Constraint Set v4, 2026-09-23/24)

| ID | Topic (short) | Decision | Reason | Closest anchors |
|----|---------------|----------|--------|-----------------|
| W2-1 | Likelihood top-k false stability / rank ordering | **MODIFY→KEEP** | W₂ dissociation falsified; minority rank order supported on GMM | MAD; Feng; LSF |
| W2-2…W2-9 | See `PATH2_DISCOVERY.md` Wave 2 | **KILL** | Named / crowded / prior Path-2 | — |

Paper: `paper/PAPER.md`. Experiment: `w2_fs`.

### Path 4 — archive deliverable (2026-09-23)

| ID | Action | Decision |
|----|--------|----------|
| P4 | Stop; treat kill-log + frozen stacks + path terminals as deliverable | **COMPLETE** — `PATH4_ARCHIVE.md` |

No KEEP. No manuscript authorized.

### Additional exclusions

(Exclusions **1–30** unchanged above in this file’s earlier sections.)

31. Few-step / low-NFE quality cliffs as standalone novelty  
32. Within-condition sample twin asymmetry / seed-quality lottery at low NFE  
33. Class-varying conditional Schedule Deviation / denoising inconsistency (Pfrommer neighborhood)  
34. Fixed-noise class-label swap / condition-vs-noise layout split as novelty  
35. Mid-trajectory condition SWITCH / MDP-style path editing as novelty  
36. Train-time label-noise robustness for conditional diffusion (TDSM / SBDC neighborhood)  
37. Flow-matching / CFM Euler (or higher-order) step-count quality cliffs as novelty  
38. Tabular synth correlation / dependency distortion & rare-mode collapse as novelty (CTGAN/TVAE/CTAB-GAN+ neighborhood)  
39. β-VAE / KL-weight diversity–fidelity (or class-acc) tradeoff as novelty  

**v3 round 1:** no new exclusion number added.  

---

## How to use this file

- Historical record for the project owner  
- Guardrail against re-proposing killed objects under new names  
- **Not** a substitute for a fresh literature review if constraints change

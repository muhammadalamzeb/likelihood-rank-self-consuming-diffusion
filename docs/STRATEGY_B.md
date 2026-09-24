# Strategy B — bottom-up discovery (active)

**Chosen:** 2026-09-23 (owner asked to pick best of A vs B)  
**Why B over A:** Three v2 top-down waves (15/15 KILL) showed “thin niches” still collide with named papers. Bottom-up starts from a **runnable** stack and only forms a research claim after **observed** failure modes survive adversarial kill-tests.

## Frozen runnable stack (v0)

| Piece | Choice | Rationale |
|-------|--------|-----------|
| Task | Class-conditional image generation | Non-text GenAI under v2; small enough for Colab |
| Data | **CIFAR-10** (public, torchvision / HF) | Standard, free, reproducible |
| Model | Diffusers **UNet2DModel** class-conditional DDPM (train from scratch *or* load public ckpt) | Colab-feasible; inspectable |
| Optional ckpt | `Ketansomewhere/cifar10_conditional_diffusion1` | Immediate failure mining without full train |
| Alt toolkit | [`smalldiffusion`](https://github.com/yuanchenyang/smalldiffusion) CIFAR U-Net | Lightweight ablation / schedule / sampler probes |
| Compute default | Colab T4/A100; log GPU, steps, seeds | Core must stay reproducible |
| Metrics (core) | FID / IS (torch-fidelity or clean-fid), class-conditional accuracy of a fixed classifier on samples, visual grids | No proprietary API for core |

**Out of scope for this stack (already excluded or too heavy):** video foundations, 3DGS, joint AV sync, large SDXL training, proprietary judges as sole evidence.

## Process (mandatory order)

1. **Observe** — run fixed probes; log failures with configs/seeds (no novelty claim).  
2. **Cluster** — group failures into candidate scientific questions.  
3. **Kill-test** — adversarial literature check **after** observation, not before.  
4. **Lock** — only on KEEP / strong MODIFY under Constraint Set v2.

Do **not** invent results. Empty logs mean “not run yet,” not “no failures.”

## Probe battery v0 (observation only)

Run each with ≥2 seeds; save grids + scalar metrics.

| Probe ID | Intervention | What to watch |
|----------|--------------|---------------|
| P1 | CFG scale sweep {1, 1.5, 3, 7.5} | Class leakage, diversity collapse, oversaturation |
| P2 | DDIM steps {10, 20, 50, 100, 1000} | Quality cliff location; class-dependent cliffs |
| P3 | Train-time label noise {0%, 5%, 20%} (if training) | Whether errors concentrate on visually similar classes |
| P4 | Schedule swap linear ↔ cosine (fixed budget) | Which classes degrade first |
| P5 | Class imbalance (e.g. 10× fewer samples for 2 classes) | Undergeneration vs mode copy from majority |
| P6 | Wrong-class conditioning at inference | How “wrong” samples look; residual correct-class features |

## Anti-patterns (do not promote to paper claim)

- Stacking SpeeD / Data Warmup / Patch Forcing → exclusion 16  
- “Synthetic data helps/hurts” regime paper → exclusion 26  
- Continual diffusion forgetting → exclusion 29  
- Generic FID improvement without a defensible novelty statement  

## Status

- Strategy: **B locked**  
- Stack **v0** (CIFAR cond. DDPM): FC1–FC5 **KILL**  
- Stack **v1** (CIFAR FM): FM1 **KILL**  
- Stack **v2** (Wine CVAE): T3 **KILL**; T1/T2 no KEEP candidate  
- Stack **v3** (Digits CVAE β-sweep): D1 **KILL** (β-VAE tradeoff)  
- Stack **v4** (SBM tiny GraphVAE): G1/G3/G4 **KILL** (GraphVAE decoder limits)  
- Stack **v5** (Digits CVAE erase/τ/cap): F1–F3 **KILL**; Fashion-MNIST attempt aborted  
- Stack **v6** (2D GMM DDPM): M3 **KILL**; M1/M2 no claim; V3-21 meta **KILL**  
- Stack **v7** (synth spectrogram CVAE): S1/S2 **KILL**; S3 no claim  
- Stack **v8** (SEM tabular CVAE→ATE): C1–C3 **KILL**  
- Topic lock: **none**  
- Path-1 **terminal:** [`TERMINAL_FINDING_V3.md`](TERMINAL_FINDING_V3.md)  
- Exclusions through **39** (proposed 40–50 pending)  
- Owner forks: Paths **2–4** in `PUBLICATION_STATUS.md`

## Frozen runnable stack (v2 — logged)

| Piece | Choice |
|-------|--------|
| Task | Class-conditional tabular generation |
| Data | `sklearn.datasets.load_wine` |
| Model | Tiny Conditional VAE (train from scratch) |
| Probes | T1 imbalance, T2 support leakage, T3 corr sign flips |

v0/v1 freeze details retained above for provenance; do not re-mine NFE/label-swap/FM-step cliffs.

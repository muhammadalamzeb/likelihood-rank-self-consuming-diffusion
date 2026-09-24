# Experiment inventory (frozen stacks)

**Freeze date:** 2026-09-23  
**Mode:** Path 4 archive — stacks **frozen**; logs/configs/scripts **preserved**; claims in terminal dossiers **not rewritten**.

All stacks below produced **KILL** or **no claim**. None are topic-locked. Do not treat this inventory as a paper outline.

## Stack summary

| Stack | Script(s) | Logs / configs | What it actually established |
|-------|-----------|----------------|------------------------------|
| **v0** CIFAR cond. DDPM | `run_probes_p1_p2.py`, `run_probe_p6.py`, `quantify_fc1_fc2.py` | `p1p2_*`, `p6_*` | NFE / twin / class-difficulty / label-swap / SWITCH-style probes → **KILL** (excl. 31–35 neighborhood) |
| **v1** CIFAR flow-matching | `run_probe_fm_v1.py` | `fm_v1_*` | Euler step-count quality cliff → **KILL** (excl. 37; same object as NFE) |
| **v2** Wine CVAE | `run_probe_tab_v2.py` | `tab_v2_*` | Corr sign flips under synth → **KILL** (tabular dependency distortion; excl. 38). T1/T2 no claim |
| **v3** Digits CVAE β | `run_probe_digits_v3.py` | `digits_v3_*` | β↑ → diversity↓, class-acc↑ → **KILL** (β-VAE; excl. 39) |
| **v4** SBM GraphVAE | `run_probe_graph_v4.py` | `graph_v4_*` | Sparse empty-graph collapse; strong-SBM two-clique mode collapse → **KILL** (GraphVAE decoder limits) |
| **v5** Digits erase/τ/N | `run_probe_digits_v5.py` | `digits_v5_*` | Erase weak; τ↑ diversity↑; few-shot N cliff → **KILL**. Fashion-MNIST attempt **aborted** (download stall); script kept |
| **v6** 2D GMM DDPM | `run_probe_gmm_v6.py` | `gmm_v6_*` | Mode imbalance ρ≥10 → uncovered modes → **KILL**. M1/M2 no failure in quick regime |
| **v7** Synth spectrogram CVAE | `run_probe_spec_v7.py` | `spec_v7_*` | Train SNR / f0-jitter → **KILL**; time-warp no claim |
| **v8** SEM tabular→ATE | `run_probe_sem_v8.py` | `sem_v8_*` | Joint CVAE: finite recon MSE, large ATE bias → **KILL** as Amad/STEAM instance (not new law) |

## Discovery / Path-3 dossiers (not experiments)

| Doc | Established |
|-----|-------------|
| `DISCOVERY_V3.md` | Lit candidates V3-1…V3-20 **KILL**; V3-21 meta **KILL** |
| `PATH3_CAUSAL_BOUNDARY.md` | P3-C1…C3 **KILL**; Path 3 **TERMINAL** 0/3 |
| `PATH2_DISCOVERY.md` | P2-1…P2-5 **KILL**; Path 2 **NO KEEP** |
| `PATH4_ARCHIVE.md` | Project archive index; Path 4 **COMPLETE** |
| `TERMINAL_FINDING_V3.md` | Path 1 practical exhaustion under v3 |
| `KILL_LOG.md` | Hard excl. 1–39; proposed 40–50 pending |

## Provenance layout (do not delete)

```
experiments/
  scripts/     # probe runners (reusable patterns)
  configs/     # per-seed JSON
  logs/        # jsonl observation records
  NOTES.md     # human-readable stack outcomes
  README.md    # index
docs/          # constraints, kill log, terminals, intake
.venv/         # local env (D: drive)
```

## Separation: tooling vs claims

| Reusable (tooling) | Not reusable as novelty (claims) |
|--------------------|----------------------------------|
| Tiny CVAE / MLP DDPM / GraphVAE skeletons | “NFE cliffs”, “β-VAE tradeoff”, “GraphVAE empty graphs” |
| Seeded train/eval loops, jsonl logging | “Recon MSE tracks ATE” / “recon≠ATE as new boundary” |
| Offline sklearn digits / synthetic SEM / SBM / GMM / synth-spec generators | Path-3 recon↔ATE paper framing |
| LogisticRegression / OLS ATE helpers | Any KEEP claim from v0–v8 |

External directions may **reuse code patterns and logging**; they must **not** inherit killed novelty statements.

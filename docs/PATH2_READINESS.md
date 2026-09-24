# Path 2 readiness

**Date:** 2026-09-23  
**Mode:** External direction — **CLOSED**. Path 2 **NO KEEP**; Path 4 archive **COMPLETE**.

See [`PATH2_DISCOVERY.md`](PATH2_DISCOVERY.md), [`PATH4_ARCHIVE.md`](PATH4_ARCHIVE.md).  
No `PATH2_SURVIVOR.md` (no survivor).

## Frozen

| Item | State |
|------|-------|
| Experimental stacks v0–v8 | **Frozen** — no further probes |
| Logs / configs / scripts / data under `experiments/` | **Preserved** — do not delete |
| Path 1 terminal claims | Unchanged (`TERMINAL_FINDING_V3.md`) |
| Path 3 recon↔ATE claims | Unchanged; dossier marked **TERMINAL** 0/3 (`PATH3_CAUSAL_BOUNDARY.md`) |
| Path 2 discovery | **Exhausted — NO KEEP** |
| Paper drafting | **Off** |
| New candidate sweeps | **Off** pending explicit bar change or out-of-neighborhood owner direction |

## Prepared artifacts (historical)

| Doc | Role |
|-----|------|
| [`EXTERNAL_DIRECTION_INTAKE.md`](EXTERNAL_DIRECTION_INTAKE.md) | Closed — autonomous search recorded |
| [`EXPERIMENT_INVENTORY.md`](EXPERIMENT_INVENTORY.md) | What each stack established; tooling vs claims |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | Python / package versions |
| [`requirements-freeze.txt`](requirements-freeze.txt) | Full `pip freeze` |

## Reusable infrastructure (patterns only)

- Tiny PyTorch CVAE / MLP denoiser / GraphVAE training loops  
- Seeded jsonl logging + per-seed JSON configs  
- Offline generators: sklearn digits, Wine, SBM, 2D GMM, synth spectrograms, SEM tabular  
- Evaluators: LogisticRegression class-acc, pairwise diversity, OLS ATE, graph stats (NetworkX)  
- Env: `.venv` on D: with Diffusers / Transformers / datasets available if needed  

## Not to reuse as novelty

Any KILL claim in `KILL_LOG.md`, `NOTES.md`, Path-3 dossier, Discovery rounds 1–4 / V3-21, or Path-2 candidates **P2-1…P2-5**.

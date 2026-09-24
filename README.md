# Machine Learning Paper

## Status: PUBLICATION-READY (workshop / empirical L2 scope)

**Active constraint set:** v4 (`docs/CONSTRAINTS.md`)  
**Survivor:** W2-1 likelihood-rank ordering (`docs/RESEARCH_SURVIVOR.md`)  
**Paper:** [`paper/PAPER.md`](paper/PAPER.md)  
**Research log:** [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md)

| Track | State |
|-------|--------|
| Paths 1–4 (historical) | Archived evidence; stacks v0–v8 frozen |
| Wave-2 discovery | KEEP → MODIFY → empirical support for revised claim |
| Principal experiments | `w2_fs` (GMM); `w2_fs_digits` inconclusive |

## Claim (one sentence)

Under matched budgets in self-consuming DDPM training on imbalanced GMMs, minority-mode retention obeys **bottom‑k > random‑k > top‑k** when ranking synthetics by a denoising likelihood proxy; W₂ “false stability” for top‑k is **falsified**.

## Reproduce principal results

```powershell
cd "D:\Machine Learning Paper"
.\.venv\Scripts\python.exe -m pip install -r docs\requirements-freeze.txt
# matplotlib needed for figures:
.\.venv\Scripts\python.exe -m pip install matplotlib
.\.venv\Scripts\python.exe experiments\scripts\run_w2_false_stability.py --out experiments --seeds 0 1 2 --rhos 5.0 10.0 --policies mix top_k rand_k replace bottom_k --generations 5 --train-steps 600
.\.venv\Scripts\python.exe experiments\scripts\analyze_w2_fs.py
.\.venv\Scripts\python.exe experiments\scripts\make_w2_figures.py
```

Configs: `experiments/configs/w2_fs.json`  
Logs: `experiments/logs/w2_fs.jsonl`  
Summary: `experiments/analysis/w2_fs_summary.json`  
Figures: `paper/figures/`

Optional Digits transfer (inconclusive): `experiments/scripts/run_w2_fs_digits.py`

## Environment

See `docs/ENVIRONMENT.md` (CPU torch 2.14 / Python 3.14). Seeded runs; no fabricated metrics.

## Historical archive

Path terminals and kill log remain under `docs/` (Path 1–4). Do not repackage frozen v0–v8 kills as this paper’s novelty.

# Machine Learning Paper

## Status: FIRST-AUTHOR SUBMISSION-READY (arXiv / workshop L2)

**Active constraint set:** v4 (`docs/CONSTRAINTS.md`)  
**Survivor:** W2-1 likelihood-rank ordering (`docs/RESEARCH_SURVIVOR.md`)  
**Paper:** [`paper/PAPER.md`](paper/PAPER.md) · LaTeX [`paper/main.tex`](paper/main.tex)  
**Submit checklist:** [`paper/SUBMIT.md`](paper/SUBMIT.md)  
**Author:** Muhammad Alamzeb (sole / first)  
**Research log:** [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md)

| Track | State |
|-------|--------|
| Paths 1–4 (historical) | Archived evidence; stacks v0–v8 frozen |
| Wave-2 discovery | KEEP → MODIFY → empirical support for revised claim |
| Principal experiments | `w2_fs` (GMM); `w2_fs_hd`; `w2_fs_digits_pca` (supported); `w2_fs_digits` CVAE inconclusive |
| Publication | Submission package complete; **arXiv upload needs your account** |

## Claim (one sentence)

Under matched budgets in self-consuming DDPM training on imbalanced GMMs, minority-mode retention obeys **bottom‑k > random‑k > top‑k** when ranking synthetics by a denoising likelihood proxy; W₂ “false stability” for top‑k is **falsified**.

## Reproduce principal results

```bash
python scripts/reproduce_w2.py
# or: bash scripts/reproduce_w2.sh
# Windows: powershell -File scripts/reproduce_w2.ps1
```

```powershell
cd "D:\Machine Learning Paper"
.\.venv\Scripts\python.exe -m pip install -r docs\requirements-freeze.txt
.\.venv\Scripts\python.exe scripts\reproduce_w2.py
```

Configs: `experiments/configs/w2_fs.json`  
Logs: `experiments/logs/w2_fs.jsonl` (n=228; primary analysis uses 8 seed×ρ cells)  
Summary: `experiments/analysis/w2_fs_summary.json`  
Stats: `paper/table_stats_retention.csv`  
Figures: `paper/figures/`

Optional Digits-PCA transfer: `experiments/scripts/run_w2_fs_digits_pca.py`  
(Legacy Digits CVAE, inconclusive: `run_w2_fs_digits.py`)

## Environment

See `docs/ENVIRONMENT.md` (CPU torch 2.14 / Python 3.14). Seeded runs; no fabricated metrics.

## Historical archive

Path terminals and kill log remain under `docs/` (Path 1–4). Do not repackage frozen v0–v8 kills as this paper’s novelty.

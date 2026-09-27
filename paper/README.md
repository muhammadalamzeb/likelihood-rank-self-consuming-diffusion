# Paper package

Companion materials for *Likelihood-Ranked Synthetic Selection in Self-Consuming Diffusion: Minority Extinction Ordering at Moderate Imbalance*.

**Author:** Muhammad Alamzeb  
**DOI:** [10.5281/zenodo.22956890](https://doi.org/10.5281/zenodo.22956890)

| File | Role |
|------|------|
| `PAPER.md` | Manuscript (Markdown) |
| `main.tex` + `refs.bib` | LaTeX source |
| `main.pdf` | Compiled PDF |
| `SUBMIT.md` | Publication checklist |
| `figures/` | Figures used in the paper |
| `table_*.csv` | Tables exported from analysis scripts |
| `zenodo/` | Zenodo deposit files and metadata |

## Compile LaTeX

```powershell
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

For arXiv upload, use `arxiv_source.zip` (see `SUBMIT.md`).

All reported numbers are computed from `experiments/logs/*.jsonl` via the analysis scripts under `experiments/scripts/`.

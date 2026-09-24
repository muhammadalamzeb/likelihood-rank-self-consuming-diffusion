# Paper package

| File | Role |
|------|------|
| `PAPER.md` | Primary manuscript (markdown) |
| `main.tex` + `refs.bib` | LaTeX for workshop / arXiv |
| `SUBMIT.md` | First-author arXiv checklist |
| `figures/` | Generated plots |
| `table_*.csv` | Tables from analysis scripts |

**Author:** Muhammad Alamzeb (sole / first)

Compile LaTeX (if TeX installed):

```powershell
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or zip `main.tex`, `refs.bib`, `figures/` for arXiv cloud compile (`SUBMIT.md`).

All numbers come from `experiments/logs/*.jsonl` via analysis scripts.

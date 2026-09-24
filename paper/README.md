# Paper package

| File | Role |
|------|------|
| `PAPER.md` | Primary manuscript (markdown) |
| `main.tex` + `refs.bib` | LaTeX draft for workshop / arXiv |
| `figures/` | Generated plots |
| `table_*.csv` | Tables from analysis scripts |

Compile LaTeX (if TeX installed):

```powershell
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

All numbers come from `experiments/logs/*.jsonl` via `experiments/scripts/analyze_w2_fs.py` (and alpha tagging script output).

# Submission package — first-author checklist

**Author (sole / first):** Muhammad Alamzeb  
**Contact:** shayankhanmahar@gmail.com  
**Manuscript:** `paper/main.tex` + `paper/PAPER.md`  
**PDF:** `paper/main.pdf`  
**arXiv zip:** `paper/arxiv_source.zip`  
**Abstract paste:** `paper/arxiv_abstract.txt`  
**Scope:** Workshop / empirical short paper / **arXiv cs.LG** note (L2 empirical).  
**Not claiming:** NeurIPS/ICML SOTA method; ImageNet-scale results.

## Package complete (local)

- [x] Sole first-author name on title page
- [x] Abstract + intro + method + experiments + limitations + conclusion
- [x] Figures (`paper/figures/*.png`)
- [x] Tables from analysis CSVs (numbers match `table_retention.csv` / Wilcoxon `p≈0.042`)
- [x] Bibliography with full author lists (`refs.bib` + `main.bbl` in zip)
- [x] Compiled PDF (`paper/main.pdf`)
- [x] arXiv source zip (`paper/arxiv_source.zip`: `main.tex`, `refs.bib`, `main.bbl`, `00README.txt`, `figures/`)
- [x] Plain-text abstract for the arXiv form (`arxiv_abstract.txt`)
- [x] Reproducibility script + logs
- [x] Honest scope / limitations documented
- [x] PDF metadata (title/author via hyperref)

## Only remaining: human arXiv account step

1. Open [arxiv.org/submit](https://arxiv.org/submit) and log in (new cs.LG accounts may need endorsement).
2. Upload **`paper/arxiv_source.zip`**.
3. Paste abstract from **`paper/arxiv_abstract.txt`**.
4. Category: **cs.LG** (optional secondary: **stat.ML**).
5. License: **CC BY 4.0**.
6. After ID issues, optionally index on Hugging Face Paper Pages.

## Venue guidance

| Target | Fit |
|--------|-----|
| arXiv preprint | **Yes — ready to upload** |
| GenAI / synthetic-data workshop | **Yes** |
| Top conference main track | **No** without larger-scale evidence |

## Cover-letter sentence

Under matched budgets in self-consuming DDPM training, minority-mode retention obeys bottom-k > random-k > top-k when ranking synthetics by a denoising likelihood proxy; W₂ “false stability” for top-k is falsified.

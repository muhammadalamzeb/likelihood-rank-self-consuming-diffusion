# Submission package — first-author checklist

**Author (sole / first):** Muhammad Alamzeb  
**Contact:** shayankhanmahar@gmail.com  
**Manuscript:** `paper/main.tex` + `paper/PAPER.md`  
**PDF:** `paper/main.pdf`  
**arXiv zip:** `paper/arxiv_source.zip`  
**Scope:** Workshop / empirical short paper / **arXiv cs.LG** note (L2 empirical).  
**Not claiming:** NeurIPS/ICML SOTA method; ImageNet-scale results.

## Ready now

- [x] Sole first-author name on title page
- [x] Abstract + intro + method + experiments + limitations + conclusion
- [x] Figures (`paper/figures/*.png`)
- [x] Tables from analysis CSVs (no hand-edited metrics)
- [x] Bibliography with full author lists (`refs.bib`)
- [x] Compiled PDF (`paper/main.pdf`)
- [x] arXiv source zip (`paper/arxiv_source.zip`)
- [x] Reproducibility script (`scripts/reproduce_w2.ps1`)
- [x] Logged experiments: `w2_fs`, `w2_fs_hd`, `w2_fs_digits_pca`, ablations
- [x] Honest scope / limitations documented

## You must do to go live (credentials / account)

1. Open [arXiv submit](https://arxiv.org/submit) and log in (or create account; cs.LG may need endorsement).
2. Upload **`paper/arxiv_source.zip`** (contains `main.tex`, `refs.bib`, `figures/`).
3. Optionally attach `main.pdf` as well.
4. **Category:** `cs.LG` (primary); optional `stat.ML`.
5. **License:** Creative Commons Attribution (CC BY 4.0) recommended.
6. After the arXiv ID issues, optionally index on Hugging Face Paper Pages.

## Venue guidance

| Target | Fit |
|--------|-----|
| arXiv preprint | **Yes** — primary next step |
| GenAI / synthetic-data / workshop | **Yes** — empirical note |
| Top conference (NeurIPS/ICML/ICLR main) | **No** without larger-scale evidence |

## One-sentence claim for cover letter

Under matched budgets in self-consuming DDPM training, minority-mode retention obeys bottom-k > random-k > top-k when ranking synthetics by a denoising likelihood proxy; W₂ “false stability” for top-k is falsified.

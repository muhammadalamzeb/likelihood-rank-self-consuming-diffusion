# Submission package — first-author checklist

**Author (sole / first):** Muhammad Alamzeb  
**Contact:** shayankhanmahar@gmail.com  
**Manuscript:** `paper/main.tex` + `paper/PAPER.md`  
**PDF:** `paper/main.pdf`  
**arXiv zip:** `paper/arxiv_source.zip`  
**Abstract paste:** `paper/arxiv_abstract.txt`  
**Public code:** https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion  
**Git commit (repro pin):** see `git rev-parse HEAD` on `master`  
**Scope:** Workshop / empirical short paper / **arXiv cs.LG** note.  
**Not claiming:** NeurIPS/ICML SOTA; ImageNet-scale results.

## Review revision (2026-09-25)

Addressed second-pass review: per-ρ breakdown (order at ρ=5 only); α caption/text fixes; α labeled illustrative (n=2); inline W₂ table gens 1–5; r defined before results; double-blind GitHub note.

## Anonymity

- **arXiv:** named author is correct (keep as-is).
- **Double-blind workshop/conference:** strip name/email from title page before submission.

## Package complete (local)

- [x] Sole first-author name on title page (arXiv)
- [x] Stats/methods clarifications from review
- [x] Figures + tables (incl. `table_seed_rho_grid.csv`, `table_w2_*.csv`, effect-size stats)
- [x] Cross-platform reproduce: `scripts/reproduce_w2.py` (+ `.sh` / `.ps1`)
- [x] Compiled PDF + arXiv zip with `.bbl`
- [x] Public GitHub repo
- [ ] **Live arXiv upload** (your account) — upload `paper/arxiv_source.zip`

## Upload steps

1. [arxiv.org/submit](https://arxiv.org/submit) → upload `paper/arxiv_source.zip`
2. Paste `paper/arxiv_abstract.txt`
3. Category **cs.LG** (optional **stat.ML**); license **CC BY 4.0**
4. In comments, you may note: code at https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion

## Cover-letter sentence

Under matched budgets in self-consuming DDPM training, minority-mode retention obeys bottom-k > random-k > top-k when ranking synthetics by a denoising likelihood proxy; top-k does not improve sliced W₂ vs random-k in the primary eight-cell matrix.

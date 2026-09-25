# Submission package — first-author checklist

**Author (sole / first):** Muhammad Alamzeb  
**Contact:** shayankhanmahar@gmail.com  
**Manuscript:** `paper/main.tex` + `paper/PAPER.md`  
**PDF:** `paper/main.pdf`  
**arXiv zip:** `paper/arxiv_source.zip`  
**Abstract paste:** `paper/arxiv_abstract.txt`  
**Git commit (repro pin):** see `git rev-parse HEAD` after pull  
**Scope:** Workshop / empirical short paper / **arXiv cs.LG** note.  
**Not claiming:** NeurIPS/ICML SOTA; ImageNet-scale results.

## Review revision (2026-09-25)

Addressed reviewer/self-audit items: explicit 8-cell seed×ρ grid; effect sizes + bootstrap CIs + Holm note; softened “falsification” language with W₂ Wilcoxon; transfer mean±std; retention \(r>1\) definition; mix vs rand_k; sliced-W₂ (24 proj / ≤400 pts); α table numbers; clearer novelty statement; cross-platform `reproduce_w2.py`; repo URL pending public hosting.

## Anonymity

- **arXiv:** named author is correct (keep as-is).
- **Double-blind workshop/conference:** strip name/email from title page before submission.

## Package complete (local)

- [x] Sole first-author name on title page (arXiv)
- [x] Stats/methods clarifications from review
- [x] Figures + tables (incl. `table_seed_rho_grid.csv`, `table_w2_*.csv`, effect-size stats)
- [x] Cross-platform reproduce: `scripts/reproduce_w2.py` (+ `.sh` / `.ps1`)
- [x] Compiled PDF + arXiv zip with `.bbl`
- [ ] Public GitHub/Zenodo URL (add when hosted; archive ships with manuscript)
- [ ] **Live arXiv upload** (your account)

## Upload steps

1. [arxiv.org/submit](https://arxiv.org/submit) → upload `paper/arxiv_source.zip`
2. Paste `paper/arxiv_abstract.txt`
3. Category **cs.LG** (optional **stat.ML**); license **CC BY 4.0**
4. After ID issues, publish code archive and link URL in a revision

## Cover-letter sentence

Under matched budgets in self-consuming DDPM training, minority-mode retention obeys bottom-k > random-k > top-k when ranking synthetics by a denoising likelihood proxy; top-k does not improve sliced W₂ vs random-k in the primary eight-cell matrix.

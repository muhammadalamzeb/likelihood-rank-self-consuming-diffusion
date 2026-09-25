00README.txt for arXiv

Title: Likelihood-Ranked Synthetic Selection in Self-Consuming Diffusion:
       Minority Extinction Ordering on Imbalanced Mixtures
Author: Muhammad Alamzeb (sole / first author)
Contact: shayankhanmahar@gmail.com
Code: https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion

Build:
  pdflatex main && bibtex main && pdflatex main && pdflatex main
Or let arXiv compile (main.bbl included).

Files in this zip:
  main.tex, refs.bib, main.bbl, 00README.txt, figures/

Primary category: cs.LG
Optional: stat.ML
License: CC BY 4.0 recommended

Notes:
  Primary retention analysis is stratified by rho (see Tables in main.tex).
  Exact Wilcoxon p-values via sign enumeration (make_submission_stats.py).

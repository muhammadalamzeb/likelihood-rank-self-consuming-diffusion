# Likelihood-Ranked Synthetic Selection in Self-Consuming Diffusion

**Minority Extinction Ordering at Moderate Imbalance**

Author: **Muhammad Alamzeb**  
Preprint: [https://doi.org/10.5281/zenodo.22956890](https://doi.org/10.5281/zenodo.22956890)  
Code release: [v1.0.0](https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion/releases/tag/v1.0.0)

## Abstract

Self-consuming training loops—retraining generative models on their own outputs—can induce model collapse, including loss of rare modes. This repository accompanies an empirical study of unlabeled selection of synthetic samples by a denoising likelihood proxy in self-consuming DDPM training on imbalanced mixtures, with emphasis on **moderate** imbalance.

Define the minority-mass ratio \(r = m^{(1)}/m^{(0)}\). Results are stratified by imbalance \(\rho\). At \(\rho=5\) (\(n=5\) seeds), generation-1 minority mass obeys \(\mathrm{bottom}>\mathrm{rand}>\mathrm{top}\) on all seeds, with large paired effects versus random-\(k\) (top−rand: Cohen’s \(d_z \approx -1.96\); bottom−rand: \(d_z \approx +1.65\)). At \(\rho=10\) (\(n=3\)), the full order does not hold and contrasts are near null. Top-\(k\) does not improve sliced \(W_2\) versus random-\(k\) at generation 1. Directional transfers (8D GMM, Digits-PCA) match the \(\rho=5\) ordering.

## Paper

| Resource | Location |
|----------|----------|
| Manuscript (Markdown) | [`paper/PAPER.md`](paper/PAPER.md) |
| LaTeX source | [`paper/main.tex`](paper/main.tex), [`paper/refs.bib`](paper/refs.bib) |
| PDF | [`paper/main.pdf`](paper/main.pdf) |
| Figures / tables | [`paper/figures/`](paper/figures/), [`paper/table_*.csv`](paper/) |
| Zenodo deposit kit | [`paper/zenodo/`](paper/zenodo/) |
| Submission notes | [`paper/SUBMIT.md`](paper/SUBMIT.md) |

## Citation

```bibtex
@misc{alamzeb2026likelihood,
  author       = {Alamzeb, Muhammad},
  title        = {Likelihood-Ranked Synthetic Selection in Self-Consuming Diffusion:
                  Minority Extinction Ordering at Moderate Imbalance},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22956890},
  url          = {https://doi.org/10.5281/zenodo.22956890}
}
```

## Reproduce

```bash
python -m pip install -r docs/requirements-freeze.txt
python scripts/reproduce_w2.py
```

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r docs\requirements-freeze.txt
.\.venv\Scripts\python.exe scripts\reproduce_w2.py
# or: powershell -File scripts\reproduce_w2.ps1
```

| Artifact | Path |
|----------|------|
| Primary config | `experiments/configs/w2_fs.json` |
| Primary logs | `experiments/logs/w2_fs.jsonl` |
| Summary | `experiments/analysis/w2_fs_summary.json` |
| Retention stats | `paper/table_stats_retention_stratified.csv` |
| Environment notes | [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) |

Optional transfers: `experiments/scripts/run_w2_fs_hd.py` (8D), `experiments/scripts/run_w2_fs_digits_pca.py` (Digits-PCA).

## Repository layout

```
paper/           Manuscript, figures, tables, Zenodo kit
experiments/     Configs, runners, logs, analysis
scripts/         End-to-end reproduce helpers
docs/            Environment freeze and research notes
```

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Muhammad Alamzeb.

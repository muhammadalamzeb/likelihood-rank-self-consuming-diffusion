# Likelihood-Ranked Selection in Self-Consuming Diffusion

**A \(W_2\) Negative Result and a Narrow Minority-Retention Regime**

Author: **Muhammad Alamzeb**  
Contact: shayankhanmahar@gmail.com  
Preprint: [https://doi.org/10.5281/zenodo.22956890](https://doi.org/10.5281/zenodo.22956890)  
Code release: [v1.2.1](https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion/releases/tag/v1.2.1)

## Abstract

Self-consuming training loops can induce model collapse. This repository accompanies a workshop-scale empirical study of unlabeled likelihood ranking of synthetic samples in self-consuming DDPM training on imbalanced mixtures.

**Primary result:** top-\(k\) selection by a denoising likelihood proxy does **not** improve sliced \(W_2\) versus random-\(k\) at generation 1 (\(23/23\) seed–imbalance cells; bootstrap 95% CI excludes zero).

**Secondary, post-hoc / narrow-regime observation:** after an original “false stability” hypothesis was falsified, the full \(\mathrm{bottom}>\mathrm{rand}>\mathrm{top}\) order appears at \(\rho=5\) (\(n=10\), \(10/10\); exact \(p=0.002\)) and fails for most seeds at \(\rho=7\) / \(\rho=10\). Treat as exploratory (HARKing risk), not a pre-registered law. Not shown on ImageNet-scale diffusion.

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
  title        = {Likelihood-Ranked Selection in Self-Consuming Diffusion:
                  A W2 Negative Result and a Narrow Minority-Retention Regime},
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
python experiments/scripts/make_submission_stats.py
```

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r docs\requirements-freeze.txt
.\.venv\Scripts\python.exe scripts\reproduce_w2.py
.\.venv\Scripts\python.exe experiments\scripts\make_submission_stats.py
```

| Artifact | Path |
|----------|------|
| Primary config | `experiments/configs/w2_fs.json` |
| Primary logs | `experiments/logs/w2_fs.jsonl` |
| Stratified stats | `paper/table_stats_retention_stratified.csv` |
| \(W_2\) table | `paper/table_w2_top_minus_rand.csv` |
| Environment | [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) |

Optional: `experiments/scripts/run_w2_fs_hd.py`, `run_w2_fs_digits_pca.py`, `run_w2_fs_digits_ddpm.py`.

## Repository layout

```
paper/           Manuscript, figures, tables, Zenodo kit
experiments/     Configs, runners, logs, analysis
scripts/         End-to-end reproduce helpers
docs/            Environment freeze (and historical research notes)
```

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Muhammad Alamzeb.

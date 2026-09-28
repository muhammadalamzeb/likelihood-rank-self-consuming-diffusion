# Likelihood-Ranked Selection in Self-Consuming Diffusion

**A \(W_2\) Negative Result and a Narrow Minority-Retention Regime**

Author: **Muhammad Alamzeb** · Contact: shayankhanmahar@gmail.com  
Preprint: [https://doi.org/10.5281/zenodo.22956890](https://doi.org/10.5281/zenodo.22956890)  
Release: [v1.4.0](https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion/releases/tag/v1.4.0)

## Abstract

Workshop-scale study of unlabeled likelihood ranking in self-consuming DDPM training on imbalanced mixtures (**not** ImageNet-scale diffusion).

**Primary:** top-\(k\) does **not** improve sliced \(W_2\) vs random-\(k\) at generation 1 (\(38/38\) cells; bootstrap CI excludes zero).

**Secondary:** minority retention bottom>rand>top is strongest at moderate imbalance \(\rho=5\) (exploratory \(n=10\), \(10/10\), \(p=0.002\); held-out seeds \(10\)–\(14\): \(4/5\)). Weaker at \(\rho=7\) (\(6/10\)) and \(\rho=10\) (\(4/13\)). Retention claim is post-hoc relative to a falsified “false stability” hypothesis.

## Paper & reproduce

| | |
|--|--|
| Manuscript | [`paper/PAPER.md`](paper/PAPER.md), [`paper/main.tex`](paper/main.tex), [`paper/main.pdf`](paper/main.pdf) |
| Stats | `python experiments/scripts/make_submission_stats.py` |
| Full reproduce | `python scripts/reproduce_w2.py` |
| Env freeze | [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) |

```bibtex
@misc{alamzeb2026likelihood,
  author       = {Alamzeb, Muhammad},
  title        = {Likelihood-Ranked Selection in Self-Consuming Diffusion:
                  A W2 Negative Result and a Narrow Minority-Retention Regime},
  year         = {2026},
  doi          = {10.5281/zenodo.22956890},
  url          = {https://doi.org/10.5281/zenodo.22956890}
}
```

## License

MIT — Copyright (c) 2026 Muhammad Alamzeb.

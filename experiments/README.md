# Experiments

Code and logs for the likelihood-rank self-consuming DDPM study.

| Experiment | Script | Logs |
|------------|--------|------|
| Primary GMM (`w2_fs`) | `scripts/run_w2_false_stability.py` | `logs/w2_fs.jsonl` |
| 8D transfer | `scripts/run_w2_fs_hd.py` | `logs/w2_fs_hd.jsonl` |
| \(\alpha\) / \(k_{\mathrm{frac}}\) sensitivity | same runner, `--log-name` | `logs/w2_fs_alpha.jsonl`, `logs/w2_fs_kfrac.jsonl` |
| Digits pixel DDPM | `scripts/run_w2_fs_digits_ddpm.py` | `logs/w2_fs_digits_ddpm.jsonl` |
| Digits-PCA transfer | `scripts/run_w2_fs_digits_pca.py` | `logs/w2_fs_digits_pca.jsonl` |
| Digits CVAE (inconclusive) | `scripts/run_w2_fs_digits.py` | `logs/w2_fs_digits.jsonl` |
| CIFAR-10 tiny-UNet (appendix) | `scripts/run_w2_fs_cifar.py` | `logs/w2_fs_cifar.jsonl` |
| Proxy vs exact GMM density | `scripts/validate_likelihood_proxy.py` | `logs/w2_fs_proxy_val.jsonl` |
| Continuous ρ sweep | `scripts/run_w2_false_stability.py` | `logs/w2_fs_rho_sweep.jsonl` |
| Accumulate / oracle baselines | same runner (`accumulate`, `oracle_*`) | `logs/w2_fs_baselines.jsonl` |
| Analysis / figures | `analyze_w2_fs.py`, `analyze_rho_sweep_baselines.py`, `make_w2_figures.py` | `analysis/` |

End-to-end reproduce from the repository root:

```bash
python scripts/reproduce_w2.py
```

Do not overwrite `logs/w2_fs.jsonl` unless appending with `--append`.  
Earlier exploratory stacks are listed in `docs/EXPERIMENT_INVENTORY.md` and are not part of the paper’s claims.

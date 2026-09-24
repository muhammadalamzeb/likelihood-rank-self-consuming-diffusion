# Experiments

**Active (W2-1):** self-consuming likelihood-rank study — do not overwrite `logs/w2_fs.jsonl` without `--append`.

| Experiment | Script | Logs | Status |
|------------|--------|------|--------|
| **w2_fs** GMM DDPM | `scripts/run_w2_false_stability.py` | `logs/w2_fs.jsonl` | KEEP support |
| w2_fs_hd (8D) | `scripts/run_w2_fs_hd.py` | `logs/w2_fs_hd.jsonl` | transfer support |
| w2_fs_alpha / kfrac | same runner, `--log-name` | `logs/w2_fs_alpha.jsonl`, `w2_fs_kfrac.jsonl` | sensitivity |
| w2_fs_digits_pca | `scripts/run_w2_fs_digits_pca.py` | `logs/w2_fs_digits_pca.jsonl` | Digits-PCA transfer support |
| w2_fs_digits (CVAE) | `scripts/run_w2_fs_digits.py` | `logs/w2_fs_digits.jsonl` | inconclusive (superseded) |
| analyze / figures | `analyze_w2_fs.py`, `analyze_w2_fs_digits_pca.py`, `make_w2_figures.py` | `analysis/` | — |

Reproduce: `scripts/reproduce_w2.ps1` (repo root).

## Frozen historical stacks (v0–v8)

Preserved; not novelty for the W2-1 paper. See `docs/EXPERIMENT_INVENTORY.md`.

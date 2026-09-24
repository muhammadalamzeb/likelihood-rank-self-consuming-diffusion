# Experiments

**Active (W2-1):** self-consuming likelihood-rank study — do not overwrite `logs/w2_fs.jsonl` without `--append`.

| Experiment | Script | Logs | Status |
|------------|--------|------|--------|
| **w2_fs** GMM DDPM | `scripts/run_w2_false_stability.py` | `logs/w2_fs.jsonl` | KEEP support |
| w2_fs_digits | `scripts/run_w2_fs_digits.py` | `logs/w2_fs_digits.jsonl` | inconclusive |
| analyze / figures | `analyze_w2_fs.py`, `make_w2_figures.py` | `analysis/` | — |

Reproduce: `scripts/reproduce_w2.ps1` (repo root).

## Frozen historical stacks (v0–v8)

Preserved; not novelty for the W2-1 paper. See `docs/EXPERIMENT_INVENTORY.md`.

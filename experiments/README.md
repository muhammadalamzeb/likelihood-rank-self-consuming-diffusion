# Experiments

**Status (2026-09-23): FROZEN — Path 4 archive complete.**  
Do not delete scripts, logs, configs, or datasets. Do not run novelty probes unless research is explicitly resumed per [`docs/PATH4_ARCHIVE.md`](../docs/PATH4_ARCHIVE.md).

| Stack | Script | Logs | Claim status |
|-------|--------|------|--------------|
| v0 CIFAR DDPM | `scripts/run_probes_p1_p2.py`, `run_probe_p6.py`, `quantify_fc1_fc2.py` | `logs/p1p2_*`, `p6_*` | KILL (frozen) |
| v1 CIFAR FM | `scripts/run_probe_fm_v1.py` | `logs/fm_v1_*` | KILL (frozen) |
| v2 Wine CVAE | `scripts/run_probe_tab_v2.py` | `logs/tab_v2_*` | KILL (frozen) |
| v3 Digits β-CVAE | `scripts/run_probe_digits_v3.py` | `logs/digits_v3_*` | KILL (frozen) |
| v4 SBM GraphVAE | `scripts/run_probe_graph_v4.py` | `logs/graph_v4_*` | KILL (frozen) |
| v5 Digits erase/τ/N | `scripts/run_probe_digits_v5.py` | `logs/digits_v5_*` | KILL (frozen) |
| v5 Fashion attempt | `scripts/run_probe_fmnist_v5.py` | — (download aborted) | aborted; script kept |
| v6 2D GMM DDPM | `scripts/run_probe_gmm_v6.py` | `logs/gmm_v6_*` | KILL (frozen) |
| v7 synth spectrogram | `scripts/run_probe_spec_v7.py` | `logs/spec_v7_*` | KILL (frozen) |
| v8 SEM→ATE | `scripts/run_probe_sem_v8.py` | `logs/sem_v8_*` | KILL (frozen; Amad instance) |

Inventory: [`docs/EXPERIMENT_INVENTORY.md`](../docs/EXPERIMENT_INVENTORY.md).  
Env: [`docs/ENVIRONMENT.md`](../docs/ENVIRONMENT.md).  
Archive: [`docs/PATH4_ARCHIVE.md`](../docs/PATH4_ARCHIVE.md).

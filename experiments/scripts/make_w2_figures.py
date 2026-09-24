#!/usr/bin/env python3
"""Generate tables/figures from w2_fs_summary.json and raw jsonl."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "w2_fs.jsonl"
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

recs = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
summary = json.loads((OUT / "w2_fs_summary.json").read_text(encoding="utf-8"))

# Table: retention ratios
lines = ["policy,mean_retention_g1_g0,std,n"]
for pol, v in summary["retention_g1_over_g0"].items():
    lines.append(f"{pol},{v['mean']:.4f},{v['std']:.4f},{v['n']}")
(OUT / "table_retention.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Table: primary rho=5 g=1
lines = ["seed,top_k_minority,rand_k_minority,bottom_k_minority,top_k_majority,rand_k_majority"]
for r in summary["primary_rho5_g1"]:
    lines.append(
        f"{r['seed']},{r['top_min']:.4f},{r['rand_min']:.4f},{r['bottom_min']:.4f},{r['top_maj']:.4f},{r['rand_maj']:.4f}"
    )
(OUT / "table_primary_rho5_g1.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Figure: minority_mean vs generation for rho=5, averaged over seeds
policies = ["top_k", "rand_k", "bottom_k", "mix"]
gens = list(range(0, 6))
plt.figure(figsize=(6.5, 4.0))
for pol in policies:
    ys = []
    yerr = []
    for g in gens:
        vals = [r["minority_mean"] for r in recs if r["rho"] == 5.0 and r["policy"] == pol and r["generation"] == g]
        ys.append(float(np.mean(vals)))
        yerr.append(float(np.std(vals)))
    plt.errorbar(gens, ys, yerr=yerr, marker="o", label=pol)
plt.xlabel("Generation")
plt.ylabel("Mean minority-mode mass")
plt.title(r"Self-consuming DDPM on imbalanced GMM ($\rho=5$)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "minority_vs_generation_rho5.png", dpi=150)
plt.close()

# Figure: retention bars
plt.figure(figsize=(6.0, 3.8))
pols = ["bottom_k", "mix", "rand_k", "top_k"]
means = [summary["retention_g1_over_g0"][p]["mean"] for p in pols]
stds = [summary["retention_g1_over_g0"][p]["std"] for p in pols]
plt.bar(pols, means, yerr=stds, capsize=4, color=["#2a9d8f", "#264653", "#e9c46a", "#e76f51"])
plt.ylabel("Minority mass retention (g1/g0)")
plt.title("Likelihood-rank ordering of synthetic selection")
plt.tight_layout()
plt.savefig(FIG / "retention_bars.png", dpi=150)
plt.close()

print("wrote tables and figures under", OUT)

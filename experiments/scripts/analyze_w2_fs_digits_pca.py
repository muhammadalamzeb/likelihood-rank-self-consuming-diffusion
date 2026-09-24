#!/usr/bin/env python3
"""Summarize w2_fs_digits_pca retention ordering."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "w2_fs_digits_pca.jsonl"
OUT = ROOT / "analysis"
PAPER = ROOT.parent / "paper"
OUT.mkdir(parents=True, exist_ok=True)

recs = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
by = {(r["seed"], r["policy"], r["generation"]): r for r in recs}
seeds = sorted({r["seed"] for r in recs})
policies = ["top_k", "rand_k", "bottom_k"]

holds = []
rets = {p: [] for p in policies}
maj_g1 = {p: [] for p in policies}
lines = ["seed,policy,m0,m1,retention,majority_g1"]
for s in seeds:
    m1 = {p: by[(s, p, 1)]["minority_mean"] for p in policies}
    m0 = {p: by[(s, p, 0)]["minority_mean"] for p in policies}
    ok = m1["bottom_k"] > m1["rand_k"] > m1["top_k"]
    holds.append(ok)
    for p in policies:
        r = m1[p] / max(m0[p], 1e-9)
        rets[p].append(r)
        maj_g1[p].append(by[(s, p, 1)]["majority_mass"])
        lines.append(
            f"{s},{p},{m0[p]},{m1[p]},{r},{by[(s, p, 1)]['majority_mass']}"
        )

summary = {
    "experiment": "w2_fs_digits_pca",
    "n_records": len(recs),
    "seeds": seeds,
    "order_g1_bottom_gt_rand_gt_top": {"holds": int(sum(holds)), "n": len(holds)},
    "retention_g1_over_g0": {
        p: {
            "mean": float(np.mean(rets[p])),
            "std": float(np.std(rets[p])),
            "values": rets[p],
        }
        for p in policies
    },
    "majority_g1_mean": {p: float(np.mean(maj_g1[p])) for p in policies},
    "contrasts_vs_rand_retention": {
        "top_minus_rand": float(np.mean(np.array(rets["top_k"]) - np.array(rets["rand_k"]))),
        "bottom_minus_rand": float(
            np.mean(np.array(rets["bottom_k"]) - np.array(rets["rand_k"]))
        ),
    },
}
(OUT / "w2_fs_digits_pca_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
csv = "\n".join(lines) + "\n"
(OUT / "table_digits_pca_retention.csv").write_text(csv, encoding="utf-8")
PAPER.mkdir(parents=True, exist_ok=True)
(PAPER / "table_digits_pca_retention.csv").write_text(csv, encoding="utf-8")
print(json.dumps(summary, indent=2))

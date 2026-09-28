#!/usr/bin/env python3
"""Analyze rho sweep + accumulate/oracle baselines for paper revision."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "experiments" / "logs"
PAPER = ROOT / "paper"
ANALYSIS = ROOT / "experiments" / "analysis"


def load_jsonl(*names: str):
    rows = []
    for name in names:
        p = LOGS / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def retention_table(rows, policies=("bottom_k", "rand_k", "top_k"), seeds=None):
    by = {(r["seed"], float(r["rho"]), r["policy"], r["generation"]): r for r in rows}
    rhos = sorted({float(r["rho"]) for r in rows})
    out_lines = ["rho,n,bottom,rand,top,full_order,bottom_gt_rand,rand_gt_top,mean_w2_bottom,mean_w2_rand,mean_w2_top"]
    summary = []
    for rho in rhos:
        seed_set = sorted({s for (s, rr, p, g) in by if rr == rho and p == policies[0]})
        if seeds is not None:
            seed_set = [s for s in seed_set if s in seeds]
        ok = []
        rets = {p: [] for p in policies}
        w2s = {p: [] for p in policies}
        full = br = rt = 0
        for s in seed_set:
            if not all((s, rho, p, 0) in by and (s, rho, p, 1) in by for p in policies):
                continue
            ok.append(s)
            rr = {}
            for p in policies:
                m0 = by[(s, rho, p, 0)]["minority_mean"]
                m1 = by[(s, rho, p, 1)]["minority_mean"]
                r = m1 / m0 if m0 > 0 else float("nan")
                rr[p] = r
                rets[p].append(r)
                w2s[p].append(by[(s, rho, p, 1)]["w2_sliced"])
            if rr["bottom_k"] > rr["rand_k"] > rr["top_k"]:
                full += 1
            if rr["bottom_k"] > rr["rand_k"]:
                br += 1
            if rr["rand_k"] > rr["top_k"]:
                rt += 1
        if not ok:
            continue
        line = (
            f"{rho:g},{len(ok)},"
            f"{np.mean(rets['bottom_k']):.4f},{np.mean(rets['rand_k']):.4f},{np.mean(rets['top_k']):.4f},"
            f"{full}/{len(ok)},{br}/{len(ok)},{rt}/{len(ok)},"
            f"{np.mean(w2s['bottom_k']):.4f},{np.mean(w2s['rand_k']):.4f},{np.mean(w2s['top_k']):.4f}"
        )
        out_lines.append(line)
        summary.append(
            {
                "rho": rho,
                "n": len(ok),
                "full_order": f"{full}/{len(ok)}",
                "full_rate": full / len(ok),
                "mean_r": {p: float(np.mean(rets[p])) for p in policies},
            }
        )
    return "\n".join(out_lines) + "\n", summary


def baseline_table(rows, rho=5.0, seeds=(0, 1, 2, 3, 4)):
    by = {(r["seed"], float(r["rho"]), r["policy"], r["generation"]): r for r in rows}
    policies = sorted({r["policy"] for r in rows if float(r["rho"]) == rho})
    lines = ["seed,policy,m0,m1,retention,w2_g1,train_size_g1"]
    means = defaultdict(list)
    for s in seeds:
        for p in policies:
            if (s, rho, p, 0) not in by or (s, rho, p, 1) not in by:
                continue
            m0 = by[(s, rho, p, 0)]["minority_mean"]
            m1 = by[(s, rho, p, 1)]["minority_mean"]
            r = m1 / m0 if m0 > 0 else float("nan")
            w2 = by[(s, rho, p, 1)]["w2_sliced"]
            ts = by[(s, rho, p, 1)].get("train_size", "")
            lines.append(f"{s},{p},{m0:.6f},{m1:.6f},{r:.6f},{w2:.6f},{ts}")
            means[p].append((r, w2))
    sum_lines = ["policy,n,mean_retention,mean_w2_g1"]
    for p, xs in sorted(means.items()):
        sum_lines.append(f"{p},{len(xs)},{np.mean([a for a,_ in xs]):.6f},{np.mean([b for _,b in xs]):.6f}")
    return "\n".join(lines) + "\n", "\n".join(sum_lines) + "\n"


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    # Combined sweep: primary matrix + new sweep rhos; restrict to seeds 0-4 for fair curve
    sweep_rows = load_jsonl("w2_fs.jsonl", "w2_fs_rho_sweep.jsonl")
    # keep only ranking policies
    sweep_rows = [r for r in sweep_rows if r["policy"] in ("top_k", "rand_k", "bottom_k")]
    csv, summary = retention_table(sweep_rows, seeds=set(range(5)))
    (ANALYSIS / "table_rho_sweep.csv").write_text(csv, encoding="utf-8")
    (PAPER / "table_rho_sweep.csv").write_text(csv, encoding="utf-8")
    print("=== rho sweep (seeds 0-4) ===")
    print(csv)

    base_rows = load_jsonl("w2_fs_baselines.jsonl", "w2_fs.jsonl")
    # For baselines compare accumulate/oracle vs top/rand/bottom/mix at rho=5 seeds 0-4
    detail, summ = baseline_table(base_rows, rho=5.0, seeds=(0, 1, 2, 3, 4))
    # filter summary to policies of interest if present
    (ANALYSIS / "table_baselines_rho5.csv").write_text(detail, encoding="utf-8")
    (ANALYSIS / "table_baselines_rho5_summary.csv").write_text(summ, encoding="utf-8")
    (PAPER / "table_baselines_rho5_summary.csv").write_text(summ, encoding="utf-8")
    print("=== baselines summary ===")
    print(summ)


if __name__ == "__main__":
    main()

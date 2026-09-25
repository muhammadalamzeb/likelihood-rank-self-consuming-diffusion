#!/usr/bin/env python3
"""Compute stratified retention, exact Wilcoxon, W2 CIs, kfrac table for paper revision."""
from __future__ import annotations

import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "experiments" / "logs" / "w2_fs.jsonl"
KLOG = ROOT / "experiments" / "logs" / "w2_fs_kfrac.jsonl"
PAPER = ROOT / "paper"
ANALYSIS = ROOT / "experiments" / "analysis"


def wilcoxon_exact(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "W": None, "p_exact": None, "p_approx": None}
    absd = np.abs(d)
    order = np.argsort(absd)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[order[j + 1]] == absd[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    signs = np.sign(d)
    w_obs = float(min(ranks[signs > 0].sum(), ranks[signs < 0].sum()))
    count = 0
    total = 1 << n
    for mask in range(total):
        s = np.array([1 if (mask >> k) & 1 else -1 for k in range(n)])
        w = min(ranks[s > 0].sum(), ranks[s < 0].sum())
        if w <= w_obs + 1e-12:
            count += 1
    p_exact = count / total
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_obs - mean + 0.5) / sqrt(var) if var > 0 else 0.0
    p_approx = float(erfc(abs(z) / sqrt(2.0)))
    return {"n": n, "W": w_obs, "p_exact": p_exact, "p_approx": p_approx}


def boot_ci(d: np.ndarray, rng: np.random.Generator, B: int = 20000):
    d = np.asarray(d, dtype=float)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return np.percentile(boots, [2.5, 97.5])


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    for rank, i in enumerate(order):
        adj[i] = min(1.0, ps[i] * (m - rank))
    for j in range(1, m):
        prev, cur = order[j - 1], order[j]
        adj[cur] = max(adj[cur], adj[prev])
    return adj


def load_jsonl(path: Path):
    text = path.read_text(encoding="utf-8")
    # handle accidental literal \n in file
    if "\\n{" in text and text.count("\n") < 5:
        text = text.replace("\\n", "\n")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def main():
    recs = load_jsonl(LOG)
    by = {(r["seed"], r["rho"], r["policy"], r["generation"]): r for r in recs}
    cells = sorted(
        {(r["seed"], r["rho"]) for r in recs if r["policy"] == "top_k" and r["generation"] == 0}
    )
    rng = np.random.default_rng(0)

    def retention(s, rho, p):
        m0 = by[(s, rho, p, 0)]["minority_mean"]
        m1 = by[(s, rho, p, 1)]["minority_mean"]
        return m1 / max(m0, 1e-9)

    # --- retention stratified ---
    ret_lines = [
        "stratum,n,contrast,mean_diff,std_diff,cohen_dz,boot_ci_low,boot_ci_high,wilcoxon_W,p_exact,p_approx_normal"
    ]
    for rho in [5.0, 10.0]:
        sub = [(s, rh) for s, rh in cells if rh == rho]
        for name, pa, pb in [("top_minus_rand", "top_k", "rand_k"), ("bottom_minus_rand", "bottom_k", "rand_k")]:
            d = np.array([retention(s, rho, pa) - retention(s, rho, pb) for s, _ in sub])
            lo, hi = boot_ci(d, rng)
            w = wilcoxon_exact(d)
            dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
            ret_lines.append(
                f"rho{int(rho)},{len(sub)},{name},{d.mean():.6f},{d.std(ddof=1):.6f},{dz:.4f},{lo:.6f},{hi:.6f},{w['W']},{w['p_exact']:.6f},{w['p_approx']:.6f}"
            )
            print(f"rho={rho} {name}: mean={d.mean():.4f} dz={dz:.3f} exact_p={w['p_exact']:.4f}")

    # pooled (for supplemental comparison only)
    for name, pa, pb in [("top_minus_rand", "top_k", "rand_k"), ("bottom_minus_rand", "bottom_k", "rand_k")]:
        d = np.array([retention(s, rho, pa) - retention(s, rho, pb) for s, rho in cells])
        lo, hi = boot_ci(d, rng)
        w = wilcoxon_exact(d)
        dz = float(d.mean() / d.std(ddof=1))
        ret_lines.append(
            f"pooled8,8,{name},{d.mean():.6f},{d.std(ddof=1):.6f},{dz:.4f},{lo:.6f},{hi:.6f},{w['W']},{w['p_exact']:.6f},{w['p_approx']:.6f}"
        )
        print(f"pooled {name}: mean={d.mean():.4f} dz={dz:.3f} exact_p={w['p_exact']:.4f}")

    # Holm within rho=5 for the two contrasts
    # (already only two)

    (PAPER / "table_stats_retention_stratified.csv").write_text("\n".join(ret_lines) + "\n", encoding="utf-8")
    (ANALYSIS / "table_stats_retention_stratified.csv").write_text("\n".join(ret_lines) + "\n", encoding="utf-8")

    # --- W2 by generation ---
    w2_lines = [
        "generation,mean_delta,std,cohen_dz,boot_ci_low,boot_ci_high,n_pos,n,p_exact,p_approx_normal,holm_p_exact"
    ]
    rows = []
    for g in range(1, 6):
        d = np.array(
            [
                by[(s, rho, "top_k", g)]["w2_sliced"] - by[(s, rho, "rand_k", g)]["w2_sliced"]
                for s, rho in cells
            ]
        )
        lo, hi = boot_ci(d, rng)
        w = wilcoxon_exact(d)
        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
        rows.append((g, d.mean(), d.std(ddof=1), dz, lo, hi, int((d > 0).sum()), w))
        print(
            f"W2 g={g}: mean={d.mean():.4f} dz={dz:.3f} npos={(d>0).sum()}/8 exact={w['p_exact']:.4f}"
        )

    holm_p = holm([r[7]["p_exact"] for r in rows])
    for i, (g, mean, std, dz, lo, hi, npos, w) in enumerate(rows):
        w2_lines.append(
            f"{g},{mean:.6f},{std:.6f},{dz:.4f},{lo:.6f},{hi:.6f},{npos},8,{w['p_exact']:.6f},{w['p_approx']:.6f},{holm_p[i]:.6f}"
        )

    (PAPER / "table_w2_top_minus_rand.csv").write_text("\n".join(w2_lines) + "\n", encoding="utf-8")
    (ANALYSIS / "table_w2_top_minus_rand.csv").write_text("\n".join(w2_lines) + "\n", encoding="utf-8")

    # --- kfrac ---
    krecs = load_jsonl(KLOG)
    kby = {}
    for r in krecs:
        kby[(r["seed"], r["k_frac"], r["policy"], r["generation"])] = r

    k_lines = ["k_frac,seed,policy,m0,m1,retention,order_bottom_gt_rand_gt_top"]
    summary = ["k_frac,policy,mean_retention,std,n,order_holds"]
    for kf in [0.25, 0.5, 0.75]:
        order_ok = 0
        rets = {"top_k": [], "rand_k": [], "bottom_k": []}
        for seed in [0, 1]:
            m1 = {p: kby[(seed, kf, p, 1)]["minority_mean"] for p in rets}
            m0 = {p: kby[(seed, kf, p, 0)]["minority_mean"] for p in rets}
            ok = m1["bottom_k"] > m1["rand_k"] > m1["top_k"]
            order_ok += int(ok)
            for p in rets:
                r = m1[p] / max(m0[p], 1e-9)
                rets[p].append(r)
                k_lines.append(f"{kf},{seed},{p},{m0[p]},{m1[p]},{r},{int(ok)}")
        for p in rets:
            v = np.array(rets[p])
            summary.append(
                f"{kf},{p},{v.mean():.4f},{v.std(ddof=1):.4f},{len(v)},{order_ok}/2"
            )
        print(f"kfrac={kf}: order {order_ok}/2 means bottom/rand/top={np.mean(rets['bottom_k']):.3f}/{np.mean(rets['rand_k']):.3f}/{np.mean(rets['top_k']):.3f}")

    (PAPER / "table_kfrac_retention.csv").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (PAPER / "table_kfrac_by_seed.csv").write_text("\n".join(k_lines) + "\n", encoding="utf-8")
    (ANALYSIS / "table_kfrac_retention.csv").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (ANALYSIS / "table_kfrac_by_seed.csv").write_text("\n".join(k_lines) + "\n", encoding="utf-8")
    print("wrote all tables")


if __name__ == "__main__":
    main()

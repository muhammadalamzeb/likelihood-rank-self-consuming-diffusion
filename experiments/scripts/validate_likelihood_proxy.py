#!/usr/bin/env python3
"""
Validate denoising-MSE proxy s(x) against exact GMM log-density.
Experiment ID: w2_fs_proxy_val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_w2_false_stability import elbo_proxy, make_gmm, sample, train_ddpm  # noqa: E402


def gmm_logpdf(X: np.ndarray, means: np.ndarray, weights: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Exact mixture log-density under isotropic Gaussian components."""
    # X: [N,2], means: [K,2]
    var = sigma**2
    log_norm = -np.log(2 * np.pi * var)  # 2D isotropic: -d/2 log(2 pi sigma^2) with d=2 → -log(2 pi var)
    # component log dens: log_norm - ||x-mu||^2 / (2 var)
    d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)  # [N,K]
    log_comp = log_norm - 0.5 * d2 / var + np.log(weights[None, :])
    m = log_comp.max(axis=1, keepdims=True)
    return (m.squeeze(1) + np.log(np.exp(log_comp - m).sum(axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rho", type=float, default=5.0)
    ap.add_argument("--n-data", type=int, default=1500)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--n-t", type=int, default=6)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--delta", type=float, default=1.5)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--k-frac", type=float, default=0.5)
    args = ap.parse_args()

    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    (args.out / "analysis").mkdir(parents=True, exist_ok=True)
    (args.out / "configs").mkdir(parents=True, exist_ok=True)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["experiment"] = "w2_fs_proxy_val"
    with open(args.out / "configs" / "w2_fs_proxy_val.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    rows = []
    for seed in args.seeds:
        X, _, means, weights = make_gmm(args.n_data, K=args.K, delta=args.delta, rho=args.rho, seed=seed)
        model = train_ddpm(X, seed=seed, steps=args.train_steps)
        synth = sample(model, args.n_eval, n_steps=args.n_steps, seed=seed + 101)
        s = elbo_proxy(model, synth, n_t=args.n_t)  # lower = higher likelihood
        logp = gmm_logpdf(synth, means, weights, sigma=args.sigma)
        neglogp = -logp
        # expect positive Spearman: high s ↔ high -logp (low density)
        rho_s, p_s = spearmanr(s, neglogp)
        k = max(1, int(round(args.k_frac * len(synth))))
        # top likelihood = lowest s / lowest -logp
        top_s = set(np.argsort(s)[:k].tolist())
        top_true = set(np.argsort(neglogp)[:k].tolist())
        bot_s = set(np.argsort(s)[-k:].tolist())
        bot_true = set(np.argsort(neglogp)[-k:].tolist())
        j_top = len(top_s & top_true) / k
        j_bot = len(bot_s & bot_true) / k
        rec = {
            "experiment": "w2_fs_proxy_val",
            "seed": seed,
            "rho": args.rho,
            "spearman_s_vs_neglogp": float(rho_s),
            "spearman_p": float(p_s),
            "top_k_jaccard": float(j_top),
            "bottom_k_jaccard": float(j_bot),
            "k": k,
            "n_eval": int(len(synth)),
            "mean_s": float(s.mean()),
            "mean_neglogp": float(neglogp.mean()),
        }
        rows.append(rec)
        print(json.dumps(rec), flush=True)

    log_path = args.out / "logs" / "w2_fs_proxy_val.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    sp = np.array([r["spearman_s_vs_neglogp"] for r in rows])
    jt = np.array([r["top_k_jaccard"] for r in rows])
    jb = np.array([r["bottom_k_jaccard"] for r in rows])
    summary = {
        "n_seeds": len(rows),
        "spearman_mean": float(sp.mean()),
        "spearman_std": float(sp.std(ddof=1)) if len(sp) > 1 else 0.0,
        "spearman_min": float(sp.min()),
        "top_k_jaccard_mean": float(jt.mean()),
        "bottom_k_jaccard_mean": float(jb.mean()),
    }
    with open(args.out / "analysis" / "proxy_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # CSV for paper
    lines = ["seed,spearman_s_vs_neglogp,top_k_jaccard,bottom_k_jaccard"]
    for r in rows:
        lines.append(
            f"{r['seed']},{r['spearman_s_vs_neglogp']:.6f},{r['top_k_jaccard']:.6f},{r['bottom_k_jaccard']:.6f}"
        )
    lines.append(
        f"mean,{summary['spearman_mean']:.6f},{summary['top_k_jaccard_mean']:.6f},{summary['bottom_k_jaccard_mean']:.6f}"
    )
    csv_text = "\n".join(lines) + "\n"
    (args.out / "analysis" / "table_proxy_validation.csv").write_text(csv_text, encoding="utf-8")
    Path("paper/table_proxy_validation.csv").write_text(csv_text, encoding="utf-8")
    print("summary", json.dumps(summary), flush=True)
    print(f"wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()

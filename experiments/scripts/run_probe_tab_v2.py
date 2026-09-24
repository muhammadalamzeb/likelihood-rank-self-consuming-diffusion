#!/usr/bin/env python3
"""
Strategy B stack v2 — tabular Conditional VAE on sklearn Wine (observation only).
No pre-trained CIFAR/image ODE stack. Train ~1–2 min on CPU.

Probes:
  T1  Rare-class collapse under imbalance
  T2  Feature-support leakage (samples outside train min/max)
  T3  Pairwise correlation sign flips vs real data

Usage:
  .venv\\Scripts\\python experiments/scripts/run_probe_tab_v2.py --quick --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CVAE(nn.Module):
    def __init__(self, x_dim: int, n_classes: int, z_dim: int = 8, h: int = 64):
        super().__init__()
        self.n_classes = n_classes
        self.enc = nn.Sequential(
            nn.Linear(x_dim + n_classes, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.mu = nn.Linear(h, z_dim)
        self.logvar = nn.Linear(h, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim + n_classes, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, x_dim),
        )

    def encode(self, x, y_oh):
        h = self.enc(torch.cat([x, y_oh], dim=-1))
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z, y_oh):
        return self.dec(torch.cat([z, y_oh], dim=-1))

    def forward(self, x, y_oh):
        mu, logvar = self.encode(x, y_oh)
        z = self.reparam(mu, logvar)
        return self.decode(z, y_oh), mu, logvar


def one_hot(y: torch.Tensor, n: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(y, n).float()


def corr_sign_flips(a: np.ndarray, b: np.ndarray, thr: float = 0.15) -> int:
    """Count pairs where |corr| > thr in real and sign(corr) differs in synth."""
    d = a.shape[1]
    flips = 0
    checked = 0
    for i in range(d):
        for j in range(i + 1, d):
            cr = np.corrcoef(a[:, i], a[:, j])[0, 1]
            cs = np.corrcoef(b[:, i], b[:, j])[0, 1]
            if abs(cr) < thr:
                continue
            checked += 1
            if np.sign(cr) != np.sign(cs) and abs(cs) > 0.05:
                flips += 1
    return flips, checked


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("experiments"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick and args.epochs == 200:
        args.epochs = 80

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    data = load_wine()
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    # T1: artificial imbalance — drop 80% of class 2 from training
    mask = ~((y_tr == 2) & (np.random.rand(len(y_tr)) < 0.8))
    X_imb, y_imb = X_tr_s[mask], y_tr[mask]
    counts = {int(c): int((y_imb == c).sum()) for c in range(3)}

    device = torch.device("cpu")
    x_dim, n_classes = X.shape[1], 3
    model = CVAE(x_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    xt = torch.from_numpy(X_imb)
    yt = torch.from_numpy(y_imb)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        losses = []
        for i in range(0, len(xt), 32):
            idx = perm[i : i + 32]
            x, yb = xt[idx], yt[idx]
            y_oh = one_hot(yb, n_classes)
            recon, mu, logvar = model(x, y_oh)
            recon_loss = nn.functional.mse_loss(recon, x, reduction="sum") / len(x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.1 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if (ep + 1) % 20 == 0:
            print(f"epoch {ep+1} loss={np.mean(losses):.4f}")

    model.eval()
    n_per = 200
    with torch.no_grad():
        samples = {}
        for c in range(n_classes):
            z = torch.randn(n_per, 8)
            y_oh = one_hot(torch.full((n_per,), c), n_classes)
            xs = model.decode(z, y_oh).numpy()
            samples[c] = xs

    # Metrics in original feature space
    real_orig = scaler.inverse_transform(X_tr_s)
    lo, hi = real_orig.min(axis=0), real_orig.max(axis=0)
    # slight padding
    pad = 0.05 * (hi - lo + 1e-8)
    lo, hi = lo - pad, hi + pad

    t1 = {}
    t2 = {}
    t3 = {}
    for c in range(n_classes):
        syn = scaler.inverse_transform(samples[c])
        t1[c] = {
            "train_count_imbalanced": counts[c],
            "mean_l2_to_class_centroid": float(
                np.linalg.norm(syn.mean(0) - real_orig[y_tr == c].mean(0))
            ),
        }
        outside = ((syn < lo) | (syn > hi)).any(axis=1).mean()
        t2[c] = {"frac_any_feature_outside_train_support": float(outside)}
        flips, checked = corr_sign_flips(real_orig[y_tr == c], syn)
        t3[c] = {"corr_sign_flips": flips, "pairs_checked": checked}

    rec = {
        "probe": "TAB-v2",
        "stack": "v2",
        "dataset": "sklearn.wine",
        "seed": args.seed,
        "epochs": args.epochs,
        "imbalance_counts": counts,
        "T1_rare_class": t1,
        "T2_support_leakage": t2,
        "T3_corr_sign_flips": t3,
        "note": "Observation only; kill-test before any novelty claim.",
    }
    log_path = out / "logs" / f"tab_v2_seed{args.seed}.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    cfg_path = out / "configs" / f"tab_v2_seed{args.seed}.json"
    cfg_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    print("log", log_path)


if __name__ == "__main__":
    main()

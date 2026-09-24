#!/usr/bin/env python3
"""
Strategy B stack v3 — sklearn digits (offline). Tiny conditional VAE + pixel MLP-AR proxy.
Observation: β (KL weight) vs class-conditional sample accuracy / diversity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CVAE(nn.Module):
    def __init__(self, x_dim=64, n_classes=10, z_dim=16, h=128):
        super().__init__()
        self.n_classes = n_classes
        self.enc = nn.Sequential(nn.Linear(x_dim + n_classes, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, z_dim)
        self.logvar = nn.Linear(h, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim + n_classes, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, x_dim))

    def forward(self, x, y_oh, beta: float):
        h = self.enc(torch.cat([x, y_oh], -1))
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        recon = self.dec(torch.cat([z, y_oh], -1))
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, recon_loss + beta * kl, recon_loss.detach(), kl.detach()

    @torch.no_grad()
    def sample(self, y: torch.Tensor, n: int, z_dim=16):
        y_oh = torch.nn.functional.one_hot(y, self.n_classes).float()
        z = torch.randn(n, z_dim)
        return self.dec(torch.cat([z, y_oh], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 40

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    # Frozen evaluator on real data
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xtr, ytr)
    real_acc = float(clf.score(Xte, yte))

    betas = [0.01, 0.1, 1.0, 4.0]
    records = []
    for beta in betas:
        model = CVAE()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        xt = torch.from_numpy(Xtr)
        yt = torch.from_numpy(ytr)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 64):
                idx = perm[i : i + 64]
                x, yb = xt[idx], yt[idx]
                y_oh = torch.nn.functional.one_hot(yb, 10).float()
                _, loss, _, _ = model(x, y_oh, beta)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        syn_x, syn_y = [], []
        with torch.no_grad():
            for c in range(10):
                xs = model.sample(torch.full((100,), c), 100).numpy()
                syn_x.append(xs)
                syn_y.append(np.full(100, c))
        syn_x = np.concatenate(syn_x)
        syn_y = np.concatenate(syn_y)
        pred = clf.predict(syn_x)
        acc = float((pred == syn_y).mean())
        # diversity: mean pairwise L2 within class (higher = more diverse)
        divs = []
        for c in range(10):
            xc = syn_x[syn_y == c]
            # subsample pairs
            d = np.mean(np.linalg.norm(xc[None, :, :] - xc[:, None, :], axis=-1))
            divs.append(float(d))
        # class confusion: off-diagonal rate
        confuse = float((pred != syn_y).mean())
        rec = {
            "probe": "DIGITS-v3-beta",
            "seed": args.seed,
            "beta": beta,
            "epochs": args.epochs,
            "real_clf_acc": real_acc,
            "synth_class_acc": acc,
            "mean_within_class_l2": float(np.mean(divs)),
            "per_class_diversity": divs,
            "confusion_rate": confuse,
        }
        records.append(rec)
        print(rec)

    path = out / "logs" / f"digits_v3_seed{args.seed}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out / "configs" / f"digits_v3_seed{args.seed}.json").write_text(
        json.dumps({"records": records}, indent=2), encoding="utf-8"
    )
    print("wrote", path)


if __name__ == "__main__":
    main()

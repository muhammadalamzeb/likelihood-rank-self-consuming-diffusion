#!/usr/bin/env python3
"""
Strategy B stack v5 — Fashion-MNIST tiny conditional VAE.
Observation probes (not β — that is excl. 39 / stack v3):
  F1: train-time random-erase area fraction
  F2: sampling prior scale τ (z ~ N(0, τ²I))
  F3: per-class train count cap (few-shot regime)
No novelty claim until kill-test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torchvision import datasets, transforms


class CVAE(nn.Module):
    def __init__(self, x_dim=784, n_classes=10, z_dim=32, h=256):
        super().__init__()
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.enc = nn.Sequential(nn.Linear(x_dim + n_classes, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, z_dim)
        self.logvar = nn.Linear(h, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim + n_classes, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, x_dim)
        )

    def forward(self, x, y_oh, beta: float = 1.0):
        h = self.enc(torch.cat([x, y_oh], -1))
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        logits = self.dec(torch.cat([z, y_oh], -1))
        recon = nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return logits, recon + beta * kl, recon.detach(), kl.detach()

    @torch.no_grad()
    def sample(self, y: torch.Tensor, tau: float = 1.0):
        y_oh = torch.nn.functional.one_hot(y, self.n_classes).float()
        z = tau * torch.randn(y.shape[0], self.z_dim)
        return torch.sigmoid(self.dec(torch.cat([z, y_oh], -1)))


def random_erase_batch(x: torch.Tensor, max_area: float, rng: np.random.Generator) -> torch.Tensor:
    """x: (B, 784) in [0,1]. Erase a random rectangle with area ≤ max_area * H*W."""
    if max_area <= 0:
        return x
    B = x.shape[0]
    img = x.view(B, 1, 28, 28).clone()
    for i in range(B):
        area = float(rng.uniform(0.02, max_area))
        target = int(area * 28 * 28)
        for _ in range(10):
            ah = int(rng.integers(1, 28))
            aw = max(1, target // ah)
            if aw >= 28:
                continue
            top = int(rng.integers(0, 28 - ah + 1))
            left = int(rng.integers(0, 28 - aw + 1))
            img[i, :, top : top + ah, left : left + aw] = 0.0
            break
    return img.view(B, 784)


def load_fmnist(root: Path, seed: int):
    ds = datasets.FashionMNIST(root=str(root), train=True, download=True, transform=transforms.ToTensor())
    X = ds.data.numpy().astype(np.float32).reshape(-1, 784) / 255.0
    y = ds.targets.numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = 5000
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def subsample_per_class(X, y, cap: int, seed: int):
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(10):
        ix = np.where(y == c)[0]
        rng.shuffle(ix)
        parts.append(ix[: min(cap, len(ix))])
    sel = np.concatenate(parts)
    rng.shuffle(sel)
    return X[sel], y[sel]


def train_cvae(X, y, seed: int, epochs: int, erase: float, beta: float = 1.0) -> CVAE:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CVAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            b = perm[i : i + 128]
            xb = random_erase_batch(xt[b], erase, rng)
            yb = torch.nn.functional.one_hot(yt[b], 10).float()
            _, loss, _, _ = model(xb, yb, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def eval_gen(model: CVAE, clf: LogisticRegression, tau: float, n_per: int, seed: int) -> dict:
    torch.manual_seed(seed + 123)
    ys = torch.arange(10).repeat_interleave(n_per)
    samp = model.sample(ys, tau=tau).numpy()
    pred = clf.predict(samp)
    acc = float(np.mean(pred == ys.numpy()))
    # within-class pairwise L2 diversity
    divs = []
    for c in range(10):
        sc = samp[c * n_per : (c + 1) * n_per]
        d = 0.0
        m = 0
        for i in range(n_per):
            for j in range(i + 1, n_per):
                d += float(np.linalg.norm(sc[i] - sc[j]))
                m += 1
        divs.append(d / max(m, 1))
    return {
        "class_acc": acc,
        "mean_pairwise_l2": float(np.mean(divs)),
        "std_pairwise_l2": float(np.std(divs)),
        "samp_mean": float(samp.mean()),
        "samp_std": float(samp.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 8

    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xva, yva = load_fmnist(args.data_root, args.seed)
    clf = LogisticRegression(max_iter=400)
    # fit on subset for speed
    rng = np.random.default_rng(args.seed)
    fit_ix = rng.choice(len(Xtr), size=min(15000, len(Xtr)), replace=False)
    clf.fit(Xtr[fit_ix], ytr[fit_ix])
    real_acc = float(clf.score(Xva[:3000], yva[:3000]))

    cfg = {
        "stack": "v5_fmnist_cvae",
        "seed": args.seed,
        "epochs": args.epochs,
        "real_clf_acc": real_acc,
        "probes": ["F1_random_erase", "F2_prior_tau", "F3_per_class_cap"],
    }
    with open(out / "configs" / f"fmnist_v5_seed{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    records = []
    n_per = 20 if args.quick else 40

    # F1: random erase
    for erase in [0.0, 0.15, 0.35, 0.55]:
        model = train_cvae(Xtr[:8000], ytr[:8000], seed=args.seed, epochs=args.epochs, erase=erase)
        ev = eval_gen(model, clf, tau=1.0, n_per=n_per, seed=args.seed)
        rec = {"probe": "F1", "erase": erase, **ev}
        records.append(rec)
        print(json.dumps(rec))

    # F2: prior tau (fixed model, no erase)
    model = train_cvae(Xtr[:8000], ytr[:8000], seed=args.seed + 7, epochs=args.epochs, erase=0.0)
    for tau in [0.5, 1.0, 1.5, 2.5]:
        ev = eval_gen(model, clf, tau=tau, n_per=n_per, seed=args.seed)
        rec = {"probe": "F2", "tau": tau, **ev}
        records.append(rec)
        print(json.dumps(rec))

    # F3: per-class cap
    for cap in [50, 150, 500, 2000]:
        Xs, ys = subsample_per_class(Xtr, ytr, cap=cap, seed=args.seed)
        model = train_cvae(Xs, ys, seed=args.seed + 11, epochs=args.epochs, erase=0.0)
        ev = eval_gen(model, clf, tau=1.0, n_per=n_per, seed=args.seed)
        rec = {"probe": "F3", "per_class_cap": cap, "n_train": int(len(Xs)), **ev}
        records.append(rec)
        print(json.dumps(rec))

    log_path = out / "logs" / f"fmnist_v5_seed{args.seed}.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()

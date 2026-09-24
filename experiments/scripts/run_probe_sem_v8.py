#!/usr/bin/env python3
"""
Strategy B stack v8 — synthetic SEM tabular + tiny CVAE (offline).
SCM: Y = τ A + β·X + ε; A ~ Bern(sigmoid(γ·X)).
Probes:
  C1: true ATE τ sweep vs recovered ATE from synthetic data
  C2: CVAE latent dim (capacity) vs ATE error at fixed recon MSE
  C3: train size N vs ATE error
Observation only → expect kill vs causal-synth lit (Amad et al. 2026).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression


def simulate_sem(n: int, tau: float, seed: int, d: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    logits = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    A = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.float32)
    eps = rng.normal(scale=0.5, size=n).astype(np.float32)
    Y = (tau * A + 0.7 * X[:, 0] - 0.4 * X[:, 2] + eps).astype(np.float32)
    # features for VAE: [X, A, Y]
    data = np.concatenate([X, A[:, None], Y[:, None]], axis=1)
    return data, A, Y, X


def ate_ols(A, Y, X) -> float:
    # outcome regression with covariates
    Z = np.concatenate([A[:, None], X], axis=1)
    reg = LinearRegression().fit(Z, Y)
    return float(reg.coef_[0])


class TabCVAE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int = 8, h: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(nn.Linear(x_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, z_dim)
        self.logvar = nn.Linear(h, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, x_dim))

    def forward(self, x, beta: float = 1.0):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        recon = self.dec(z)
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, recon_loss + beta * kl, float(recon_loss.detach()), float(kl.detach())

    @torch.no_grad()
    def sample(self, n: int):
        z = torch.randn(n, self.z_dim)
        return self.dec(z).numpy()


def train(X: np.ndarray, z_dim: int, seed: int, epochs: int) -> tuple[TabCVAE, float]:
    torch.manual_seed(seed)
    model = TabCVAE(x_dim=X.shape[1], z_dim=z_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(X)
    last_recon = 0.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            b = xt[perm[i : i + 128]]
            _, loss, recon, _ = model(b)
            last_recon = recon
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, last_recon


def ate_from_synth(samp: np.ndarray, d: int = 4) -> float:
    X = samp[:, :d]
    A = (samp[:, d] > 0.5).astype(np.float64)
    Y = samp[:, d + 1].astype(np.float64)
    if A.sum() < 5 or (1 - A).sum() < 5:
        return float("nan")
    return ate_ols(A, Y, X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 35

    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    cfg = {
        "stack": "v8_sem_tab_cvae",
        "seed": args.seed,
        "epochs": args.epochs,
        "probes": ["C1_tau", "C2_zdim", "C3_n"],
    }
    with open(out / "configs" / f"sem_v8_seed{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    records = []
    n_samp = 2000

    # C1: true tau sweep
    for tau in [0.0, 0.5, 1.0, 2.0]:
        data, A, Y, X = simulate_sem(3000, tau=tau, seed=args.seed)
        real_ate = ate_ols(A, Y, X)
        model, recon = train(data, z_dim=8, seed=args.seed, epochs=args.epochs)
        samp = model.sample(n_samp)
        syn_ate = ate_from_synth(samp)
        rec = {
            "probe": "C1",
            "tau_true": tau,
            "ate_real": real_ate,
            "ate_synth": syn_ate,
            "ate_abs_err": float(abs(syn_ate - tau)) if syn_ate == syn_ate else None,
            "recon_mse": recon,
        }
        records.append(rec)
        print(json.dumps(rec))

    # C2: capacity
    data, A, Y, X = simulate_sem(3000, tau=1.0, seed=args.seed + 1)
    real_ate = ate_ols(A, Y, X)
    for zdim in [2, 4, 8, 16]:
        model, recon = train(data, z_dim=zdim, seed=args.seed + 1, epochs=args.epochs)
        samp = model.sample(n_samp)
        syn_ate = ate_from_synth(samp)
        rec = {
            "probe": "C2",
            "z_dim": zdim,
            "ate_real": real_ate,
            "ate_synth": syn_ate,
            "ate_abs_err": float(abs(syn_ate - 1.0)) if syn_ate == syn_ate else None,
            "recon_mse": recon,
        }
        records.append(rec)
        print(json.dumps(rec))

    # C3: N
    for n in [200, 500, 1500, 4000]:
        data, A, Y, X = simulate_sem(n, tau=1.0, seed=args.seed + 2)
        real_ate = ate_ols(A, Y, X)
        model, recon = train(data, z_dim=8, seed=args.seed + 2, epochs=args.epochs)
        samp = model.sample(n_samp)
        syn_ate = ate_from_synth(samp)
        rec = {
            "probe": "C3",
            "n": n,
            "ate_real": real_ate,
            "ate_synth": syn_ate,
            "ate_abs_err": float(abs(syn_ate - 1.0)) if syn_ate == syn_ate else None,
            "recon_mse": recon,
        }
        records.append(rec)
        print(json.dumps(rec))

    log_path = out / "logs" / f"sem_v8_seed{args.seed}.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()

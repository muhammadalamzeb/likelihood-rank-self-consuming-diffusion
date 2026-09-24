#!/usr/bin/env python3
"""
Strategy B stack v6 — synthetic 2D Gaussian mixtures + tiny MLP DDPM (offline).
Probes (observation only):
  M1: mode separation δ
  M2: number of modes K
  M3: mode-weight imbalance ratio ρ
Metrics: sample-to-mode assignment entropy, uncovered-mode rate, MMD proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def make_gmm(n: int, K: int, delta: float, rho: float, seed: int) -> np.ndarray:
    """K modes on a circle of radius delta; mode0 weight boosted by rho."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    means = np.stack([delta * np.cos(angles), delta * np.sin(angles)], axis=1)
    weights = np.ones(K, dtype=np.float64)
    weights[0] *= rho
    weights /= weights.sum()
    counts = rng.multinomial(n, weights)
    parts = []
    for k, c in enumerate(counts):
        if c == 0:
            continue
        parts.append(means[k] + 0.15 * rng.normal(size=(c, 2)))
    X = np.concatenate(parts, axis=0).astype(np.float32)
    rng.shuffle(X)
    return X, means.astype(np.float32)


class TinyDenoiser(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2)
        )

    def forward(self, x, t):
        # t in [0,1]
        return self.net(torch.cat([x, t], -1))


def q_sample(x0, t, noise):
    # linear α_bar = 1-t
    a = (1.0 - t).clamp(min=1e-4)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise, a


def train_ddpm(X: np.ndarray, seed: int, steps: int) -> TinyDenoiser:
    torch.manual_seed(seed)
    model = TinyDenoiser()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    xt = torch.from_numpy(X)
    for _ in range(steps):
        model.train()
        idx = torch.randint(0, len(xt), (256,))
        x0 = xt[idx]
        t = torch.rand(len(x0), 1)
        noise = torch.randn_like(x0)
        xt_noisy, _ = q_sample(x0, t, noise)
        pred = model(xt_noisy, t)
        loss = nn.functional.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def sample(model: TinyDenoiser, n: int, n_steps: int = 40, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    x = torch.randn(n, 2)
    for i in range(n_steps, 0, -1):
        t = torch.full((n, 1), i / float(n_steps))
        eps = model(x, t)
        a = (1.0 - t).clamp(min=1e-4)
        a_prev = torch.full((n, 1), max(1e-4, 1.0 - (i - 1) / float(n_steps)))
        x0_pred = (x - (1 - a).sqrt() * eps) / a.sqrt()
        if i > 1:
            noise = torch.randn_like(x)
            x = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * noise
        else:
            x = x0_pred
    return x.numpy()


def eval_modes(samples: np.ndarray, means: np.ndarray, assign_thresh: float = 0.6) -> dict:
    # nearest mode
    d = ((samples[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    nn = d.argmin(1)
    mind = d.min(1)
    covered = []
    for k in range(len(means)):
        covered.append(bool(np.any((nn == k) & (mind < assign_thresh**2))))
    # soft occupancy
    counts = np.bincount(nn, minlength=len(means)).astype(np.float64)
    p = counts / max(counts.sum(), 1)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    # MMD-like: mean distance of samples to nearest train-mode center
    return {
        "uncovered_modes": int(len(means) - sum(covered)),
        "coverage_rate": float(sum(covered) / len(means)),
        "assign_entropy": ent,
        "max_entropy": float(np.log(len(means))),
        "mean_nn_dist": float(np.sqrt(mind).mean()),
        "mode_count_std": float(counts.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.train_steps = 400

    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    cfg = {
        "stack": "v6_gmm_2d_ddpm",
        "seed": args.seed,
        "train_steps": args.train_steps,
        "probes": ["M1_separation", "M2_n_modes", "M3_imbalance"],
    }
    with open(out / "configs" / f"gmm_v6_seed{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    records = []
    n_data, n_samp = 2000, 500

    # M1: separation
    for delta in [0.4, 0.8, 1.5, 2.5]:
        X, means = make_gmm(n_data, K=4, delta=delta, rho=1.0, seed=args.seed)
        model = train_ddpm(X, seed=args.seed, steps=args.train_steps)
        samp = sample(model, n_samp, seed=args.seed + 1)
        ev = eval_modes(samp, means)
        rec = {"probe": "M1", "delta": delta, "K": 4, "rho": 1.0, **ev}
        records.append(rec)
        print(json.dumps(rec))

    # M2: number of modes
    for K in [2, 4, 6, 8]:
        X, means = make_gmm(n_data, K=K, delta=1.5, rho=1.0, seed=args.seed + 3)
        model = train_ddpm(X, seed=args.seed + 3, steps=args.train_steps)
        samp = sample(model, n_samp, seed=args.seed + 4)
        ev = eval_modes(samp, means)
        rec = {"probe": "M2", "delta": 1.5, "K": K, "rho": 1.0, **ev}
        records.append(rec)
        print(json.dumps(rec))

    # M3: imbalance
    for rho in [1.0, 3.0, 10.0, 30.0]:
        X, means = make_gmm(n_data, K=4, delta=1.5, rho=rho, seed=args.seed + 5)
        model = train_ddpm(X, seed=args.seed + 5, steps=args.train_steps)
        samp = sample(model, n_samp, seed=args.seed + 6)
        ev = eval_modes(samp, means)
        rec = {"probe": "M3", "delta": 1.5, "K": 4, "rho": rho, **ev}
        records.append(rec)
        print(json.dumps(rec))

    log_path = out / "logs" / f"gmm_v6_seed{args.seed}.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Digits pixel-space DDPM transfer (8x8 images, 64-D MLP denoiser).
Imbalanced class subset; mode masses via nearest empirical class mean in pixel space.
Experiment ID: w2_fs_digits_ddpm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits


def make_digits_pixels(n: int, classes: list[int], rho: float, seed: int):
    rng = np.random.default_rng(seed)
    digits = load_digits()
    X_raw = (digits.data.astype(np.float32) / 16.0)  # [0,1]
    y_raw = digits.target.astype(np.int64)
    mask = np.isin(y_raw, classes)
    X_raw, y_raw = X_raw[mask], y_raw[mask]
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
    K = len(classes)
    means = np.stack([X_raw[y == k].mean(axis=0) for k in range(K)], axis=0).astype(np.float32)
    weights = np.ones(K, dtype=np.float64)
    weights[0] *= rho
    weights /= weights.sum()
    counts = rng.multinomial(n, weights)
    parts, labels = [], []
    for k, c in enumerate(counts):
        pool = X_raw[y == k]
        if len(pool) == 0 or c == 0:
            continue
        idx = rng.choice(len(pool), size=c, replace=True)
        parts.append(pool[idx])
        labels.append(np.full(c, k, dtype=np.int64))
    X = np.concatenate(parts, axis=0)
    ylab = np.concatenate(labels, axis=0)
    perm = rng.permutation(len(X))
    return X[perm], ylab[perm], means, weights


class TinyDenoiser(nn.Module):
    def __init__(self, dim: int, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], -1))


def q_sample(x0, t, noise):
    a = (1.0 - t).clamp(min=1e-4)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise, a


def train_ddpm(X: np.ndarray, seed: int, steps: int, h: int) -> TinyDenoiser:
    torch.manual_seed(seed)
    dim = X.shape[1]
    model = TinyDenoiser(dim=dim, h=h)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    xt = torch.from_numpy(X.astype(np.float32))
    bs = min(256, len(xt))
    for _ in range(steps):
        model.train()
        idx = torch.randint(0, len(xt), (bs,))
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
def sample(model: TinyDenoiser, n: int, dim: int, n_steps: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    x = torch.randn(n, dim)
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
    return x.clamp(0, 1).numpy().astype(np.float32)


@torch.no_grad()
def elbo_proxy(model: TinyDenoiser, X: np.ndarray, n_t: int = 6, chunk: int = 256) -> np.ndarray:
    scores = []
    t_grid = torch.linspace(0.05, 0.95, n_t).view(-1, 1)
    for i in range(0, len(X), chunk):
        x0 = torch.from_numpy(X[i : i + chunk])
        errs = []
        for t1 in t_grid:
            t = t1.expand(len(x0), 1)
            noise = torch.randn_like(x0)
            xt, _ = q_sample(x0, t, noise)
            pred = model(xt, t)
            errs.append(((pred - noise) ** 2).mean(dim=1))
        scores.append(torch.stack(errs, 0).mean(0).numpy())
    return np.concatenate(scores, 0)


def mode_masses(X: np.ndarray, means: np.ndarray) -> np.ndarray:
    d = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    K = means.shape[0]
    return np.array([(lab == k).mean() for k in range(K)], dtype=np.float64)


def select_policy(synth: np.ndarray, scores: np.ndarray, k: int, policy: str, rng: np.random.Generator):
    n = len(synth)
    k = min(k, n)
    if policy == "top_k":
        idx = np.argsort(scores)[:k]
    elif policy == "bottom_k":
        idx = np.argsort(scores)[-k:]
    elif policy == "rand_k":
        idx = rng.choice(n, size=k, replace=False)
    else:
        raise ValueError(policy)
    return synth[idx]


def build_next_train(
    real_buf: np.ndarray,
    synth: np.ndarray,
    scores: np.ndarray,
    policy: str,
    alpha_real: float,
    k_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_target = len(real_buf)
    n_real = int(round(alpha_real * n_target))
    n_syn = n_target - n_real
    k = max(n_syn, int(round(k_frac * len(synth))))
    selected = select_policy(synth, scores, k=k, policy=policy, rng=rng)
    if len(selected) < n_syn:
        extra = synth[rng.choice(len(synth), size=n_syn - len(selected), replace=True)]
        selected = np.concatenate([selected, extra], axis=0)
    syn_idx = rng.choice(len(selected), size=n_syn, replace=False)
    real_idx = rng.choice(len(real_buf), size=n_real, replace=False)
    out = np.concatenate([real_buf[real_idx], selected[syn_idx]], axis=0)
    rng.shuffle(out)
    return out


def run_one(seed: int, rho: float, policy: str, args, classes: list[int]):
    rng = np.random.default_rng(seed)
    X0, _, means, true_w = make_digits_pixels(args.n_data, classes, rho, seed)
    real_buf = X0.copy()
    train = X0.copy()
    dim = train.shape[1]
    out = []
    for g in range(args.generations + 1):
        model = train_ddpm(train, seed=seed + 17 * g, steps=args.train_steps, h=args.hidden)
        samp = sample(model, args.n_synth, dim, args.n_steps, seed=seed + 101 * g)
        scores = elbo_proxy(model, samp, n_t=args.n_t)
        masses = mode_masses(samp, means)
        minority_mean = float(masses[1:].mean())
        rec = {
            "experiment": "w2_fs_digits_ddpm",
            "seed": seed,
            "rho": rho,
            "policy": policy,
            "generation": g,
            "minority_mean": minority_mean,
            "majority_mass": float(masses[0]),
            "mode_masses": masses.tolist(),
            "true_weights": true_w.tolist(),
        }
        out.append(rec)
        print(json.dumps({k: rec[k] for k in ("seed", "rho", "policy", "generation", "minority_mean", "majority_mass")}))
        if g == args.generations:
            break
        train = build_next_train(
            real_buf, samp, scores, policy, args.alpha_real, args.k_frac, rng
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rhos", type=float, nargs="+", default=[5.0])
    ap.add_argument("--policies", nargs="+", default=["top_k", "rand_k", "bottom_k"])
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--n-data", type=int, default=1200)
    ap.add_argument("--n-synth", type=int, default=1200)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--alpha-real", type=float, default=0.5)
    ap.add_argument("--k-frac", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-t", type=int, default=6)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--log-name", type=str, default="w2_fs_digits_ddpm.jsonl")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    (args.out / "configs").mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["experiment"] = "w2_fs_digits_ddpm"
    with open(args.out / "configs" / "w2_fs_digits_ddpm.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    all_recs = []
    for seed in args.seeds:
        for rho in args.rhos:
            for policy in args.policies:
                all_recs.extend(run_one(seed, rho, policy, args, list(args.classes)))

    path = args.out / "logs" / args.log_name
    mode = "a" if args.append and path.exists() else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path} n={len(all_recs)} mode={mode}")


if __name__ == "__main__":
    main()

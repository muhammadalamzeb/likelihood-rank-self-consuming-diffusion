#!/usr/bin/env python3
"""
W2-1 Digits-PCA transfer: imbalanced sklearn Digits in PCA space with tiny MLP DDPM.
Mode masses via nearest empirical class mean (no classifier on collapsed samples).
Experiment ID: w2_fs_digits_pca
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def make_digits_pca(
    n: int,
    classes: list[int],
    dim: int,
    rho: float,
    seed: int,
    pca: PCA | None = None,
    scaler: StandardScaler | None = None,
):
    """Build imbalanced Digits subset in PCA coords; return X, y, means, weights, fitted transforms."""
    rng = np.random.default_rng(seed)
    digits = load_digits()
    X_raw = digits.data.astype(np.float64)
    y_raw = digits.target.astype(np.int64)
    mask = np.isin(y_raw, classes)
    X_raw, y_raw = X_raw[mask], y_raw[mask]
    # Remap labels to 0..K-1 with majority = classes[0]
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)

    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_raw)
    else:
        Xs = scaler.transform(X_raw)
    if pca is None:
        pca = PCA(n_components=dim, random_state=seed)
        Xp = pca.fit_transform(Xs).astype(np.float32)
    else:
        Xp = pca.transform(Xs).astype(np.float32)

    K = len(classes)
    means = np.stack([Xp[y == k].mean(axis=0) for k in range(K)], axis=0).astype(np.float32)

    # Target multinomial weights: mode 0 boosted by rho
    weights = np.ones(K, dtype=np.float64)
    weights[0] *= rho
    weights /= weights.sum()
    counts = rng.multinomial(n, weights)

    parts, labels = [], []
    for k, c in enumerate(counts):
        pool = Xp[y == k]
        if len(pool) == 0 or c == 0:
            continue
        idx = rng.choice(len(pool), size=c, replace=True)
        parts.append(pool[idx])
        labels.append(np.full(c, k, dtype=np.int64))
    X = np.concatenate(parts, axis=0)
    ylab = np.concatenate(labels, axis=0)
    perm = rng.permutation(len(X))
    return X[perm], ylab[perm], means, weights, scaler, pca


class TinyDenoiser(nn.Module):
    def __init__(self, dim: int, h: int = 128):
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
    return x.numpy().astype(np.float32)


@torch.no_grad()
def elbo_proxy(model: TinyDenoiser, X: np.ndarray, n_t: int = 6, chunk: int = 256) -> np.ndarray:
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32))
    ts = torch.linspace(0.05, 0.95, n_t).view(1, n_t, 1)
    scores = np.zeros(len(xt), dtype=np.float64)
    for start in range(0, len(xt), chunk):
        x0 = xt[start : start + chunk]
        B = x0.shape[0]
        dim = x0.shape[1]
        x0e = x0.unsqueeze(1).expand(B, n_t, dim).reshape(B * n_t, dim)
        t = ts.expand(B, n_t, 1).reshape(B * n_t, 1)
        noise = torch.randn_like(x0e)
        xt_noisy, _ = q_sample(x0e, t, noise)
        pred = model(xt_noisy, t)
        mse = ((pred - noise) ** 2).mean(dim=1).view(B, n_t).mean(dim=1)
        scores[start : start + B] = mse.cpu().numpy()
    return scores


def occupancy(samples: np.ndarray, means: np.ndarray) -> np.ndarray:
    d = ((samples[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    nn = d.argmin(1)
    counts = np.bincount(nn, minlength=len(means)).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def wasserstein2_1d_proj(a: np.ndarray, b: np.ndarray, n_proj: int = 24, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    dims = a.shape[1]
    total = 0.0
    n = min(len(a), len(b), 400)
    a, b = a[:n], b[:n]
    for _ in range(n_proj):
        v = rng.normal(size=dims)
        v /= np.linalg.norm(v) + 1e-12
        pa = np.sort(a @ v)
        pb = np.sort(b @ v)
        total += float(np.mean((pa - pb) ** 2))
    return total / n_proj


def select_policy(synth, scores, k, policy, rng):
    k = min(k, len(synth))
    if policy == "top_k":
        idx = np.argsort(scores)[:k]
    elif policy == "bottom_k":
        idx = np.argsort(scores)[-k:]
    else:
        idx = rng.choice(len(synth), size=k, replace=False)
    return synth[idx]


def build_next_train(real_buf, synth, scores, policy, alpha_real, k_frac, rng):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--rho", type=float, default=5.0)
    ap.add_argument("--policies", nargs="+", default=["top_k", "rand_k", "bottom_k"])
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--n-data", type=int, default=1200)
    ap.add_argument("--n-synth", type=int, default=1200)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--alpha-real", type=float, default=0.5)
    ap.add_argument("--k-frac", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--log-name", type=str, default="w2_fs_digits_pca.jsonl")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["experiment"] = "w2_fs_digits_pca"
    cfg["note"] = "Mode masses = nearest empirical Digits class mean in PCA space"
    with open(out / "configs" / "w2_fs_digits_pca.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    all_recs = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        X0, _, means, true_w, scaler, pca = make_digits_pca(
            args.n_data, args.classes, args.dim, args.rho, seed
        )
        real_buf = X0.copy()
        X_hold, _, _, _, _, _ = make_digits_pca(
            args.n_data,
            args.classes,
            args.dim,
            args.rho,
            seed + 999,
            pca=pca,
            scaler=scaler,
        )
        for policy in args.policies:
            train = X0.copy()
            for g in range(args.generations + 1):
                model = train_ddpm(train, seed + 17 * g, args.train_steps, args.hidden)
                samp = sample(model, args.n_synth, args.dim, args.n_steps, seed + 101 * g)
                masses = occupancy(samp, means)
                rec = {
                    "experiment": "w2_fs_digits_pca",
                    "seed": seed,
                    "dim": args.dim,
                    "classes": args.classes,
                    "rho": args.rho,
                    "policy": policy,
                    "alpha_real": args.alpha_real,
                    "generation": g,
                    "m_min": float(masses.min()),
                    "minority_mean": float(masses[1:].mean()),
                    "majority_mass": float(masses[0]),
                    "mode_masses": masses.tolist(),
                    "true_weights": true_w.tolist(),
                    "w2_sliced": wasserstein2_1d_proj(samp, X_hold, seed=seed + g),
                    "pca_explained_var_sum": float(pca.explained_variance_ratio_.sum()),
                }
                all_recs.append(rec)
                print(
                    json.dumps(
                        {
                            k: rec[k]
                            for k in (
                                "seed",
                                "policy",
                                "generation",
                                "minority_mean",
                                "majority_mass",
                                "w2_sliced",
                            )
                        }
                    ),
                    flush=True,
                )
                if g == args.generations:
                    break
                scores = elbo_proxy(model, samp)
                train = build_next_train(
                    real_buf, samp, scores, policy, args.alpha_real, args.k_frac, rng
                )

    log_path = out / "logs" / args.log_name
    mode = "a" if args.append and log_path.exists() else "w"
    with open(log_path, mode, encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {log_path} n={len(all_recs)} mode={mode}", flush=True)


if __name__ == "__main__":
    main()

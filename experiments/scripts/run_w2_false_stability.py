#!/usr/bin/env python3
"""
W2-1 False stability: likelihood top-k vs random-k in self-consuming DDPM on imbalanced 2D GMM.

Hypothesis: top-k by denoising ELBO proxy can stabilize/improve aggregate W2 while
accelerating minority mode mass collapse vs size-matched random retention.

Experiment ID: w2_fs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def make_gmm(n: int, K: int, delta: float, rho: float, seed: int):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    means = np.stack([delta * np.cos(angles), delta * np.sin(angles)], axis=1).astype(np.float32)
    weights = np.ones(K, dtype=np.float64)
    weights[0] *= rho
    weights /= weights.sum()
    counts = rng.multinomial(n, weights)
    parts = []
    labels = []
    for k, c in enumerate(counts):
        if c == 0:
            continue
        parts.append(means[k] + 0.15 * rng.normal(size=(c, 2)).astype(np.float32))
        labels.append(np.full(c, k, dtype=np.int64))
    X = np.concatenate(parts, axis=0)
    y = np.concatenate(labels, axis=0)
    perm = rng.permutation(len(X))
    return X[perm], y[perm], means, weights.astype(np.float64)


class TinyDenoiser(nn.Module):
    def __init__(self, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], -1))


def q_sample(x0, t, noise):
    a = (1.0 - t).clamp(min=1e-4)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise, a


def train_ddpm(X: np.ndarray, seed: int, steps: int, h: int = 64) -> TinyDenoiser:
    torch.manual_seed(seed)
    model = TinyDenoiser(h=h)
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
    return x.numpy().astype(np.float32)


@torch.no_grad()
def elbo_proxy(model: TinyDenoiser, X: np.ndarray, n_t: int = 8, chunk: int = 256) -> np.ndarray:
    """Lower = better fit under denoising score-matching loss (likelihood proxy)."""
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32))
    ts = torch.linspace(0.05, 0.95, n_t).view(1, n_t, 1)
    scores = np.zeros(len(xt), dtype=np.float64)
    for start in range(0, len(xt), chunk):
        x0 = xt[start : start + chunk]  # [B,2]
        B = x0.shape[0]
        x0e = x0.unsqueeze(1).expand(B, n_t, 2).reshape(B * n_t, 2)
        t = ts.expand(B, n_t, 1).reshape(B * n_t, 1)
        noise = torch.randn_like(x0e)
        xt_noisy, _ = q_sample(x0e, t, noise)
        pred = model(xt_noisy, t)
        mse = ((pred - noise) ** 2).mean(dim=1).view(B, n_t).mean(dim=1)
        scores[start : start + B] = mse.cpu().numpy()
    return scores


def assign_modes(samples: np.ndarray, means: np.ndarray):
    d = ((samples[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    nn = d.argmin(1)
    mind = d.min(1)
    return nn, mind


def mode_masses(samples: np.ndarray, means: np.ndarray, thresh: float = 0.6) -> np.ndarray:
    nn, mind = assign_modes(samples, means)
    K = len(means)
    masses = np.zeros(K, dtype=np.float64)
    for k in range(K):
        masses[k] = float(np.mean((nn == k) & (mind < thresh**2)))
    # renormalize among assigned-in-ball; also report raw occupancy
    return masses


def occupancy(samples: np.ndarray, means: np.ndarray) -> np.ndarray:
    nn, _ = assign_modes(samples, means)
    counts = np.bincount(nn, minlength=len(means)).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def wasserstein2_1d_proj(a: np.ndarray, b: np.ndarray, n_proj: int = 16, seed: int = 0) -> float:
    """Sliced W2 approximation via 1D projections (CPU-friendly)."""
    rng = np.random.default_rng(seed)
    dims = a.shape[1]
    total = 0.0
    n = min(len(a), len(b), 400)
    a = a[:n]
    b = b[:n]
    for _ in range(n_proj):
        v = rng.normal(size=dims)
        v /= np.linalg.norm(v) + 1e-12
        pa = np.sort(a @ v)
        pb = np.sort(b @ v)
        total += float(np.mean((pa - pb) ** 2))
    return total / n_proj


def select_policy(synth: np.ndarray, scores: np.ndarray, k: int, policy: str, rng: np.random.Generator):
    n = len(synth)
    k = min(k, n)
    if policy == "top_k":
        idx = np.argsort(scores)[:k]  # lowest ELBO proxy
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
    if policy == "replace":
        return synth[:n_target].copy()
    if policy == "mix":
        n_real = int(round(alpha_real * n_target))
        n_syn = n_target - n_real
        syn_idx = rng.choice(len(synth), size=n_syn, replace=False)
        real_idx = rng.choice(len(real_buf), size=n_real, replace=False)
        out = np.concatenate([real_buf[real_idx], synth[syn_idx]], axis=0)
        rng.shuffle(out)
        return out
    # top_k / rand_k / bottom_k: mix fixed real fraction with selected synth
    n_real = int(round(alpha_real * n_target))
    n_syn = n_target - n_real
    k = max(n_syn, int(round(k_frac * len(synth))))
    selected = select_policy(synth, scores, k=k, policy=policy, rng=rng)
    if len(selected) < n_syn:
        # pad if needed
        extra = synth[rng.choice(len(synth), size=n_syn - len(selected), replace=True)]
        selected = np.concatenate([selected, extra], axis=0)
    syn_idx = rng.choice(len(selected), size=n_syn, replace=False)
    real_idx = rng.choice(len(real_buf), size=n_real, replace=False)
    out = np.concatenate([real_buf[real_idx], selected[syn_idx]], axis=0)
    rng.shuffle(out)
    return out


def run_one(seed: int, rho: float, policy: str, args) -> list[dict]:
    rng = np.random.default_rng(seed)
    X0, _, means, true_w = make_gmm(args.n_data, K=args.K, delta=args.delta, rho=rho, seed=seed)
    real_buf = X0.copy()
    # held-out real for metrics
    X_hold, _, _, _ = make_gmm(args.n_data, K=args.K, delta=args.delta, rho=rho, seed=seed + 999)
    train = X0.copy()
    records = []
    for g in range(args.generations + 1):
        model = train_ddpm(train, seed=seed + 17 * g, steps=args.train_steps, h=args.hidden)
        samp = sample(model, args.n_synth, n_steps=args.n_steps, seed=seed + 101 * g)
        masses = occupancy(samp, means)
        m_min = float(masses.min())
        m_max = float(masses.max())
        # minority = all non-boosted modes (modes 1..K-1); report mean and min
        minority_mean = float(masses[1:].mean()) if args.K > 1 else m_min
        w2 = wasserstein2_1d_proj(samp, X_hold, n_proj=24, seed=seed + g)
        rec = {
            "experiment": "w2_fs",
            "seed": seed,
            "rho": rho,
            "policy": policy,
            "generation": g,
            "m_min": m_min,
            "m_max": m_max,
            "minority_mean": minority_mean,
            "majority_mass": float(masses[0]),
            "mode_masses": masses.tolist(),
            "true_weights": true_w.tolist(),
            "w2_sliced": w2,
            "train_size": int(len(train)),
        }
        records.append(rec)
        print(json.dumps({k: rec[k] for k in ("seed", "rho", "policy", "generation", "m_min", "minority_mean", "majority_mass", "w2_sliced")}))
        if g == args.generations:
            break
        scores = elbo_proxy(model, samp, n_t=args.n_t)
        train = build_next_train(
            real_buf=real_buf,
            synth=samp,
            scores=scores,
            policy=policy,
            alpha_real=args.alpha_real,
            k_frac=args.k_frac,
            rng=rng,
        )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rhos", type=float, nargs="+", default=[5.0, 10.0])
    ap.add_argument(
        "--policies",
        nargs="+",
        default=["mix", "top_k", "rand_k", "replace"],
    )
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--n-data", type=int, default=1500)
    ap.add_argument("--n-synth", type=int, default=1500)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--delta", type=float, default=1.5)
    ap.add_argument("--alpha-real", type=float, default=0.5)
    ap.add_argument("--k-frac", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-t", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to existing w2_fs.jsonl instead of overwriting",
    )
    args = ap.parse_args()
    if args.quick:
        args.train_steps = 250
        args.generations = 4
        args.n_data = 800
        args.n_synth = 800
        args.seeds = args.seeds[:2]
        args.rhos = args.rhos[:1]

    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["experiment"] = "w2_fs"
    with open(out / "configs" / "w2_fs.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    all_recs = []
    for seed in args.seeds:
        for rho in args.rhos:
            for policy in args.policies:
                all_recs.extend(run_one(seed, rho, policy, args))

    log_path = out / "logs" / "w2_fs.jsonl"
    mode = "a" if args.append and log_path.exists() else "w"
    with open(log_path, mode, encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {log_path} n={len(all_recs)} mode={mode}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CIFAR-10 imbalanced self-consuming DDPM with a tiny U-Net denoiser.
Experiment ID: w2_fs_cifar
Mode masses: nearest empirical class mean in pixel space (4-class subset).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def load_cifar_pool(data_root: Path, classes: list[int]):
    tfm = transforms.Compose([transforms.ToTensor()])  # [0,1], CHW
    ds = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tfm)
    xs, ys = [], []
    remap = {c: i for i, c in enumerate(classes)}
    for x, y in ds:
        if int(y) in remap:
            xs.append(x.numpy().astype(np.float32))
            ys.append(remap[int(y)])
    X = np.stack(xs, axis=0)  # [N,3,32,32]
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def make_imbalanced(X_pool: np.ndarray, y_pool: np.ndarray, n: int, rho: float, seed: int):
    rng = np.random.default_rng(seed)
    K = int(y_pool.max()) + 1
    means = np.stack([X_pool[y_pool == k].mean(axis=0) for k in range(K)], axis=0).astype(np.float32)
    weights = np.ones(K, dtype=np.float64)
    weights[0] *= rho
    weights /= weights.sum()
    counts = rng.multinomial(n, weights)
    parts, labels = [], []
    for k, c in enumerate(counts):
        pool = X_pool[y_pool == k]
        if c == 0 or len(pool) == 0:
            continue
        idx = rng.choice(len(pool), size=c, replace=True)
        parts.append(pool[idx])
        labels.append(np.full(c, k, dtype=np.int64))
    X = np.concatenate(parts, axis=0)
    y = np.concatenate(labels, axis=0)
    perm = rng.permutation(len(X))
    return X[perm], y[perm], means, weights


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + h)


class TinyUNet(nn.Module):
    """Very small U-Net for 32x32 RGB; t embedded as channel bias."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.base = base
        self.in_conv = nn.Conv2d(3 + 1, base, 3, padding=1)  # +1 time channel
        self.down1 = nn.Sequential(ResBlock(base), nn.Conv2d(base, base * 2, 4, stride=2, padding=1))
        self.mid = ResBlock(base * 2)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), ResBlock(base))
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x, t):
        # x: [B,3,H,W], t: [B,1] in (0,1]
        B, _, H, W = x.shape
        tmap = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        h = torch.cat([x, tmap], dim=1)
        h0 = F.silu(self.in_conv(h))
        h1 = self.down1(h0)
        h2 = self.mid(h1)
        h3 = self.up1(h2)
        return self.out(h3 + h0)


def q_sample(x0, t, noise):
    # t: [B,1,1,1]
    a = (1.0 - t).clamp(min=1e-4)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise, a


def train_ddpm(X: np.ndarray, seed: int, steps: int, base: int, lr: float, device: torch.device) -> TinyUNet:
    torch.manual_seed(seed)
    model = TinyUNet(base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(X.astype(np.float32))
    bs = min(64, len(xt))
    for _ in range(steps):
        model.train()
        idx = torch.randint(0, len(xt), (bs,))
        x0 = xt[idx].to(device)
        t = torch.rand(bs, 1, 1, 1, device=device)
        noise = torch.randn_like(x0)
        xt_noisy, _ = q_sample(x0, t, noise)
        t_flat = t.view(bs, 1)
        pred = model(xt_noisy, t_flat)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def sample(model: TinyUNet, n: int, n_steps: int, seed: int, device: torch.device) -> np.ndarray:
    torch.manual_seed(seed)
    model.eval()
    x = torch.randn(n, 3, 32, 32, device=device)
    bs = 64
    outs = []
    for start in range(0, n, bs):
        xb = x[start : start + bs]
        B = xb.shape[0]
        for i in range(n_steps, 0, -1):
            tval = i / float(n_steps)
            t = torch.full((B, 1), tval, device=device)
            eps = model(xb, t)
            a = torch.full((B, 1, 1, 1), max(1e-4, 1.0 - tval), device=device)
            a_prev = torch.full((B, 1, 1, 1), max(1e-4, 1.0 - (i - 1) / float(n_steps)), device=device)
            x0_pred = (xb - (1 - a).sqrt() * eps) / a.sqrt()
            if i > 1:
                noise = torch.randn_like(xb)
                xb = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * noise
            else:
                xb = x0_pred
        outs.append(xb.clamp(0, 1).cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def elbo_proxy(model: TinyUNet, X: np.ndarray, n_t: int, chunk: int, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    t_grid = torch.linspace(0.05, 0.95, n_t, device=device)
    for i in range(0, len(X), chunk):
        x0 = torch.from_numpy(X[i : i + chunk]).to(device)
        B = x0.shape[0]
        errs = []
        for tv in t_grid:
            t = torch.full((B, 1), float(tv), device=device)
            t4 = t.view(B, 1, 1, 1)
            noise = torch.randn_like(x0)
            xt, _ = q_sample(x0, t4, noise)
            pred = model(xt, t)
            errs.append(((pred - noise) ** 2).mean(dim=(1, 2, 3)))
        scores.append(torch.stack(errs, 0).mean(0).cpu().numpy())
    return np.concatenate(scores, 0)


def mode_masses(X: np.ndarray, means: np.ndarray) -> np.ndarray:
    # X, means: [N,3,32,32] / [K,3,32,32]
    flat = X.reshape(len(X), -1)
    mflat = means.reshape(len(means), -1)
    d = ((flat[:, None, :] - mflat[None, :, :]) ** 2).sum(-1)
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


def run_one(seed, rho, policy, args, X_pool, y_pool, device):
    rng = np.random.default_rng(seed)
    X0, _, means, true_w = make_imbalanced(X_pool, y_pool, args.n_data, rho, seed)
    real_buf = X0.copy()
    train = X0.copy()
    out = []
    for g in range(args.generations + 1):
        model = train_ddpm(train, seed + 17 * g, args.train_steps, args.base, args.lr, device)
        samp = sample(model, args.n_synth, args.n_steps, seed + 101 * g, device)
        scores = elbo_proxy(model, samp, args.n_t, args.chunk, device)
        masses = mode_masses(samp, means)
        minority_mean = float(masses[1:].mean())
        rec = {
            "experiment": "w2_fs_cifar",
            "seed": seed,
            "rho": rho,
            "policy": policy,
            "generation": g,
            "minority_mean": minority_mean,
            "majority_mass": float(masses[0]),
            "mode_masses": masses.tolist(),
            "true_weights": true_w.tolist(),
            "classes": args.classes,
        }
        out.append(rec)
        print(
            json.dumps(
                {k: rec[k] for k in ("seed", "rho", "policy", "generation", "minority_mean", "majority_mass")}
            ),
            flush=True,
        )
        if g == args.generations:
            break
        train = build_next_train(real_buf, samp, scores, policy, args.alpha_real, args.k_frac, rng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--data-root", type=Path, default=Path("experiments/data/cifar10"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rhos", type=float, nargs="+", default=[5.0])
    ap.add_argument("--policies", nargs="+", default=["top_k", "rand_k", "bottom_k"])
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--n-data", type=int, default=1024)
    ap.add_argument("--n-synth", type=int, default=1024)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--alpha-real", type=float, default=0.5)
    ap.add_argument("--k-frac", type=float, default=0.5)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-t", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--log-name", type=str, default="w2_fs_cifar.jsonl")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    (args.out / "configs").mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["experiment"] = "w2_fs_cifar"
    cfg["device"] = str(device)
    with open(args.out / "configs" / "w2_fs_cifar.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("loading CIFAR-10 pool...", flush=True)
    X_pool, y_pool = load_cifar_pool(args.data_root, list(args.classes))
    print(f"pool n={len(X_pool)} classes={args.classes}", flush=True)

    path = args.out / "logs" / args.log_name
    mode = "a" if args.append and path.exists() else "w"
    n_wrote = 0
    with open(path, mode, encoding="utf-8") as f:
        for seed in args.seeds:
            for rho in args.rhos:
                for policy in args.policies:
                    for r in run_one(seed, rho, policy, args, X_pool, y_pool, device):
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        n_wrote += 1
    print(f"wrote {path} n={n_wrote} mode={mode}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Strategy B stack v7 — synthetic class-conditional 'spectrogram' patches (offline).
Each digit class = harmonic template (f0, overtones) → 16x16 log-mag image.
Probes:
  S1: train additive SNR (dB)
  S2: class f0 jitter σ (Hz)
  S3: time-warp stretch range
Observation only → kill-test after logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


def synth_patch(
    cls: int,
    rng: np.random.Generator,
    snr_db: float = 20.0,
    f0_jitter: float = 0.0,
    warp: float = 0.0,
    side: int = 16,
) -> np.ndarray:
    """Build a crude time-frequency patch for class 0..9."""
    t = np.linspace(0, 1, side, dtype=np.float64)
    if warp > 0:
        w = 1.0 + float(rng.uniform(-warp, warp))
        t = np.clip(t * w, 0, 1)
    f0 = 80.0 + 35.0 * cls
    if f0_jitter > 0:
        f0 = f0 + float(rng.normal(0, f0_jitter))
    # frequency axis as rows
    freqs = np.linspace(40, 500, side)
    mag = np.zeros((side, side), dtype=np.float64)
    for h in (1, 2, 3):
        fh = f0 * h
        # Gaussian ridge in frequency, amplitude envelope in time
        env = 0.5 + 0.5 * np.sin(2 * np.pi * (cls + 1) * t / 10.0)
        ridge = np.exp(-0.5 * ((freqs - fh) / 18.0) ** 2)
        mag += np.outer(ridge, env)
    # noise
    signal_pow = np.mean(mag**2) + 1e-12
    snr_lin = 10 ** (snr_db / 10.0)
    noise_pow = signal_pow / snr_lin
    mag = mag + rng.normal(0, np.sqrt(noise_pow), size=mag.shape)
    mag = np.log1p(np.maximum(mag, 0))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag.astype(np.float32).reshape(-1)


def make_dataset(n_per: int, snr_db: float, f0_jitter: float, warp: float, seed: int):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in range(10):
        for _ in range(n_per):
            Xs.append(synth_patch(c, rng, snr_db=snr_db, f0_jitter=f0_jitter, warp=warp))
            ys.append(c)
    X = np.stack(Xs)
    y = np.array(ys, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


class CVAE(nn.Module):
    def __init__(self, x_dim=256, n_classes=10, z_dim=16, h=128):
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
        recon = self.dec(torch.cat([z, y_oh], -1))
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, recon_loss + beta * kl, recon_loss.detach(), kl.detach()

    @torch.no_grad()
    def sample(self, y: torch.Tensor):
        y_oh = torch.nn.functional.one_hot(y, self.n_classes).float()
        z = torch.randn(y.shape[0], self.z_dim)
        return self.dec(torch.cat([z, y_oh], -1)).clamp(0, 1)


def train_cvae(X, y, seed: int, epochs: int) -> CVAE:
    torch.manual_seed(seed)
    model = CVAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i : i + 64]
            yb = torch.nn.functional.one_hot(yt[b], 10).float()
            _, loss, _, _ = model(xt[b], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def eval_gen(model: CVAE, clf: LogisticRegression, n_per: int, seed: int) -> dict:
    torch.manual_seed(seed + 99)
    ys = torch.arange(10).repeat_interleave(n_per)
    samp = model.sample(ys).numpy()
    pred = clf.predict(samp)
    acc = float(np.mean(pred == ys.numpy()))
    divs = []
    for c in range(10):
        sc = samp[c * n_per : (c + 1) * n_per]
        d = 0.0
        m = 0
        for i in range(min(n_per, 15)):
            for j in range(i + 1, min(n_per, 15)):
                d += float(np.linalg.norm(sc[i] - sc[j]))
                m += 1
        divs.append(d / max(m, 1))
    return {
        "class_acc": acc,
        "mean_pairwise_l2": float(np.mean(divs)),
        "samp_std": float(samp.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 25

    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    n_per_train = 80 if args.quick else 120
    n_eval = 20

    cfg = {
        "stack": "v7_synth_spec_cvae",
        "seed": args.seed,
        "epochs": args.epochs,
        "probes": ["S1_snr", "S2_f0_jitter", "S3_time_warp"],
    }
    with open(out / "configs" / f"spec_v7_seed{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    records = []

    # clean eval classifier trained on high-SNR unjittered data
    Xc, yc = make_dataset(n_per_train, snr_db=30.0, f0_jitter=0.0, warp=0.0, seed=args.seed + 1000)
    clf = LogisticRegression(max_iter=400)
    clf.fit(Xc, yc)

    for snr in [30.0, 10.0, 0.0, -5.0]:
        X, y = make_dataset(n_per_train, snr_db=snr, f0_jitter=0.0, warp=0.0, seed=args.seed)
        model = train_cvae(X, y, seed=args.seed, epochs=args.epochs)
        ev = eval_gen(model, clf, n_eval, seed=args.seed)
        rec = {"probe": "S1", "snr_db": snr, **ev}
        records.append(rec)
        print(json.dumps(rec))

    for jit in [0.0, 5.0, 15.0, 40.0]:
        X, y = make_dataset(n_per_train, snr_db=25.0, f0_jitter=jit, warp=0.0, seed=args.seed + 2)
        model = train_cvae(X, y, seed=args.seed + 2, epochs=args.epochs)
        ev = eval_gen(model, clf, n_eval, seed=args.seed)
        rec = {"probe": "S2", "f0_jitter": jit, **ev}
        records.append(rec)
        print(json.dumps(rec))

    for warp in [0.0, 0.1, 0.25, 0.5]:
        X, y = make_dataset(n_per_train, snr_db=25.0, f0_jitter=0.0, warp=warp, seed=args.seed + 4)
        model = train_cvae(X, y, seed=args.seed + 4, epochs=args.epochs)
        ev = eval_gen(model, clf, n_eval, seed=args.seed)
        rec = {"probe": "S3", "warp": warp, **ev}
        records.append(rec)
        print(json.dumps(rec))

    log_path = out / "logs" / f"spec_v7_seed{args.seed}.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()

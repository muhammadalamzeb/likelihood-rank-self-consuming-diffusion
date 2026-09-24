#!/usr/bin/env python3
"""
W2-1 Digits transfer: self-consuming CVAE on sklearn digits with likelihood selection.
Experiment ID: w2_fs_digits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TinyCVAE(nn.Module):
    def __init__(self, d_in=64, z=8, h=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.mu = nn.Linear(h, z)
        self.logvar = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d_in))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar


def elbo_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="none").sum(dim=1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return recon_loss + kl


def train_cvae(X: np.ndarray, seed: int, steps: int) -> TinyCVAE:
    torch.manual_seed(seed)
    model = TinyCVAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(X.astype(np.float32))
    bs = min(128, len(xt))
    for _ in range(steps):
        model.train()
        idx = torch.randint(0, len(xt), (bs,))
        x = xt[idx]
        recon, mu, logvar = model(x)
        loss = elbo_loss(recon, x, mu, logvar).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def sample_cvae(model: TinyCVAE, n: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    z = torch.randn(n, 8)
    return model.dec(z).numpy().astype(np.float32)


@torch.no_grad()
def nll_proxy(model: TinyCVAE, X: np.ndarray, chunk: int = 256) -> np.ndarray:
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32))
    scores = np.zeros(len(xt), dtype=np.float64)
    for start in range(0, len(xt), chunk):
        x = xt[start : start + chunk]
        recon, mu, logvar = model(x)
        scores[start : start + len(x)] = elbo_loss(recon, x, mu, logvar).cpu().numpy()
    return scores


def class_masses(samples: np.ndarray, clf, scaler) -> np.ndarray:
    pred = clf.predict(scaler.transform(samples))
    counts = np.bincount(pred, minlength=10).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def select(synth, scores, k, policy, rng):
    k = min(k, len(synth))
    if policy == "top_k":
        idx = np.argsort(scores)[:k]
    elif policy == "bottom_k":
        idx = np.argsort(scores)[-k:]
    else:
        idx = rng.choice(len(synth), size=k, replace=False)
    return synth[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--policies", nargs="+", default=["top_k", "rand_k", "bottom_k", "mix"])
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--n-synth", type=int, default=1000)
    ap.add_argument("--alpha-real", type=float, default=0.5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.train_steps = 200
        args.generations = 3
        args.seeds = args.seeds[:2]

    digits = load_digits()
    X_all = digits.data.astype(np.float32) / 16.0
    y_all = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3, random_state=0, stratify=y_all)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    clf = LogisticRegression(max_iter=500, random_state=0)
    clf.fit(X_tr_s, y_tr)

    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    with open(out / "configs" / "w2_fs_digits.json", "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=2)

    records = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        real_buf = X_tr.copy()
        for policy in args.policies:
            train = real_buf.copy()
            for g in range(args.generations + 1):
                model = train_cvae(train, seed=seed + 11 * g, steps=args.train_steps)
                samp = sample_cvae(model, args.n_synth, seed=seed + 33 * g)
                masses = class_masses(samp, clf, scaler)
                # rare classes = those with lowest true prior on train
                true = np.bincount(y_tr, minlength=10).astype(np.float64)
                true /= true.sum()
                rare = np.argsort(true)[:3]
                maj = int(np.argmax(true))
                rec = {
                    "experiment": "w2_fs_digits",
                    "seed": seed,
                    "policy": policy,
                    "generation": g,
                    "rare_mean_mass": float(masses[rare].mean()),
                    "majority_mass": float(masses[maj]),
                    "entropy": float(-(masses * np.log(masses + 1e-12)).sum()),
                    "class_masses": masses.tolist(),
                }
                records.append(rec)
                print(
                    json.dumps(
                        {
                            k: rec[k]
                            for k in (
                                "seed",
                                "policy",
                                "generation",
                                "rare_mean_mass",
                                "majority_mass",
                                "entropy",
                            )
                        }
                    ),
                    flush=True,
                )
                if g == args.generations:
                    break
                scores = nll_proxy(model, samp)
                n_target = len(real_buf)
                n_real = int(round(args.alpha_real * n_target))
                n_syn = n_target - n_real
                if policy == "mix":
                    syn = samp[rng.choice(len(samp), size=n_syn, replace=False)]
                else:
                    syn = select(samp, scores, k=max(n_syn, len(samp) // 2), policy=policy, rng=rng)
                    syn = syn[rng.choice(len(syn), size=n_syn, replace=False)]
                real = real_buf[rng.choice(len(real_buf), size=n_real, replace=False)]
                train = np.concatenate([real, syn], axis=0)
                rng.shuffle(train)

    path = out / "logs" / "w2_fs_digits.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("wrote", path, "n=", len(records))


if __name__ == "__main__":
    main()

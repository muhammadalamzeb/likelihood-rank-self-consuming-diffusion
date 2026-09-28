#!/usr/bin/env python3
"""
CIFAR tiny-UNet self-consuming run with Inception FID at g=0 and g=1.
Experiment ID: w2_fs_cifar_fid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_w2_fs_cifar import (  # noqa: E402
    build_next_train,
    elbo_proxy,
    load_cifar_pool,
    make_imbalanced,
    mode_masses,
    sample,
    train_ddpm,
)


class InceptionFeat(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT
        net = models.inception_v3(weights=weights, transform_input=False)
        net.fc = nn.Identity()
        net.eval()
        self.net = net
        self.tfm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        # x01: [B,3,H,W] in [0,1] -> resize 299
        x = torch.nn.functional.interpolate(x01, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.tfm(x)
        out = self.net(x)
        if isinstance(out, tuple):
            out = out[0]
        return out


def frechet(mu1, sigma1, mu2, sigma2) -> float:
    diff = mu1 - mu2
    # matrix square root via eigen
    covmean_sq, _ = np.linalg.eig(sigma1 @ sigma2)
    # more stable: scipy would be better; use numpy svd sqrt
    from scipy.linalg import sqrtm

    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr = np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(diff @ diff + tr)


@torch.no_grad()
def inception_stats(feat_net: InceptionFeat, X: np.ndarray, device: torch.device, bs: int = 32):
    feats = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i : i + bs]).to(device)
        feats.append(feat_net(xb).cpu().numpy())
    F = np.concatenate(feats, axis=0).astype(np.float64)
    mu = F.mean(axis=0)
    sigma = np.cov(F, rowvar=False)
    return mu, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--data-root", type=Path, default=Path("experiments/data/cifar10"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rhos", type=float, nargs="+", default=[5.0])
    ap.add_argument("--policies", nargs="+", default=["top_k", "rand_k", "bottom_k"])
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--n-data", type=int, default=512)
    ap.add_argument("--n-synth", type=int, default=512)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--alpha-real", type=float, default=0.5)
    ap.add_argument("--k-frac", type=float, default=0.5)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-t", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--log-name", type=str, default="w2_fs_cifar_fid.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    (args.out / "configs").mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["experiment"] = "w2_fs_cifar_fid"
    with open(args.out / "configs" / "w2_fs_cifar_fid.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("loading CIFAR + Inception...", flush=True)
    X_pool, y_pool = load_cifar_pool(args.data_root, list(args.classes))
    feat_net = InceptionFeat().to(device)
    feat_net.eval()

    path = args.out / "logs" / args.log_name
    n_wrote = 0
    with open(path, "w", encoding="utf-8") as f:
        for seed in args.seeds:
            for rho in args.rhos:
                X0, _, means, true_w = make_imbalanced(X_pool, y_pool, args.n_data, rho, seed)
                # held-out real for FID: another draw from the same class pool (balanced-ish holdout)
                rng = np.random.default_rng(seed + 999)
                hold_idx = rng.choice(len(X_pool), size=min(args.n_data, len(X_pool)), replace=False)
                X_hold = X_pool[hold_idx]
                mu_r, sig_r = inception_stats(feat_net, X_hold, device)
                for policy in args.policies:
                    real_buf = X0.copy()
                    train = X0.copy()
                    rng_p = np.random.default_rng(seed)
                    for g in range(args.generations + 1):
                        model = train_ddpm(train, seed + 17 * g, args.train_steps, args.base, args.lr, device)
                        samp = sample(model, args.n_synth, args.n_steps, seed + 101 * g, device)
                        scores = elbo_proxy(model, samp, args.n_t, args.chunk, device)
                        masses = mode_masses(samp, means)
                        mu_s, sig_s = inception_stats(feat_net, samp, device)
                        fid = frechet(mu_r, sig_r, mu_s, sig_s)
                        rec = {
                            "experiment": "w2_fs_cifar_fid",
                            "seed": seed,
                            "rho": rho,
                            "policy": policy,
                            "generation": g,
                            "minority_mean": float(masses[1:].mean()),
                            "majority_mass": float(masses[0]),
                            "mode_masses": masses.tolist(),
                            "true_weights": true_w.tolist(),
                            "fid": fid,
                            "classes": args.classes,
                        }
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        n_wrote += 1
                        print(
                            json.dumps(
                                {
                                    k: rec[k]
                                    for k in (
                                        "seed",
                                        "rho",
                                        "policy",
                                        "generation",
                                        "minority_mean",
                                        "majority_mass",
                                        "fid",
                                    )
                                }
                            ),
                            flush=True,
                        )
                        if g == args.generations:
                            break
                        train = build_next_train(
                            real_buf, samp, scores, policy, args.alpha_real, args.k_frac, rng_p
                        )
    print(f"wrote {path} n={n_wrote}", flush=True)


if __name__ == "__main__":
    main()

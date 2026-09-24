#!/usr/bin/env python3
"""
Strategy B stack v4 — synthetic SBM graphs + tiny GraphVAE (adjacency upper-tri).
Observation only: community strength / density / label-noise interventions.
No novelty claim until post-hoc kill-test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


def triu_dim(n: int) -> int:
    return n * (n - 1) // 2


def adj_to_vec(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    idx = np.triu_indices(n, k=1)
    return A[idx].astype(np.float32)


def vec_to_adj(v: np.ndarray, n: int, thresh: float = 0.5) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float32)
    idx = np.triu_indices(n, k=1)
    edges = (v >= thresh).astype(np.float32)
    A[idx] = edges
    A = A + A.T
    return A


def sbm_graph(n: int, p_in: float, p_out: float, seed: int) -> nx.Graph:
    sizes = [n // 2, n - n // 2]
    probs = [[p_in, p_out], [p_out, p_in]]
    return nx.stochastic_block_model(sizes, probs, seed=seed)


def graph_stats(A: np.ndarray) -> dict:
    G = nx.from_numpy_array(A)
    n = A.shape[0]
    dens = float(A.sum() / (n * (n - 1)))
    try:
        ass = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        ass = float("nan")
    # binary modularity proxy: split nodes 0..n/2-1 vs rest (planted for SBM)
    c0 = list(range(n // 2))
    c1 = list(range(n // 2, n))
    try:
        q = float(nx.community.modularity(G, [c0, c1]))
    except Exception:
        q = float("nan")
    n_comp = nx.number_connected_components(G)
    return {
        "density": dens,
        "assortativity": ass,
        "planted_modularity": q,
        "n_components": int(n_comp),
        "n_edges": int(A.sum() // 2),
    }


class GraphVAE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int = 16, h: int = 128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, z_dim)
        self.logvar = nn.Linear(h, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, x_dim)
        )

    def forward(self, x: torch.Tensor, beta: float = 1.0):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        logits = self.dec(z)
        recon = nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return logits, recon + beta * kl, recon.detach(), kl.detach()

    @torch.no_grad()
    def sample(self, n: int, z_dim: int = 16) -> torch.Tensor:
        z = torch.randn(n, z_dim)
        return torch.sigmoid(self.dec(z))


def make_dataset(n_nodes: int, p_in: float, p_out: float, n_graphs: int, seed: int) -> np.ndarray:
    X = []
    for i in range(n_graphs):
        G = sbm_graph(n_nodes, p_in, p_out, seed=seed + i * 17)
        A = nx.to_numpy_array(G, dtype=np.float32)
        X.append(adj_to_vec(A))
    return np.stack(X, axis=0)


def train_vae(X: np.ndarray, seed: int, epochs: int, beta: float = 1.0) -> GraphVAE:
    torch.manual_seed(seed)
    model = GraphVAE(x_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(X)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            batch = xt[perm[i : i + 64]]
            _, loss, _, _ = model(batch, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def eval_samples(model: GraphVAE, n_nodes: int, n_samp: int, seed: int) -> dict:
    torch.manual_seed(seed + 999)
    probs = model.sample(n_samp).numpy()
    stats = []
    for i in range(n_samp):
        A = vec_to_adj(probs[i], n_nodes, thresh=0.5)
        stats.append(graph_stats(A))
    keys = ["density", "assortativity", "planted_modularity", "n_components", "n_edges"]
    out = {}
    for k in keys:
        vals = np.array([s[k] for s in stats], dtype=np.float64)
        out[f"gen_mean_{k}"] = float(np.nanmean(vals))
        out[f"gen_std_{k}"] = float(np.nanstd(vals))
    # pairwise Hamming diversity on upper-tri binary
    bins = (probs >= 0.5).astype(np.float32)
    if n_samp >= 2:
        dsum = 0.0
        c = 0
        for i in range(n_samp):
            for j in range(i + 1, n_samp):
                dsum += float(np.mean(bins[i] != bins[j]))
                c += 1
        out["pairwise_hamming"] = dsum / max(c, 1)
    else:
        out["pairwise_hamming"] = float("nan")
    return out


def train_stats(X: np.ndarray, n_nodes: int) -> dict:
    stats = [graph_stats(vec_to_adj(X[i], n_nodes, thresh=0.5)) for i in range(len(X))]
    out = {}
    for k in ["density", "assortativity", "planted_modularity", "n_components", "n_edges"]:
        vals = np.array([s[k] for s in stats], dtype=np.float64)
        out[f"train_mean_{k}"] = float(np.nanmean(vals))
        out[f"train_std_{k}"] = float(np.nanstd(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-nodes", type=int, default=20)
    ap.add_argument("--n-graphs", type=int, default=400)
    ap.add_argument("--n-samp", type=int, default=64)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 40
        args.n_graphs = 200
        args.n_samp = 32

    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    cfg = {
        "stack": "v4_sbm_graphvae",
        "seed": args.seed,
        "epochs": args.epochs,
        "n_nodes": args.n_nodes,
        "n_graphs": args.n_graphs,
        "n_samp": args.n_samp,
        "probes": ["G1_community_strength", "G3_density", "G4_label_noise_proxy"],
    }
    with open(out / "configs" / f"graph_v4_seed{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    records = []
    n = args.n_nodes

    # G1: community strength sweep (p_out fixed, p_in varies)
    p_out = 0.05
    for p_in in [0.15, 0.35, 0.55, 0.75]:
        X = make_dataset(n, p_in, p_out, args.n_graphs, seed=args.seed * 1000 + int(p_in * 100))
        tr = train_stats(X, n)
        model = train_vae(X, seed=args.seed, epochs=args.epochs)
        gen = eval_samples(model, n, args.n_samp, seed=args.seed)
        rec = {
            "probe": "G1",
            "p_in": p_in,
            "p_out": p_out,
            **tr,
            **gen,
            "delta_modularity": gen["gen_mean_planted_modularity"] - tr["train_mean_planted_modularity"],
            "delta_assortativity": gen["gen_mean_assortativity"] - tr["train_mean_assortativity"],
            "delta_density": gen["gen_mean_density"] - tr["train_mean_density"],
        }
        records.append(rec)
        print(json.dumps(rec))

    # G3: density sweep at fixed weak communities (p_in≈p_out+eps → ER-like)
    for p in [0.1, 0.2, 0.35, 0.5]:
        X = make_dataset(n, p, p, args.n_graphs, seed=args.seed * 2000 + int(p * 100))
        tr = train_stats(X, n)
        model = train_vae(X, seed=args.seed + 1, epochs=args.epochs)
        gen = eval_samples(model, n, args.n_samp, seed=args.seed + 1)
        rec = {
            "probe": "G3",
            "p_er": p,
            **tr,
            **gen,
            "delta_density": gen["gen_mean_density"] - tr["train_mean_density"],
            "delta_assortativity": gen["gen_mean_assortativity"] - tr["train_mean_assortativity"],
        }
        records.append(rec)
        print(json.dumps(rec))

    # G4: proxy — train on strong SBM, evaluate modularity under increasing Bernoulli edge flips
    # (inference-time corruption of latents via mixing train with ER) as label-noise analogue
    X_strong = make_dataset(n, 0.7, 0.05, args.n_graphs, seed=args.seed * 3000)
    model = train_vae(X_strong, seed=args.seed + 2, epochs=args.epochs)
    for noise in [0.0, 0.1, 0.25, 0.5]:
        # mix decoded samples toward ER density ~ train density (post-hoc edge flip)
        torch.manual_seed(args.seed + 50 + int(noise * 100))
        probs = model.sample(args.n_samp).numpy()
        dens_t = float(np.mean(X_strong))
        stats = []
        for i in range(args.n_samp):
            A = vec_to_adj(probs[i], n, thresh=0.5)
            if noise > 0:
                mask = np.triu(np.random.rand(n, n) < noise, k=1)
                flip = mask | mask.T
                A = np.where(flip, 1.0 - A, A).astype(np.float32)
                np.fill_diagonal(A, 0.0)
            stats.append(graph_stats(A))
        gen = {
            "gen_mean_planted_modularity": float(np.nanmean([s["planted_modularity"] for s in stats])),
            "gen_mean_density": float(np.nanmean([s["density"] for s in stats])),
            "gen_mean_assortativity": float(np.nanmean([s["assortativity"] for s in stats])),
            "train_density_ref": dens_t,
        }
        rec = {"probe": "G4", "edge_flip_noise": noise, **gen}
        records.append(rec)
        print(json.dumps(rec))

    log_path = out / "logs" / f"graph_v4_seed{args.seed}.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Strategy B — probe P6: wrong-class conditioning at inference.
Observation only. Fixed noise × varied class labels.

Usage:
  .venv\\Scripts\\python experiments/scripts/run_probe_p6.py --quick --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torchvision.utils import make_grid, save_image


REPO_ID = "Ketansomewhere/cifar10_conditional_diffusion1"
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def sample_from_noise(
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    x0: torch.Tensor,
    class_labels: torch.Tensor,
    num_inference_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Denoise a fixed initial noise tensor under given class labels."""
    device = next(unet.parameters()).device
    x = x0.clone()
    scheduler.set_timesteps(num_inference_steps, device=device)
    for t in scheduler.timesteps:
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        eps = unet(x, t_batch, class_labels=class_labels).sample
        x = scheduler.step(eps, t, x, generator=generator).prev_sample
    return (x.clamp(-1, 1) + 1) / 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("experiments"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-noise", type=int, default=4, help="fixed noise columns")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--quick",
        action="store_true",
        help="CPU-safe: 2 noise cols, steps=20, all 10 labels",
    )
    args = p.parse_args()

    if args.quick:
        args.n_noise = 2
        if args.steps == 50:
            args.steps = 20

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = args.out
    (out / "grids").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    print(f"device={device} loading {REPO_ID} ...")
    unet = UNet2DModel.from_pretrained(REPO_ID, subfolder="unet").to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(REPO_ID, subfolder="scheduler")
    sample_size = unet.config.sample_size
    ch = unet.config.in_channels

    g = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        args.n_noise, ch, sample_size, sample_size, device=device, generator=g
    )

    # Grid layout: row = conditioning class, col = fixed noise index
    rows = []
    for c in range(10):
        labels = torch.full((args.n_noise,), c, device=device, dtype=torch.long)
        print(f"sampling class={c} ({CIFAR_CLASSES[c]}) steps={args.steps} ...")
        imgs = sample_from_noise(unet, scheduler, noise, labels, args.steps)
        rows.append(imgs)
    all_imgs = torch.cat(rows, dim=0)  # 10 * n_noise
    grid = make_grid(all_imgs, nrow=args.n_noise)
    name = f"p6_wrongclass_steps{args.steps}_n{args.n_noise}_seed{args.seed}.png"
    path = out / "grids" / name
    save_image(grid, path)

    # Pairwise swap panel: similar-class confusions (same noise, swap labels)
    pairs = [(3, 5), (1, 9), (0, 8), (2, 4)]  # cat-dog, auto-truck, plane-ship, bird-deer
    pair_imgs = []
    pair_meta = []
    for a, b in pairs:
        for label in (a, b):
            labels = torch.full((args.n_noise,), label, device=device, dtype=torch.long)
            imgs = sample_from_noise(unet, scheduler, noise, labels, args.steps)
            pair_imgs.append(imgs)
            pair_meta.append({"cond": label, "name": CIFAR_CLASSES[label], "pair": [a, b]})
    pair_all = torch.cat(pair_imgs, dim=0)
    pair_grid = make_grid(pair_all, nrow=args.n_noise)
    pair_name = f"p6_pairs_steps{args.steps}_n{args.n_noise}_seed{args.seed}.png"
    pair_path = out / "grids" / pair_name
    save_image(pair_grid, pair_path)

    rec = {
        "probe": "P6",
        "mode": "quick" if args.quick else "full",
        "steps": args.steps,
        "seed": args.seed,
        "n_noise": args.n_noise,
        "layout": "rows=class0..9, cols=fixed_noise",
        "pairs": pairs,
        "pair_row_order": pair_meta,
        "repo": REPO_ID,
        "grid": str(path),
        "pair_grid": str(pair_path),
        "num_class_embeds": getattr(unet.config, "num_class_embeds", None),
        "device": str(device),
        "note": "Same initial noise across rows; only class_labels change.",
    }
    log_path = out / "logs" / f"p6_seed{args.seed}.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    cfg_path = out / "configs" / f"p6_seed{args.seed}.json"
    cfg_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("wrote", path)
    print("wrote", pair_path)
    print("log", log_path)


if __name__ == "__main__":
    main()

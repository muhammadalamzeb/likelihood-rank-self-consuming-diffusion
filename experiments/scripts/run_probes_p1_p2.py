#!/usr/bin/env python3
"""
Strategy B — probe battery P1 (CFG) + P2 (DDIM steps).
Observation only. Run on Colab (GPU). Does not claim novelty.

Usage (Colab):
  !pip -q install diffusers transformers accelerate torchvision pillow
  !python experiments/scripts/run_probes_p1_p2.py --out experiments
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from torchvision.utils import make_grid, save_image


REPO_ID = "Ketansomewhere/cifar10_conditional_diffusion1"
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def sample_batch(
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    class_labels: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """CFG for class-conditional UNet: eps = eps_uncond + s * (eps_cond - eps_uncond).
    Uncond uses a null label = num_classes (requires embedding size num_classes+1).
    If the checkpoint has only num_classes embeds, we approximate uncond by
    averaging predictions over all class labels (documented fallback).
    """
    device = next(unet.parameters()).device
    b = class_labels.shape[0]
    sample_size = unet.config.sample_size
    x = torch.randn(
        b, unet.config.in_channels, sample_size, sample_size,
        device=device, generator=generator,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    num_classes = 10
    has_null = (
        getattr(unet.config, "num_class_embeds", None) is not None
        and unet.config.num_class_embeds > num_classes
    )

    for t in scheduler.timesteps:
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        if guidance_scale <= 1.0:
            eps = unet(x, t_batch, class_labels=class_labels).sample
        elif has_null:
            null = torch.full_like(class_labels, num_classes)
            eps_u = unet(x, t_batch, class_labels=null).sample
            eps_c = unet(x, t_batch, class_labels=class_labels).sample
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        else:
            # Fallback: no dedicated null class — document in log JSON.
            eps_c = unet(x, t_batch, class_labels=class_labels).sample
            eps_u = 0
            for c in range(num_classes):
                labels = torch.full_like(class_labels, c)
                eps_u = eps_u + unet(x, t_batch, class_labels=labels).sample
            eps_u = eps_u / num_classes
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        x = scheduler.step(eps, t, x, generator=generator).prev_sample

    x = (x.clamp(-1, 1) + 1) / 2
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("experiments"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-class", type=int, default=4, help="images per class in each grid")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--quick",
        action="store_true",
        help="CPU-safe subset: CFG=1 only, steps {10,20,50}, 2 imgs/class",
    )
    args = p.parse_args()

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

    if args.quick:
        cfg_values = [1.0]
        step_values = [10, 20, 50]
        if args.per_class == 4:
            args.per_class = 2
    else:
        # P1: CFG sweep at fixed steps; P2: step sweep at CFG=1 (and CFG=3 if null exists)
        cfg_values = [1.0, 1.5, 3.0, 7.5]
        step_values = [10, 20, 50, 100]
    records = []

    labels = torch.arange(10, device=device).repeat_interleave(args.per_class)

    for steps in step_values:
        for cfg in cfg_values:
            if not args.quick and steps != 50 and cfg not in (1.0, 3.0):
                continue  # keep Colab budget small: full CFG sweep only at 50 steps
            g = torch.Generator(device=device).manual_seed(args.seed)
            print(f"sampling steps={steps} cfg={cfg} ...")
            imgs = sample_batch(unet, scheduler, labels, steps, cfg, g)
            grid = make_grid(imgs, nrow=args.per_class)
            name = f"p1p2_steps{steps}_cfg{cfg}_seed{args.seed}.png"
            path = out / "grids" / name
            save_image(grid, path)
            rec = {
                "probe": "P1+P2",
                "mode": "quick" if args.quick else "full",
                "steps": steps,
                "guidance_scale": cfg,
                "seed": args.seed,
                "per_class": args.per_class,
                "repo": REPO_ID,
                "grid": str(path),
                "num_class_embeds": getattr(unet.config, "num_class_embeds", None),
                "device": str(device),
            }
            records.append(rec)
            print("wrote", path)

    log_path = out / "logs" / f"p1p2_seed{args.seed}.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    cfg_path = out / "configs" / f"p1p2_seed{args.seed}.json"
    cfg_path.write_text(json.dumps({"repo": REPO_ID, "records": records}, indent=2), encoding="utf-8")
    print("log", log_path)
    print("Next: inspect grids; note class leakage / step cliffs in experiments/NOTES.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Strategy B stack v1 — unconditional flow-matching CIFAR probe (observation only).
Repo: FrankCCCCC/cfm-cifar10-32

Usage:
  .venv\\Scripts\\python experiments/scripts/run_probe_fm_v1.py --quick --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from torchvision.utils import make_grid, save_image


REPO_ID = "FrankCCCCC/cfm-cifar10-32"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("experiments"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--quick",
        action="store_true",
        help="CPU-safe: n=8, step sets {5,10,20}",
    )
    args = p.parse_args()

    step_values = [5, 10, 20] if args.quick else [5, 10, 20, 50]
    if args.quick:
        args.n = 8

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = args.out
    (out / "grids").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    print(f"device={device} loading {REPO_ID} ...")
    pipe = DiffusionPipeline.from_pretrained(REPO_ID)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    records = []
    for steps in step_values:
        g = torch.Generator(device=device).manual_seed(args.seed)
        print(f"sampling n={args.n} steps={steps} ...")
        # Diffusers DDPMPipeline / generic: num_inference_steps
        result = pipe(
            batch_size=args.n,
            num_inference_steps=steps,
            generator=g,
            output_type="pt",
        )
        imgs = result.images
        if isinstance(imgs, list):
            import torchvision.transforms.functional as TF

            imgs = torch.stack([TF.to_tensor(im) for im in imgs])
        elif not torch.is_tensor(imgs):
            imgs = torch.as_tensor(imgs)
            if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):
                imgs = imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW
            if imgs.dtype != torch.float32:
                imgs = imgs.float()
            if imgs.max() > 1.5:
                imgs = imgs / 255.0
        grid = make_grid(imgs, nrow=4)
        name = f"fm_v1_steps{steps}_seed{args.seed}.png"
        path = out / "grids" / name
        save_image(grid, path)
        rec = {
            "probe": "FM-v1-steps",
            "stack": "v1",
            "repo": REPO_ID,
            "steps": steps,
            "seed": args.seed,
            "n": args.n,
            "grid": str(path),
            "device": str(device),
            "scheduler": type(pipe.scheduler).__name__,
        }
        records.append(rec)
        print("wrote", path)

    log_path = out / "logs" / f"fm_v1_seed{args.seed}.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    cfg_path = out / "configs" / f"fm_v1_seed{args.seed}.json"
    cfg_path.write_text(json.dumps({"repo": REPO_ID, "records": records}, indent=2), encoding="utf-8")
    print("log", log_path)


if __name__ == "__main__":
    main()

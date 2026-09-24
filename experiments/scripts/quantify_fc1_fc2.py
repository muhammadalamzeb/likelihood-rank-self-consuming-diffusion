#!/usr/bin/env python3
"""
Quantify Strategy B clusters FC1/FC2 on CIFAR-10 conditional samples.

- Loads a frozen torchvision ResNet18 (CIFAR-10 fine-tuned if available via
  simple train on CIFAR test-as-proxy is forbidden — we train briefly on
  CIFAR-10 *train* once and freeze, or use a public ckpt).
- Regenerates samples with the same probe settings as quick P2.
- Metrics:
  FC1: mean classifier P(correct class) vs NFE
  FC2: within-class twin disagreement rate and |p1-p2| gap at each NFE

Observation only. No novelty claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import save_image

REPO_ID = "Ketansomewhere/cifar10_conditional_diffusion1"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_classifier(device: torch.device) -> nn.Module:
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m.to(device)


def train_classifier(device: torch.device, epochs: int, data_root: Path) -> nn.Module:
    tfm = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    train = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tfm)
    loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
    model = build_classifier(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        total, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        print(f"clf epoch {ep+1}/{epochs} loss={total/n:.4f} acc={correct/n:.3f}")
    model.eval()
    return model


@torch.no_grad()
def sample_batch(unet, scheduler, class_labels, steps, generator):
    device = next(unet.parameters()).device
    b = class_labels.shape[0]
    ss = unet.config.sample_size
    x = torch.randn(
        b, unet.config.in_channels, ss, ss, device=device, generator=generator
    )
    scheduler.set_timesteps(steps, device=device)
    for t in scheduler.timesteps:
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        eps = unet(x, t_batch, class_labels=class_labels).sample
        x = scheduler.step(eps, t, x, generator=generator).prev_sample
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def classify(model, images_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """images in [0,1] → preds, probs for intended analysis."""
    x = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)(images_01)
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    return logits.argmax(1), probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments"))
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--twins", type=int, default=2, help="samples per class")
    ap.add_argument("--clf-epochs", type=int, default=3)
    ap.add_argument("--clf-ckpt", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "grids").mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.clf_ckpt or (out / "checkpoints" / "cifar_resnet18.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists():
        print("loading classifier", ckpt_path)
        clf = build_classifier(device)
        clf.load_state_dict(torch.load(ckpt_path, map_location=device))
        clf.eval()
    else:
        print("training frozen CIFAR classifier (few epochs) ...")
        clf = train_classifier(device, args.clf_epochs, args.data_root)
        torch.save(clf.state_dict(), ckpt_path)
        print("saved", ckpt_path)

    print("loading diffusion", REPO_ID)
    unet = UNet2DModel.from_pretrained(REPO_ID, subfolder="unet").to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(REPO_ID, subfolder="scheduler")

    labels = torch.arange(10, device=device).repeat_interleave(args.twins)
    rows = []

    for seed in args.seeds:
        for steps in args.steps:
            g = torch.Generator(device=device).manual_seed(seed)
            print(f"sample seed={seed} steps={steps}")
            imgs = sample_batch(unet, scheduler, labels, steps, g)
            preds, probs = classify(clf, imgs)
            intended = labels
            correct = (preds == intended).float()
            # twin metrics: reshape [10, twins]
            correct_tw = correct.view(10, args.twins)
            probs_int = probs.gather(1, intended.view(-1, 1)).squeeze(1).view(10, args.twins)
            twin_disagree = (preds.view(10, args.twins)[:, 0] != preds.view(10, args.twins)[:, 1]).float()
            both_wrong = ((1 - correct_tw).prod(dim=1))
            one_good_one_bad = (correct_tw.sum(dim=1) == 1).float()
            conf_gap = (probs_int[:, 0] - probs_int[:, 1]).abs()

            rec = {
                "seed": seed,
                "steps": steps,
                "acc_mean": float(correct.mean()),
                "p_intended_mean": float(probs_int.mean()),
                "twin_pred_disagree_rate": float(twin_disagree.mean()),
                "twin_one_good_one_bad_rate": float(one_good_one_bad.mean()),
                "twin_both_wrong_rate": float(both_wrong.mean()),
                "twin_conf_gap_mean": float(conf_gap.mean()),
                "per_class_acc": correct_tw.mean(dim=1).tolist(),
                "per_class_one_good_one_bad": one_good_one_bad.tolist(),
            }
            rows.append(rec)
            print(
                f"  acc={rec['acc_mean']:.3f} "
                f"twin_disagree={rec['twin_pred_disagree_rate']:.3f} "
                f"one_good_one_bad={rec['twin_one_good_one_bad_rate']:.3f} "
                f"conf_gap={rec['twin_conf_gap_mean']:.3f}"
            )
            save_image(imgs, out / "grids" / f"fc_quant_steps{steps}_seed{seed}.png", nrow=args.twins)

    # aggregate FC1/FC2 summaries
    by_steps: dict[int, list] = {}
    for r in rows:
        by_steps.setdefault(r["steps"], []).append(r)
    summary = {"fc1_acc_vs_nfe": {}, "fc2_one_good_one_bad_vs_nfe": {}, "rows": rows}
    for s, lst in sorted(by_steps.items()):
        summary["fc1_acc_vs_nfe"][str(s)] = {
            "mean": sum(x["acc_mean"] for x in lst) / len(lst),
            "seeds": [x["seed"] for x in lst],
        }
        summary["fc2_one_good_one_bad_vs_nfe"][str(s)] = {
            "mean": sum(x["twin_one_good_one_bad_rate"] for x in lst) / len(lst),
            "disagree_mean": sum(x["twin_pred_disagree_rate"] for x in lst) / len(lst),
            "conf_gap_mean": sum(x["twin_conf_gap_mean"] for x in lst) / len(lst),
        }

    path = out / "logs" / "fc1_fc2_quant.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", path)
    print("FC1 acc vs NFE:", summary["fc1_acc_vs_nfe"])
    print("FC2 one-good-one-bad vs NFE:", summary["fc2_one_good_one_bad_vs_nfe"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cross-platform reproduction entry for principal W2-1 results."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=ROOT)


def main() -> None:
    py = sys.executable
    run([py, "-m", "pip", "install", "matplotlib", "--disable-pip-version-check", "-q"])
    run(
        [
            py,
            "experiments/scripts/run_w2_false_stability.py",
            "--out",
            "experiments",
            "--seeds",
            "0",
            "1",
            "2",
            "--rhos",
            "5.0",
            "10.0",
            "--policies",
            "mix",
            "top_k",
            "rand_k",
            "replace",
            "bottom_k",
            "--generations",
            "5",
            "--train-steps",
            "600",
            "--n-data",
            "1500",
            "--n-synth",
            "1500",
        ]
    )
    run([py, "experiments/scripts/analyze_w2_fs.py"])
    run([py, "experiments/scripts/make_w2_figures.py"])
    paper_fig = ROOT / "paper" / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "experiments" / "analysis" / "figures").glob("*.png"):
        shutil.copy2(p, paper_fig / p.name)
    for p in (ROOT / "experiments" / "analysis").glob("table_*.csv"):
        shutil.copy2(p, ROOT / "paper" / p.name)
    print("Done. See paper/PAPER.md and experiments/analysis/", flush=True)


if __name__ == "__main__":
    main()

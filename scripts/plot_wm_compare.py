"""Overlay several world-model training logs on one figure (one panel per metric).

Complements plot_wm_log.py, which plots a single run's train-vs-test curves. Here the
splits are collapsed (test only, solid; train dashed) so that runs can be compared.

Usage:
    python scripts/plot_wm_compare.py \
        installation/logs/wm_resnet_25083707_*.log \
        installation/logs/wm_dino_25083733_*.log \
        -o installation/logs/wm_sweep_25083707_25083733.png
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRIC_RE = re.compile(r"Steps:\s*(\d+),\s*Metrics:\s*(\{.*\})\s*$")

PANELS = ["loss", "latent_dynamics", "value", "action", "reward", "debug/latent_norm"]

# array index -> latent_dim, matching train_wm_manip_{dino,resnet_sweep}.sbatch
LATENT_DIMS = {0: 64, 1: 128, 2: 256}

# categorical assignment: backbone -> hue, latent_dim -> lightness
COLORS = {
    ("resnet", 64): "#9ec5f5", ("resnet", 128): "#5b8fe0", ("resnet", 256): "#1f4e9c",
    ("dino", 64): "#f6c689", ("dino", 128): "#e0902f", ("dino", 256): "#a35c05",
}


def parse_log(path: Path) -> list[dict]:
    records = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            # tqdm writes carriage returns; the metrics line can be glued to a bar
            for chunk in line.split("\r"):
                m = METRIC_RE.search(chunk.strip())
                if m is None:
                    continue
                try:
                    metrics = ast.literal_eval(m.group(2))
                except (ValueError, SyntaxError):
                    continue
                records.append({"steps": int(m.group(1)), **metrics})
    records.sort(key=lambda r: r["steps"])
    return records


def label_of(path: Path) -> tuple[str, int]:
    """('resnet'|'dino', latent_dim) from the wm_<enc>_<jobid>_<arrayidx>.log name."""
    stem = path.stem.split("_")
    enc = stem[1]
    idx = int(stem[-1]) if stem[-1].isdigit() else 0
    return enc, LATENT_DIMS.get(idx, idx)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--x", default="steps", choices=["steps", "epoch"])
    p.add_argument("--linear", action="store_true", help="linear y instead of log")
    args = p.parse_args()

    runs = [(label_of(f), parse_log(f)) for f in args.logs]
    runs = [(k, r) for k, r in runs if r]
    if not runs:
        raise SystemExit("no metric lines parsed")

    ncol = 3
    nrow = -(-len(PANELS) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow), squeeze=False)

    for ax, name in zip(axes.ravel(), PANELS):
        for (enc, lat), recs in runs:
            color = COLORS.get((enc, lat), "#888888")
            xs = [r[args.x] for r in recs]
            for split, ls in (("test", "-"), ("train", "--")):
                key = f"{split}/{name}"
                if key not in recs[0]:
                    continue
                ax.plot(xs, [r[key] for r in recs], ls, color=color, lw=1.6,
                        alpha=1.0 if split == "test" else 0.45,
                        label=f"{enc} l{lat}" if split == "test" else None)
        ax.set_title(name)
        ax.set_xlabel(args.x)
        if not args.linear and "latent_norm" not in name:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)

    for ax in axes.ravel()[len(PANELS):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=8, ncol=2)
    fig.suptitle("world-model sweep — solid = test, dashed = train", y=1.0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

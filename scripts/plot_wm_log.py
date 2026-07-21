"""Parse and plot world-model training metrics from an sbatch log.

The trainer emits one line per epoch of the form:

    Steps: 73840, Metrics: {'steps_per_second': ..., 'train/loss': ..., 'epoch': 19}

Usage:
    python scripts/plot_wm_log.py installation/logs/wm_manip_24729830.log
    python scripts/plot_wm_log.py <log> -o plots/wm.png --x epoch --linear
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

# fixed categorical assignment (validated: CVD dE 28.3, normal dE 33.5 on light surface)
SPLIT_COLOR = {"train": "#3b6fd6", "test": "#d97706"}


def parse_log(path: str | Path) -> list[dict]:
    """Return one dict per logged epoch: {'steps': int, **metrics}."""
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


def _metric_names(records: list[dict]) -> list[str]:
    """Metric suffixes that appear under train/ or test/, loss first."""
    names = {k.split("/", 1)[1] for r in records for k in r if k.startswith(("train/", "test/"))}
    ordered = ["loss"] if "loss" in names else []
    return ordered + sorted(names - set(ordered))


def plot_metrics(
    records: list[dict],
    out_path: str | Path,
    x_key: str = "steps",
    log_y: bool = True,
    title: str | None = None,
):
    """Small multiples: one panel per metric, train vs test overlaid."""
    if not records:
        raise ValueError("no metric lines parsed from log")

    names = _metric_names(records)
    x = [r[x_key] for r in records]

    ncols = min(3, len(names))
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows), squeeze=False)

    for ax, name in zip(axes.flat, names):
        for split in ("train", "test"):
            key = f"{split}/{name}"
            ys = [r.get(key) for r in records]
            if all(y is None for y in ys):
                continue
            pts = [(xi, yi) for xi, yi in zip(x, ys) if yi is not None]
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=SPLIT_COLOR[split],
                lw=2,
                label=split,
            )
        ax.set_title(name, fontsize=10, color="#2a2a28")
        if log_y and all(
            (r.get(f"{s}/{name}") or 1) > 0 for r in records for s in ("train", "test")
        ):
            ax.set_yscale("log")
        ax.set_xlabel(x_key, fontsize=8, color="#6b6b66")
        ax.grid(True, lw=0.5, color="#e6e6e2")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c9c4")
        ax.tick_params(labelsize=8, colors="#6b6b66")
        ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, color="#2a2a28")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log")
    p.add_argument("-o", "--out", default=None, help="output png (default: <log>.png next to log)")
    p.add_argument("--x", default="steps", choices=["steps", "epoch"])
    p.add_argument("--linear", action="store_true", help="linear y-axis instead of log")
    args = p.parse_args()

    log = Path(args.log)
    records = parse_log(log)
    out = args.out or log.with_suffix(".png")
    path = plot_metrics(records, out, x_key=args.x, log_y=not args.linear, title=log.stem)
    print(f"{len(records)} metric points parsed -> {path}")


if __name__ == "__main__":
    main()

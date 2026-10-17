"""
Visual sanity check of the preprocessed HipMRI slices.

Saves to assets/:
    slices_<split>.png      a grid of random slices from each split, shown
                            upright (superior up) to check orientation by eye
    intensity_histogram.png per-split histogram of normalised pixel intensities

Usage: python scripts/inspect_data.py [root] [--n 16] [--seed 0]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from dataset import build_datasets, to_display  # noqa: E402


def plot_slice_grid(ds, n, rng, out_path):
    idx = np.sort(rng.choice(len(ds), size=n, replace=False))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.2 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idx):
        r = ds.records[i]
        ax.imshow(to_display(ds[i]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"case {r.case:03d}  week {r.week}  slice {r.slice}", fontsize=9)
    fig.suptitle(f"{ds.split}: {n} random slices (superior up)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_histogram(datasets, out_path, bins=100):
    fig, ax = plt.subplots(figsize=(7, 4))
    edges = np.linspace(0, 1, bins + 1)
    for split, ds in datasets.items():
        counts, _ = np.histogram(ds.images.numpy(), bins=edges)
        ax.stairs(counts / counts.sum(), edges, label=f"{split} ({len(ds)} slices)")
    ax.set_yscale("log")
    ax.set_xlabel("normalised intensity")
    ax.set_ylabel("fraction of pixels (log)")
    ax.set_title("Pixel intensities after per-slice 99.5th percentile clip")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?",
                        default=os.path.join(PROJECT_DIR, "data", "keras_slices_data"))
    parser.add_argument("--n", type=int, default=16, help="slices per grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "assets"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    datasets = build_datasets(args.root)

    for split, ds in datasets.items():
        plot_slice_grid(ds, args.n, rng, os.path.join(args.out, f"slices_{split}.png"))
    plot_histogram(datasets, os.path.join(args.out, "intensity_histogram.png"))
    print(f"Saved figures to {args.out}")


if __name__ == "__main__":
    main()

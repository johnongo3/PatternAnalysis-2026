"""
Shared helpers for training and evaluation: image quality metrics, code usage
statistics and figures.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
from torchmetrics.functional.image import (peak_signal_noise_ratio,
                                           structural_similarity_index_measure)

from dataset import to_display


def ssim(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image SSIM (11 x 11 Gaussian window, sigma 1.5) for images in [0, 1]."""
    return structural_similarity_index_measure(recon.clamp(0, 1), target,
                                               data_range=1.0, reduction="none")


def psnr(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for images in [0, 1]."""
    return peak_signal_noise_ratio(recon.clamp(0, 1), target, data_range=1.0,
                                   reduction="none", dim=(1, 2, 3))


def perplexity_from_counts(counts: torch.Tensor) -> float:
    """exp(entropy) of a code usage histogram; equals K when all K codes are used evenly."""
    probs = counts.float() / counts.sum()
    probs = probs[probs > 0]
    return math.exp(-(probs * probs.log()).sum().item())


def plot_history(history: list[dict], path: str, levels: tuple[str, ...]):
    """
    Plot per-epoch losses, code perplexity, and validation SSIM and PSNR for
    the full reconstruction (and the top level alone, for two-level models).
    """
    epochs = [h["epoch"] for h in history]

    def series(key):
        # nan for epochs logged before a metric existed (e.g. a resumed run)
        return [h.get(key, float("nan")) for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    ax = axes[0, 0]
    ax.plot(epochs, series("train_recon"), label="train")
    ax.plot(epochs, series("val_recon"), label="validate")
    ax.set_yscale("log")
    ax.set_title("Reconstruction loss (MSE)")
    ax.legend()

    ax = axes[0, 1]
    for level in levels:
        ax.plot(epochs, series(f"train_commit_{level}"), label=level)
    ax.set_yscale("log")
    ax.set_title("Commitment loss (train)")
    ax.legend()

    ax = axes[0, 2]
    for i, level in enumerate(levels):
        ax.plot(epochs, series(f"train_perplexity_{level}"), color=f"C{i}", label=f"{level} train")
        ax.plot(epochs, series(f"val_perplexity_{level}"), color=f"C{i}", linestyle="--",
                label=f"{level} validate")
    ax.set_title("Code perplexity")
    ax.legend()

    best = int(np.nanargmax(series("val_ssim")))
    for ax, metric, title in [(axes[1, 0], "ssim", "Validation SSIM"),
                              (axes[1, 1], "psnr", "Validation PSNR (dB)")]:
        ax.plot(epochs, series(f"val_{metric}"), label="full")
        if "top" in levels:
            ax.plot(epochs, series(f"val_{metric}_top_only"), label="top only")
        ax.axvline(epochs[best], color="grey", linestyle=":",
                   label=f"best val SSIM (epoch {epochs[best]})")
        ax.set_title(title)
        ax.legend()
    fig.delaxes(axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_reconstructions(columns: dict[str, torch.Tensor], titles: list[str], path: str):
    """
    Plot a grid with one row per slice and one column per entry of columns,
    e.g. {"original": x, "top only": ..., "full": ..., "|error|": ...}.
    Columns whose name starts with "|" are drawn as error maps with a colour bar.
    """
    names = list(columns)
    n = len(titles)
    fig, axes = plt.subplots(n, len(names), figsize=(3.2 * len(names), 1.8 * n),
                             squeeze=False, layout="constrained")
    for col, name in enumerate(names):
        images = columns[name].detach().cpu()
        is_error = name.startswith("|")
        vmax = float(images.quantile(0.995)) if is_error else 1.0
        for row in range(n):
            ax = axes[row, col]
            im = ax.imshow(to_display(images[row]), cmap="inferno" if is_error else "gray",
                           vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(titles[row], fontsize=8)
        if is_error:
            fig.colorbar(im, ax=axes[:, col].tolist(), shrink=0.6)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

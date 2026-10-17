"""
Shared helpers for training and evaluation: config, device and checkpoint
handling, image quality metrics, code usage statistics and figures.

Run layout: a stage-1 run directory holds its own checkpoints, the code maps
encoded with it, and the priors trained on those codes:

    checkpoints/<vqvae_run>/{best,last}.pt, history.csv, loss_curves.png
    checkpoints/<vqvae_run>/codes/{train,validate,test}.pt
    checkpoints/<vqvae_run>/prior_top/{best,last}.pt, ...
    checkpoints/<vqvae_run>/prior_bottom/{best,last}.pt, ...
    checkpoints/<vqvae_run>/samples/samples_t<temperature>.pt
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
import yaml
from torchmetrics.functional.image import (peak_signal_noise_ratio,
                                           structural_similarity_index_measure)

from dataset import to_display
from modules import PixelCNNPrior, VQVAE2, build_prior, build_vqvae


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_atomic(obj, path: str):
    """torch.save via a temp file, so an interrupted save never corrupts a checkpoint."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def codes_dir(vqvae_run: str) -> str:
    return os.path.join(vqvae_run, "codes")


def prior_dir(vqvae_run: str, level: str) -> str:
    return os.path.join(vqvae_run, f"prior_{level}")


def load_checkpoint(run_dir: str, name: str = "best.pt") -> dict:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint at {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_vqvae(run_dir: str, device: torch.device) -> tuple[VQVAE2, dict]:
    """The best-SSIM stage-1 model of a run, frozen in eval mode, and its checkpoint."""
    ckpt = load_checkpoint(run_dir)
    model = build_vqvae(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.requires_grad_(False)
    return model.to(device).eval(), ckpt


def load_prior(vqvae_run: str, level: str, device: torch.device) -> tuple[PixelCNNPrior, dict]:
    """The best-NLL top or bottom prior trained on a stage-1 run's codes."""
    ckpt = load_checkpoint(prior_dir(vqvae_run, level))
    model = build_prior(level, ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


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


def plot_vqvae_history(history: list[dict], path: str, levels: tuple[str, ...]):
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


def plot_prior_history(histories: dict[str, list[dict]], path: str):
    """
    Plot train and validation NLL in bits per code, one panel per prior level,
    e.g. {"top": top_history, "bottom": bottom_history}.
    """
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4), squeeze=False)
    for ax, (level, history) in zip(axes[0], histories.items()):
        epochs = [h["epoch"] for h in history]
        val = [h["val_bits_per_code"] for h in history]
        ax.plot(epochs, [h["train_bits_per_code"] for h in history], label="train")
        ax.plot(epochs, val, label="validate")
        best = int(np.argmin(val))
        ax.axvline(epochs[best], color="grey", linestyle=":",
                   label=f"best {val[best]:.3f} (epoch {epochs[best]})")
        ax.set_title(f"{level.capitalize()} prior NLL")
        ax.set_xlabel("epoch")
        ax.set_ylabel("bits per code")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_samples(samples: dict[float, torch.Tensor], path: str):
    """Grid of generated slices, one row per sampling temperature."""
    temperatures = list(samples)
    n = len(samples[temperatures[0]])
    fig, axes = plt.subplots(len(temperatures), n, figsize=(2.6 * n, 1.5 * len(temperatures)),
                             squeeze=False, layout="constrained")
    for row, t in enumerate(temperatures):
        for col in range(n):
            ax = axes[row, col]
            ax.imshow(to_display(samples[t][col].cpu()), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"T = {t:g}")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

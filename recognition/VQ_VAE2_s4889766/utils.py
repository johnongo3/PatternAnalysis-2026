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
import csv
import math
import os
import warnings
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
import yaml
from scipy import linalg
from torchmetrics.functional.image import (peak_signal_noise_ratio,
                                           structural_similarity_index_measure)

from dataset import MASK_LABELS, to_display
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


def code_usage(codes: torch.Tensor, codebook_size: int) -> dict:
    """Perplexity and number of distinct codes used across a set of code maps."""
    counts = torch.bincount(codes.flatten().long(), minlength=codebook_size)
    return {"perplexity": perplexity_from_counts(counts), "codes_used": int((counts > 0).sum()),
            "codebook_size": codebook_size}


def read_history(run_dir: str) -> list[dict]:
    """
    Per-epoch history from history.csv, with numbers as floats (nan where an
    epoch has no value). Cheaper than loading last.pt just for the history.
    """
    with open(os.path.join(run_dir, "history.csv"), newline="") as f:
        return [{k: float(v) if v != "" else float("nan") for k, v in row.items()}
                for row in csv.DictReader(f)]


def summarise(values: torch.Tensor) -> dict:
    """Mean, standard deviation, median, range and count of a 1-D tensor."""
    values = values.double()
    sd = values.std().item() if len(values) > 1 else 0.0
    return {"mean": values.mean().item(), "sd": sd, "median": values.median().item(),
            "min": values.min().item(), "max": values.max().item(), "n": len(values)}


def summarise_by_group(values: torch.Tensor, groups: list) -> dict:
    """summarise() per group label, e.g. per patient."""
    return {g: summarise(values[[i for i, x in enumerate(groups) if x == g]])
            for g in sorted(set(groups))}


def synchronize(device: torch.device):
    """Wait for queued GPU work, so wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@lru_cache(maxsize=None)
def _inception(device: str):
    from torchmetrics.image.fid import NoTrainInceptionV3 # needs torch-fidelity
    return NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()


@torch.no_grad()
def inception_features(images: torch.Tensor, device: torch.device,
                       batch_size: int = 64) -> torch.Tensor:
    """
    2048-d pool features of the standard FID Inception-v3 network for (N, 1, H, W)
    images in [0, 1]. The grey channel is repeated three times, and the network
    resizes to 299 x 299 itself.
    """
    net = _inception(str(device))
    features = []
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size].to(device).float().clamp(0, 1)
        x = (x * 255).round().to(torch.uint8).repeat(1, 3, 1, 1)
        features.append(net(x).cpu().double())
    return torch.cat(features)


def frechet_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    FID between two sets of features:
    ||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a S_b)^(1/2)).
    With fewer images than the 2048 feature dimensions (e.g. 512 samples) the
    covariances are singular; the square root is still well defined, as in the
    standard implementations, but FID is biased upward at small sample sizes.
    """
    a, b = a.double().numpy(), b.double().numpy()
    mu_a, mu_b = a.mean(0), b.mean(0)
    s_a, s_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning) # singular, see above
        covmean = linalg.sqrtm(s_a @ s_b)
        if not np.isfinite(covmean).all():
            offset = np.eye(len(s_a)) * 1e-6
            covmean = linalg.sqrtm((s_a + offset) @ (s_b + offset))
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(s_a + s_b - 2 * covmean.real))


@torch.no_grad()
def nearest_neighbours(queries: torch.Tensor, references: torch.Tensor, device: torch.device,
                       k: int = 5, chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """k nearest references to each query by Euclidean pixel distance: (distances, indices)."""
    ref = references.flatten(1).to(device).float()
    ref_sq = ref.pow(2).sum(1)
    dists, idxs = [], []
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk].flatten(1).to(device).float()
        d = (q.pow(2).sum(1, keepdim=True) - 2 * q @ ref.t() + ref_sq).clamp_min(0)
        v, i = d.topk(k, dim=1, largest=False)
        dists.append(v.sqrt().cpu())
        idxs.append(i.cpu())
    return torch.cat(dists), torch.cat(idxs)


@torch.no_grad()
def nearest_by_ssim(queries: torch.Tensor, references: torch.Tensor, device: torch.device,
                    k: int = 5, chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Most similar reference to each query by SSIM: shortlist the k closest by
    pixel distance, then keep the shortlisted one with the highest SSIM.
    Returns (ssim, index) per query.
    """
    _, shortlist = nearest_neighbours(queries, references, device, k)
    q = queries.repeat_interleave(k, dim=0)
    r_idx = shortlist.flatten()
    scores = torch.cat([ssim(q[s:s + chunk].to(device).float(),
                             references[r_idx[s:s + chunk].to(references.device)].to(device).float()).cpu()
                        for s in range(0, len(q), chunk)]).view(-1, k)
    best, j = scores.max(1)
    return best, shortlist.gather(1, j[:, None]).squeeze(1)


@torch.no_grad()
def max_code_agreement(queries: torch.Tensor, references: torch.Tensor, device: torch.device,
                       chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each query code map, the fraction of positions matching its closest
    reference code map, and that reference's index. 1.0 is an exact copy.
    """
    ref = references.flatten(1).to(device)
    best, idxs = [], []
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk].flatten(1).to(device)
        matches = (q[:, None, :] == ref[None]).sum(2) # (chunk, n_references)
        v, i = matches.max(1)
        best.append(v.cpu().double() / ref.shape[1])
        idxs.append(i.cpu())
    return torch.cat(best), torch.cat(idxs)


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


def plot_samples(samples: dict[str, torch.Tensor], path: str):
    """Grid of generated slices, one row per temperature key ('0.8' or top/bottom '0.9/0.7')."""
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
        axes[row, 0].set_ylabel(f"T = {t}")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_nn_histogram(similarities: dict[str, torch.Tensor], reference: str, path: str):
    """
    Overlaid histograms of each set's nearest-training-slice SSIM, with the
    reference set's maximum marked: anything to its right is closer to a
    training slice than any unseen patient's slice gets.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 51)
    for i, (label, values) in enumerate(similarities.items()):
        ax.hist(values.numpy(), bins=bins, density=True, histtype="step", linewidth=1.5,
                color="black" if label == reference else f"C{i}", label=label)
    ax.axvline(float(similarities[reference].max()), color="black", linestyle=":",
               label=f"{reference} max")
    ax.set_xlabel("SSIM to nearest training slice")
    ax.set_ylabel("density")
    ax.set_title("Nearest-neighbour audit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_nn_pairs(queries: torch.Tensor, neighbours: torch.Tensor, captions: list[str],
                  path: str):
    """Each query (top row) above its nearest training slice (bottom row)."""
    n = len(queries)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 3.4), squeeze=False, layout="constrained")
    for col in range(n):
        for row, image in enumerate((queries[col], neighbours[col])):
            ax = axes[row, col]
            ax.imshow(to_display(image.float()), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, col].set_title(captions[col], fontsize=8)
    axes[0, 0].set_ylabel("sample")
    axes[1, 0].set_ylabel("nearest train")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_candidates(images: torch.Tensor, captions: list[str], path: str, per_row: int = 6):
    """Generated slices in a grid, each titled with its caption (e.g. its sample index)."""
    rows = math.ceil(len(images) / per_row)
    fig, axes = plt.subplots(rows, per_row, figsize=(2.6 * per_row, 1.6 * rows), squeeze=False,
                             layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, caption in zip(axes.flat, images, captions):
        ax.imshow(to_display(image.float()), cmap="gray", vmin=0, vmax=1)
        ax.set_title(caption, fontsize=8)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# Outline colour and line style per mask label. Body is left out: its margin
# already shows in the image. Three hues that stay distinct under colour-vision
# deficiency, with prostate dashed so it is not told from rectum by colour alone.
OUTLINES = {2: ("#3987e5", "solid"), 3: ("#d95926", "solid"), 4: ("#199e70", "solid"),
            5: ("#199e70", "dashed")}


def plot_autopsy(columns: dict[str, torch.Tensor], masks: np.ndarray, captions: list[str],
                 outlined: tuple[str, ...], path: str):
    """
    One row per sample and one column per entry of columns, each (N, 1, H, W).
    Columns named in outlined get the structures of the matching mask (N, H, W)
    outlined over them. Columns whose name starts with "|" are error maps, drawn
    in grey (not inferno) so the coloured outlines stay visible on them.
    """
    names = list(columns)
    n = len(captions)
    fig, axes = plt.subplots(n, len(names), figsize=(2.6 * len(names), 1.5 * n), squeeze=False,
                             layout="constrained")
    for col, name in enumerate(names):
        images = columns[name].detach().cpu().float()
        is_error = name.startswith("|")
        vmax = float(images.quantile(0.995)) if is_error else 1.0
        for row in range(n):
            ax = axes[row, col]
            im = ax.imshow(to_display(images[row]), cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if name in outlined:
                mask = to_display(masks[row])
                for label, (colour, style) in OUTLINES.items():
                    if (mask == label).any():
                        ax.contour((mask == label).astype(float), levels=[0.5], colors=colour,
                                   linestyles=style, linewidths=1.0)
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(captions[row], fontsize=7)
        if is_error:
            fig.colorbar(im, ax=axes[:, col].tolist(), shrink=0.6)
    handles = [plt.Line2D([], [], color=colour, linestyle=style, label=MASK_LABELS[label])
               for label, (colour, style) in OUTLINES.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=8, frameon=False, title="nearest training slice's mask", title_fontsize=8)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_interpolations(frames: torch.Tensor, alphas: torch.Tensor, captions: list[str],
                        path: str):
    """One row per pair of (P, S, 1, H, W) frames, one column per blend weight."""
    pairs, steps = frames.shape[:2]
    fig, axes = plt.subplots(pairs, steps, figsize=(1.9 * steps, 1.2 * pairs), squeeze=False,
                             layout="constrained")
    for p in range(pairs):
        for s in range(steps):
            ax = axes[p, s]
            ax.imshow(to_display(frames[p, s]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if p == 0:
                ax.set_title(f"a = {float(alphas[s]):.2f}", fontsize=8)
        axes[p, 0].set_ylabel(captions[p], fontsize=7)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_step_changes(changes: torch.Tensor, captions: list[str], path: str):
    """1 - SSIM between consecutive interpolation frames, one line per pair."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for p, caption in enumerate(captions):
        ax.plot(range(1, changes.shape[1] + 1), changes[p].numpy(), marker="o", label=caption)
    ax.set_xlabel("interpolation step")
    ax.set_ylabel("1 - SSIM to previous frame")
    ax.set_title("Interpolation smoothness (a flat line is a smooth path)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_embedding(points: np.ndarray, labels: list[str], path: str, title: str):
    """2-D scatter of feature embeddings (e.g. t-SNE), coloured by set."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, label in enumerate(dict.fromkeys(labels)):
        mask = np.array([l == label for l in labels])
        real = label.startswith("real")
        ax.scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.5 if real else 0.7,
                   color="grey" if label == "real train" else f"C{i}", label=label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)

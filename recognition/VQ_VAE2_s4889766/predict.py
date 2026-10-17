"""
Evaluate a trained VQ-VAE-2 run on the test split and save figures.

Usage: python predict.py RUN_DIR [--root DATA_ROOT] [--n 6] [--seed 0] [--out assets]

Loads RUN_DIR/best.pt (the highest validation SSIM) and reports test MSE, SSIM
and PSNR for the full reconstruction, plus the top level alone for two-level
models. Metrics are written to RUN_DIR/test_metrics.json, and to --out:
    loss_curves_<run>.png      training curves from RUN_DIR/last.pt
    reconstructions_<run>.png  test slices: original, top only, full, |error|
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HipMRISlices
from train import PROJECT_DIR, build_model, pick_device
from utils import plot_history, plot_reconstructions, psnr, ssim


def load_run(run_dir: str, device: torch.device):
    """Load the best-SSIM model and its config from a run directory."""
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=True)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Mean and std over test images of MSE, SSIM and PSNR, per reconstruction type."""
    per_image = {}

    def add(name, recon, x):
        recon = recon.clamp(0, 1)
        for metric, values in [("mse", F.mse_loss(recon, x, reduction="none").mean((1, 2, 3))),
                               ("ssim", ssim(recon, x)), ("psnr", psnr(recon, x))]:
            per_image.setdefault(f"{name}_{metric}", []).append(values.cpu())

    for x in loader:
        x = x.to(device)
        add("full", model(x)["recon"], x)
        if model.levels == 2:
            add("top_only", model.reconstruct_top_only(x), x)

    results = {}
    for key, chunks in per_image.items():
        values = torch.cat(chunks)
        results[key] = {"mean": values.mean().item(), "std": values.std().item()}
    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate a VQ-VAE-2 run on the test split.")
    p.add_argument("run_dir")
    p.add_argument("--root", help="data root (default: data.root in the run's config)")
    p.add_argument("--n", type=int, default=6, help="test slices in the reconstruction grid")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(PROJECT_DIR, "assets"))
    args = p.parse_args()

    device = pick_device()
    model, ckpt = load_run(args.run_dir, device)
    cfg = ckpt["config"]
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    print(f"Loaded {run_name} from epoch {ckpt['epoch']} "
          f"(val SSIM {ckpt['metrics']['val_ssim']:.4f})")

    test = HipMRISlices(args.root or cfg["data"]["root"], "test", cfg["data"]["size"],
                        cfg["data"]["clip_percentile"])
    loader = DataLoader(test, batch_size=cfg["train"]["batch_size"])
    results = evaluate(model, loader, device)
    with open(os.path.join(args.run_dir, "test_metrics.json"), "w") as f:
        json.dump({"epoch": ckpt["epoch"], "n_test": len(test), **results}, f, indent=2)
    for key, v in results.items():
        print(f"test {key:>15}: {v['mean']:.4f} ± {v['std']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    last_path = os.path.join(args.run_dir, "last.pt")
    if os.path.exists(last_path):
        history = torch.load(last_path, map_location="cpu", weights_only=True)["history"]
        plot_history(history, os.path.join(args.out, f"loss_curves_{run_name}.png"),
                     model.level_names)

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(test), size=args.n, replace=False))
    x = torch.stack([test[i] for i in idx]).to(device)
    with torch.no_grad():
        full = model(x)["recon"].clamp(0, 1)
        columns = {"original": x}
        if model.levels == 2:
            columns["top only"] = model.reconstruct_top_only(x).clamp(0, 1)
        columns["full"] = full
        columns["|full - original|"] = (full - x).abs()
    titles = [f"case {test.records[i].case:03d} wk {test.records[i].week} "
              f"sl {test.records[i].slice}" for i in idx]
    plot_reconstructions(columns, titles, os.path.join(args.out, f"reconstructions_{run_name}.png"))
    print(f"Saved figures to {args.out}")


if __name__ == "__main__":
    main()

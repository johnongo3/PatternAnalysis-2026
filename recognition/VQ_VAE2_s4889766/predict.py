"""
Evaluate and sample from a trained VQ-VAE-2 run.

Usage:
    python predict.py RUN_DIR [--root DATA_ROOT] [--n 6] [--seed 0] [--out assets]
    python predict.py RUN_DIR --sample [--temperatures 0.8 1.0] [--n-samples N]

Reconstruction (the default) loads RUN_DIR/best.pt (the highest validation
SSIM) and reports test MSE, SSIM and PSNR for the full reconstruction, plus the
top level alone for two-level models. Metrics go to RUN_DIR/test_metrics.json,
and to --out:
    loss_curves_<run>.png      stage-1 training curves
    reconstructions_<run>.png  test slices: original, top only, full, |error|

Sampling (--sample) needs the run's priors from train.py: --stage top and
--stage bottom for a two-level run, --stage bottom alone for the single-level
baseline. For each temperature it samples top codes from the top prior, then
bottom codes from the bottom prior given those (or the baseline's codes from
its one prior), then decodes them with the stage-1 decoder, in batches from a
fixed seed. Images and codes go to
RUN_DIR/samples/samples_t<T>.pt, and to --out:
    samples_<run>.png          the first --n samples at each temperature
    prior_nll_<run>.png        train and validation bits per code of each prior
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HipMRISlices
from utils import (load_checkpoint, load_prior, load_vqvae, pick_device, plot_prior_history,
                   plot_reconstructions, plot_samples, plot_vqvae_history, prior_dir, psnr,
                   save_atomic, ssim)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


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


@torch.no_grad()
def sample_images(vqvae, priors: dict, n: int, temperature: float, seed: int,
                  batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Generate n slices from the priors, then decode them with the stage-1 decoder.
    Two-level: the top prior, then the bottom prior given the top codes.
    Single-level: the one prior over the bottom codes. One generator seeded once
    drives every batch, so a given (seed, batch_size) always gives the same samples.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    out = {"images": [], **{level: [] for level in priors}}
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        if "top" in priors:
            top = priors["top"].sample(b, temperature, generator)
            codes = {"top": top, "bottom": priors["bottom"].sample(top, temperature, generator)}
        else:
            codes = {"bottom": priors["bottom"].sample(b, temperature, generator)}
        out["images"].append(vqvae.decode_codes(codes["bottom"], codes.get("top")).clamp(0, 1).cpu())
        for level, c in codes.items():
            out[level].append(c.to(torch.int16).cpu())
        print(f"  T={temperature:g}: {start + b}/{n} samples")
    return {k: torch.cat(v) for k, v in out.items()}


def reconstruct(args, device: torch.device):
    model, ckpt = load_vqvae(args.run_dir, device)
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
    if os.path.exists(os.path.join(args.run_dir, "last.pt")):
        history = load_checkpoint(args.run_dir, "last.pt")["history"]
        plot_vqvae_history(history, os.path.join(args.out, f"loss_curves_{run_name}.png"),
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


def sample(args, device: torch.device):
    vqvae, _ = load_vqvae(args.run_dir, device)
    priors, ckpts = {}, {}
    for level in vqvae.level_names:
        priors[level], ckpts[level] = load_prior(args.run_dir, level, device)
        print(f"Loaded {level} prior from epoch {ckpts[level]['epoch']} "
              f"(val {ckpts[level]['metrics']['val_bits_per_code']:.4f} bits/code)")
    sample_cfg = ckpts["bottom"]["config"]["sample"]
    temperatures = args.temperatures or sample_cfg["temperatures"]
    n = args.n_samples or sample_cfg["n_samples"]
    run_name = os.path.basename(os.path.normpath(args.run_dir))

    samples_dir = os.path.join(args.run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    previews = {}
    for t in temperatures:
        out = sample_images(vqvae, priors, n, t, sample_cfg["seed"],
                            sample_cfg["batch_size"], device)
        save_atomic({"images": out["images"].half(), **{lv: out[lv] for lv in priors},
                     "temperature": t, "seed": sample_cfg["seed"]},
                    os.path.join(samples_dir, f"samples_t{t:g}.pt"))
        previews[t] = out["images"][:args.n]

    plot_samples(previews, os.path.join(args.out, f"samples_{run_name}.png"))
    histories = {level: load_checkpoint(prior_dir(args.run_dir, level), "last.pt")["history"]
                 for level in priors}
    plot_prior_history(histories, os.path.join(args.out, f"prior_nll_{run_name}.png"))
    print(f"Saved {n} samples per temperature to {samples_dir} and figures to {args.out}")


def main():
    p = argparse.ArgumentParser(description="Evaluate or sample from a VQ-VAE-2 run.")
    p.add_argument("run_dir", help="stage-1 run directory, e.g. checkpoints/vqvae2_L2_ssim0")
    p.add_argument("--sample", action="store_true",
                   help="generate new slices with the trained priors instead of reconstructing")
    p.add_argument("--root", help="data root for reconstruction (default: the run's config)")
    p.add_argument("--n", type=int, default=6,
                   help="slices per row of the reconstruction or sample figure")
    p.add_argument("--seed", type=int, default=0, help="choice of test slices to reconstruct")
    p.add_argument("--temperatures", type=float, nargs="+",
                   help="sampling temperatures (default: sample.temperatures in the config)")
    p.add_argument("--n-samples", type=int,
                   help="samples per temperature (default: sample.n_samples in the config)")
    p.add_argument("--out", default=os.path.join(PROJECT_DIR, "assets"))
    args = p.parse_args()

    device = pick_device()
    if args.sample:
        sample(args, device)
    else:
        reconstruct(args, device)


if __name__ == "__main__":
    main()

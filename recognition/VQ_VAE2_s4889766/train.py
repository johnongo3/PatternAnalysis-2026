"""
Train every stage of the VQ-VAE-2 pipeline on the HipMRI slices.

Stages, run in this order:
    vqvae   stage 1, the VQ-VAE-2 itself (the default)
    encode  encode every split with the frozen stage-1 model into integer code maps
    top     stage 2, the top prior over the top codes
    bottom  stage 2, the bottom prior over the bottom codes given the true top codes

Usage:
    python train.py [--levels 1|2] [--ssim-loss-weight W] [--resume]
    python train.py --stage encode [--vqvae-run checkpoints/vqvae2_L2_ssim0]
    python train.py --stage top    [--vqvae-run checkpoints/vqvae2_L2_ssim0] [--resume]
    python train.py --stage bottom [--vqvae-run checkpoints/vqvae2_L2_ssim0] [--resume]

For the single-level baseline, encode as above and train its one unconditional
prior (prior_single in the config) with:
    python train.py --stage bottom --vqvae-run checkpoints/vqvae2_L1_ssim0 [--resume]

Every training stage runs through the same loop (fit): train and validate each
epoch, log to <run_dir>/history.csv and loss_curves.png, keep best.pt on the
monitored validation metric (SSIM for stage 1, NLL in bits per code for the
priors) and last.pt for --resume, and stop early after `patience` epochs without
improvement. The prior stages encode the codes first if they are missing or
came from an older stage-1 checkpoint.
"""
import argparse
import csv
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SPLITS, CodeDataset, build_datasets, encode_dataset
from modules import BottomPrior, build_prior, build_vqvae, prior_section
from utils import (codes_dir, load_checkpoint, load_config, load_vqvae, perplexity_from_counts,
                   pick_device, plot_prior_history, plot_vqvae_history, prior_dir, psnr,
                   save_atomic, ssim)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# Per-epoch functions: one pass over a loader, training when given an optimizer.

def vqvae_epoch(model, loader, device, optimizer=None, ssim_weight=0.0, limit_batches=None):
    """
    One pass of the VQ-VAE-2. Evaluation (no optimizer) makes no codebook
    updates and also computes SSIM and PSNR. Losses and metrics are averaged per
    image, and perplexity is taken from the code usage histogram over the whole
    epoch rather than averaged over batches.
    """
    training = optimizer is not None
    model.train(training)
    prefix = "train" if training else "val"
    sums = {}
    counts = {level: torch.zeros(model.quant_b.codebook_size, device=device)
              for level in model.level_names}
    n_images = 0

    def add(key, value):
        sums[key] = sums.get(key, 0.0) + float(value)

    with torch.set_grad_enabled(training):
        # disable=None hides the bar when not on a terminal (e.g. Slurm logs)
        for i, x in enumerate(tqdm(loader, desc=prefix, leave=False, disable=None)):
            if limit_batches is not None and i >= limit_batches:
                break
            x = x.to(device, non_blocking=True)
            n = x.shape[0]
            out = model(x)
            recon_loss = F.mse_loss(out["recon"], x)
            loss = recon_loss + sum(out["commitment"].values())

            if training:
                if ssim_weight > 0:
                    loss = loss + ssim_weight * (1 - ssim(out["recon"], x).mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                add("ssim", ssim(out["recon"], x).sum())
                add("psnr", psnr(out["recon"], x).sum())
                if model.levels == 2:
                    top_only = model.reconstruct_top_only(x)
                    add("ssim_top_only", ssim(top_only, x).sum())
                    add("psnr_top_only", psnr(top_only, x).sum())

            add("recon", recon_loss.item() * n)
            for level in model.level_names:
                add(f"commit_{level}", out["commitment"][level].item() * n)
                counts[level] += torch.bincount(out["codes"][level].flatten(),
                                                minlength=len(counts[level]))
            n_images += n

    metrics = {f"{prefix}_{k}": v / n_images for k, v in sums.items()}
    for level, c in counts.items():
        metrics[f"{prefix}_perplexity_{level}"] = perplexity_from_counts(c)
        metrics[f"{prefix}_codes_used_{level}"] = int((c > 0).sum())
    return metrics


def prior_epoch(model, level, loader, device, optimizer=None, limit_batches=None):
    """
    One pass of a prior over batches of code maps, modelling the codes of
    `level`. The bottom prior of a two-level model is conditioned on the true
    top codes (teacher forcing); the top prior and the single-level baseline's
    prior are unconditional. Returns NLL in bits per code and next-code accuracy.
    """
    training = optimizer is not None
    model.train(training)
    prefix = "train" if training else "val"
    nll_sum, correct, n_codes = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for i, batch in enumerate(tqdm(loader, desc=prefix, leave=False, disable=None)):
            if limit_batches is not None and i >= limit_batches:
                break
            codes = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = codes[level]
            if isinstance(model, BottomPrior):
                logits = model(target, codes["top"])
            else:
                logits = model(target)
            loss = F.cross_entropy(logits, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            nll_sum += loss.item() * target.numel()
            correct += (logits.argmax(1) == target).sum().item()
            n_codes += target.numel()

    return {f"{prefix}_bits_per_code": nll_sum / n_codes / math.log(2),
            f"{prefix}_accuracy": correct / n_codes}


def format_vqvae_epoch(r: dict, epochs: int, levels: tuple[str, ...], improved: bool) -> str:
    commit = " ".join(f"{lv[0]} {r[f'train_commit_{lv}']:.2e}" for lv in levels)
    ppl = " ".join(f"{lv[0]} {r[f'train_perplexity_{lv}']:.0f}/{r[f'val_perplexity_{lv}']:.0f}"
                   for lv in levels)
    return (f"epoch {r['epoch']:3d}/{epochs}  recon {r['train_recon']:.5f}/{r['val_recon']:.5f}"
            f"  commit {commit}  val SSIM {r['val_ssim']:.4f} PSNR {r['val_psnr']:.2f}"
            f"  ppl {ppl}  {r['seconds']:.0f}s" + ("  * best" if improved else ""))


def format_prior_epoch(r: dict, epochs: int, improved: bool) -> str:
    return (f"epoch {r['epoch']:3d}/{epochs}"
            f"  bits/code {r['train_bits_per_code']:.4f}/{r['val_bits_per_code']:.4f}"
            f"  accuracy {r['train_accuracy']:.3f}/{r['val_accuracy']:.3f}"
            f"  {r['seconds']:.0f}s" + ("  * best" if improved else ""))


# The epoch loop shared by every training stage.

def write_history(history: list[dict], path: str):
    # union of keys, so a resumed run that gained new metrics still writes
    fieldnames = list(dict.fromkeys(k for record in history for k in record))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def fit(model, optimizer, run_epoch, train_loader, val_loader, loader_rng, run_dir: str,
        cfg: dict, train_cfg: dict, monitor: str, mode: str, describe, plot, resume: bool):
    """
    Train, validate, log, checkpoint and early-stop, resuming from
    <run_dir>/last.pt when asked.

    run_epoch(loader, optimizer=None): one pass, returning train_/val_ metrics
    monitor: validation metric that selects best.pt and drives early stopping
    mode: "max" if a higher monitor value is better, "min" if lower is better
    describe(record, improved): the console line for an epoch
    plot(history, path): draws the training curves
    """
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")
    epochs, patience, min_delta = train_cfg["epochs"], train_cfg["patience"], train_cfg["min_delta"]
    sign = 1 if mode == "max" else -1
    history, best, stale_epochs, start_epoch = [], None, 0, 1

    if os.path.exists(last_path):
        if not resume:
            raise SystemExit(f"{last_path} already exists. Pass --resume to continue "
                             "that run, or choose a new run directory.")
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        loader_rng.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        history, stale_epochs = state["history"], state["stale_epochs"]
        start_epoch = state["epoch"] + 1
        best = state.get("best", state.get("best_ssim")) # best_ssim: older stage-1 runs
        print(f"Resumed {run_dir} from epoch {state['epoch']} (best {monitor} {best:.4f})")
        if stale_epochs >= patience:
            print(f"Run already stopped early ({stale_epochs} epochs without improvement)")
            return
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        start = time.time()
        record = {"epoch": epoch, **run_epoch(train_loader, optimizer), **run_epoch(val_loader)}
        record["seconds"] = round(time.time() - start, 1)
        history.append(record)

        improved = best is None or sign * (record[monitor] - best) > min_delta
        if improved:
            best, stale_epochs = record[monitor], 0
            save_atomic({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                         "metrics": record}, best_path)
        else:
            stale_epochs += 1

        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": cfg, "epoch": epoch, "history": history, "best": best,
                     "stale_epochs": stale_epochs, "loader_rng": loader_rng.get_state(),
                     "torch_rng": torch.get_rng_state()}, last_path)
        write_history(history, os.path.join(run_dir, "history.csv"))
        plot(history, os.path.join(run_dir, "loss_curves.png"))
        print(describe(record, improved))

        if stale_epochs >= patience:
            print(f"Early stopping: {monitor} has not improved by {min_delta} "
                  f"for {patience} epochs (best {best:.4f})")
            break

    print(f"Done. Best {monitor} {best:.4f}; checkpoints in {run_dir}")


# Stage setup.

def make_loaders(train_set, val_set, batch_size: int, seed: int, num_workers: int,
                 device: torch.device):
    """Shuffled train and ordered validation loaders, plus the shuffle RNG for resuming."""
    loader_rng = torch.Generator().manual_seed(seed)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    return (DataLoader(train_set, shuffle=True, generator=loader_rng, **kwargs),
            DataLoader(val_set, shuffle=False, **kwargs), loader_rng)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def stage_train_config(cfg: dict, section: str, args) -> dict:
    """The stage's training section, with --epochs and --patience applied."""
    if args.epochs is not None:
        cfg[section]["epochs"] = args.epochs
    if args.patience is not None:
        cfg[section]["patience"] = args.patience
    return cfg[section]


def use_saved_architecture(run_dir: str, cfg: dict, section: str, resume: bool):
    """When resuming, rebuild the model with the architecture it was trained with."""
    last_path = os.path.join(run_dir, "last.pt")
    if resume and os.path.exists(last_path):
        saved = torch.load(last_path, map_location="cpu", weights_only=True)["config"]
        if saved[section] != cfg[section]:
            print(f"Resuming with the {section} config saved in the checkpoint")
        cfg[section] = saved[section]


def default_vqvae_run(cfg: dict) -> str:
    return f"vqvae2_L{cfg['vqvae']['levels']}_ssim{cfg['vqvae']['ssim_loss_weight']:g}"


def count_params(model) -> str:
    return f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params"


def train_vqvae(args, cfg: dict, device: torch.device):
    run_dir = os.path.join(args.checkpoint_dir, args.run_name or default_vqvae_run(cfg))
    use_saved_architecture(run_dir, cfg, "vqvae", args.resume)
    train_cfg = stage_train_config(cfg, "train", args)
    seed_everything(train_cfg["seed"])

    datasets = build_datasets(args.root or cfg["data"]["root"], cfg["data"]["size"],
                              cfg["data"]["clip_percentile"])
    train_loader, val_loader, loader_rng = make_loaders(
        datasets["train"], datasets["validate"], train_cfg["batch_size"], train_cfg["seed"],
        args.num_workers, device)
    model = build_vqvae(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    levels, ssim_weight = model.level_names, cfg["vqvae"]["ssim_loss_weight"]
    print(f"Training {run_dir} on {device}: {len(datasets['train'])} train / "
          f"{len(datasets['validate'])} validate slices, levels {levels}, {count_params(model)}")

    fit(model, optimizer,
        lambda loader, opt=None: vqvae_epoch(model, loader, device, opt, ssim_weight,
                                             args.limit_batches),
        train_loader, val_loader, loader_rng, run_dir, cfg, train_cfg,
        monitor="val_ssim", mode="max",
        describe=lambda r, improved: format_vqvae_epoch(r, train_cfg["epochs"], levels, improved),
        plot=lambda history, path: plot_vqvae_history(history, path, levels),
        resume=args.resume)


def encode_codes(vqvae_run: str, root: str | None, device: torch.device):
    """Encode every split with the frozen stage-1 model into <vqvae_run>/codes/<split>.pt."""
    model, ckpt = load_vqvae(vqvae_run, device)
    cfg = ckpt["config"]
    datasets = build_datasets(root or cfg["data"]["root"], cfg["data"]["size"],
                              cfg["data"]["clip_percentile"])
    os.makedirs(codes_dir(vqvae_run), exist_ok=True)
    for split, ds in datasets.items():
        codes = encode_dataset(model, ds, device, cfg["train"]["batch_size"])
        save_atomic({**codes, "names": [os.path.basename(r.path) for r in ds.records],
                     "vqvae_epoch": ckpt["epoch"]},
                    os.path.join(codes_dir(vqvae_run), f"{split}.pt"))
        shapes = ", ".join(f"{level} {tuple(c.shape[1:])}" for level, c in codes.items())
        print(f"{split:>8}: {len(ds)} slices -> {shapes} codes (stage-1 epoch {ckpt['epoch']})")


def load_code_datasets(vqvae_run: str, root: str | None,
                       device: torch.device) -> dict[str, CodeDataset]:
    """Code datasets for every split, encoding first if missing or made from an older best.pt."""
    best_epoch = load_checkpoint(vqvae_run)["epoch"]
    paths = {split: os.path.join(codes_dir(vqvae_run), f"{split}.pt") for split in SPLITS}
    try:
        datasets = {split: CodeDataset(path) for split, path in paths.items()}
        if all(ds.vqvae_epoch == best_epoch for ds in datasets.values()):
            return datasets
        print("Saved codes came from an older stage-1 checkpoint, re-encoding")
    except FileNotFoundError:
        print("No saved codes yet, encoding")
    encode_codes(vqvae_run, root, device)
    return {split: CodeDataset(path) for split, path in paths.items()}


def train_prior(level: str, args, cfg: dict, device: torch.device):
    vqvae_cfg = load_checkpoint(args.vqvae_run)["config"]
    if vqvae_cfg["vqvae"]["levels"] == 1 and level != "bottom":
        raise SystemExit("A single-level stage-1 model only has bottom codes; "
                         "train its prior with --stage bottom")
    # stage-1 settings come from its checkpoint, prior settings from the config file
    cfg["data"], cfg["vqvae"] = vqvae_cfg["data"], vqvae_cfg["vqvae"]
    run_dir = prior_dir(args.vqvae_run, level)
    use_saved_architecture(run_dir, cfg, prior_section(level, cfg), args.resume)
    train_cfg = stage_train_config(cfg, "train_prior", args)
    seed_everything(train_cfg["seed"])

    codes = load_code_datasets(args.vqvae_run, args.root, device)
    train_loader, val_loader, loader_rng = make_loaders(
        codes["train"], codes["validate"], train_cfg["batch_size"], train_cfg["seed"],
        args.num_workers, device)
    model = build_prior(level, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    print(f"Training {run_dir} on {device}: {len(codes['train'])} train / "
          f"{len(codes['validate'])} validate code maps of {model.shape}, {count_params(model)}")

    fit(model, optimizer,
        lambda loader, opt=None: prior_epoch(model, level, loader, device, opt,
                                             args.limit_batches),
        train_loader, val_loader, loader_rng, run_dir, cfg, train_cfg,
        monitor="val_bits_per_code", mode="min",
        describe=lambda r, improved: format_prior_epoch(r, train_cfg["epochs"], improved),
        plot=lambda history, path: plot_prior_history({level: history}, path),
        resume=args.resume)


def parse_args():
    p = argparse.ArgumentParser(description="Train the VQ-VAE-2 pipeline on HipMRI slices.")
    p.add_argument("--stage", choices=("vqvae", "encode", "top", "bottom"), default="vqvae")
    p.add_argument("--config", default=os.path.join(PROJECT_DIR, "configs", "vqvae2.yml"))
    p.add_argument("--root", help="data root (default: data.root in the config)")
    p.add_argument("--levels", type=int, choices=(1, 2), help="override vqvae.levels")
    p.add_argument("--ssim-loss-weight", type=float, help="override vqvae.ssim_loss_weight")
    p.add_argument("--epochs", type=int, help="override the stage's epochs")
    p.add_argument("--patience", type=int, help="override the stage's patience")
    p.add_argument("--checkpoint-dir", default=os.path.join(PROJECT_DIR, "checkpoints"))
    p.add_argument("--run-name", help="stage-1 run name (default: vqvae2_L<levels>_ssim<weight>)")
    p.add_argument("--vqvae-run", help="stage-1 run directory for the encode, top and bottom "
                                       "stages (default: <checkpoint-dir>/<default run name>)")
    p.add_argument("--resume", action="store_true",
                   help="continue from the stage's last.pt if it exists")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-batches", type=int, help="cap batches per epoch, for smoke tests")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.levels is not None:
        cfg["vqvae"]["levels"] = args.levels
    if args.ssim_loss_weight is not None:
        cfg["vqvae"]["ssim_loss_weight"] = args.ssim_loss_weight
    args.vqvae_run = args.vqvae_run or os.path.join(args.checkpoint_dir, default_vqvae_run(cfg))
    device = pick_device()

    if args.stage == "vqvae":
        train_vqvae(args, cfg, device)
    elif args.stage == "encode":
        encode_codes(args.vqvae_run, args.root, device)
    else:
        train_prior(args.stage, args, cfg, device)


if __name__ == "__main__":
    main()

"""
Train the VQ-VAE-2 on the HipMRI slices.

Usage:
    python train.py [--config configs/vqvae2.yml] [--root DATA_ROOT] [--levels 1|2]
                    [--ssim-loss-weight W] [--run-name NAME] [--resume]

Each epoch logs, to the console and <run_dir>/history.csv: train and validation
reconstruction loss, train commitment loss and code perplexity per level, and
validation SSIM, PSNR and perplexity per level. best.pt keeps the weights with
the highest validation SSIM, and last.pt holds the full training state for
--resume. Training stops early once validation SSIM has not improved by
min_delta for `patience` epochs.
"""
import argparse
import csv
import os
import random
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from modules import VQVAE2
from utils import perplexity_from_counts, plot_history, psnr, ssim

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: dict) -> VQVAE2:
    """Build a VQVAE2 from the vqvae section of the config."""
    kwargs = {k: v for k, v in cfg["vqvae"].items() if k != "ssim_loss_weight"}
    return VQVAE2(in_channels=1, **kwargs)


def save_atomic(obj, path: str):
    """torch.save via a temp file, so an interrupted save never corrupts a checkpoint."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def run_epoch(model, loader, device, optimizer=None, ssim_weight=0.0, limit_batches=None):
    """
    One pass over loader. Trains when an optimizer is given; otherwise evaluates
    (no codebook updates) and also computes SSIM and PSNR. Losses and metrics
    are averaged per image, and perplexity is taken from the code usage
    histogram over the whole epoch rather than averaged over batches.
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


def write_history(history: list[dict], path: str):
    # union of keys, so a resumed run that gained new metrics still writes
    fieldnames = list(dict.fromkeys(k for record in history for k in record))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def format_epoch(r: dict, epochs: int, levels: tuple[str, ...], improved: bool) -> str:
    commit = " ".join(f"{lv[0]} {r[f'train_commit_{lv}']:.2e}" for lv in levels)
    ppl = " ".join(f"{lv[0]} {r[f'train_perplexity_{lv}']:.0f}/{r[f'val_perplexity_{lv}']:.0f}"
                   for lv in levels)
    return (f"epoch {r['epoch']:3d}/{epochs}  recon {r['train_recon']:.5f}/{r['val_recon']:.5f}"
            f"  commit {commit}  val SSIM {r['val_ssim']:.4f} PSNR {r['val_psnr']:.2f}"
            f"  ppl {ppl}  {r['seconds']:.0f}s" + ("  * best" if improved else ""))


def parse_args():
    p = argparse.ArgumentParser(description="Train the VQ-VAE-2 on HipMRI slices.")
    p.add_argument("--config", default=os.path.join(PROJECT_DIR, "configs", "vqvae2.yml"))
    p.add_argument("--root", help="data root (default: data.root in the config)")
    p.add_argument("--levels", type=int, choices=(1, 2), help="override vqvae.levels")
    p.add_argument("--ssim-loss-weight", type=float, help="override vqvae.ssim_loss_weight")
    p.add_argument("--epochs", type=int, help="override train.epochs")
    p.add_argument("--patience", type=int, help="override train.patience")
    p.add_argument("--run-name", help="default: vqvae2_L<levels>_ssim<weight>")
    p.add_argument("--checkpoint-dir", default=os.path.join(PROJECT_DIR, "checkpoints"))
    p.add_argument("--resume", action="store_true",
                   help="continue from <run_dir>/last.pt if it exists")
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
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience

    run_name = args.run_name or (f"vqvae2_L{cfg['vqvae']['levels']}"
                                 f"_ssim{cfg['vqvae']['ssim_loss_weight']:g}")
    run_dir = os.path.join(args.checkpoint_dir, run_name)
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")

    state = None
    if os.path.exists(last_path):
        if not args.resume:
            raise SystemExit(f"{last_path} already exists. Pass --resume to continue "
                             "that run, or choose a new --run-name.")
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["config"]["vqvae"] != cfg["vqvae"]:
            print("Resuming with the model config saved in the checkpoint")
        cfg["vqvae"] = state["config"]["vqvae"]
    os.makedirs(run_dir, exist_ok=True)

    train_cfg = cfg["train"]
    epochs, patience = train_cfg["epochs"], train_cfg["patience"]
    min_delta, ssim_weight = train_cfg["min_delta"], cfg["vqvae"]["ssim_loss_weight"]
    random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    device = pick_device()

    datasets = build_datasets(args.root or cfg["data"]["root"], cfg["data"]["size"],
                              cfg["data"]["clip_percentile"])
    loader_rng = torch.Generator().manual_seed(train_cfg["seed"])
    loader_kwargs = dict(batch_size=train_cfg["batch_size"], num_workers=args.num_workers,
                         pin_memory=device.type == "cuda",
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(datasets["train"], shuffle=True, generator=loader_rng,
                              **loader_kwargs)
    val_loader = DataLoader(datasets["validate"], shuffle=False, **loader_kwargs)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    levels = model.level_names

    history, best_ssim, stale_epochs, start_epoch = [], float("-inf"), 0, 1
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        loader_rng.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        history, best_ssim = state["history"], state["best_ssim"]
        stale_epochs, start_epoch = state["stale_epochs"], state["epoch"] + 1
        print(f"Resumed {run_name} from epoch {state['epoch']} "
              f"(best val SSIM {best_ssim:.4f})")
        if stale_epochs >= patience:
            print(f"Run already stopped early ({stale_epochs} epochs without improvement)")
            return

    print(f"Training {run_name} on {device}: {len(datasets['train'])} train / "
          f"{len(datasets['validate'])} validate slices, levels {levels}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    for epoch in range(start_epoch, epochs + 1):
        start = time.time()
        record = {"epoch": epoch}
        record.update(run_epoch(model, train_loader, device, optimizer, ssim_weight,
                                args.limit_batches))
        record.update(run_epoch(model, val_loader, device, limit_batches=args.limit_batches))
        record["seconds"] = round(time.time() - start, 1)
        history.append(record)

        improved = record["val_ssim"] > best_ssim + min_delta
        if improved:
            best_ssim, stale_epochs = record["val_ssim"], 0
            save_atomic({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                         "metrics": record}, best_path)
        else:
            stale_epochs += 1

        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": cfg, "epoch": epoch, "history": history,
                     "best_ssim": best_ssim, "stale_epochs": stale_epochs,
                     "loader_rng": loader_rng.get_state(),
                     "torch_rng": torch.get_rng_state()}, last_path)
        write_history(history, os.path.join(run_dir, "history.csv"))
        plot_history(history, os.path.join(run_dir, "loss_curves.png"), levels)
        print(format_epoch(record, epochs, levels, improved))

        if stale_epochs >= patience:
            print(f"Early stopping: val SSIM has not improved by {min_delta} "
                  f"for {patience} epochs (best {best_ssim:.4f})")
            break

    print(f"Done. Best val SSIM {best_ssim:.4f}; checkpoints in {run_dir}")


if __name__ == "__main__":
    main()

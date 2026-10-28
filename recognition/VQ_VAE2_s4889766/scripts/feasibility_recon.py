"""
Stage-1 reconstruction evidence for the feasibility review, on the validation
split only (the test split stays untouched until the final evaluation).

Reads the cached validation tensor directly, so it runs without the raw NIfTI
folder. Prints SSIM/PSNR overall and per patient, and saves
assets/feasibility_recon.png: original, reconstruction and absolute error for
the slice nearest the median SSIM in each of the first --rows patients.

Usage: python scripts/feasibility_recon.py [run_dir] [--rows 2]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from dataset import DEFAULT_CACHE_DIR, parse_filename, to_display  # noqa: E402
from utils import load_vqvae, pick_device, psnr, ssim, summarise_by_group  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", nargs="?",
                        default=os.path.join(PROJECT_DIR, "checkpoints", "vqvae2_L2_ssim0"))
    parser.add_argument("--rows", type=int, default=2, help="patients shown, one slice each")
    parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "assets", "feasibility_recon.png"))
    args = parser.parse_args()

    device = pick_device()
    model, ckpt = load_vqvae(args.run_dir, device)
    rows, cols = ckpt["config"]["data"]["size"]
    clip = ckpt["config"]["data"]["clip_percentile"]
    cached = torch.load(os.path.join(DEFAULT_CACHE_DIR, f"validate_{rows}x{cols}_p{clip}.pt"),
                        mmap=True, weights_only=True)
    images, records = cached["images"], [parse_filename(n) for n in cached["names"]]

    with torch.no_grad():
        recon = torch.cat([model(batch.to(device))["recon"].clamp(0, 1).cpu()
                           for batch in images.split(64)])

    patients = [f"{r.case:03d}" for r in records]
    s, p = ssim(recon, images), psnr(recon, images)
    print(f"{args.run_dir}, epoch {ckpt['epoch']}, {len(images)} validation slices")
    print(f"  SSIM {s.mean():.3f} ± {s.std():.3f} (min {s.min():.3f}), "
          f"PSNR {p.mean():.2f} ± {p.std():.2f} dB")
    for patient, stats in summarise_by_group(s, patients).items():
        print(f"  patient {patient}: SSIM {stats['mean']:.3f} (min {stats['min']:.3f}, "
              f"n={stats['n']})")

    # One slice per patient, nearest the median SSIM, so the figure is typical
    # rather than best case.
    gap = (s - s.median()).abs()
    idx = torch.tensor([min((i for i, q in enumerate(patients) if q == patient),
                            key=lambda i: float(gap[i]))
                        for patient in sorted(set(patients))[:args.rows]])
    columns = {"original": images[idx], "reconstruction": recon[idx],
               "|error|": (recon[idx] - images[idx]).abs()}
    vmax = float(columns["|error|"].quantile(0.995))

    # Sized to be shown at about half the page width, hence the large fonts.
    fig, axes = plt.subplots(args.rows, 3, figsize=(7.5, 1.35 * args.rows), squeeze=False,
                             layout="constrained")
    for c, (name, ims) in enumerate(columns.items()):
        error = name.startswith("|")
        for r in range(args.rows):
            ax = axes[r, c]
            im = ax.imshow(to_display(ims[r]), cmap="inferno" if error else "gray",
                           vmin=0, vmax=vmax if error else 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=15)
            if c == 0:
                rec = records[int(idx[r])]
                ax.set_ylabel(f"case {rec.case:03d}", fontsize=13)
        if error:
            bar = fig.colorbar(im, ax=axes[:, c].tolist(), shrink=0.9, pad=0.01)
            bar.ax.tick_params(labelsize=11)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

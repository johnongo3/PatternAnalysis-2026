"""
Evaluate trained VQ-VAE-2 runs: reconstruction, sampling, a memorisation audit,
FID, latent interpolations and resource use.

Usage, where RUN is a stage-1 run directory such as checkpoints/vqvae2_L2_ssim0:
    python predict.py reconstruct RUN   test SSIM and PSNR (mean ± SD, per patient)
                                        and codebook perplexity per level
    python predict.py sample RUN        timed sampling with the priors, per temperature
                                        (--bottom-temperatures sweeps the priors separately)
    python predict.py audit RUN         nearest-neighbour and code-copy memorisation checks
    python predict.py fid RUN           FID against the training slices, t-SNE of features
    python predict.py interpolate RUN   latent interpolations between test slices
    python predict.py all RUN           the five tasks above, in order
    python predict.py autopsy RUN       flawed samples at one --setting beside their nearest
                                        training slice
    python predict.py resources RUN...  parameters, training time, peak VRAM, sampling time
    python predict.py report RUN...     markdown tables comparing runs, from their results

Each task writes RUN/eval/<task>.json and saves figures to --out (assets/),
named after the run. reconstruct, audit, fid and interpolate read the slices
(--root, default: the run's config). sample needs the priors from train.py;
audit and fid need the samples.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset import CodeDataset, HipMRISlices, build_datasets
from modules import VQVAE2, BottomPrior, build_prior, build_vqvae
from utils import (code_usage, codes_dir, frechet_distance, inception_features, load_checkpoint,
                   load_prior, load_vqvae, max_code_agreement, nearest_by_ssim, pick_device,
                   plot_candidates, plot_embedding, plot_interpolations,
                   plot_nn_histogram, plot_nn_pairs, plot_prior_history, plot_reconstructions,
                   plot_samples, plot_step_changes, plot_vqvae_history, prior_dir, psnr,
                   read_history, save_atomic, ssim, summarise, summarise_by_group, synchronize)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS = ("reconstruct", "sample", "audit", "fid", "interpolate")


# Shared helpers.

def run_name(run_dir: str) -> str:
    return os.path.basename(os.path.normpath(run_dir))


def model_label(cfg: dict) -> str:
    """The model's name in results and tables, with the SSIM loss weight when it is used."""
    label = "VQ-VAE-2 (2 levels)" if cfg["vqvae"]["levels"] == 2 else "Baseline (1 level)"
    weight = cfg["vqvae"].get("ssim_loss_weight", 0)
    return f"{label}, SSIM loss {weight:g}" if weight else label


def figure_path(args, name: str) -> str:
    os.makedirs(args.out, exist_ok=True)
    return os.path.join(args.out, f"{name}_{run_name(args.run_dir)}.png")


def write_results(run_dir: str, task: str, results: dict):
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
    with open(os.path.join(run_dir, "eval", f"{task}.json"), "w") as f:
        json.dump(results, f, indent=2)


def read_results(run_dir: str, task: str) -> dict | None:
    path = os.path.join(run_dir, "eval", f"{task}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_split(args, cfg: dict, split: str) -> HipMRISlices:
    return HipMRISlices(args.root or cfg["data"]["root"], split, cfg["data"]["size"],
                        cfg["data"]["clip_percentile"])


def temperature_key(temperature: dict[str, float]) -> str:
    """
    Name of a sampling setting, given each level's temperature: '0.8' when every
    level samples at 0.8 (the form older results use), else 'top/bottom', e.g. '0.9/0.7'.
    """
    values = set(temperature.values())
    if len(values) == 1:
        return f"{values.pop():g}"
    return f"{temperature['top']:g}/{temperature['bottom']:g}"


def key_order(key: str) -> tuple[float, float]:
    """Sort temperature keys by top, then bottom temperature."""
    parts = [float(p) for p in key.split("/")]
    return parts[0], parts[-1]


def file_tag(key: str) -> str:
    """A temperature key made safe for file names: '0.9/0.7' becomes '0.9_0.7'."""
    return key.replace("/", "_")


def load_samples(run_dir: str) -> dict[str, dict]:
    """Saved samples per temperature key, as written by the sample task."""
    folder = os.path.join(run_dir, "samples")
    files = sorted(f for f in os.listdir(folder) if f.startswith("samples_t")) \
        if os.path.isdir(folder) else []
    if not files:
        raise SystemExit(f"No samples in {folder}; run: python predict.py sample {run_dir}")
    samples = [torch.load(os.path.join(folder, f), weights_only=True) for f in files]
    # older files hold one float temperature shared by every level
    keyed = {s["temperature"] if isinstance(s["temperature"], str) else f"{s['temperature']:g}": s
             for s in samples}
    return {key: keyed[key] for key in sorted(keyed, key=key_order)}


def load_codes(run_dir: str, split: str) -> dict[str, torch.Tensor]:
    path = os.path.join(codes_dir(run_dir), f"{split}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"No codes at {path}; run: python train.py --stage encode "
                         f"--vqvae-run {run_dir}")
    return CodeDataset(path).codes


@torch.no_grad()
def reconstruct_all(model, images: torch.Tensor, device, batch_size: int = 64) -> torch.Tensor:
    return torch.cat([model(images[s:s + batch_size].to(device))["recon"].clamp(0, 1).cpu()
                      for s in range(0, len(images), batch_size)])


def pm(stats: dict, digits: int = 3) -> str:
    return f"{stats['mean']:.{digits}f} ± {stats['sd']:.{digits}f}"


# Reconstruction.

@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, dict]:
    """
    Per-image MSE, SSIM and PSNR of the full reconstruction (and of the top level
    alone, for two-level models), plus every code map, over a loader.
    """
    per_image, codes = {}, {level: [] for level in model.level_names}

    def add(name, recon, x):
        recon = recon.clamp(0, 1)
        for metric, values in [("mse", F.mse_loss(recon, x, reduction="none").mean((1, 2, 3))),
                               ("ssim", ssim(recon, x)), ("psnr", psnr(recon, x))]:
            per_image.setdefault(f"{name}_{metric}", []).append(values.cpu())

    for x in loader:
        x = x.to(device)
        out = model(x)
        add("full", out["recon"], x)
        for level in codes:
            codes[level].append(out["codes"][level].cpu())
        if model.levels == 2:
            add("top_only", model.reconstruct_top_only(x), x)
    return ({k: torch.cat(v) for k, v in per_image.items()},
            {k: torch.cat(v) for k, v in codes.items()})


def reconstruct(args, device):
    model, ckpt = load_vqvae(args.run_dir, device)
    cfg = ckpt["config"]
    test = load_split(args, cfg, "test")
    per_image, codes = evaluate(model, DataLoader(test, batch_size=cfg["train"]["batch_size"]),
                                device)
    patients = [f"{r.case:03d}" for r in test.records]
    results = {
        "model": model_label(cfg), "epoch": ckpt["epoch"], "n_test": len(test),
        "overall": {k: summarise(v) for k, v in per_image.items()},
        "per_patient": {k: summarise_by_group(v, patients) for k, v in per_image.items()},
        "codebook": {level: code_usage(c, cfg["vqvae"]["codebook_size"])
                     for level, c in codes.items()},
    }
    write_results(args.run_dir, "reconstruct", results)

    print(f"{results['model']}, epoch {ckpt['epoch']}, {len(test)} test slices")
    for key, stats in results["overall"].items():
        by_patient = "  ".join(f"{p}: {pm(s)}" for p, s in results["per_patient"][key].items())
        print(f"  {key:>15}: {pm(stats, 4)}   [{by_patient}]")
    for level, usage in results["codebook"].items():
        print(f"  {level} codebook: perplexity {usage['perplexity']:.1f}, "
              f"{usage['codes_used']}/{usage['codebook_size']} codes used")

    plot_vqvae_history(read_history(args.run_dir), figure_path(args, "loss_curves"),
                       model.level_names)
    idx = np.sort(np.random.default_rng(args.seed).choice(len(test), size=args.n, replace=False))
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
    plot_reconstructions(columns, titles, figure_path(args, "reconstructions"))


# Sampling.

def temperature_settings(levels: tuple[str, ...], temperatures: list[float],
                         bottom_temperatures: list[float] | None) -> list[dict[str, float]]:
    """
    Each level's temperature for every setting to sample. Without
    bottom_temperatures, every level shares each of temperatures; with them, the
    top prior takes each of temperatures and the bottom prior each of
    bottom_temperatures, for every pair.
    """
    if not bottom_temperatures:
        return [{level: t for level in levels} for t in temperatures]
    if "top" not in levels:
        raise SystemExit("--bottom-temperatures needs a two-level model")
    return [{"top": t_top, "bottom": t_bottom}
            for t_top in temperatures for t_bottom in bottom_temperatures]


@torch.no_grad()
def sample_images(vqvae, priors: dict, n: int, temperature: dict[str, float], seed: int,
                  batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Generate n slices from the priors, then decode them with the stage-1 decoder.
    Two-level: the top prior, then the bottom prior given the top codes.
    Single-level: the one prior over the bottom codes. temperature maps each
    level to its sampling temperature. One generator seeded once drives every
    batch, so a given (seed, batch_size) always gives the same samples, and
    settings that share a top temperature share their top code maps.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    out = {"images": [], **{level: [] for level in priors}}
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        if "top" in priors:
            top = priors["top"].sample(b, temperature["top"], generator)
            codes = {"top": top,
                     "bottom": priors["bottom"].sample(top, temperature["bottom"], generator)}
        else:
            codes = {"bottom": priors["bottom"].sample(b, temperature["bottom"], generator)}
        out["images"].append(vqvae.decode_codes(codes["bottom"], codes.get("top")).clamp(0, 1).cpu())
        for level, c in codes.items():
            out[level].append(c.to(torch.int16).cpu())
        print(f"  T={temperature_key(temperature)}: {start + b}/{n} samples")
    return {k: torch.cat(v) for k, v in out.items()}


def sample(args, device):
    """
    Sample every temperature setting, saving each one as soon as it finishes. A
    setting already saved with the same number of samples, seed and batch size
    is skipped (unless --overwrite), so a job cut off by its time limit can be
    resubmitted and carries on from the next setting. With --bottom-temperatures,
    the two priors are swept separately (see temperature_settings).
    """
    vqvae, ckpt = load_vqvae(args.run_dir, device)
    priors, prior_ckpts = {}, {}
    for level in vqvae.level_names:
        priors[level], prior_ckpts[level] = load_prior(args.run_dir, level, device)
        print(f"Loaded {level} prior from epoch {prior_ckpts[level]['epoch']} "
              f"(val {prior_ckpts[level]['metrics']['val_bits_per_code']:.4f} bits/code)")
    sample_cfg = prior_ckpts["bottom"]["config"]["sample"]
    sweep = temperature_settings(vqvae.level_names, args.temperatures or sample_cfg["temperatures"],
                                 args.bottom_temperatures)
    n = args.n_samples or sample_cfg["n_samples"]

    seed, batch_size = sample_cfg["seed"], sample_cfg["batch_size"]
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type

    samples_dir = os.path.join(args.run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    previous = read_results(args.run_dir, "sample") or {}
    timing = previous.get("temperatures", {})
    previews = {}
    for temperature in sweep:
        key = temperature_key(temperature)
        path = os.path.join(samples_dir, f"samples_t{file_tag(key)}.pt")
        done = timing.get(key)
        # older results files keep these settings once, at the top level
        settings = {k: (done or {}).get(k, previous.get(k))
                    for k in ("n_samples", "seed", "batch_size")}
        if (not args.overwrite and done and os.path.exists(path)
                and settings == {"n_samples": n, "seed": seed, "batch_size": batch_size}):
            print(f"  T={key}: already sampled, skipping (--overwrite to redo)")
            previews[key] = torch.load(path, weights_only=True)["images"][:args.n].float()
            continue

        synchronize(device)
        start = time.perf_counter()
        out = sample_images(vqvae, priors, n, temperature, seed, batch_size, device)
        synchronize(device)
        seconds = time.perf_counter() - start
        print(f"  T={key}: {seconds:.0f}s, {seconds / n:.2f}s per image")
        save_atomic({"images": out["images"].half(), **{lv: out[lv] for lv in priors},
                     "temperature": key, "level_temperatures": temperature, "seed": seed}, path)
        timing[key] = {"seconds": seconds, "seconds_per_image": seconds / n,
                       "level_temperatures": temperature, "n_samples": n, "seed": seed,
                       "batch_size": batch_size, "device": device_name}
        write_results(args.run_dir, "sample", {
            "model": model_label(ckpt["config"]), "device": device_name, "n_samples": n,
            "batch_size": batch_size, "seed": seed, "temperatures": timing})
        previews[key] = out["images"][:args.n]

    plot_samples(previews, figure_path(args, "samples"))
    plot_prior_history({level: read_history(prior_dir(args.run_dir, level)) for level in priors},
                       figure_path(args, "prior_nll"))


# Memorisation audit.

def audit(args, device):
    """
    Nearest-neighbour audit: the SSIM of each sample to its most similar
    training slice. The reference is the test slices of unseen patients, passed
    through the same decoder so its blur affects both sides equally; a sample
    closer to a training slice than any reference slice gets is a possible copy.

    Code-copy check: the fraction of positions each sampled code map shares with
    its closest training code map, against the same fraction for test code maps.
    """
    vqvae, ckpt = load_vqvae(args.run_dir, device)
    cfg = ckpt["config"]
    slices = build_datasets(args.root or cfg["data"]["root"], cfg["data"]["size"],
                            cfg["data"]["clip_percentile"])
    train_images, train_records = slices["train"].images.to(device), slices["train"].records
    levels, k = vqvae.level_names, cfg["vqvae"]["codebook_size"]

    test_recon = reconstruct_all(vqvae, slices["test"].images, device)
    ref_ssim, _ = nearest_by_ssim(test_recon, train_images, device)
    ref_max, ref_p99 = float(ref_ssim.max()), float(ref_ssim.quantile(0.99))
    train_codes, test_codes = load_codes(args.run_dir, "train"), load_codes(args.run_dir, "test")
    results = {"model": model_label(cfg), "reference": {
        "nn_ssim": {**summarise(ref_ssim), "p99": ref_p99},
        "code_agreement": {lv: summarise(max_code_agreement(test_codes[lv], train_codes[lv],
                                                            device)[0]) for lv in levels},
        "code_usage": {lv: code_usage(test_codes[lv], k) for lv in levels}},
        "temperatures": {}}
    similarities = {"test (reference)": ref_ssim}

    for key, s in load_samples(args.run_dir).items():
        images = s["images"].float()
        nn_ssim, nn_idx = nearest_by_ssim(images, train_images, device)
        agreement = {lv: max_code_agreement(s[lv], train_codes[lv], device)[0] for lv in levels}
        results["temperatures"][key] = {
            "n": len(images),
            "nn_ssim": summarise(nn_ssim),
            "above_reference_max": int((nn_ssim > ref_max).sum()),
            "above_reference_p99": int((nn_ssim > ref_p99).sum()),
            "code_agreement": {lv: {**summarise(a), "exact_copies": int((a == 1).sum())}
                               for lv, a in agreement.items()},
            "code_usage": {lv: code_usage(s[lv], k) for lv in levels},
        }
        similarities[f"samples T={key}"] = nn_ssim

        closest = nn_ssim.argsort(descending=True)[:args.n]
        captions = [f"SSIM {nn_ssim[i]:.3f}\ntrain case {train_records[nn_idx[i]].case:03d} "
                    f"sl {train_records[nn_idx[i]].slice}" for i in closest]
        plot_nn_pairs(images[closest], train_images[nn_idx[closest].to(device)].cpu(), captions,
                      figure_path(args, f"nn_pairs_t{file_tag(key)}"))

    write_results(args.run_dir, "audit", results)
    plot_nn_histogram(similarities, "test (reference)", figure_path(args, "nn_audit"))
    print(f"Reference (test reconstructions): nearest-train SSIM "
          f"median {results['reference']['nn_ssim']['median']:.3f}, max {ref_max:.3f}")
    for t, r in results["temperatures"].items():
        copies = ", ".join(f"{lv} max {a['max']:.3f} ({a['exact_copies']} exact)"
                           for lv, a in r["code_agreement"].items())
        print(f"  T={t}: nearest-train SSIM median {r['nn_ssim']['median']:.3f}, "
              f"max {r['nn_ssim']['max']:.3f}, {r['above_reference_max']}/{r['n']} above the "
              f"reference max; code agreement {copies}")


# Distribution: FID and t-SNE.

def fid(args, device):
    """
    FID of each temperature's samples against the training slices, with two
    references at a similar sample size: unseen test slices (the floor from
    patient differences alone) and their reconstructions (adding decoder blur,
    which samples cannot avoid). Secondary to the audit: FID cannot detect copies.
    """
    vqvae, ckpt = load_vqvae(args.run_dir, device)
    cfg = ckpt["config"]
    slices = build_datasets(args.root or cfg["data"]["root"], cfg["data"]["size"],
                            cfg["data"]["clip_percentile"])
    samples = load_samples(args.run_dir)
    features = {
        "real train": inception_features(slices["train"].images, device),
        "real test": inception_features(slices["test"].images, device),
        "test reconstructions": inception_features(
            reconstruct_all(vqvae, slices["test"].images, device), device),
        **{f"samples T={key}": inception_features(s["images"], device)
           for key, s in samples.items()},
    }
    train = features["real train"]
    results = {"model": model_label(cfg), "n_train": len(train),
               "reference": {"test_vs_train": frechet_distance(features["real test"], train),
                             "test_reconstructions_vs_train":
                                 frechet_distance(features["test reconstructions"], train)},
               "temperatures": {key: frechet_distance(features[f"samples T={key}"], train)
                                for key in samples}}
    write_results(args.run_dir, "fid", results)
    print(f"FID vs train: test {results['reference']['test_vs_train']:.1f}, test reconstructions "
          f"{results['reference']['test_reconstructions_vs_train']:.1f}, " +
          ", ".join(f"T={t} {v:.1f}" for t, v in results["temperatures"].items()))

    # t-SNE of a training subset, the test slices and the samples
    rng = np.random.default_rng(args.seed)
    keep = torch.from_numpy(rng.choice(len(train), size=min(1000, len(train)), replace=False))
    sets = {"real train": train[keep], "real test": features["real test"],
            **{k: v for k, v in features.items() if k.startswith("samples")}}
    x = torch.cat(list(sets.values())).numpy()
    labels = [name for name, f in sets.items() for _ in range(len(f))]
    x = PCA(n_components=min(50, *x.shape), random_state=args.seed).fit_transform(x)
    points = TSNE(n_components=2, init="pca", random_state=args.seed,
                  perplexity=min(30.0, (len(x) - 1) / 3)).fit_transform(x)
    plot_embedding(points, labels, figure_path(args, "tsne"),
                   f"t-SNE of Inception features, {model_label(cfg)}")


# Failure autopsy.

def parse_setting(setting: str) -> str:
    """A --setting such as '1.1/0.5', '1.0/0.5' or '0.8' as its temperature key."""
    values = [float(v) for v in setting.split("/")]
    return temperature_key(dict(zip(("top", "bottom"), values)))


def autopsy(args, device):
    """
    Failure autopsy at one sampling setting (--setting): flawed samples beside
    their nearest training slice and the difference between them, so a
    distorted, misplaced or missing structure shows against real anatomy.
    The flawed samples are --indices, or by default the --n least like any
    training slice (lowest nearest-train SSIM), a fixed rule rather than a hand
    pick. A sheet of the 4 * --n lowest helps choose --indices instead.

    Settings sharing the top temperature share the top code maps, so the same
    sample at the other bottom temperatures shows whether a flaw comes from the
    top prior (it persists) or the bottom prior (it changes).
    """
    if not args.setting:
        raise SystemExit("autopsy needs --setting, e.g. --setting 1.1/0.5")
    cfg = load_checkpoint(args.run_dir)["config"]
    train = load_split(args, cfg, "train")
    samples = load_samples(args.run_dir)
    key = parse_setting(args.setting)
    if key not in samples:
        raise SystemExit(f"No samples at T={key}; saved: {', '.join(samples)}")
    s = samples[key]
    images = s["images"].float()
    nn_ssim, nn_idx = nearest_by_ssim(images, train.images.to(device), device)
    nn_idx = nn_idx.cpu()
    order = nn_ssim.argsort()  # least like any training slice first
    chosen = args.indices or order[:args.n].tolist()
    if not all(0 <= i < len(images) for i in chosen):
        raise SystemExit(f"--indices must be in [0, {len(images)})")

    # other bottom temperatures with this sample's top codes
    idx = torch.tensor(chosen)
    same_top = [k for k in samples if k != key and "top" in s
                and key_order(k)[0] == key_order(key)[0] and len(samples[k]["top"]) > max(chosen)
                and torch.equal(samples[k]["top"][idx], s["top"][idx])]
    persistence = {k: ssim(samples[k]["images"].float(), images) for k in same_top}

    neighbours = train.images[nn_idx[idx]]
    columns = {f"sample, T={key}": images[idx],
               **{f"same top codes, T={k}": samples[k]["images"][idx].float() for k in same_top},
               "nearest training slice": neighbours,
               "|sample - nearest|": (images[idx] - neighbours).abs()}
    records = [train.records[nn_idx[i]] for i in chosen]
    captions = [f"sample {i}\nnearest: case {r.case:03d} wk {r.week} sl {r.slice}\n"
                f"SSIM {nn_ssim[i]:.3f}" for i, r in zip(chosen, records)]
    plot_reconstructions(columns, captions, figure_path(args, f"autopsy_t{file_tag(key)}"))
    pool = order[:4 * args.n]
    plot_candidates(images[pool], [f"sample {i}, SSIM {nn_ssim[i]:.3f}" for i in pool.tolist()],
                    figure_path(args, f"autopsy_candidates_t{file_tag(key)}"))

    rows = [{"index": i, "nearest": os.path.basename(r.path), "nearest_ssim": float(nn_ssim[i]),
             "same_top_ssim": {k: float(p[i]) for k, p in persistence.items()}}
            for i, r in zip(chosen, records)]
    write_results(args.run_dir, f"autopsy_t{file_tag(key)}", {
        "model": model_label(cfg), "temperature": key, "n": len(images),
        "selection": "--indices" if args.indices else "lowest nearest-train SSIM",
        "typical_same_top_ssim": {k: float(p.median()) for k, p in persistence.items()},
        "samples": rows})

    print(f"T={key}, {len(images)} samples; nearest-train SSIM median {nn_ssim.median():.3f}")
    for row in rows:
        print(f"  sample {row['index']}: nearest {row['nearest']} (SSIM {row['nearest_ssim']:.3f})")
        if row["same_top_ssim"]:
            print("    SSIM to the same top codes at " + ", ".join(
                f"T={k}: {v:.2f} (usual {persistence[k].median():.2f})"
                for k, v in row["same_top_ssim"].items()))


# Latent interpolation.

def choose_pairs(records, n: int, rng) -> list[tuple[int, int]]:
    """
    n random pairs of test slices from different patients at the same slice
    index, or of any two different slices when there is only one patient.
    """
    by_slice = {}
    for i, r in enumerate(records):
        by_slice.setdefault(r.slice, []).append(i)
    usable = [s for s, idx in by_slice.items() if len({records[i].case for i in idx}) > 1]
    if not usable:
        return [tuple(int(i) for i in rng.choice(len(records), size=2, replace=False))
                for _ in range(n)]
    pairs = []
    for s in rng.choice(sorted(usable), size=min(n, len(usable)), replace=False):
        a = int(rng.choice(by_slice[s]))
        b = int(rng.choice([i for i in by_slice[s] if records[i].case != records[a].case]))
        pairs.append((a, b))
    return pairs


def interpolate(args, device):
    """
    Blend the encoder outputs of two test slices before quantisation at every
    level, then quantise and decode. Smoothness is the change between
    consecutive frames (1 - SSIM); a path that jumps has a few large steps, so
    its max/median step ratio is high.
    """
    vqvae, ckpt = load_vqvae(args.run_dir, device)
    cfg = ckpt["config"]
    test = load_split(args, cfg, "test")
    rng = np.random.default_rng(args.seed)
    alphas = torch.linspace(0, 1, args.steps)

    frames, changes, captions, pairs = [], [], [], []
    for a, b in choose_pairs(test.records, args.pairs, rng):
        f, codes = vqvae.interpolate(test[a][None].to(device), test[b][None].to(device), alphas)
        f = f.clamp(0, 1)
        step = (1 - ssim(f[1:], f[:-1])).cpu()
        code_step = {lv: (c[1:] != c[:-1]).flatten(1).cpu().double().mean(1)
                     for lv, c in codes.items()}
        ra, rb = test.records[a], test.records[b]
        caption = f"{ra.case:03d}/{rb.case:03d} sl {ra.slice}"
        frames.append(f.cpu())
        changes.append(step)
        captions.append(caption)
        pairs.append({"a": os.path.basename(ra.path), "b": os.path.basename(rb.path),
                      "step_change": step.tolist(),
                      "max_over_median_step": float(step.max() / step.median().clamp_min(1e-8)),
                      "code_change_per_step": {lv: c.tolist() for lv, c in code_step.items()}})

    ratios = torch.tensor([p["max_over_median_step"] for p in pairs])
    write_results(args.run_dir, "interpolate", {
        "model": model_label(cfg), "alphas": alphas.tolist(), "pairs": pairs,
        "max_over_median_step": summarise(ratios)})
    plot_interpolations(torch.stack(frames), alphas, captions, figure_path(args, "interpolation"))
    plot_step_changes(torch.stack(changes), captions, figure_path(args, "interpolation_steps"))
    for p in pairs:
        print(f"  {p['a']} -> {p['b']}: max/median step {p['max_over_median_step']:.2f}")


# Resources.

def components(run_dir: str):
    """(name, model directory, model, training batch size, best epoch) per trained model."""
    ckpt = load_checkpoint(run_dir)
    cfg = ckpt["config"]
    yield "stage-1 VQ-VAE", run_dir, build_vqvae(cfg), cfg["train"]["batch_size"], ckpt["epoch"]
    for level in ("top", "bottom") if cfg["vqvae"]["levels"] == 2 else ("bottom",):
        if os.path.exists(os.path.join(prior_dir(run_dir, level), "best.pt")):
            prior = load_checkpoint(prior_dir(run_dir, level))
            yield (f"{level} prior", prior_dir(run_dir, level),
                   build_prior(level, prior["config"]),
                   prior["config"]["train_prior"]["batch_size"], prior["epoch"])


def training_loss(model, batch_size: int, image_size: tuple[int, int], device):
    """The training loss on a random batch shaped like the real data."""
    if isinstance(model, VQVAE2):
        x = torch.rand(batch_size, 1, *image_size, device=device)
        out = model(x)
        return F.mse_loss(out["recon"], x) + sum(out["commitment"].values())
    codes = torch.randint(model.codebook_size, (batch_size, *model.shape), device=device)
    if isinstance(model, BottomPrior):
        top = torch.randint(model.codebook_size,
                            (batch_size, model.shape[0] // 2, model.shape[1] // 2), device=device)
        return F.cross_entropy(model(codes, top), codes)
    return F.cross_entropy(model(codes), codes)


def profile_training(model, batch_size: int, image_size: tuple[int, int], device,
                     steps: int) -> tuple[float, float]:
    """Peak GPU memory (GB) and median seconds per step over a few training steps."""
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(steps):
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        training_loss(model, batch_size, image_size, device).backward()
        optimizer.step()
        synchronize(device)
        times.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    model.cpu()
    return peak, float(np.median(times[1:] or times))


def resources(args, device):
    """
    Parameters, epochs and wall-clock training time (from history.csv), peak
    training VRAM (measured by profiling a few training steps on CUDA, since
    training did not record it), and sampling time per image (from the sample task).
    """
    for run_dir in args.run_dirs:
        cfg = load_checkpoint(run_dir)["config"]
        rows = []
        for name, model_dir, model, batch_size, best_epoch in components(run_dir):
            history = read_history(model_dir)
            row = {"component": name, "params": sum(p.numel() for p in model.parameters()),
                   "batch_size": batch_size, "epochs": len(history), "best_epoch": best_epoch,
                   "training_hours": sum(h["seconds"] for h in history) / 3600,
                   "peak_vram_gb": None, "seconds_per_step": None}
            if device.type == "cuda":
                row["peak_vram_gb"], row["seconds_per_step"] = profile_training(
                    model, batch_size, cfg["data"]["size"], device, args.profile_steps)
            rows.append(row)
            print(f"{model_label(cfg)} {name}: {row['params'] / 1e6:.2f}M params, "
                  f"{row['epochs']} epochs (best {row['best_epoch']}), "
                  f"{row['training_hours']:.2f} h, peak VRAM {row['peak_vram_gb']}")
        sampling = read_results(run_dir, "sample")
        write_results(run_dir, "resources", {
            "model": model_label(cfg),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
            "components": rows,
            "sampling_seconds_per_image": (
                float(np.mean([t["seconds_per_image"] for t in sampling["temperatures"].values()]))
                if sampling else None),
            "sampling_device": sampling["device"] if sampling else None})


# Report.

def report(args, device):
    """Markdown tables comparing runs, from their eval/*.json results."""
    lines = []

    def table(title, header, rows):
        if rows:
            lines.extend([f"### {title}", "", "| " + " | ".join(header) + " |",
                          "|" + "---|" * len(header), *("| " + " | ".join(r) + " |" for r in rows),
                          ""])

    results = {run: {task: read_results(run, task) for task in (*TASKS, "resources")}
               for run in args.run_dirs}

    rows, patient_rows = [], []
    for r in results.values():
        rec = r["reconstruct"]
        if not rec:
            continue
        o, cb = rec["overall"], rec["codebook"]
        rows.append([rec["model"], pm(o["full_ssim"]), pm(o["full_psnr"], 2),
                     pm(o["top_only_ssim"]) if "top_only_ssim" in o else "–",
                     f"{cb['top']['perplexity']:.0f}" if "top" in cb else "–",
                     f"{cb['bottom']['perplexity']:.0f}"])
        for patient, s in rec["per_patient"]["full_ssim"].items():
            patient_rows.append([rec["model"], patient, str(s["n"]), pm(s),
                                 pm(rec["per_patient"]["full_psnr"][patient], 2)])
    table("Reconstruction (test set, mean ± SD)",
          ["Model", "SSIM", "PSNR (dB)", "Top-only SSIM", "Perplexity top", "Perplexity bottom"],
          rows)
    table("Reconstruction per test patient",
          ["Model", "Patient", "Slices", "SSIM", "PSNR (dB)"], patient_rows)

    rows = []
    for r in results.values():
        aud, fd, smp = r["audit"], r["fid"], r["sample"]
        if not (aud or fd):
            continue
        model = (aud or fd)["model"]
        if aud:
            ref = aud["reference"]
            agree = " / ".join(f"{a['max']:.3f}" for a in ref["code_agreement"].values())
            rows.append([f"{model}, test slices (reference)", "–",
                         f"{fd['reference']['test_vs_train']:.1f}" if fd else "–",
                         f"{ref['nn_ssim']['median']:.3f} / {ref['nn_ssim']['max']:.3f}", "–",
                         agree, "–", "–"])
        for t in (aud or fd)["temperatures"]:
            a = aud["temperatures"][t] if aud else None
            rows.append([
                model, t, f"{fd['temperatures'][t]:.1f}" if fd else "–",
                f"{a['nn_ssim']['median']:.3f} / {a['nn_ssim']['max']:.3f}" if a else "–",
                f"{a['above_reference_max']}/{a['n']}" if a else "–",
                " / ".join(f"{c['max']:.3f}" for c in a["code_agreement"].values()) if a else "–",
                str(sum(c["exact_copies"] for c in a["code_agreement"].values())) if a else "–",
                f"{smp['temperatures'][t]['seconds_per_image']:.2f}"
                if smp and t in smp["temperatures"] else "–"])
    table("Generation and memorisation audit",
          ["Model", "T (top/bottom)", "FID ↓", "Nearest-train SSIM (median / max)",
           "Above reference max",
           "Max code agreement (top / bottom)", "Exact code copies", "Sampling s/image"], rows)

    rows = []
    for r in results.values():
        interp = r["interpolate"]
        if interp:
            rows.append([interp["model"], str(len(interp["pairs"])),
                         pm(interp["max_over_median_step"], 2),
                         f"{interp['max_over_median_step']['max']:.2f}"])
    table("Latent interpolation smoothness",
          ["Model", "Pairs", "Max/median step change", "Worst pair"], rows)

    rows, sampling_rows = [], []
    for r in results.values():
        res = r["resources"]
        if not res:
            continue
        for c in res["components"]:
            vram = f"{c['peak_vram_gb']:.1f}" if c["peak_vram_gb"] is not None else "–"
            rows.append([res["model"], c["component"], f"{c['params'] / 1e6:.2f}M",
                         str(c["batch_size"]), f"{c['epochs']} ({c['best_epoch']})",
                         f"{c['training_hours']:.2f}", vram])
        smp = r["sample"]
        if smp:
            per_image = np.mean([t["seconds_per_image"] for t in smp["temperatures"].values()])
            sampling_rows.append([res["model"], smp["device"], str(smp["batch_size"]),
                                  str(smp["n_samples"]), f"{per_image:.2f}"])
    profiled_on = next((r["resources"]["device"] for r in results.values() if r["resources"]), "")
    table(f"Training resources (peak VRAM profiled on {profiled_on})",
          ["Model", "Component", "Params", "Batch", "Epochs (best)", "Training time (h)",
           "Peak VRAM (GB)"], rows)
    table("Sampling time (top prior, bottom prior and decoder; mean over temperatures)",
          ["Model", "Device", "Batch", "Samples per temperature", "Seconds per image"],
          sampling_rows)

    text = "\n".join(lines)
    print(text)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(text)


def run_all(args, device):
    for task in TASKS:
        print(f"=== {task}")
        globals()[task](args, device)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained VQ-VAE-2 runs.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=os.path.join(PROJECT_DIR, "assets"),
                        help="where figures (and report's results.md) are saved")
    single = argparse.ArgumentParser(add_help=False, parents=[common])
    single.add_argument("run_dir", help="stage-1 run directory, e.g. checkpoints/vqvae2_L2_ssim0")
    single.add_argument("--root", help="data root (default: data.root in the run's config)")
    single.add_argument("--n", type=int, default=6, help="slices per figure row")
    single.add_argument("--seed", type=int, default=0,
                        help="choice of slices for figures, interpolation pairs and t-SNE")
    single.add_argument("--temperatures", type=float, nargs="+",
                        help="sampling temperatures shared by both priors, or the top prior's "
                             "with --bottom-temperatures (default: sample.temperatures in the config)")
    single.add_argument("--bottom-temperatures", type=float, nargs="+",
                        help="bottom-prior temperatures: sample every (top, bottom) pair of "
                             "--temperatures and these, e.g. --temperatures 0.7 0.8 0.9 "
                             "--bottom-temperatures 0.7 0.8 0.9")
    single.add_argument("--n-samples", type=int,
                        help="samples per temperature (default: sample.n_samples in the config)")
    single.add_argument("--overwrite", action="store_true",
                        help="resample temperatures that are already saved (sample)")
    single.add_argument("--setting",
                        help="sampling setting to autopsy: a shared temperature or top/bottom, "
                             "e.g. 1.1/0.5")
    single.add_argument("--indices", type=int, nargs="+",
                        help="samples to autopsy (default: the --n least like any training slice)")
    single.add_argument("--pairs", type=int, default=4, help="interpolation pairs")
    single.add_argument("--steps", type=int, default=9, help="frames per interpolation")
    multi = argparse.ArgumentParser(add_help=False, parents=[common])
    multi.add_argument("run_dirs", nargs="+", help="stage-1 run directories")
    multi.add_argument("--profile-steps", type=int, default=5,
                       help="training steps profiled per model for peak VRAM (resources)")

    sub = p.add_subparsers(dest="task", required=True)
    for task in (*TASKS, "all", "autopsy"):
        sub.add_parser(task, parents=[single])
    for task in ("resources", "report"):
        sub.add_parser(task, parents=[multi])
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    task = {"all": run_all, "resources": resources, "report": report}.get(args.task)
    (task or globals()[args.task])(args, device)


if __name__ == "__main__":
    main()

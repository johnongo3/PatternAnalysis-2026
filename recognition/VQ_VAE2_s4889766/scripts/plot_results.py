"""
README comparison plots from a results.md written by predict.py report, read
from its reconstruction tables and its "Generation and memorisation audit" table:

    reconstruction.png        test-set SSIM and PSNR per model, overall and per patient
    fid_shared.png            FID against the shared sampling temperature, per model
    memorisation_shared.png   the nearest-neighbour audit against the same temperatures
    tradeoff.png              FID against flagged samples for every sampling setting

Usage: python scripts/plot_results.py RESULTS_MD [--out assets/results]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Short name and fixed colour per model, so a model keeps its colour in every
# plot. The three hues stay distinct under colour-vision deficiency.
MODELS = {"Baseline (1 level)": ("Baseline (1 level)", "#2a78d6"),
          "VQ-VAE-2 (2 levels)": ("VQ-VAE-2", "#eb6834"),
          "VQ-VAE-2 (2 levels), SSIM loss 0.01": ("VQ-VAE-2 + SSIM loss", "#1baf7a")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REFERENCE_SUFFIX = ", test slices (reference)"


def number(text: str) -> float | None:
    return None if text in ("–", "-", "") else float(text)


def mean_sd(text: str) -> tuple[float, float]:
    mean, sd = (float(v) for v in text.split(" ± "))
    return mean, sd


def read_table(path: str, title: str) -> list[list[str]]:
    """Body rows of the table under the first "### " heading starting with title."""
    with open(path) as f:
        sections = f.read().split("### ")
    table = next(s for s in sections if s.startswith(title))
    return [[c.strip() for c in line.strip().strip("|").split("|")]
            for line in table.splitlines() if line.startswith("| ")][1:]  # skip the header


def read_reconstruction(path: str) -> list[dict]:
    """
    Test-set SSIM and PSNR (mean, SD over slices) per model, over every test
    slice (group "all") and per test patient.
    """
    rows = [{"model": model, "group": "all", "n": None, "ssim": mean_sd(s), "psnr": mean_sd(p)}
            for model, s, p, *_ in read_table(path, "Reconstruction (test set")]
    rows += [{"model": model, "group": patient, "n": int(n), "ssim": mean_sd(s),
              "psnr": mean_sd(p)}
             for model, patient, n, s, p in read_table(path, "Reconstruction per test patient")]
    return rows


def read_audit(path: str) -> tuple[list[dict], dict[str, dict]]:
    """
    The audit table's sample rows, one per (model, temperature key), and each
    model's reference row (its decoder's reconstructions of the test slices).
    """
    samples, references = [], {}
    for model, key, fid, nn, flagged, *_ in read_table(path, "Generation and memorisation audit"):
        median, maximum = (float(v) for v in nn.split(" / "))
        if model.endswith(REFERENCE_SUFFIX):
            references[model.removesuffix(REFERENCE_SUFFIX)] = {
                "fid": number(fid), "nn_median": median, "nn_max": maximum}
            continue
        count, n = (int(v) for v in flagged.split("/"))
        top, bottom = (float(v) for v in key.split("/")) if "/" in key else (float(key),) * 2
        samples.append({"model": model, "key": key, "shared": "/" not in key, "top": top,
                        "bottom": bottom, "fid": number(fid), "nn_median": median,
                        "flagged": count, "n": n})
    return samples, references


def shared_rows(samples: list[dict], model: str, temperatures=None) -> list[dict]:
    rows = [s for s in samples if s["model"] == model and s["shared"]
            and (temperatures is None or s["top"] in temperatures)]
    return sorted(rows, key=lambda s: s["top"])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_reconstruction(rows, path):
    """
    SSIM and PSNR side by side, each model a dot with a ±1 SD bar over the test
    slices, first over every test slice and then per patient.
    """
    slices = {r["group"]: r["n"] for r in rows if r["group"] != "all"}
    slices["all"] = sum(slices.values())
    groups = ["all", *sorted(slices.keys() - {"all"})]
    models = [m for m in MODELS if any(r["model"] == m for r in rows)]
    width = 0.22  # horizontal offset between models within a group

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    for ax, field, label in zip(axes, ("ssim", "psnr"), ("SSIM to the original", "PSNR (dB)")):
        for k, model in enumerate(models):
            name, colour = MODELS[model]
            stats = {r["group"]: r[field] for r in rows if r["model"] == model}
            x = [g + (k - (len(models) - 1) / 2) * width for g in range(len(groups))]
            ax.errorbar(x, [stats[g][0] for g in groups], yerr=[stats[g][1] for g in groups],
                        fmt="o", color=colour, markersize=6, elinewidth=1.5, capsize=0,
                        label=name)
        ax.axvline(0.5, color=GRID, linewidth=1)  # all slices | per patient
        ax.set_xticks(range(len(groups)),
                      [f"{'all test slices' if g == 'all' else 'patient ' + g}\n{slices[g]} slices"
                       for g in groups], fontsize=8)
        ax.set_ylabel(f"{label} (higher is better)")
        style(ax)
    axes[0].set_title("Structural similarity", fontsize=10, loc="left")
    axes[1].set_title("Peak signal-to-noise ratio", fontsize=10, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(models), fontsize=8,
               frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fid_shared(samples, references, temperatures, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    for model, (name, colour) in MODELS.items():
        rows = shared_rows(samples, model, temperatures)
        if not rows:
            continue
        ax.plot([r["top"] for r in rows], [r["fid"] for r in rows], color=colour, linewidth=2,
                marker="o", markersize=6, label=name)
        best = min(rows, key=lambda r: r["fid"])  # label only each model's best
        ax.annotate(f"{best['fid']:.1f}", (best["top"], best["fid"]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8, color=INK)
    floor = next(r["fid"] for r in references.values() if r["fid"] is not None)
    ax.axhline(floor, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"real test slices, {floor:.1f}", (temperatures[0], floor), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xticks(temperatures)
    ax.set_xlim(temperatures[0] - 0.06, temperatures[-1] + 0.03)
    ax.set_ylim(0, None)
    ax.set_xlabel("shared sampling temperature $T$")
    ax.set_ylabel("FID against training slices (lower is better)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    style(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_memorisation_shared(samples, references, temperatures, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True,
                                      layout="constrained")
    medians = [r["nn_median"] for r in references.values()]
    left.axhspan(min(medians) - 0.001, max(medians) + 0.001, color=GRID, linewidth=0)
    left.annotate("test patients (reference)", (temperatures[-1], max(medians)), xytext=(0, 4),
                  textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    for model, (name, colour) in MODELS.items():
        rows = shared_rows(samples, model, temperatures)
        if not rows:
            continue
        for ax, field in ((left, "nn_median"), (right, "flagged")):
            ax.plot([r["top"] for r in rows], [r[field] for r in rows], color=colour,
                    linewidth=2, marker="o", markersize=6, label=name)
    left.set_ylabel("median SSIM to nearest training slice")
    left.set_title("Closeness of the typical sample", fontsize=10, loc="left")
    n = samples[0]["n"]
    right.set_ylabel(f"samples above the reference max (of {n})")
    right.set_title("Samples closer to training than any test patient", fontsize=10,
                    loc="left")
    right.set_ylim(0, None)
    left.legend(fontsize=8, frameon=False)
    for ax in (left, right):
        ax.set_xticks(temperatures)
        ax.set_xlim(temperatures[0] - 0.03, temperatures[-1] + 0.03)
        ax.set_xlabel("shared sampling temperature $T$")
        style(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(samples, references, labelled, max_fid, path):
    """
    FID against flagged samples. Shared-temperature points are joined in
    temperature order; points past max_fid are noted at the right edge.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
    handles = []
    for model, (name, colour) in MODELS.items():
        rows = shared_rows(samples, model)
        if not rows:
            continue
        ax.plot([r["fid"] for r in rows], [r["flagged"] for r in rows], color=colour,
                linewidth=1.5, marker="o", markersize=6, alpha=0.9)
        below = model.startswith("Baseline")  # its points sit on the x axis among others
        for r in rows:
            if r["fid"] > max_fid:
                ax.annotate(f"$T$ = {r['key']}: FID {r['fid']:.0f} →", (max_fid, r["flagged"]),
                            xytext=(-2, 22), textcoords="offset points", ha="right",
                            fontsize=7, color=MUTED)
                continue
            ax.annotate(f"{r['key']}", (r["fid"], r["flagged"]),
                        xytext=(0, -11) if below else (5, 3), ha="center" if below else "left",
                        textcoords="offset points", fontsize=7, color=MUTED)
        handles.append(Line2D([], [], color=colour, marker="o", linewidth=1.5,
                              label=f"{name}, shared $T$"))
        separate = [s for s in samples if s["model"] == model and not s["shared"]]
        if separate:
            ax.scatter([s["fid"] for s in separate], [s["flagged"] for s in separate],
                       marker="s", s=36, facecolors="white", edgecolors=colour, linewidths=1.5,
                       zorder=3)
            for s in separate:
                if s["key"] not in labelled:
                    continue
                # left of the point, unless another point sits there
                crowded = any(0 < s["fid"] - o["fid"] < 15 and abs(s["flagged"] - o["flagged"]) < 5
                              for o in separate)
                ax.annotate(f"{s['key']}", (s["fid"], s["flagged"]),
                            xytext=(6, 2) if crowded else (-6, 2), ha="left" if crowded else "right",
                            textcoords="offset points", fontsize=8, color=INK, fontweight="bold")
            handles.append(Line2D([], [], color=colour, marker="s", markerfacecolor="white",
                                  linewidth=0, label=f"{name}, separate $T$ (top/bottom)"))
    floor = next(r["fid"] for r in references.values() if r["fid"] is not None)
    ax.axvline(floor, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"real test slices\n{floor:.1f}", (floor, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points", va="top", fontsize=8, color=MUTED)
    ax.set_xlabel("FID against training slices (realism, lower is better)")
    ax.set_ylabel(f"samples closer than any test patient (of {samples[0]['n']})")
    ax.set_xlim(floor - 5, max_fid)
    ax.set_ylim(-9, None)
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="center right")
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", help="results.md from predict.py report")
    parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "assets", "results"))
    parser.add_argument("--shared", type=float, nargs="+", default=[0.8, 0.9, 1.0],
                        help="shared temperatures in the per-temperature plots")
    parser.add_argument("--label", nargs="+", default=["0.9/0.7", "1/0.5", "1.1/0.5"],
                        help="separate-temperature settings named in the trade-off plot")
    parser.add_argument("--max-fid", type=float, default=200,
                        help="right edge of the trade-off plot; points past it are noted")
    args = parser.parse_args()

    samples, references = read_audit(args.results)
    os.makedirs(args.out, exist_ok=True)
    plot_reconstruction(read_reconstruction(args.results),
                        os.path.join(args.out, "reconstruction.png"))
    plot_fid_shared(samples, references, args.shared, os.path.join(args.out, "fid_shared.png"))
    plot_memorisation_shared(samples, references, args.shared,
                             os.path.join(args.out, "memorisation_shared.png"))
    plot_tradeoff(samples, references, args.label, args.max_fid,
                  os.path.join(args.out, "tradeoff.png"))
    print(f"Saved plots to {args.out}")


if __name__ == "__main__":
    main()

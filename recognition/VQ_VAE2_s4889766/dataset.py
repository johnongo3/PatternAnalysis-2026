"""
Data loading for the HipMRI 2D prostate slices.

Expects the keras_slices_data layout, where each split lives in its own folder:

    <root>/keras_slices_train/case_004_week_0_slice_0.nii.gz
    <root>/keras_slices_validate/...
    <root>/keras_slices_test/...

Each slice is loaded with nibabel, has any stray third dimension dropped, is
center-cropped (or zero-padded) to a fixed size, then normalised per slice by
clipping at a high percentile and scaling to [0, 1].

Preprocessed splits are cached as one tensor file each (outside the repo, under
HIPMRI_CACHE or ~/.cache/hipmri_vqvae2) so later runs skip the per-file decode.

The slices are coronal. Arrays are stored as (256, 128) with superior pointing
along +columns; use to_display() to rotate a slice upright for plotting.
"""
import os
import re
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SPLITS = ("train", "validate", "test")
FILENAME_PATTERN = re.compile(r"^case_(\d+)_week_(\d+)_slice_(\d+)\.nii(?:\.gz)?$")
DEFAULT_CACHE_DIR = os.environ.get("HIPMRI_CACHE",
                                   os.path.expanduser("~/.cache/hipmri_vqvae2"))


@dataclass(frozen=True)
class SliceRecord:
    path: str
    case: int
    week: int
    slice: int


def parse_filename(path: str) -> SliceRecord:
    """Parse case (patient) ID, week and slice index from a slice filename."""
    match = FILENAME_PATTERN.match(os.path.basename(path))
    if match is None:
        raise ValueError(f"Unexpected slice filename: {path}")
    case, week, slice_idx = (int(g) for g in match.groups())
    return SliceRecord(path, case, week, slice_idx)


def load_slice(path: str) -> np.ndarray:
    """Load a single 2D slice as float32, removing the extra dim some files have."""
    image = nib.load(path).get_fdata(caching="unchanged")
    if image.ndim == 3:
        image = image[:, :, 0] # sometimes extra dims in HipMRI_study data
    return image.astype(np.float32)


def fit_to_size(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Center-crop or zero-pad an image to size.

    One training volume (case 019, week 1) is 256 x 144 rather than 256 x 128.
    Its pixel spacing matches the rest of the data, so cropping keeps the
    anatomy at the same scale where resizing would squash it along one axis.
    """
    out = np.zeros(size, dtype=image.dtype)
    src, dst = [], []
    for have, want in zip(image.shape, size):
        if have >= want:
            start = (have - want) // 2
            src.append(slice(start, start + want))
            dst.append(slice(0, want))
        else:
            start = (want - have) // 2
            src.append(slice(0, have))
            dst.append(slice(start, start + have))
    out[tuple(dst)] = image[tuple(src)]
    return out


def normalise(image: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """Clip a slice at its clip_percentile intensity and min-max scale to [0, 1]."""
    high = np.percentile(image, clip_percentile)
    image = np.clip(image, None, high)
    low = image.min()
    if high - low < 1e-8: # blank slice
        return np.zeros_like(image)
    return (image - low) / (high - low)


def to_display(image):
    """Rotate a (H, W) or (1, H, W) slice so superior is up, for plotting only."""
    image = np.asarray(image).squeeze()
    return np.rot90(image)


# Segmentation labels of the HipMRI masks.
MASK_LABELS = {0: "background", 1: "body", 2: "bone", 3: "bladder", 4: "rectum", 5: "prostate"}


def mask_path(record: SliceRecord) -> str:
    """
    A slice's segmentation mask, in the matching keras_slices_seg_<split> folder:
    keras_slices_train/case_004_week_0_slice_0.nii.gz has its mask at
    keras_slices_seg_train/seg_case_004_week_0_slice_0.nii.gz.
    """
    folder, name = os.path.split(record.path)
    seg_folder = os.path.join(os.path.dirname(folder), os.path.basename(folder).replace(
        "keras_slices_", "keras_slices_seg_"))
    for candidate in (f"seg_{name}", name):
        path = os.path.join(seg_folder, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No mask for {name} in {seg_folder}")


def load_mask(record: SliceRecord, size=(256, 128)) -> np.ndarray:
    """A slice's mask as integer labels (MASK_LABELS), cropped or padded like its image."""
    return fit_to_size(np.rint(load_slice(mask_path(record))).astype(np.int64), tuple(size))


class HipMRISlices(Dataset):
    """
    One split of the HipMRI 2D slices, returning (1, H, W) float tensors in [0, 1].

    root: folder containing the keras_slices_{train, validate, test} folders
    split: one of "train", "validate", "test"
    size: (rows, cols) every slice is cropped or padded to
    clip_percentile: per-slice upper intensity percentile to clip at
    preload: load and preprocess every slice into memory up front, rather
        than reading from disk in __getitem__ (train is ~1.5 GB as float32)
    cache_dir: where preloaded splits are cached as one tensor file each, or
        None to disable caching. Only used when preload is True.
    """

    def __init__(self, root: str, split: str, size=(256, 128),
                 clip_percentile: float = 99.5, preload: bool = True,
                 cache_dir: str | None = DEFAULT_CACHE_DIR):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.size = tuple(size)
        self.clip_percentile = clip_percentile

        folder = os.path.join(root, f"keras_slices_{split}")
        names = [n for n in os.listdir(folder) if n.endswith((".nii", ".nii.gz"))]
        if not names:
            raise FileNotFoundError(f"No NIfTI slices found in {folder}")
        self.records = sorted((parse_filename(os.path.join(folder, n)) for n in names),
                              key=lambda r: (r.case, r.week, r.slice))

        self.images = None
        if preload and cache_dir is not None:
            self.images = self._load_cached(cache_dir)
        elif preload:
            self.images = self._load_all()

    @property
    def cases(self) -> set[int]:
        return {r.case for r in self.records}

    def _load_all(self) -> torch.Tensor:
        return torch.stack([self._load(r.path)
                            for r in tqdm(self.records, desc=f"Loading {self.split}")])

    def _load_cached(self, cache_dir: str) -> torch.Tensor:
        """
        Load this split from its cache file, building it on first use.

        The file name encodes size and clip_percentile, and the cached filenames
        are checked against the folder, so changing either rebuilds the cache.
        """
        rows, cols = self.size
        cache_path = os.path.join(
            cache_dir, f"{self.split}_{rows}x{cols}_p{self.clip_percentile}.pt")
        names = [os.path.basename(r.path) for r in self.records]

        if os.path.exists(cache_path):
            cached = torch.load(cache_path, mmap=True, weights_only=True)
            if cached["names"] == names:
                return cached["images"]
            print(f"Cache {cache_path} does not match {self.split} files, rebuilding")

        images = self._load_all()
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = cache_path + ".tmp"
        torch.save({"images": images, "names": names}, tmp_path)
        os.replace(tmp_path, cache_path) # no half-written cache if interrupted
        return images

    def _load(self, path: str) -> torch.Tensor:
        image = fit_to_size(load_slice(path), self.size)
        image = normalise(image, self.clip_percentile)
        return torch.from_numpy(image).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.images is not None:
            return self.images[idx]
        return self._load(self.records[idx].path)


def check_no_patient_leakage(datasets: dict[str, HipMRISlices]) -> None:
    """Assert that no case (patient) ID appears in more than one split."""
    names = list(datasets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = datasets[a].cases & datasets[b].cases
            assert not shared, f"Patients {sorted(shared)} appear in both {a} and {b}"


def build_datasets(root: str, size=(256, 128), clip_percentile: float = 99.5,
                   preload: bool = True,
                   cache_dir: str | None = DEFAULT_CACHE_DIR) -> dict[str, HipMRISlices]:
    """Build the train, validate and test splits and check they share no patients."""
    datasets = {split: HipMRISlices(root, split, size, clip_percentile, preload, cache_dir)
                for split in SPLITS}
    check_no_patient_leakage(datasets)
    return datasets


@torch.no_grad()
def encode_dataset(model, dataset: Dataset, device: torch.device,
                   batch_size: int = 64) -> dict[str, torch.Tensor]:
    """
    Encode every slice with a frozen VQ-VAE-2, in dataset order. Returns the
    code map of each level (e.g. "top": (N, 32, 16), "bottom": (N, 64, 32)) as
    int16, which holds any codebook of up to 32768 codes.
    """
    model.eval()
    codes = {level: [] for level in model.level_names}
    for x in tqdm(DataLoader(dataset, batch_size=batch_size), desc="Encoding", disable=None):
        for level, out in model.encode(x.to(device)).items():
            codes[level].append(out.codes.to(torch.int16).cpu())
    return {level: torch.cat(chunks) for level, chunks in codes.items()}


class CodeDataset(Dataset):
    """
    Saved code maps of one split, as written by train.py --stage encode.
    Returns a dict of code maps per level as long tensors: "top" and "bottom"
    for a two-level model, only "bottom" for the single-level baseline.
    vqvae_epoch records which stage-1 checkpoint made the codes.
    """

    def __init__(self, path: str):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.codes = {level: saved[level] for level in ("top", "bottom") if level in saved}
        self.names, self.vqvae_epoch = saved["names"], saved["vqvae_epoch"]

    def __len__(self) -> int:
        return len(self.codes["bottom"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {level: codes[idx].long() for level, codes in self.codes.items()}


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/keras_slices_data"
    datasets = build_datasets(root)
    for split, ds in datasets.items():
        x = ds.images
        print(f"{split:>8}: {len(ds):>5} slices, {len(ds.cases):>2} patients, "
              f"shape {tuple(x.shape[1:])}, range [{x.min():.3f}, {x.max():.3f}]")
    print("No patient appears in more than one split.")

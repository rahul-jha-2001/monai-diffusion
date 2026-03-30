"""
Brain MRI 2D Slice Dataset
--------------------------
Loads 3D volumetric T1 NIfTI files, preprocesses them, and serves
individual 2D axial slices for diffusion model training.

Pipeline per volume:
  1. Classify & filter  — keep 3D volumetric, drop thick-slice / empty / outliers
  2. Crop               — trim to brain z-extent (first/last slice with fill > 1%) ± margin
  3. Resample in-plane  — to 1.0 × 1.0 mm using MONAI Spacing
  4. Resize             — pad or crop in-plane to TARGET_SIZE × TARGET_SIZE
  5. Normalise          — clip at 99.5th percentile within brain, min-max → [0, 1]
  6. Slice extraction   — keep axial slices with fill > MIN_SLICE_FILL
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Spacing,
    ResizeWithPadOrCrop,
    ScaleIntensityRangePercentiles,
    ToTensor,
)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent / "brain_only"
TARGET_SIZE    = 256          # output in-plane size (pixels)
TARGET_SPACING = 1.0          # target in-plane resolution (mm)
MIN_SLICE_FILL = 0.05         # slice must have >5% non-zero voxels to be usable
MAX_FILL_MIN   = 0.15         # volume quality: peak fill must exceed 15%
MIN_USABLE_R   = 0.10         # volume quality: usable/total ratio must exceed 10%
CROP_MARGIN    = 3            # extra slices added above/below brain extent
TRAIN_FRAC     = 0.80
VAL_FRAC       = 0.10
# TEST_FRAC    = 0.10  (remainder)


# ── Acquisition classifier ───────────────────────────────────────────────────

def classify_acquisition(nifti_path: Path) -> str:
    """Return 'thick' or '3d' using a 2-of-3 voting classifier."""
    img = nib.load(nifti_path)
    dx, dy, dz = img.header.get_zooms()[:3]
    D = img.shape[2]
    thick_slice = dz > 2.5
    anisotropic = (dz / ((dx + dy) / 2)) > 2.0
    few_slices  = D < 60
    return "thick" if sum([thick_slice, anisotropic, few_slices]) >= 2 else "3d"


# ── Subject filtering (Step 1) ───────────────────────────────────────────────

def get_subject_list(base_dir: Path = BASE_DIR) -> list[Path]:
    """
    Scan base_dir, apply all quality filters, and return paths to T1 files
    for clean 3D subjects only.

    Filters applied:
      - Must have t1_brain.nii.gz
      - classify_acquisition → must be '3d'
      - max fill across all slices > MAX_FILL_MIN  (drops partial brains)
      - usable / total ratio   > MIN_USABLE_R      (drops oversized FOV)
      - at least one usable slice                  (drops empty volumes)
    """
    kept, dropped = [], []

    for subj_dir in sorted(base_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        t1_path = subj_dir / "t1_brain.nii.gz"
        if not t1_path.exists():
            dropped.append((subj_dir.name, "no_t1"))
            continue

        if classify_acquisition(t1_path) == "thick":
            dropped.append((subj_dir.name, "thick_slice"))
            continue

        nii  = nib.load(t1_path)
        H, W, D = nii.shape
        data = nii.get_fdata(dtype=np.float32)
        fills = np.array(
            [np.count_nonzero(data[..., z]) / (H * W) for z in range(D)]
        )

        usable       = int((fills >= 0.01).sum())   # 1% threshold for counting
        usable_ratio = usable / D
        max_fill     = float(fills.max())

        if usable == 0:
            dropped.append((subj_dir.name, "empty_volume"))
        elif max_fill < MAX_FILL_MIN:
            dropped.append((subj_dir.name, "low_peak_fill"))
        elif usable_ratio < MIN_USABLE_R:
            dropped.append((subj_dir.name, "low_usable_ratio"))
        else:
            kept.append(t1_path)

    print(f"Subjects kept : {len(kept)}")
    print(f"Subjects dropped: {len(dropped)}")
    return kept


# ── Train / Val / Test split (Step 4) ────────────────────────────────────────

def make_splits(
    subject_paths: list[Path],
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Subject-level random split: 80 / 10 / 10.
    Split is done at subject level to prevent data leakage between splits.
    """
    paths = sorted(subject_paths)   # deterministic order before shuffle
    rng   = random.Random(seed)
    rng.shuffle(paths)

    n       = len(paths)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train = paths[:n_train]
    val   = paths[n_train : n_train + n_val]
    test  = paths[n_train + n_val :]

    print(f"Split — train: {len(train)}  val: {len(val)}  test: {len(test)}")
    return train, val, test


# ── Per-volume preprocessing (Steps 2–3) ─────────────────────────────────────

def _build_transforms(pixdim_z: float) -> Compose:
    """
    Build MONAI transform pipeline for a single volume.
    Resamples in-plane to TARGET_SPACING mm, pads/crops to TARGET_SIZE.
    pixdim_z is passed through unchanged (irrelevant for 2D slice extraction).
    """
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),   # (H,W,D) → (1,H,W,D)
        Spacing(
            pixdim=(TARGET_SPACING, TARGET_SPACING, pixdim_z),
            mode="bilinear",
        ),
        ResizeWithPadOrCrop(
            spatial_size=(TARGET_SIZE, TARGET_SIZE, -1),  # -1 = keep D unchanged
            mode="constant",
            constant_values=0,
        ),
    ])


def _crop_to_brain(data: np.ndarray, margin: int = CROP_MARGIN) -> np.ndarray:
    """
    Crop volume along z to the brain extent.
    data shape: (H, W, D). Returns cropped array.
    """
    H, W, D = data.shape
    fills   = np.array([np.count_nonzero(data[..., z]) / (H * W) for z in range(D)])
    usable  = np.where(fills >= 0.01)[0]
    if len(usable) == 0:
        return data
    z_start = max(0,   usable[0]  - margin)
    z_end   = min(D,   usable[-1] + margin + 1)
    return data[..., z_start:z_end]


def _normalise_slice(
    slc: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """
    Clip intensities to [lower, upper] percentile, then min-max scale to [0, 1].
    Computed on non-zero voxels only (within brain).
    """
    nonzero = slc[slc > 0]
    if len(nonzero) == 0:
        return slc.astype(np.float32)
    lo = np.percentile(nonzero, lower_percentile)
    hi = np.percentile(nonzero, upper_percentile)
    slc = np.clip(slc, lo, hi)
    slc = (slc - lo) / (hi - lo + 1e-8)
    return slc.astype(np.float32)


def preprocess_volume(t1_path: Path) -> list[np.ndarray]:
    """
    Full preprocessing pipeline for one subject.
    Returns list of 2D numpy arrays, each shape (TARGET_SIZE, TARGET_SIZE),
    normalised to [0, 1], with fill > MIN_SLICE_FILL.
    """
    nii      = nib.load(t1_path)
    data     = nii.get_fdata(dtype=np.float32)          # (H, W, D)
    pixdim_z = float(nii.header.get_zooms()[2])

    # Step 2a — crop to brain z-extent
    data = _crop_to_brain(data)

    # Step 2b — resample in-plane + resize to TARGET_SIZE
    tfm  = _build_transforms(pixdim_z)
    vol  = tfm(data)                                    # (1, TARGET_SIZE, TARGET_SIZE, D')
    vol  = vol.numpy()[0]                               # (TARGET_SIZE, TARGET_SIZE, D')

    # Step 3 — extract 2D slices, filter by fill
    H, W, D = vol.shape
    slices  = []
    for z in range(D):
        slc  = vol[..., z]
        fill = np.count_nonzero(slc) / (H * W)
        if fill < MIN_SLICE_FILL:
            continue
        slc = _normalise_slice(slc)
        slices.append(slc)

    return slices


# ── PyTorch Dataset (Step 5) ─────────────────────────────────────────────────

class BrainSliceDataset(Dataset):
    """
    Lazily loads 3D T1 volumes and serves individual 2D axial slices.

    Each item is a dict:
        {
            "slice":       torch.Tensor  shape (1, 256, 256), float32 in [0, 1]
            "subject_id":  str
            "slice_idx":   int
        }

    Args:
        subject_paths: list of Path objects pointing to t1_brain.nii.gz files
        augment:       if True, apply random horizontal flip (training only)
    """

    def __init__(
        self,
        subject_paths: list[Path],
        augment: bool = False,
    ) -> None:
        self.augment = augment
        self._index: list[tuple[str, int]] = []   # (subject_id, slice_idx)
        self._cache: dict[str, list[np.ndarray]] = {}
        self._paths: dict[str, Path] = {}

        for p in subject_paths:
            uid = p.parent.name
            self._paths[uid] = p

        self._build_index(subject_paths)

    def _build_index(self, subject_paths: list[Path]) -> None:
        """Pre-compute (subject_id, slice_idx) pairs for all usable slices."""
        print(f"Building slice index for {len(subject_paths)} subjects...")
        for p in subject_paths:
            uid    = p.parent.name
            slices = preprocess_volume(p)
            self._cache[uid] = slices
            for i in range(len(slices)):
                self._index.append((uid, i))
        print(f"Total usable slices: {len(self._index):,}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        uid, slice_idx = self._index[idx]
        slc = self._cache[uid][slice_idx].copy()   # (H, W)

        if self.augment and random.random() > 0.5:
            slc = np.fliplr(slc).copy()

        tensor = torch.from_numpy(slc).unsqueeze(0)  # (1, H, W)

        return {
            "slice":      tensor,
            "subject_id": uid,
            "slice_idx":  slice_idx,
        }


# ── Convenience builder ──────────────────────────────────────────────────────

def build_datasets(
    base_dir: Path = BASE_DIR,
    seed: int = 42,
) -> tuple[BrainSliceDataset, BrainSliceDataset, BrainSliceDataset]:
    """
    End-to-end: filter subjects → split → build train/val/test datasets.

    Returns:
        train_ds, val_ds, test_ds
    """
    subjects          = get_subject_list(base_dir)
    train_p, val_p, test_p = make_splits(subjects, seed=seed)

    train_ds = BrainSliceDataset(train_p, augment=True)
    val_ds   = BrainSliceDataset(val_p,   augment=False)
    test_ds  = BrainSliceDataset(test_p,  augment=False)

    return train_ds, val_ds, test_ds


# ── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== Running smoke test on 3 subjects ===\n")

    subjects = get_subject_list()
    if len(subjects) < 3:
        print("Not enough subjects found.")
        sys.exit(1)

    rng      = random.Random(42)
    sample   = rng.sample(subjects, 3)
    train_ds = BrainSliceDataset(sample, augment=False)

    print(f"\nDataset size: {len(train_ds)} slices from 3 subjects")

    item = train_ds[0]
    print(f"\nSample item:")
    print(f"  slice shape : {item['slice'].shape}")
    print(f"  dtype       : {item['slice'].dtype}")
    print(f"  value range : [{item['slice'].min():.3f}, {item['slice'].max():.3f}]")
    print(f"  subject_id  : {item['subject_id']}")
    print(f"  slice_idx   : {item['slice_idx']}")

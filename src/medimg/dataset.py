"""Torch datasets that build phantom patches on demand.

The dataset is generated, not stored: a spec (an integer seed) deterministically
produces a volume, a lesion slice and a set of patches.  That keeps the repo
free of binary fixtures while still giving the training loop real, varying data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.medimg.augment import AugmentConfig, DEFAULT_AUGMENT, augment_image
from src.medimg.patching import Patch, PatchSampler, patch_array
from src.medimg.phantom import PhantomConfig, brain_phantom, lesion_slice_index
from src.medimg.windowing import apply_preset, normalize_array


@dataclass(frozen=True)
class SyntheticSpec:
    """One dataset item: which phantom to build and how to sample from it."""

    index: int
    seed: int
    positive_fraction: float = 0.5
    axis: int = 2

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be non-negative")
        if not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be in [0, 1]")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")


def build_specs(
    n: int,
    *,
    seed: int = 0,
    positive_fraction: float = 0.5,
) -> list[SyntheticSpec]:
    """Create ``n`` specs whose seeds are derived from ``seed`` deterministically."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n)
    return [
        SyntheticSpec(index=i, seed=int(s), positive_fraction=positive_fraction)
        for i, s in enumerate(seeds)
    ]


def slice_for_spec(
    spec: SyntheticSpec,
    cfg: PhantomConfig,
    *,
    window: str = "brain",
    normalize: str = "minmax",
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed lesion slice and its mask for one spec.

    The slice is taken through the lesion centre, so every generated image has
    a visible lesion; the negative class comes from the sampler, not from
    lesion-free slices.
    """
    phantom_cfg = PhantomConfig(
        shape=cfg.shape,
        spacing=cfg.spacing,
        lesion_radius_mm=cfg.lesion_radius_mm,
        noise_sigma=cfg.noise_sigma,
        seed=spec.seed,
        head_radius_fraction=cfg.head_radius_fraction,
        lesion_intensity=cfg.lesion_intensity,
        noise_model=cfg.noise_model,
    )
    data, lesion_mask = brain_phantom(phantom_cfg)
    index = lesion_slice_index(lesion_mask, axis=spec.axis)
    image = np.take(data, index, axis=spec.axis)
    mask = np.take(lesion_mask, index, axis=spec.axis)
    windowed = apply_preset(image, window)
    normalised = normalize_array(windowed, normalize)
    return normalised.astype(np.float32), mask.astype(bool)


class SyntheticPatchDataset(Dataset):
    """Patch dataset over a list of specs.

    Slices are cached per seed because regenerating a 64x64x48 phantom for
    every patch would dominate training time.  Augmentation happens after the
    cache lookup so each epoch still sees different pixels.
    """

    def __init__(
        self,
        specs: Sequence[SyntheticSpec],
        *,
        patch_size: int = 32,
        patches_per_spec: int = 1,
        cfg: PhantomConfig | None = None,
        window: str = "brain",
        normalize: str = "minmax",
        augment: AugmentConfig | None = DEFAULT_AUGMENT,
        seed: int = 0,
    ) -> None:
        if not specs:
            raise ValueError("specs must not be empty")
        if patches_per_spec <= 0:
            raise ValueError("patches_per_spec must be positive")
        self.specs = list(specs)
        self.patch_size = int(patch_size)
        self.patches_per_spec = int(patches_per_spec)
        self.cfg = cfg or PhantomConfig()
        self.window = window
        self.normalize = normalize
        self.augment = augment
        self.seed = int(seed)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._items = self._build_index()

    def _build_index(self) -> list[tuple[SyntheticSpec, int]]:
        rng = np.random.default_rng(self.seed)
        items: list[tuple[SyntheticSpec, int]] = []
        for spec in self.specs:
            for _ in range(self.patches_per_spec):
                items.append((spec, int(rng.integers(0, 2**31 - 1))))
        return items

    def _slice(self, spec: SyntheticSpec) -> tuple[np.ndarray, np.ndarray]:
        if spec.seed not in self._cache:
            self._cache[spec.seed] = slice_for_spec(
                spec, self.cfg, window=self.window, normalize=self.normalize
            )
        return self._cache[spec.seed]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        spec, item_seed = self._items[index]
        image, mask = self._slice(spec)
        sampler = PatchSampler(
            image,
            mask,
            self.patch_size,
            seed=item_seed,
            positive_fraction=spec.positive_fraction,
        )
        patch = sampler.sample(1)[0]
        if self.augment is not None:
            augmented = augment_image(patch.image, self.augment, seed=item_seed)
            patch = Patch(augmented, patch.label, patch.center, patch.source_index)
        return {
            "image": torch.from_numpy(patch.image[None, :, :].copy()),
            "label": torch.tensor(float(patch.label), dtype=torch.float32),
        }

    def labels(self) -> np.ndarray:
        """Materialise every label (cheap: no pixels are decoded for this)."""
        labels = []
        for spec, item_seed in self._items:
            image, mask = self._slice(spec)
            sampler = PatchSampler(
                image,
                mask,
                self.patch_size,
                seed=item_seed,
                positive_fraction=spec.positive_fraction,
            )
            patch = sampler.sample(1)[0]
            labels.append(patch.label)
        return np.asarray(labels, dtype=np.int64)


def collate_patches(batch: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack dataset items into batched tensors."""
    items = list(batch)
    if not items:
        raise ValueError("cannot collate an empty batch")
    images = torch.stack([item["image"] for item in items], dim=0)
    labels = torch.stack([item["label"] for item in items], dim=0)
    return {"image": images, "label": labels}


def make_dataloaders(
    specs: Sequence[SyntheticSpec],
    *,
    batch_size: int = 16,
    val_fraction: float = 0.25,
    patch_size: int = 32,
    cfg: PhantomConfig | None = None,
    augment: AugmentConfig | None = DEFAULT_AUGMENT,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split specs into train/validation loaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(specs))
    n_val = max(1, int(round(len(specs) * val_fraction)))
    val_specs = [specs[int(i)] for i in order[:n_val]]
    train_specs = [specs[int(i)] for i in order[n_val:]]
    if not train_specs:
        raise ValueError("validation split consumed every spec; add more specs")
    train_ds = SyntheticPatchDataset(
        train_specs, patch_size=patch_size, cfg=cfg, augment=augment, seed=seed
    )
    val_ds = SyntheticPatchDataset(
        val_specs, patch_size=patch_size, cfg=cfg, augment=None, seed=seed + 1
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_patches
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_patches
    )
    return train_loader, val_loader


def dataset_statistics(dataset: SyntheticPatchDataset) -> dict[str, float]:
    """Class balance of a dataset, computed without loading full batches."""
    labels = dataset.labels()
    positives = int(labels.sum())
    return {
        "items": float(len(labels)),
        "positives": float(positives),
        "negatives": float(len(labels) - positives),
        "positive_rate": float(positives / max(1, len(labels))),
    }


def arrays_from_dataset(
    dataset: SyntheticPatchDataset,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a dataset into NumPy arrays (used by tests and the demo)."""
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    images = []
    labels = []
    for i in range(n):
        item = dataset[i]
        images.append(item["image"].numpy())
        labels.append(float(item["label"]))
    return np.stack(images, axis=0), np.asarray(labels, dtype=np.float32)


def patch_label_counts(patches: Iterable[Patch]) -> dict[int, int]:
    """Count labels in an explicit patch list."""
    counts = {0: 0, 1: 0}
    for patch in patches:
        counts[patch.label] += 1
    return counts

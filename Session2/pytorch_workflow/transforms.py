from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def normalize_ct(image: np.ndarray, lower: float = -1000, upper: float = 400) -> np.ndarray:
    """Clip lung CT intensities and map them to a stable roughly [-1, 1] range."""

    image = np.clip(image.astype(np.float32), lower, upper)
    return 2.0 * (image - lower) / (upper - lower) - 1.0


def crop_or_pad(
    image: np.ndarray,
    target: np.ndarray,
    patch_size: tuple[int, int, int],
    center: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop/pad channel-first volumes to a fixed D,H,W patch."""

    _, depth, height, width = image.shape
    pd, ph, pw = patch_size
    if center is None:
        center = (depth // 2, height // 2, width // 2)

    pads = (
        (0, 0),
        (max(0, pd - depth) // 2, max(0, pd - depth + 1) // 2),
        (max(0, ph - height) // 2, max(0, ph - height + 1) // 2),
        (max(0, pw - width) // 2, max(0, pw - width + 1) // 2),
    )
    image = np.pad(image, pads, mode="constant")
    target = np.pad(target, pads, mode="constant")

    _, depth, height, width = image.shape
    cz, cy, cx = center
    cz += pads[1][0]
    cy += pads[2][0]
    cx += pads[3][0]

    z0 = min(max(cz - pd // 2, 0), depth - pd)
    y0 = min(max(cy - ph // 2, 0), height - ph)
    x0 = min(max(cx - pw // 2, 0), width - pw)
    return (
        image[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw],
        target[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw],
    )


@dataclass
class SegmentationTransform:
    patch_size: tuple[int, int, int] | None
    training: bool = False
    seed: int = 2026

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        image = sample["image"]
        target = sample["target"]

        if self.patch_size is not None:
            center = None
            if self.training and target.sum() > 0 and self.rng.random() < 0.8:
                coords = np.argwhere(target[0] > 0)
                z, y, x = coords[self.rng.randrange(len(coords))]
                center = (int(z), int(y), int(x))
            image, target = crop_or_pad(image, target, self.patch_size, center=center)

        if self.training:
            for axis in (1, 2, 3):
                if self.rng.random() < 0.5:
                    image = np.flip(image, axis=axis).copy()
                    target = np.flip(target, axis=axis).copy()
            scale = self.rng.uniform(0.9, 1.1)
            shift = self.rng.uniform(-0.05, 0.05)
            image[0] = np.clip(image[0] * scale + shift, -1.5, 1.5)

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
        }


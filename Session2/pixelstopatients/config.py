from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TrainingConfig:
    """Small, serializable config for a reproducible 3D segmentation run."""

    nifti_root: Path = Path("aapm_nsclc/nifti")
    output_dir: Path = Path("Session2/tempworkspace/runs/gtv_unet3d")
    target_mask: str = "gtv"
    context_masks: Sequence[str] = ("lung_l", "lung_r", "heart", "esophagus", "cord")
    patch_size: tuple[int, int, int] = (32, 128, 128)  # D, H, W
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    batch_size: int = 1
    epochs: int = 3
    learning_rate: float = 1e-3
    seed: int = 2026
    num_workers: int = 0
    threshold: float = 0.5

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["nifti_root"] = str(self.nifti_root)
        data["output_dir"] = str(self.output_dir)
        data["context_masks"] = list(self.context_masks)
        data["patch_size"] = list(self.patch_size)
        return data


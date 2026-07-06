from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import SegmentationTransform, normalize_ct


@dataclass(frozen=True)
class CaseRecord:
    """One exported Session 1 image series and its available masks."""

    case_id: str
    patient_id: str
    study_id: str
    series_id: str
    image_path: Path
    mask_paths: Mapping[str, Path]
    metadata_path: Path | None = None

    @property
    def has_metadata(self) -> bool:
        return self.metadata_path is not None and self.metadata_path.exists()


def find_nifti_root(search_from: str | Path | None = None) -> Path | None:
    """Locate the Session 1 NIfTI export without hard-coding one relative path.

    Session 1 writes its export to ``Session1/aapm_nsclc/nifti``. Session 2 may be
    launched from the repository root or from the ``Session2`` folder, so the export
    sits at a different relative path depending on where you start. Walk up from
    ``search_from`` (the current directory by default) and return the first matching
    tree, or ``None`` if nothing is found.
    """

    start = Path(search_from).resolve() if search_from else Path.cwd().resolve()
    for base in (start, *start.parents):
        for candidate in (
            base / "Session1" / "aapm_nsclc" / "nifti",
            base / "aapm_nsclc" / "nifti",
        ):
            if candidate.is_dir():
                return candidate
    return None


def discover_cases(
    nifti_root: str | Path,
    required_masks: Sequence[str] = ("gtv",),
) -> list[CaseRecord]:
    """Scan the Session 1 NIfTI tree and return cases ready for learning."""

    root = Path(nifti_root)
    records: list[CaseRecord] = []
    for image_path in sorted(root.rglob("image.nii.gz")):
        series_dir = image_path.parent
        masks_dir = series_dir / "masks"
        mask_paths = {
            p.name[:-7] if p.name.endswith(".nii.gz") else p.stem: p
            for p in sorted(masks_dir.glob("*.nii.gz"))
        } if masks_dir.exists() else {}
        if any(name not in mask_paths for name in required_masks):
            continue

        rel_parts = series_dir.relative_to(root).parts
        patient_id = rel_parts[0] if len(rel_parts) > 0 else "unknown_patient"
        study_id = rel_parts[1] if len(rel_parts) > 1 else "unknown_study"
        series_id = rel_parts[2] if len(rel_parts) > 2 else series_dir.name
        case_id = "__".join(rel_parts)
        metadata_path = series_dir / "metadata.json"
        records.append(
            CaseRecord(
                case_id=case_id,
                patient_id=patient_id,
                study_id=study_id,
                series_id=series_id,
                image_path=image_path,
                mask_paths=mask_paths,
                metadata_path=metadata_path if metadata_path.exists() else None,
            )
        )
    return records


def split_by_patient(
    records: Sequence[CaseRecord],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 2026,
) -> dict[str, list[CaseRecord]]:
    """Split by patient hash to prevent train/val/test leakage."""

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be in [0, 1).")

    patients = sorted({record.patient_id for record in records})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = max(1, round(n * test_fraction)) if n >= 3 and test_fraction > 0 else 0
    n_val = max(1, round(n * val_fraction)) if n >= 3 and val_fraction > 0 else 0
    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test:n_test + n_val])

    splits = {"train": [], "val": [], "test": []}
    for record in records:
        if record.patient_id in test_patients:
            splits["test"].append(record)
        elif record.patient_id in val_patients:
            splits["val"].append(record)
        else:
            splits["train"].append(record)
    return splits


def write_index_csv(records: Iterable[CaseRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "patient_id", "study_id", "series_id", "image_path", "masks"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "case_id": record.case_id,
                    "patient_id": record.patient_id,
                    "study_id": record.study_id,
                    "series_id": record.series_id,
                    "image_path": str(record.image_path),
                    "masks": json.dumps({k: str(v) for k, v in record.mask_paths.items()}),
                }
            )


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Install nibabel to read Session 1 NIfTI exports.") from exc

    arr = nib.load(str(path)).get_fdata(dtype=np.float32)
    return np.asarray(arr, dtype=np.float32).transpose(2, 1, 0)  # x,y,z -> z,y,x


class NiftiSegmentationDataset(Dataset):
    """PyTorch dataset for CT-to-GTV 3D segmentation from Session 1 exports."""

    def __init__(
        self,
        records: Sequence[CaseRecord],
        target_mask: str = "gtv",
        context_masks: Sequence[str] = (),
        patch_size: tuple[int, int, int] | None = (32, 128, 128),
        training: bool = False,
        seed: int = 2026,
    ) -> None:
        self.records = list(records)
        self.target_mask = target_mask
        self.context_masks = tuple(context_masks)
        self.transform = SegmentationTransform(
            patch_size=patch_size,
            training=training,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = normalize_ct(_load_nifti(record.image_path))
        target = (_load_nifti(record.mask_paths[self.target_mask]) > 0.5).astype(np.float32)

        channels = [image]
        for mask_name in self.context_masks:
            path = record.mask_paths.get(mask_name)
            if path is None:
                channels.append(np.zeros_like(image, dtype=np.float32))
            else:
                channels.append((_load_nifti(path) > 0.5).astype(np.float32))

        sample = self.transform(
            {
                "image": np.stack(channels, axis=0),
                "target": target[None, ...],
            }
        )
        sample["case_id"] = record.case_id
        sample["patient_id"] = record.patient_id
        return sample

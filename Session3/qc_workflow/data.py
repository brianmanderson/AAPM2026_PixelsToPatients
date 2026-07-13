"""Index and load Session 1 exports for QC — the same tree Session 2 trains on.

This mirrors ``Session2/pytorch_workflow/data.py`` (case discovery, patient-safe splits,
z,y,x voxel order) so Session 3 reads the *identical* dataset. The one addition is voxel
spacing: Session 2 works in patch space and can ignore it, but QC features are physical
(volume in cc, distances in mm), so we read spacing from the NIfTI affine.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    patient_id: str
    image_path: Path
    mask_paths: dict[str, Path]
    metadata_path: Path | None = None


def find_nifti_root(search_from: str | Path | None = None) -> Path | None:
    """Walk up from ``search_from`` to locate Session 1's ``.../nifti`` export."""
    start = Path(search_from).resolve() if search_from else Path.cwd().resolve()
    for base in (start, *start.parents):
        for candidate in (base / "Session1" / "aapm_nsclc" / "nifti", base / "aapm_nsclc" / "nifti"):
            if candidate.is_dir():
                return candidate
    return None


def discover_cases(nifti_root: str | Path, required_masks: Sequence[str] = ("gtv",)) -> list[CaseRecord]:
    """Scan the NIfTI tree; return cases that have every mask in ``required_masks``."""
    root = Path(nifti_root)
    records: list[CaseRecord] = []
    for image_path in sorted(root.rglob("image.nii.gz")):
        series_dir = image_path.parent
        masks_dir = series_dir / "masks"
        mask_paths = {
            p.name[:-7]: p for p in sorted(masks_dir.glob("*.nii.gz"))
        } if masks_dir.exists() else {}
        if any(name not in mask_paths for name in required_masks):
            continue
        rel = series_dir.relative_to(root).parts
        meta = series_dir / "metadata.json"
        records.append(CaseRecord(
            case_id="__".join(rel),
            patient_id=rel[0] if rel else series_dir.name,
            image_path=image_path,
            mask_paths=mask_paths,
            metadata_path=meta if meta.exists() else None,
        ))
    return records


def split_by_patient(records: Sequence[CaseRecord], holdout_fraction: float = 0.3,
                     seed: int = 2026) -> dict[str, list[CaseRecord]]:
    """Split into ``reference`` (build the check) and ``holdout`` (test it) by patient hash,
    so no patient contributes to both — the same leakage rule Session 2 uses for train/test."""
    patients = sorted({r.patient_id for r in records})
    random.Random(seed).shuffle(patients)
    n_holdout = max(1, round(len(patients) * holdout_fraction)) if len(patients) >= 2 else 0
    holdout = set(patients[:n_holdout])
    out: dict[str, list[CaseRecord]] = {"reference": [], "holdout": []}
    for r in records:
        out["holdout" if r.patient_id in holdout else "reference"].append(r)
    return out


def load_nifti(path: str | Path) -> tuple[object, tuple[float, float, float]]:
    """Return ``(array_zyx, spacing_zyx)``. Array is float32 in z,y,x order (matches Session 2)."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32).transpose(2, 1, 0)
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    return arr, (sz, sy, sx)


def load_case(record: CaseRecord, masks: Sequence[str]):
    """Load the image plus the requested masks. Returns ``(image_zyx, {name: mask_zyx}, spacing)``.
    Missing masks are returned as ``None`` so the caller can decide (context masks are optional)."""
    image, spacing = load_nifti(record.image_path)
    loaded = {}
    for name in masks:
        path = record.mask_paths.get(name)
        loaded[name] = load_nifti(path)[0] if path is not None else None
    return image, loaded, spacing

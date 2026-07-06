from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sphere(shape: tuple[int, int, int], center: tuple[int, int, int], radius: int) -> np.ndarray:
    z, y, x = np.indices(shape)
    cz, cy, cx = center
    return ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2).astype(np.float32)


def write_nifti(array_zyx: np.ndarray, path: Path) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    array_xyz = array_zyx.transpose(2, 1, 0)
    affine = np.diag([1.0, 1.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(array_xyz.astype(np.float32), affine), str(path))


def create_synthetic_session1_export(root: str | Path, n_patients: int = 6) -> Path:
    """Create a tiny NIfTI tree that matches the Session 1 export contract."""

    root = Path(root)
    shape = (24, 48, 48)
    rng = np.random.default_rng(2026)

    for idx in range(n_patients):
        patient = f"patient_{idx:03d}"
        study = "study_000"
        series = "series_000"
        series_dir = root / patient / study / series
        masks_dir = series_dir / "masks"

        center = (
            int(rng.integers(8, 16)),
            int(rng.integers(18, 30)),
            int(rng.integers(18, 30)),
        )
        gtv = _sphere(shape, center, radius=4)
        lung_l = _sphere(shape, (12, 24, 16), radius=13)
        lung_r = _sphere(shape, (12, 24, 32), radius=13)
        heart = _sphere(shape, (13, 30, 24), radius=7)

        image = rng.normal(-780, 80, size=shape).astype(np.float32)
        image += gtv * 900
        image += heart * 250

        write_nifti(image, series_dir / "image.nii.gz")
        write_nifti(gtv, masks_dir / "gtv.nii.gz")
        write_nifti(lung_l, masks_dir / "lung_l.nii.gz")
        write_nifti(lung_r, masks_dir / "lung_r.nii.gz")
        write_nifti(heart, masks_dir / "heart.nii.gz")
        (series_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "PatientAge": "065Y",
                    "PatientSex": "O",
                    "Manufacturer": "Synthetic",
                    "SliceThickness": "3.0",
                },
                indent=2,
            )
            + "\n"
        )

    return root


if __name__ == "__main__":
    create_synthetic_session1_export(Path(__file__).parents[1] / "tempworkspace" / "synthetic_nifti")


"""Reference-free feature extraction for a contour.

Given a binary mask (and its CT), compute a compact geometry + intensity signature — the
same idea as radiomics, but a lightweight numpy/scipy implementation so the session runs
with no extra dependencies. These features are what the QC check characterizes and scores;
none of them needs a ground-truth reference, which is the whole point (at deployment there
is no reference to compare against).

Methodology follows "Reference-Free QC of Organ-at-Risk Contours via Radiomic and Positional
Signatures" (Elguindi et al.) — a shape/appearance vector for the OOD check, plus the mask
centroid (mm) for the positional atlas.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as _ndi
except ImportError:  # scipy is expected, but degrade gracefully rather than crash the demo
    _ndi = None

# The ordered feature contract the reference model is fit on and scored against.
SHAPE_FEATURES = (
    "volume_cc", "voxel_count", "surface_area_mm2", "surface_to_volume",
    "sphericity", "extent", "elongation", "flatness",
    "major_axis_mm", "minor_axis_mm", "least_axis_mm", "bbox_z_mm", "bbox_y_mm", "bbox_x_mm",
    "hu_mean", "hu_std", "hu_p10", "hu_p50", "hu_p90",
)


def _surface_area(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> float:
    """Approximate surface area (mm^2) as boundary voxels weighted by face area."""
    if mask.sum() == 0:
        return 0.0
    if _ndi is not None:
        border = mask & ~_ndi.binary_erosion(mask)
    else:
        border = mask & ~np.pad(mask, 1)[2:, 1:-1, 1:-1][:, :, :]  # crude fallback
    sz, sy, sx = spacing_zyx
    face = (sy * sx + sz * sx + sz * sy) / 3.0  # mean face area of an anisotropic voxel
    return float(border.sum()) * face


def _axes_lengths_mm(coords_mm: np.ndarray) -> tuple[float, float, float]:
    """Principal-axis extents (mm) from the eigenvalues of the coordinate covariance."""
    if len(coords_mm) < 2:
        return 0.0, 0.0, 0.0
    cov = np.cov((coords_mm - coords_mm.mean(0)).T)
    ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
    return tuple(float(4.0 * np.sqrt(v)) for v in ev)  # ~ +/-2 sigma extent


def extract_features(image_zyx: np.ndarray, mask_zyx: np.ndarray,
                     spacing_zyx: tuple[float, float, float]) -> dict:
    """Compute the shape + intensity signature and the centroid (mm) of ``mask_zyx``."""
    mask = np.asarray(mask_zyx) > 0.5
    sz, sy, sx = spacing_zyx
    n = int(mask.sum())
    f: dict[str, float] = dict.fromkeys(SHAPE_FEATURES, 0.0)
    f["centroid_mm"] = [0.0, 0.0, 0.0]
    if n == 0:
        return f

    voxel_vol = sz * sy * sx
    V = n * voxel_vol
    A = _surface_area(mask, spacing_zyx)
    f["voxel_count"] = float(n)
    f["volume_cc"] = V / 1000.0
    f["surface_area_mm2"] = A
    f["surface_to_volume"] = A / max(V, 1e-9)
    f["sphericity"] = (np.pi ** (1 / 3) * (6 * V) ** (2 / 3)) / max(A, 1e-9)

    zz, yy, xx = np.where(mask)
    coords_mm = np.column_stack([zz * sz, yy * sy, xx * sx])
    centroid = coords_mm.mean(0)
    f["centroid_mm"] = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
    bbox = ((np.ptp(zz) + 1) * sz, (np.ptp(yy) + 1) * sy, (np.ptp(xx) + 1) * sx)
    f["bbox_z_mm"], f["bbox_y_mm"], f["bbox_x_mm"] = map(float, bbox)
    f["extent"] = V / max(bbox[0] * bbox[1] * bbox[2], 1e-9)
    major, minor, least = _axes_lengths_mm(coords_mm)
    f["major_axis_mm"], f["minor_axis_mm"], f["least_axis_mm"] = major, minor, least
    f["elongation"] = minor / max(major, 1e-9)
    f["flatness"] = least / max(major, 1e-9)

    hu = np.asarray(image_zyx, dtype=np.float32)[mask]
    f["hu_mean"], f["hu_std"] = float(hu.mean()), float(hu.std())
    f["hu_p10"], f["hu_p50"], f["hu_p90"] = (float(v) for v in np.percentile(hu, [10, 50, 90]))
    return f


def shape_vector(features: dict) -> np.ndarray:
    """The ordered SHAPE_FEATURES as a float vector (NaN for anything missing)."""
    return np.array([features.get(k, np.nan) for k in SHAPE_FEATURES], dtype=float)

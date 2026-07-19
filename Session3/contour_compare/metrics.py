"""Reference-based comparison of two contours of the same structure.

Unlike ``qc_workflow`` (which scores a single contour with *no* reference), this compares
two masks on the same grid — typically an AI contour against the physician's manual or
corrected contour — and reports the metrics that track *editing effort*:

  * volumetric Dice                        — interior overlap
  * surface Dice at 1 / 2 / 3 mm tolerance — Nikolov et al. 2018/2021
  * added / total path length (APL / TPL)  — Vaassen et al. 2020

Volumetric Dice weights the interior, which an editor rarely touches. Surface Dice and APL
weight the *boundary*, which is where correction time actually goes: in Vaassen et al., APL
correlated R=0.87 with adaptation time, and manual path length R=0.90 with from-scratch time —
both far stronger than volumetric Dice. Everything here is physical (mm/cm), so voxel spacing
is threaded through, exactly as ``qc_workflow/features.py`` does.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as _ndi
except ImportError as exc:  # scipy drives erosion + distance transforms; it is required here
    raise ImportError("contour_compare.metrics needs scipy (pip install scipy)") from exc

Spacing = tuple[float, float, float]  # (z, y, x) mm, matching qc_workflow


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0.5


def dice(reference: np.ndarray, test: np.ndarray) -> float:
    """Volumetric Dice = 2|A∩B| / (|A|+|B|). 1.0 if both empty, 0.0 if exactly one is."""
    a, b = _as_bool(reference), _as_bool(test)
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return 2.0 * int((a & b).sum()) / (na + nb)


def _exposed_face_area(mask: np.ndarray, spacing: Spacing) -> np.ndarray:
    """Per-voxel exposed surface area (mm²): sum of face areas whose 6-neighbour is background.

    A boundary voxel with an exposed in-plane-normal face contributes ``sz*sy`` or ``sz*sx`` mm²;
    an exposed axial face contributes ``sy*sx`` mm². Weighting boundary voxels by this area —
    rather than counting them — keeps surface Dice honest on anisotropic (thick-slice) grids,
    where a raw count would over-weight the densely-sampled in-plane directions. Off-volume
    neighbours count as background, so the volume edge is a real surface.
    """
    m = _as_bool(mask)
    sz, sy, sx = spacing
    face_area = {0: sy * sx, 1: sz * sx, 2: sz * sy}  # area of a face with normal along axis a
    area = np.zeros(m.shape, dtype=float)
    for axis, a in face_area.items():
        pad = [(1, 1) if ax == axis else (0, 0) for ax in range(3)]
        p = np.pad(m, pad, constant_values=False)
        lo = [slice(None)] * 3; lo[axis] = slice(0, -2)
        hi = [slice(None)] * 3; hi[axis] = slice(2, None)
        exposed = (m & ~p[tuple(lo)]).astype(float) + (m & ~p[tuple(hi)]).astype(float)
        area += exposed * a
    return area


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Border voxels: in the mask but touching background."""
    return _exposed_face_area(mask, (1.0, 1.0, 1.0)) > 0


def surface_distances(reference: np.ndarray, test: np.ndarray, spacing: Spacing):
    """Symmetric surface-to-surface distances (mm) between the two mask borders.

    Returns ``(d_ref_to_test, d_test_to_ref)`` — for every border voxel of one mask, the
    distance to the nearest border voxel of the other. ``distance_transform_edt`` honours
    anisotropic ``spacing``, so distances are true millimetres. Either array is empty if the
    corresponding mask has no surface.
    """
    surf_ref, surf_test = _surface_voxels(reference), _surface_voxels(test)
    if not surf_ref.any() or not surf_test.any():
        empty = np.array([], dtype=float)
        return (empty if not surf_ref.any() else np.full(int(surf_ref.sum()), np.inf),
                empty if not surf_test.any() else np.full(int(surf_test.sum()), np.inf))
    # EDT measures distance to the nearest zero, so invert the *other* surface to get
    # "distance from every voxel to the nearest border voxel of that other surface".
    dt_to_test = _ndi.distance_transform_edt(~surf_test, sampling=spacing)
    dt_to_ref = _ndi.distance_transform_edt(~surf_ref, sampling=spacing)
    return dt_to_test[surf_ref], dt_to_ref[surf_test]


def surface_dice(reference: np.ndarray, test: np.ndarray, spacing: Spacing,
                 tolerances_mm: tuple[float, ...] = (1.0, 2.0, 3.0)) -> dict[float, float]:
    """Area-weighted surface Dice at each tolerance τ (Nikolov et al.).

    ``SDSC_τ = (A_ref within τ of S_test + A_test within τ of S_ref) / (A_ref + A_test)``

    i.e. the fraction of *both* surfaces' area that agrees to within τ mm. Each border voxel is
    weighted by its exposed face area (see ``_exposed_face_area``), which is the anisotropy-aware
    lightweight stand-in for the marching-cubes surface area used by reference implementations.
    """
    area_ref = _exposed_face_area(reference, spacing)
    area_test = _exposed_face_area(test, spacing)
    surf_ref, surf_test = area_ref > 0, area_test > 0
    if not surf_ref.any() and not surf_test.any():  # both empty → surfaces trivially agree
        return {float(t): 1.0 for t in tolerances_mm}
    if not surf_ref.any() or not surf_test.any():    # one empty → no agreement
        return {float(t): 0.0 for t in tolerances_mm}
    d_rt = _ndi.distance_transform_edt(~surf_test, sampling=spacing)[surf_ref]
    d_tr = _ndi.distance_transform_edt(~surf_ref, sampling=spacing)[surf_test]
    w_ref, w_test = area_ref[surf_ref], area_test[surf_test]
    total = float(w_ref.sum() + w_test.sum())
    return {
        float(t): float((w_ref[d_rt <= t].sum() + w_test[d_tr <= t].sum()) / total)
        for t in tolerances_mm
    }


def _border_2d(mask2d: np.ndarray) -> np.ndarray:
    if not mask2d.any():
        return mask2d
    return mask2d & ~_ndi.binary_erosion(mask2d)


def added_path_length(reference: np.ndarray, test: np.ndarray, spacing: Spacing,
                      tolerance_mm: float = 1.0) -> dict[str, float]:
    """Added and total path length (cm), computed slice-by-slice in the axial plane.

    Contours are drawn per axial slice, so APL is a 2-D, slice-wise quantity (Vaassen et al.):

      * **TPL** — total length of the *reference* (manual/corrected) contour, i.e. what a
        clinician would draw from scratch.
      * **APL** — the length of that reference contour lying **> τ mm from the test contour**:
        the segments the editor must add or redraw. This captures both expansion and shrinkage
        (a shrunk region leaves the corrected boundary far from the AI boundary too). Vaassen
        used τ = 1 mm (one voxel) for auto-vs-manual comparison.

    Path length per boundary pixel ≈ the mean in-plane voxel size, so length = (#pixels) ×
    ``mean(spacing_y, spacing_x)``. Returns ``{apl_cm, tpl_cm, apl_over_tpl}``.
    """
    ref, test = _as_bool(reference), _as_bool(test)
    _, sy, sx = spacing
    pixel_len_mm = 0.5 * (sy + sx)  # length contributed by one traced boundary pixel
    added_px = 0
    total_px = 0
    for z in range(ref.shape[0]):
        ref_border = _border_2d(ref[z])
        n_ref = int(ref_border.sum())
        if n_ref == 0:
            continue
        total_px += n_ref
        test_border = _border_2d(test[z])
        if not test_border.any():
            added_px += n_ref  # nothing to match against → whole slice contour must be drawn
            continue
        dist = _ndi.distance_transform_edt(~test_border, sampling=(sy, sx))
        added_px += int(np.sum(dist[ref_border] > tolerance_mm))
    tpl_cm = total_px * pixel_len_mm / 10.0
    apl_cm = added_px * pixel_len_mm / 10.0
    return {
        "apl_cm": apl_cm,
        "tpl_cm": tpl_cm,
        "apl_over_tpl": (apl_cm / tpl_cm) if tpl_cm > 0 else 0.0,
    }

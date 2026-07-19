"""Demonstrate ``contour_compare`` on a real NSCLC-Radiomics case.

Reference-based comparison needs two contours of the same structure. Here we take one real
case's **manual GTV as the gold standard**, synthesize a realistic **AI GTV** by perturbing it,
and grade the AI against the manual: Dice, surface Dice @1/2/3 mm, added/total path length,
estimated time saved, and a suggested 1–5 clinical-acceptability rating.

The synthesized AI contour is written to ``<series>/masks_ai/gtv.nii.gz`` — a real, on-disk
second data point next to the Session 1 export — then we sweep the perturbation severity to
walk the rubric from "use as-is" (5) down to "unusable" (1).

    python Session3/compare_contours.py

Reads the Session 1 export (``Session1/aapm_nsclc/nifti``); writes the report under
``Session3/outputs/``. No GPU or trained model needed.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Make both Session 3 packages importable from the repo root or from Session3/.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from qc_workflow import QCConfig, discover_cases, find_nifti_root, load_case, split_by_patient
from contour_compare import compare_masks


def synth_ai_contour(mask_zyx: np.ndarray, grow_iters: int = 0, shift_vox: int = 0,
                     region_frac: float = 1.0) -> np.ndarray:
    """Fake an AI contour from a manual one with a realistic, *localized in-plane* edit.

    Real AI-vs-manual differences are (a) drawn in the axial plane — clinicians edit slice by
    slice — and (b) usually confined to a region, not the whole surface. So we grow/shrink and
    shift only *in-plane* (a 2-D structuring element, no z growth) and apply it to a contiguous
    band covering ``region_frac`` of the mask's slices; the rest keeps the manual contour.

    ``grow_iters`` > 0 over-segments (dilate), < 0 under-segments (erode), one voxel = one
    in-plane pixel; ``shift_vox`` slides in-plane (a systematic offset / wrong-place error).
    """
    from scipy import ndimage as ndi

    m = np.asarray(mask_zyx) > 0.5
    se2d = np.zeros((3, 3, 3), bool)
    se2d[1] = ndi.generate_binary_structure(2, 1)  # in-plane cross only — no growth in z

    edited = m
    if grow_iters > 0:
        edited = ndi.binary_dilation(m, se2d, iterations=grow_iters)
    elif grow_iters < 0:
        edited = ndi.binary_erosion(m, se2d, iterations=-grow_iters)
    if shift_vox:
        edited = ndi.shift(edited.astype(np.float32), (0, shift_vox, shift_vox), order=0) > 0.5

    if region_frac >= 1.0:
        return edited.astype(np.float32)
    zs = np.where(m.any(axis=(1, 2)))[0]
    if zs.size == 0:
        return edited.astype(np.float32)
    band = max(1, round((zs.max() - zs.min() + 1) * region_frac))
    out = m.copy()
    out[zs.min():zs.min() + band] = edited[zs.min():zs.min() + band]  # edit a contiguous band
    return out.astype(np.float32)


def save_mask_like(reference_nifti: Path, mask_zyx: np.ndarray, out_path: Path) -> None:
    """Write ``mask_zyx`` (z,y,x) next to the case, reusing the original NIfTI's affine."""
    import nibabel as nib

    orig = nib.load(str(reference_nifti))
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0)).astype(np.uint8)  # back to nibabel x,y,z order
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask_xyz, orig.affine), str(out_path))


def _pick_case(cfg: QCConfig):
    """A held-out case with a reasonably large GTV, so path lengths are meaningful."""
    root = find_nifti_root() or cfg.nifti_root
    records = discover_cases(root, required_masks=(cfg.target_mask,))
    if len(records) < 4:
        raise SystemExit(f"Found {len(records)} cases under {root}. Run Session 1's export first.")
    holdout = split_by_patient(records, holdout_fraction=0.3, seed=cfg.seed)["holdout"]
    best, best_vox = None, -1
    for rec in holdout:
        _, masks, _ = load_case(rec, (cfg.target_mask,))
        vox = int((masks[cfg.target_mask] > 0.5).sum()) if masks[cfg.target_mask] is not None else 0
        if vox > best_vox:
            best, best_vox = rec, vox
    return best


def main() -> None:
    cfg = QCConfig()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rec = _pick_case(cfg)
    image, masks, spacing = load_case(rec, (cfg.target_mask,))
    manual = masks[cfg.target_mask]
    print(f"Case: {rec.case_id}")
    print(f"Spacing (z,y,x mm): {tuple(round(s, 2) for s in spacing)}")
    print(f"Manual GTV: {int((manual > 0.5).sum())} voxels\n")

    # --- Edit one data point: write a realistic 'AI' GTV to disk, then grade it ------------
    ai_gtv = synth_ai_contour(manual, grow_iters=2, shift_vox=1, region_frac=0.6)
    ai_path = rec.image_path.parent / "masks_ai" / "gtv.nii.gz"
    save_mask_like(rec.mask_paths[cfg.target_mask], ai_gtv, ai_path)
    print(f"Wrote synthesized AI contour -> {ai_path.relative_to(find_nifti_root() or cfg.nifti_root)}\n")

    report = compare_masks(manual, ai_gtv, spacing, structure="gtv (AI vs manual)",
                           rate_cm_min=3.0)
    print(report.summary(), "\n")

    # --- Sweep severity to walk the 5→1 rubric ------------------------------------------
    sweep = [
        ("near-perfect", dict(grow_iters=1, shift_vox=0, region_frac=0.15)),
        ("minor",        dict(grow_iters=1, shift_vox=1, region_frac=0.30)),
        ("moderate",     dict(grow_iters=2, shift_vox=1, region_frac=0.60)),
        ("poor",         dict(grow_iters=3, shift_vox=3, region_frac=0.85)),
        ("unusable",     dict(grow_iters=8, shift_vox=12, region_frac=1.00)),
    ]
    rows = [report.to_row() | {"case_id": rec.case_id, "variant": "on-disk AI"}]
    print(f"{'variant':<13}{'Dice':>6}{'sDSC2mm':>9}{'APL/TPL':>9}{'%saved':>8}{'rating':>8}")
    print("-" * 53)
    for name, kw in sweep:
        rep = compare_masks(manual, synth_ai_contour(manual, **kw), spacing,
                            structure=f"gtv:{name}", rate_cm_min=3.0)
        print(f"{name:<13}{rep.dice:>6.2f}{rep.surface_dice[2.0]:>9.2f}"
              f"{rep.apl_over_tpl:>9.0%}{rep.timing.pct_saved:>7.0f}%{rep.rating.score:>6}/5")
        rows.append(rep.to_row() | {"case_id": rec.case_id, "variant": name})

    report_path = out / "contour_comparison.csv"
    with report_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()

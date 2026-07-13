"""Apply the reference-free check to a case — the deployment-time half of Session 3.

Two questions a deployed model can't answer about itself:

1. **Input validation** — *is this even the kind of data I was trained on?* A volume whose
   intensities aren't in a real-CT HU range, or whose spacing is far outside the training
   range, is out-of-distribution *before* the model runs. Silent failures often start here.
2. **Output sanity** — *is the predicted contour plausible?* Score its shape (radiomic OOD)
   and its position (wrong-place residual) against the frozen training distribution.

The verdict is the worst band across the checks, with the reasons attached — a prioritizer
for human review, not an auto-accept gate.
"""
from __future__ import annotations

import numpy as np

from .config import QCConfig
from .features import extract_features, shape_vector
from .reference import ReferenceCheck, position_zscore, shape_zscore

_ORDER = {"pass": 0, "review": 1, "fail": 2}


def _worst(bands: list[str]) -> str:
    return max(bands, key=lambda b: _ORDER.get(b, 0)) if bands else "pass"


def input_validation(image_zyx, spacing_zyx, cfg: QCConfig, profile_spacing: dict | None = None) -> dict:
    """Is this a CT the model can trust? Check HU range and (optionally) spacing vs. training."""
    hu = np.asarray(image_zyx, float)
    p1, p99 = np.percentile(hu, [1, 99])
    # A real CT's signature is air (~-1000 HU) together with soft tissue (> 0). An MR has no
    # negative air (background ~0); a normalized/pre-scaled volume has no -1000. Keying on the
    # air floor separates all three without assuming a high-HU (bone/contrast) tail is in view.
    ct_plausible = bool(p1 < -500 and p99 > -100 and hu.min() < -700)
    checks = {"ct_plausible": ct_plausible, "hu_p1": float(p1), "hu_p99": float(p99)}
    reasons = []
    band = "pass"
    if not ct_plausible:
        band = "fail"
        reasons.append(f"input does not look like CT (HU p1={p1:.0f}, p99={p99:.0f}) — model trained on CT")

    if profile_spacing:
        for i, ax in enumerate(("z", "y", "x")):
            rng = profile_spacing.get(ax)
            if rng and not (rng["min"] - 1e-6 <= spacing_zyx[i] <= rng["max"] + 1e-6):
                band = _worst([band, "review"])
                reasons.append(f"{ax}-spacing {spacing_zyx[i]:.2f} mm outside training range "
                               f"{rng['min']:.2f}–{rng['max']:.2f}")
    checks["band"] = band
    checks["reasons"] = reasons
    return checks


def check_contour(image_zyx, gtv_mask_zyx, context_centroids: dict, spacing_zyx,
                  reference: ReferenceCheck, cfg: QCConfig, profile_spacing: dict | None = None) -> dict:
    """Full QC report for one predicted contour: input + shape + position -> verdict."""
    report: dict = {"input": input_validation(image_zyx, spacing_zyx, cfg, profile_spacing)}

    if gtv_mask_zyx is None or np.asarray(gtv_mask_zyx).sum() == 0:
        report["shape"] = {"evaluated": False, "reason": "empty contour"}
        report["position"] = {"evaluated": False, "reason": "empty contour"}
        report["verdict"] = _worst([report["input"]["band"], "fail"])
        report["reasons"] = report["input"]["reasons"] + ["contour is empty"]
        return report

    feats = extract_features(image_zyx, gtv_mask_zyx, spacing_zyx)

    # Shape axis
    z_shape = shape_zscore(shape_vector(feats), reference.shape)
    shape_band = ("fail" if z_shape >= cfg.shape_z_fail
                  else "review" if z_shape >= cfg.shape_z_review else "pass")
    report["shape"] = {"evaluated": True, "z": round(float(z_shape), 2), "band": shape_band,
                       "volume_cc": round(feats["volume_cc"], 1)}

    # Position axis
    constellation = {reference.target: feats["centroid_mm"], **context_centroids}
    pos = position_zscore(constellation, reference.atlas, target=reference.target)
    pos_band = "pass"
    if pos.get("evaluated") and pos.get("z") is not None:
        pos["wrong_place"] = bool(pos["z"] > cfg.position_z_flag)
        pos_band = "fail" if pos["wrong_place"] else "pass"
    report["position"] = pos

    reasons = list(report["input"]["reasons"])
    if shape_band != "pass":
        reasons.append(f"shape is a training outlier (z={z_shape:.1f}; volume {feats['volume_cc']:.1f} cc)")
    if pos.get("wrong_place"):
        reasons.append(f"contour is far from where a {reference.target} normally sits "
                       f"(residual {pos['residual_mm']} mm, z={pos['z']})")
    report["reasons"] = reasons
    report["verdict"] = _worst([report["input"]["band"], shape_band, pos_band])
    return report


def _centroid_mm(mask, spacing_zyx):
    zz, yy, xx = np.where(np.asarray(mask) > 0.5)
    return [float(zz.mean() * spacing_zyx[0]), float(yy.mean() * spacing_zyx[1]),
            float(xx.mean() * spacing_zyx[2])]


def _lateral_partner(name: str) -> str | None:
    """The L/R partner of a lateralized name (Lung_L<->Lung_R), or None."""
    if name.endswith(("_l", "_L")):
        return name[:-1] + ("R" if name[-1].isupper() else "r")
    if name.endswith(("_r", "_R")):
        return name[:-1] + ("L" if name[-1].isupper() else "l")
    return None


def lateral_check(name: str, centroids: dict, atlas: dict, min_shared: int = 3) -> dict:
    """Catch a left/right *swap*, which the symmetric position residual can miss.

    Align the case using only the **non-paired midline anatomy** (excluding both the structure
    and its L/R partner, so a swap can't corrupt the frame), then ask whether the structure
    landed nearer its own consensus position or its partner's. Nearer the partner's ⇒ wrong side.
    """
    from .reference import _fit_similarity

    partner = _lateral_partner(name)
    T = atlas.get("template", {})
    if partner is None or name not in centroids or name not in T or partner not in T:
        return {"applicable": False}
    anchors = [o for o in centroids if o in T and o not in (name, partner)]
    if len(anchors) < min_shared:
        return {"applicable": False}
    s, t = _fit_similarity(np.array([centroids[o] for o in anchors]),
                           np.array([T[o] for o in anchors]))
    aligned = s * np.asarray(centroids[name], float) + t
    d_self = float(np.linalg.norm(aligned - np.array(T[name])))
    d_partner = float(np.linalg.norm(aligned - np.array(T[partner])))
    return {"applicable": True, "wrong_side": bool(d_partner < d_self),
            "d_self_mm": round(d_self, 1), "d_partner_mm": round(d_partner, 1), "partner": partner}


def check_panel(image_zyx, masks: dict, spacing_zyx, panel, cfg: QCConfig,
                profile_spacing: dict | None = None) -> dict:
    """Run the whole panel on one case → ``{structure: report}``.

    Each structure is checked against its own shape reference; its position is judged by
    aligning the *other* structures (the anchor anatomy) to the shared atlas. So a
    swapped Lung_L/Lung_R shows up as wrong-place on both sides, since each lands where its
    partner belongs.
    """
    centroids = {name: _centroid_mm(masks[name], spacing_zyx)
                 for name in panel.structures
                 if masks.get(name) is not None and np.asarray(masks[name]).sum() > 0}
    out = {}
    for name, ref in panel.references.items():
        mask = masks.get(name)
        if mask is None or np.asarray(mask).sum() == 0:
            out[name] = {"verdict": "skipped", "reasons": ["structure absent from this case"],
                         "shape": {"evaluated": False}, "position": {"evaluated": False}}
            continue
        context = {o: c for o, c in centroids.items() if o != name}
        report = check_contour(image_zyx, mask, context, spacing_zyx, ref, cfg, profile_spacing)
        lat = lateral_check(name, centroids, panel.atlas, cfg.position_min_shared)
        if lat.get("applicable"):
            report["laterality"] = lat
            if lat["wrong_side"]:
                report["verdict"] = _worst([report["verdict"], "fail"])
                report.setdefault("reasons", []).append(
                    f"wrong side — nearer where {lat['partner']} sits ({lat['d_partner_mm']} mm) "
                    f"than {name} ({lat['d_self_mm']} mm)")
        out[name] = report
    return out

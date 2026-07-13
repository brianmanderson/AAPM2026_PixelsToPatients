"""Build the reference-free check from the training cohort.

Two orthogonal axes, following Elguindi et al.:

* **Shape** — is the contour's radiomic signature an *outlier* versus the training GTVs?
  A robust Mahalanobis model over standardized shape/appearance features; the distance is
  reported as a z-score against the training distribution (median/MAD-calibrated).
* **Position** — is the contour *where a GTV normally sits*? A generalized-Procrustes atlas
  co-registers each case's organ-centroid constellation (GTV + lungs/heart/cord/esophagus)
  to a consensus, and the GTV's distance from its consensus position is the wrong-place
  residual — orthogonal to shape, and the thing a perfectly-shaped-but-mislabelled contour
  fails on.

Neither axis needs a ground-truth reference at scoring time: the "reference" is the training
*distribution*, baked into this object. That is what makes it deployable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AXES = ("z", "y", "x")


# --------------------------------------------------------------------------- #
# Shape axis — robust Mahalanobis OOD over the radiomic feature vector.
# --------------------------------------------------------------------------- #
def fit_shape_reference(X: np.ndarray, feature_names: list[str]) -> dict:
    """Fit a robust per-feature outlier model over the training features.

    Each feature is standardized by its **median / MAD** (robust to a few odd training cases),
    and a contour is scored by the RMS of its per-feature robust z — a diagonal Mahalanobis that
    stays stable at the small cohort sizes this workshop runs at, where a full covariance is
    rank-deficient. That RMS distance is already in robust-σ units (≈ 1 for an in-distribution
    contour), so it is thresholded **directly** (see ``QCConfig.shape_z_*``) rather than
    re-calibrated against a tiny sample — which keeps the verdict stable at small n. The
    production system (Elguindi et al.) uses PCA + full covariance once hundreds of cases exist;
    this is the same idea, robust to small n.
    """
    X = np.asarray(X, float)
    median = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - median), axis=0) * 1.4826
    mad = np.maximum(mad, 0.10 * np.abs(median))  # relative floor: a feature that barely varies
    mad[mad < 1e-9] = 1.0                          # in the reference can't dominate the distance
    d = np.sqrt(np.nanmean(_clip_z((X - median) / mad) ** 2, axis=1))
    return {
        "feature_names": list(feature_names),
        "median": median.tolist(), "mad": mad.tolist(),
        "dist_median": float(np.median(d)),  # informational (typical in-distribution distance)
    }


def _clip_z(z: np.ndarray, cap: float = 8.0) -> np.ndarray:
    """Winsorize per-feature z so one degenerate feature can't dominate the distance."""
    return np.clip(np.nan_to_num(z), -cap, cap)


def shape_zscore(features_vec: np.ndarray, ref: dict) -> float:
    """Robust RMS distance (≈ σ) of a contour from the training centre; ≈ 1 in-distribution."""
    median = np.array(ref["median"])
    mad = np.array(ref["mad"])
    return float(np.sqrt(np.nanmean(_clip_z((np.asarray(features_vec, float) - median) / mad) ** 2)))


# --------------------------------------------------------------------------- #
# Position axis — generalized-Procrustes atlas of organ-centroid constellations.
# --------------------------------------------------------------------------- #
def _fit_similarity(C: np.ndarray, G: np.ndarray):
    """Isotropic scale + translation (no rotation, so L/R and S/I stay meaningful)."""
    cb, gb = C.mean(0), G.mean(0)
    Cc, Gc = C - cb, G - gb
    s = (Cc * Gc).sum() / max((Cc * Cc).sum(), 1e-9)
    return s, gb - s * cb


def fit_position_atlas(constellations: list[dict], min_shared: int = 3, iters: int = 10) -> dict:
    """Build a consensus atlas from per-case ``{organ: [z, y, x] mm}`` maps.

    Returns per-organ consensus positions + the GTV residual distribution (median/MAD),
    used to z-score how far a new GTV sits from where it normally does.
    """
    maps = {i: {o: np.asarray(c, float) for o, c in m.items()}
            for i, m in enumerate(constellations) if len(m) >= min_shared}
    if not maps:
        return {}
    organs_by_size = sorted(maps, key=lambda i: -len(maps[i]))
    template = {o: v.copy() for o, v in maps[organs_by_size[0]].items()}

    for _ in range(iters):
        aligned = {}
        for i, m in maps.items():
            shared = [o for o in m if o in template]
            if len(shared) < min_shared:
                continue
            s, t = _fit_similarity(np.array([m[o] for o in shared]),
                                   np.array([template[o] for o in shared]))
            aligned[i] = {o: s * v + t for o, v in m.items()}
        organs = {o for a in aligned.values() for o in a}
        template = {o: np.mean([a[o] for a in aligned.values() if o in a], 0) for o in organs}

    # Leave-one-out residuals: to judge whether organ o is where it belongs, align each case
    # to the atlas using the OTHER organs (o excluded), then measure o's distance. This is what
    # makes a mislocated target detectable — it can't pull its own alignment and hide.
    residuals = {o: [] for o in template}
    for m in maps.values():
        shared = [o for o in m if o in template]
        for o in shared:
            anchors = [a for a in shared if a != o]
            if len(anchors) < min_shared:
                continue
            s, t = _fit_similarity(np.array([m[a] for a in anchors]),
                                   np.array([template[a] for a in anchors]))
            residuals[o].append(float(np.linalg.norm((s * m[o] + t) - template[o])))
    # Floor the MAD so a tiny cohort's tight residual spread doesn't make a normal case look
    # far (a few mm of variation shouldn't read as "wrong place"); the atlas is for catching
    # gross mislocation, not sub-cm jitter.
    stats = {}
    for o, r in residuals.items():
        if len(r) < 3:
            continue
        med = float(np.median(r))
        mad = float(np.median(np.abs(np.array(r) - med)) * 1.4826)
        stats[o] = {"median": med, "mad": max(mad, 0.5 * med, 3.0), "n": len(r)}
    return {"template": {o: v.tolist() for o, v in template.items()},
            "residual_stats": stats, "min_shared": min_shared}


def position_zscore(constellation: dict, atlas: dict, target: str = "gtv") -> dict:
    """Align one case to the atlas and z-score the target's wrong-place residual."""
    if not atlas or not atlas.get("template"):
        return {"evaluated": False, "reason": "no atlas"}
    T = {o: np.array(v) for o, v in atlas["template"].items()}
    present = {o: np.asarray(c, float) for o, c in constellation.items() if o in T}
    anchors = [o for o in present if o != target]
    if target not in present or len(anchors) < atlas["min_shared"]:
        return {"evaluated": False,
                "reason": f"need the {target} + >= {atlas['min_shared']} anchor organs to align "
                          f"(have {len(anchors)} anchors)"}
    # Align on the anchor anatomy only, then measure the target — mirrors the LOO calibration.
    s, t = _fit_similarity(np.array([present[o] for o in anchors]),
                           np.array([T[o] for o in anchors]))
    residual = float(np.linalg.norm((s * present[target] + t) - T[target]))
    st = atlas["residual_stats"].get(target)
    z = (residual - st["median"]) / (st["mad"] or 1.0) if st else None
    return {"evaluated": True, "residual_mm": round(residual, 2),
            "z": round(z, 2) if z is not None else None}


# --------------------------------------------------------------------------- #
# The bundled check (shape + position) — save/load like a model artifact.
# --------------------------------------------------------------------------- #
@dataclass
class ReferenceCheck:
    shape: dict
    atlas: dict
    target: str
    n_reference: int

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "shape": self.shape, "atlas": self.atlas,
            "target": self.target, "n_reference": self.n_reference,
        }))
        return path

    @classmethod
    def load(cls, path: str | Path) -> ReferenceCheck:
        d = json.loads(Path(path).read_text())
        return cls(shape=d["shape"], atlas=d["atlas"], target=d["target"], n_reference=d["n_reference"])

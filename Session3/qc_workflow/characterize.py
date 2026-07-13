"""Characterize the training data, then build the reference-free check from it.

Session 1 already surfaces cohort-level outliers in its manifest (odd spacing, ROI volumes
far from the norm). Session 3 turns that idea into a *deployable* check: it distills the
training cohort into a profile — acquisition ranges (spacing, HU) and the GTV's shape and
positional distributions — and freezes that profile into a ``ReferenceCheck`` that scores
future contours with no ground truth. This is the "know your training distribution, then
watch for anything unlike it" half of model QC.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import QCConfig
from .data import CaseRecord, load_case
from .features import SHAPE_FEATURES, extract_features, shape_vector
from .reference import ReferenceCheck, fit_position_atlas, fit_shape_reference


@dataclass
class CohortProfile:
    """A frozen description of the training cohort — the reference distribution."""

    n_cases: int
    feature_names: list[str]
    feature_matrix: np.ndarray                 # (n_cases, n_features) GTV shape vectors
    constellations: list[dict]                 # per-case {organ: centroid_zyx}
    case_ids: list[str]
    spacing: dict = field(default_factory=dict)      # per-axis {min,p1,p50,p99,max}
    shape_stats: dict = field(default_factory=dict)  # per-feature {median,p1,p99}
    context_coverage: dict = field(default_factory=dict)  # how often each context mask is present

    def to_json_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "feature_names": self.feature_names,
            "spacing": self.spacing,
            "shape_stats": self.shape_stats,
            "context_coverage": self.context_coverage,
        }

    def summary(self) -> str:
        lines = [f"Cohort profile: {self.n_cases} reference cases"]
        sp = self.spacing.get("z", {})
        if sp:
            lines.append(f"  slice spacing z: {sp['min']:.2f}–{sp['max']:.2f} mm "
                         f"(typical {sp['p1']:.2f}–{sp['p99']:.2f})")
        gv = self.shape_stats.get("volume_cc", {})
        if gv:
            lines.append(f"  GTV volume: median {gv['median']:.1f} cc "
                         f"(typical {gv['p1']:.1f}–{gv['p99']:.1f})")
        cov = ", ".join(f"{k} {v:.0%}" for k, v in self.context_coverage.items())
        if cov:
            lines.append(f"  context masks present: {cov}")
        return "\n".join(lines)


def _pctl(a: np.ndarray) -> dict:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    p1, p50, p99 = np.percentile(a, [1, 50, 99])
    return {"min": float(a.min()), "p1": float(p1), "median": float(p50),
            "p99": float(p99), "max": float(a.max())}


def characterize_cohort(records: Sequence[CaseRecord], cfg: QCConfig) -> CohortProfile:
    """Load the reference cases and distill them into a :class:`CohortProfile`."""
    all_masks = (cfg.target_mask, *cfg.context_masks)
    rows, constellations, case_ids, spacings = [], [], [], []
    ctx_present = dict.fromkeys(cfg.context_masks, 0)

    for rec in records:
        image, masks, spacing = load_case(rec, all_masks)
        gtv = masks.get(cfg.target_mask)
        if gtv is None or gtv.sum() == 0:
            continue
        feats = extract_features(image, gtv, spacing)
        rows.append(shape_vector(feats))
        case_ids.append(rec.case_id)
        spacings.append(spacing)

        constellation = {cfg.target_mask: feats["centroid_mm"]}
        for m in cfg.context_masks:
            arr = masks.get(m)
            if arr is not None and arr.sum() > 0:
                ctx_present[m] += 1
                zz, yy, xx = np.where(arr > 0.5)
                constellation[m] = [float(zz.mean() * spacing[0]),
                                    float(yy.mean() * spacing[1]),
                                    float(xx.mean() * spacing[2])]
        constellations.append(constellation)

    X = np.array(rows, dtype=float) if rows else np.empty((0, len(SHAPE_FEATURES)))
    n = len(rows)
    sp_arr = np.array(spacings, float) if spacings else np.empty((0, 3))
    profile = CohortProfile(
        n_cases=n,
        feature_names=list(SHAPE_FEATURES),
        feature_matrix=X,
        constellations=constellations,
        case_ids=case_ids,
        spacing={ax: _pctl(sp_arr[:, i]) for i, ax in enumerate(("z", "y", "x"))} if n else {},
        shape_stats={name: _pctl(X[:, i]) for i, name in enumerate(SHAPE_FEATURES)} if n else {},
        context_coverage={m: (ctx_present[m] / n if n else 0.0) for m in cfg.context_masks},
    )
    return profile


def build_reference(profile: CohortProfile, cfg: QCConfig) -> ReferenceCheck:
    """Fit the shape OOD model + positional atlas from a characterized cohort."""
    if profile.n_cases < 3:
        raise ValueError(f"Need >= 3 reference cases to build a check; got {profile.n_cases}.")
    shape = fit_shape_reference(profile.feature_matrix, profile.feature_names)
    atlas = fit_position_atlas(profile.constellations, min_shared=cfg.position_min_shared)
    return ReferenceCheck(shape=shape, atlas=atlas, target=cfg.target_mask, n_reference=profile.n_cases)

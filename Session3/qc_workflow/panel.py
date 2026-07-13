"""QC a whole structure set, not just the GTV.

A segmentation model rarely outputs one structure — Session 2 (assumed here to train the
GTV *and* the thoracic OARs) produces GTV + Lung_L/R + Heart + Esophagus + Cord. So the
deployable check is a **panel**: one shape reference per structure, plus a single shared
positional atlas (the same generalized-Procrustes constellation, which already knows where
every organ sits relative to the others). OARs are anatomically stereotyped, so their shape
distributions are tight and outliers are meaningful; the shared atlas makes classic errors
like a swapped Lung_L/Lung_R fall out as wrong-place on both sides.

The panel is just ``{structure: ReferenceCheck}`` sharing one atlas — so it reuses the exact
single-structure machinery (``fit_shape_reference``, ``fit_position_atlas``, ``check_contour``).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .characterize import _pctl
from .config import QCConfig
from .data import CaseRecord, load_case
from .features import SHAPE_FEATURES, extract_features, shape_vector
from .reference import ReferenceCheck, fit_position_atlas, fit_shape_reference


@dataclass
class PanelProfile:
    """Per-structure training distributions + the shared anatomical constellations."""

    structures: list[str]
    feature_matrices: dict                     # {structure: (n, n_features)}
    constellations: list[dict]                 # per-case {structure: centroid_zyx}
    n_cases: int
    spacing: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)   # how often each structure is present

    def to_json_dict(self) -> dict:
        return {"structures": self.structures, "n_cases": self.n_cases, "spacing": self.spacing,
                "coverage": self.coverage,
                "counts": {s: int(len(m)) for s, m in self.feature_matrices.items()}}


@dataclass
class QCPanel:
    """A reference-free check for every structure the model produces (shared atlas)."""

    references: dict                           # {structure: ReferenceCheck}
    structures: list[str]
    n_cases: int

    @property
    def atlas(self) -> dict:
        return next(iter(self.references.values())).atlas if self.references else {}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "structures": self.structures, "n_cases": self.n_cases,
            "atlas": self.atlas,  # shared across structures — stored once
            "shape": {name: ref.shape for name, ref in self.references.items()},
            "n_reference": {name: ref.n_reference for name, ref in self.references.items()},
        }))
        return path

    @classmethod
    def load(cls, path: str | Path) -> QCPanel:
        d = json.loads(Path(path).read_text())
        refs = {name: ReferenceCheck(shape=sh, atlas=d["atlas"], target=name,
                                     n_reference=d["n_reference"][name])
                for name, sh in d["shape"].items()}
        return cls(references=refs, structures=d["structures"], n_cases=d["n_cases"])


def characterize_panel(records: Sequence[CaseRecord], cfg: QCConfig) -> PanelProfile:
    """Extract a shape vector for every structure present in each case, plus its constellation."""
    structures = [cfg.target_mask, *cfg.context_masks]
    feats: dict[str, list] = {s: [] for s in structures}
    constellations, spacings = [], []
    coverage = dict.fromkeys(structures, 0)

    for rec in records:
        image, masks, spacing = load_case(rec, structures)
        constellation = {}
        for s in structures:
            arr = masks.get(s)
            if arr is None or arr.sum() == 0:
                continue
            coverage[s] += 1
            fe = extract_features(image, arr, spacing)
            feats[s].append(shape_vector(fe))
            constellation[s] = fe["centroid_mm"]
        if constellation:
            constellations.append(constellation)
            spacings.append(spacing)

    n = len(constellations)
    fmats = {s: (np.array(v, float) if v else np.empty((0, len(SHAPE_FEATURES))))
             for s, v in feats.items()}
    sp = np.array(spacings, float) if spacings else np.empty((0, 3))
    return PanelProfile(
        structures=structures, feature_matrices=fmats, constellations=constellations, n_cases=n,
        spacing={ax: _pctl(sp[:, i]) for i, ax in enumerate(("z", "y", "x"))} if n else {},
        coverage={s: (coverage[s] / n if n else 0.0) for s in structures},
    )


def build_panel(profile: PanelProfile, cfg: QCConfig) -> QCPanel:
    """Fit one shape reference per structure with enough examples + the shared position atlas."""
    atlas = fit_position_atlas(profile.constellations, min_shared=cfg.position_min_shared)
    references = {}
    for s in profile.structures:
        X = profile.feature_matrices.get(s)
        if X is None or len(X) < 3:            # too few examples for a stable reference
            continue
        references[s] = ReferenceCheck(shape=fit_shape_reference(X, list(SHAPE_FEATURES)),
                                       atlas=atlas, target=s, n_reference=len(X))
    if not references:
        raise ValueError("No structure had >= 3 reference examples to build a check.")
    return QCPanel(references=references, structures=list(references), n_cases=profile.n_cases)

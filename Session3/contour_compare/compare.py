"""One call to compare two contours end-to-end: overlap → effort → time → rubric score.

``compare_masks(reference, test, spacing)`` runs the whole chain and returns a
``ComparisonReport``: volumetric Dice, surface Dice at each tolerance, APL/TPL, the
time-saving estimate, and a suggested 1–5 rubric score. ``reference`` is the physician's
manual/corrected contour (the gold standard); ``test`` is the AI contour being graded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import Spacing, added_path_length, dice, surface_dice
from .rubric import RubricThresholds, RatingSuggestion, suggest_rating
from .timesaving import DEFAULT_RATE_CM_MIN, TimeSaving, time_saved


@dataclass(frozen=True)
class ComparisonReport:
    structure: str
    dice: float
    surface_dice: dict[float, float]      # tolerance mm → surface Dice
    apl_cm: float
    tpl_cm: float
    apl_over_tpl: float
    timing: TimeSaving
    rating: RatingSuggestion
    surface_tolerances_mm: tuple[float, ...] = field(default=(1.0, 2.0, 3.0))

    def to_row(self) -> dict:
        """Flat dict for a CSV/DataFrame row."""
        row = {
            "structure": self.structure,
            "dice": round(self.dice, 4),
            "apl_cm": round(self.apl_cm, 2),
            "tpl_cm": round(self.tpl_cm, 2),
            "pct_saved": round(self.timing.pct_saved, 1),
            "time_saved_min": round(self.timing.saved_min, 1),
            "rating": self.rating.score,
            "rating_label": self.rating.label,
        }
        for t in self.surface_tolerances_mm:
            row[f"sdsc_{t:g}mm"] = round(self.surface_dice[t], 4)
        return row

    def summary(self) -> str:
        sd = "  ".join(f"{t:g}mm={self.surface_dice[t]:.2f}" for t in self.surface_tolerances_mm)
        return (
            f"{self.structure}\n"
            f"  volumetric Dice : {self.dice:.3f}\n"
            f"  surface Dice    : {sd}\n"
            f"  APL / TPL       : {self.apl_cm:.1f} / {self.tpl_cm:.1f} cm  "
            f"({self.apl_over_tpl:.0%} of contour needs editing)\n"
            f"  time saved      : {self.timing.saved_min:.1f} min  "
            f"({self.timing.pct_saved:.0f}%  @ {self.timing.rate_cm_min:g} cm/min)\n"
            f"  suggested rating: {self.rating.score}/5 — {self.rating.label}  "
            f"[{self.rating.rationale}]"
        )


def compare_masks(reference: np.ndarray, test: np.ndarray, spacing: Spacing, *,
                  structure: str = "structure",
                  rate_cm_min: float = DEFAULT_RATE_CM_MIN,
                  review_overhead_min: float = 0.0,
                  surface_tolerances_mm: tuple[float, ...] = (1.0, 2.0, 3.0),
                  apl_tolerance_mm: float = 1.0,
                  thresholds: RubricThresholds = RubricThresholds()) -> ComparisonReport:
    """Compare an AI contour (``test``) to a manual/corrected reference on the same grid.

    ``spacing`` is (z, y, x) mm. See the submodules for each metric's definition and citation.
    """
    d = dice(reference, test)
    sdsc = surface_dice(reference, test, spacing, surface_tolerances_mm)
    paths = added_path_length(reference, test, spacing, tolerance_mm=apl_tolerance_mm)
    timing = time_saved(paths["apl_cm"], paths["tpl_cm"], rate_cm_min=rate_cm_min,
                        review_overhead_min=review_overhead_min)
    rating = suggest_rating(sdsc, timing.pct_saved, thresholds)
    return ComparisonReport(
        structure=structure,
        dice=d,
        surface_dice=sdsc,
        apl_cm=paths["apl_cm"],
        tpl_cm=paths["tpl_cm"],
        apl_over_tpl=paths["apl_over_tpl"],
        timing=timing,
        rating=rating,
        surface_tolerances_mm=tuple(float(t) for t in surface_tolerances_mm),
    )

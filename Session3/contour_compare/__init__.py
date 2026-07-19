"""Reference-based contour comparison — Session 3 companion to ``qc_workflow``.

Where ``qc_workflow`` grades one contour with no reference, this compares two contours of the
same structure (AI vs. the physician's manual/corrected contour) and reports the metrics that
predict *editing effort*: volumetric Dice, surface Dice at 1/2/3 mm, added/total path length,
an estimated time saved, and a suggested 1–5 clinical-acceptability rating.

Grounded in:
  * Vaassen et al., "Evaluation of measures for assessing time-saving of automatic OAR
    segmentation" (Phys Imaging Radiat Oncol 2020) — surface DSC + APL vs. correction time.
  * Nikolov et al., "Clinically applicable segmentation of head and neck anatomy" (2018/2021)
    — surface Dice at tolerance τ.
  * Baroudi et al., "Automated Contouring and Planning: What Is Clinically Acceptable?"
    (Diagnostics 2023) — the 5-point clinical-acceptability scale.
"""
from __future__ import annotations

from .compare import ComparisonReport, compare_masks
from .metrics import added_path_length, dice, surface_dice, surface_distances
from .rubric import RUBRIC, RUBRIC_TEXT, RatingSuggestion, RubricThresholds, suggest_rating
from .timesaving import DEFAULT_RATE_CM_MIN, TimeSaving, time_saved

__all__ = [
    "compare_masks", "ComparisonReport",
    "dice", "surface_dice", "surface_distances", "added_path_length",
    "time_saved", "TimeSaving", "DEFAULT_RATE_CM_MIN",
    "suggest_rating", "RatingSuggestion", "RubricThresholds", "RUBRIC", "RUBRIC_TEXT",
]

"""A 1–5 clinical-acceptability rubric for AI contours, plus a geometry-driven suggestion.

The 5-point scale below is the semi-standard one used clinically at MD Anderson and reproduced
in Baroudi et al., *"Automated Contouring and Planning in Radiation Therapy: What Is Clinically
Acceptable?"* (Diagnostics 2023) — a Likert scale of reviewer effort. It is the human judgment
that actually counts; there is no consensus formula that replaces it (scales in the literature
range from 2 to 11 points).

``suggest_rating`` offers a *prior* for that reviewer, not a verdict. It leans on one anchor
the rubric itself provides: level 3 is defined as "edits that can be made in **less time than
starting from scratch**" — i.e. exactly ``time_saved > 0`` (``APL < TPL``). Surface Dice then
separates the upper bands (stylistic vs. use-as-is). Thresholds are the obvious thing to tune
to a given structure and clinic; they are gathered in ``RubricThresholds`` for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass

# (score, short label, description) — verbatim intent of the MD Anderson 5-point scale.
RUBRIC: tuple[tuple[int, str, str], ...] = (
    (5, "Use as-is", "Clinically acceptable; usable for treatment without change."),
    (4, "Minor stylistic edits", "Minor edits that are not necessary — stylistic, not clinically important."),
    (3, "Minor necessary edits", "Necessary edits, but faster to correct than to redraw from scratch."),
    (2, "Major edits", "Substantial edits needed for safe treatment; user would rather start from scratch."),
    (1, "Unusable", "So inaccurate the contour cannot be used; redraw entirely."),
)

RUBRIC_TEXT: dict[int, str] = {score: f"{label} — {desc}" for score, label, desc in RUBRIC}


@dataclass(frozen=True)
class RubricThresholds:
    """Cut-points mapping geometry → a suggested 1–5 score. Tune per structure/clinic.

    ``surface_tolerance_mm`` selects which surface-Dice tolerance drives the score. ``pct_*``
    are percent-of-contour saved (100·(TPL−APL)/TPL). The bands follow the rubric's own wording:
    a lot saved → minor edits (5/4/3); little saved → major (2); a grossly wrong shape,
    caught by the surface-Dice floor, is unusable regardless of path length (1).
    """
    surface_tolerance_mm: float = 2.0
    sdsc_use_as_is: float = 0.95     # ≥ this surface Dice AND pct ≥ pct_use_as_is → 5
    pct_use_as_is: float = 95.0
    sdsc_stylistic: float = 0.90     # ≥ this AND pct ≥ pct_stylistic → 4
    pct_stylistic: float = 80.0
    pct_minor: float = 50.0          # ≥ this pct saved → 3 (clearly faster than scratch)
    pct_major: float = 15.0          # ≥ this pct saved (and shape not grossly wrong) → 2
    sdsc_unusable: float = 0.20      # below this surface Dice → 1 (not the structure; redraw)


@dataclass(frozen=True)
class RatingSuggestion:
    score: int
    label: str
    rationale: str


def suggest_rating(surface_dice_by_tol: dict[float, float], pct_saved: float,
                   thresholds: RubricThresholds = RubricThresholds()) -> RatingSuggestion:
    """Suggest a 1–5 rubric score from surface Dice + percent time saved.

    A *prior for the human reviewer*, not a clinical verdict. Returns the score, its rubric
    label, and a one-line rationale naming the numbers that drove it.
    """
    sdsc = surface_dice_by_tol.get(thresholds.surface_tolerance_mm)
    if sdsc is None:  # fall back to the closest available tolerance
        sdsc = surface_dice_by_tol[min(surface_dice_by_tol, key=lambda t: abs(t - thresholds.surface_tolerance_mm))]
    tol = thresholds.surface_tolerance_mm

    if sdsc < thresholds.sdsc_unusable:
        score = 1  # shape floor: too far off to be this structure, regardless of path length
    elif sdsc >= thresholds.sdsc_use_as_is and pct_saved >= thresholds.pct_use_as_is:
        score = 5
    elif sdsc >= thresholds.sdsc_stylistic and pct_saved >= thresholds.pct_stylistic:
        score = 4
    elif pct_saved >= thresholds.pct_minor:
        score = 3
    elif pct_saved >= thresholds.pct_major:
        score = 2
    else:
        score = 1

    label = RUBRIC_TEXT[score].split(" — ")[0]
    rationale = f"surface Dice@{tol:g}mm={sdsc:.2f}, {pct_saved:.0f}% of contour length saved"
    return RatingSuggestion(score=score, label=label, rationale=rationale)

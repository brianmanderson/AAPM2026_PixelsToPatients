"""Estimate reviewer time saved by an AI contour, from path length and a drawing rate.

Vaassen et al. showed contouring time is well modelled as *proportional to path length*:
manual time ∝ TPL (R=0.90), correction time ∝ APL (R=0.87). Pick a drawing **rate**
(cm of contour per minute) and both times follow:

    manual_min = TPL / rate            # draw the whole thing from scratch
    edit_min   = APL / rate + review   # only redraw the segments the AI got wrong
    saved_min  = manual_min - edit_min
    pct_saved  = 100 * (TPL - APL) / TPL         # rate-independent

``pct_saved`` needs no rate at all — it is purely geometric — which is why it is the robust
headline number (the HN pilot led with the percent, not the minutes). The rate only converts
that fraction into wall-clock minutes, so it should reflect *your* clinic's contouring speed.
"""
from __future__ import annotations

from dataclasses import dataclass

# Illustrative default contouring speed (cm of drawn boundary per minute). Contouring speed
# varies widely by structure, tool, and operator — treat this as a placeholder and pass your
# own measured rate(s). Documented here so the number is never silently assumed.
DEFAULT_RATE_CM_MIN = 4.0


@dataclass(frozen=True)
class TimeSaving:
    manual_min: float   # modelled from-scratch time
    edit_min: float     # modelled AI-assisted (correct-only) time
    saved_min: float    # manual - edit (can go negative when the AI hurts)
    pct_saved: float    # rate-independent: 100 * (TPL - APL) / TPL
    rate_cm_min: float
    review_overhead_min: float


def time_saved(apl_cm: float, tpl_cm: float, rate_cm_min: float = DEFAULT_RATE_CM_MIN,
               review_overhead_min: float = 0.0) -> TimeSaving:
    """Convert APL/TPL into a time-saving estimate at a given drawing ``rate`` (cm/min).

    ``review_overhead_min`` is a fixed cost of *reviewing* the AI contour even when no edit is
    needed — set it > 0 to model that an accepted contour still has to be looked at. With a
    positive overhead a near-perfect AI contour can still show a small saving rather than zero.
    """
    if tpl_cm <= 0:  # nothing to contour → nothing to save
        return TimeSaving(0.0, 0.0, 0.0, 0.0, rate_cm_min, review_overhead_min)
    rate = max(rate_cm_min, 1e-9)
    manual_min = tpl_cm / rate
    edit_min = apl_cm / rate + review_overhead_min
    pct_saved = 100.0 * (tpl_cm - apl_cm) / tpl_cm
    return TimeSaving(
        manual_min=manual_min,
        edit_min=edit_min,
        saved_min=manual_min - edit_min,
        pct_saved=pct_saved,
        rate_cm_min=rate_cm_min,
        review_overhead_min=review_overhead_min,
    )

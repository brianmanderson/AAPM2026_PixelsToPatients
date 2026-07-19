"""Self-tests for ``contour_compare`` — run directly (``python Session3/test_contour_compare.py``)
or under pytest. No data or GPU needed; everything runs on tiny synthetic masks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contour_compare import added_path_length, compare_masks, dice, surface_dice

SPACING = (3.0, 1.0, 1.0)


def _box(shape=(10, 40, 80), z=(2, 8), y=(10, 30), x=(10, 30)):
    m = np.zeros(shape, np.float32)
    m[z[0]:z[1], y[0]:y[1], x[0]:x[1]] = 1
    return m


def test_identity_is_perfect():
    a = _box()
    assert dice(a, a) == 1.0
    assert surface_dice(a, a, SPACING) == {1.0: 1.0, 2.0: 1.0, 3.0: 1.0}
    p = added_path_length(a, a, SPACING)
    assert p["apl_cm"] == 0.0 and p["tpl_cm"] > 0


def test_empty_test_needs_full_redraw():
    a, empty = _box(), np.zeros((10, 40, 80), np.float32)
    assert dice(a, empty) == 0.0
    assert surface_dice(a, empty, SPACING)[2.0] == 0.0
    p = added_path_length(a, empty, SPACING)
    assert abs(p["apl_cm"] - p["tpl_cm"]) < 1e-9  # everything must be drawn


def test_both_empty_agree():
    empty = np.zeros((10, 40, 80), np.float32)
    assert dice(empty, empty) == 1.0
    assert surface_dice(empty, empty, SPACING)[2.0] == 1.0


def test_disjoint_is_unusable():
    a = _box()
    b = _box(x=(55, 75))  # far away, no overlap and no wrap-around
    r = compare_masks(a, b, SPACING)
    assert r.dice == 0.0 and r.surface_dice[3.0] == 0.0 and r.rating.score == 1


def test_metrics_are_monotonic_in_shift():
    a = _box()
    prev_sdsc, prev_pct = 1.01, 101.0
    for shift in (0, 1, 2, 4):
        b = _box(x=(10 + shift, 30 + shift))
        r = compare_masks(a, b, SPACING)
        assert r.surface_dice[2.0] <= prev_sdsc + 1e-9
        assert r.timing.pct_saved <= prev_pct + 1e-9
        prev_sdsc, prev_pct = r.surface_dice[2.0], r.timing.pct_saved


def test_pct_saved_is_rate_independent():
    a, b = _box(), _box(x=(12, 32))
    slow = compare_masks(a, b, SPACING, rate_cm_min=2.0)
    fast = compare_masks(a, b, SPACING, rate_cm_min=6.0)
    assert abs(slow.timing.pct_saved - fast.timing.pct_saved) < 1e-9  # geometry, not speed
    assert fast.timing.saved_min < slow.timing.saved_min              # minutes scale with rate


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")

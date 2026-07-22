"""Regression test for the pass2.py overlay-pruning bug: when refine_track
finds zero arcs in a window (real detections too sparse/scattered to ever
cluster into ball_arc_min_pts within ball_arc_gap_frames — the far-baseline
ball case), the caller must NOT wipe every already-picked per-frame ball
detection. That silently turned a sparse-but-real ball track into a blank
one, which downstream (refine module) reads as a fixed, content-independent
"no ball ever" signal indistinguishable from a genuinely ball-less window.

This mirrors the exact snippet in pass2.py's Stage 3-4 block rather than
importing pass2 itself (which needs a real video + models to run).
"""
from __future__ import annotations

from ..ball_filter import refine_track


def _prune(entries: dict[int, dict], ball_obs: list[tuple], refined) -> None:
    """The fixed pass2.py snippet: only prune off-arc points when at least
    one arc was actually fitted to judge them against."""
    kept = {p.frame for p in refined.points}
    if refined.arcs:
        for bf, _, _, _ in ball_obs:
            if bf not in kept and bf in entries:
                entries[bf].pop("ball", None)


def test_zero_arcs_keeps_sparse_real_detections():
    # Three real detections, each far more than ball_arc_gap_frames (10 px
    # frames) apart in time -> _segment splits them into three 1-point
    # segments, none reaching min_arc_pts=4 -> zero arcs, by design.
    ball_obs = [(0, 100.0, 100.0, 8.0),
                (50, 300.0, 250.0, 8.0),
                (100, 500.0, 400.0, 8.0)]
    entries = {f: {"f": f, "ball": [x, y, d, 1.0]} for f, x, y, d in ball_obs}

    refined = refine_track(ball_obs)
    assert refined.arcs == []          # confirms this hits the zero-arc case

    _prune(entries, ball_obs, refined)
    for f, _, _, _ in ball_obs:
        assert "ball" in entries[f], f"frame {f} was wiped despite zero arcs"


def test_real_arc_still_prunes_off_arc_noise():
    # A genuine 5-point arc (real shot) plus one isolated noise point far
    # from it in time -> the noise point must still be dropped once we HAVE
    # a real trajectory to judge it against.
    arc_obs = [(f, 100.0 + f * 5, 100.0 + f * 2 - 0.05 * f * f, 8.0)
               for f in range(5)]
    noise_obs = [(200, 900.0, 900.0, 8.0)]
    ball_obs = arc_obs + noise_obs
    entries = {f: {"f": f, "ball": [x, y, d, 1.0]} for f, x, y, d in ball_obs}

    refined = refine_track(ball_obs)
    assert refined.arcs, "expected the 5-point shot to fit an arc"

    _prune(entries, ball_obs, refined)
    assert "ball" not in entries[200]
    assert all("ball" in entries[f] for f, *_ in arc_obs)

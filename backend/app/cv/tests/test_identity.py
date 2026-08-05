"""Regression test for the player-identity swap bug confirmed on real data
(singles clip 9ac7be49..., near-team slot 2 -> slot 1 flip at frame 10453):
a continuously-tracked identity lost to one dormant for 1467 frames because a
single noisy torso-histogram reading outweighed a decisive position gap. The
fix (STICKY_BONUS in identity.py) must keep the recently-tracked identity
unless the challenger's cost is winning by more than the bonus can offset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..identity import PlayerIdentityTracker, _Identity


@dataclass
class _Obs:
    court_xy: tuple[float, float]


def test_sticky_bonus_keeps_continuously_tracked_identity():
    t = PlayerIdentityTracker()
    frame_idx = 10453

    # slot A: matched one frame ago, 0.4m from the new detection (real player).
    a = _Identity(id=1, team="near", court_xy=(0.65, 12.88), last_frame=frame_idx - 1)
    # slot B: dormant for 1467 frames, 3.1m from the new detection (stale ghost).
    b = _Identity(id=2, team="near", court_xy=(3.06, 14.18), last_frame=frame_idx - 1467)
    t._identities = [a, b]
    t._next_id = 3

    det = _Obs(court_xy=(0.24, 12.87))

    # Rig appearance so, on cost alone (pre-fix), B would win: perfect
    # appearance match to B (app=0), worst possible to A (app=1) — the exact
    # kind of single-frame histogram noise observed on the real clip.
    cost_a = t._cost(det, hist=None, idn=a, frame_idx=frame_idx)  # hist=None -> app defaults to 1.0
    cost_b_no_bonus_equiv = 3.11 / t.POS_SCALE + 0.0  # app=0 case for B

    # With the sticky bonus, A's continuity discount must be enough to beat
    # B even though B's raw (pos+app) cost is near its floor.
    assert cost_a < cost_b_no_bonus_equiv, (
        "hysteresis bonus should let the continuously-tracked identity win "
        "over a long-dormant one even when the challenger's appearance term "
        "is as favorable as it can possibly be")


def test_sticky_bonus_decays_so_real_occlusion_recovery_still_works():
    t = PlayerIdentityTracker()
    frame_idx = 500
    # Both identities long-dormant (past STICKY_WINDOW) -> no bonus applies,
    # position/appearance alone decide, exactly as before the fix.
    a = _Identity(id=1, team="near", court_xy=(0.0, 0.0), last_frame=frame_idx - 50)
    b = _Identity(id=2, team="near", court_xy=(5.0, 5.0), last_frame=frame_idx - 50)
    t._identities = [a, b]

    det = _Obs(court_xy=(5.1, 5.1))  # clearly closer to b
    assert t._cost(det, None, a, frame_idx) > t._cost(det, None, b, frame_idx)


@dataclass
class _FullObs:
    court_xy: tuple[float, float]
    team: str
    box: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)
    keypoints: list = None
    slot: int | None = None
    number: int | None = None


def test_max_per_team_caps_singles_at_one_identity_per_side():
    """Root-cause regression for the 'singles clip tracks 4 players' bug:
    max_per_team=1 (singles) must never create a 2nd persistent identity on
    the same side, even when a 2nd detection is real and well-separated —
    it force-binds to the existing slot (or is dropped as noise) instead,
    exactly as MAX_PER_TEAM=2 already did for a 3rd doubles detection."""
    t = PlayerIdentityTracker(max_per_team=1)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    o1 = _FullObs(court_xy=(0.0, 0.0), team="near")
    t.assign(frame, [o1], frame_idx=0)
    assert o1.slot is not None

    # A second, clearly distinct near-side detection one frame later — in
    # doubles (max_per_team=2) this would get its own slot; in singles it
    # must not.
    o2 = _FullObs(court_xy=(8.0, 8.0), team="near")
    t.assign(frame, [o2], frame_idx=1)
    assert len(t._team_ids("near")) == 1, "singles must never grow past 1 identity per side"


def test_max_per_team_two_is_unchanged_default():
    t = PlayerIdentityTracker()  # default, same as before this change
    assert t.MAX_PER_TEAM == 2


if __name__ == "__main__":
    test_sticky_bonus_keeps_continuously_tracked_identity()
    test_sticky_bonus_decays_so_real_occlusion_recovery_still_works()
    test_max_per_team_caps_singles_at_one_identity_per_side()
    test_max_per_team_two_is_unchanged_default()
    print("identity self-check OK")

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

from ..identity import PlayerIdentityTracker, _Identity, _circular_ema, _dominant_color, _hue_dist


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
    cost_a = t._cost(det, hist=None, dom=None, idn=a, frame_idx=frame_idx)  # -> app defaults to 1.0
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
    assert t._cost(det, None, None, a, frame_idx) > t._cost(det, None, None, b, frame_idx)


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


def test_hue_dist_is_circular():
    # Red sits at both ends of OpenCV's 0-180 hue range — a naive |a-b|
    # would call these maximally different; circularly they're near-identical.
    assert _hue_dist(2.0, 178.0) < 0.05
    assert _hue_dist(0.0, 90.0) == 1.0  # opposite hues, max distance


def test_circular_ema_crosses_the_hue_boundary_correctly():
    # Averaging 2 and 178 linearly gives ~90 (a totally different colour).
    # The circular EMA must step toward the short way around (through 0/180).
    result = _circular_ema(178.0, 2.0, alpha=0.5)
    assert result < 10 or result > 170, f"expected near the 0/180 boundary, got {result}"


def test_dominant_color_none_when_hist_is_none():
    assert _dominant_color(None) is None


def test_same_kit_doubles_pair_flagged_ambiguous_on_crossing():
    """The failure case the task calls out by name: two same-team players in
    near-identical kit, crossing paths (both position and colour close) —
    must be flagged, not silently resolved to a coin-flip pick."""
    t = PlayerIdentityTracker(max_per_team=2)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    a = _Identity(id=1, team="near", court_xy=(0.0, 0.0),
                  dom_hue=60.0, dom_sat=120.0, dom_conf=0.3, last_frame=-1)
    b = _Identity(id=2, team="near", court_xy=(0.3, 0.3),
                  dom_hue=61.0, dom_sat=118.0, dom_conf=0.3, last_frame=-1)
    t._identities = [a, b]
    t._next_id = 3

    # Two detections near BOTH identities' positions, with no usable crop
    # (keypoints=None -> _torso_hist/-dominant_color both return None for the
    # tiny synthetic frame) — position alone can't separate a 0.3m gap either.
    o1 = _FullObs(court_xy=(0.05, 0.05), team="near")
    o2 = _FullObs(court_xy=(0.25, 0.25), team="near")
    t._assign_team(frame, [o1, o2], "near", frame_idx=0)

    assert o1.slot is not None and o2.slot is not None
    assert o1.identity_ambiguous and o2.identity_ambiguous, (
        "a genuine near-tie between two same-team identities must be flagged")


def test_decisive_match_is_not_flagged_ambiguous():
    """Sanity check the flag isn't just always True: identities far apart in
    both position and colour must NOT be flagged."""
    t = PlayerIdentityTracker(max_per_team=2)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    a = _Identity(id=1, team="near", court_xy=(0.0, 0.0), last_frame=-1)
    b = _Identity(id=2, team="near", court_xy=(10.0, 10.0), last_frame=-1)
    t._identities = [a, b]
    t._next_id = 3

    o1 = _FullObs(court_xy=(0.1, 0.1), team="near")
    o2 = _FullObs(court_xy=(9.9, 9.9), team="near")
    t._assign_team(frame, [o1, o2], "near", frame_idx=0)

    assert not o1.identity_ambiguous and not o2.identity_ambiguous


if __name__ == "__main__":
    test_sticky_bonus_keeps_continuously_tracked_identity()
    test_sticky_bonus_decays_so_real_occlusion_recovery_still_works()
    test_max_per_team_caps_singles_at_one_identity_per_side()
    test_max_per_team_two_is_unchanged_default()
    test_hue_dist_is_circular()
    test_circular_ema_crosses_the_hue_boundary_correctly()
    test_dominant_color_none_when_hist_is_none()
    test_same_kit_doubles_pair_flagged_ambiguous_on_crossing()
    test_decisive_match_is_not_flagged_ambiguous()
    print("identity self-check OK")

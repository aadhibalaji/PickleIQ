"""Synthetic ContactEvent lists -> shot-side % and rally-length bucket math.
No DB involved: pure functions only, same pattern as test_position_metrics."""
from __future__ import annotations

from ..contracts import ContactEvent
from ..shot_metrics import (
    RALLY_MEDIUM_MAX_SHOTS,
    RALLY_SHORT_MAX_SHOTS,
    bucket_rally_length,
    compute_rally_shot_metrics,
    shot_side_pct,
)


def _ev(player_id, side, rally_id="r1"):
    return ContactEvent(contact_id="c", video_id="v", rally_id=rally_id, frame_idx=0,
                         timestamp_s=0.0, player_id=player_id, team="near", court_x=0.0,
                         court_y=0.0, shot_side=side, hand_speed_peak=0.0,
                         confidence=1.0, is_serve=False)


def test_bucket_boundaries():
    assert bucket_rally_length(1) == "short"
    assert bucket_rally_length(RALLY_SHORT_MAX_SHOTS) == "short"        # 5 -> short
    assert bucket_rally_length(RALLY_SHORT_MAX_SHOTS + 1) == "medium"   # 6 -> medium
    assert bucket_rally_length(RALLY_MEDIUM_MAX_SHOTS) == "medium"      # 10 -> medium
    assert bucket_rally_length(RALLY_MEDIUM_MAX_SHOTS + 1) == "long"    # 11 -> long
    assert bucket_rally_length(50) == "long"


def test_shot_side_pct_math():
    events = [_ev(1, "left"), _ev(1, "left"), _ev(1, "left"), _ev(1, "right")]
    pct = shot_side_pct(events)
    assert pct[1]["left"] == 0.75
    assert pct[1]["right"] == 0.25


def test_shot_side_pct_per_player_independent():
    events = [_ev(1, "left"), _ev(1, "right"), _ev(2, "right"), _ev(2, "right")]
    pct = shot_side_pct(events)
    assert pct[1] == {"left": 0.5, "right": 0.5}
    assert pct[2] == {"left": 0.0, "right": 1.0}


def test_compute_rally_shot_metrics_short():
    events = [_ev(1, "left") for _ in range(3)]
    rm = compute_rally_shot_metrics("r1", "v1", events)
    assert rm.shot_count == 3
    assert rm.rally_length_bucket == "short"
    assert rm.shot_side_pct[1]["left"] == 1.0


def test_compute_rally_shot_metrics_medium_and_long_boundaries():
    medium_events = [_ev(1, "left") for _ in range(RALLY_MEDIUM_MAX_SHOTS)]
    long_events = [_ev(1, "left") for _ in range(RALLY_MEDIUM_MAX_SHOTS + 1)]
    assert compute_rally_shot_metrics("r1", "v1", medium_events).rally_length_bucket == "medium"
    assert compute_rally_shot_metrics("r2", "v1", long_events).rally_length_bucket == "long"


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    failures = 0
    for name in dir(mod):
        if name.startswith("test_"):
            try:
                getattr(mod, name)()
                print(f"ok  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)

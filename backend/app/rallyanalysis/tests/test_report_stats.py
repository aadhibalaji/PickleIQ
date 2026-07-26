"""Pure-function tests for report_stats.py: velocity finite-diff, kitchen
arrival crossing, zone bucketing, and null-omission (never zero-fill). No DB,
same pattern as test_coaching.py / test_readiness.py."""
from __future__ import annotations

from ...cv.court import KITCHEN, NET_Y
from ..report_stats import (ZONE_BACKCOURT, ZONE_KITCHEN, ZONE_MIDCOURT,
                             _zone_for_court_y, forward_momentum_series,
                             kitchen_arrival_s, shot_zone_counts)


# --------------------------------------------------------- forward momentum

def test_forward_momentum_first_sample_has_no_velocity():
    samples = [(0, 3.0, 12.0, "near"), (30, 3.0, 9.0, "near")]
    series = forward_momentum_series(samples, fps=30.0)
    assert series[0]["velocity_y"] is None
    assert series[0]["court_y"] == 12.0


def test_forward_momentum_finite_difference():
    samples = [(0, 3.0, 12.0, "near"), (30, 3.0, 9.0, "near"), (60, 3.0, 6.5, "near")]
    series = forward_momentum_series(samples, fps=30.0)
    assert abs(series[1]["velocity_y"] - (-3.0)) < 1e-9   # (9-12)/1s
    assert abs(series[2]["velocity_y"] - (-2.5)) < 1e-9   # (6.5-9)/1s


def test_forward_momentum_empty_samples_yields_empty_series():
    assert forward_momentum_series([], fps=30.0) == []


# ----------------------------------------------------------- kitchen arrival

def test_kitchen_arrival_near_team_crossing():
    samples = [(0, 3.0, 12.0, "near"), (30, 3.0, 9.0, "near"), (60, 3.0, 6.5, "near")]
    # NET_Y + KITCHEN ~= 8.8392 -> first sample below that is frame 60
    arrival = kitchen_arrival_s(samples, rally_start_frame=0, fps=30.0)
    assert arrival == 2.0


def test_kitchen_arrival_far_team_crossing():
    samples = [(0, 3.0, 0.5, "far"), (30, 3.0, 4.0, "far"), (60, 3.0, 5.0, "far")]
    # NET_Y - KITCHEN ~= 4.572 -> first sample >= that is frame 60
    arrival = kitchen_arrival_s(samples, rally_start_frame=0, fps=30.0)
    assert arrival == 2.0


def test_kitchen_arrival_never_reached_is_null():
    samples = [(0, 3.0, 12.0, "near"), (30, 3.0, 11.5, "near")]
    assert kitchen_arrival_s(samples, rally_start_frame=0, fps=30.0) is None


def test_kitchen_arrival_no_samples_is_null_never_zero():
    assert kitchen_arrival_s([], rally_start_frame=0, fps=30.0) is None


def test_kitchen_arrival_infers_side_when_team_missing():
    samples = [(0, 3.0, 12.0, None), (30, 3.0, 6.0, None)]  # near side by starting y
    arrival = kitchen_arrival_s(samples, rally_start_frame=0, fps=30.0)
    assert arrival == 1.0


# ----------------------------------------------------------------- zoning

def test_zone_for_court_y_kitchen_midcourt_backcourt():
    assert _zone_for_court_y(NET_Y) == ZONE_KITCHEN
    assert _zone_for_court_y(NET_Y - KITCHEN + 0.001) == ZONE_KITCHEN  # just inside the line
    assert _zone_for_court_y(NET_Y - KITCHEN - 0.01) == ZONE_MIDCOURT
    assert _zone_for_court_y(0.0) == ZONE_BACKCOURT
    assert _zone_for_court_y(2 * NET_Y) == ZONE_BACKCOURT  # near baseline


# ---------------------------------------------------------- shot zone counts

def test_shot_zone_counts_buckets_by_side_zone_and_serve():
    contacts = [
        {"player_id": 1, "shot_side": "left", "court_x": 1.0, "court_y": NET_Y, "is_serve": False},
        {"player_id": 1, "shot_side": "left", "court_x": 1.0, "court_y": 0.0, "is_serve": True},
        {"player_id": 2, "shot_side": "right", "court_x": 2.0, "court_y": NET_Y, "is_serve": False},
    ]
    counts = shot_zone_counts(contacts)
    assert counts[1]["left"][ZONE_KITCHEN]["non_serve"] == 1
    assert counts[1]["left"][ZONE_BACKCOURT]["serve"] == 1
    assert counts[2]["right"][ZONE_KITCHEN]["non_serve"] == 1


def test_shot_zone_counts_excludes_null_position_only():
    contacts = [
        {"player_id": 1, "shot_side": "left", "court_x": None, "court_y": None, "is_serve": False},
        {"player_id": 1, "shot_side": "right", "court_x": 1.0, "court_y": NET_Y, "is_serve": False},
    ]
    counts = shot_zone_counts(contacts)
    assert "left" not in counts[1]
    assert counts[1]["right"][ZONE_KITCHEN]["non_serve"] == 1


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

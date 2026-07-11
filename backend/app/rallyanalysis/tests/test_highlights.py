"""Synthetic RallyMetrics / overlay position data -> tag classification.
No DB involved: pure functions only, same pattern as test_position_metrics /
test_shot_metrics."""
from __future__ import annotations

from ..highlights import (
    FIREFIGHT_MIN_SHOTS_PER_S,
    TAG_FIREFIGHT,
    TAG_LONG_RALLY,
    TAG_POACH,
    aggregate_rally_positions,
    classify_rally_tags,
    team_has_poach,
)


def test_long_rally_tag_on_bucket_only():
    assert classify_rally_tags("long", 1, 1.0, {}, {}) == [TAG_LONG_RALLY]
    assert classify_rally_tags("medium", 1, 1.0, {}, {}) == []


def test_firefight_needs_both_shot_rate_and_sustained_net_position():
    near_net = {1: 0.8, 2: 0.8}
    team_x = {"near": {1: [3.0]}, "far": {2: [3.0]}}
    fast_enough = FIREFIGHT_MIN_SHOTS_PER_S * 10  # comfortably over threshold, 10s rally
    assert classify_rally_tags("short", fast_enough, 10.0, near_net, team_x) == [TAG_FIREFIGHT]

    # slow shot rate -> no Firefight even with both players camped at net
    assert classify_rally_tags("short", 1, 10.0, near_net, team_x) == []

    # fast shot rate but neither player sustained net position -> no Firefight
    assert classify_rally_tags("short", fast_enough, 10.0, {1: 0.1, 2: 0.1}, team_x) == []

    # fast shot rate, only one team represented in position data -> no Firefight
    assert classify_rally_tags("short", fast_enough, 10.0, near_net, {"near": {1: [3.0]}, "far": {}}) == []


def test_poach_detects_crossing_past_partner_home_side():
    # player 1 sits left (mean x < center), player 2 sits right; player 1
    # then crosses well past center into player 2's side.
    team_x = {1: [1.0, 1.0, 5.5], 2: [5.0, 5.0, 5.0]}
    assert team_has_poach(team_x) is True
    assert classify_rally_tags("short", 0, 1.0, {}, {"near": team_x, "far": {}}) == [TAG_POACH]


def test_no_poach_when_no_crossing_or_no_partner_split():
    # stays on own side the whole rally
    assert team_has_poach({1: [1.0, 1.5], 2: [5.0, 4.5]}) is False
    # both players nominally on the same side -> can't tell home sides apart
    assert team_has_poach({1: [1.0, 1.2], 2: [1.5, 1.3]}) is False
    # only one player tracked -> can't determine a partner split
    assert team_has_poach({1: [1.0, 5.5]}) is False


def test_aggregate_rally_positions_from_overlay_entries():
    entries = [
        {"f": 0, "players": [{"id": 1, "team": "near", "cxy": [3.0, 7.0]},   # in near kitchen
                              {"id": 2, "team": "far", "cxy": [3.0, 12.0]}]},  # far from own kitchen
        {"f": 1, "players": [{"id": 1, "team": "near", "cxy": [3.0, 12.0]},
                              {"id": 2, "team": "far", "cxy": [3.0, 12.0]}]},
    ]
    near_net_frac, team_x = aggregate_rally_positions(entries)
    assert near_net_frac[1] == 0.5
    assert near_net_frac[2] == 0.0
    assert team_x["near"][1] == [3.0, 3.0]
    assert team_x["far"][2] == [3.0, 3.0]


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

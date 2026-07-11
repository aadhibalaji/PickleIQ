"""Synthetic per-frame position data -> aggregation math. No DB involved:
these exercise the pure functions (aggregate_frames, infer_serving_team,
_aggregate_rallies) directly, same pattern as refine/tests."""
from __future__ import annotations

from ..position_metrics import (_aggregate_rallies, _depth_from_baseline,
                                _in_kitchen, aggregate_frames,
                                infer_serving_team)


def _frame(f, near_y=None, far_y=None, near_x=3.0, far_x=3.0):
    players = []
    if near_y is not None:
        players.append({"id": 1, "team": "near", "cxy": [near_x, near_y]})
    if far_y is not None:
        players.append({"id": 2, "team": "far", "cxy": [far_x, far_y]})
    return {"f": f, "players": players}


def test_in_kitchen_and_depth_from_baseline():
    assert _in_kitchen("near", (3.0, 7.0))       # just inside near-side NVZ
    assert not _in_kitchen("near", (3.0, 9.0))   # past the kitchen, still their side
    assert _in_kitchen("far", (3.0, 6.0))
    assert not _in_kitchen("far", (3.0, 13.0))   # near the NEAR baseline, nowhere near the far side's NVZ
    assert _depth_from_baseline("near", 13.4112) == 0.0   # standing on own baseline
    assert _depth_from_baseline("far", 0.0) == 0.0
    assert _depth_from_baseline("near", 0.0) > _depth_from_baseline("near", 10.0)  # near baseline is y=COURT_L


def test_aggregate_frames_kitchen_arrival_true_once_reached():
    entries = [
        _frame(0, near_y=12.5, far_y=1.0),   # both deep, no one in kitchen
        _frame(1, near_y=6.9, far_y=1.0),    # near player steps into their kitchen
        _frame(2, near_y=12.0, far_y=1.0),   # near player retreats — arrival already recorded
    ]
    kitchen, pressure, teams = aggregate_frames(entries)
    assert kitchen == {1: True, 2: False}
    assert teams == {1: "near", 2: "far"}
    assert pressure[2] == _depth_from_baseline("far", 1.0)  # far player never moved


def test_infer_serving_team_picks_deepest_player_at_serve_formation():
    # Near team both deep (server + partner just behind baseline); far team
    # one up near kitchen, one deep — a valid is_serve_formation snapshot.
    entries = [{"f": 0, "players": [
        {"id": 1, "team": "near", "cxy": [1.5, 13.0]},
        {"id": 3, "team": "near", "cxy": [4.5, 12.0]},
        {"id": 2, "team": "far", "cxy": [1.5, 1.0]},
        {"id": 4, "team": "far", "cxy": [4.5, 6.5]},
    ]}]
    assert infer_serving_team(entries) == "near"


def test_infer_serving_team_none_when_no_formation_found():
    entries = [_frame(0, near_y=6.9, far_y=6.9)]  # both at the net, no formation
    assert infer_serving_team(entries) is None


def test_kitchen_arrival_pct_100_when_serving_and_always_in_kitchen():
    """A player whose team serves every rally and who reaches the kitchen in
    every one of those rallies must show 100% arrival."""
    serve_snapshot = [
        {"id": 1, "team": "near", "cxy": [1.5, 13.0]},
        {"id": 3, "team": "near", "cxy": [4.5, 12.0]},
        {"id": 2, "team": "far", "cxy": [1.5, 1.0]},
        {"id": 4, "team": "far", "cxy": [4.5, 6.5]},
    ]  # near team serving in every rally (deepest player is near)
    rallies = []
    for i in range(3):
        entries = [
            {"f": 0, "players": serve_snapshot},
            {"f": 1, "players": [{"id": 3, "team": "near", "cxy": [4.5, 7.0]},  # id 3 reaches kitchen
                                  {"id": 4, "team": "far", "cxy": [4.5, 6.0]}]},  # id 4 never does (far kitchen would need y in [5.57,6.71])
        ]
        rallies.append((f"rally{i}", entries))

    result = _aggregate_rallies(rallies)
    players = result["players"]
    assert players[3]["kitchen_arrival_pct"] == 1.0
    assert players[3]["kitchen_serving_rallies"] == 3
    # id 1 never revisited after frame 0 (stayed at the baseline, never in kitchen)
    assert players[1]["kitchen_arrival_pct"] == 0.0


def test_forward_pressure_averages_across_all_rallies_regardless_of_serve():
    rallies = [
        ("r1", [_frame(0, near_y=13.0)]),   # depth 0
        ("r2", [_frame(0, near_y=3.0)]),    # depth = COURT_L - 3.0
    ]
    result = _aggregate_rallies(rallies)
    p1 = result["players"][1]
    assert p1["forward_pressure_rallies"] == 2
    expected = (_depth_from_baseline("near", 13.0) + _depth_from_baseline("near", 3.0)) / 2
    assert abs(p1["forward_pressure_avg_m"] - expected) < 1e-9


if __name__ == "__main__":
    import sys
    import types

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

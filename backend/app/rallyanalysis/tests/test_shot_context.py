"""Shot-type classification + rally-outcome annotation (shot_context.py) —
pure functions, no DB. Also pins that report_stats' zone helper is the same
function (moved, not forked)."""
from __future__ import annotations

from ...cv.court import KITCHEN, NET_Y
from ..report_stats import _zone_for_court_y
from ..shot_context import (MIN_TEMPO_SAMPLES, ZONE_BACKCOURT, ZONE_KITCHEN, ZONE_MIDCOURT,
                             annotate_rally_context, classify_shot_types, zone_for_court_y)


# ------------------------------------------------------------------- zones

def test_zone_cutoffs_match_court_constants():
    assert zone_for_court_y(NET_Y) == ZONE_KITCHEN
    assert zone_for_court_y(NET_Y - KITCHEN + 0.001) == ZONE_KITCHEN
    assert zone_for_court_y(NET_Y - KITCHEN - 0.01) == ZONE_MIDCOURT
    assert zone_for_court_y(0.0) == ZONE_BACKCOURT
    assert zone_for_court_y(2 * NET_Y) == ZONE_BACKCOURT


def test_report_stats_zone_helper_is_the_same_function():
    # Moved to shot_context, re-exported — never two diverging copies.
    assert _zone_for_court_y is zone_for_court_y


# ----------------------------------------------------------- shot typing

def _c(cid, pid=1, court_y=NET_Y, speed=100.0):
    return {"contact_id": cid, "player_id": pid, "court_y": court_y,
            "hand_speed_peak": speed}


def test_classify_tempo_is_relative_to_the_players_own_median():
    # Player 1: four fast-ish shots + one much slower one. The slow one must
    # be "soft" BY THIS PLAYER'S standard, regardless of absolute px/s.
    contacts = [_c(f"a{i}", speed=100.0 + i) for i in range(MIN_TEMPO_SAMPLES)]
    contacts.append(_c("slow", court_y=0.0, speed=10.0))
    types = classify_shot_types(contacts)
    assert types["slow"] == f"{ZONE_BACKCOURT}_soft"
    assert types[f"a{MIN_TEMPO_SAMPLES - 1}"] == f"{ZONE_KITCHEN}_fast"


def test_classify_two_players_get_independent_medians():
    # Far-side players have smaller pixel speeds (camera distance) — the same
    # absolute speed must be able to read fast for one player, soft for
    # another.
    near = [_c(f"n{i}", pid=1, speed=200.0 + i) for i in range(MIN_TEMPO_SAMPLES)]
    far = [_c(f"f{i}", pid=2, speed=20.0 + i) for i in range(MIN_TEMPO_SAMPLES)]
    types = classify_shot_types(near + far)
    assert types[f"n0"].endswith("_soft")   # 200 is slow for player 1
    assert types[f"f{MIN_TEMPO_SAMPLES - 1}"].endswith("_fast")  # 23 is fast for player 2


def test_classify_too_few_speed_samples_yields_zone_only_type():
    contacts = [_c("only", court_y=0.0, speed=500.0)]
    assert classify_shot_types(contacts)["only"] == ZONE_BACKCOURT  # no "_fast"/"_soft" invented


def test_classify_no_court_position_yields_no_type():
    contacts = [_c(f"a{i}") for i in range(MIN_TEMPO_SAMPLES)] + [_c("nopos", court_y=None)]
    types = classify_shot_types(contacts)
    assert "nopos" not in types


# ------------------------------------------------------------ rally context

def _rally(rid="r1", start=0, end=300):
    return {"id": rid, "start_frame": start, "end_frame": end}


def test_annotate_orders_indexes_and_flags_the_last_shot():
    contacts = [
        {"contact_id": "third", "rally_id": "r1", "frame_idx": 200},
        {"contact_id": "first", "rally_id": "r1", "frame_idx": 10},
        {"contact_id": "second", "rally_id": "r1", "frame_idx": 100},
    ]
    ann = {c["contact_id"]: c for c in annotate_rally_context(contacts, [_rally()], fps=30.0)}
    assert ann["first"]["rally_shot_index"] == 1
    assert ann["second"]["rally_shot_index"] == 2
    assert ann["third"]["rally_shot_index"] == 3
    assert ann["third"]["is_rally_ending"] is True
    assert ann["first"]["is_rally_ending"] is False
    assert all(c["rally_shot_count"] == 3 for c in ann.values())


def test_annotate_seconds_until_rally_end_from_rally_end_frame():
    contacts = [{"contact_id": "c1", "rally_id": "r1", "frame_idx": 240}]
    ann = annotate_rally_context(contacts, [_rally(end=300)], fps=30.0)
    assert ann[0]["seconds_until_rally_end"] == 2.0


def test_annotate_unknown_rally_passes_through_without_context():
    contacts = [{"contact_id": "orphan", "rally_id": "gone", "frame_idx": 5}]
    ann = annotate_rally_context(contacts, [_rally()], fps=30.0)
    assert ann[0]["contact_id"] == "orphan"
    assert "rally_shot_index" not in ann[0]     # not dropped, not fabricated


def test_annotate_does_not_mutate_inputs():
    contacts = [{"contact_id": "c1", "rally_id": "r1", "frame_idx": 10}]
    annotate_rally_context(contacts, [_rally()], fps=30.0)
    assert "rally_shot_index" not in contacts[0]

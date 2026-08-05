"""Positioning-focus metrics: absolute band math, per-metric gates, span
moment extraction and pseudo-contact anchoring. Pure — no DB, no cv2, no
anthropic — same pattern as test_coaching.py."""
from __future__ import annotations

from ...cv.court import COURT_L, KITCHEN, NET_Y
from ..positioning import (BANDS, CATEGORY_METRICS, CENTER_X, METRIC_FIELDS,
                            MIN_BOTH_VISIBLE_FRAMES, MIN_SPAN_SAMPLES,
                            NEVER_ARRIVED_CAP_S, _abs_dev, _approach_moments,
                            _dwell_moment, _recovery_moment, _score, _soft_dev,
                            _stagger_moments, band_for, build_vectors,
                            contact_metrics, span_moments)

MID_Y_NEAR = NET_Y + KITCHEN + 1.0   # midcourt band, near side
RALLY = {"id": "r1", "start_frame": 0, "end_frame": 300}


def _raw(**overrides):
    raw = {f: None for f in METRIC_FIELDS}
    raw["contact_id"] = overrides.pop("contact_id", "c1")
    raw.update(overrides)
    return raw


# ------------------------------------------------------------- band math

def test_abs_dev_zero_inside_band_signed_excess_outside():
    band = (0.0, 2.0, 1.0)
    assert _abs_dev(1.0, band) == 0.0
    assert _abs_dev(3.5, band) == 1.5     # above hi, in scale units
    assert _abs_dev(-0.5, band) == -0.5   # below lo -> negative
    assert _abs_dev(None, band) is None   # no reading -> no comparison, never 0


def test_soft_dev_is_continuous_inside_the_band():
    band = (0.0, 2.0, 1.0)
    assert _soft_dev(1.0, band) == 0.0    # at band center
    assert _soft_dev(2.0, band) == 1.0    # at the edge — nonzero, unlike _abs_dev
    assert _soft_dev(None, band) is None


def test_singles_band_override_widens_kitchen_distance_only():
    assert band_for("kitchen_distance_at_contact_m", "singles")[1] == 3.0
    assert band_for("kitchen_distance_at_contact_m", "doubles") == BANDS["kitchen_distance_at_contact_m"]
    assert band_for("kitchen_distance_at_contact_m", None) == BANDS["kitchen_distance_at_contact_m"]
    assert band_for("return_depth_m", "singles") == BANDS["return_depth_m"]


# --------------------------------------------------- contact-anchored gates

def test_kitchen_distance_gated_to_established_rally_phase():
    at_line = {"court_y": NET_Y + KITCHEN, "team": "near"}
    assert contact_metrics({**at_line, "rally_shot_index": 5}, None)["kitchen_distance_at_contact_m"] == 0.0
    # Shots 1-4 are legitimately deep — never measured, not measured-and-forgiven.
    assert contact_metrics({**at_line, "rally_shot_index": 4}, None)["kitchen_distance_at_contact_m"] is None
    assert contact_metrics({**at_line, "rally_shot_index": None}, None)["kitchen_distance_at_contact_m"] is None
    assert contact_metrics({"court_y": None, "team": "near", "rally_shot_index": 6},
                            None)["kitchen_distance_at_contact_m"] is None


def test_kitchen_distance_mirrors_by_team_and_goes_negative_inside_kitchen():
    deep_far = contact_metrics({"court_y": 0.5, "team": "far", "rally_shot_index": 6}, None)
    assert deep_far["kitchen_distance_at_contact_m"] > 3.5   # far baseline camp
    inside = contact_metrics({"court_y": NET_Y + 1.0, "team": "near", "rally_shot_index": 6}, None)
    assert inside["kitchen_distance_at_contact_m"] < 0       # inside the kitchen: fine, dev 0
    assert _abs_dev(inside["kitchen_distance_at_contact_m"],
                     BANDS["kitchen_distance_at_contact_m"]) == 0.0


def test_return_depth_only_on_the_return_shot():
    ret = {"court_y": COURT_L - 1.0, "team": "near", "rally_shot_index": 2}
    assert contact_metrics(ret, None)["return_depth_m"] == 1.0
    assert contact_metrics({**ret, "rally_shot_index": 3}, None)["return_depth_m"] is None
    far_ret = contact_metrics({"court_y": 4.0, "team": "far", "rally_shot_index": 2}, None)
    assert far_ret["return_depth_m"] == 4.0  # metres inside own (far, y=0) baseline


# ------------------------------------------------------------ span moments

def _near_samples(frames, y=MID_Y_NEAR, x=3.0):
    return [(f, x, y, "near") for f in frames]


def test_dwell_flags_longest_midcourt_run_and_anchors_mid_run():
    samples = _near_samples(range(0, 91, 3))  # 3s parked in midcourt
    m = _dwell_moment("v", RALLY, 1, samples, 30.0, 1.0)
    assert m["span"]["value"] == 3.0
    assert m["kind"] == "span"
    assert m["contact_id"] == f"pos_r1_1_transition_dwell_s_{m['frame_idx']}"
    assert 30 <= m["frame_idx"] <= 60  # anchored inside the run, not at an edge
    assert m["timestamp_s"] == m["frame_idx"] / 30.0


def test_dwell_gap_breaks_the_run_instead_of_bridging_unseen_time():
    samples = _near_samples(range(0, 46, 3)) + _near_samples(range(90, 136, 3))
    m = _dwell_moment("v", RALLY, 1, samples, 30.0, 1.0)
    assert m["span"]["value"] == 1.5  # two 1.5s runs, never one fabricated 4.5s


def test_dwell_requires_min_samples_and_emits_in_band_moments():
    assert _dwell_moment("v", RALLY, 1, _near_samples(range(MIN_SPAN_SAMPLES - 1)), 30.0, 1.0) is None
    # A clean player (never in midcourt) still emits a dev-0 moment — feeds
    # praise exemplars and general aggregates, mirroring form's score-everything.
    clean = _near_samples(range(0, 60, 2), y=NET_Y + KITCHEN - 0.1)  # at the line
    m = _dwell_moment("v", RALLY, 1, clean, 30.0, 1.0)
    assert m is not None and m["span"]["value"] == 0.0


def test_approach_measured_for_returning_team_only_with_never_arrival_cap():
    contacts = [{"contact_id": "ret", "rally_shot_index": 2, "team": "near",
                 "player_id": 1, "frame_idx": 30}]
    # Returner (near) parked deep the whole rally; server side (far) also deep.
    near = _near_samples(range(0, 301, 3), y=COURT_L - 0.5)
    far = [(f, 3.0, 0.5, "far") for f in range(0, 301, 3)]
    ms = _approach_moments("v", RALLY, {1: near, 2: far}, contacts, 30.0, {1: 1.0, 2: 1.0})
    assert [m["player_id"] for m in ms] == [1]  # serving team correctly stays back
    assert ms[0]["span"]["value"] == NEVER_ARRIVED_CAP_S  # capped, not rally-length
    assert ms[0]["frame_idx"] == 30 + round(3.5 * 30)     # anchored where they should be in


def test_approach_skipped_when_rally_ends_too_soon_after_return():
    short_rally = {"id": "r2", "start_frame": 0, "end_frame": 100}
    contacts = [{"contact_id": "ret", "rally_shot_index": 2, "team": "near",
                 "player_id": 1, "frame_idx": 30}]
    near = _near_samples(range(0, 101, 3), y=COURT_L - 0.5)
    assert _approach_moments("v", short_rally, {1: near}, contacts, 30.0, {1: 1.0}) == []


def test_approach_arrival_time_measured_from_return_contact():
    contacts = [{"contact_id": "ret", "rally_shot_index": 2, "team": "near",
                 "player_id": 1, "frame_idx": 30}]
    # Deep until frame 90, at the kitchen line from frame 90 on -> 2.0s.
    samples = (_near_samples(range(0, 90, 3), y=COURT_L - 0.5)
               + _near_samples(range(90, 301, 3), y=NET_Y + KITCHEN))
    ms = _approach_moments("v", RALLY, {1: samples}, contacts, 30.0, {1: 1.0})
    assert len(ms) == 1 and ms[0]["span"]["value"] == 2.0


def test_recovery_singles_only_and_measured_at_opponents_next_contact():
    contacts = [{"contact_id": "wide", "player_id": 1, "team": "near",
                 "court_x": CENTER_X + 2.5, "frame_idx": 10},
                {"contact_id": "opp", "player_id": 2, "team": "far",
                 "court_x": CENTER_X, "frame_idx": 40}]
    parked_wide = [(f, CENTER_X + 2.5, NET_Y + 3, "near") for f in range(0, 60, 2)]
    assert _recovery_moment("v", RALLY, 1, parked_wide, contacts, 30.0, "doubles", 1.0) is None
    assert _recovery_moment("v", RALLY, 1, parked_wide, contacts, 30.0, None, 1.0) is None
    m = _recovery_moment("v", RALLY, 1, parked_wide, contacts, 30.0, "singles", 1.0)
    assert abs(m["span"]["value"] - 2.5) < 1e-9
    assert m["frame_idx"] == 40  # anchored at the opponent's strike
    # Recovered back to center by the opponent's contact -> dev 0, still emitted.
    recovered = ([(f, CENTER_X + 2.5, NET_Y + 3, "near") for f in range(0, 20, 2)]
                 + [(f, CENTER_X, NET_Y + 3, "near") for f in range(20, 60, 2)])
    m = _recovery_moment("v", RALLY, 1, recovered, contacts, 30.0, "singles", 1.0)
    assert m["span"]["value"] == 0.0


def test_stagger_doubles_only_flags_back_player_needs_both_visible():
    up = [(f, 2.0, NET_Y + KITCHEN, "near") for f in range(0, 90)]
    back = [(f, 4.0, COURT_L - 0.5, "near") for f in range(0, 90)]
    ms = _stagger_moments("v", RALLY, {1: up, 2: back}, 30.0, "doubles", {1: 1.0, 2: 1.0})
    assert len(ms) == 1
    assert ms[0]["player_id"] == 2                   # the back player owns the tip
    assert ms[0]["span"]["value"] > 4.0
    assert _stagger_moments("v", RALLY, {1: up, 2: back}, 30.0, "singles", {}) == []
    # Graceful degrade: partners never visible together long enough -> nothing.
    sparse_back = [(f, 4.0, COURT_L - 0.5, "near") for f in range(0, MIN_BOTH_VISIBLE_FRAMES - 1)]
    assert _stagger_moments("v", RALLY, {1: up, 2: sparse_back}, 30.0, "doubles", {}) == []


def test_stagger_transient_split_never_flags():
    # Partners together except a 0.5s poach split — rolling minimum holds low.
    a = [(f, 2.0, NET_Y + KITCHEN, "near") for f in range(0, 90)]
    b = [(f, 4.0, NET_Y + KITCHEN + (3.0 if 40 <= f < 55 else 0.3), "near") for f in range(0, 90)]
    ms = _stagger_moments("v", RALLY, {1: a, 2: b}, 30.0, "doubles", {})
    assert not ms or ms[0]["span"]["value"] < 1.0


# ------------------------------------------------------- scoring / vectors

def test_score_picks_category_of_worst_metric_and_lists_all_nulls():
    vec = _score(_raw(transition_dwell_s=5.0), None)
    assert vec.category == "transition"
    assert vec.deviation_score == 2.5
    assert vec.deviating_metrics == ["transition_dwell_s"]
    assert set(vec.low_confidence_metrics) == set(METRIC_FIELDS) - {"transition_dwell_s"}
    assert vec.no_comparison_metrics == [] and vec.mixed_baseline_metrics == []
    assert vec.shot_type is None
    assert _score(_raw(), None) is None  # nothing real anywhere -> dropped


def test_category_metrics_cover_every_metric_field_exactly_once():
    covered = [f for fields in CATEGORY_METRICS.values() for f in fields]
    assert sorted(covered) == sorted(METRIC_FIELDS)


def test_build_vectors_combines_contacts_and_span_moments():
    contacts = [{"contact_id": "c1", "rally_id": "r1", "player_id": 1, "team": "near",
                 "court_y": COURT_L - 2.0, "rally_shot_index": 6, "frame_idx": 100,
                 "timestamp_s": 100 / 30.0, "confidence": 0.9, "is_serve": False}]
    samples = {1: _near_samples(range(0, 91, 3))}  # 3s midcourt dwell
    vm = build_vectors(contacts, [RALLY], {"r1": samples}, 30.0, None, "v")
    by_id = {c["contact_id"]: vec for vec, c in vm}
    assert "c1" in by_id and by_id["c1"].category == "net_position"
    span_ids = [cid for cid in by_id if cid.startswith("pos_r1_1_transition_dwell_s_")]
    assert len(span_ids) == 1
    assert by_id[span_ids[0]].category == "transition"


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

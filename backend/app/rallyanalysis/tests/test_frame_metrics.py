"""Continuous frame-by-frame metric tracking (frame_metrics.py): absolute
per-frame targets, spike grouping (incl. gap-breaking), dominant-metric
attribution, average-vs-target, established-phase/singles/doubles gating.
Pure — no DB. Same pattern as test_positioning.py."""
from __future__ import annotations

from ...cv.court import KITCHEN, NET_Y
from ..frame_metrics import (CENTER_X, ELBOW_HITTING_SIDE_WINDOW_S, FORM_CATEGORY,
                              FORM_METRIC_THRESHOLDS, FORM_TARGETS, GAP_BREAK_S,
                              POSITIONING_CATEGORY, _contact_elbow_readings, _hitting_side,
                              average_context, build_frame_series, detect_spikes,
                              established_from_frame, form_frame_values, frame_deviations,
                              frame_max_excess, frame_sum, positioning_frame_values,
                              targets_and_category)

FPS = 30.0


def _series(frame, knee, elbow=None):
    values = {"knee_bend_deg": knee, "elbow_extension_deg": elbow}
    devs = frame_deviations(values, FORM_TARGETS, None)
    return {"frame": frame, "values": values, "devs": devs}


# ------------------------------------------------------------- band math

def test_frame_deviations_zero_inside_band_signed_excess_outside():
    devs = frame_deviations({"knee_bend_deg": 180.0, "elbow_extension_deg": None},
                             FORM_TARGETS, None)
    assert devs["knee_bend_deg"] == 2.0          # (180-168)/6
    assert devs["elbow_extension_deg"] is None   # no reading -> no comparison


def test_frame_sum_ignores_none_and_sums_absolute_values():
    assert frame_sum({"knee_bend_deg": 2.0, "elbow_extension_deg": -1.5}) == 3.5
    assert frame_sum({"knee_bend_deg": None, "elbow_extension_deg": None}) == 0.0


def test_frame_max_excess_is_the_worst_fraction_over_its_own_threshold():
    # knee at exactly its own bar -> excess 1.0; elbow further under its own
    # (larger) bar -> smaller fraction; max_excess picks the worse of the two.
    devs = {"knee_bend_deg": 1.5, "elbow_extension_deg": -0.75}
    assert frame_max_excess(devs, FORM_METRIC_THRESHOLDS) == 1.0
    assert frame_max_excess({"knee_bend_deg": None, "elbow_extension_deg": None},
                             FORM_METRIC_THRESHOLDS) == 0.0
    assert frame_max_excess({}, FORM_METRIC_THRESHOLDS) == 0.0


def test_targets_and_category_selects_by_focus():
    t, c = targets_and_category("positioning")
    assert set(t) == {"kitchen_distance_m", "transition_zone_occupancy",
                       "lateral_offset_m", "partner_depth_stagger_m"}
    assert c is POSITIONING_CATEGORY
    t2, c2 = targets_and_category("form")
    assert set(t2) == {"knee_bend_deg", "elbow_extension_deg"}
    assert c2 is FORM_CATEGORY


# ------------------------------------------------------------ spike math
# (2026-08-05: detection is now INDEPENDENT per-metric — detect_spikes takes
# a {metric: own_threshold} dict, not a single shared bar; see
# FORM_METRIC_THRESHOLDS.)

def test_detect_spikes_groups_consecutive_over_threshold_frames():
    series = [_series(f, 180.0) for f in range(100, 105)]  # knee dev 2.0 > its own 1.5 bar
    spikes = detect_spikes(series, FORM_METRIC_THRESHOLDS, fps=FPS, category_map=FORM_CATEGORY)
    assert len(spikes) == 1
    assert spikes[0]["start_frame"] == 100 and spikes[0]["end_frame"] == 104
    assert spikes[0]["dominant_metric"] == "knee_bend_deg"
    assert spikes[0]["category"] == "footwork"
    assert spikes[0]["values_at_peak"]["knee_bend_deg"] == 180.0


def test_detect_spikes_gap_larger_than_gap_break_splits_the_run():
    series = [_series(f, 180.0) for f in range(0, 5)]
    gap = int(GAP_BREAK_S * FPS) + 5
    series += [_series(5 + gap, 180.0)]
    spikes = detect_spikes(series, FORM_METRIC_THRESHOLDS, fps=FPS, category_map=FORM_CATEGORY)
    assert len(spikes) == 2


def test_detect_spikes_gap_within_tolerance_bridges_the_run():
    series = [_series(f, 180.0) for f in range(0, 5)]
    gap = int(GAP_BREAK_S * FPS) - 2
    series += [_series(5 + gap, 180.0)]
    spikes = detect_spikes(series, FORM_METRIC_THRESHOLDS, fps=FPS, category_map=FORM_CATEGORY)
    assert len(spikes) == 1


def test_detect_spikes_below_threshold_never_spikes():
    series = [_series(f, 130.0) for f in range(0, 10)]
    assert detect_spikes(series, FORM_METRIC_THRESHOLDS, fps=FPS, category_map=FORM_CATEGORY) == []


def test_detect_spikes_dominant_metric_is_the_worst_fraction_over_its_own_bar():
    # Both clear their own bar (knee frac 2.0/1.5=1.33, elbow frac
    # 1.1/1.0=1.10), but knee is further past its own line PROPORTIONALLY
    # -> knee wins even though elbow's raw |deviation| alone isn't tiny.
    devs = frame_deviations({"knee_bend_deg": 180.0, "elbow_extension_deg": 48.5},
                             FORM_TARGETS, None)
    series = [{"frame": 0, "values": {"knee_bend_deg": 180.0, "elbow_extension_deg": 48.5},
               "devs": devs}]
    spikes = detect_spikes(series, FORM_METRIC_THRESHOLDS, fps=FPS, category_map=FORM_CATEGORY)
    assert len(spikes) == 1
    assert spikes[0]["dominant_metric"] == "knee_bend_deg"


def test_detect_spikes_moderate_multi_metric_pileup_no_longer_flags():
    # THE FIX this revision exists for: two metrics each individually UNDER
    # their own threshold, whose SUM would have crossed the old shared bar
    # (1.0), must NOT spike now that detection is independent per-metric.
    devs = frame_deviations({"knee_bend_deg": 173.4, "elbow_extension_deg": 51.5},
                             FORM_TARGETS, None)
    assert abs(devs["knee_bend_deg"]) < FORM_METRIC_THRESHOLDS["knee_bend_deg"]
    assert abs(devs["elbow_extension_deg"]) < FORM_METRIC_THRESHOLDS["elbow_extension_deg"]
    assert frame_sum(devs) > 1.0  # would have crossed the retired shared bar
    series = [{"frame": 0, "values": {"knee_bend_deg": 173.4, "elbow_extension_deg": 51.5},
               "devs": devs}]
    assert detect_spikes(series, FORM_METRIC_THRESHOLDS, fps=FPS, category_map=FORM_CATEGORY) == []


def test_detect_spikes_one_metric_alone_clearing_its_bar_still_flags():
    # A single metric clearing its OWN bar still spikes regardless of what
    # any other metric is doing (independent, not a joint requirement).
    devs = frame_deviations({"knee_bend_deg": 180.0, "elbow_extension_deg": 60.0},
                             FORM_TARGETS, None)
    assert abs(devs["elbow_extension_deg"]) < FORM_METRIC_THRESHOLDS["elbow_extension_deg"]
    series = [{"frame": 0, "values": {"knee_bend_deg": 180.0, "elbow_extension_deg": 60.0},
               "devs": devs}]
    spikes = detect_spikes(series, FORM_METRIC_THRESHOLDS, fps=FPS, category_map=FORM_CATEGORY)
    assert len(spikes) == 1 and spikes[0]["dominant_metric"] == "knee_bend_deg"


# --------------------------------------------------------- average-vs-target

def test_average_context_flags_a_session_average_outside_the_band():
    series = [_series(f, 180.0) for f in range(10)]
    ctx = average_context(series, focus="form", match_type=None)
    assert ctx["footwork"]["avg_value"] == 180.0
    assert ctx["footwork"]["deviation"] == 2.0
    assert ctx["footwork"]["flagged_fraction"] == 1.0
    assert ctx["footwork"]["flagged_avg_deviation"] == 2.0
    assert ctx["footwork"]["impact_score"] == 2.0
    assert ctx["footwork"]["n_frames"] == 10


def test_average_context_in_band_average_yields_zero_deviation_and_impact():
    series = [_series(f, 130.0) for f in range(10)]
    ctx = average_context(series, focus="form", match_type=None)
    assert ctx["footwork"]["deviation"] == 0.0
    assert ctx["footwork"]["flagged_fraction"] == 0.0
    assert ctx["footwork"]["flagged_avg_deviation"] is None
    assert ctx["footwork"]["impact_score"] == 0.0


def test_average_context_absent_metric_is_simply_missing_not_zero_filled():
    series = [_series(f, 180.0) for f in range(10)]  # elbow always None
    ctx = average_context(series, focus="form", match_type=None)
    assert "contact_point" not in ctx


def test_average_context_impact_score_survives_dilution_the_plain_average_cannot():
    # A metric flagged only 10% of the session (locked-straight knee) buried
    # in 90% normal frames: the plain average stays comfortably in-band
    # (deviation 0.0), but impact_score must still surface it as real via
    # flagged_fraction/flagged_avg_deviation -- this is the whole reason
    # impact_score exists instead of ranking categories by plain deviation.
    series = ([_series(f, 130.0) for f in range(90)]
              + [_series(f, 180.0) for f in range(90, 100)])
    ctx = average_context(series, focus="form", match_type=None)
    footwork = ctx["footwork"]
    assert footwork["deviation"] == 0.0
    assert abs(footwork["flagged_fraction"] - 0.10) < 1e-9
    assert footwork["flagged_avg_deviation"] == 2.0
    assert footwork["impact_score"] > 0.15


def test_average_context_category_pick_uses_impact_not_plain_deviation():
    # footwork: never flagged, moderately elevated average (deviation
    # nonzero but impact 0). contact_point: flagged a chunk of the session,
    # clearly worse by impact despite a smaller plain-average deviation.
    # Picking by plain |deviation| would wrongly favor footwork; impact_score
    # must not.
    footwork_series = [{"frame": f, "values": {"knee_bend_deg": 167.0, "elbow_extension_deg": None},
                         "devs": frame_deviations({"knee_bend_deg": 167.0, "elbow_extension_deg": None},
                                                   FORM_TARGETS, None)} for f in range(50)]
    contact_series = ([{"frame": f, "values": {"knee_bend_deg": None, "elbow_extension_deg": 120.0},
                         "devs": frame_deviations({"knee_bend_deg": None, "elbow_extension_deg": 120.0},
                                                   FORM_TARGETS, None)} for f in range(40)]
                       + [{"frame": f, "values": {"knee_bend_deg": None, "elbow_extension_deg": 40.0},
                           "devs": frame_deviations({"knee_bend_deg": None, "elbow_extension_deg": 40.0},
                                                     FORM_TARGETS, None)} for f in range(40, 50)])
    ctx = average_context(footwork_series + contact_series, focus="form", match_type=None)
    assert ctx["footwork"]["impact_score"] == 0.0
    assert ctx["contact_point"]["impact_score"] > 0.0


# ------------------------------------------------------- positioning gating

def test_established_from_frame_requires_the_gating_shot_index():
    assert established_from_frame([{"frame_idx": 40, "rally_shot_index": 5}]) == 40
    assert established_from_frame([{"frame_idx": 40, "rally_shot_index": 4}]) is None
    assert established_from_frame([]) is None


def test_positioning_frame_values_gates_lateral_offset_to_singles_only():
    doubles = positioning_frame_values(cx=CENTER_X + 2.0, cy=NET_Y + 5.0, team="near",
                                        established=True, partner_cy=None, match_type="doubles")
    singles = positioning_frame_values(cx=CENTER_X + 2.0, cy=NET_Y + 5.0, team="near",
                                        established=True, partner_cy=None, match_type="singles")
    assert doubles["lateral_offset_m"] is None
    assert singles["lateral_offset_m"] == 2.0


def test_positioning_frame_values_kitchen_and_transition_gated_to_established_phase():
    early = positioning_frame_values(cx=CENTER_X, cy=NET_Y + KITCHEN + 1.0, team="near",
                                      established=False, partner_cy=None, match_type=None)
    late = positioning_frame_values(cx=CENTER_X, cy=NET_Y + KITCHEN + 1.0, team="near",
                                     established=True, partner_cy=None, match_type=None)
    assert early["kitchen_distance_m"] is None and early["transition_zone_occupancy"] is None
    assert late["kitchen_distance_m"] is not None
    assert late["transition_zone_occupancy"] == 1.0  # midcourt band


def test_positioning_frame_values_partner_stagger_needs_a_partner():
    solo = positioning_frame_values(cx=CENTER_X, cy=NET_Y + 3.0, team="near",
                                     established=True, partner_cy=None, match_type="doubles")
    with_partner = positioning_frame_values(cx=CENTER_X, cy=NET_Y + 3.0, team="near",
                                             established=True, partner_cy=NET_Y + 7.0,
                                             match_type="doubles")
    assert solo["partner_depth_stagger_m"] is None
    assert with_partner["partner_depth_stagger_m"] == 4.0


# --------------------------------------------------------- build_frame_series

def _pdict(knee_kp_offset=0.0):
    kp = [[0.0, 0.0]] * 17
    kp[11], kp[13], kp[15] = [100.0, 100.0], [100.0, 150.0 + knee_kp_offset], [100.0, 200.0]
    kp[12], kp[14], kp[16] = [140.0, 100.0], [140.0, 150.0 + knee_kp_offset], [140.0, 200.0]
    return kp


def test_build_frame_series_excludes_frames_in_the_given_ranges():
    entries = [{"f": f, "players": [{"id": 1, "kp": _pdict()}]} for f in range(0, 30)]
    series_by_pid = build_frame_series(entries, [], FPS, None, "form",
                                        exclude_frame_ranges=[(5, 15)])
    frames = {fs["frame"] for fs in series_by_pid[1]}
    assert not (frames & set(range(5, 16)))
    assert 0 in frames and 20 in frames


def test_build_frame_series_positioning_folds_doubles_partner_from_same_frame():
    entries = [{"f": 0, "players": [
        {"id": 1, "cxy": [CENTER_X, NET_Y + 2.0], "team": "near"},
        {"id": 2, "cxy": [CENTER_X, NET_Y + 6.0], "team": "near"},
    ]}]
    contacts = [{"frame_idx": 0, "rally_shot_index": 5}]  # established immediately
    series_by_pid = build_frame_series(entries, contacts, FPS, "doubles", "positioning")
    assert series_by_pid[1][0]["values"]["partner_depth_stagger_m"] == 4.0
    assert series_by_pid[2][0]["values"]["partner_depth_stagger_m"] == 4.0


# ------------------------------------------- contact-anchored elbow (2026-08-06)
# Left arm ("resting"): fixed every frame, genuinely cramped geometry (~45deg
# — would flag under the old continuous worse-of-both-arms design). Right arm
# ("hitting"): wrist drifts across the window (real motion -> _hitting_side
# picks it), fully extended (~180deg, in-band) exactly at the contact frame.
CONTACT_FRAME = 10
# _kp_xy treats any x<=0 or y<=0 as "missing" (matches the overlay's own
# "no detection" convention), so every point here must stay positive.
L_SHOULDER_XY, L_ELBOW_XY, L_WRIST_XY = (100.0, 100.0), (110.0, 100.0), (105.0, 105.0)  # ~45deg, cramped
R_SHOULDER_XY, R_ELBOW_XY = (300.0, 100.0), (310.0, 100.0)                              # extended arm anchor


def _hitting_test_kp(f):
    kp = [[1.0, 1.0]] * 17
    kp[5], kp[7], kp[9] = list(L_SHOULDER_XY), list(L_ELBOW_XY), list(L_WRIST_XY)  # left: fixed, cramped
    kp[6], kp[8] = list(R_SHOULDER_XY), list(R_ELBOW_XY)
    kp[10] = [320.0 + 5.0 * (f - CONTACT_FRAME), 100.0]  # right wrist: drifts, extended at f==CONTACT_FRAME
    return kp


def _hitting_test_entries():
    return [{"f": f, "players": [{"id": 1, "kp": _hitting_test_kp(f)}]} for f in range(0, 21)]


def test_hitting_side_picks_the_wrist_that_actually_moved():
    entries = {e["f"]: e["players"][0] for e in _hitting_test_entries()}
    window = round(ELBOW_HITTING_SIDE_WINDOW_S * FPS)
    frames_by_index = {f: entries[f] for f in range(CONTACT_FRAME - window, CONTACT_FRAME + window + 1)
                        if f in entries}
    assert _hitting_side(frames_by_index) == "R"


def test_contact_elbow_readings_uses_the_hitting_arm_not_the_cramped_resting_one():
    entries = _hitting_test_entries()
    contacts = [{"player_id": 1, "frame_idx": CONTACT_FRAME}]
    readings = _contact_elbow_readings(entries, contacts, FPS)
    assert (1, CONTACT_FRAME) in readings
    # If this picked the resting (left) arm instead, it would read ~45deg
    # (cramped, flagged) — the whole bug this revision fixes.
    assert readings[(1, CONTACT_FRAME)] > 170.0


def test_contact_elbow_readings_absent_for_a_contact_with_no_pose_at_its_instant():
    entries = [e for e in _hitting_test_entries() if e["f"] != CONTACT_FRAME]
    contacts = [{"player_id": 1, "frame_idx": CONTACT_FRAME}]
    assert _contact_elbow_readings(entries, contacts, FPS) == {}


def test_build_frame_series_elbow_is_none_everywhere_except_at_the_contact_frame():
    # Pose (incl. the cramped resting left arm) is present on EVERY frame,
    # but elbow_extension_deg must be None everywhere except exactly the
    # contact frame -- this is the core of the fix: a normal bent-elbow
    # ready posture held for the rest of the rally must never register at
    # all anymore, not just fail to flag.
    entries = _hitting_test_entries()
    contacts = [{"player_id": 1, "frame_idx": CONTACT_FRAME}]
    series_by_pid = build_frame_series(entries, contacts, FPS, None, "form")
    by_frame = {fs["frame"]: fs["values"]["elbow_extension_deg"] for fs in series_by_pid[1]}
    assert by_frame[CONTACT_FRAME] is not None and by_frame[CONTACT_FRAME] > 170.0
    for f, v in by_frame.items():
        if f != CONTACT_FRAME:
            assert v is None, f"frame {f} should have no elbow reading, got {v}"


def test_form_frame_values_never_computes_elbow_directly():
    # form_frame_values is the CONTINUOUS per-frame path; elbow must always
    # come back None from it regardless of how cramped the pose looks --
    # only build_frame_series's contact-anchored injection may ever set it.
    pdict = {"kp": _hitting_test_kp(CONTACT_FRAME)}
    assert form_frame_values(pdict)["elbow_extension_deg"] is None


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

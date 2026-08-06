"""debug_timeline (coaching.debug_timeline / GET .../coaching-debug): the
frame-by-frame decision-timeline diagnostic (2026-08-05 redesign) — every
per-frame reading, the spikes detected from them, the flag threshold, and
which spikes were sent to the LLM. No LLM call. Must never let a serve's
swing window contribute a single frame to the continuous series.

Same real-temp-sqlite-DB fixture pattern as test_rally_source_default.py,
since this exercises the DB-touching orchestration layer
(_build_vectors_meta), not just pure functions (see test_frame_metrics.py
for those)."""
from __future__ import annotations

import pytest

from ... import db
from ...cv import overlays
from .. import coaching

VIDEO = "vid1"
FPS = 30.0
RALLY_END = 299

# Straight leg (hip-knee-ankle colinear) -> ~180 deg -> flags REFERENCE_BANDS-
# style continuous target; bent leg -> comfortably inside the acceptable band.
STRAIGHT_KNEE_KP = [[0.0, 0.0]] * 17
STRAIGHT_KNEE_KP[11], STRAIGHT_KNEE_KP[13], STRAIGHT_KNEE_KP[15] = [100.0, 100.0], [100.0, 150.0], [100.0, 200.0]
STRAIGHT_KNEE_KP[12], STRAIGHT_KNEE_KP[14], STRAIGHT_KNEE_KP[16] = [140.0, 100.0], [140.0, 150.0], [140.0, 200.0]

BENT_KNEE_KP = [[0.0, 0.0]] * 17
BENT_KNEE_KP[11], BENT_KNEE_KP[13], BENT_KNEE_KP[15] = [100.0, 100.0], [90.0, 150.0], [130.0, 180.0]
BENT_KNEE_KP[12], BENT_KNEE_KP[14], BENT_KNEE_KP[16] = [140.0, 100.0], [150.0, 150.0], [110.0, 180.0]

# Frames 0-279 are straight-knee; a serve sits at frame 10 (PRE_WINDOW_S=0.6s/
# POST_WINDOW_S=0.45s @ 30fps -> exclusion window [0,23]) so frames 0-23 must
# NEVER surface in the continuous series regardless of their (bad) value --
# if serve exclusion is broken, the detected spike would start at frame 0
# instead of frame 24. Frames 280-299 are bent-knee (in range), pulling the
# session average back toward acceptable but not all the way (still >81% bad).
BAD_UNTIL_FRAME = 279


def _kp_for_frame(f):
    return STRAIGHT_KNEE_KP if f <= BAD_UNTIL_FRAME else BENT_KNEE_KP


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, fps, uploaded_at) VALUES (?,?,?,?,?)",
            (VIDEO, "v.mp4", "/tmp/v.mp4", FPS, db.now()))
        con.execute(
            "INSERT INTO rally_labels (id, video_id, source, start_frame, end_frame)"
            " VALUES (?,?,?,?,?)",
            ("r1", VIDEO, "pass1", 0, RALLY_END))
        con.execute(
            "INSERT INTO jobs (id, video_id, kind, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (db.new_id(), VIDEO, "pass2", "done", 1.0, 1.0))
        con.execute(
            "INSERT INTO contact_events (contact_id, video_id, rally_id, frame_idx,"
            " timestamp_s, player_id, team, court_x, court_y, shot_side,"
            " hand_speed_peak, confidence, is_serve) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("serve1", VIDEO, "r1", 10, 10 / FPS, 1, "near", 3.0, 6.0, "left", 100.0, 0.9, 1))
    overlays.write_entries(VIDEO, "pass2", [
        {"f": f, "players": [{"id": 1, "box": [0, 0, 50, 200], "cxy": [3.0, 6.0], "team": "near",
                               "kp": _kp_for_frame(f)}]}
        for f in range(0, RALLY_END + 1)
    ])
    yield


def test_debug_timeline_never_lets_a_serve_window_produce_a_spike():
    result = coaching.debug_timeline(VIDEO)
    assert result["focus"] == "form"
    assert result["threshold"] == coaching.WORST_DEVIATION_THRESHOLD
    assert set(result["metric_fields"]) == {"knee_bend_deg", "elbow_extension_deg"}

    spikes = [s for s in result["shots"] if s["category"] == "footwork"]
    assert len(spikes) == 1
    spike = spikes[0]
    # The spike must start AFTER the serve window (frame 23), never at/before
    # it, even though frames 0-23 also carry the straight-knee reading.
    assert 24 <= spike["frame_idx"] <= BAD_UNTIL_FRAME
    assert spike["flagged"] is True
    assert spike["selected"] is True
    assert spike["metrics"]["knee_bend_deg"]["value"] == 180.0
    assert spike["duration_s"] > 0


def test_debug_timeline_frame_series_never_contains_a_serve_window_frame():
    result = coaching.debug_timeline(VIDEO)
    series = result["frame_series"]["Player 1"]
    frames_seen = {fs["frame_idx"] for fs in series}
    assert not (frames_seen & set(range(0, 24)))  # the whole serve window is absent
    assert 100 in frames_seen and 290 in frames_seen


def test_debug_timeline_frame_series_sum_crosses_threshold_only_outside_the_good_tail():
    result = coaching.debug_timeline(VIDEO)
    series = result["frame_series"]["Player 1"]
    by_frame = {fs["frame_idx"]: fs["sum"] for fs in series}
    assert by_frame[100] >= coaching.WORST_DEVIATION_THRESHOLD
    assert by_frame[290] == 0.0


def test_debug_timeline_general_context_averages_the_whole_session():
    result = coaching.debug_timeline(VIDEO)
    ctx = result["general_context"]["Player 1"]
    footwork = ctx["by_category"]["footwork"]
    # >81% of the (non-excluded) session is straight-knee -> the average
    # itself crosses the target band, not just individual frames.
    assert footwork["avg_value"] > 168.0
    assert footwork["deviation"] > 0
    assert ctx["strongest_negative_category"] == "footwork"


def test_debug_timeline_selected_shots_match_a_fresh_select_tip_candidates_call():
    # The endpoint's whole value is "selected" being trustworthy — assert it
    # against an independently-computed select_tip_candidates call rather
    # than hand-picked expectations, so the two can never silently drift.
    result = coaching.debug_timeline(VIDEO)
    vectors_meta, focus, *_ = coaching._build_vectors_meta(VIDEO)
    expected = coaching.select_tip_candidates(
        vectors_meta, limit=coaching.MAX_LLM_CANDIDATES,
        fill_to_limit=coaching.FILL_TO_LIMIT, category_cap=coaching.CATEGORY_CAP)
    expected_ids = {c["contact_id"] for c in expected}
    selected_ids = {s["contact_id"] for s in result["shots"] if s["selected"]}
    assert selected_ids == expected_ids


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

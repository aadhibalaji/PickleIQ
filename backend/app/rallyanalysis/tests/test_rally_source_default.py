"""Regression test for the rally_source="pass2" literal-default bug:
detect_contacts_for_video and generate_coaching_report used to default
rally_source to the literal string "pass2", which readiness.get_analyzable_rallies
treats as a direct `rally_labels WHERE source='pass2'` query -- bypassing the
human -> pass2 -> pass1 override fallback that only kicks in when rally_source
is None. Since Pass 2 never writes rally_labels rows anywhere in this codebase
(Pass 1 is the sole source of rally boundaries), that literal default made both
functions silently operate over zero rallies for every real video, whose
rally_labels only ever have source='pass1'.

Uses the same real-temp-sqlite-DB fixture pattern as test_readiness.py, since
this exercises the DB-touching orchestration layer, not just pure functions."""
from __future__ import annotations

import pytest

from ... import db
from ...cv import overlays
from .. import coaching
from ..coaching import generate_coaching_report
from ..contact_detect import detect_contacts_for_video

VIDEO = "vid1"
FPS = 30.0


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
            ("r1", VIDEO, "pass1", 0, 20))
        con.execute(
            "INSERT INTO jobs (id, video_id, kind, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (db.new_id(), VIDEO, "pass2", "done", 1.0, 1.0))
    # Pass 2 overlay covering the rally, with a clear wrist swing at frame 10
    # so detect_contacts_for_video has something to find.
    entries = []
    for f in range(0, 21):
        rw = [300.0, 50.0] if f == 10 else [50.0, 50.0]
        kp = [[100.0, 100.0]] * 17
        kp[9] = [50.0, 50.0]
        kp[10] = rw
        entries.append({"f": f, "players": [
            {"id": 1, "box": [0, 0, 50, 200], "cxy": [1.0, 3.0], "team": "near", "kp": kp},
        ]})
    overlays.write_entries(VIDEO, "pass2", entries)
    yield


def test_detect_contacts_for_video_finds_pass1_only_rallies_by_default():
    # This is the normal case for every real video: rally_labels only has
    # source='pass1' rows, never 'pass2' or 'human'. Leaving rally_source at
    # its default must still find and use this rally.
    events = detect_contacts_for_video(VIDEO)
    assert len(events) == 1
    assert events[0].rally_id == "r1"


def test_generate_coaching_report_finds_pass1_only_rallies_by_default(monkeypatch):
    events = detect_contacts_for_video(VIDEO)
    from ..contact_detect import save_events
    save_events(VIDEO, events)

    # Stub the adaptive selection (its own worst/best cap logic is covered by
    # test_coaching.py) and the Anthropic call (no ANTHROPIC_API_KEY in the
    # test env) so this test isolates what it actually guards: that the
    # pass1-only rally was found and its contact reached feature-vector
    # extraction and tip generation at all.
    monkeypatch.setattr(coaching, "select_tip_candidates",
                         lambda vectors_meta: [coaching._shot_summary(vec, c) for vec, c in vectors_meta])
    monkeypatch.setattr(coaching, "generate_llm_tips",
                         lambda shots: {s["contact_id"]: {"tip": "t", "tone": "praise"} for s in shots})

    report = generate_coaching_report(VIDEO)
    assert len(report.tips) == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

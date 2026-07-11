"""Readiness gating: synthetic jobs-table rows exercising each readiness
reason, and a synthetic overlay range short of full coverage. Uses a real
(temp) sqlite DB via db.connect(), same schema as production — readiness.py
is DB-touching (jobs/overlays/rally_labels), unlike the pure-function cores
the other rallyanalysis test files exercise."""
from __future__ import annotations

import pytest

from ... import db
from ...cv import overlays
from .. import readiness

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
    yield


def _add_job(status: str, created_at: float, kind: str = "pass2") -> None:
    with db.connect() as con:
        con.execute(
            "INSERT INTO jobs (id, video_id, kind, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (db.new_id(), VIDEO, kind, status, created_at, created_at))


def _add_rally(rally_id: str, start_frame: int, end_frame: int, source: str = "pass2") -> None:
    with db.connect() as con:
        con.execute(
            "INSERT INTO rally_labels (id, video_id, source, start_frame, end_frame)"
            " VALUES (?,?,?,?,?)",
            (rally_id, VIDEO, source, start_frame, end_frame))


def _write_overlay_frames(frames: range) -> None:
    overlays.write_entries(VIDEO, "pass2", [{"f": f, "players": []} for f in frames])


def test_no_pass2_job_not_ready():
    _add_rally("r1", 100, 200)
    assert readiness.get_analyzable_rallies(VIDEO, overlay_source="pass2") == []
    report = readiness.readiness_report(VIDEO, overlay_source="pass2")
    assert report["r1"] == {"ready": False, "reason": "pass2_not_done"}


def test_pass2_running_not_ready():
    _add_rally("r1", 100, 200)
    _write_overlay_frames(range(100, 201))
    _add_job("running", created_at=1.0)
    report = readiness.readiness_report(VIDEO, overlay_source="pass2")
    assert report["r1"] == {"ready": False, "reason": "pass2_running"}
    assert readiness.get_analyzable_rallies(VIDEO, overlay_source="pass2") == []


def test_pass2_error_not_ready():
    _add_rally("r1", 100, 200)
    _write_overlay_frames(range(100, 201))
    _add_job("error", created_at=1.0)
    report = readiness.readiness_report(VIDEO, overlay_source="pass2")
    assert report["r1"]["reason"] == "pass2_not_done"


def test_most_recent_job_wins_over_older_done():
    _add_rally("r1", 100, 200)
    _write_overlay_frames(range(100, 201))
    _add_job("done", created_at=1.0)
    _add_job("running", created_at=2.0)  # a newer re-run is now in flight
    report = readiness.readiness_report(VIDEO, overlay_source="pass2")
    assert report["r1"]["reason"] == "pass2_running"


def test_done_with_full_coverage_is_ready():
    _add_rally("r1", 100, 200)
    _write_overlay_frames(range(100, 201))
    _add_job("done", created_at=1.0)
    report = readiness.readiness_report(VIDEO, overlay_source="pass2")
    assert report["r1"] == {"ready": True, "reason": "ready"}
    rallies = readiness.get_analyzable_rallies(VIDEO, overlay_source="pass2")
    assert [r["id"] for r in rallies] == ["r1"]


def test_done_but_overlay_short_of_coverage_is_stale():
    # rally is 100-200 but overlay only covers up to 150 -- nowhere near the
    # tail end, well past the tolerance -> excluded, not partially included.
    _add_rally("r1", 100, 200)
    _write_overlay_frames(range(100, 151))
    _add_job("done", created_at=1.0)
    report = readiness.readiness_report(VIDEO, overlay_source="pass2")
    assert report["r1"] == {"ready": False, "reason": "overlay_stale"}
    assert readiness.get_analyzable_rallies(VIDEO, overlay_source="pass2") == []


def test_done_with_coverage_within_tolerance_of_edges_is_ready():
    # Overlay stops a few frames short of end_frame -- within tolerance, so
    # Pass 2 simply not emitting an entry for literally every frame doesn't
    # falsely mark this rally stale.
    _add_rally("r1", 100, 200)
    _write_overlay_frames(range(100, 200 - readiness.COVERAGE_TOLERANCE_FRAMES + 1))
    _add_job("done", created_at=1.0)
    report = readiness.readiness_report(VIDEO, overlay_source="pass2")
    assert report["r1"] == {"ready": True, "reason": "ready"}


def test_multiple_rallies_only_ready_one_included():
    _add_rally("r_ready", 0, 100)
    _add_rally("r_stale", 200, 400)
    _write_overlay_frames(range(0, 101))
    _write_overlay_frames(range(200, 250))  # far short of 400
    _add_job("done", created_at=1.0)
    rallies = readiness.get_analyzable_rallies(VIDEO, overlay_source="pass2")
    assert [r["id"] for r in rallies] == ["r_ready"]
    report = readiness.readiness_report(VIDEO, overlay_source="pass2")
    assert report["r_ready"]["ready"] is True
    assert report["r_stale"]["ready"] is False and report["r_stale"]["reason"] == "overlay_stale"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

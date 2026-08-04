"""Covers jobs.py's timeout safeguard: a hung job must be marked 'error'
(not left stuck in 'running' forever) without blocking subsequent queued
jobs, and a timed-out pass2_range job must fail its coaching_selections row
too (via pipeline._set_selection_status, not duplicated logic)."""
from __future__ import annotations

import threading
import time

import pytest

from .. import config, db, jobs


VIDEO = "vid1"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, fps, uploaded_at) VALUES (?,?,?,?,?)",
            (VIDEO, "v.mp4", "/tmp/v.mp4", 30.0, db.now()))
    monkeypatch.setattr(config, "JOB_TIMEOUT_DEFAULT_S", 0.2)
    monkeypatch.setattr(config, "JOB_TIMEOUT_PASS2_RANGE_S", 0.2)
    yield


def _job_row(job_id: str) -> dict:
    with db.connect() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row)


def test_hung_job_marked_failed_and_queue_continues():
    runner = jobs.JobRunner()
    runner.start()

    hang_forever = threading.Event()

    def hangs(handle):
        hang_forever.wait()  # never set; simulates a true hang

    done = threading.Event()

    def quick(handle):
        done.set()

    hung_id = runner.submit(VIDEO, "some_kind", hangs)
    next_id = runner.submit(VIDEO, "other_kind", quick)

    for _ in range(50):
        if done.is_set():
            break
        time.sleep(0.1)
    assert done.is_set(), "second queued job never ran — the hung job blocked the queue"

    row = _job_row(hung_id)
    assert row["status"] == "error"
    assert "timeout" in row["message"]

    row2 = _job_row(next_id)
    assert row2["status"] == "done"


def test_timed_out_pass2_range_job_fails_its_selection():
    with db.connect() as con:
        con.execute(
            "INSERT INTO coaching_selections (id, video_id, start_frame, end_frame, source,"
            " status, rally_label_id, overlay_source, error_message, created_at)"
            " VALUES ('sel1', ?, 0, 10, 'user_drawn', 'processing', NULL, NULL, NULL, ?)",
            (VIDEO, db.now()))

    runner = jobs.JobRunner()
    runner.start()
    hang_forever = threading.Event()
    runner.submit(VIDEO, "pass2_range_sel1", lambda handle: hang_forever.wait())

    for _ in range(50):
        with db.connect() as con:
            row = con.execute(
                "SELECT status FROM coaching_selections WHERE id='sel1'").fetchone()
        if row["status"] == "failed":
            break
        time.sleep(0.1)

    with db.connect() as con:
        row = con.execute("SELECT * FROM coaching_selections WHERE id='sel1'").fetchone()
    assert row["status"] == "failed"
    assert row["error_message"] == "job exceeded timeout, likely hung"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

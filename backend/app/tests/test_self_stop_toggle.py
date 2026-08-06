"""Covers the self-stop toggle: default-on (unchanged behavior), persists
across get/set, and stop_this_instance() skips the EC2 call entirely (no
boto3/metadata-service reach) once disabled."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from .. import db, self_stop


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(self_stop, "_STATE_PATH", tmp_path / "self_stop_state.json")
    monkeypatch.setattr(self_stop, "SELF_STOP_PASS1_GRACE_S", 600)
    monkeypatch.setattr(self_stop, "SELF_STOP_COACHING_GRACE_S", 600)
    # Empty-queue helpers that need a DB — point at an isolated sqlite file.
    db_path = tmp_path / "app.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_db()
    yield


def test_defaults_enabled_when_never_set():
    assert self_stop.get_enabled() is True


def test_set_enabled_persists():
    self_stop.set_enabled(False)
    assert self_stop.get_enabled() is False
    self_stop.set_enabled(True)
    assert self_stop.get_enabled() is True


def test_stop_this_instance_skips_ec2_call_when_disabled():
    self_stop.set_enabled(False)
    with patch("requests.put") as mock_put:
        self_stop.stop_this_instance()
        mock_put.assert_not_called()


def test_on_mode_stops_when_idle_no_grace():
    self_stop.set_enabled(True)
    with patch.object(self_stop, "_pending_pass2_s3_requests", return_value=False), \
         patch.object(self_stop, "_awaiting_pass2_state", return_value=(False, False)):
        assert self_stop.should_auto_stop() is True


def test_on_mode_defers_during_pass1_grace():
    self_stop.set_enabled(True)
    with patch.object(self_stop, "_pending_pass2_s3_requests", return_value=False), \
         patch.object(self_stop, "_awaiting_pass2_state", return_value=(True, False)):
        assert self_stop.should_auto_stop() is False


def test_on_mode_defers_during_coaching_grace():
    # Pass 2 just finished, no coaching report yet -- the user hasn't had a
    # chance to click "Generate report" yet, so this must defer.
    self_stop.set_enabled(True)
    with patch.object(self_stop, "_pending_pass2_s3_requests", return_value=False), \
         patch.object(self_stop, "_awaiting_pass2_state", return_value=(False, False)), \
         patch.object(self_stop, "_awaiting_coaching_report_state", return_value=(True, False)):
        assert self_stop.should_auto_stop() is False


def test_off_mode_never_stops_regardless_of_queue_or_grace_state():
    # A real kill switch: OFF means never, checked before any grace-window
    # logic even runs -- doesn't matter what state the queue/graces are in.
    self_stop.set_enabled(False)
    with patch.object(self_stop, "_pending_pass2_s3_requests", return_value=False), \
         patch.object(self_stop, "_awaiting_pass2_state", return_value=(False, True)), \
         patch.object(self_stop, "_awaiting_coaching_report_state", return_value=(False, True)):
        assert self_stop.should_auto_stop() is False


def test_awaiting_pass2_within_grace_from_db():
    vid = db.new_id()
    now = db.now()
    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, uploaded_at) VALUES (?,?,?,?)",
            (vid, "t.mp4", "", now))
        con.execute(
            "INSERT INTO jobs (id, video_id, kind, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (db.new_id(), vid, "pass1", "done", now, now))
    within, expired = self_stop._awaiting_pass2_state()
    assert within is True
    assert expired is False


def test_awaiting_coaching_report_within_grace_from_db():
    # Pass 2 just finished, no coaching_reports row yet -> within grace.
    vid = db.new_id()
    now = db.now()
    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, uploaded_at) VALUES (?,?,?,?)",
            (vid, "t.mp4", "", now))
        con.execute(
            "INSERT INTO jobs (id, video_id, kind, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (db.new_id(), vid, "pass2", "done", now, now))
    within, expired = self_stop._awaiting_coaching_report_state()
    assert within is True
    assert expired is False


def test_awaiting_coaching_report_expires_and_clears_once_report_exists():
    vid = db.new_id()
    stale = db.now() - 700  # older than the 600s test grace
    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, uploaded_at) VALUES (?,?,?,?)",
            (vid, "t.mp4", "", stale))
        con.execute(
            "INSERT INTO jobs (id, video_id, kind, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (db.new_id(), vid, "pass2", "done", stale, stale))
    within, expired = self_stop._awaiting_coaching_report_state()
    assert within is False
    assert expired is True

    # Once a report exists, this video no longer counts as awaiting one --
    # even past the grace window, it's done, not abandoned.
    with db.connect() as con:
        con.execute(
            "INSERT INTO coaching_reports (video_id, tips_json, created_at) VALUES (?,?,?)",
            (vid, "[]", db.now()))
    within, expired = self_stop._awaiting_coaching_report_state()
    assert within is False
    assert expired is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

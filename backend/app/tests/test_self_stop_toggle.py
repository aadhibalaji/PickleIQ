"""Covers the self-stop toggle + smart Pass-1→Pass-2 grace policy."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from .. import db, self_stop


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(self_stop, "_STATE_PATH", tmp_path / "self_stop_state.json")
    monkeypatch.setattr(self_stop, "_armed_after_pass2_chain", False)
    monkeypatch.setattr(self_stop, "SELF_STOP_PASS1_GRACE_S", 600)
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


def test_stop_this_instance_skips_ec2_when_should_auto_stop_false():
    self_stop.set_enabled(False)
    with patch.object(self_stop, "should_auto_stop", return_value=False), \
         patch("requests.put") as mock_put:
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


def test_off_mode_stops_only_after_pass2_armed():
    self_stop.set_enabled(False)
    with patch.object(self_stop, "_pending_pass2_s3_requests", return_value=False), \
         patch.object(self_stop, "_awaiting_pass2_state", return_value=(False, False)):
        assert self_stop.should_auto_stop() is False
        self_stop.arm_after_pass2_chain()
        assert self_stop.should_auto_stop() is True


def test_off_mode_stops_when_pass1_grace_expired():
    self_stop.set_enabled(False)
    with patch.object(self_stop, "_pending_pass2_s3_requests", return_value=False), \
         patch.object(self_stop, "_awaiting_pass2_state", return_value=(False, True)):
        assert self_stop.should_auto_stop() is True


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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

"""Covers the hybrid (S3 worker -> local UI-only container) progress-sync
fixes: Pass 1 rally windows merging in progressively via the same live
progress snapshot (not just at chain end), Pass 2's mirrored job actually
getting a numeric progress fraction instead of just a message, and Pass 1
alone (no auto-chained Pass 2) still reaching a terminal mirrored status
instead of getting stuck on the last live 'running' snapshot forever."""
from __future__ import annotations

import pytest

from .. import db, pipeline, results_export, storage_s3

VIDEO = "vid1"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, fps, uploaded_at) VALUES (?,?,?,?,?)",
            (VIDEO, "v.mp4", "/tmp/v.mp4", 30.0, db.now()))
    yield


def _rally_row(rid: str, s: int, e: int) -> dict:
    return {"id": rid, "start_frame": s, "end_frame": e, "meta": db.dumps({})}


def test_merge_rally_labels_replaces_prior_snapshot_for_source():
    results_export.merge_rally_labels(VIDEO, "pass1", [_rally_row("r1", 0, 100)])
    with db.connect() as con:
        rows = con.execute(
            "SELECT start_frame, end_frame FROM rally_labels WHERE video_id=? AND source='pass1'",
            (VIDEO,)).fetchall()
    assert [(r["start_frame"], r["end_frame"]) for r in rows] == [(0, 100)]

    # A later, larger snapshot (more rallies found as Pass 1 keeps running)
    # replaces the earlier partial one rather than accumulating duplicates.
    results_export.merge_rally_labels(
        VIDEO, "pass1", [_rally_row("r1", 0, 100), _rally_row("r2", 200, 300)])
    with db.connect() as con:
        rows = con.execute(
            "SELECT start_frame, end_frame FROM rally_labels WHERE video_id=? AND source='pass1'"
            " ORDER BY start_frame", (VIDEO,)).fetchall()
    assert [(r["start_frame"], r["end_frame"]) for r in rows] == [(0, 100), (200, 300)]


def test_mirror_cloud_progress_merges_rally_labels_from_snapshot():
    pipeline.mirror_cloud_progress(VIDEO, {
        "stage": "pass1", "status": "running", "progress": 0.5, "message": "Pass 1: 1 rally so far",
        "rally_labels": [_rally_row("r1", 0, 100)],
    })
    with db.connect() as con:
        rows = con.execute(
            "SELECT start_frame, end_frame FROM rally_labels WHERE video_id=? AND source='pass1'",
            (VIDEO,)).fetchall()
        job = con.execute(
            "SELECT status, progress FROM jobs WHERE video_id=? AND kind='pass1'",
            (VIDEO,)).fetchone()
    assert [(r["start_frame"], r["end_frame"]) for r in rows] == [(0, 100)]
    assert job["status"] == "running" and job["progress"] == pytest.approx(0.5)


def test_mark_cloud_pass2_active_sets_real_progress_fraction():
    pipeline.mark_cloud_pass2_active(VIDEO, window_index=2, total_windows=8)
    with db.connect() as con:
        job = con.execute(
            "SELECT progress, message FROM jobs WHERE video_id=? AND kind='pass2'",
            (VIDEO,)).fetchone()
    # window_index is 0-based; the 3rd of 8 windows synced -> 3/8.
    assert job["progress"] == pytest.approx(3 / 8)
    assert "window 3 synced" in job["message"]


def test_mirror_cloud_progress_done_status_flips_local_job_to_done():
    # Simulates poll-cloud-windows picking up the terminal snapshot
    # _upload_pass1_terminal_progress writes when Pass 1 finishes on its
    # own (no auto-chained Pass 2 to produce a chain-end result.json) — a
    # UI-only container's only other way to learn Pass 1 is actually done.
    pipeline.mirror_cloud_progress(VIDEO, {
        "stage": "pass1", "status": "running", "progress": 0.98, "message": "almost there",
    })
    pipeline.mirror_cloud_progress(VIDEO, {
        "stage": "pass1", "status": "done", "progress": 1.0, "message": "",
    })
    with db.connect() as con:
        job = con.execute(
            "SELECT status, progress FROM jobs WHERE video_id=? AND kind='pass1'",
            (VIDEO,)).fetchone()
    assert job["status"] == "done" and job["progress"] == pytest.approx(1.0)


@pytest.mark.parametrize("status,message,expected_progress", [
    ("done", "", 1.0),
    ("error", "boom", 0.0),
    ("cancelled", "", 0.0),
])
def test_upload_pass1_terminal_progress_payload_shape(monkeypatch, status, message, expected_progress):
    calls = []
    monkeypatch.setattr(storage_s3, "upload_progress", lambda vid, data: calls.append((vid, data)))
    pipeline._upload_pass1_terminal_progress(VIDEO, status, message)
    assert calls == [(VIDEO, {
        "stage": "pass1", "status": status, "progress": expected_progress, "message": message,
    })]


def test_upload_pass1_terminal_progress_never_raises_on_s3_failure(monkeypatch):
    # Best-effort: a mirroring nicety for a UI-only container that may not
    # even exist must never take down the real Pass 1 job over it.
    def _boom(vid, data):
        raise RuntimeError("s3 unavailable")
    monkeypatch.setattr(storage_s3, "upload_progress", _boom)
    pipeline._upload_pass1_terminal_progress(VIDEO, "done", "")  # must not raise


def test_mark_cloud_pass2_active_without_total_leaves_progress_untouched():
    with db.connect() as con:
        con.execute(
            "INSERT INTO jobs (id, video_id, kind, status, progress, message, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (db.new_id(), VIDEO, "pass2", "running", 0.4, "old", db.now(), db.now()))
    pipeline.mark_cloud_pass2_active(VIDEO, window_index=4, total_windows=None)
    with db.connect() as con:
        job = con.execute(
            "SELECT progress FROM jobs WHERE video_id=? AND kind='pass2'", (VIDEO,)).fetchone()
    assert job["progress"] == pytest.approx(0.4)

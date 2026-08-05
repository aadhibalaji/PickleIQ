"""Export/import contract for the hybrid architecture: the cloud (EC2) side
serializes a video's pipeline output into one JSON blob and uploads it to S3
(storage_s3.upload_results); the local side downloads and merges it back into
its own SQLite DB (api/media.py's sync-from-cloud endpoint). Both directions
share this module so the shape never drifts between export and import.

Deliberately captures whatever exists, even for a chain that partially
failed (e.g. Pass 1 done, Pass 2 errored on missing calibration) — a partial
result surfaced beats a partial result silently stranded on a stopped
instance. chain_status is read from the jobs table (the same table the local
UI already polls), so "what happened" is reported the same way here as
everywhere else in the app, not re-derived from guessing at row presence.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from . import db
from .cv.overlays import chunk_index

log = logging.getLogger("results_export")

_JOB_KINDS = ("pass1", "pass2", "contacts")
_STAGE_LABELS = {"pass1": "Pass 1", "pass2": "Pass 2", "contacts": "contacts", "coaching": "coaching"}


def _job_statuses(con: "db.sqlite3.Connection", video_id: str) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for kind in _JOB_KINDS:
        row = con.execute(
            "SELECT status, message FROM jobs WHERE video_id=? AND kind=?"
            " ORDER BY created_at DESC LIMIT 1", (video_id, kind)).fetchone()
        statuses[kind] = {"status": row["status"], "message": row["message"]} if row \
            else {"status": "not_run", "message": None}
    return statuses


def export_video_results(video_id: str,
                          status_override: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Snapshots everything this app knows about video_id into one JSON-safe
    dict. Called regardless of whether processing fully succeeded.

    status_override lets a caller assert a stage's status directly instead of
    reading it off the jobs table — needed when exporting from INSIDE the
    still-running job's own exception handler (a cancelled/errored Pass 2),
    where the jobs row for that job hasn't been updated yet (jobs.py's
    JobRunner only writes the final status AFTER job.fn() returns/raises, and
    export happens before that point so it can complete before self-stop —
    see pipeline._build_do_pass2). Without the override, chain_status would
    misreport that stage as still 'running'."""
    with db.connect() as con:
        rally_labels = db.rows_to_dicts(con.execute(
            "SELECT id, source, start_frame, end_frame, meta FROM rally_labels"
            " WHERE video_id=? AND source IN ('pass1','pass2')", (video_id,)).fetchall())

        overlay_rows = con.execute(
            "SELECT source, chunk_index, payload FROM overlays WHERE video_id=?",
            (video_id,)).fetchall()
        overlays: dict[str, list[dict]] = {}
        for r in overlay_rows:
            overlays.setdefault(r["source"], []).append(
                {"chunk_index": r["chunk_index"], "payload": r["payload"]})

        contact_events = db.rows_to_dicts(con.execute(
            "SELECT * FROM contact_events WHERE video_id=?", (video_id,)).fetchall())

        coaching_row = con.execute(
            "SELECT * FROM coaching_reports WHERE video_id=?", (video_id,)).fetchone()
        coaching_report = dict(coaching_row) if coaching_row else None

        # "the calibration used" — same resolution order as pipeline.effective_calibration:
        # this video's own override if it has one, else the active global calibration.
        vrow = con.execute("SELECT calibration_id FROM videos WHERE id=?", (video_id,)).fetchone()
        cal_row = None
        if vrow and vrow["calibration_id"]:
            cal_row = con.execute("SELECT * FROM calibrations WHERE id=?",
                                   (vrow["calibration_id"],)).fetchone()
        if cal_row is None:
            cal_row = con.execute(
                "SELECT * FROM calibrations WHERE scope='global'"
                " ORDER BY created_at DESC LIMIT 1").fetchone()
        calibration = dict(cal_row) if cal_row is not None else None

        chain_status = _job_statuses(con, video_id)
        chain_status["coaching"] = {"status": "done" if coaching_row else "not_run", "message": None}
        if status_override:
            for kind, info in status_override.items():
                chain_status[kind] = {"status": info["status"], "message": info.get("message")}

    payload = {
        "video_id": video_id,
        "exported_at": time.time(),
        "chain_status": chain_status,
        "rally_labels": rally_labels,
        "overlays": overlays,
        "contact_events": contact_events,
        "coaching_report": coaching_report,
        "calibration": calibration,
    }
    log.info("exported results for %s: %s", video_id,
              {k: v["status"] for k, v in chain_status.items()})
    return payload


def describe_chain_status(chain_status: dict[str, dict[str, Any]], rally_labels: list[dict]) -> str:
    """Human summary for the sync response, e.g. 'Pass 1: 4 rallies, Pass 2:
    error — missing calibration' or 'Pass 1 + Pass 2 + contacts + coaching:
    all complete'."""
    stages = ("pass1", "pass2", "contacts", "coaching")
    if all(chain_status.get(s, {}).get("status") == "done" for s in stages):
        return "Pass 1 + Pass 2 + contacts + coaching: all complete"

    n_rallies = sum(1 for r in rally_labels if r["source"] == "pass1")
    parts = []
    for s in stages:
        info = chain_status.get(s, {"status": "not_run", "message": None})
        status = info.get("status", "not_run")
        label = _STAGE_LABELS[s]
        if status == "done":
            if s == "pass1":
                parts.append(f"{label}: {n_rallies} rall{'y' if n_rallies == 1 else 'ies'}")
            else:
                parts.append(f"{label}: done")
        elif status == "not_run":
            parts.append(f"{label}: not run")
        else:
            detail = f" — {info['message']}" if info.get("message") else ""
            parts.append(f"{label}: {status}{detail}")
    return ", ".join(parts)


def merge_rally_labels(video_id: str, source: str, rows: list[dict]) -> None:
    """Delete-then-insert this video's rally_labels for one source (pass1/
    pass2), same pattern cv/pass1.py and merge_video_results below use
    locally — shared so the live progress-snapshot path (mirror_cloud_progress)
    and the chain-end full sync (merge_video_results) can't drift apart."""
    with db.connect() as con:
        con.execute("DELETE FROM rally_labels WHERE video_id=? AND source=?", (video_id, source))
        if rows:
            con.executemany(
                "INSERT INTO rally_labels (id, video_id, source, start_frame, end_frame, meta)"
                " VALUES (?,?,?,?,?,?)",
                [(r["id"], video_id, source, r["start_frame"], r["end_frame"], r["meta"])
                 for r in rows])


def merge_video_results(video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Upserts a downloaded result.json into the local DB. Safe to call
    repeatedly: cloud-authoritative tables (rally_labels sourced from
    pass1/pass2, overlays, contact_events) are replaced per-source/video
    wholesale before the fresh rows are inserted — the same delete-then-insert
    pattern cv/pass1.py, cv/pass2.py and contact_detect.save_events already
    use locally, so a video reprocessed and re-synced doesn't accumulate
    stale rows from a previous cloud run. Human-sourced rally_labels
    (source='human') are never touched — the export only ever contains
    pass1/pass2 rows in the first place.

    The calibration is imported as a clip-local override tagged only to this
    video (deterministic id, never the cloud instance's own row id) rather
    than as a 'global' row — importing a foreign global calibration could
    silently change which calibration every OTHER video resolves to via
    effective_calibration()'s "newest global" lookup. It is not applied to
    videos.calibration_id automatically; it's there to inspect/assign
    explicitly via the existing calibration UI if wanted.
    """
    if payload.get("video_id") != video_id:
        raise ValueError(
            f"result.json video_id {payload.get('video_id')!r} does not match {video_id!r}")

    rally_labels = payload.get("rally_labels", [])
    overlays = payload.get("overlays", {})
    contact_events = payload.get("contact_events", [])
    coaching_report = payload.get("coaching_report")
    calibration = payload.get("calibration")

    for source in ("pass1", "pass2"):
        rows = [r for r in rally_labels if r["source"] == source]
        if rows:
            merge_rally_labels(video_id, source, rows)

    with db.connect() as con:
        for source, chunks in overlays.items():
            con.execute("DELETE FROM overlays WHERE video_id=? AND source=?",
                        (video_id, source))
            con.executemany(
                "INSERT INTO overlays (video_id, source, chunk_index, payload) VALUES (?,?,?,?)",
                [(video_id, source, c["chunk_index"], c["payload"]) for c in chunks])

        if contact_events:
            con.execute("DELETE FROM contact_events WHERE video_id=?", (video_id,))
            cols = list(contact_events[0].keys())
            placeholders = ",".join("?" for _ in cols)
            con.executemany(
                f"INSERT INTO contact_events ({','.join(cols)}) VALUES ({placeholders})",
                [tuple(e[c] for c in cols) for e in contact_events])

        if coaching_report:
            # .get(): old cloud exports predate themes_json/stale_reason/
            # focus/general_json — import them as NULL rather than dropping
            # the whole row (or, as before this fix, silently dropping the
            # columns even when the export DID carry them).
            con.execute(
                "INSERT OR REPLACE INTO coaching_reports"
                " (video_id, tips_json, themes_json, general_json, focus, created_at, stale_reason)"
                " VALUES (?,?,?,?,?,?,?)",
                (video_id, coaching_report["tips_json"],
                 coaching_report.get("themes_json"), coaching_report.get("general_json"),
                 coaching_report.get("focus"), coaching_report["created_at"],
                 coaching_report.get("stale_reason")))

        if calibration:
            sync_cal_id = f"sync-{video_id}"
            con.execute(
                "INSERT OR REPLACE INTO calibrations"
                " (id, scope, video_id, corner_points, homography, focal_px, created_at)"
                " VALUES (?,'clip_local',?,?,?,?,?)",
                (sync_cal_id, video_id, calibration["corner_points"], calibration["homography"],
                 calibration["focal_px"], calibration["created_at"]))

    chain_status = payload.get("chain_status", {})
    counts = {
        "rally_labels": len(rally_labels),
        "overlay_chunks": sum(len(v) for v in overlays.values()),
        "contact_events": len(contact_events),
        "coaching_report": bool(coaching_report),
        "calibration": bool(calibration),
    }
    log.info("merged cloud results for %s: %s (%s)", video_id,
              {k: v["status"] for k, v in chain_status.items()}, counts)
    return {"video_id": video_id, "chain_status": chain_status, "counts": counts,
            "message": describe_chain_status(chain_status, rally_labels)}


def export_window_results(video_id: str, source: str, window_index: int,
                           start_frame: int, end_frame: int,
                           total_windows: int | None = None) -> dict[str, Any]:
    """Same overlays shape as export_video_results, scoped to the overlay
    chunks touched by one Pass 2 window — for live incremental visibility
    while Pass 2 is still running (see cv/pass2.py's per-window upload, right
    after overlays.write_entries() flushes that window). NOT a substitute
    for the full export at chain end: it carries no rally_labels, contacts,
    coaching, or calibration — just this window's detection overlays.

    total_windows (the count Pass 2 built its window list from) rides along
    so a UI-only container can turn "window N synced" into an actual N/total
    fraction for its Pass 2 progress bar (see pipeline.mark_cloud_pass2_active)
    — without it, that container has no way to know how many windows there
    will be in total, only that one more just landed."""
    c0, c1 = chunk_index(start_frame), chunk_index(end_frame)
    with db.connect() as con:
        overlay_rows = con.execute(
            "SELECT chunk_index, payload FROM overlays"
            " WHERE video_id=? AND source=? AND chunk_index BETWEEN ? AND ?",
            (video_id, source, c0, c1)).fetchall()
    payload = {
        "video_id": video_id,
        "window_index": window_index,
        "total_windows": total_windows,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "exported_at": time.time(),
        "overlays": {source: [{"chunk_index": r["chunk_index"], "payload": r["payload"]}
                               for r in overlay_rows]},
    }
    log.info("exported window %d results for %s (%d overlay chunk(s))",
              window_index, video_id, len(overlay_rows))
    return payload


def merge_window_results(video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Upserts one incremental window's overlay chunks into the local DB.
    Deletes only the SPECIFIC chunk indices present in this payload before
    reinserting them — unlike merge_video_results' per-source wholesale
    delete, this must not touch chunks written by other already-merged
    windows, since a window-scoped payload only ever covers a subset of a
    source's chunks. Safe to call repeatedly for the same window (including
    a reprocessed run's window at the same index with different content)."""
    if payload.get("video_id") != video_id:
        raise ValueError(
            f"window result video_id {payload.get('video_id')!r} does not match {video_id!r}")

    overlays = payload.get("overlays", {})
    with db.connect() as con:
        for source, chunks in overlays.items():
            if not chunks:
                continue
            idxs = [c["chunk_index"] for c in chunks]
            placeholders = ",".join("?" for _ in idxs)
            con.execute(
                f"DELETE FROM overlays WHERE video_id=? AND source=? AND chunk_index IN ({placeholders})",
                (video_id, source, *idxs))
            con.executemany(
                "INSERT INTO overlays (video_id, source, chunk_index, payload) VALUES (?,?,?,?)",
                [(video_id, source, c["chunk_index"], c["payload"]) for c in chunks])

    n_chunks = sum(len(v) for v in overlays.values())
    log.info("merged window %s results for %s (%d chunk(s))",
              payload.get("window_index"), video_id, n_chunks)
    return {"window_index": payload.get("window_index"),
            "total_windows": payload.get("total_windows"),
            "start_frame": payload.get("start_frame"), "end_frame": payload.get("end_frame"),
            "overlay_chunks": n_chunks}

"""Upload, video streaming (range requests), job status, reprocess."""
from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from .. import db, pipeline, results_export, self_stop, storage_s3
from ..config import RUN_PROCESSING, STORAGE_BACKEND, VIDEO_DIR
from ..cv.video_io import probe
from ..jobs import runner

log = logging.getLogger("api.media")

router = APIRouter()


class VideoUpdateIn(BaseModel):
    match_type: str | None = None
    coaching_focus: str | None = None


# Allowed values per PATCHable videos column. None is always allowed (unset).
_VIDEO_FIELD_VALUES = {
    "match_type": ("singles", "doubles"),
    "coaching_focus": ("form", "positioning"),
}


@router.post("/videos")
async def upload_video(file: UploadFile, match_type: str = Form(...)):
    # Required at upload, enforced server-side (not just in the frontend
    # selector) — this is what cv/pass2.py reads to cap player-identity
    # capacity per side (1 for singles, 2 for doubles); a missing/wrong value
    # here is how a singles clip used to end up tracking 4 "players" (see
    # docs audit). No default: an upload without it is rejected outright
    # rather than silently guessing a match type.
    if match_type not in ("singles", "doubles"):
        raise HTTPException(400, "match_type is required and must be 'singles' or 'doubles'")
    video_id = db.new_id()
    safe_name = os.path.basename(file.filename or "video.mp4")
    s3_key = None

    if STORAGE_BACKEND == "s3":
        # Written straight to the scratch path Pass 1 will read from — the
        # same file serves both as the S3 upload source and, since it's
        # already local, Pass 1 doesn't need to download anything back down
        # for this video (see pipeline._ensure_local_video).
        dest = storage_s3.scratch_path(video_id)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            meta = probe(str(dest))
        except ValueError as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, str(exc)) from exc
        try:
            s3_key = storage_s3.upload_video(dest, video_id)
        except RuntimeError as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(500, str(exc)) from exc
    else:
        dest_dir = VIDEO_DIR / video_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_name
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            meta = probe(str(dest))
        except ValueError as exc:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(400, str(exc)) from exc

    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, s3_key, fps, width, height, frame_count,"
            " uploaded_at, match_type)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (video_id, safe_name, str(dest), s3_key, meta.fps, meta.width, meta.height,
             meta.frame_count, db.now(), match_type),
        )
    if RUN_PROCESSING:
        # Immediately kick off Pass 1 in the background — the user goes
        # straight to the labeling UI while this runs.
        job = pipeline.start_processing(video_id)
    else:
        # This container only uploads/displays/syncs — the S3 write above
        # (unconditional, happens regardless of RUN_PROCESSING) is what
        # wakes the EC2 worker; see pipeline.seed_cloud_placeholder for why
        # this isn't a runner.submit call.
        pipeline.seed_cloud_placeholder(video_id)
        job = {"job_id": None, "queued_for_cloud": True}
    return {"id": video_id, "filename": safe_name, "fps": meta.fps,
            "width": meta.width, "height": meta.height,
            "frame_count": meta.frame_count, **job}


@router.get("/videos")
def list_videos():
    with db.connect() as con:
        rows = con.execute("SELECT * FROM videos ORDER BY uploaded_at DESC").fetchall()
    return db.rows_to_dicts(rows)


@router.get("/videos/{video_id}")
def get_video(video_id: str):
    with db.connect() as con:
        row = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "video not found")
        jobs = con.execute(
            "SELECT * FROM jobs WHERE video_id=? ORDER BY created_at DESC", (video_id,)
        ).fetchall()
    positions = runner.queue_positions()
    out_jobs = []
    for j in db.rows_to_dicts(jobs):
        j["queue_ahead"] = positions.get(j["id"]) if j["status"] == "queued" else None
        out_jobs.append(j)
    # Normalized player-detection exclusion polygons so the overlay can draw them.
    from ..cv.exclusions import zones_as_lists
    # RUN_PROCESSING=0: jobs rows here are informational mirrors of cloud
    # state (see pipeline.seed_cloud_placeholder/mirror_cloud_chain_status),
    # not real jobs.JobRunner entries — cancelling them locally would no-op
    # silently. The frontend uses this to hide the Cancel button for them.
    return {**dict(row), "jobs": out_jobs, "exclusion_zones": zones_as_lists(),
            "run_processing": RUN_PROCESSING}


@router.patch("/videos/{video_id}")
def update_video(video_id: str, body: VideoUpdateIn):
    """Per-clip metadata that isn't produced by a pass — match_type (used by
    the coaching layer to defend against player-identity mis-tracking on
    singles clips) and coaching_focus (which metric family coaching reports
    evaluate — see rallyanalysis/coaching.py). Partial update: only fields
    present in the request body are written, so PATCHing one field never
    clears another."""
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(400, "no fields to update")
    for name, value in fields.items():
        allowed = _VIDEO_FIELD_VALUES[name]
        if value is not None and value not in allowed:
            raise HTTPException(400, f"{name} must be one of {' | '.join(allowed)}")
    with db.connect() as con:
        if con.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is None:
            raise HTTPException(404, "video not found")
        sets = ", ".join(f"{name}=?" for name in fields)
        con.execute(f"UPDATE videos SET {sets} WHERE id=?", (*fields.values(), video_id))
    return {"id": video_id, **fields}


@router.delete("/videos/{video_id}")
def delete_video(video_id: str):
    with db.connect() as con:
        row = con.execute("SELECT path, s3_key FROM videos WHERE id=?", (video_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "video not found")
        runner.cancel_video(video_id)
        # Every table that keys off video_id — no FK cascade is declared
        # (see db.py SCHEMA), so each has to be cleared explicitly.
        for table in (
            "jobs", "rally_labels", "overlays", "calibrations", "user_notes",
            "discrepancies", "reports", "ball_corrections", "ball_tracks",
            "window_grades", "contact_events", "coaching_reports",
        ):
            con.execute(f"DELETE FROM {table} WHERE video_id=?", (video_id,))
        con.execute("DELETE FROM videos WHERE id=?", (video_id,))

    if STORAGE_BACKEND == "s3":
        storage_s3.delete_video_data(video_id)
        storage_s3.cleanup_scratch(video_id)
    else:
        path = Path(row["path"])
        shutil.rmtree(path.parent, ignore_errors=True)
    return {"deleted": video_id}


@router.post("/videos/{video_id}/reprocess")
def reprocess(video_id: str, which: str = "pass2"):
    if which not in ("pass1", "pass2"):
        raise HTTPException(400, "which must be pass1 or pass2")
    if not RUN_PROCESSING:
        # This container has no mechanism to ask the cloud worker to
        # reprocess (no equivalent of the S3-upload trigger for "redo this
        # video") — refuse loudly rather than silently running it here,
        # which is exactly what RUN_PROCESSING=0 exists to prevent.
        raise HTTPException(
            400, "reprocess is not available on this container (RUN_PROCESSING=0) — "
                 "there is currently no way to trigger cloud reprocessing from here")
    with db.connect() as con:
        if con.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is None:
            raise HTTPException(404, "video not found")
    return pipeline.reprocess(video_id, which)


@router.post("/videos/{video_id}/sync-from-cloud")
def sync_from_cloud(video_id: str):
    """Manual trigger (UI button) — checks S3 for this video's exported
    cloud-processing results (see results_export.py, pipeline._finish_chain)
    and merges whatever is there into the local DB. Safe to call repeatedly:
    merge_video_results upserts, it never blind-inserts.

    Only meaningful on a RUN_PROCESSING=0 (UI-only) container: a
    RUN_PROCESSING=1 container either ran this itself (nothing to sync — its
    own DB already has the answer) or is the EC2 worker that WROTE the
    results in the first place. Rejecting the mismatch here, rather than
    just no-op'ing, avoids a subtler problem: mirror_cloud_chain_status
    below writes synthetic jobs rows to fake "active" status for a container
    that never runs jobs — on a container that DOES run real jobs via
    jobs.JobRunner, those synthetic rows would sit alongside/shadow the real
    ones for the same video and confuse Workspace.jsx's newest-row-per-kind
    display logic."""
    if STORAGE_BACKEND != "s3":
        raise HTTPException(400, "sync-from-cloud only applies when STORAGE_BACKEND=s3")
    if RUN_PROCESSING:
        raise HTTPException(
            400, "sync-from-cloud only applies on a RUN_PROCESSING=0 (UI-only) container")
    with db.connect() as con:
        if con.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is None:
            raise HTTPException(404, "video not found")
    try:
        payload = storage_s3.download_results(video_id)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    if payload is None:
        return {"found": False, "video_id": video_id,
                "message": "no cloud results yet — it may still be processing, or hasn't "
                           "started"}
    summary = results_export.merge_video_results(video_id, payload)
    pipeline.mirror_cloud_chain_status(video_id, summary["chain_status"])
    return {"found": True, **summary}


# video_id -> {window_index: etag} already merged. In-memory only, on
# purpose (see poll_cloud_windows docstring) — this is a cheap "don't
# re-download what we've already applied" cursor, not a durability
# guarantee; a process restart just means the next poll re-merges
# already-seen windows once, which merge_window_results makes harmless.
_merged_window_etags: dict[str, dict[int, str]] = {}


@router.post("/videos/{video_id}/poll-cloud-windows")
def poll_cloud_windows(video_id: str):
    """Lightweight incremental pull for live Pass 2 visibility while cloud
    processing is still in flight (see cv/pass2.py's per-window upload and
    results_export.merge_window_results). Meant to be called on a short
    interval (frontend uses the same ~3s tick it already polls job status
    on) while a video's cloud job looks active; distinct from
    sync-from-cloud, which merges the one final full export at chain end.

    Tracked by S3 object etag rather than "highest index seen": a
    reprocessed run's Pass 2 reuses the same window_N.json keys with new
    content, so an index-only watermark would wrongly skip re-merging a
    window that changed.

    Also pulls the live Pass 1 progress snapshot (see storage_s3.
    upload_progress / cv/pass1.py) if one exists — Pass 1 has no per-window
    checkpoints of its own, so without this the local UI shows nothing but
    "waiting for cloud worker" for however long Pass 1 takes, even though
    it's actively running.

    Only meaningful on a RUN_PROCESSING=0 (UI-only) container — same
    reasoning as sync-from-cloud's docstring: on a container that runs jobs
    for real, the synthetic activity nudges below (mark_cloud_pass2_active,
    mirror_cloud_progress) would collide with genuine jobs.JobRunner rows
    for the same video."""
    if STORAGE_BACKEND != "s3":
        raise HTTPException(400, "poll-cloud-windows only applies when STORAGE_BACKEND=s3")
    if RUN_PROCESSING:
        raise HTTPException(
            400, "poll-cloud-windows only applies on a RUN_PROCESSING=0 (UI-only) container")
    with db.connect() as con:
        if con.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is None:
            raise HTTPException(404, "video not found")

    try:
        progress_payload = storage_s3.download_progress(video_id)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    if progress_payload is not None:
        pipeline.mirror_cloud_progress(video_id, progress_payload)

    try:
        objects = storage_s3.list_window_objects(video_id)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc

    seen = _merged_window_etags.setdefault(video_id, {})
    merged = []
    for obj in objects:
        idx, etag = obj["index"], obj["etag"]
        if seen.get(idx) == etag:
            continue
        payload = storage_s3.download_window_results(video_id, idx)
        if payload is None:
            continue
        merged.append(results_export.merge_window_results(video_id, payload))
        seen[idx] = etag
    if merged:
        pipeline.mark_cloud_pass2_active(video_id, merged[-1]["window_index"],
                                          merged[-1]["total_windows"])
    return {"video_id": video_id, "new_windows": merged,
            "pass1_progress": progress_payload}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return {"cancelled": runner.cancel(job_id)}


@router.post("/videos/{video_id}/cancel")
def cancel_video_jobs(video_id: str):
    return {"cancelled": runner.cancel_video(video_id)}


class SelfStopIn(BaseModel):
    enabled: bool


@router.get("/self-stop")
def get_self_stop():
    """Current state of the EC2 self-stop-on-empty-queue toggle (see
    self_stop.py) — testing/ops control, unrelated to manual start/stop."""
    return {"enabled": self_stop.get_enabled()}


@router.post("/self-stop")
def set_self_stop(body: SelfStopIn):
    self_stop.set_enabled(body.enabled)
    if STORAGE_BACKEND == "s3":
        # Mirrors to S3 so the separate EC2 worker instance (which never sees
        # this container's local filesystem directly) picks it up on its next
        # cloud_ingest poll — see cloud_ingest._sync_self_stop.
        try:
            storage_s3.upload_self_stop_state(body.enabled)
        except Exception:
            log.exception("failed to sync self-stop toggle to S3 (non-fatal)")
    return {"enabled": body.enabled}


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK = 1024 * 1024


@router.get("/videos/{video_id}/stream")
def stream_video(video_id: str, request: Request):
    with db.connect() as con:
        row = con.execute("SELECT path, s3_key FROM videos WHERE id=?", (video_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "video not found")

    if STORAGE_BACKEND == "s3":
        # 302 to a presigned S3 URL rather than proxying bytes through this
        # instance: the <video> tag's src is a plain URL (VideoPlayer.jsx),
        # so the browser follows the redirect transparently — including
        # subsequent Range requests, which S3 supports natively, so scrubbing
        # still works with no frontend changes.
        if not row["s3_key"]:
            raise HTTPException(404, "video has no S3 object recorded")
        url = storage_s3.generate_presigned_url(row["s3_key"])
        return RedirectResponse(url, status_code=302)

    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(404, "video file missing on disk")
    size = path.stat().st_size

    range_header = request.headers.get("range")
    start, end = 0, size - 1
    status = 200
    if range_header:
        m = _RANGE_RE.match(range_header)
        if m:
            if m.group(1):
                start = int(m.group(1))
            if m.group(2):
                end = min(int(m.group(2)), size - 1)
            status = 206

    def iter_file():
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = f.read(min(_CHUNK, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": "video/mp4",
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(iter_file(), status_code=status, headers=headers)

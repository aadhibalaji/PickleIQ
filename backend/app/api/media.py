"""Upload, video streaming (range requests), job status, reprocess."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from .. import db, pipeline
from ..config import VIDEO_DIR
from ..cv.video_io import probe
from ..jobs import runner

router = APIRouter()


@router.post("/videos")
async def upload_video(file: UploadFile):
    video_id = db.new_id()
    dest_dir = VIDEO_DIR / video_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = os.path.basename(file.filename or "video.mp4")
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
            "INSERT INTO videos (id, filename, path, fps, width, height, frame_count, uploaded_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (video_id, safe_name, str(dest), meta.fps, meta.width, meta.height,
             meta.frame_count, db.now()),
        )
    # Immediately kick off Pass 1 in the background — the user goes straight
    # to the labeling UI while this runs.
    job = pipeline.start_processing(video_id)
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
    return {**dict(row), "jobs": out_jobs, "exclusion_zones": zones_as_lists()}


@router.post("/videos/{video_id}/reprocess")
def reprocess(video_id: str, which: str = "pass2"):
    if which not in ("pass1", "pass2"):
        raise HTTPException(400, "which must be pass1 or pass2")
    with db.connect() as con:
        if con.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is None:
            raise HTTPException(404, "video not found")
    return pipeline.reprocess(video_id, which)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return {"cancelled": runner.cancel(job_id)}


@router.post("/videos/{video_id}/cancel")
def cancel_video_jobs(video_id: str):
    return {"cancelled": runner.cancel_video(video_id)}


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK = 1024 * 1024


@router.get("/videos/{video_id}/stream")
def stream_video(video_id: str, request: Request):
    with db.connect() as con:
        row = con.execute("SELECT path FROM videos WHERE id=?", (video_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "video not found")
    return _stream_file(Path(row["path"]), request)


# ------------------------------------------------ rallies-only cropped video ---

@router.post("/videos/{video_id}/cropped")
def build_cropped_video(video_id: str):
    """Queue the rallies-only export (kept refined segments concatenated).
    Progress shows up in the normal jobs panel; kind='crop'."""
    from .. import export
    status = export.cropped_status(video_id)
    if status["n_rallies"] == 0:
        raise HTTPException(400, "no kept rallies to export — run Pass 2 first")

    def do_crop(handle):
        export.build_cropped(video_id, handle)

    job_id = runner.submit(video_id, "crop", do_crop)
    return {"job_id": job_id, **status}


@router.get("/videos/{video_id}/cropped")
def cropped_video_status(video_id: str):
    from .. import export
    try:
        return export.cropped_status(video_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/videos/{video_id}/cropped/stream")
def stream_cropped_video(video_id: str, request: Request):
    from .. import export
    try:
        path = export.cropped_path(video_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _stream_file(path, request)


def _stream_file(path: Path, request: Request):
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

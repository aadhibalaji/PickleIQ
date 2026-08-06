"""EC2 worker side of the hybrid handoff: discovers videos a RUN_PROCESSING=0
UI-only container uploaded to S3 and starts local Pass 1/2 processing for
them on this instance.

This is the piece that was missing before: picklecv-start-ec2 (the Lambda
behind the bucket's s3:ObjectCreated notification) only powers the instance
on — it doesn't pass along a video_id or call anything on the instance. And
this instance's own startup only ever cleaned up ITS OWN stale jobs (see
main.py); it never checked S3 or its DB for unprocessed work. Before
RUN_PROCESSING existed, that gap was invisible because whichever container
happened to receive the upload HTTP request ran Pass 1/2 on itself
in-process — often the wrong one. Now that local containers correctly never
do that, something has to actually notice a new S3 upload and act on it —
this is that something.

Runs as a background thread, started from main.py's startup() ONLY when
RUN_PROCESSING is true AND STORAGE_BACKEND=='s3' (see config.py) — a
RUN_PROCESSING=0 UI-only container has no business discovering and starting
CV work; it only uploads, displays, and syncs.
"""
from __future__ import annotations

import logging
import re
import threading
import time

import yaml

from . import db, pipeline, self_stop, storage_s3
from .config import DATA_DIR, S3_VIDEO_BUCKET
from .cv.video_io import probe
from .refine.config import override_path

log = logging.getLogger("cloud_ingest")

# How often this instance checks S3 for videos it hasn't started yet.
# Cheap by design (one list_objects_v2 call per tick, skipped entirely once
# a video_id is already known locally) — no need for anything tighter than
# this while a human is waiting on the other end for Pass 1 to even start.
POLL_INTERVAL_S = 20

_KEY_RE = re.compile(r"^" + re.escape(storage_s3.DEFAULT_USER_ID) + r"/([0-9a-f]{32})\.mp4$")


def _known_video_ids() -> set[str]:
    with db.connect() as con:
        return {r["id"] for r in con.execute("SELECT id FROM videos")}


def _register_without_processing(video_id: str, s3_key: str) -> None:
    """Adds video_id to this instance's own DB WITHOUT downloading it or
    starting processing — used when a final result.json already exists in
    S3 for it (see _tick's results_exist check). Keeps _known_video_ids()'s
    dedup effective for this video from now on without re-fetching/
    re-checking it every tick forever; no video bytes are needed since this
    instance is deliberately not going to process it."""
    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, s3_key, uploaded_at) VALUES (?,?,?,?,?)",
            (video_id, f"{video_id}.mp4", "", s3_key, db.now()))


def _ingest_video(video_id: str, s3_key: str) -> None:
    """Downloads the video (needed locally for Pass 1/2 regardless — see
    storage_s3.py), inserts the videos row this instance's own DB was
    missing, and starts the normal processing chain exactly the way a local
    upload would. Original filename isn't recoverable from just the S3 key
    (user_id/video_id.mp4 carries no filename) — a placeholder is used; it's
    cosmetic only, nothing keys off it."""
    log.info("cloud ingest: new video %s at s3://%s/%s — downloading + starting Pass 1",
              video_id, S3_VIDEO_BUCKET, s3_key)
    local_path = storage_s3.download_to_scratch(s3_key, video_id)
    meta = probe(str(local_path))
    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, s3_key, fps, width, height, frame_count, uploaded_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (video_id, f"{video_id}.mp4", str(local_path), s3_key, meta.fps, meta.width,
             meta.height, meta.frame_count, db.now()))
    pipeline.start_processing(video_id)
    log.info("cloud ingest: Pass 1 queued for %s", video_id)


def _sync_calibration() -> None:
    """Mirrors the current global CameraCalibration from S3 into this
    instance's own local DB. Calibration is meant to be set once (fixed
    camera setup) and reused for every video from it, but it's normally
    saved through a separate RUN_PROCESSING=0 UI-only container — whose
    local calibrations table this worker instance never otherwise sees —
    so without this, effective_calibration() here would find nothing and
    every video would process uncalibrated. Runs every tick, same cadence
    as the video poll: one S3 GET, a no-op once this instance already has
    the current calibration id, so cheap enough not to need its own
    interval. Only INSERTs (never UPDATEs) since calibration ids are
    immutable once saved (see api/annotate.py's save_calibration) — a new
    global calibration always gets a new id, which naturally becomes the
    effective one since effective_calibration() picks the newest by
    created_at."""
    try:
        cal = storage_s3.download_calibration()
    except Exception:
        log.exception("cloud ingest: calibration S3 fetch failed (non-fatal, retrying next tick)")
        return
    if cal is None:
        return
    with db.connect() as con:
        existing = con.execute(
            "SELECT 1 FROM calibrations WHERE id=?", (cal["id"],)).fetchone()
        if existing:
            return
        con.execute(
            "INSERT INTO calibrations (id, scope, video_id, corner_points, homography,"
            " focal_px, created_at) VALUES (?,?,?,?,?,?,?)",
            (cal["id"], cal["scope"], cal.get("video_id"),
             db.dumps(cal["corner_points"]), db.dumps(cal["homography"]),
             cal.get("focal_px"), cal["created_at"]))
    log.info("cloud ingest: synced global calibration %s from S3", cal["id"])


def _sync_tracknet_weights() -> None:
    """Mirrors a fine-tuned TrackNet checkpoint from S3 into this instance's
    own data/models/tracknet_v2.pt — same reasoning as _sync_calibration:
    the weights are produced/placed on a dev machine
    (scripts/upload_tracknet_weights.py), not through this worker, so
    without this it would never pick them up. Cheap no-op once the local
    file matches (size check, same as download_to_scratch's dedup); only
    engages ball_tracknet.weights_path() if TRACKNET_ENABLED is also set —
    this just makes sure the file is THERE when it is."""
    try:
        got = storage_s3.download_tracknet_weights(DATA_DIR / "models" / "tracknet_v2.pt")
    except Exception:
        log.exception("cloud ingest: tracknet weights S3 fetch failed (non-fatal, retrying next tick)")
        return
    if got:
        log.info("cloud ingest: synced TrackNet weights from S3")


def _sync_refine_config() -> None:
    """Mirrors the applied refine config.override.yaml (serve_gate_min etc.,
    see api/annotate.py's write_refine_override) from S3 into this instance's
    own data/refine/config.override.yaml — same reasoning as _sync_calibration:
    normally applied through a separate RUN_PROCESSING=0 UI-only container.
    refine.config.load_config re-reads this file on every Pass 2/refine call
    (see pipeline.py), so no restart is needed once it lands here — just a
    poll tick. Content-compared (no natural version/id here) so this is a
    no-op once the local file already matches."""
    try:
        remote = storage_s3.download_refine_config()
    except Exception:
        log.exception("cloud ingest: refine config S3 fetch failed (non-fatal, retrying next tick)")
        return
    if remote is None:
        return
    path = override_path()
    current = yaml.safe_load(path.read_text()) if path.exists() else None
    if current == remote:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(remote, sort_keys=False))
    log.info("cloud ingest: synced refine config override from S3: %s", remote)


def _sync_self_stop() -> None:
    """Mirrors the self-stop toggle from S3 into this instance's own local
    state (self_stop.py) — same reasoning as _sync_calibration: normally
    flipped through a separate RUN_PROCESSING=0 UI-only container. Read
    directly by self_stop.stop_this_instance() right before it would
    otherwise call ec2:StopInstances, so this poll cycle is what lets a
    toggle flipped mid-testing actually take effect on the worker."""
    try:
        remote = storage_s3.download_self_stop_state()
    except Exception:
        log.exception("cloud ingest: self-stop toggle S3 fetch failed (non-fatal, retrying next tick)")
        return
    if remote is None:
        return
    enabled = bool(remote.get("enabled", True))
    if self_stop.get_enabled() == enabled:
        return
    self_stop.set_enabled(enabled)
    log.info("cloud ingest: synced self-stop toggle from S3: enabled=%s", enabled)


def _ensure_selection_rows(video_id: str, selections: list[dict]) -> list[dict]:
    """Upserts coaching_selections (+ user_drawn rally_labels) from an S3
    Pass 2 work request onto this worker's local DB — the UI container that
    created them has a separate DB, so without this the worker would have
    nothing to process."""
    out = []
    with db.connect() as con:
        for s in selections:
            sid = s["id"]
            existing = con.execute(
                "SELECT id FROM coaching_selections WHERE id=?", (sid,)).fetchone()
            if not existing:
                con.execute(
                    "INSERT INTO coaching_selections (id, video_id, start_frame, end_frame, source,"
                    " status, rally_label_id, overlay_source, error_message, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (sid, video_id, s["start_frame"], s["end_frame"], s.get("source") or "pass1_accepted",
                     "pending", s.get("rally_label_id"), s.get("overlay_source"), None, db.now()))
            if s.get("source") == "user_drawn" and not s.get("rally_label_id"):
                rid = db.new_id()
                con.execute(
                    "INSERT INTO rally_labels (id, video_id, source, start_frame, end_frame, meta)"
                    " VALUES (?,?,?,?,?,?)",
                    (rid, video_id, "pass1", s["start_frame"], s["end_frame"],
                     db.dumps({"user_drawn": True})))
                con.execute("UPDATE coaching_selections SET rally_label_id=? WHERE id=?",
                            (rid, sid))
                s = {**s, "rally_label_id": rid}
            row = con.execute("SELECT * FROM coaching_selections WHERE id=?", (sid,)).fetchone()
            if row:
                out.append(dict(row))
    return out


def _ingest_pass2_request(video_id: str, req: dict) -> None:
    """Enqueues the matching Pass 2 job locally, then claims the S3 request
    (overwrite marker — this role often lacks s3:DeleteObject). Claim-after-
    enqueue means a crash between the two can double-submit once; JobRunner
    supersedes same (video_id, kind), so that's safe."""
    if req.get("claimed"):
        return
    mode = req.get("mode") or "selections"
    from .learning import params as learned_params
    from .jobs import runner
    _version, params = learned_params.get_latest()
    params = pipeline._with_hw_overrides(params)

    if mode == "selections":
        sels = _ensure_selection_rows(video_id, req.get("selections") or [])
        if not sels:
            log.warning("cloud ingest: pass2 request for %s had no selections", video_id)
            storage_s3.claim_pass2_request(video_id)
            return
        job_id = runner.submit(
            video_id, "pass2",
            pipeline._build_do_pass2_selections(video_id, params, sels))
        storage_s3.claim_pass2_request(video_id)
        log.info("cloud ingest: Pass 2 selections queued for %s job=%s (%d ranges)",
                 video_id, job_id, len(sels))
        return

    if mode == "single_range":
        sel = req.get("selection") or {}
        sels = _ensure_selection_rows(video_id, [sel])
        if not sels:
            log.warning("cloud ingest: pass2 single_range request for %s missing selection", video_id)
            storage_s3.claim_pass2_request(video_id)
            return
        s = sels[0]
        overlay_source = sel.get("overlay_source") or f"pass2_range_{s['id']}"
        job_id = runner.submit(
            video_id, f"pass2_range_{s['id']}",
            pipeline._build_do_pass2_range(video_id, params, s["id"],
                                            s["start_frame"], s["end_frame"], overlay_source))
        storage_s3.claim_pass2_request(video_id)
        log.info("cloud ingest: Pass 2 single_range queued for %s job=%s", video_id, job_id)
        return

    if mode == "full_pass2":
        job_id = runner.submit(video_id, "pass2", pipeline._build_do_pass2(video_id, params))
        storage_s3.claim_pass2_request(video_id)
        log.info("cloud ingest: full Pass 2 queued for %s job=%s", video_id, job_id)
        return

    log.warning("cloud ingest: unknown pass2 request mode %r for %s", mode, video_id)
    storage_s3.claim_pass2_request(video_id)


def _tick_pass2_requests() -> None:
    try:
        video_ids = storage_s3.list_pass2_request_video_ids()
    except Exception:
        log.exception("cloud ingest: pass2-request S3 list failed (non-fatal, retrying next tick)")
        return
    known = _known_video_ids()
    for video_id in video_ids:
        if video_id not in known:
            # Pass 1 may still be mid-ingest this tick; retry next cycle.
            log.info("cloud ingest: pass2 request for unknown video %s — waiting until ingested",
                     video_id)
            continue
        try:
            req = storage_s3.download_pass2_request(video_id)
            if req is None:
                continue
            _ingest_pass2_request(video_id, req)
        except Exception:
            log.exception("cloud ingest: failed to ingest pass2 request for %s "
                          "(non-fatal, retrying next tick)", video_id)


def _tick_kill_requests() -> None:
    """Handles api/media.py's DELETE /videos/{id} for a video that's actively
    processing on THIS worker — that endpoint's own JobRunner.cancel_video
    can only stop a job its own container started, so it relays a kill
    request here instead of reaching into this process directly. This is
    the only piece that can actually stop the computation (deleting S3
    files doesn't touch an already-downloaded local scratch copy a job is
    mid-read on — see storage_s3.download_to_scratch); the UI container
    deliberately leaves the video/results S3 objects alone until this runs,
    so cancel always happens before delete, never racing it.

    Checked before _tick_pass2_requests in _tick() — no point starting new
    work for a video that's about to be killed."""
    from .jobs import runner  # local import: same reasoning as _ingest_pass2_request's
    try:
        video_ids = storage_s3.list_kill_request_video_ids()
    except Exception:
        log.exception("cloud ingest: kill-request S3 list failed (non-fatal, retrying next tick)")
        return
    for video_id in video_ids:
        try:
            cancelled = runner.cancel_video(video_id)
            with db.connect() as con:
                db.delete_video_rows(video_id, con=con)
            storage_s3.delete_video_data(video_id)
            storage_s3.cleanup_scratch(video_id)
            storage_s3.delete_kill_request(video_id)
            log.info("cloud ingest: killed %s (%d local job(s) cancelled), S3 data removed",
                     video_id, cancelled)
        except Exception:
            log.exception("cloud ingest: failed to process kill request for %s "
                          "(non-fatal, retrying next tick)", video_id)


def _tick() -> None:
    _tick_kill_requests()
    _sync_calibration()
    _sync_tracknet_weights()
    _sync_refine_config()
    _sync_self_stop()
    try:
        keys = storage_s3.list_video_keys()
    except Exception:
        log.exception("cloud ingest: S3 list failed (non-fatal, retrying next tick)")
        return

    known = _known_video_ids()
    for key in keys:
        m = _KEY_RE.match(key)
        if not m:
            continue
        video_id = m.group(1)
        if video_id in known:
            continue
        try:
            # Defense against re-processing an already-finished video: the
            # dedup above relies entirely on THIS instance's own local DB
            # (videos it has ever ingested), which only protects against
            # duplicate work as long as that DB survives across this
            # instance's stop/start cycles. If it doesn't — e.g. the
            # container is recreated fresh on each boot rather than reusing
            # a persistent volume for DATA_DIR — every video ever uploaded
            # would look "new" again on the next boot, including ones fully
            # completed long ago. Checking S3 directly for a final
            # result.json is independent of this instance's own DB state,
            # so it still catches that case even if the root persistence
            # issue (outside this codebase — check how the container is
            # launched on boot) isn't fixed.
            if storage_s3.results_exist(video_id):
                log.info("cloud ingest: %s already has a final result.json in S3 (from a "
                          "prior run) — registering locally without reprocessing", video_id)
                _register_without_processing(video_id, key)
                continue
            _ingest_video(video_id, key)
        except Exception:
            # Must not kill the poller thread — a bad object today shouldn't
            # block every future video from ever being noticed. Retried
            # next tick since the video_id still won't be in `known`.
            log.exception("cloud ingest: failed to start %s (non-fatal, retrying next tick)",
                          video_id)

    # After video discovery so a request that arrived with a brand-new
    # upload can resolve once the video row exists (same tick if ingest
    # succeeded above; otherwise next tick).
    _tick_pass2_requests()


def _loop() -> None:
    while True:
        _tick()
        time.sleep(POLL_INTERVAL_S)


def start() -> None:
    threading.Thread(target=_loop, daemon=True, name="cloud-ingest-poller").start()
    log.info("cloud ingest poller started — checking s3://%s/%s/ every %ds",
              S3_VIDEO_BUCKET, storage_s3.DEFAULT_USER_ID, POLL_INTERVAL_S)

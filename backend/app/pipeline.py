"""Processing orchestration. Every upload loads the CURRENT global
LearnedParams and the effective CameraCalibration (clip-local override if
one exists, else the shared global one) before Pass 1 — this is how every
new clip automatically inherits all accumulated improvements."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import db, results_export, storage_s3
from .config import STORAGE_BACKEND
from .cv import overlays
from .cv.court import CameraModel
from .cv.pass1 import run_pass1
from .cv.pass2 import run_pass2
from .jobs import JobCancelled, JobHandle, runner
from .learning import params as learned_params
from .hw_profile import active_profile

log = logging.getLogger("pipeline")


def seed_cloud_placeholder(video_id: str) -> None:
    """RUN_PROCESSING=0 only — called from api/media.py's upload_video in
    place of start_processing(). Inserts one informational jobs row so the
    UI shows "queued/waiting" immediately after upload instead of looking
    like nothing happened (Workspace.jsx's activity panel and its
    active-job poll are both driven purely off the jobs table). Deliberately
    NOT submitted via runner.submit — that both records AND actually
    executes a job function on this container's own worker thread, which is
    exactly what RUN_PROCESSING=0 exists to prevent. mirror_cloud_chain_status
    and mark_cloud_pass2_active (called from the poll-cloud-windows /
    sync-from-cloud endpoints) keep this row honest as real progress comes
    back from the cloud worker."""
    with db.connect() as con:
        con.execute(
            "INSERT INTO jobs (id, video_id, kind, status, message, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (db.new_id(), video_id, "pass1", "queued",
             "waiting for cloud worker to pick this up", db.now(), db.now()))


def _upsert_mirror_job(video_id: str, kind: str, status: str, message: str,
                        progress: float | None = None) -> None:
    """Updates this video's most recent jobs row of `kind`, or inserts one if
    none exists yet — used only by the RUN_PROCESSING=0 mirroring helpers
    below, never by real job execution (that always goes through
    jobs.JobRunner, which owns its own row per submitted job). progress is
    optional since mirror_cloud_chain_status has no fraction at all (just a
    terminal status)."""
    with db.connect() as con:
        row = con.execute(
            "SELECT id FROM jobs WHERE video_id=? AND kind=? ORDER BY created_at DESC LIMIT 1",
            (video_id, kind)).fetchone()
        if row:
            if progress is not None:
                con.execute("UPDATE jobs SET status=?, progress=?, message=?, updated_at=? WHERE id=?",
                            (status, progress, message, db.now(), row["id"]))
            else:
                con.execute("UPDATE jobs SET status=?, message=?, updated_at=? WHERE id=?",
                            (status, message, db.now(), row["id"]))
        else:
            con.execute(
                "INSERT INTO jobs (id, video_id, kind, status, progress, message, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (db.new_id(), video_id, kind, status, progress if progress is not None else 0.0,
                 message, db.now(), db.now()))


def mirror_cloud_chain_status(video_id: str, chain_status: dict[str, dict]) -> None:
    """RUN_PROCESSING=0 only — called from sync-from-cloud once a full
    result.json is found. Reflects the cloud's definitive pass1/pass2/
    contacts status into this container's own jobs table (coaching isn't a
    jobs-table concept — see results_export._JOB_KINDS), so the existing
    jobs-table-driven "is this video active" UI logic keeps working
    unmodified even though this container never actually runs what it's
    displaying. Stages still 'not_run' on the cloud are left alone rather
    than overwritten, so an earlier optimistic mark_cloud_pass2_active nudge
    (below) isn't clobbered back to nothing before the real status lands."""
    for kind in ("pass1", "pass2", "contacts"):
        info = chain_status.get(kind)
        if not info or info["status"] == "not_run":
            continue
        _upsert_mirror_job(video_id, kind, info["status"], info.get("message") or "")


def mirror_cloud_progress(video_id: str, progress_payload: dict) -> None:
    """RUN_PROCESSING=0 only — called from poll-cloud-windows when it finds a
    live progress snapshot (see storage_s3.upload_progress /
    cv/pass1.py's periodic upload). Unlike mark_cloud_pass2_active below,
    this carries a real progress fraction, so the local UI shows genuine
    "Pass 1: N rallies so far, 47%" status instead of just "waiting for
    cloud worker" for however long Pass 1 takes. The snapshot object is
    deleted by the cloud side once Pass 1 finishes (see
    storage_s3.delete_progress) specifically so this never fires again with
    stale 'running' state after the video has moved on — mirror_cloud_progress
    itself always trusts whatever payload it's given without re-checking
    staleness, by design; that's delete_progress's job, not this function's.

    Also merges any rally_labels riding along in the same snapshot (see
    cv/pass1.py's _upload_progress_incremental) — this is what lets the
    Pass 1 timeline track populate progressively on a UI-only container
    instead of staying empty until the full chain-end sync."""
    stage = progress_payload.get("stage", "pass1")
    rally_labels = progress_payload.get("rally_labels")
    if rally_labels is not None:
        results_export.merge_rally_labels(video_id, stage, rally_labels)
    _upsert_mirror_job(video_id, stage,
                        progress_payload.get("status", "running"),
                        progress_payload.get("message", ""),
                        progress=progress_payload.get("progress", 0.0))


def mark_cloud_pass2_active(video_id: str, window_index: int, total_windows: int | None) -> None:
    """RUN_PROCESSING=0 only — called from poll-cloud-windows whenever it
    merges at least one new window. Window payloads carry no chain_status
    (see results_export.export_window_results), so unlike
    mirror_cloud_chain_status this isn't reading a real status field — it's
    inferring "Pass 2 must be running" purely from the fact that a window
    landed, to keep the UI showing real activity in the gap between the
    upload-time placeholder and the eventual definitive status from a full
    sync. sync-from-cloud's mirror_cloud_chain_status always supersedes this
    once the real result.json lands.

    total_windows (from the same window payload — see cv/pass2.py's
    _upload_window_incremental) lets this set a real (window_index+1)/
    total_windows fraction instead of leaving progress untouched: without
    it, only the message updated ("window N synced") while the mirrored
    job's progress column stayed at whatever it was seeded at (0.0),
    so the local Pass 2 progress bar's width never moved even though
    windows kept landing. Falls back to leaving progress alone only if a
    caller genuinely has no total (shouldn't happen once every window
    payload carries one)."""
    _upsert_mirror_job(video_id, "pass1", "done", "")
    message = f"Pass 2 processing in the cloud (window {window_index + 1} synced)"
    progress = (window_index + 1) / total_windows if total_windows else None
    _upsert_mirror_job(video_id, "pass2", "running", message, progress=progress)


def _ensure_local_video(video_id: str) -> None:
    """S3 mode only: pass1.py/pass2.py read vrow["path"] as a real local file
    (they have no S3 awareness, by design — see storage_s3.py docstring).
    That local file may be gone by the time CV code runs it: the upload-time
    scratch copy gets deleted once the pass1->pass2->contacts chain finishes
    (see the chain wiring below), and a fresh instance booted by the
    s3:ObjectCreated trigger has no scratch dir at all. So re-download here
    if needed, then repoint the DB row so pass1.py/pass2.py keep working
    unmodified."""
    if STORAGE_BACKEND != "s3":
        return
    with db.connect() as con:
        vrow = con.execute("SELECT path, s3_key FROM videos WHERE id=?", (video_id,)).fetchone()
    path = Path(vrow["path"])
    if path.exists() and path.stat().st_size > 0:
        return
    if not vrow["s3_key"]:
        raise RuntimeError(f"video {video_id}: STORAGE_BACKEND=s3 but no s3_key recorded")
    local = storage_s3.download_to_scratch(vrow["s3_key"], video_id)
    with db.connect() as con:
        con.execute("UPDATE videos SET path=? WHERE id=?", (str(local), video_id))


def _finish_chain(video_id: str, *, run_contacts: bool = True,
                   status_override: dict[str, dict] | None = None) -> None:
    """End of the automatic pass1->pass2->contacts chain. Coaching reports
    run later, on-demand (api/rally_analysis.py), off already-stored DB data
    — no local video access — so it's safe to clean up here regardless.

    In S3 mode, _finish_chain runs synchronously inside the "contacts" job's
    fn (see the runner.submit call in start_processing/reprocess below) —
    the job runner only fires self_stop.stop_this_instance() AFTER job.fn()
    returns and the queue is empty (see jobs.py's _loop). So the export here
    is guaranteed to complete before this instance can be stopped; nothing
    further needs to await or block on self-stop. Exported/uploaded
    regardless of whether the chain fully succeeded — see
    results_export.py's docstring.

    run_contacts=False / status_override are used by _build_do_pass2's
    cancel/error handler (a partial-chain export, not the normal completion
    path — see there): contacts detection is skipped because
    readiness.get_analyzable_rallies() gates on the pass2 job showing
    status='done', so it would find zero analyzable rallies for a
    cancelled/errored run anyway — and contact_detect.save_events
    unconditionally DELETEs this video's existing contact_events before
    inserting, which would silently wipe a prior good Pass 2 + contacts run
    for no benefit. status_override lets the caller assert pass2's real
    status (see results_export.export_video_results's docstring for why the
    jobs table can't be trusted yet at that point).

    On the normal completion path (run_contacts=True), the SAME "jobs table
    can't be trusted yet" problem applies to the contacts job's own status,
    not just pass2's: jobs.py's _loop only flips this job's DB row to 'done'
    AFTER this whole function returns, and _export_results below runs
    BEFORE that return — so a live DB read there is guaranteed to see
    'running' for contacts on every single successful chain, not
    occasionally. Confirmed for real: a live end-to-end run's exported
    result.json showed contacts stuck at 'running' despite 214 real
    contact_events already stored — the detection had genuinely finished,
    only the exported label was stale, and since this export is the ONE
    chain-end snapshot (nothing re-exports after), that stale label would
    otherwise persist forever in every synced copy. Asserted explicitly
    here instead, same pattern as the cancel/error paths."""
    if run_contacts:
        _detect_contacts_after_pass2(video_id)
        status_override = {**(status_override or {}),
                           "contacts": {"status": "done", "message": ""}}
    if STORAGE_BACKEND == "s3":
        _export_results(video_id, status_override=status_override)
        storage_s3.cleanup_scratch(video_id)


def _export_results(video_id: str, status_override: dict[str, dict] | None = None) -> None:
    try:
        payload = results_export.export_video_results(video_id, status_override=status_override)
        storage_s3.upload_results(video_id, payload)
    except Exception:
        # Must not block cleanup/self-stop — a failed export is a lost
        # result, not a reason to strand the instance running (billed) with
        # no way to retry from here. Visible in docker logs either way.
        log.exception("cloud results export failed for %s (non-fatal)", video_id)


def _with_hw_overrides(params: dict) -> dict:
    """Overlay the active hardware profile's p1_scale/p1_frame_step (if set)
    onto a COPY of the LearnedParams dict for this run only. Never written
    back to learned_params — these are speed knobs, not learned tunables
    (see hw_profile.py); autolearn's stored baseline is untouched."""
    profile = active_profile()
    if not profile:
        return params
    merged = dict(params)
    for key in ("p1_scale", "p1_frame_step"):
        if profile.get(key) is not None:
            merged[key] = learned_params.clamp(key, profile[key])
    return merged


def effective_calibration(video_id: str):
    """Returns (H | None, CameraModel | None, calibration_id | None)."""
    with db.connect() as con:
        vrow = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        cal = None
        if vrow and vrow["calibration_id"]:
            cal = con.execute("SELECT * FROM calibrations WHERE id=?",
                              (vrow["calibration_id"],)).fetchone()
        if cal is None:
            cal = con.execute(
                "SELECT * FROM calibrations WHERE scope='global'"
                " ORDER BY created_at DESC LIMIT 1").fetchone()
    if cal is None or vrow is None:
        return None, None, None
    H = np.array(db.loads(cal["homography"]))
    cam = CameraModel.from_homography(H, vrow["width"], vrow["height"])
    if cal["focal_px"]:
        cam.focal_px = float(cal["focal_px"])
    return H, cam, cal["id"]


def _window_refiner(video_id: str):
    """Best-effort per-window refine hook for run_pass2: refines each Pass 1
    window as soon as its overlays land, so verdicts + refined boundaries show
    up in the UI while the pass is still scanning. Calling the hook with -1
    just clears any stale refine JSON (done before the pass starts). The full
    refine after the pass remains the canonical output. Never fails the pass."""
    state: dict = {"r": None}

    def hook(upto_frame: int) -> None:
        try:
            if state["r"] is None:
                from .refine.config import load_config, override_path
                from .refine.refine import IncrementalRefiner
                state["r"] = IncrementalRefiner(video_id,
                                                load_config(override_path()))
            state["r"].advance(upto_frame)
        except Exception:
            log.exception("incremental refine failed (non-fatal)")

    return hook


def _refine_after_pass2(video_id: str) -> None:
    """Rally refinement over the fresh Pass 2 tracking data (pass2/refine
    module). JSON-only output for now — it writes data/refine/<id>.json plus
    diagnostics and NO rally_labels rows, so Pass 1 stays the app's boundary
    source until the refiner is validated. Never fails the pass."""
    try:
        from .refine import run_refine_json
        from .refine.config import load_config, override_path
        cfg = load_config(override_path())  # picks up any applied serve-gate override
        out = run_refine_json(video_id, cfg, plot=True)
        log.info("rally refinement written to %s", out)
    except Exception:
        log.exception("rally refinement failed (non-fatal)")


def _detect_contacts_after_pass2(video_id: str) -> None:
    """Contact detection over the fresh Pass 2 overlays. Readiness-gated (see
    rallyanalysis/readiness.py) on the jobs table showing this video's most
    recent pass2 job as 'done' — but JobRunner only flips jobs.status to
    'done' AFTER the do_pass2 closure returns, so this must NOT be called
    synchronously inside do_pass2 (it would always see 'running' and detect
    nothing). Instead it's submitted as its own chained job (same pattern
    do_pass1 already uses to chain do_pass2), so by the time the single
    worker thread gets to it, the pass2 job row is already finalized to
    'done'. Never fails the pass — a bug here must not take down someone's
    Pass 2 job, and running as a separate job (rather than inline) makes
    that structurally true, not just try/except-shaped."""
    try:
        from .rallyanalysis import contact_detect
        events = contact_detect.detect_contacts_for_video(video_id)
        contact_detect.save_events(video_id, events)
        log.info("contact detection stored %d events for %s", len(events), video_id)
    except Exception:
        log.exception("contact detection failed (non-fatal)")


def _build_do_pass2(video_id: str, params: dict):
    """Shared by start_processing (chained after Pass 1) and reprocess
    (standalone 'pass2' re-run) so the cancel/error handling below only has
    to be written — and reviewed — once.

    run_pass2 raises JobCancelled (via handle.update()'s cooperative check,
    called roughly every 100 frames — see jobs.py) when the user hits
    Cancel; that exception propagates straight out of this function, so the
    "if not handle.cancelled(): runner.submit(...)" line below it never
    runs. That used to mean the "contacts" job — the only thing that calls
    _finish_chain, where the S3 export lives — never got submitted, so any
    overlay data Pass 2 already flushed for completed windows was stranded:
    on a cloud instance, self-stop still fires once the queue empties, with
    nothing exported. Same gap for a genuine mid-run failure (bad frame,
    OOM, etc.) — jobs.py's `except Exception` branch caught it but nothing
    chained the export there either.

    Fix: catch both here and export inline (not via a new queued job) before
    re-raising. Running inline still preserves the same export-before-
    self-stop ordering as the happy path: jobs.py's _loop only calls
    self_stop.stop_this_instance() after job.fn() (this function) returns OR
    raises, AND the queue is empty — so finishing the export here, before
    the raise, guarantees it completes first either way. The exception is
    re-raised unchanged afterward so jobs.py's own status bookkeeping
    ('cancelled' vs 'error') behaves exactly as before this change."""

    def do_pass2(handle: JobHandle) -> None:
        _ensure_local_video(video_id)
        H, cam, _ = effective_calibration(video_id)
        on_window = _window_refiner(video_id)
        on_window(-1)   # drop stale refine JSON before the pass starts
        try:
            run_pass2(video_id, params, H, cam, handle, on_window=on_window)
        except JobCancelled:
            _finish_chain(video_id, run_contacts=False,
                          status_override={"pass2": {"status": "cancelled",
                                                       "message": "cancelled by user"}})
            raise
        except Exception as exc:
            _finish_chain(video_id, run_contacts=False,
                          status_override={"pass2": {"status": "error",
                                                       "message": f"{type(exc).__name__}: {exc}"}})
            raise
        _refine_after_pass2(video_id)
        # Chained as its own job (not called inline) so it only runs once the
        # jobs table shows this pass2 job as 'done' — see
        # _detect_contacts_after_pass2's docstring.
        if not handle.cancelled():
            runner.submit(video_id, "contacts", lambda h: _finish_chain(video_id))

    return do_pass2


def _set_selection_status(selection_id: str, status: str, error_message: str | None = None) -> None:
    with db.connect() as con:
        con.execute(
            "UPDATE coaching_selections SET status=?, error_message=? WHERE id=?",
            (status, error_message, selection_id))


def _build_do_pass2_range(video_id: str, params: dict, selection_id: str,
                           start_frame: int, end_frame: int, overlay_source: str):
    """Runs Pass 2 over a single user-specified (video_id, start_frame,
    end_frame) range that does NOT come from a rally_labels row — a manual
    coaching selection. Unlike _build_do_pass2:
      - writes overlays under `overlay_source` (never "pass2" — that would
        wipe the real Pass 1-driven run for this video).
      - does NOT chain _refine_after_pass2 or a "contacts" job: both
        ultimately depend on rally_labels rows (refine reads Pass 1 windows;
        contacts is gated by readiness.get_analyzable_rallies, which only
        enumerates rally_labels rows), so chaining them here would silently
        do nothing for this range.
      - verifies overlay data actually landed before marking the selection
        'ready' (see module-level note in the caller) instead of treating a
        no-op run as success.

    TODO: once contacts-for-manual-ranges is built, it should call
    contact_detect.detect_contacts_for_rally(video_id, selection_id, entries,
    fps) directly for this range, bypassing readiness.get_analyzable_rallies
    entirely (that gate will never see a rally_labels-less range)."""
    # Guard, not just a comment: run_pass2 unconditionally does
    # overlays.clear(video_id, source) at the top of every run. The one
    # current caller (api/rally_analysis.py's process_selection) always
    # builds overlay_source as f"pass2_range_{selection_id}", so this never
    # fires today — but nothing inside this function enforced that on its
    # own, and a future call site (or a refactor) passing overlay_source=
    # "pass2"/None/"" here would silently wipe the real Pass 1-driven Pass 2
    # overlay for this video with no error. Fail loudly at build time
    # instead, before a job is even queued.
    if not overlay_source or overlay_source == "pass2":
        raise ValueError(
            f"_build_do_pass2_range: overlay_source must be a distinct, "
            f"non-empty string, got {overlay_source!r} for video {video_id!r} "
            f"— refusing to risk clearing the real 'pass2' overlay source")

    def do_pass2_range(handle: JobHandle) -> None:
        _ensure_local_video(video_id)
        H, cam, _ = effective_calibration(video_id)
        _set_selection_status(selection_id, "processing")
        try:
            run_pass2(video_id, params, H, cam, handle, source=overlay_source,
                      windows_override=[(start_frame, end_frame)])
        except JobCancelled:
            _set_selection_status(selection_id, "failed", "cancelled by user")
            raise
        except Exception as exc:
            _set_selection_status(selection_id, "failed", f"{type(exc).__name__}: {exc}")
            raise

        # Completeness check: a job that finishes without raising but wrote
        # no overlay data for this range must not look like a real success
        # (e.g. a degenerate/zero-length range, or every frame silently
        # dropped upstream) — mark it failed instead of ready.
        entries = overlays.read_range(video_id, overlay_source, start_frame, end_frame)
        if not entries:
            _set_selection_status(
                selection_id, "failed",
                "pass2 completed but produced no overlay data for this range")
            return
        _set_selection_status(selection_id, "ready")

    return do_pass2_range


def start_processing(video_id: str) -> dict:
    """Kick off Pass 1 immediately (background); Pass 2 chains after it."""
    version, params = learned_params.get_latest()
    params = _with_hw_overrides(params)
    with db.connect() as con:
        con.execute("UPDATE videos SET learned_params_version=? WHERE id=?",
                    (version, video_id))

    def do_pass1(handle: JobHandle) -> None:
        _ensure_local_video(video_id)
        H, _, _ = effective_calibration(video_id)
        try:
            run_pass1(video_id, params, H, handle)
        finally:
            # Covers normal completion AND any exception (JobCancelled or a
            # real failure) — run_pass1's periodic progress snapshot (see
            # cv/pass1.py's _upload_progress_incremental) must never be left
            # behind showing stale 'running' state once this job is no
            # longer actually running, or a later poll-cloud-windows tick
            # could re-mirror it and flip an already-finalized local job
            # status back to 'running' — see storage_s3.delete_progress.
            if STORAGE_BACKEND == "s3":
                storage_s3.delete_progress(video_id)
        # Chain Pass 2 automatically (unless Pass 1 was cancelled). It will
        # fail with a clear message if the court is not calibrated yet.
        if not handle.cancelled():
            runner.submit(video_id, "pass2", _build_do_pass2(video_id, params))

    job_id = runner.submit(video_id, "pass1", do_pass1)
    return {"job_id": job_id, "learned_params_version": version}


def reprocess(video_id: str, which: str) -> dict:
    """Re-run with the CURRENT params/calibration (used by Regenerate and
    after calibration). which: 'pass1' (chains pass2) or 'pass2'."""
    if which == "pass1":
        return start_processing(video_id)
    version, params = learned_params.get_latest()
    params = _with_hw_overrides(params)
    job_id = runner.submit(video_id, "pass2", _build_do_pass2(video_id, params))
    return {"job_id": job_id, "learned_params_version": version}

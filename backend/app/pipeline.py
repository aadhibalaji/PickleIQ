"""Processing orchestration. Every upload loads the CURRENT global
LearnedParams and the effective CameraCalibration (clip-local override if
one exists, else the shared global one) before Pass 1 — this is how every
new clip automatically inherits all accumulated improvements."""
from __future__ import annotations

import logging

import numpy as np

from . import db
from .cv.court import CameraModel
from .cv.pass1 import run_pass1
from .cv.pass2 import run_pass2
from .jobs import JobHandle, runner
from .learning import params as learned_params

log = logging.getLogger("pipeline")


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


def start_processing(video_id: str) -> dict:
    """Kick off Pass 1 immediately (background); Pass 2 chains after it."""
    version, params = learned_params.get_latest()
    with db.connect() as con:
        con.execute("UPDATE videos SET learned_params_version=? WHERE id=?",
                    (version, video_id))

    def do_pass1(handle: JobHandle) -> None:
        H, _, _ = effective_calibration(video_id)
        run_pass1(video_id, params, H, handle)
        # Chain Pass 2 automatically (unless Pass 1 was cancelled). It will
        # fail with a clear message if the court is not calibrated yet.
        if not handle.cancelled():
            runner.submit(video_id, "pass2", do_pass2)

    def do_pass2(handle: JobHandle) -> None:
        H, cam, _ = effective_calibration(video_id)
        on_window = _window_refiner(video_id)
        on_window(-1)   # drop stale refine JSON before the pass starts
        run_pass2(video_id, params, H, cam, handle, on_window=on_window)
        _refine_after_pass2(video_id)
        # Chained as its own job (not called inline) so it only runs once the
        # jobs table shows this pass2 job as 'done' — see
        # _detect_contacts_after_pass2's docstring.
        if not handle.cancelled():
            runner.submit(video_id, "contacts", lambda h: _detect_contacts_after_pass2(video_id))

    job_id = runner.submit(video_id, "pass1", do_pass1)
    return {"job_id": job_id, "learned_params_version": version}


def reprocess(video_id: str, which: str) -> dict:
    """Re-run with the CURRENT params/calibration (used by Regenerate and
    after calibration). which: 'pass1' (chains pass2) or 'pass2'."""
    if which == "pass1":
        return start_processing(video_id)
    version, params = learned_params.get_latest()

    def do_pass2(handle: JobHandle) -> None:
        H, cam, _ = effective_calibration(video_id)
        on_window = _window_refiner(video_id)
        on_window(-1)   # drop stale refine JSON before the pass starts
        run_pass2(video_id, params, H, cam, handle, on_window=on_window)
        _refine_after_pass2(video_id)
        # Chained as its own job (not called inline) so it only runs once the
        # jobs table shows this pass2 job as 'done' — see
        # _detect_contacts_after_pass2's docstring.
        if not handle.cancelled():
            runner.submit(video_id, "contacts", lambda h: _detect_contacts_after_pass2(video_id))

    job_id = runner.submit(video_id, "pass2", do_pass2)
    return {"job_id": job_id, "learned_params_version": version}

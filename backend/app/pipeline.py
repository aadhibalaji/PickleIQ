"""Processing orchestration. Every upload loads the CURRENT global
LearnedParams and the effective CameraCalibration (clip-local override if
one exists, else the shared global one) before Pass 1 — this is how every
new clip automatically inherits all accumulated improvements."""
from __future__ import annotations

import logging

import numpy as np

from . import db
from .cv import audio_events
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
        run_pass2(video_id, params, H, cam, handle)

    # Audio onset extraction has no calibration dependency and is cheap
    # (~4% of realtime). Submit it FIRST so the single worker finishes it
    # before Pass 2 (which fuses it) chains off Pass 1. Idempotent per video.
    runner.submit(video_id, "audio",
                  lambda h: audio_events.compute_and_store(video_id, params))
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
        run_pass2(video_id, params, H, cam, handle)

    # Ensure audio onsets exist for a pass2-only rerun (idempotent skip if so).
    runner.submit(video_id, "audio",
                  lambda h: audio_events.compute_and_store(video_id, params))
    job_id = runner.submit(video_id, "pass2", do_pass2)
    return {"job_id": job_id, "learned_params_version": version}

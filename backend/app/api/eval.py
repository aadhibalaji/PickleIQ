"""Ball-eval runs: kick off the ball_filter harness as a background job and
read back its stored results.

The harness sweeps every frame of every rally window (twice, when --diagnose
renders the failure gallery), so it never runs inside a request handler — it
goes through the same single-worker JobRunner as pass1/pass2, under the job
kind 'ball_eval'. The frontend polls GET /videos/{id} for job status exactly
as it does for the passes, then re-lists the runs when the job finishes.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import db
from ..analysis.ball_eval import (DEFAULT_TRACKNET_WEIGHTS, EvalOptions,
                                  EvalSkipped, eval_video, headline, json_safe,
                                  lever_for, load_video_row, resolve_params)
from ..jobs import JobHandle, runner

router = APIRouter()

# `tag` names a directory under data/eval_diag/<vid8>/ AND a path segment of the
# static gallery URL, so it must not escape either. Reject anything but a plain
# label rather than trying to sanitize it.
_SAFE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")

# Columns the list endpoint reads — deliberately never `result`, which can be
# megabytes once diagnose attaches slices and the bounce-event trace.
_LIST_COLUMNS = ("id, video_id, tag, detector, params_version, diagnose, status,"
                 " message, n_marks, det_rate, pos_err_med_px, bounce_p,"
                 " bounce_r, inout_acc, created_at")


class EvalRunRequest(BaseModel):
    detector: str = "classical"          # 'classical' | 'tracknet'
    diagnose: bool = True
    tag: str = "run"
    params_version: int = 0              # 0 = latest
    param_overrides: dict[str, float] = Field(default_factory=dict)
    pos_frame_tol: int = 1
    bounce_frame_tol: int = 5
    bounce_dist_m: float = 1.0


def _insert(video_id: str, req: EvalRunRequest, version: int,
            status: str, message: str, m: dict | None) -> None:
    h = headline(m) if m else {}
    with db.connect() as con:
        con.execute(
            "INSERT INTO ball_eval (id, video_id, tag, detector, params_version,"
            " diagnose, status, message, n_marks, det_rate, pos_err_med_px,"
            " bounce_p, bounce_r, inout_acc, result, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (db.new_id(), video_id, req.tag, req.detector, version,
             int(req.diagnose), status, message,
             h.get("n_marks"), h.get("det_rate"), h.get("pos_err_med_px"),
             h.get("bounce_p"), h.get("bounce_r"), h.get("inout_acc"),
             db.dumps(json_safe(m)) if m else None, db.now()),
        )


@router.post("/videos/{video_id}/eval/run")
def run_eval(video_id: str, body: EvalRunRequest):
    if body.detector not in ("classical", "tracknet"):
        raise HTTPException(400, "detector must be classical or tracknet")
    if not _SAFE_TAG.match(body.tag):
        raise HTTPException(400, "tag must be a short alphanumeric label "
                                 "([A-Za-z0-9][A-Za-z0-9_.-]*)")
    with db.connect() as con:
        if con.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is None:
            raise HTTPException(404, "video not found")

    def job(handle: JobHandle) -> None:
        version, params = resolve_params(body.params_version, body.param_overrides)
        opts = EvalOptions(
            detector=body.detector, tracknet_weights=str(DEFAULT_TRACKNET_WEIGHTS),
            diagnose=body.diagnose, tag=body.tag,
            pos_frame_tol=body.pos_frame_tol,
            bounce_frame_tol=body.bounce_frame_tol,
            bounce_dist_m=body.bounce_dist_m,
            param_overrides=body.param_overrides)
        try:
            v = load_video_row(video_id)
            # handle.update() raises JobCancelled when the user cancels, which
            # unwinds the frame sweep — do NOT catch it here.
            m = eval_video(v, params, opts,
                           on_progress=lambda frac, msg: handle.update(frac, msg))
        except EvalSkipped as exc:
            # Not an error: the clip legitimately can't be evaluated yet (no
            # calibration, no rally windows, no TrackNet weights). Record it so
            # the UI can say why instead of showing an empty list.
            _insert(video_id, body, 0, "skipped", str(exc), None)
            handle.update(1.0, f"skipped: {exc}")
            return
        _insert(video_id, body, version, "ok", "", m)

    job_id = runner.submit(video_id, "ball_eval", job)
    return {"job_id": job_id, "kind": "ball_eval"}


@router.get("/videos/{video_id}/eval")
def list_evals(video_id: str):
    """Past runs for a clip, newest first. Headline metrics only."""
    with db.connect() as con:
        rows = db.rows_to_dicts(con.execute(
            f"SELECT {_LIST_COLUMNS} FROM ball_eval WHERE video_id=?"
            " ORDER BY created_at DESC LIMIT 50", (video_id,)).fetchall())
    for r in rows:
        r["diagnose"] = bool(r["diagnose"])
    return rows


@router.get("/videos/{video_id}/eval/{eval_id}")
def get_eval(video_id: str, eval_id: str):
    """One run in full: the stored eval_video() payload, plus the failure
    attribution re-sorted and joined to its suggested lever (done here rather
    than at write time so older rows pick up lever-table changes)."""
    with db.connect() as con:
        row = con.execute("SELECT * FROM ball_eval WHERE id=? AND video_id=?",
                          (eval_id, video_id)).fetchone()
    if row is None:
        raise HTTPException(404, "eval run not found")
    out = dict(row)
    out["diagnose"] = bool(out["diagnose"])
    result = db.loads(out.pop("result"), None)
    out["result"] = result

    diag = (result or {}).get("diag") or {}
    out["attribution"] = [
        {"category": cat, "count": n, "lever": lever_for(cat)}
        for cat, n in sorted((diag.get("failures") or {}).items(),
                             key=lambda kv: -kv[1])]
    out["bounce_attribution"] = [
        {"category": cat, "count": n, "lever": lever_for(cat)}
        for cat, n in sorted((diag.get("bounce_failures") or {}).items(),
                             key=lambda kv: -kv[1])]
    return out

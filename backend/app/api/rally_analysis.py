"""Rally Analysis & Coaching endpoints. Read-only against rally_labels,
overlays, ball_corrections/ball_tracks and calibration data — never writes
to those tables. New tables only: contact_events, coaching_reports,
coaching_marks.

Endpoints:
  GET  /rally-analysis/health                       liveness check
  POST /rally-analysis/{video_id}/detect-contacts    pose-based contact detection -> contact_events
  GET  /rally-analysis/{video_id}/contact-events     list stored contact_events
  GET  /rally-analysis/{video_id}/readiness          per-rally Pass 2 readiness (waiting/stale/ready)
  POST /rally-analysis/{video_id}/coaching-report    generate report (1 Anthropic call) -> coaching_reports
  GET  /rally-analysis/{video_id}/coaching-report    cached report, or {"report": null}
  GET  /rally-analysis/{video_id}/coaching-debug     per-metric decision timeline (no LLM call, nothing persisted)
  GET  /rally-analysis/{video_id}/coaching-marks     expert ground-truth moment labels
  POST /rally-analysis/{video_id}/coaching-marks     store one expert verdict over a frame span
  DEL  /rally-analysis/coaching-marks/{mark_id}      remove a mark
  GET  /rally-analysis/{video_id}/report-stats       forward-momentum / kitchen-arrival / shot-zone aggregate stats
  GET  /rally-analysis/{video_id}/trajectories       per-rally 3D ball trajectory samples (ballistic reconstruction)
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from .. import db
from ..rallyanalysis import coaching, contact_detect, report_stats, trajectory
from ..rallyanalysis.contracts import CoachingReport
from ..rallyanalysis.readiness import readiness_report

router = APIRouter()


def _report_payload(report: CoachingReport) -> dict:
    tips = [asdict(t) for t in report.tips]
    general = [asdict(g) for g in report.general]
    # Union: a player can carry general feedback without landing any per-shot
    # tip — they must still appear in the UI's player filter tabs.
    players = sorted({t["player_label"] for t in tips}
                     | {g["player_label"] for g in general})
    return {"video_id": report.video_id, "tips": tips, "players": players,
            "player_themes": [asdict(t) for t in report.player_themes],
            "general_feedback": general,
            "focus": report.focus,
            "stale_reason": report.stale_reason}


@router.get("/rally-analysis/health")
def health():
    return {"status": "ok"}


@router.post("/rally-analysis/{video_id}/process-single-rally")
def process_single_rally(video_id: str):
    """Coaching-only fast path for a PRE-TRIMMED clip containing exactly one
    rally: writes a pass1-source rally label spanning the whole clip
    (replacing any prior pass1 labels for this video — that's why this must
    only be used on single-rally clips, never a full match), then submits
    Pass 2 directly via pipeline.reprocess. Pass 2 reads its processing
    windows from pass1-source rally_labels (see cv/pass2.py run_pass2), so
    this skips Pass 1's rally detection entirely while the rest of the chain
    (pass2 overlays -> contacts job -> readiness gate -> Generate report)
    runs completely unchanged. Court calibration is still required, exactly
    as for a normal Pass 2 run."""
    from .. import pipeline
    with db.connect() as con:
        vrow = con.execute("SELECT frame_count FROM videos WHERE id=?", (video_id,)).fetchone()
        if vrow is None:
            raise HTTPException(status_code=404, detail="video not found")
        if not vrow["frame_count"]:
            raise HTTPException(status_code=409, detail="video has no frame_count recorded")
        con.execute("DELETE FROM rally_labels WHERE video_id=? AND source='pass1'", (video_id,))
        con.execute(
            "INSERT INTO rally_labels (id, video_id, source, start_frame, end_frame, meta)"
            " VALUES (?,?,?,?,?,?)",
            (db.new_id(), video_id, "pass1", 0, int(vrow["frame_count"]) - 1,
             db.dumps({"whole_clip_rally": True})))
    result = pipeline.reprocess(video_id, "pass2")
    return {"video_id": video_id, "rally_frames": [0, int(vrow["frame_count"]) - 1], **result}


@router.post("/rally-analysis/{video_id}/detect-contacts")
def detect_contacts(video_id: str, source: str | None = None):
    """Runs pose-only contact detection for every rally of `source` on this
    video and persists the result to contact_events (replacing any prior run
    for this video)."""
    events = contact_detect.detect_contacts_for_video(video_id, rally_source=source)
    contact_detect.save_events(video_id, events)
    return {"video_id": video_id, "source": source, "count": len(events),
            "events": [asdict(e) for e in events]}


@router.get("/rally-analysis/{video_id}/contact-events")
def list_contact_events(video_id: str):
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM contact_events WHERE video_id=? ORDER BY frame_idx",
            (video_id,)).fetchall()
    return {"video_id": video_id, "events": db.rows_to_dicts(rows)}


@router.get("/rally-analysis/{video_id}/readiness")
def readiness(video_id: str):
    """Per-rally Pass 2 readiness, so the frontend can show a 'waiting on
    Pass 2' state instead of silently showing nothing or stale data.
    {rally_id: {"ready": bool, "reason": "pass2_not_done"|"pass2_running"|
    "overlay_stale"|"ready"}}."""
    return readiness_report(video_id)


@router.post("/rally-analysis/{video_id}/coaching-report")
def generate_coaching_report(video_id: str, source: str | None = None,
                              focus: str | None = None):
    """Triggers coaching report generation (one Anthropic API call) and
    persists it to coaching_reports, replacing any prior report for this
    video. Not run automatically anywhere — explicit user action only.
    focus overrides the video's stored coaching_focus for this generation
    only ('form' | 'positioning'); omitted = the stored value, then form."""
    if focus is not None and focus not in (coaching.FOCUS_FORM, coaching.FOCUS_POSITIONING):
        raise HTTPException(status_code=422,
                            detail="focus must be 'form' or 'positioning'")
    report = coaching.generate_coaching_report(video_id, rally_source=source, focus=focus)
    return _report_payload(report)


@router.get("/rally-analysis/{video_id}/coaching-report")
def get_coaching_report(video_id: str):
    """Returns the cached report, or null if none has been generated yet —
    regeneration only happens via the POST endpoint."""
    report = coaching.get_report(video_id)
    return {"video_id": video_id, "report": _report_payload(report) if report else None}


@router.get("/rally-analysis/{video_id}/coaching-debug")
def get_coaching_debug(video_id: str, source: str | None = None, focus: str | None = None):
    """Diagnostic view into HOW a coaching decision gets made: every scored
    moment (not just the ones that made a report), each metric's raw value
    and deviation, the flag threshold, and which moments were selected as
    LLM candidates. Never calls the LLM — pure deterministic computation,
    safe to call repeatedly while iterating on thresholds/bands. See
    coaching.debug_timeline for the payload shape."""
    if focus is not None and focus not in (coaching.FOCUS_FORM, coaching.FOCUS_POSITIONING):
        raise HTTPException(status_code=422,
                            detail="focus must be 'form' or 'positioning'")
    return coaching.debug_timeline(video_id, rally_source=source, focus=focus)


@router.post("/rally-analysis/{video_id}/tip-feedback")
def submit_tip_feedback(video_id: str, payload: dict):
    """Stores one thumbs verdict for one tip: {contact_id, rating: 1|-1,
    category?, tone?, tip?}. Re-rating the same tip replaces the prior row
    (PRIMARY KEY (video_id, contact_id)); category/tone/tip text are
    snapshotted so the signal stays analyzable after the report is
    regenerated. This is the measurement layer for tip-selection quality —
    see get_tip_feedback's aggregate."""
    contact_id = payload.get("contact_id")
    rating = payload.get("rating")
    if not contact_id or rating not in (1, -1):
        raise HTTPException(status_code=422,
                            detail="contact_id and rating (1 or -1) are required")
    with db.connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO tip_feedback"
            " (video_id, contact_id, rating, category, tone, tip_text, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (video_id, contact_id, int(rating), payload.get("category"),
             payload.get("tone"), payload.get("tip"), db.now()))
    return {"video_id": video_id, "contact_id": contact_id, "rating": rating}


@router.get("/rally-analysis/{video_id}/tip-feedback")
def get_tip_feedback(video_id: str):
    """{ratings: {contact_id: 1|-1}, summary: {helpful, not_helpful}} — the
    ratings map restores button state in the UI; the summary is the report-
    level hit-rate that turns tip-selection changes into a measurable number
    instead of vibes."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT contact_id, rating FROM tip_feedback WHERE video_id=?",
            (video_id,)).fetchall()
    ratings = {r["contact_id"]: r["rating"] for r in rows}
    return {"video_id": video_id, "ratings": ratings,
            "summary": {"helpful": sum(1 for v in ratings.values() if v == 1),
                         "not_helpful": sum(1 for v in ratings.values() if v == -1)}}


# --------------------------------------------------- expert coaching marks ---
# Ground-truth labels on coaching MOMENTS (see db.coaching_marks). Deliberately
# separate from tip_feedback above: that rates tip COPY for tips the LLM chose
# to surface, so it can never tell us about a moment the detector missed. These
# marks rate the PLAY over a frame span, including spans nothing flagged, which
# is what makes precision AND recall measurable (backend/scripts/eval_coaching.py).
#
# Spikes to review come from the existing coaching-debug endpoint (its shots[]
# carry contact_id + start_frame/end_frame) — no separate listing endpoint.

_VERDICTS = ("bad", "fine", "good")
_SOURCES = ("spike", "freeform")
_FOCUSES = ("form", "positioning")


@router.get("/rally-analysis/{video_id}/coaching-marks")
def list_coaching_marks(video_id: str, focus: str | None = None):
    """All expert marks for this clip, optionally one focus. Ordered by start
    frame so the UI can render them along the timeline directly."""
    sql = "SELECT * FROM coaching_marks WHERE video_id=?"
    args: list = [video_id]
    if focus is not None:
        if focus not in _FOCUSES:
            raise HTTPException(status_code=422, detail=f"focus must be one of {_FOCUSES}")
        sql += " AND focus=?"
        args.append(focus)
    with db.connect() as con:
        rows = con.execute(sql + " ORDER BY start_frame", args).fetchall()
    return {"video_id": video_id, "marks": db.rows_to_dicts(rows)}


@router.post("/rally-analysis/{video_id}/coaching-marks")
def add_coaching_mark(video_id: str, payload: dict):
    """Stores one expert verdict over a frame span:
    {player_id, focus, start_frame, end_frame, verdict, source,
     rally_id?, metric?, note?, spike_id?}.

    A spike-sourced mark re-marked later REPLACES the earlier verdict for that
    spike (one expert opinion per spike), while freeform marks always append —
    an expert may legitimately mark several distinct spans in one stretch."""
    try:
        player_id = int(payload["player_id"])
        start_frame = int(payload["start_frame"])
        end_frame = int(payload["end_frame"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=422,
                            detail="player_id, start_frame and end_frame are required integers")
    focus, verdict = payload.get("focus"), payload.get("verdict")
    source = payload.get("source", "freeform")
    if focus not in _FOCUSES:
        raise HTTPException(status_code=422, detail=f"focus must be one of {_FOCUSES}")
    if verdict not in _VERDICTS:
        raise HTTPException(status_code=422, detail=f"verdict must be one of {_VERDICTS}")
    if source not in _SOURCES:
        raise HTTPException(status_code=422, detail=f"source must be one of {_SOURCES}")
    spike_id = payload.get("spike_id")
    if source == "spike" and not spike_id:
        raise HTTPException(status_code=422, detail="a spike-sourced mark needs spike_id")
    if end_frame < start_frame:
        raise HTTPException(status_code=422, detail="end_frame must be >= start_frame")

    mark_id = db.new_id()
    with db.connect() as con:
        if source == "spike":
            con.execute("DELETE FROM coaching_marks WHERE video_id=? AND spike_id=?",
                        (video_id, spike_id))
        con.execute(
            "INSERT INTO coaching_marks (id, video_id, rally_id, player_id, focus,"
            " start_frame, end_frame, verdict, metric, note, source, spike_id, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mark_id, video_id, payload.get("rally_id"), player_id, focus,
             start_frame, end_frame, verdict, payload.get("metric"),
             payload.get("note"), source, spike_id if source == "spike" else None,
             db.now()))
    return {"id": mark_id, "video_id": video_id, "verdict": verdict, "source": source}


@router.delete("/rally-analysis/coaching-marks/{mark_id}")
def delete_coaching_mark(mark_id: str):
    with db.connect() as con:
        con.execute("DELETE FROM coaching_marks WHERE id=?", (mark_id,))
    return {"deleted": True}


@router.get("/rally-analysis/{video_id}/report-stats")
def get_report_stats(video_id: str, source: str | None = None):
    """Aggregate coaching-report stats (forward momentum, kitchen arrival,
    shot counts by side/zone), restricted to rallies readiness.
    get_analyzable_rallies() currently clears. Computed fresh each call
    (cheap aggregation, no LLM call) — nothing is persisted."""
    return report_stats.get_report_stats(video_id, rally_source=source)


@router.get("/rally-analysis/{video_id}/trajectories")
def get_trajectories(video_id: str, source: str | None = None):
    """Per-rally ball trajectory samples for the 3D trajectory view:
    {video_id, rallies: [{rally_id, points: [{frame, x, y, height_m,
    confidence, player_id}]}]}. height_m comes from the ballistic
    reconstruction (refine/ballistic.py); confidence and player_id are
    documented in rallyanalysis/trajectory.py. Computed fresh each call
    (nothing persisted), restricted to rallies readiness clears."""
    return trajectory.get_video_trajectories(video_id, rally_source=source)

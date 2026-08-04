"""Rally Analysis & Coaching endpoints. Read-only against rally_labels,
overlays, ball_corrections/ball_tracks and calibration data — never writes
to those tables. New tables only: contact_events, coaching_reports,
coaching_selections.

Endpoints:
  GET  /rally-analysis/health                       liveness check
  POST /rally-analysis/{video_id}/detect-contacts    pose-based contact detection -> contact_events
  GET  /rally-analysis/{video_id}/contact-events     list stored contact_events
  GET  /rally-analysis/{video_id}/readiness          per-rally Pass 2 readiness (waiting/stale/ready)
  POST /rally-analysis/{video_id}/coaching-report    generate report (1 Anthropic call) -> coaching_reports
  GET  /rally-analysis/{video_id}/coaching-report    cached report, or {"report": null}
  GET  /rally-analysis/{video_id}/report-stats       forward-momentum / kitchen-arrival / shot-zone aggregate stats
  GET  /rally-analysis/{video_id}/trajectories       per-rally 3D ball trajectory samples (ballistic reconstruction)
  GET  /rally-analysis/{video_id}/pass1-candidates   Pass 1 rally_labels rows, as acceptable candidates
  POST /rally-analysis/{video_id}/selections         create coaching_selections rows (accept a Pass 1 rally or draw one)
  GET  /rally-analysis/{video_id}/selections         list coaching_selections with current status
  POST /rally-analysis/{video_id}/selections/{selection_id}/process   enqueue Pass 2 for one selection
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db, pipeline
from ..cv import overlays
from ..jobs import runner
from ..learning import params as learned_params
from ..rallyanalysis import coaching, contact_detect, report_stats, trajectory
from ..rallyanalysis.contracts import CoachingReport
from ..rallyanalysis.readiness import covers_rally, readiness_report

router = APIRouter()


def _report_payload(report: CoachingReport) -> dict:
    tips = [asdict(t) for t in report.tips]
    players = sorted({t["player_label"] for t in tips})
    return {"video_id": report.video_id, "tips": tips, "players": players,
            "stale_reason": report.stale_reason}


@router.get("/rally-analysis/health")
def health():
    return {"status": "ok"}


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
def generate_coaching_report(video_id: str, source: str | None = None):
    """Triggers coaching report generation (one Anthropic API call) and
    persists it to coaching_reports, replacing any prior report for this
    video. Not run automatically anywhere — explicit user action only."""
    report = coaching.generate_coaching_report(video_id, rally_source=source)
    return _report_payload(report)


@router.get("/rally-analysis/{video_id}/coaching-report")
def get_coaching_report(video_id: str):
    """Returns the cached report, or null if none has been generated yet —
    regeneration only happens via the POST endpoint."""
    report = coaching.get_report(video_id)
    return {"video_id": video_id, "report": _report_payload(report) if report else None}


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


# ------------------------------------------------- coaching selections ---
# User-driven coaching selection: a manual range does NOT need to trace back
# to a rally_labels row (see coaching_selections.source='user_drawn'). Pass 2
# for these ranges runs via pipeline._build_do_pass2_range under a
# selection-specific overlay source, never "pass2" itself.

class SelectionIn(BaseModel):
    source: str                       # 'pass1_accepted' | 'user_drawn'
    rally_label_id: str | None = None  # required when source='pass1_accepted'
    start_frame: int | None = None     # required when source='user_drawn'
    end_frame: int | None = None


@router.get("/rally-analysis/{video_id}/pass1-candidates")
def pass1_candidates(video_id: str):
    """Pass 1 rally_labels rows for this video, for the UI to offer as
    acceptable selections."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT id, start_frame, end_frame, meta FROM rally_labels"
            " WHERE video_id=? AND source='pass1' ORDER BY start_frame",
            (video_id,)).fetchall()
    return {"video_id": video_id, "candidates": [
        {"id": r["id"], "start_frame": r["start_frame"], "end_frame": r["end_frame"],
         "meta": db.loads(r["meta"], {})} for r in rows]}


@router.post("/rally-analysis/{video_id}/selections")
def create_selections(video_id: str, items: list[SelectionIn]):
    """Creates one coaching_selections row per item. Each item is either
    {source: 'pass1_accepted', rally_label_id} — start/end are looked up from
    that rally_labels row — or {source: 'user_drawn', start_frame,
    end_frame}. Rows start 'pending'; nothing is enqueued until /process."""
    created = []
    with db.connect() as con:
        for item in items:
            if item.source == "pass1_accepted":
                if not item.rally_label_id:
                    raise HTTPException(400, "rally_label_id required for source='pass1_accepted'")
                row = con.execute(
                    "SELECT start_frame, end_frame FROM rally_labels WHERE id=? AND video_id=?",
                    (item.rally_label_id, video_id)).fetchone()
                if row is None:
                    raise HTTPException(404, f"rally_label {item.rally_label_id} not found for this video")
                start_frame, end_frame, rally_label_id = row["start_frame"], row["end_frame"], item.rally_label_id
            elif item.source == "user_drawn":
                if item.start_frame is None or item.end_frame is None:
                    raise HTTPException(400, "start_frame and end_frame required for source='user_drawn'")
                if item.end_frame <= item.start_frame:
                    raise HTTPException(400, f"end_frame must be after start_frame ({item.start_frame})")
                start_frame, end_frame, rally_label_id = item.start_frame, item.end_frame, None
            else:
                raise HTTPException(400, "source must be 'pass1_accepted' or 'user_drawn'")

            selection_id = db.new_id()
            con.execute(
                "INSERT INTO coaching_selections (id, video_id, start_frame, end_frame, source,"
                " status, rally_label_id, overlay_source, error_message, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (selection_id, video_id, start_frame, end_frame, item.source, "pending",
                 rally_label_id, None, None, db.now()))
            created.append(selection_id)
    return {"video_id": video_id, "created": created}


@router.get("/rally-analysis/{video_id}/selections")
def list_selections(video_id: str):
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM coaching_selections WHERE video_id=? ORDER BY created_at",
            (video_id,)).fetchall()
    return {"video_id": video_id, "selections": db.rows_to_dicts(rows)}


@router.post("/rally-analysis/{video_id}/selections/{selection_id}/process")
def process_selection(video_id: str, selection_id: str):
    """Enqueues Pass 2 for one selection. If it's a Pass 1-accepted rally and
    the normal 'pass2' overlay source already covers its exact range, skip
    reprocessing entirely (avoid wasted compute) and mark it ready
    immediately. Otherwise runs Pass 2 fresh under a selection-specific
    overlay source (see pipeline._build_do_pass2_range) — job kind is keyed
    on selection_id so two different selections on the same video never
    supersede each other in the queue (jobs.py's superseding logic keys on
    (video_id, kind))."""
    with db.connect() as con:
        sel = con.execute(
            "SELECT * FROM coaching_selections WHERE id=? AND video_id=?",
            (selection_id, video_id)).fetchone()
    if sel is None:
        raise HTTPException(404, "selection not found")

    if sel["source"] == "pass1_accepted":
        existing = overlays.read_range(video_id, "pass2", sel["start_frame"], sel["end_frame"])
        if covers_rally(existing, sel["start_frame"], sel["end_frame"]):
            with db.connect() as con:
                con.execute(
                    "UPDATE coaching_selections SET status='ready', overlay_source='pass2',"
                    " error_message=NULL WHERE id=?", (selection_id,))
            return {"video_id": video_id, "selection_id": selection_id, "status": "ready",
                    "reused_existing_pass2": True}

    overlay_source = f"pass2_range_{selection_id}"
    version, params = learned_params.get_latest()
    with db.connect() as con:
        con.execute(
            "UPDATE coaching_selections SET status='pending', overlay_source=?,"
            " error_message=NULL WHERE id=?", (overlay_source, selection_id))
    job_id = runner.submit(
        video_id, f"pass2_range_{selection_id}",
        pipeline._build_do_pass2_range(video_id, params, selection_id,
                                        sel["start_frame"], sel["end_frame"], overlay_source))
    return {"video_id": video_id, "selection_id": selection_id, "job_id": job_id,
            "status": "queued", "reused_existing_pass2": False}

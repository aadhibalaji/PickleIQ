"""Rally Analysis & Coaching endpoints. Read-only against rally_labels,
overlays, ball_corrections/ball_tracks and calibration data — never writes
to those tables. New tables only: contact_events, coaching_reports.

Endpoints:
  GET  /rally-analysis/health                       liveness check
  POST /rally-analysis/{video_id}/detect-contacts    pose-based contact detection -> contact_events
  GET  /rally-analysis/{video_id}/contact-events     list stored contact_events
  GET  /rally-analysis/{video_id}/readiness          per-rally Pass 2 readiness (waiting/stale/ready)
  POST /rally-analysis/{video_id}/coaching-report    generate report (1 Anthropic call) -> coaching_reports
  GET  /rally-analysis/{video_id}/coaching-report    cached report, or {"report": null}
  GET  /rally-analysis/{video_id}/report-stats       forward-momentum / kitchen-arrival / shot-zone aggregate stats
  GET  /rally-analysis/{video_id}/trajectories       per-rally 3D ball trajectory samples (ballistic reconstruction)
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from .. import db
from ..rallyanalysis import coaching, contact_detect, report_stats, trajectory
from ..rallyanalysis.contracts import CoachingReport
from ..rallyanalysis.readiness import readiness_report

router = APIRouter()


def _report_payload(report: CoachingReport) -> dict:
    tips = [asdict(t) for t in report.tips]
    players = sorted({t["player_label"] for t in tips})
    return {"video_id": report.video_id, "tips": tips, "players": players,
            "player_themes": [asdict(t) for t in report.player_themes],
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

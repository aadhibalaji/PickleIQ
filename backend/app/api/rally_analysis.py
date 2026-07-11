"""Rally Analysis & Coaching endpoints. Read-only against rally_labels,
overlays, ball_corrections/ball_tracks and calibration data — never writes
to those tables. New tables only: contact_events, coaching_reports."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from .. import db
from ..rallyanalysis import coaching, contact_detect
from ..rallyanalysis.position_metrics import compute_video_metrics
from ..rallyanalysis.shot_metrics import compute_video_shot_metrics

router = APIRouter()


@router.get("/rally-analysis/health")
def health():
    return {"status": "ok"}


@router.post("/rally-analysis/{video_id}/detect-contacts")
def detect_contacts(video_id: str, source: str = "pass2"):
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


@router.get("/rally-analysis/{video_id}/position-metrics")
def position_metrics(video_id: str):
    """Per-player Kitchen Arrival % and Forward Pressure, video-level, plus
    a 0-1 'dominance' share per metric (each player's value / sum of all
    players' values) sized for a gradient/dominance-bar UI."""
    data = compute_video_metrics(video_id)
    players = data["players"]

    def shares(key: str) -> dict:
        vals = {pid: p[key] for pid, p in players.items() if p[key] is not None}
        total = sum(vals.values())
        return {pid: (v / total if total else None) for pid, v in vals.items()}

    kitchen_dom, pressure_dom = shares("kitchen_arrival_pct"), shares("forward_pressure_avg_m")
    return {
        "video_id": video_id,
        "overlay_source": data["overlay_source"],
        "players": {
            str(pid): {
                "team": p["team"],
                "kitchen_arrival_pct": p["kitchen_arrival_pct"],
                "kitchen_serving_rallies": p["kitchen_serving_rallies"],
                "kitchen_dominance": kitchen_dom.get(pid),
                "forward_pressure_avg_m": p["forward_pressure_avg_m"],
                "forward_pressure_rallies": p["forward_pressure_rallies"],
                "forward_pressure_dominance": pressure_dom.get(pid),
            }
            for pid, p in players.items()
        },
    }


@router.get("/rally-analysis/{video_id}/shot-metrics")
def shot_metrics(video_id: str):
    """Per-rally shot count / length bucket / shot-side split, plus a
    video-level per-player shot-side rollup. Reads contact_events directly
    (Wave 1's detect-contacts endpoint must have run first)."""
    data = compute_video_shot_metrics(video_id)
    return {
        "video_id": video_id,
        "rallies": {
            rm.rally_id: {
                "shot_count": rm.shot_count,
                "rally_length_bucket": rm.rally_length_bucket,
                "shot_side_pct": {str(pid): v for pid, v in rm.shot_side_pct.items()},
            }
            for rm in data["rally_metrics"]
        },
        "players": {
            str(pid): {
                "team": p["team"],
                "shot_count": p["shot_count"],
                "shot_side_pct": p["shot_side_pct"],
            }
            for pid, p in data["players"].items()
        },
    }


@router.post("/rally-analysis/{video_id}/coaching-report")
def generate_coaching_report(video_id: str, source: str = "pass2"):
    """Triggers coaching report generation (one Anthropic API call) and
    persists it to coaching_reports, replacing any prior report for this
    video. Not run automatically anywhere — explicit user action only."""
    report = coaching.generate_coaching_report(video_id, rally_source=source)
    return asdict(report)


@router.get("/rally-analysis/{video_id}/coaching-report")
def get_coaching_report(video_id: str):
    """Returns the cached report, or null if none has been generated yet —
    regeneration only happens via the POST endpoint."""
    report = coaching.get_report(video_id)
    return {"video_id": video_id, "report": asdict(report) if report else None}

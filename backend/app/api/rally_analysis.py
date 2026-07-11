"""Rally Analysis & Coaching endpoints. Read-only against rally_labels,
overlays, ball_corrections/ball_tracks and calibration data — never writes
to those tables. New tables only: contact_events, coaching_reports."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/rally-analysis/health")
def health():
    return {"status": "ok"}

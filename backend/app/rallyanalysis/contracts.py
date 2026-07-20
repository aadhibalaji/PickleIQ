"""Shared dataclasses for the Rally Analysis & Coaching feature set. Every
module under rallyanalysis/ imports these rather than inventing parallel
shapes — frozen here first so Wave 1/2/3 work can run in parallel without
drifting.

team/side values match the existing cv/players.py convention: 'near' | 'far'
(not 'serving'/'receiving' — rally_labels has no serving-team column, so
serving team is derived per-rally from is_serve_formation at rally start)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContactEvent:
    """One detected swing/shot, pose-based (never derived from bounce.py or
    ball tracking — ball detection is unreliable, especially far-baseline)."""
    contact_id: str
    video_id: str
    rally_id: str
    frame_idx: int
    timestamp_s: float
    player_id: int          # identity slot (PlayerObs.slot), falls back to track_id
    team: str                # 'near' | 'far'
    court_x: float
    court_y: float
    shot_side: str            # 'left' | 'right'
    hand_speed_peak: float
    confidence: float          # 0-1
    is_serve: bool


@dataclass
class ShotFeatureVector:
    """One per ContactEvent, feeding the coaching pipeline."""
    contact_id: str
    category: str  # 'footwork' | 'paddle_prep' | 'contact_point' | 'follow_through'
    knee_bend_deg: float
    shoulder_rotation_deg: float
    elbow_extension_deg: float
    weight_transfer_score: float
    contact_height_m: float
    follow_through_completion_score: float
    split_step_delay_s: float
    deviation_score: float  # signed distance from good-form reference range


@dataclass
class CoachingTip:
    """One plain-language, per-shot coaching tip, anchored to the contact
    event it was generated from so the UI can seek the video there and
    attribute the tip to a player."""
    contact_id: str
    frame: int
    timestamp_s: float
    category: str  # 'footwork' | 'paddle_prep' | 'contact_point' | 'follow_through'
    tip: str
    tone: str  # 'improve' | 'praise'
    player_id: int
    player_label: str  # e.g. "Player 1" — display label, derived from player_id


@dataclass
class CoachingReport:
    """One per video, generated once (explicit user action — LLM call)."""
    video_id: str
    tips: list = field(default_factory=list)  # list[CoachingTip]

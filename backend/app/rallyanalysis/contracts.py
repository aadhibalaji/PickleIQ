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
class RallyMetrics:
    """One per rally. Position metrics (Wave 1) and shot metrics (Wave 2)
    both populate this same shape — see position_metrics.py / shot_metrics.py."""
    rally_id: str
    video_id: str
    kitchen_arrival: dict = field(default_factory=dict)      # {player_id: bool}
    forward_pressure: dict = field(default_factory=dict)     # {player_id: float}
    shot_side_pct: dict = field(default_factory=dict)        # {player_id: {"left": float, "right": float}}
    shot_count: int = 0
    rally_length_bucket: str | None = None  # 'short' | 'medium' | 'long'
    active_duration_s: float = 0.0


@dataclass
class CoachingReport:
    """One per video, generated once (explicit user action — LLM call)."""
    video_id: str
    category_ratios: dict = field(default_factory=dict)  # {category: {flagged_count, active_rally_seconds, ratio}}
    worst_shots: list = field(default_factory=list)  # [{contact_id, category, deviation_score, timestamp_s, rally_id, feature_vector}]
    best_shots: list = field(default_factory=list)
    narrative: str = ""

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
    """One per ContactEvent, feeding the coaching pipeline.

    knee_bend_deg/shoulder_rotation_deg/elbow_extension_deg/weight_transfer_score/
    follow_through_completion_score/split_step_delay_s are None when this shot
    had no confident pose support for that specific metric (missing keypoint,
    or below coaching.KP_CONF_MIN) — see low_confidence_metrics. They are NEVER
    backfilled with a fabricated "neutral/in-range" placeholder; a metric with
    no real reading is simply absent, not silently reported as fine.
    contact_height_m is a court-position proxy from the contact event itself
    (not keypoint-derived), so it's always present.

    deviation_score is judged against THIS PLAYER's own median/spread for
    this metric across their other shots in THIS video — never a fixed
    universal range — see coaching._rel_dev. 0 = within this player's own
    typical spread for that metric this session."""
    contact_id: str
    category: str  # 'footwork' | 'paddle_prep' | 'contact_point' | 'follow_through'
    knee_bend_deg: float | None
    shoulder_rotation_deg: float | None
    elbow_extension_deg: float | None
    weight_transfer_score: float | None
    contact_height_m: float
    follow_through_completion_score: float | None
    split_step_delay_s: float | None
    deviation_score: float  # signed deviation from this player's own median/spread, this video
    # Names (dataclass field names above) of metrics excluded from this shot
    # for missing/low-confidence pose data — the explicit "we didn't actually
    # see this" flag consumed by aggregate stats and the coaching narrative.
    low_confidence_metrics: list = field(default_factory=list)
    # Names of metrics that DID get a confident reading on this shot, but
    # where this player didn't have enough of their OWN valid shots for that
    # metric in this video (see coaching.MIN_SAMPLES_FOR_DISTRIBUTION) to
    # know what "typical" looks like for them — so no relative comparison
    # could be judged. Distinct from low_confidence_metrics: the pose reading
    # itself is trusted here, there's just no personal baseline to compare it
    # against yet.
    no_comparison_metrics: list = field(default_factory=list)


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
class PlayerTheme:
    """One per player per report: the session's dominant pattern for that
    player in coach language, produced by the same single LLM call as the
    tips. player_id/player_label are attached from our own contact_events
    data (the model's echoed label is only used to match, never trusted as
    attribution — see coaching.build_report)."""
    player_id: int
    player_label: str
    theme: str


@dataclass
class CoachingReport:
    """One per video, generated once (explicit user action — LLM call)."""
    video_id: str
    tips: list = field(default_factory=list)  # list[CoachingTip]
    player_themes: list = field(default_factory=list)  # list[PlayerTheme]
    # None = current (built on the confidence-aware pipeline). Otherwise a
    # short machine-readable reason this report predates a data-accuracy fix
    # and may contain fabricated feature values — see db.STALE_BEFORE_CONFIDENCE_FIX.
    stale_reason: str | None = None

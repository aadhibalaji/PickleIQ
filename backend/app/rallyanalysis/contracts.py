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
    # Coarse pose-derived stratification key ("kitchen_soft",
    # "backcourt_fast", zone-only "midcourt", ...) — see
    # shot_context.classify_shot_types. None = couldn't be typed (no court
    # position). A stratification key, NOT a verified stroke label.
    shot_type: str | None = None
    # Metrics whose deviation_score comparison fell back to the player's
    # ALL-shots baseline because their same-shot-type stratum didn't have
    # enough samples this session (or the shot couldn't be typed at all).
    # Distinct from no_comparison_metrics: a comparison DID happen here, it
    # just mixes shot types — cross-type claims ("short follow-through" on a
    # dink judged against drives) should be made more cautiously downstream.
    mixed_baseline_metrics: list = field(default_factory=list)
    # Names of metrics that deviated (non-zero _rel_dev) on this shot — not
    # just the single worst one that picked the category. Multiple metrics
    # deviating together is strong evidence of a real form breakdown rather
    # than one noisy keypoint reading; coaching._salience uses the count to
    # rank tip candidates.
    deviating_metrics: list = field(default_factory=list)
    # Continuous distance from the middle of this player's typical range, in
    # the same units as deviation_score — unlike deviation_score it does NOT
    # clamp to 0 inside the fence, so in-range shots can still be ranked
    # ("which of these is closest to drifting"). Ranking signal for the
    # fill-to-limit candidate top-up ONLY (see coaching.select_tip_candidates)
    # — never a flag, never shown as evidence of a problem. None when no
    # metric had both a reading and a baseline.
    soft_deviation: float | None = None
    # Metrics whose deviation on THIS shot came from a conservative UNIVERSAL
    # reference check (coaching.REFERENCE_BANDS — gross fundamentals only:
    # straight legs, jammed contact, arm-only swing / cut-off follow-through
    # on hard swings) rather than the player's own Tukey fence. This is what
    # lets a CONSISTENT flaw flag at all: a habit repeated on every shot is,
    # by definition, typical for that player and invisible to per-player
    # fences. Also the only metrics the LLM may coach as correct-form claims.
    reference_flagged_metrics: list = field(default_factory=list)
    # {metric_field_name: deviation_or_None} for EVERY metric this shot was
    # scored on, not just the one whose |deviation| picked the category —
    # the diagnostic detail category/deviation_score alone throw away.
    # Feeds coaching.debug_timeline (the per-metric-over-time chart); no
    # selection/LLM logic reads this, only display/eval tooling.
    all_metric_devs: dict = field(default_factory=dict)


@dataclass
class PositioningFeatureVector:
    """One per evaluated positioning moment — either a real contact event
    (contact-anchored metrics) or a synthetic span moment (contact_id
    "pos_<rally>_<pid>_<metric>_<frame>", kind 'span' on its meta dict; see
    positioning.span_moments). Field-compatible with ShotFeatureVector
    everywhere the selection/LLM path reads it (category, deviation_score,
    the four metric-name lists, shot_type, soft_deviation) so the two flow
    through the same select/salience/LLM pipeline.

    deviation_score here is judged against ABSOLUTE court-geometry bands
    (positioning.BANDS) — the court is a fixed size, unlike body mechanics —
    NOT against the player's own session spread. no_comparison_metrics /
    mixed_baseline_metrics are therefore always empty; low_confidence_metrics
    lists every metric field that is None on this moment (not applicable to
    this moment, or no position data to compute it from)."""
    contact_id: str
    category: str  # 'net_position' | 'transition' | 'court_coverage' | 'team_spacing'
    kitchen_distance_at_contact_m: float | None
    return_depth_m: float | None
    transition_dwell_s: float | None
    kitchen_approach_after_return_s: float | None
    recovery_offset_m: float | None
    partner_depth_stagger_m: float | None
    deviation_score: float
    low_confidence_metrics: list = field(default_factory=list)
    no_comparison_metrics: list = field(default_factory=list)
    shot_type: str | None = None
    mixed_baseline_metrics: list = field(default_factory=list)
    deviating_metrics: list = field(default_factory=list)
    soft_deviation: float | None = None
    # Always empty for positioning (EVERY positioning deviation is already
    # judged against absolute court geometry) — present only so the two
    # vector types stay field-compatible through _shot_summary.
    reference_flagged_metrics: list = field(default_factory=list)
    # {metric_field_name: deviation_or_None} for EVERY metric this shot was
    # scored on, not just the one whose |deviation| picked the category —
    # the diagnostic detail category/deviation_score alone throw away.
    # Feeds coaching.debug_timeline (the per-metric-over-time chart); no
    # selection/LLM logic reads this, only display/eval tooling.
    all_metric_devs: dict = field(default_factory=dict)


@dataclass
class FrameSpikeVector:
    """One per detected SPIKE — a run of consecutive frames whose combined
    per-metric deviation from an ABSOLUTE per-frame target (see
    frame_metrics.py) crossed the flag threshold. This is the frame-by-frame
    counterpart of ShotFeatureVector/PositioningFeatureVector and is
    field-compatible with both through _shot_summary — coaching.py's
    selection/salience/LLM pipeline treats a spike exactly like any other
    scored moment (rendered downstream as a "span" pseudo-contact, same
    mechanism positioning.py's span moments already use).

    deviation_score is the SIGNED deviation of the spike's dominant metric
    at its peak frame (the metric that decided the category) — not the
    unsigned peak_sum, which callers can recover from all_metric_devs the
    same way debug_timeline's metric_sum does for every other vector type.
    There is no personal-baseline mode for a frame stream, so
    reference_flagged_metrics/no_comparison_metrics/mixed_baseline_metrics
    are always empty — every deviation here is already against an absolute
    target by construction."""
    contact_id: str
    category: str
    deviation_score: float
    duration_s: float
    low_confidence_metrics: list = field(default_factory=list)
    no_comparison_metrics: list = field(default_factory=list)
    shot_type: str | None = None
    mixed_baseline_metrics: list = field(default_factory=list)
    deviating_metrics: list = field(default_factory=list)
    soft_deviation: float | None = None
    reference_flagged_metrics: list = field(default_factory=list)
    all_metric_devs: dict = field(default_factory=dict)
    # Raw metric values at the spike's peak frame — all_metric_devs alone
    # (deviations) can't reconstruct what the reading actually was; debug
    # tooling and the LLM's feature_vector both need both.
    all_metric_values: dict = field(default_factory=dict)


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
class GeneralFeedback:
    """One per player per report: exactly one positive and one negative piece
    of session-wide feedback, grounded in deterministic per-player metric
    aggregates (coaching._frame_general_context) and written by the same
    single LLM call as the tips. Replaces PlayerTheme on new reports.
    Attribution rules match PlayerTheme: player_id/player_label come from
    our own contact data, the model's echoed label is only a join key."""
    player_id: int
    player_label: str
    praise: str = ""   # what's working — the strongest positive session pattern
    improve: str = ""  # the one thing to fix — the strongest negative pattern


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
    # Which metric family this report evaluated — 'form' | 'positioning'.
    # None on reports generated before selectable focus existed (= form).
    focus: str | None = None
    # list[GeneralFeedback] — per-player praise+improve pairs. Empty on
    # legacy reports (which carry player_themes instead) and on reports
    # whose LLM call degraded.
    general: list = field(default_factory=list)
    # {player_label: jpeg_b64} — one representative photo per player (see
    # keyframes.extract_identity_photos), so the UI can show who's who
    # instead of an unlabeled seat number. None when extraction wasn't
    # possible (no video path, unreadable file, no player boxes yet).
    player_photos: dict | None = None

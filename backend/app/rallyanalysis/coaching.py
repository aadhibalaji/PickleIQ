"""Coaching report generation (Wave 2/3): turns ContactEvents into per-shot
biomechanical feature vectors, aggregates flag ratios per category, picks
representative worst/best shots via an adaptive threshold, and (one call per
video) asks the Anthropic API for a narrative. Mirrors analysis/reporter.py's
client/schema/degrade pattern exactly — see generate_llm_narrative below.

Read-only against rally_labels, overlays, contact_events, court geometry.
Writes only to coaching_reports (never cv/*, refine/*, rally_labels, overlays).
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict

import numpy as np

from .. import db
from ..config import ANTHROPIC_API_KEY, ANTHROPIC_COACHING_MODEL
from ..cv import overlays
from ..cv.court import NET_Y
from . import positioning
from .contracts import CoachingReport, CoachingTip, GeneralFeedback, PlayerTheme, ShotFeatureVector
from .readiness import _default_overlay_source, get_analyzable_rallies
from .shot_context import annotate_rally_context, classify_shot_types, position_samples

log = logging.getLogger("rallyanalysis.coaching")

# COCO-17 keypoint indices (same convention as contact_detect.py / players.py).
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# ---------------------------------------------------------------------------
# PER-PLAYER RELATIVE SCORING — a shot is judged against THIS PLAYER's own
# median/spread for each metric, computed across their other shots in THIS
# video, never a fixed universal range. Two labeled clips gave contradictory
# answers on which joint is weakest across singles vs doubles play, so there
# is no validated universal cutoff to compare against yet — a player's own
# session is the only baseline we can trust today. See _rel_dev/_robust_fence.
# ---------------------------------------------------------------------------
METRIC_FIELDS = (
    "knee_bend_deg", "shoulder_rotation_deg", "elbow_extension_deg",
    "weight_transfer_score", "contact_height_m",
    "follow_through_completion_score", "split_step_delay_s",
)

# Minimum non-null values a player needs for a given metric, this video,
# before we trust a median/spread computed from them enough to score shots
# against it — below this, 1-2 points would make an unstable "distribution"
# look meaningful. JUDGMENT CALL: 3 is a small, workable floor for a
# handful-of-shots-per-video session; raise if per-player samples turn out
# noisier than expected in practice.
MIN_SAMPLES_FOR_DISTRIBUTION = 3

# Tukey outlier fence multiplier: a shot's metric only counts as deviating
# once it falls beyond Q1 - 1.5*IQR / Q3 + 1.5*IQR of THIS PLAYER's own
# values for that metric this video — the standard robust-statistics
# convention for "notably atypical," reused here instead of inventing a new
# one. Deviation is then expressed as (excess distance beyond the fence) in
# IQR units, matching the old fixed-range function's "0 inside, signed
# excess width outside" shape.
FENCE_IQR_MULTIPLIER = 1.5

# Normalization constants (turn raw pixel/metre displacements into 0-1 scores).
WEIGHT_TRANSFER_NORM_M = 1.0   # forward court-plane movement treated as "full" transfer
FOLLOW_THROUGH_NORM = 1.8      # post-contact wrist path length, as a multiple of the
                                # player's own on-screen box height, treated as "full"

PRE_WINDOW_S = 0.6   # how far before contact we look for split-step / prep
POST_WINDOW_S = 0.45  # how far after contact we look for follow-through

WORST_DEVIATION_THRESHOLD = 1.0  # |deviation_score| >= this many of the PLAYER's OWN
                                   # IQR-widths beyond their own Tukey fence for that
                                   # metric counts as "worst" — adaptive PER CATEGORY and
                                   # PER PLAYER (each category/player/video combo has its
                                   # own fence baked into _rel_dev's normalization), not a
                                   # fixed top-N count: a clean session naturally clears
                                   # 0-1 shots, a rough one can clear many.
MIN_BEST_CONFIDENCE = 0.3  # absolute floor: a barely-detected contact is never held up as exemplary

MAX_TIPS = 8       # hard cap on TIPS SHOWN — keeps the UI list deep enough to
                    # matter but never overwhelming. Also written into the LLM
                    # prompt's selection budget (see SYSTEM) — keep in sync.
MAX_PRAISE_TIPS = 2  # total praise across the report (and never more than one
                      # per player) — the cap exists to spend slots on flaws
MAX_LLM_CANDIDATES = 16  # statistical shortlist size handed to the LLM, which
                          # makes the final cut down to MAX_TIPS using the
                          # frames (see build_report / SYSTEM selection rules)

# Product decision (2026-08-04): when fewer than MAX_LLM_CANDIDATES moments
# clear the flag threshold, top the shortlist up with the highest
# |soft_deviation| moments anyway, each marked below_threshold in its LLM
# summary. The LLM final cut keeps full skip authority (see the SYSTEM
# selection rules), so borderline filler still gets dropped when the frames
# don't back it up — this widens what the LLM sees, not what gets flagged.
FILL_TO_LIMIT = True

# ---------------------------------------------------------------------------
# Salience — ranks WHICH flagged shots deserve one of the MAX_TIPS slots.
# Deviation magnitude alone finds "most atypical", not "most obviously a
# mistake"; these bonuses close part of that gap: a shot the rally ended on
# is far more likely an actual error than an equal mid-rally outlier, and
# several metrics deviating together is a real form breakdown, where a
# single-metric outlier is often one noisy keypoint. Units are IQR-widths
# (same scale as deviation_score) so the bonuses stay comparable.
# ---------------------------------------------------------------------------
RALLY_ENDING_BONUS = 1.0
MULTI_METRIC_BONUS = 0.5   # per deviating metric beyond the first
MULTI_METRIC_BONUS_CAP = 2  # at most this many bonus-earning extra metrics

# category -> the ShotFeatureVector metric fields that drive it. Single
# source of truth shared by _classify (category assignment) and the praise
# exemplar picker (does this in-range shot have a REAL reading to praise?).
CATEGORY_METRICS = {
    "footwork": ("split_step_delay_s", "knee_bend_deg"),
    "paddle_prep": ("shoulder_rotation_deg",),
    "contact_point": ("elbow_extension_deg", "weight_transfer_score", "contact_height_m"),
    "follow_through": ("follow_through_completion_score",),
}

# ---------------------------------------------------------------------------
# UNIVERSAL REFERENCE BANDS — the deliberate, narrow exception to the
# per-player-relative rule above, closing its structural blind spot: a flaw a
# player repeats on EVERY shot is, by definition, typical for them, so their
# own Tukey fence can never flag it (and a player with < MIN_SAMPLES_FOR_
# DISTRIBUTION shots has no fence at all). These bands catch only GROSS
# fundamentals that are wrong under any reasonable camera projection and any
# standard of play — standing straight-legged through a shot, a contact
# jammed into the body, an arm-only swing or a chopped-off follow-through on
# a hard drive. Everything subtler stays per-player-relative: 2D image-space
# joint angles are projection-dependent, which is exactly why the original
# design refused universal cutoffs for ordinary readings.
#
# Shape: (lo, hi, scale, fast_only). Deviation = signed excess beyond
# [lo, hi] in `scale` units (same units as _rel_dev / positioning._abs_dev,
# so WORST_DEVIATION_THRESHOLD and salience apply unchanged). fast_only
# bands are checked only on shots typed *_fast (weight transfer and a full
# follow-through are non-negotiable on drives, but legitimately tiny on
# dinks/resets — an untempoed shot gets no check, conservative by default).
# Constants are JUDGMENT CALLS pending labeled-clip validation, same status
# as positioning.BANDS:
#   knee_bend_deg  hip-knee-ankle ~180 = standing dead straight. 168+ reads
#       as visually straight legs from any angle; scale 6 puts fully locked
#       (180) at dev 2 ("clearly off").
#   elbow_extension_deg  shoulder-elbow-wrist < 65 = contact jammed right
#       into the body; normal volleys/dinks sit ~90-130, so 65 flags only
#       the cramped extreme.
#   weight_transfer_score  ~0 forward movement INTO a hard swing = all-arm.
#   follow_through_completion_score  near-zero wrist path after a hard
#       swing = chopped-off finish.
# ---------------------------------------------------------------------------
REFERENCE_BANDS = {
    "knee_bend_deg": (0.0, 168.0, 6.0, False),
    "elbow_extension_deg": (65.0, 180.0, 15.0, False),
    "weight_transfer_score": (0.1, 1.0, 0.05, True),
    "follow_through_completion_score": (0.15, 1.0, 0.1, True),
}


def _reference_dev(metric: str, value: float | None, shot_type: str | None) -> float | None:
    """Signed deviation beyond the universal reference band in scale units —
    0 inside, None when the metric has no band, no reading, or (for
    fast_only bands) the shot isn't verifiably fast."""
    band = REFERENCE_BANDS.get(metric)
    if band is None or value is None:
        return None
    lo, hi, scale, fast_only = band
    if fast_only and not (shot_type or "").endswith("_fast"):
        return None
    if value > hi:
        return (value - hi) / scale
    if value < lo:
        return (value - lo) / scale
    return 0.0


MIN_TIP_GAP_S = 4.0  # minimum spacing, in seconds, between two selected tip
                      # moments for the SAME player (worst+best combined, any
                      # category) — keeps rendered coaching moments from
                      # clustering a couple seconds apart. The more instructive
                      # of a too-close pair wins (larger |deviation_score| for
                      # worst tips, higher confidence for best tips; worst
                      # always outranks best, matching select_tip_candidates'
                      # own worst-first budgeting).

# Minimum YOLO per-keypoint confidence (cv/players.py's r.keypoints.conf,
# stored as overlay "kp_conf") for a keypoint to support a biomechanics
# metric below. A SINGLE global cutoff, not per-joint: the only two
# already-processed clips checked so far disagree on which joints are
# weakest (one clip: wrists/ankles worst; the other: ankles/knees worst and
# flatter across the board) — a per-joint threshold today would encode that
# small, inconsistent sample as a stable joint hierarchy it doesn't actually
# support yet. 0.5 is a conservative, easy-to-reason-about floor (YOLO itself
# is less than a coin flip confident below it) applied uniformly until more
# labeled clips justify per-joint tuning.
# THIS IS A STARTING POINT, NOT A CALIBRATED VALUE — revisit once more clips
# have been processed with kp_conf capture (see cv/players.py PlayerObs.kp_conf).
KP_CONF_MIN = 0.5


# ---------------------------------------------------------------------------
# Pure pose-window helpers (no DB) — unit-tested directly on synthetic overlays.
# ---------------------------------------------------------------------------

def _kp_xy(pdict: dict, idx: int) -> tuple[float, float] | None:
    """A keypoint counts as usable only if it's present AND (when confidence
    was captured — see cv/players.py's kp_conf) at least KP_CONF_MIN. Older
    overlay rows written before kp_conf capture existed have no "kp_conf" key
    at all; those fall back to the old presence-only check rather than being
    treated as automatically failing a threshold they never recorded."""
    kp = pdict.get("kp")
    if not kp or len(kp) <= idx:
        return None
    pt = kp[idx]
    if not pt or pt[0] <= 0 or pt[1] <= 0:
        return None
    conf = pdict.get("kp_conf")
    if conf and len(conf) > idx and conf[idx] < KP_CONF_MIN:
        return None
    return (float(pt[0]), float(pt[1]))


def _angle_deg(a: tuple, b: tuple, c: tuple) -> float | None:
    """Angle at vertex b (degrees), formed by rays b->a and b->c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def _knee_angle(pdict: dict) -> float | None:
    angles = []
    for hip_i, knee_i, ankle_i in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        hip, knee, ankle = _kp_xy(pdict, hip_i), _kp_xy(pdict, knee_i), _kp_xy(pdict, ankle_i)
        if hip and knee and ankle:
            a = _angle_deg(hip, knee, ankle)
            if a is not None:
                angles.append(a)
    return sum(angles) / len(angles) if angles else None


def _elbow_angle(pdict: dict, side: str) -> float | None:
    shoulder_i, elbow_i, wrist_i = (L_SHOULDER, L_ELBOW, L_WRIST) if side == "L" else (R_SHOULDER, R_ELBOW, R_WRIST)
    shoulder, elbow, wrist = _kp_xy(pdict, shoulder_i), _kp_xy(pdict, elbow_i), _kp_xy(pdict, wrist_i)
    if shoulder and elbow and wrist:
        return _angle_deg(shoulder, elbow, wrist)
    return None


def _shoulder_rotation(pdict: dict) -> float | None:
    """Angle between the shoulder line and hip line — a proxy for trunk coil
    (0 = fully square/no twist, larger = more rotated)."""
    ls, rs = _kp_xy(pdict, L_SHOULDER), _kp_xy(pdict, R_SHOULDER)
    lh, rh = _kp_xy(pdict, L_HIP), _kp_xy(pdict, R_HIP)
    if not (ls and rs and lh and rh):
        return None
    shoulder_vec = (rs[0] - ls[0], rs[1] - ls[1])
    hip_vec = (rh[0] - lh[0], rh[1] - lh[1])
    n1, n2 = math.hypot(*shoulder_vec), math.hypot(*hip_vec)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = max(-1.0, min(1.0, (shoulder_vec[0] * hip_vec[0] + shoulder_vec[1] * hip_vec[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def _player_frames(entries: list[dict], player_id) -> list[tuple[int, dict]]:
    out = []
    for e in entries:
        for p in e.get("players", []):
            if p.get("id") == player_id:
                out.append((e["f"], p))
                break
    out.sort(key=lambda fp: fp[0])
    return out


def _nearest_valid(frames: list[tuple[int, dict]], target_frame: int, extractor):
    """Value from the frame nearest target_frame (ties favour earlier) whose
    extractor(pdict) is not None; None if no frame in `frames` qualifies."""
    for f, pdict in sorted(frames, key=lambda fp: (abs(fp[0] - target_frame), fp[0])):
        v = extractor(pdict)
        if v is not None:
            return v
    return None


def _hitting_side(frames: list[tuple[int, dict]]) -> str:
    """Whichever wrist moved more across the window is treated as the
    swinging/hitting arm (same "faster wrist" idea contact_detect.py uses)."""
    totals = {"L": 0.0, "R": 0.0}
    prev = {"L": None, "R": None}
    for _f, pdict in frames:
        for side, idx in (("L", L_WRIST), ("R", R_WRIST)):
            pt = _kp_xy(pdict, idx)
            if pt and prev[side]:
                totals[side] += math.hypot(pt[0] - prev[side][0], pt[1] - prev[side][1])
            if pt:
                prev[side] = pt
    return "L" if totals["L"] > totals["R"] else "R"


def _extract_raw_metrics(contact: dict, entries: list[dict], fps: float) -> dict | None:
    """contact: one contact_events row (dict) — contact_id, frame_idx,
    player_id, court_x, court_y, .... entries: overlay rows covering a
    [contact_frame - PRE_WINDOW_S, contact_frame + POST_WINDOW_S] window
    (any order, any players). Returns None only when this player has no pose
    data anywhere in the window (contact gets skipped, not fabricated).

    Returns the raw (un-classified, un-scored) metric values plus
    low_confidence_metrics — the metric-extraction half of what
    compute_feature_vector used to do in one step, split out so the caller
    can build every shot's raw metrics first, derive each player's own
    per-metric fence across their whole video, and only then classify/score
    each shot relative to that fence (see generate_coaching_report).

    Individual metrics that lack CONFIDENT pose support (missing keypoint, or
    below KP_CONF_MIN) are excluded — left None, never backfilled with a
    fabricated neutral value. A metric we didn't confidently see is absent,
    not silently reported as "in range". Excluded metric names are collected
    in low_confidence_metrics.
    """
    frames = _player_frames(entries, contact["player_id"])
    if not frames:
        return None

    contact_frame = contact["frame_idx"]
    pre_frames = [fp for fp in frames if fp[0] <= contact_frame]
    post_frames = [fp for fp in frames if fp[0] >= contact_frame]
    hitting_side = _hitting_side(frames)
    low_confidence_metrics: list[str] = []

    knee_bend_deg = _nearest_valid(frames, contact_frame, _knee_angle)
    if knee_bend_deg is None:
        low_confidence_metrics.append("knee_bend_deg")

    shoulder_rotation_deg = _nearest_valid(frames, contact_frame, _shoulder_rotation)
    if shoulder_rotation_deg is None:
        low_confidence_metrics.append("shoulder_rotation_deg")

    elbow_extension_deg = _nearest_valid(frames, contact_frame, lambda p: _elbow_angle(p, hitting_side))
    if elbow_extension_deg is None:
        low_confidence_metrics.append("elbow_extension_deg")

    # weight_transfer_score: forward court-plane movement (toward the net) of
    # this player from the start of the pre-window to contact, normalized by
    # WEIGHT_TRANSFER_NORM_M. Uses court_x/court_y (already homography-projected
    # metres) rather than raw pixels so it's camera-distance invariant.
    start_cxy = _nearest_valid(pre_frames, pre_frames[0][0], lambda p: tuple(p["cxy"]) if p.get("cxy") else None) if pre_frames else None
    contact_cxy = (contact.get("court_x"), contact.get("court_y"))
    if start_cxy and contact_cxy[0] is not None and contact_cxy[1] is not None:
        dy = contact_cxy[1] - start_cxy[1]
        forward = -dy if contact.get("team") == "near" else dy  # near moves toward decreasing y
        weight_transfer_score = max(0.0, min(1.0, forward / WEIGHT_TRANSFER_NORM_M))
    else:
        weight_transfer_score = None
        low_confidence_metrics.append("weight_transfer_score")

    # contact_height_m: LIMITATION — no calibrated vertical (Z) coordinate is
    # available (pose keypoints + homography only give a ground-plane
    # projection, not true height). We reuse the contact event's own
    # court_y (already stored, no need to re-derive) minus NET_Y as a rough
    # proxy for "how far from the net, along the court, this contact was
    # taken" — NOT an actual contact height. Do not read literal metres of
    # vertical height into this field.
    contact_height_m = (contact.get("court_y") if contact.get("court_y") is not None else NET_Y) - NET_Y

    # follow_through_completion_score: hitting-wrist path length after
    # contact, normalized by the player's own on-screen box height at
    # contact (self-normalizing across different camera zoom/distance).
    box_height = _nearest_valid(frames, contact_frame, lambda p: (p["box"][3] - p["box"][1]) if p.get("box") else None)
    wrist_idx = L_WRIST if hitting_side == "L" else R_WRIST
    path = 0.0
    prev_pt = None
    for _f, pdict in sorted(post_frames, key=lambda fp: fp[0]):
        pt = _kp_xy(pdict, wrist_idx)
        if pt and prev_pt:
            path += math.hypot(pt[0] - prev_pt[0], pt[1] - prev_pt[1])
        if pt:
            prev_pt = pt
    if box_height and box_height > 1e-6 and prev_pt is not None:
        follow_through_completion_score = max(0.0, min(1.0, path / (box_height * FOLLOW_THROUGH_NORM)))
    else:
        follow_through_completion_score = None
        low_confidence_metrics.append("follow_through_completion_score")

    # split_step_delay_s: time from the most "planted" (lowest ankle speed)
    # pre-contact frame to contact — a proxy for split-step landing timing.
    ankle_speeds = []  # (frame, speed)
    prev = None
    for f, pdict in sorted(pre_frames, key=lambda fp: fp[0]):
        la, ra = _kp_xy(pdict, L_ANKLE), _kp_xy(pdict, R_ANKLE)
        pts = [p for p in (la, ra) if p]
        mid = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) if pts else None
        if mid and prev:
            pf, pmid = prev
            gap = f - pf
            if gap > 0:
                ankle_speeds.append((f, math.hypot(mid[0] - pmid[0], mid[1] - pmid[1]) / gap))
        if mid:
            prev = (f, mid)
    if ankle_speeds:
        plant_frame = min(ankle_speeds, key=lambda fs: fs[1])[0]
        split_step_delay_s = max(0.0, (contact_frame - plant_frame) / fps)
    else:
        split_step_delay_s = None
        low_confidence_metrics.append("split_step_delay_s")

    return {
        "knee_bend_deg": knee_bend_deg, "shoulder_rotation_deg": shoulder_rotation_deg,
        "elbow_extension_deg": elbow_extension_deg, "weight_transfer_score": weight_transfer_score,
        "contact_height_m": contact_height_m,
        "follow_through_completion_score": follow_through_completion_score,
        "split_step_delay_s": split_step_delay_s,
        "low_confidence_metrics": low_confidence_metrics,
    }


def _robust_fence(values: list[float]) -> tuple[float, float, float] | None:
    """(q1, q3, iqr) over `values` — this player's own non-null readings for
    one metric, this video — or None if there are too few
    (< MIN_SAMPLES_FOR_DISTRIBUTION) to trust a spread computed from them."""
    if len(values) < MIN_SAMPLES_FOR_DISTRIBUTION:
        return None
    q1, q3 = np.percentile(np.asarray(values, dtype=float), [25, 75])
    return float(q1), float(q3), float(q3 - q1)


def _rel_dev(value: float | None, fence: tuple[float, float, float] | None) -> float | None:
    """Signed deviation from this player's own typical spread for this
    metric, this video — expressed in units of THEIR OWN iqr so metrics with
    different units (degrees, seconds, 0-1 scores) stay comparable. 0 = within
    [q1 - FENCE_IQR_MULTIPLIER*iqr, q3 + FENCE_IQR_MULTIPLIER*iqr] (this
    player's own Tukey fence). None (no confident pose reading for this shot,
    OR this player didn't have enough of their own shots to define a fence
    for this metric) passes through as None rather than being treated as
    0/typical — there is nothing real to compare in either case."""
    if value is None or fence is None:
        return None
    q1, q3, iqr = fence
    if iqr < 1e-9:
        # This player's own readings for this metric are ~identical this
        # session — any difference is real but there's no spread to divide
        # by. ponytail: binary flag (0 or +-1 iqr-unit) rather than an
        # unbounded ratio; revisit if this over/under-flags flatline metrics.
        return 0.0 if abs(value - q1) < 1e-9 else math.copysign(1.0, value - q1)
    lo, hi = q1 - FENCE_IQR_MULTIPLIER * iqr, q3 + FENCE_IQR_MULTIPLIER * iqr
    if value < lo:
        return (value - lo) / iqr
    if value > hi:
        return (value - hi) / iqr
    return 0.0


def _classify(devs: dict[str, float | None]) -> tuple[str | None, float | None]:
    """devs: {metric_field_name: relative_deviation_or_None} for one shot
    (see _rel_dev — already computed against the player's own fence).
    category = whichever of the four categories has the single worst
    (largest-magnitude) deviation among its REAL (non-None) component
    metrics; deviation_score is that category's signed deviation. A category
    every one of whose metrics is None (excluded for low pose confidence, or
    no player fence to compare against) is never eligible to win — that would
    be the same fabricated-neutral bug at the category level even if
    individual metrics are properly excluded. A shot with every metric
    typical for this player gets an arbitrary-but-stable category (max of
    all-zero deviations) and deviation_score 0.0 (i.e. not flagged — see
    compute_category_ratios). Returns (None, None) only if literally no
    category has any real metric (compute_feature_vector treats that as
    unscoreable, same as no pose data at all)."""
    per_category = {cat: [devs[f] for f in fields]
                    for cat, fields in CATEGORY_METRICS.items()}
    category_dev = {}
    for cat, ds in per_category.items():
        real = [d for d in ds if d is not None]
        if real:
            category_dev[cat] = max(real, key=abs)
    if not category_dev:
        return None, None
    category = max(category_dev, key=lambda c: abs(category_dev[c]))
    return category, category_dev[category]


def _player_fences_by_metric(raw_by_contact: list[tuple[dict, dict]]) -> dict:
    """{player_id: {metric_field_name: fence_or_None}} — one fence per
    player per metric, built from THAT PLAYER's own non-null values for that
    metric across every contact in raw_by_contact (i.e. their whole video).
    A metric absent from a player's dict, or mapped to None, means they
    didn't have MIN_SAMPLES_FOR_DISTRIBUTION valid shots for it — _rel_dev
    treats that exactly like a missing fence (no comparison possible)."""
    values: dict = {}
    for contact, raw in raw_by_contact:
        pid = contact["player_id"]
        for field_name in METRIC_FIELDS:
            v = raw[field_name]
            if v is not None:
                values.setdefault(pid, {}).setdefault(field_name, []).append(v)
    return {pid: {field_name: _robust_fence(vals) for field_name, vals in metrics.items()}
            for pid, metrics in values.items()}


def _stratum_fences_by_metric(raw_by_contact: list[tuple[dict, dict]]) -> dict:
    """{(player_id, shot_type): {metric_field_name: fence_or_None}} — the
    same fence construction as _player_fences_by_metric, but stratified by
    the contact's pose-derived shot_type (see shot_context.classify_shot_
    types) so a soft kitchen exchange is judged against this player's OTHER
    soft kitchen exchanges, not their drives. Untyped contacts
    (shot_type None) contribute to no stratum — they only ever compare
    against the player-wide baseline (flagged mixed downstream)."""
    values: dict = {}
    for contact, raw in raw_by_contact:
        shot_type = contact.get("shot_type")
        if shot_type is None:
            continue
        key = (contact["player_id"], shot_type)
        for field_name in METRIC_FIELDS:
            v = raw[field_name]
            if v is not None:
                values.setdefault(key, {}).setdefault(field_name, []).append(v)
    return {key: {field_name: _robust_fence(vals) for field_name, vals in metrics.items()}
            for key, metrics in values.items()}


def _finalize_vector(contact: dict, raw: dict, fences: dict,
                      stratum_fences: dict | None = None) -> ShotFeatureVector | None:
    """raw: one _extract_raw_metrics() result. fences: this shot's player's
    ALL-shots {metric_field_name: fence_or_None}, this video (see
    _player_fences_by_metric). stratum_fences: the same-shot-type slice for
    this shot's (player, shot_type) (see _stratum_fences_by_metric), or
    None/{} when the shot is untyped or its stratum is empty.

    Per metric, the same-type fence wins when it exists; falling back to the
    all-shots fence still yields a real comparison but is recorded in
    mixed_baseline_metrics (the comparison crosses shot types — see
    contracts.ShotFeatureVector). Returns None when no category has any real
    (comparable) metric — same "nothing to score" degrade as before, just
    driven by relative comparability now instead of confident pose alone."""
    stratum_fences = stratum_fences or {}
    devs: dict = {}
    softs: list[float] = []
    mixed_baseline_metrics: list[str] = []
    reference_flagged_metrics: list[str] = []
    for f in METRIC_FIELDS:
        fence = stratum_fences.get(f)
        if fence is None:
            fence = fences.get(f)
            if fence is not None and raw[f] is not None:
                mixed_baseline_metrics.append(f)
        devs[f] = _rel_dev(raw[f], fence)
        # Universal reference check (see REFERENCE_BANDS): wins over the
        # personal comparison only when it's the LARGER signal — so a
        # habitual gross flaw flags even though it's "typical for them"
        # (or when they have no personal fence at all), while a shot whose
        # personal outlier-ness is the bigger story keeps that framing.
        ref = _reference_dev(f, raw[f], contact.get("shot_type"))
        if ref is not None and ref != 0.0 and (devs[f] is None or abs(ref) > abs(devs[f])):
            devs[f] = ref
            reference_flagged_metrics.append(f)
        # soft_deviation: continuous distance from the fence midpoint (never
        # clamped to 0 inside the fence) — fill-to-limit ranking only, see
        # contracts.ShotFeatureVector.soft_deviation / select_tip_candidates.
        if raw[f] is not None and fence is not None:
            q1, q3, iqr = fence
            mid = (q1 + q3) / 2.0
            if iqr >= 1e-9:
                softs.append((raw[f] - mid) / iqr)
            else:
                softs.append(0.0 if abs(raw[f] - mid) < 1e-9
                             else math.copysign(1.0, raw[f] - mid))
        if ref is not None and ref != 0.0:
            softs.append(ref)
    category, deviation_score = _classify(devs)
    if category is None:
        return None
    deviating_metrics = [f for f, d in devs.items() if d is not None and d != 0.0]
    no_comparison_metrics = [f for f in METRIC_FIELDS
                              if raw[f] is not None
                              and stratum_fences.get(f) is None and fences.get(f) is None]
    return ShotFeatureVector(
        contact_id=contact["contact_id"], category=category,
        knee_bend_deg=raw["knee_bend_deg"], shoulder_rotation_deg=raw["shoulder_rotation_deg"],
        elbow_extension_deg=raw["elbow_extension_deg"], weight_transfer_score=raw["weight_transfer_score"],
        contact_height_m=raw["contact_height_m"],
        follow_through_completion_score=raw["follow_through_completion_score"],
        split_step_delay_s=raw["split_step_delay_s"], deviation_score=deviation_score,
        low_confidence_metrics=raw["low_confidence_metrics"],
        no_comparison_metrics=no_comparison_metrics,
        shot_type=contact.get("shot_type"),
        mixed_baseline_metrics=mixed_baseline_metrics,
        deviating_metrics=deviating_metrics,
        soft_deviation=max(softs, key=abs) if softs else None,
        reference_flagged_metrics=reference_flagged_metrics)


def compute_feature_vector(contact: dict, entries: list[dict], fps: float,
                            player_fences: dict | None = None,
                            stratum_fences: dict | None = None) -> ShotFeatureVector | None:
    """Single-shot convenience wrapper around _extract_raw_metrics +
    _finalize_vector. player_fences (see _player_fences_by_metric, one
    player's slice of it): the calling video's per-metric fences for this
    shot's player, or None/omitted to score every metric as "no comparison
    available" (e.g. calling this in isolation without a video's worth of
    other shots to build a fence from). stratum_fences: the same-shot-type
    slice (see _stratum_fences_by_metric). generate_coaching_report calls
    _extract_raw_metrics/_finalize_vector directly instead, so it can build
    every shot's raw metrics once before deriving fences across the whole
    video, rather than paying for entries/pose extraction twice per shot."""
    raw = _extract_raw_metrics(contact, entries, fps)
    if raw is None:
        return None
    return _finalize_vector(contact, raw, player_fences or {}, stratum_fences)


# ---------------------------------------------------------------------------
# Adaptive worst/best selection — pure, unit-tested directly.
# ---------------------------------------------------------------------------

def player_label(player_id) -> str:
    return f"Player {player_id}"


def _salience(vec: ShotFeatureVector, contact: dict) -> float:
    """How strongly this shot deserves a tip slot — |deviation| plus a bonus
    when the rally ended on it (likely an actual error, not just an atypical
    motion) and when multiple independent metrics deviated together (a real
    form breakdown, not one noisy keypoint). Ranking only — the flagged/not
    threshold stays WORST_DEVIATION_THRESHOLD on raw deviation."""
    s = abs(vec.deviation_score)
    if contact.get("is_rally_ending"):
        s += RALLY_ENDING_BONUS
    extra = max(0, len(vec.deviating_metrics) - 1)
    s += MULTI_METRIC_BONUS * min(extra, MULTI_METRIC_BONUS_CAP)
    return s


def _shot_summary(vec: ShotFeatureVector, contact: dict) -> dict:
    summary = {
        "contact_id": vec.contact_id,
        "category": vec.category,
        "deviation_score": vec.deviation_score,
        "frame_idx": contact.get("frame_idx"),
        "timestamp_s": contact.get("timestamp_s"),
        "rally_id": contact.get("rally_id"),
        "player_id": contact.get("player_id"),
        "player_label": player_label(contact.get("player_id")),
        "team": contact.get("team"),
        "shot_side": contact.get("shot_side"),
        "is_serve": bool(contact.get("is_serve")),
        "confidence": contact.get("confidence"),
        # Shot-type stratification + rally-outcome context (see
        # shot_context.py). rally fields are None/absent-safe: contacts
        # annotated before serve-exclusion carry them; synthetic test
        # contacts without them simply pass None through.
        "shot_type": vec.shot_type,
        "rally_shot_index": contact.get("rally_shot_index"),
        "rally_shot_count": contact.get("rally_shot_count"),
        "is_rally_ending": bool(contact.get("is_rally_ending")),
        "seconds_until_rally_end": contact.get("seconds_until_rally_end"),
        # Ranking signal (see _salience) + which metrics actually deviated —
        # both surfaced so the LLM's final cut and any eval tooling can see
        # WHY a shot ranked where it did.
        "salience": _salience(vec, contact),
        "deviating_metrics": list(vec.deviating_metrics),
        # Set only on praise candidates by _select_praise_exemplars: the
        # category this shot exemplifies done WELL (the player's most-flagged
        # category). None on improve candidates.
        "exemplar_of_category": None,
        # Surfaced at top level (not just nested in feature_vector) so the LLM
        # prompt, the UI, and any aggregate stats can all see at a glance
        # which metrics this shot has NO confident reading for, without
        # having to inspect every field of feature_vector for a null.
        "low_confidence_metrics": list(vec.low_confidence_metrics),
        "no_comparison_metrics": list(vec.no_comparison_metrics),
        "mixed_baseline_metrics": list(vec.mixed_baseline_metrics),
        # Metrics whose deviation came from the conservative UNIVERSAL check
        # (REFERENCE_BANDS) rather than the player's own fence — the only
        # metrics the LLM may coach as correct-form claims.
        "reference_flagged_metrics": list(vec.reference_flagged_metrics),
        "feature_vector": asdict(vec),
    }
    # Span moments (positioning focus) are stretches of movement, not swings —
    # surfaced so keyframes.py picks wider offsets and the LLM never describes
    # them as shots (see positioning.span_moments / the positioning SYSTEM
    # blocks). Absent on real contacts, never None-filled.
    if contact.get("kind") == "span":
        summary["kind"] = "span"
        summary["span"] = contact.get("span")
    # Ball-derived context (see ball_context.py) — present only when the
    # tracker actually measured it; absence means "no ball evidence", so
    # these keys are never None-filled.
    if contact.get("tempo_verified"):
        summary["tempo_verified"] = contact["tempo_verified"]
    if contact.get("rally_end_ball"):
        summary["rally_end_ball"] = contact["rally_end_ball"]
    return summary


def _suppress_close_duplicates(worst: list[dict], best: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drops the less-instructive of any two same-player candidates (worst or
    best, any category — cross-category too, since two tips about the same
    player a second apart are near-duplicates regardless of category) whose
    timestamp_s fall within MIN_TIP_GAP_S of each other. Different players are
    never suppressed against each other. Greedy by rank: process candidates
    most-instructive-first (worst before best; within a type, higher
    salience or higher confidence first) and keep a candidate only if
    no already-kept candidate for the same player is within the gap. A
    candidate with no timestamp_s is never suppressed and never suppresses
    another (missing metric, not "close")."""
    ranked = (
        [(0, -s["salience"], s) for s in worst]
        + [(1, -(s.get("confidence") or 0.0), s) for s in best]
    )
    ranked.sort(key=lambda r: (r[0], r[1]))

    kept_ids: set[str] = set()
    kept_times: dict = {}
    for _rank, _score, s in ranked:
        ts = s.get("timestamp_s")
        pid = s["player_id"]
        times = kept_times.setdefault(pid, [])
        if ts is not None and any(abs(ts - t) < MIN_TIP_GAP_S for t in times):
            continue
        kept_ids.add(s["contact_id"])
        if ts is not None:
            times.append(ts)

    return ([s for s in worst if s["contact_id"] in kept_ids],
            [s for s in best if s["contact_id"] in kept_ids])


def _select_praise_exemplars(vectors_meta: list[tuple[ShotFeatureVector, dict]],
                              worst_summaries: list[dict],
                              category_metrics: dict = CATEGORY_METRICS) -> list[dict]:
    """At most ONE praise candidate per player, at most MAX_PRAISE_TIPS
    total — chosen to be pedagogically CONNECTED to that player's flaws, not
    random encouragement: the exemplar is their most confident in-range shot
    that has a REAL (non-null) reading in their MOST-FLAGGED category, i.e.
    "this is what your footwork looks like when it's right". Its summary
    category is re-pointed at that category (exemplar_of_category marks it)
    so the tip lines up with the improve tips it reinforces.

    A player with no flagged shots (clean session) still gets a plain
    highest-confidence exemplar — encouragement isn't dropped, just
    deprioritized: when total praise exceeds MAX_PRAISE_TIPS, players with
    MORE flagged shots keep their exemplar first (their praise is doing
    teaching work). A player with no qualifying in-range shot simply gets
    none — never fabricated. This replaces the old detection-confidence
    z-score "best shot" pick, which measured wrist-peak detection strength,
    not shot quality."""
    flagged_cats_by_player: dict = {}
    for s in worst_summaries:
        flagged_cats_by_player.setdefault(s["player_id"], []).append(s["category"])

    in_range_by_player: dict = {}
    for vec, contact in vectors_meta:
        if vec.deviation_score == 0.0:
            in_range_by_player.setdefault(contact.get("player_id"), []).append((vec, contact))

    prioritized = []  # (n_flagged, confidence, summary)
    for pid, shots in in_range_by_player.items():
        flagged = flagged_cats_by_player.get(pid, [])
        target_cat = max(set(flagged), key=flagged.count) if flagged else None
        candidates = []
        for vec, contact in shots:
            conf = float(contact.get("confidence") or 0.0)
            if conf < MIN_BEST_CONFIDENCE:
                continue
            if target_cat and all(getattr(vec, f) is None for f in category_metrics[target_cat]):
                continue  # nothing real to praise in the category that matters
            candidates.append((conf, vec, contact))
        if not candidates:
            continue
        conf, vec, contact = max(candidates, key=lambda c: c[0])
        summary = _shot_summary(vec, contact)
        if target_cat:
            summary["category"] = target_cat
            summary["exemplar_of_category"] = target_cat
        prioritized.append((len(flagged), conf, summary))

    prioritized.sort(key=lambda t: (-t[0], -t[1]))
    return [summary for _n, _c, summary in prioritized[:MAX_PRAISE_TIPS]]


def select_worst_best(vectors_meta: list[tuple[ShotFeatureVector, dict]],
                       category_metrics: dict = CATEGORY_METRICS) -> tuple[list[dict], list[dict]]:
    """Adaptive severity selection — NOT a fixed top-N. worst_shots: shots
    whose |deviation_score| clears WORST_DEVIATION_THRESHOLD reference-range-
    widths beyond their category's own good-form bound (deviation_score is
    already normalized per-category, so the same threshold constant is an
    adaptive-per-category bar, not an absolute one); they carry a salience
    score (deviation + rally-ending + multi-metric bonuses, see _salience)
    that downstream ranking uses. best_shots: per-player praise exemplars
    tied to each player's most-flagged category (see
    _select_praise_exemplars). A clean session can surface 0-1 worst shots;
    a rough one can surface many — count is whatever clears the bar, never
    a fixed N."""
    if not vectors_meta:
        return [], []

    worst = [_shot_summary(vec, contact) for vec, contact in vectors_meta
             if abs(vec.deviation_score) >= WORST_DEVIATION_THRESHOLD]
    best = _select_praise_exemplars(vectors_meta, worst, category_metrics)
    return _suppress_close_duplicates(worst, best)


def select_tip_candidates(vectors_meta: list[tuple[ShotFeatureVector, dict]],
                           limit: int = MAX_TIPS,
                           category_metrics: dict = CATEGORY_METRICS,
                           fill_to_limit: bool = False,
                           category_cap: int | None = None) -> list[dict]:
    """Bounds select_worst_best's adaptive output to at most `limit` shots —
    worst shots (the actionable ones) get priority, ranked by SALIENCE (not
    raw deviation — see _salience: rally-ending mistakes and multi-metric
    breakdowns outrank equal-magnitude single-metric outliers), with at most
    MAX_PRAISE_TIPS exemplar praise slots so a clean session still surfaces
    some encouragement rather than an empty list.

    limit defaults to MAX_TIPS (the UI budget); build_report passes
    MAX_LLM_CANDIDATES instead so the LLM — which can see the frames — makes
    the final cut down to MAX_TIPS (see the SYSTEM selection rules).

    Budget is split as evenly as practical across the distinct players
    present (not hardcoded to 2 or 4 — a video may carry any count,
    especially before Part 0's singles-merge stopgap runs). A player short on
    notable contacts simply yields their unused share back to the pool
    rather than getting a padded/fabricated tip. Returned order is flagged
    improve tips salience-first, then below-threshold fill (when
    fill_to_limit), praise last — build_report relies on that when
    hard-capping the LLM's output back to MAX_TIPS.

    fill_to_limit (see FILL_TO_LIMIT): tops the shortlist up to `limit` with
    the highest-|soft_deviation| unflagged moments, each summary marked
    "below_threshold": True so the LLM (and any eval tooling) can tell filler
    from real flags. The flagged/not-flagged bar itself never moves — this
    widens what the LLM gets to see, not what counts as a problem.

    category_cap: coverage guard — at most this many flagged candidates from
    any single category, so one dominant failure mode (70 baseline-camping
    contacts on a real clip) can't monopolize the whole shortlist and crowd
    every other category out of the LLM's view. Applied salience-first, and
    ONLY when more than one category flagged (a genuinely one-issue session
    shouldn't have its list truncated toward nothing). Session-wide counts
    (session_context / _general_context) are built from ALL of vectors_meta,
    so capped-out shots still shape the aggregate story."""
    worst, best = select_worst_best(vectors_meta, category_metrics)
    worst = sorted(worst, key=lambda s: -s["salience"])
    if category_cap is not None and len({s["category"] for s in worst}) > 1:
        kept, by_cat = [], {}
        for s in worst:
            if by_cat.get(s["category"], 0) >= category_cap:
                continue
            kept.append(s)
            by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
        worst = kept
    best = sorted(best, key=lambda s: -(s.get("confidence") or 0.0))
    players = sorted({s["player_id"] for s in worst + best})
    if not players:
        return []

    per_player = max(1, limit // len(players))
    selected: list[dict] = []
    used_ids: set[str] = set()
    for pid in players:
        p_best = [s for s in best if s["player_id"] == pid][:1]  # ≤1 praise per player by construction
        p_worst = [s for s in worst if s["player_id"] == pid][:max(0, per_player - len(p_best))]
        for s in p_worst + p_best:
            selected.append(s)
            used_ids.add(s["contact_id"])

    # Fill any budget left by players who didn't have enough candidates to
    # fill their share, salience-first (best is already ≤MAX_PRAISE_TIPS
    # total, so no praise re-fill is needed).
    for s in worst:
        if len(selected) >= limit:
            break
        if s["contact_id"] not in used_ids:
            selected.append(s)
            used_ids.add(s["contact_id"])

    # Membership sets, not deviation_score == 0 checks: below-threshold fill
    # can carry a 0.0 deviation (in-band) without being praise.
    praise_ids = {s["contact_id"] for s in best}
    improve = sorted([s for s in selected if s["contact_id"] not in praise_ids],
                     key=lambda s: -s["salience"])
    praise = [s for s in selected if s["contact_id"] in praise_ids]

    fill: list[dict] = []
    if fill_to_limit and len(improve) + len(praise) < limit:
        # Respect the same per-player MIN_TIP_GAP_S spacing the flagged
        # selection got — filler must never reintroduce a near-duplicate
        # moment that _suppress_close_duplicates already rejected.
        kept_times: dict = {}
        for s in improve + praise:
            if s.get("timestamp_s") is not None:
                kept_times.setdefault(s["player_id"], []).append(s["timestamp_s"])
        # Genuinely-unflagged moments only: a flagged shot missing from
        # `selected` was excluded deliberately (near-duplicate suppression or
        # the category cap) — re-adding it as "below_threshold" filler would
        # both mislabel it and defeat the exclusion.
        pool = [(vec, c) for vec, c in vectors_meta
                if c["contact_id"] not in used_ids and vec.soft_deviation is not None
                and abs(vec.deviation_score) < WORST_DEVIATION_THRESHOLD]
        pool.sort(key=lambda vm: (-abs(vm[0].soft_deviation),
                                   -(vm[1].get("confidence") or 0.0)))
        for vec, c in pool:
            if len(improve) + len(praise) + len(fill) >= limit:
                break
            ts = c.get("timestamp_s")
            times = kept_times.setdefault(c.get("player_id"), [])
            if ts is not None and any(abs(ts - t) < MIN_TIP_GAP_S for t in times):
                continue
            s = _shot_summary(vec, c)
            s["below_threshold"] = True
            fill.append(s)
            used_ids.add(c["contact_id"])
            if ts is not None:
                times.append(ts)

    # Flagged improve salience-first, then fill, praise at the end, capped at
    # limit — the MAX_TIPS hard cap in build_report keeps the front, so
    # over-budget LLM output sheds praise, then filler, before real flags.
    return (improve + fill + praise)[:limit]


# ---------------------------------------------------------------------------
# Anthropic API call — same client/schema/degrade pattern as
# analysis/reporter.py's generate_llm_report. Called at most once per video.
# ---------------------------------------------------------------------------

# Internal metric names that would mislead the LLM, mapped to what the value
# actually measures. contact_height_m is NOT a height (no calibrated vertical
# coordinate exists — see _extract_raw_metrics) — it's distance from the net
# along the court. The rename happens only at the LLM boundary
# (_llm_shot_payload); DB rows, contracts, and the UI keep the internal name
# so nothing downstream of coaching_reports has to migrate.
LLM_METRIC_NAMES = {"contact_height_m": "distance_from_net_m"}


def _llm_shot_payload(shot: dict) -> dict:
    """One shot summary (see _shot_summary) with internal metric names
    swapped for their honest LLM-facing names (LLM_METRIC_NAMES), in the
    feature_vector keys AND both null-tracking name lists (top-level and the
    copies nested inside feature_vector). Tolerates partial dicts (tests call
    generate_llm_tips with bare {contact_id} shots)."""
    def rename_list(names):
        return [LLM_METRIC_NAMES.get(n, n) for n in names]

    out = dict(shot)
    fv = shot.get("feature_vector")
    metric_list_keys = ("low_confidence_metrics", "no_comparison_metrics",
                         "mixed_baseline_metrics", "deviating_metrics",
                         "reference_flagged_metrics")
    if fv:
        fv = {LLM_METRIC_NAMES.get(k, k): v for k, v in fv.items()}
        for key in metric_list_keys:
            if fv.get(key):
                fv[key] = rename_list(fv[key])
        out["feature_vector"] = fv
    for key in metric_list_keys:
        if out.get(key):
            out[key] = rename_list(out[key])
    return out


def _session_context(vectors_meta: list[tuple[ShotFeatureVector, dict]]) -> dict | None:
    """Per-player whole-session aggregates for the LLM prompt, built from
    EVERY scored shot in the video (all of vectors_meta) — not just the
    <=MAX_TIPS shots that get individual tips. This is what lets the model
    write a per-player theme grounded in the whole session ("footwork was
    flagged on 5 of your 14 shots") instead of extrapolating one from the
    handful of shots it happens to see. Counts only — no raw metric values —
    so there's nothing here the no-raw-numbers prompt rules could leak.
    Returns None when there's nothing to summarize."""
    per_player: dict = {}
    for vec, contact in vectors_meta:
        entry = per_player.setdefault(contact.get("player_id"), {
            "shots_analyzed": 0, "typical_shots": 0, "flagged_by_category": {}})
        entry["shots_analyzed"] += 1
        if abs(vec.deviation_score) >= WORST_DEVIATION_THRESHOLD:
            cats = entry["flagged_by_category"]
            cats[vec.category] = cats.get(vec.category, 0) + 1
        elif vec.deviation_score == 0.0:
            entry["typical_shots"] += 1
    return {player_label(pid): e for pid, e in per_player.items()} or None


# A category needs at least this many analyzed moments for one player before
# "cleanest category" is a meaningful praise pick — 1-2 clean moments is
# absence of evidence, not evidence of a strength.
MIN_GENERAL_CATEGORY_SAMPLES = 3


def _general_context(vectors_meta: list[tuple[ShotFeatureVector, dict]]) -> dict | None:
    """Deterministic per-player session aggregates behind general_feedback
    (one praise + one improve per player) — the coarse pass: pure
    category-level math over EVERY scored moment, no LLM. Besides the raw
    counts, two deterministic picks per player: strongest_negative_category
    (most flagged — what the improve piece should be about) and
    strongest_positive_category (their best-sampled cleanest category — what
    the praise piece should be about). A None pick is an honest gap ("the
    data doesn't support a call"), never backfilled; the same category is
    never both picks."""
    per_player: dict = {}
    for vec, contact in vectors_meta:
        cats = per_player.setdefault(contact.get("player_id"), {})
        entry = cats.setdefault(vec.category, {"n": 0, "flagged": 0})
        entry["n"] += 1
        if abs(vec.deviation_score) >= WORST_DEVIATION_THRESHOLD:
            entry["flagged"] += 1

    out: dict = {}
    for pid, cats in per_player.items():
        flagged = {c: e for c, e in cats.items() if e["flagged"] > 0}
        negative = (max(flagged, key=lambda c: (flagged[c]["flagged"],
                                                 flagged[c]["flagged"] / flagged[c]["n"]))
                    if flagged else None)
        sampled = {c: e for c, e in cats.items()
                   if e["n"] >= MIN_GENERAL_CATEGORY_SAMPLES and c != negative}
        positive = (min(sampled, key=lambda c: (sampled[c]["flagged"] / sampled[c]["n"],
                                                 -sampled[c]["n"]))
                    if sampled else None)
        out[player_label(pid)] = {
            "by_category": cats,
            "strongest_negative_category": negative,
            "strongest_positive_category": positive,
        }
    return out or None


TIPS_SCHEMA = {
    "type": "object",
    "properties": {
        "tips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string"},
                    "tip": {"type": "string", "description": "2-3 short sentences from a "
                            "coach who is invested in this player, not a lab report. Shape: "
                            "(1) name what's happening for the player first, so the critique "
                            "lands as being on their side, not (2) the critique itself, kept "
                            "just as pointed and specific as the data supports — do not soften "
                            "or vague it up, then (3) a clear, actionable and encouraging close. "
                            "The critique's directness never gets diluted, only reframed. No "
                            "joint angles, degrees, anatomical terms, confidence scores, or raw "
                            "timings/velocities of any kind. Give a little of the why, not just "
                            "the instruction, while staying scannable."},
                    "tone": {"type": "string", "enum": ["improve", "praise"]},
                },
                "required": ["contact_id", "tip", "tone"],
                "additionalProperties": False,
            },
        },
        "general_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "player_label": {"type": "string", "description": "Exactly one of the "
                                     "player_label values present in the shots you were given."},
                    "praise": {"type": "string", "description": "1-2 sentences, coach voice: the "
                               "ONE thing genuinely working for this player this session — real "
                               "reinforcement of something observed, never filler. Grounded in "
                               "general_context/session_context and the shots. Same "
                               "no-raw-numbers/no-anatomy rules as tips."},
                    "improve": {"type": "string", "description": "1-2 sentences, coach voice: the "
                                "single most valuable thing for this player to fix — the dominant "
                                "negative session pattern, one pointed fix, not a list. Same "
                                "rules as tips."},
                },
                "required": ["player_label", "praise", "improve"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tips", "general_feedback"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# SYSTEM prompt — composed per focus by system_prompt() below from shared
# blocks plus focus-specific ones (which metric family the shots carry and
# how deviation_score was computed for it: form = per-player Tukey fences,
# positioning = absolute court-geometry bands — the framings are mutually
# exclusive, never show both). The form composition is bound to module-level
# SYSTEM so existing imports and the substring-pinning tests keep working.
# ---------------------------------------------------------------------------

FOCUS_FORM = "form"
FOCUS_POSITIONING = "positioning"

_SYS_INTRO_FORM = """You are a pickleball coach giving quick, spoken-style tips on
individual shots from one session — plus, for each player, one thing that's
genuinely working and one thing to fix — based on pose-derived shot
mechanics that a computer-vision pipeline extracted at each detected
contact/swing — and, when available, actual still frames of the player
taking each shot."""

_SYS_INPUT_FORM = """You receive a list of shots, each with a category (footwork, paddle_prep,
contact_point, or follow_through), a deviation_score, a feature_vector of raw
pose measurements (joint angles in degrees, 0-1 scores, seconds, metres), a
low_confidence_metrics list, and a no_comparison_metrics list.

One feature_vector field needs special care: distance_from_net_m is where on
the court (how far from the net, in court-plane metres) the player was when
they hit — it says NOTHING about how high or low they met the ball. The
pipeline has no vertical measurement at all. Never make any claim about
contact height, meeting the ball low/high, or letting the ball drop — that
information does not exist in this data. Court positioning claims ("you're
hanging back near the baseline", "you took that one from mid-court") are the
only valid use of this field."""

_SYS_FRAMES = """Some or all shots may come with labeled still frames (prep / contact /
follow-through — cropped around the hitting player, matched to the shot by
contact_id). When frames are present for a shot they are your PRIMARY
evidence: look at them first, and ground the critique in what the player is
visibly doing — stance, balance, paddle position, body shape, where they are
on the court. The pose numbers then become corroboration: where a metric
agrees with what you see, be confident; where a metric contradicts what the
frames clearly show, trust your eyes and drop the claim (2D pose readings
are noisy). Something you can clearly see in a shot's frames is real
evidence, and you may state it even if the corresponding numeric metric is
null for that shot — the null rules below ban inventing claims from missing
data, not describing what is visibly in front of you. But apply the same
honesty to the image itself: small crops can be blurry or ambiguous, so if
you can't clearly make something out, don't claim it. Speak like a coach
reviewing film with the player ("I can see you're...", "look at where your
feet are here") — never mention frames, images, crops, cameras, or that
you're an AI looking at pictures."""

_SYS_DEVIATION_FORM = """IMPORTANT — deviation_score is relative to THIS PLAYER by default, not a
universal standard: it measures how far this shot falls from THIS PLAYER's
OWN typical value for that category's metric(s), computed across their
OTHER shots in this same session. Negative/positive = notably outside their
own usual range for this category and needs work; 0 = typical for them, a
solid shot. Frame those critiques as consistency feedback (e.g. "shallower
than your usual knee bend on this shot" or "you're not getting as low as
you normally do"), never as a claim about correct/ideal form in general
(never "your knee bend should be X degrees" or "that's below the ideal
range").

The ONE exception: metrics listed in a shot's reference_flagged_metrics
failed a deliberately conservative universal check — a gross fundamental
that is off under any reasonable standard of play (standing straight-legged
through a shot, a contact jammed right into the body, an arm-only swing or
a chopped-off follow-through on a hard drive). For THOSE metrics only, you
MAY coach correct form directly ("bend your knees and get low", "step into
it and finish your swing") — that is exactly the feedback a player with a
CONSISTENT flaw needs, because a habit they repeat on every shot never
stands out from their own baseline. When the same metric keeps appearing in
reference_flagged_metrics across many shots, treat it as one recurring
habit worth a theme, not many separate discoveries. For every other metric
there is no validated universal range and personal consistency is the only
honest comparison."""

_SYS_SEVERITY = """Calibrate how strongly you word a critique to |deviation_score|, translated
qualitatively and never recited as a number: magnitude around 1-2 means the
shot drifted a bit outside their usual — note it lightly; 2-4 means it was
clearly off their normal — be direct about it; above 4 means it was way
outside anything else they did this session — treat it as the shot to talk
about. Matching your intensity to the evidence is what keeps the feedback
credible."""

_SYS_SHOT_TYPE_FORM = """Each shot may carry a shot_type — a coarse, pose-derived stratification key:
court zone plus whether the swing was soft or fast RELATIVE TO THAT PLAYER'S
OWN session (e.g. "kitchen_soft", "backcourt_fast", or just a zone like
"midcourt" when tempo couldn't be judged). It is NOT a verified stroke
label: treat "kitchen_soft" as "one of their slower swings taken up at the
kitchen" — plausibly a dink exchange — never as certainty about which stroke
they hit. Use it to keep advice appropriate to the situation (don't coach a
soft kitchen exchange like a baseline drive), and prefer situational
phrasing ("on those soft exchanges at the kitchen") over asserting stroke
names as fact.

When the player hit enough shots of the same shot_type this session,
deviation_score is computed against those same-type shots — that is what
stops a soft shot being "flagged" merely for not looking like a drive.
mixed_baseline_metrics lists the metrics where that wasn't possible and the
comparison fell back to ALL of their shots regardless of type. A critique
leaning on a metric in mixed_baseline_metrics crosses shot types, so soften
cross-type claims accordingly — especially follow-through and swing-size
claims on soft/kitchen shots, where "shorter than your usual" may only mean
"shorter than your drives"."""

_SYS_RALLY = """Rally context: rally_shot_index / rally_shot_count place the shot inside its
rally, and is_rally_ending marks the LAST detected shot of a rally — the
point ended on or right after it. The data does NOT say whether that shot
was a winner or an error, and you must never assert either (no "that one
won you the point", no "that's the shot that cost you the rally"). What you
MAY do: use it for stakes-aware framing ("this was the last shot of that
rally — worth getting right"), and for pattern-spotting — several
rally-ending shots sharing the same flaw is exactly the kind of theme worth
naming. seconds_until_rally_end is internal timing signal under the same
rules as every other raw number: never recite it."""

_SYS_BALL = """Some sessions include ball-tracking evidence; when a shot carries it:
- tempo_verified ('soft' | 'fast'): the swing's tempo measured from the
  actual tracked ball speed rather than inferred from the player's motion.
  A shot_type built on it is trustworthy about tempo — coach the situation
  accordingly, and trust it over any impression the pose numbers give.
- rally_end_ball, only ever on a rally's FINAL shot: 'landed_out' means the
  tracked ball's bounce after that swing was called outside the court — you
  MAY treat that shot as the error that ended the rally ("that one sailed
  long on you") and give it that weight; 'landed_in_unreturned' means it
  bounced in and was never returned — you MAY treat it as the shot that won
  the point and praise what made it work. These are measured outcomes, the
  ONE exception to the winner/error rule above; on any shot without
  rally_end_ball that rule applies in full.
Ball readings come from a computer-vision tracker: never recite speeds or
any other number from them, and if a shot's frames clearly contradict the
label, trust the frames."""

_SYS_NO_NUMBERS = """These raw numbers are internal signal only — NEVER mention degrees, angles, confidence/certainty scores, or anatomical terms (no "elbow extension",
"shoulder rotation", "knee bend angle", etc.) in your output. Translate them
into plain language a rec/intermediate pickleball player would actually
understand and act on (e.g. "you're rushing your paddle back" instead of
"delayed unit turn", or "plant your feet before you swing" instead of
"insufficient weight transfer")."""

_SYS_TENDENCIES = """You may also receive a player_tendencies object — small, rounded, per-player
summaries (session-level tendencies, not per-shot data) such as how quickly a
player tends to get back to the kitchen after a point, or how their shots
tend to distribute across court zones. Treat these exactly like the pose
metrics above: they are internal signal to inform what you say, never
numbers to say out loud. Never state a raw seconds value, a velocity, or any
sub-second timing — translate a slow kitchen-arrival tendency into something
like "you're slow getting back to the net after your shot," never "kitchen
arrival averages 2 seconds." Small rounded shot counts (e.g. "you rarely hit
deep to the backcourt") may inform your framing but should stay in service of
a coaching point, not be recited as a stat line. If player_tendencies is
absent or empty, ignore it — say nothing about session-level tendencies for
that player."""

_SYS_SESSION_CTX = """You may also receive a session_context object: for each player, how many of
their shots were analyzed across the WHOLE session (not just the shots shown
to you), how many were typical for them, and how often each category was
flagged. This is the ground truth for what actually kept happening — a
pattern that shows up repeatedly in session_context is a theme; a category
flagged once is a moment, not a pattern. Use it to decide each player's
general_feedback and to keep individual tips in proportion (don't declare
"you always rush your prep" off one flagged shot when session_context shows
the other ten were fine). Like everything else, these counts are internal
signal: paraphrase qualitatively ("this kept showing up", "most of your
shots were solid"), never recite tallies. If session_context is absent,
base the general feedback only on the shots you can see and keep it
modest."""

_SYS_FEEDBACK = """You may also receive tip_feedback_history: for each tip category, how
often this player has marked past tips in it helpful or not helpful across
their sessions. Use it only as a light tiebreaker when choosing among
similarly strong candidates — lean toward categories that have landed
before, and for a category that keeps getting rejected, rethink the ANGLE
of the advice (how it's framed), never water down its substance. NEVER
suppress a clearly severe moment because its category polled badly, never
mention ratings, and if tip_feedback_history is absent, ignore this
entirely."""

_SYS_NULLS = """IMPORTANT — data gaps are real and marked, not hidden: some feature_vector
fields are null. A null field means the computer-vision system did NOT get a
confident reading for that specific measurement on that specific shot (pose
tracking lost the joint, or its confidence was too low to trust) — it is not
zero, not average, not "fine", it is simply not known. low_confidence_metrics
names exactly which fields on this shot are null. You must NEVER state or
imply a specific claim about a joint/measurement listed in
low_confidence_metrics for that shot (e.g. do not say their elbow was
extended, their knees were bent well, etc. if elbow_extension_deg or
knee_bend_deg is null for that shot). The category and deviation_score you're
given are always computed from whatever real (non-null) data the shot has, so
still write a tip grounded in that category — just don't invent specifics
about the fields that are null. If nearly everything relevant to a shot's
category is null, it's fine for the tip to stay general/encouraging rather
than specific. This null-handling rule applies to player_tendencies too: a
player missing from player_tendencies (or missing a particular tendency) has
no known reading for it — never fabricate a claim about it, and never treat
their absence as "average" or "fine".

That same spirit extends to no_comparison_metrics: a field can be listed
there even though its raw reading WAS confident — it means this player
simply hasn't hit enough shots THIS SESSION for that specific metric to know
what's typical for them yet, so there is nothing real to compare this shot
against. You must NEVER claim or imply a comparison (e.g. "shallower than
your usual", "better than your average", "not as extended as normal for you")
for a field listed in a shot's no_comparison_metrics, even though you may
still know its raw value. If a shot's category is driven mostly by fields in
no_comparison_metrics, keep the tip general/encouraging about that category
rather than inventing a personal baseline that doesn't exist yet."""

_SYS_VOICE = """Write like a coach who is invested in this specific player, not a lab report
or a performance review — and NOT a compliment sandwich that pads or dilutes
the critique. Every "improve" tip follows the same three-beat shape:
  1. Acknowledge what's happening for the player right now, so the critique
     that follows reads as a coach who's paying attention to them, not a
     verdict handed down from nowhere.
  2. Deliver the actual critique — just as pointed, specific, and concrete as
     the data supports. Do NOT soften, hedge, vague-up, or water down the
     substance to make it feel gentler. If a shot is genuinely off, say
     exactly what's wrong and exactly what to do about it.
  3. Close on something actionable and encouraging — the fix, and why it'll
     help — so the player leaves knowing what to do next, not just what they
     did wrong.
The reframing is entirely in the packaging (who this is being said to, and
in what order) — never in the substance of the critique itself.

For example, given the same underlying shot data, do NOT write the flat,
clinical version:
  "You're barely putting any energy into this shot—almost no weight transfer
  and the contact point is too low. Step in, load your weight, and finish
  your swing so the ball has somewhere to go."
Instead, write the same critique in the three-beat coach voice, e.g.:
  "I can see you're not fully committing to this shot yet — there's almost
  no weight transfer and you're meeting the ball too low to drive it. Step
  in and load your weight before you swing, then finish all the way
  through — that's what gives the ball somewhere to go. Get that timing
  down and this shot picks up a lot of pop."
Notice the critique itself (low weight transfer, low contact point, the
fix) is exactly as specific and direct in both versions — only the framing
around it changed.

For each shot you SELECT (see the selection rules below) with a
deviation_score != 0, write one "improve" tip: what to change and a bit of
why it matters, in 2-3 short sentences, concrete, critical where warranted,
and encouraging in tone rather than content. For each selected shot with
deviation_score == 0, write one "praise" tip: what they're doing well and
why it works, so they know what to keep doing. Return exactly one tip per
SELECTED shot, each referencing that shot's contact_id so it can be matched
back up; shots you don't select simply get no entry."""

_SYS_GENERAL = """Also return general_feedback: exactly one entry per distinct player_label
present in the shots, and none for any label not present. Each entry is two
short pieces in the same coach voice, 1-2 sentences each: "praise" — the
ONE thing genuinely working for this player this session — and "improve" —
the single most valuable thing to fix. Ground both in the general_context
object when present: for each player it aggregates EVERY analyzed moment by
category (how many, how many flagged) and names strongest_positive_category
/ strongest_negative_category — the session-wide patterns the aggregates
actually support. Prefer those categories; a null pick means the data does
not support a call there, so stay modest and base that piece only on what
you can see — never invent a session-wide pattern from a single moment.
Praise is real reinforcement of something observed, never filler; improve
is one pointed fix, not a list. Write each tip so it reads as part of that
player's general feedback when the moment fits the pattern, but every tip
must still stand alone. All the rules above — no raw numbers, no anatomical
terms, null and no-comparison discipline — apply to general_feedback
exactly as they do to tips."""

_SYS_DISCLAIMER = """Do not overstate certainty — this is
computer-vision-derived pose data, not a substitute for in-person coaching."""

_SYS_SELECTION = f"""THE FINAL CUT IS YOURS. You may be given more shots than you
should write about — up to {MAX_LLM_CANDIDATES} candidates, pre-ranked by a
statistical salience score (larger deviations, rally-ending shots, and
multi-metric breakdowns rank higher). The numbers cannot see the video; you
can. Write tips for AT MOST {MAX_TIPS} shots total (praise included),
choosing the most instructive coaching moments:
  1. Prefer shots where the frames VISIBLY show the flagged problem — a
     mistake a player could see themselves making.
  2. Prefer rally-ending mistakes (is_rally_ending) — the moments that
     cost points teach best.
  3. Prefer covering each player and their recurring theme over piling
     several tips onto one issue.
SKIP a shot entirely — no tip at all — when its frames clearly contradict
what the metrics claim (pose-tracking noise) or show nothing coachable;
never write a tip you can't stand behind from the evidence you can see.
Never exceed {MAX_TIPS} tips, and never more than one "praise" tip per
player.

A shot with exemplar_of_category set is a PRAISE candidate: that player's
cleanest example of the category their flagged shots struggle with. If you
select it, tone is "praise" and the tip should point at what doing that
category well looks like for them, so it reinforces the improve tips
rather than being generic encouragement.

Some shots may carry below_threshold: true — they did NOT cross the
statistical flag bar and were only included to fill out your shortlist.
Treat them as borderline: pick one only when its frames clearly show
something coachable, word it lightly, and skip them freely — an
unremarkable below_threshold shot is exactly the one to leave out."""

# --------------------------- positioning focus -----------------------------

_SYS_INTRO_POSITIONING = """You are a pickleball coach giving quick, spoken-style tips on court
positioning and movement from one session — plus, for each player, one
thing that's genuinely working and one thing to fix — based on where a
computer-vision pipeline measured each player standing and moving on the
court — and, when available, actual still frames of the moments in
question."""

_SYS_INPUT_POSITIONING = """You receive a list of moments, each with a category (net_position,
transition, court_coverage, or team_spacing), a deviation_score, a
feature_vector of raw positioning measurements (metres and seconds), and a
low_confidence_metrics list naming every field with no reading for that
moment.

Two kinds of moment appear. Most are tied to a specific shot the player
hit. Others carry "kind": "span" — those are STRETCHES OF MOVEMENT between
shots (their span object gives the metric and its time range), not swings:
dwelling in no-man's-land, being slow to come in behind a return, drifting
apart from a partner. Never describe a span moment as a shot or a swing —
talk about where the player is standing and how they're moving around that
moment.

The metric names say exactly what they measure: kitchen_distance_at_contact_m
(metres behind their own non-volley-zone line when they hit — negative means
inside the kitchen), return_depth_m (metres inside their own baseline when
returning serve), transition_dwell_s (longest continuous stay in the
midcourt band), kitchen_approach_after_return_s (seconds from their return
of serve until they were established at their own kitchen line),
recovery_offset_m (lateral metres off the center line when the opponent next
struck, after this player was pulled wide — singles only), and
partner_depth_stagger_m (depth split between partners held for a second or
more — doubles only)."""

_SYS_DEVIATION_POSITIONING = """IMPORTANT — deviation_score here is measured against the COURT ITSELF,
not against this player's own habits: the court is a fixed size, and each
metric's acceptable band comes from its real geometry (where the
non-volley-zone line actually is, how long a body takes to cross the
midcourt band). 0 = within what good positioning allows; positive/negative
= outside it. Because the standard is external and real, you MAY frame
positioning feedback as what good positioning looks like in general ("you
want to be at the kitchen line by now", "that's no-man's-land — don't set
up camp there"). Do not dress it up as a comparison to this player's own
usual game — the data says where they were on the court, not what their
habits are."""

_SYS_SEVERITY_POSITIONING = """Calibrate how strongly you word a critique to |deviation_score|, translated
qualitatively and never recited as a number: magnitude around 1-2 means they
drifted a bit outside good positioning — note it lightly; 2-4 means they
were clearly out of position — be direct about it; above 4 means they were
badly out of position for a sustained stretch — treat it as the thing to
talk about. Matching your intensity to the evidence is what keeps the
feedback credible."""

_SYS_CONTEXT_POSITIONING = """Positioning moments carry no shot_type and no mixed_baseline_metrics —
every comparison here is against the court's own geometry, never across
shot types. rally_shot_index still places a shot-anchored moment inside its
rally; the early-rally shots where deep positioning is CORRECT (serve,
return, third shot) are already excluded from the depth metrics, so never
scold a player for standing deep early in a rally."""

# Focus-specific prompt blocks, keyed by focus then slot; the shared blocks
# above apply to every focus. The deviation blocks are mutually exclusive on
# purpose: form deviations are per-player-relative (Tukey fences), positioning
# deviations are absolute court geometry — composing both framings into one
# prompt would make each one a lie half the time.
_FOCUS_BLOCKS = {
    FOCUS_FORM: {
        "intro": _SYS_INTRO_FORM,
        "input": _SYS_INPUT_FORM,
        "deviation": _SYS_DEVIATION_FORM,
        "severity": _SYS_SEVERITY,
        "context": _SYS_SHOT_TYPE_FORM,
    },
    FOCUS_POSITIONING: {
        "intro": _SYS_INTRO_POSITIONING,
        "input": _SYS_INPUT_POSITIONING,
        "deviation": _SYS_DEVIATION_POSITIONING,
        "severity": _SYS_SEVERITY_POSITIONING,
        "context": _SYS_CONTEXT_POSITIONING,
    },
}


def system_prompt(focus: str = FOCUS_FORM) -> str:
    """The coaching SYSTEM prompt for one focus: the shared blocks with the
    focus-specific intro / input description / deviation semantics / extra
    context interleaved in reading order. The form composition reproduces the
    pre-split prompt verbatim (module-level SYSTEM below)."""
    if focus not in _FOCUS_BLOCKS:
        raise ValueError(f"unknown coaching focus: {focus!r}")
    b = _FOCUS_BLOCKS[focus]
    return "\n\n".join([
        b["intro"], b["input"], _SYS_FRAMES, b["deviation"], b["severity"],
        b["context"], _SYS_RALLY, _SYS_BALL, _SYS_NO_NUMBERS, _SYS_TENDENCIES,
        _SYS_SESSION_CTX, _SYS_FEEDBACK, _SYS_NULLS, _SYS_VOICE, _SYS_GENERAL,
        _SYS_DISCLAIMER, _SYS_SELECTION,
    ])


SYSTEM = system_prompt(FOCUS_FORM)


def resolve_focus(stored: str | None, override: str | None = None) -> str:
    """The focus a report should be generated under: explicit per-call
    override first (API query param), then the video's stored coaching_focus
    column, then form — a NULL/unset clip behaves exactly as it did before
    selectable focus existed."""
    focus = override or stored or FOCUS_FORM
    if focus not in (FOCUS_FORM, FOCUS_POSITIONING):
        raise ValueError(f"unknown coaching focus: {focus!r}")
    return focus


def generate_llm_tips(shots: list[dict], stats_summary: dict | None = None,
                       session_context: dict | None = None,
                       frames_by_contact: dict[str, list[dict]] | None = None,
                       system: str | None = None,
                       general_context: dict | None = None,
                       feedback_history: dict | None = None,
                       ) -> tuple[dict[str, dict], list[dict]]:
    """Returns ({contact_id: {tip, tone}}, [{player_label, praise, improve},
    ...]), or ({}, []) (with a logged reason) when the API key is missing,
    the call fails, or the response can't be parsed — degrades gracefully,
    same as analysis/reporter.py. Never trusted for frame/timestamp/player
    attribution — those are attached from our own contact_events data
    (build_report validates each general_feedback entry's player_label
    against the players we actually sent, and drops anything unrecognized).

    stats_summary (see _summarize_stats), session_context (see
    _session_context) and frames_by_contact (see keyframes.
    extract_for_candidates — {contact_id: [{label, jpeg_b64}, ...]}) are
    optional extra context folded into this same single API call — never a
    second Anthropic call — and are simply omitted from the prompt input
    when None/empty rather than zero-filled or fabricated. With frames the
    user message becomes a block list (JSON text block + labeled image
    blocks per shot); without them it stays a plain JSON string, exactly as
    before frames existed.

    Runs on ANTHROPIC_COACHING_MODEL (not the discrepancy report's
    ANTHROPIC_REPORT_MODEL): translating weak numeric signal into pointed
    coaching language under the SYSTEM prompt's never-say-X constraints is a
    nuanced writing task, and it fires at most once per video on explicit
    user action, so the heavier model's cost is bounded."""
    if not shots:
        return {}, []
    if not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set — skipping coaching tips")
        return {}, []
    import anthropic

    payload: dict = {"shots": [_llm_shot_payload(s) for s in shots]}
    if stats_summary:
        payload["player_tendencies"] = stats_summary
    if session_context:
        payload["session_context"] = session_context
    if general_context:
        payload["general_context"] = general_context
    if feedback_history:
        payload["tip_feedback_history"] = feedback_history

    content: str | list = db.dumps(payload)
    if frames_by_contact:
        blocks: list[dict] = [{"type": "text", "text": db.dumps(payload)}]
        for s in shots:
            cid = s.get("contact_id")
            for fr in frames_by_contact.get(cid) or []:
                blocks.append({"type": "text",
                               "text": f"contact_id {cid} — {fr['label']} frame:"})
                blocks.append({"type": "image",
                               "source": {"type": "base64", "media_type": "image/jpeg",
                                           "data": fr["jpeg_b64"]}})
        content = blocks

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        # thinking must be ADAPTIVE here: ANTHROPIC_COACHING_MODEL defaults to
        # claude-sonnet-5, where the old {"type": "enabled", "budget_tokens": N}
        # shape is removed and returns a 400 (it still works on the Haiku 4.5
        # discrepancy-report path in analysis/reporter.py — do not "align" the
        # two calls). Depth is steered by output_config.effort instead.
        # max_tokens caps thinking + response TOGETHER on this model family.
        # 8000 truncated the tips JSON mid-array on a real 15-candidate run,
        # then 16000 truncated again on a real 16-candidate positioning run
        # (adaptive thinking at effort=high ate the budget both times) — each
        # time degrading the whole report to zero tips. Streaming removes the
        # non-streaming HTTP-timeout ceiling, so give it real headroom.
        with client.messages.stream(
            model=ANTHROPIC_COACHING_MODEL,
            max_tokens=64000,
            thinking={"type": "adaptive"},
            # system defaults to the form prompt; build_report passes the
            # focus-composed one (system_prompt(focus)) for positioning runs.
            system=system or SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": TIPS_SCHEMA},
                            "effort": "high"},
            messages=[{"role": "user", "content": content}],
        ) as llm_stream:
            response = llm_stream.get_final_message()
    except anthropic.RateLimitError:
        log.error("Anthropic rate limit — try generating the coaching report again shortly")
        return {}, []
    except anthropic.AuthenticationError:
        log.error("Anthropic API key invalid — check .env")
        return {}, []
    except anthropic.APIStatusError as exc:
        log.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        return {}, []
    except anthropic.APIConnectionError:
        log.error("network error reaching the Anthropic API")
        return {}, []

    if response.stop_reason == "refusal":
        log.error("coaching LLM call was refused (stop_reason=refusal) — no tips generated")
        return {}, []
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = db.loads(text, {})
    except Exception:  # noqa: BLE001
        # Distinguish the two real-world causes: a truncated response (raise
        # max_tokens / lower candidate count) vs genuinely malformed output.
        if response.stop_reason == "max_tokens":
            log.error("coaching LLM response TRUNCATED at max_tokens — tips JSON incomplete; "
                      "raise max_tokens in generate_llm_tips")
        else:
            log.error("LLM returned unparseable JSON (stop_reason=%s)", response.stop_reason)
        return {}, []

    out: dict[str, dict] = {}
    for t in data.get("tips") or []:
        cid, tip, tone = t.get("contact_id"), t.get("tip"), t.get("tone")
        if cid and tip and tone in ("improve", "praise"):
            out[cid] = {"tip": tip, "tone": tone}
    general = [{"player_label": g.get("player_label"), "praise": g.get("praise"),
                "improve": g.get("improve")}
               for g in data.get("general_feedback") or []
               if g.get("player_label") and g.get("praise") and g.get("improve")]
    return out, general


def _summarize_stats(stats: dict | None) -> dict | None:
    """Compacts report_stats.get_report_stats()'s per-rally time-series output
    into the small, LLM-safe summary generate_llm_tips folds into its one
    prompt: per-player rounded (whole-second) average kitchen-arrival
    tendency plus how many rallies that's based on, and the already-small
    per-zone shot counts. Never dumps the raw forward_momentum series or
    per-rally sub-second kitchen_arrival_s values — those are aggregated away
    here, not passed through. A player/rally with no real reading for a given
    stat is simply absent from the summary, never zero-filled or averaged in
    as 0. Returns None when there's nothing worth passing to the prompt."""
    if not stats:
        return None

    arrivals: dict[str, list[float]] = {}
    for rally in (stats.get("rallies") or {}).values():
        for pid, secs in (rally.get("kitchen_arrival_s") or {}).items():
            if secs is not None:
                arrivals.setdefault(pid, []).append(secs)

    kitchen_tendencies = {
        player_label(pid): {
            "avg_kitchen_arrival_s_rounded": round(sum(vals) / len(vals)),
            "rallies_with_data": len(vals),
        }
        for pid, vals in arrivals.items()
    }

    shot_distribution = {
        player_label(pid): sides
        for pid, sides in (stats.get("shot_counts") or {}).items()
    }

    summary: dict = {}
    if kitchen_tendencies:
        summary["kitchen_arrival_tendencies"] = kitchen_tendencies
    if shot_distribution:
        summary["shot_distribution_by_zone"] = shot_distribution
    return summary or None


# ---------------------------------------------------------------------------
# Orchestration — the only DB-touching layer. build_report is the pure core
# (video_id/vectors_meta -> CoachingReport, LLM call injectable) so tests can
# exercise the ratio math and adaptive selection without a DB or anthropic.
# ---------------------------------------------------------------------------

def build_report(video_id: str, vectors_meta: list[tuple[ShotFeatureVector, dict]],
                  llm_fn=None, stats_summary: dict | None = None,
                  frames_fn=None, focus: str = FOCUS_FORM,
                  feedback_history: dict | None = None) -> CoachingReport:
    category_metrics = (positioning.CATEGORY_METRICS if focus == FOCUS_POSITIONING
                        else CATEGORY_METRICS)
    # Hand the LLM a WIDER statistical shortlist than the tip budget — it can
    # see the frames, so it makes the final cut down to MAX_TIPS (see the
    # SYSTEM selection rules), dropping pose-noise false positives that the
    # statistics alone can't distinguish from real mistakes. The category cap
    # keeps one dominant failure mode from monopolizing the shortlist.
    candidates = select_tip_candidates(vectors_meta, limit=MAX_LLM_CANDIDATES,
                                        category_metrics=category_metrics,
                                        fill_to_limit=FILL_TO_LIMIT,
                                        category_cap=max(2, MAX_LLM_CANDIDATES // 2))
    session_context = _session_context(vectors_meta)
    general_context = _general_context(vectors_meta)
    log.info("coaching: %d/%d scored moments selected as LLM candidates (focus=%s, fill_to_limit=%s)",
              len(candidates), len(vectors_meta), focus, FILL_TO_LIMIT)
    if llm_fn is None:
        # The real LLM call gets the focus-composed prompt, the general-
        # feedback aggregates and the tip-rating history; injected test
        # doubles keep their existing signatures and simply never see them
        # (including doubles monkeypatched over generate_llm_tips itself —
        # hence the same TypeError fallback the argument ladder below uses).
        sys_text = system_prompt(focus)

        def llm_fn(*args):
            try:
                return generate_llm_tips(*args, system=sys_text,
                                          general_context=general_context,
                                          feedback_history=feedback_history)
            except TypeError:
                return generate_llm_tips(*args)
    # frames_fn (candidates -> {contact_id: [{label, jpeg_b64}, ...]} | None)
    # is called AFTER selection so keyframes are only ever extracted for the
    # <=MAX_TIPS shots that actually reach the LLM, never every contact in
    # the video. Vision context is an upgrade, not a dependency: any
    # exception degrades to a frameless call rather than failing the report.
    frames_by_contact = None
    if frames_fn is not None and candidates:
        try:
            frames_by_contact = frames_fn(candidates)
        except Exception:  # noqa: BLE001
            log.warning("keyframe extraction failed — generating coaching report without frames",
                        exc_info=True)
    # llm_fn is the one Anthropic call for this video. Injected/mocked llm_fns
    # in tests may only accept the original (shots) or (shots, stats)
    # signatures -- fall back rather than requiring every test double to
    # accept every context argument.
    try:
        result = llm_fn(candidates, stats_summary, session_context, frames_by_contact)
    except TypeError:
        try:
            result = llm_fn(candidates, stats_summary, session_context)
        except TypeError:
            try:
                result = llm_fn(candidates, stats_summary)
            except TypeError:
                result = llm_fn(candidates)
    # generate_llm_tips returns (tips, general_feedback); older-style
    # injected doubles return just the tips dict -- treat that as "no
    # general feedback", not an error.
    if isinstance(result, tuple):
        tip_text, general_raw = result
    else:
        tip_text, general_raw = result, []
    # Player attribution is our own data (contact_events.player_id), not
    # trusted from the model's echoed output — attach it here regardless of
    # what the LLM response contains.
    # Candidates are ordered improve-by-salience then praise (see
    # select_tip_candidates), so if the model ignores its selection budget
    # and tips more than MAX_TIPS shots, the hard cap below keeps the most
    # salient improve tips and drops praise/low-salience overflow first.
    tips = [
        CoachingTip(contact_id=c["contact_id"], frame=c["frame_idx"], timestamp_s=c["timestamp_s"],
                    category=c["category"], tip=tip_text[c["contact_id"]]["tip"],
                    tone=tip_text[c["contact_id"]]["tone"],
                    player_id=c["player_id"], player_label=c["player_label"])
        for c in candidates if c["contact_id"] in tip_text
    ][:MAX_TIPS]
    # General feedback: the model's echoed player_label is only a JOIN KEY
    # back to the players we actually sent — an unrecognized/duplicated label
    # is dropped (first occurrence wins), and player_id always comes from our
    # own contact_events data, mirroring the tip-attribution rule above.
    # player_themes stays empty on new reports (replaced by general_feedback;
    # legacy rows still rehydrate their themes via get_report).
    label_to_pid = {c["player_label"]: c["player_id"] for c in candidates}
    general, seen_labels = [], set()
    for g in general_raw:
        lbl = g.get("player_label")
        if lbl in label_to_pid and lbl not in seen_labels and g.get("praise") and g.get("improve"):
            general.append(GeneralFeedback(player_id=label_to_pid[lbl], player_label=lbl,
                                            praise=g["praise"], improve=g["improve"]))
            seen_labels.add(lbl)
    return CoachingReport(video_id=video_id, tips=tips, general=general, focus=focus)


def exclude_serves(contacts: list[dict]) -> list[dict]:
    """Serves are excluded from the coaching narrative/worst-best selection
    entirely — filtered once here, at the earliest point the contact list is
    assembled, rather than downstream in feature-vector computation or
    worst/best selection. A rally that's all serves (or a video with zero
    non-serve contacts overall) just yields fewer/no contacts here; the rest
    of the pipeline already degrades an empty contact/vector list to an empty
    tip list (see select_tip_candidates), never an error or serve fallback."""
    return [c for c in contacts if not c.get("is_serve")]


def singles_pid_map(contacts: list[dict], match_type: str | None) -> dict:
    """{phantom_pid: top2_pid} — the aggregate form of merge_singles_players'
    per-contact fold, for data keyed by overlay player id rather than contact
    rows (positioning's per-frame span samples). Each phantom id maps to
    whichever top-2 player any of its contacts sits nearest in time, so the
    two folds agree on which real player absorbs a phantom. Empty dict when
    no folding applies (not singles, or ≤2 identities) — and a phantom id
    that appears ONLY in overlays (zero contacts) has no evidence to map it
    with, so it stays unmapped and is filtered by positioning's
    MIN_SPAN_SAMPLES floor instead."""
    if match_type != "singles":
        return {}
    player_ids = sorted({c["player_id"] for c in contacts})
    if len(player_ids) <= 2:
        return {}
    counts: dict = {}
    for c in contacts:
        counts[c["player_id"]] = counts.get(c["player_id"], 0) + 1
    top2 = sorted(counts, key=lambda pid: -counts[pid])[:2]
    top2_frames = {pid: [c["frame_idx"] for c in contacts if c["player_id"] == pid] for pid in top2}
    mapping: dict = {}
    for pid in player_ids:
        if pid in top2:
            continue
        frames = [c["frame_idx"] for c in contacts if c["player_id"] == pid]
        mapping[pid] = min(top2, key=lambda t: min(
            (abs(f - tf) for f in frames for tf in top2_frames[t]), default=float("inf")))
    return mapping


def merge_singles_players(contacts: list[dict], match_type: str | None) -> list[dict]:
    """Stopgap for player-identity mis-tracking (see docs/CONTEXT.md TODO):
    if this clip is tagged singles but contact_events carries more than 2
    distinct player_ids (spurious/phantom identities from the pose tracker),
    fold every contact from a non-top-2 player into whichever of the top-2
    players it's closest to in time. In-memory only, scoped to this report —
    never writes contact_events back. Doubles / unset clips are untouched."""
    if match_type != "singles":
        return contacts
    player_ids = sorted({c["player_id"] for c in contacts})
    if len(player_ids) <= 2:
        return contacts

    counts: dict = {}
    for c in contacts:
        counts[c["player_id"]] = counts.get(c["player_id"], 0) + 1
    top2 = sorted(counts, key=lambda pid: -counts[pid])[:2]
    top2_frames = {pid: [c["frame_idx"] for c in contacts if c["player_id"] == pid] for pid in top2}
    log.info("coaching: singles clip has %d distinct players %s — merging extras into top-2 %s",
              len(player_ids), player_ids, top2)

    merged = []
    for c in contacts:
        if c["player_id"] in top2:
            merged.append(c)
            continue
        nearest_pid = min(top2, key=lambda pid: min(
            (abs(c["frame_idx"] - f) for f in top2_frames[pid]), default=float("inf")))
        merged.append({**c, "player_id": nearest_pid})
    return merged


def _tip_feedback_history() -> dict | None:
    """{category: {"helpful": n, "not_helpful": n}} across ALL videos — the
    read-back half of the tip_feedback loop (ratings snapshot their category
    at rating time, so this survives report regeneration and contact
    re-detection). Global rather than per-video on purpose: a per-video
    history would usually be the very report being replaced. Folded into the
    LLM payload as a LIGHT selection tiebreaker (see the SYSTEM feedback
    block) — never a deterministic suppression, so a severe flag can't be
    silenced by its category having polled badly."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT category, rating, COUNT(*) AS n FROM tip_feedback"
            " WHERE category IS NOT NULL GROUP BY category, rating").fetchall()
    hist: dict = {}
    for r in rows:
        entry = hist.setdefault(r["category"], {"helpful": 0, "not_helpful": 0})
        entry["helpful" if r["rating"] == 1 else "not_helpful"] += r["n"]
    return hist or None


def generate_coaching_report(video_id: str, rally_source: str | None = None,
                              overlay_source: str | None = None,
                              focus: str | None = None) -> CoachingReport:
    """Only builds feature vectors for contacts belonging to a rally
    readiness.get_analyzable_rallies() currently clears (Pass 2 done, overlay
    coverage still matches the rally's current frame range) — a stale/partial
    rally's contacts are excluded from compute_feature_vector.

    focus: 'form' | 'positioning' | None — None resolves through the video's
    stored coaching_focus column, then defaults to form (see resolve_focus)."""
    # rally_source can be None by default (see contact_detect.py's matching
    # fix) — resolve overlay_source the same way here so the read_range calls
    # below aren't left with an unresolved None.
    overlay_source = overlay_source or rally_source or _default_overlay_source(video_id)
    rallies = get_analyzable_rallies(video_id, rally_source=rally_source, overlay_source=overlay_source)
    ready_rally_ids = {r["id"] for r in rallies}
    with db.connect() as con:
        vrow = con.execute("SELECT fps, match_type, coaching_focus, path FROM videos WHERE id=?",
                            (video_id,)).fetchone()
        contacts = db.rows_to_dicts(con.execute(
            "SELECT * FROM contact_events WHERE video_id=? ORDER BY frame_idx",
            (video_id,)).fetchall())
    focus = resolve_focus(vrow["coaching_focus"] if vrow else None, focus)
    match_type = vrow["match_type"] if vrow else None
    fps = float(vrow["fps"]) if vrow and vrow["fps"] else 30.0
    contacts = [c for c in contacts if c["rally_id"] in ready_rally_ids]
    # Rally context BEFORE serve exclusion so rally_shot_index/count describe
    # the real rally (serve = shot 1); shot types AFTER the singles merge so
    # phantom-identity contacts feed the right player's tempo median.
    contacts = annotate_rally_context(contacts, rallies, fps)
    contacts = exclude_serves(contacts)
    # pid_map from the PRE-merge contacts (it needs the phantom ids) — used
    # below to fold positioning's overlay-keyed span samples the same way
    # merge_singles_players folds the contacts themselves.
    pid_map = singles_pid_map(contacts, match_type)
    contacts = merge_singles_players(contacts, match_type)

    # One rally-span overlay read, shared by ball context (both focuses) and
    # positioning's per-frame samples.
    entries_by_rally = {r["id"]: overlays.read_range(video_id, overlay_source,
                                                      r["start_frame"], r["end_frame"])
                        for r in rallies}

    # Ball-derived enrichment (see ball_context.py) — measured swing tempo
    # (beats the wrist-speed inference in classify_shot_types) and the
    # measured rally-ending outcome, attached to the rally's final contact.
    # A clip with no ball data yields nothing here and everything downstream
    # behaves exactly as before.
    from . import ball_context
    tempo_override: dict = {}
    contacts_by_rally: dict = {}
    for c in contacts:
        contacts_by_rally.setdefault(c["rally_id"], []).append(c)
    ball_by_contact: dict = {}
    rally_end_ball: dict = {}
    for rally_id, rc in contacts_by_rally.items():
        by_c, end_ball = ball_context.annotate_contacts(
            entries_by_rally.get(rally_id) or [], rc, fps)
        ball_by_contact.update(by_c)
        if end_ball:
            rally_end_ball[rally_id] = end_ball
    for cid, ctx in ball_by_contact.items():
        if ctx.get("tempo_verified"):
            tempo_override[cid] = ctx["tempo_verified"]
    if ball_by_contact or rally_end_ball:
        log.info("coaching: ball context on %d contacts (%d tempo-verified), "
                  "%d rally outcomes", len(ball_by_contact), len(tempo_override),
                  len(rally_end_ball))

    shot_types = classify_shot_types(contacts, tempo_override=tempo_override)
    contacts = [{**c, "shot_type": shot_types.get(c["contact_id"]),
                 **ball_by_contact.get(c["contact_id"], {})} for c in contacts]
    contacts = [{**c, "rally_end_ball": rally_end_ball[c["rally_id"]]}
                if c.get("is_rally_ending") and c["rally_id"] in rally_end_ball else c
                for c in contacts]

    if focus == FOCUS_POSITIONING:
        # Positioning: absolute court-geometry scoring over per-frame court
        # positions (whole rallies, not contact windows) + the contact events
        # themselves — see positioning.py. No pose-fence passes needed.
        samples_by_rally: dict = {}
        for r in rallies:
            samples = position_samples(entries_by_rally.get(r["id"]) or [])
            if pid_map:
                folded: dict = {}
                for pid, s in samples.items():
                    folded.setdefault(pid_map.get(pid, pid), []).extend(s)
                samples = {pid: sorted(s) for pid, s in folded.items()}
            samples_by_rally[r["id"]] = samples
        vectors_meta = positioning.build_vectors(
            contacts, rallies, samples_by_rally, fps, match_type, video_id)
    else:
        pre_frames, post_frames = int(PRE_WINDOW_S * fps), int(POST_WINDOW_S * fps)
        # Two passes: first pull every shot's raw (un-scored) metrics, then derive
        # each player's own per-metric fence across their whole video, and only
        # then classify/score each shot relative to THEIR fence — a shot can't be
        # scored against a distribution that isn't fully built yet.
        raw_by_contact: list[tuple[dict, dict]] = []
        for c in contacts:
            entries = overlays.read_range(video_id, overlay_source,
                                           max(0, c["frame_idx"] - pre_frames),
                                           c["frame_idx"] + post_frames)
            raw = _extract_raw_metrics(c, entries, fps)
            if raw is not None:
                raw_by_contact.append((c, raw))

        player_fences = _player_fences_by_metric(raw_by_contact)
        stratum_fences = _stratum_fences_by_metric(raw_by_contact)
        vectors_meta = []
        for c, raw in raw_by_contact:
            vec = _finalize_vector(c, raw, player_fences.get(c["player_id"], {}),
                                    stratum_fences.get((c["player_id"], c.get("shot_type"))))
            if vec is not None:
                vectors_meta.append((vec, c))

    stats_summary = None
    try:
        from . import report_stats  # lazy: report_stats imports from this module
        stats_summary = _summarize_stats(
            report_stats.get_report_stats(video_id, rally_source=rally_source, overlay_source=overlay_source))
    except Exception:  # noqa: BLE001 — stats are an add-on; never block the report
        log.warning("report_stats unavailable for %s — generating coaching report without it", video_id, exc_info=True)

    # Vision context: crops of the hitting player at prep/contact/follow-
    # through for each SELECTED tip candidate (build_report calls this only
    # after selection). Missing/unreadable video path (e.g. an s3-mode row
    # whose local scratch file is gone) degrades to a frameless call inside
    # keyframes/build_report — never blocks the report.
    video_path = vrow["path"] if vrow and vrow["path"] else None
    frames_fn = None
    if video_path:
        from . import keyframes  # lazy: keeps cv2 out of coaching's import graph

        def frames_fn(candidates):  # noqa: F811 — deliberate conditional def
            return keyframes.extract_for_candidates(
                video_path, candidates, video_id, overlay_source, fps,
                PRE_WINDOW_S, POST_WINDOW_S)

    report = build_report(video_id, vectors_meta, stats_summary=stats_summary,
                           frames_fn=frames_fn, focus=focus,
                           feedback_history=_tip_feedback_history())
    save_report(report)
    return report


def save_report(report: CoachingReport) -> None:
    # stale_reason explicitly NULL: this report was just built by the
    # currently-running (confidence-aware) pipeline, so it's current by
    # definition — never inherit staleness from whatever row it replaces.
    with db.connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO coaching_reports"
            " (video_id, tips_json, themes_json, general_json, focus, created_at, stale_reason)"
            " VALUES (?,?,?,?,?,?,NULL)",
            (report.video_id, db.dumps([asdict(t) for t in report.tips]),
             db.dumps([asdict(t) for t in report.player_themes]),
             db.dumps([asdict(g) for g in report.general]),
             report.focus, db.now()))


def get_report(video_id: str) -> CoachingReport | None:
    with db.connect() as con:
        row = con.execute("SELECT * FROM coaching_reports WHERE video_id=?", (video_id,)).fetchone()
    if row is None:
        return None
    # themes_json / general_json / focus are NULL on rows predating their
    # migrations (columns added without a backfill) — empty lists / None,
    # not an error.
    return CoachingReport(
        video_id=row["video_id"],
        tips=[CoachingTip(**t) for t in db.loads(row["tips_json"], [])],
        player_themes=[PlayerTheme(**t) for t in db.loads(row["themes_json"] or "[]", [])],
        stale_reason=row["stale_reason"],
        focus=row["focus"],
        general=[GeneralFeedback(**g) for g in db.loads(row["general_json"] or "[]", [])])


if __name__ == "__main__":
    # ponytail self-check: a value at the player's own median must land at
    # deviation_score 0.0; too few of the player's own shots for a metric
    # must yield None (no fence, no comparison), never a fabricated 0;
    # select_tip_candidates must cap total output at MAX_TIPS even when far
    # more shots clear the bar.
    fence = _robust_fence([120.0, 125.0, 130.0, 135.0, 140.0])
    assert fence is not None
    assert _rel_dev(130.0, fence) == 0.0          # at own median -> typical
    assert _rel_dev(500.0, fence) is not None and _rel_dev(500.0, fence) > 0  # way outside own fence
    assert _robust_fence([120.0, 130.0]) is None  # below MIN_SAMPLES_FOR_DISTRIBUTION -> no fence
    assert _rel_dev(130.0, None) is None          # no fence -> no comparison, not 0

    devs_typical = {f: 0.0 for f in METRIC_FIELDS}
    cat, dev = _classify(devs_typical)
    assert dev == 0.0, dev

    devs_no_fence = {f: None for f in METRIC_FIELDS}
    cat, dev = _classify(devs_no_fence)
    assert cat is None and dev is None, (cat, dev)  # nothing comparable anywhere -> unscoreable

    bad_vectors_meta = [
        (ShotFeatureVector(contact_id=f"bad{i}", category="footwork",
                            knee_bend_deg=0.0, shoulder_rotation_deg=0.0, elbow_extension_deg=0.0,
                            weight_transfer_score=0.0, contact_height_m=0.0,
                            follow_through_completion_score=0.0, split_step_delay_s=0.0,
                            deviation_score=-5.0),
         # timestamps spread well beyond MIN_TIP_GAP_S apart -- this check is
         # about the MAX_TIPS cap, not the time-gap suppression filter.
         {"contact_id": f"bad{i}", "frame_idx": i, "timestamp_s": i * 10.0, "rally_id": "r",
          "player_id": 1, "team": "near", "shot_side": "left", "is_serve": False, "confidence": 0.9})
        for i in range(20)
    ]
    candidates = select_tip_candidates(bad_vectors_meta)
    assert len(candidates) == MAX_TIPS, len(candidates)

    # merge_singles_players: 4-player "singles" clip collapses to the 2 most-
    # active players, each extra contact folded into its nearest (by frame).
    contacts = (
        [{"player_id": 1, "frame_idx": f} for f in (0, 10, 20, 30)]
        + [{"player_id": 2, "frame_idx": f} for f in (5, 15, 25, 35)]
        + [{"player_id": 3, "frame_idx": 6}]   # phantom id, nearest to player 1 (f=10, d=4) vs player 2 (f=5, d=1)
        + [{"player_id": 4, "frame_idx": 31}]  # phantom id, nearest to player 1 (f=30, d=1)
    )
    merged = merge_singles_players(contacts, "singles")
    assert {c["player_id"] for c in merged} == {1, 2}
    by_frame = {c["frame_idx"]: c["player_id"] for c in merged}
    assert by_frame[6] == 2
    assert by_frame[31] == 1
    assert merge_singles_players(contacts, "doubles") == contacts
    assert merge_singles_players(contacts, None) == contacts
    print("coaching self-check OK")

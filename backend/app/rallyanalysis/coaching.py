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
from ..config import ANTHROPIC_API_KEY, ANTHROPIC_REPORT_MODEL
from ..cv import overlays
from ..cv.court import NET_Y
from .contracts import CoachingReport, CoachingTip, ShotFeatureVector
from .readiness import _default_overlay_source, get_analyzable_rallies

log = logging.getLogger("rallyanalysis.coaching")

# COCO-17 keypoint indices (same convention as contact_detect.py / players.py).
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# ---------------------------------------------------------------------------
# HEURISTIC REFERENCE RANGES — starting-point biomechanical guardrails only.
# These are NOT validated against professional or ground-truth pickleball
# data; a human coach should sanity-check/re-tune them before trusting
# flagged output. Values are chosen to be plausible defaults, nothing more.
# ---------------------------------------------------------------------------
KNEE_BEND_GOOD_RANGE_DEG = (100.0, 155.0)          # hip-knee-ankle angle at contact
SHOULDER_ROTATION_GOOD_RANGE_DEG = (15.0, 70.0)    # shoulder-line vs hip-line twist at contact
ELBOW_EXTENSION_GOOD_RANGE_DEG = (110.0, 175.0)    # shoulder-elbow-wrist angle, hitting arm, at contact
WEIGHT_TRANSFER_GOOD_RANGE = (0.35, 1.0)           # 0-1 score, see compute_feature_vector
CONTACT_HEIGHT_GOOD_RANGE_M = (-1.5, 1.5)          # proxy — see contact_height_m note below
FOLLOW_THROUGH_GOOD_RANGE = (0.4, 1.0)             # 0-1 completion score
SPLIT_STEP_GOOD_RANGE_S = (0.05, 0.35)             # seconds between plant and contact

# Normalization constants (turn raw pixel/metre displacements into 0-1 scores).
WEIGHT_TRANSFER_NORM_M = 1.0   # forward court-plane movement treated as "full" transfer
FOLLOW_THROUGH_NORM = 1.8      # post-contact wrist path length, as a multiple of the
                                # player's own on-screen box height, treated as "full"

PRE_WINDOW_S = 0.6   # how far before contact we look for split-step / prep
POST_WINDOW_S = 0.45  # how far after contact we look for follow-through

WORST_DEVIATION_THRESHOLD = 1.0  # |deviation_score| >= this many reference-range-widths
                                   # beyond the category's own good-form bound counts as
                                   # "worst" — adaptive PER CATEGORY (each category has its
                                   # own reference range baked into _norm_dev's normalization),
                                   # not a fixed top-N count: a clean session naturally clears
                                   # 0-1 shots, a rough one can clear many.
K_STD_BEST = 0.5          # best-shot z-score threshold over confidence (in-range shots only)
MIN_BEST_CONFIDENCE = 0.3

MAX_TIPS = 8       # keeps the UI list short and readable regardless of rally count
MAX_PRAISE_TIPS = 3  # reserves most of the cap for actionable ("improve") tips

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


def _neutral(rng: tuple[float, float]) -> float:
    return (rng[0] + rng[1]) / 2.0


def compute_feature_vector(contact: dict, entries: list[dict], fps: float) -> ShotFeatureVector | None:
    """contact: one contact_events row (dict) — contact_id, frame_idx,
    player_id, court_x, court_y, .... entries: overlay rows covering a
    [contact_frame - PRE_WINDOW_S, contact_frame + POST_WINDOW_S] window
    (any order, any players). Returns None only when this player has no pose
    data anywhere in the window (contact gets skipped, not fabricated).

    Individual metrics that lack CONFIDENT pose support (missing keypoint, or
    below KP_CONF_MIN) are excluded — left None, never backfilled with the
    midpoint of their reference range. A metric we didn't confidently see is
    absent, not silently reported as "in range". Excluded metric names are
    collected in low_confidence_metrics.
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

    category, deviation_score = _classify(
        knee_bend_deg, shoulder_rotation_deg, elbow_extension_deg,
        weight_transfer_score, contact_height_m, follow_through_completion_score,
        split_step_delay_s)
    if category is None:
        # Every category's metrics were excluded — nothing real to score.
        # (In practice unreachable today since contact_height_m always has a
        # real reading, but this is what "no scoreable data" degrades to if
        # that ever changes, rather than fabricating a category/score.)
        return None

    return ShotFeatureVector(
        contact_id=contact["contact_id"], category=category,
        knee_bend_deg=knee_bend_deg, shoulder_rotation_deg=shoulder_rotation_deg,
        elbow_extension_deg=elbow_extension_deg, weight_transfer_score=weight_transfer_score,
        contact_height_m=contact_height_m,
        follow_through_completion_score=follow_through_completion_score,
        split_step_delay_s=split_step_delay_s, deviation_score=deviation_score,
        low_confidence_metrics=low_confidence_metrics)


def _norm_dev(value: float | None, rng: tuple[float, float]) -> float | None:
    """Signed deviation from [lo, hi], expressed in units of range-width so
    metrics with different units (degrees, seconds, 0-1 scores) are
    comparable. 0 = within range. None (metric excluded — no confident pose
    support) passes through as None rather than being treated as 0/in-range."""
    if value is None:
        return None
    lo, hi = rng
    width = hi - lo
    if value < lo:
        return (value - lo) / width
    if value > hi:
        return (value - hi) / width
    return 0.0


def _classify(knee_bend_deg, shoulder_rotation_deg, elbow_extension_deg,
              weight_transfer_score, contact_height_m, follow_through_completion_score,
              split_step_delay_s) -> tuple[str | None, float | None]:
    """category = whichever of the four categories has the single worst
    (largest-magnitude) normalized deviation among its REAL (non-excluded)
    component metrics; deviation_score is that category's signed deviation.
    A category every one of whose metrics was excluded for missing/low-
    confidence pose data is never eligible to win — that would be the same
    fabricated-neutral bug at the category level even if individual metrics
    are properly excluded. A shot with every metric in range gets an
    arbitrary-but-stable category (max of all-zero deviations) and
    deviation_score 0.0 (i.e. not flagged — see compute_category_ratios).
    Returns (None, None) only if literally no category has any real metric
    (compute_feature_vector treats that as unscoreable, same as no pose data
    at all)."""
    per_category = {
        "footwork": [_norm_dev(split_step_delay_s, SPLIT_STEP_GOOD_RANGE_S),
                     _norm_dev(knee_bend_deg, KNEE_BEND_GOOD_RANGE_DEG)],
        "paddle_prep": [_norm_dev(shoulder_rotation_deg, SHOULDER_ROTATION_GOOD_RANGE_DEG)],
        "contact_point": [_norm_dev(elbow_extension_deg, ELBOW_EXTENSION_GOOD_RANGE_DEG),
                           _norm_dev(weight_transfer_score, WEIGHT_TRANSFER_GOOD_RANGE),
                           _norm_dev(contact_height_m, CONTACT_HEIGHT_GOOD_RANGE_M)],
        "follow_through": [_norm_dev(follow_through_completion_score, FOLLOW_THROUGH_GOOD_RANGE)],
    }
    category_dev = {}
    for cat, devs in per_category.items():
        real = [d for d in devs if d is not None]
        if real:
            category_dev[cat] = max(real, key=abs)
    if not category_dev:
        return None, None
    category = max(category_dev, key=lambda c: abs(category_dev[c]))
    return category, category_dev[category]


# ---------------------------------------------------------------------------
# Adaptive worst/best selection — pure, unit-tested directly.
# ---------------------------------------------------------------------------

def player_label(player_id) -> str:
    return f"Player {player_id}"


def _shot_summary(vec: ShotFeatureVector, contact: dict) -> dict:
    return {
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
        # Surfaced at top level (not just nested in feature_vector) so the LLM
        # prompt, the UI, and any aggregate stats can all see at a glance
        # which metrics this shot has NO confident reading for, without
        # having to inspect every field of feature_vector for a null.
        "low_confidence_metrics": list(vec.low_confidence_metrics),
        "feature_vector": asdict(vec),
    }


def select_worst_best(vectors_meta: list[tuple[ShotFeatureVector, dict]]) -> tuple[list[dict], list[dict]]:
    """Adaptive severity selection — NOT a fixed top-N. worst_shots: shots
    whose |deviation_score| clears WORST_DEVIATION_THRESHOLD reference-range-
    widths beyond their category's own good-form bound (deviation_score is
    already normalized per-category, so the same threshold constant is an
    adaptive-per-category bar, not an absolute one). best_shots: in-range
    shots (deviation_score == 0) whose detection confidence clears
    mean + K_STD_BEST * std of confidence among this video's in-range shots,
    with an absolute confidence floor so a low-confidence "in range"
    detection never gets held up as exemplary. A clean session can surface
    0-1 worst shots; a rough one can surface many — count is whatever
    clears the bar, never a fixed N."""
    if not vectors_meta:
        return [], []

    worst = [_shot_summary(vec, contact) for vec, contact in vectors_meta
             if abs(vec.deviation_score) >= WORST_DEVIATION_THRESHOLD]

    in_range = [(vec, contact) for vec, contact in vectors_meta if vec.deviation_score == 0.0]
    best = []
    if in_range:
        confs = np.array([float(contact.get("confidence") or 0.0) for _vec, contact in in_range])
        threshold = confs.mean() + K_STD_BEST * confs.std()
        for (vec, contact), conf in zip(in_range, confs):
            if conf >= threshold and conf >= MIN_BEST_CONFIDENCE:
                best.append(_shot_summary(vec, contact))

    return worst, best


def select_tip_candidates(vectors_meta: list[tuple[ShotFeatureVector, dict]]) -> list[dict]:
    """Bounds select_worst_best's adaptive output to at most MAX_TIPS shots so
    the tip list stays short — worst shots (the actionable ones) get
    priority, with at most MAX_PRAISE_TIPS praise slots reserved so a clean
    session still surfaces some encouragement rather than an empty list.

    Budget is split as evenly as practical across the distinct players
    present (not hardcoded to 2 or 4 — a video may carry any count,
    especially before Part 0's singles-merge stopgap runs). A player short on
    notable contacts simply yields their unused share back to the pool
    rather than getting a padded/fabricated tip."""
    worst, best = select_worst_best(vectors_meta)
    worst = sorted(worst, key=lambda s: -abs(s["deviation_score"]))
    best = sorted(best, key=lambda s: -(s.get("confidence") or 0.0))
    players = sorted({s["player_id"] for s in worst + best})
    if not players:
        return []

    per_player = max(1, MAX_TIPS // len(players))
    per_player_praise = max(1, MAX_PRAISE_TIPS // len(players))
    selected: list[dict] = []
    used_ids: set[str] = set()
    for pid in players:
        p_best = [s for s in best if s["player_id"] == pid][:per_player_praise]
        p_worst = [s for s in worst if s["player_id"] == pid][:max(0, per_player - len(p_best))]
        for s in p_worst + p_best:
            selected.append(s)
            used_ids.add(s["contact_id"])

    # Fill any budget left by players who didn't have enough candidates to
    # fill their share, worst-first, then praise (still capped by MAX_TIPS
    # and MAX_PRAISE_TIPS overall).
    n_praise = sum(1 for s in selected if s["deviation_score"] == 0.0)
    for s in worst:
        if len(selected) >= MAX_TIPS:
            break
        if s["contact_id"] not in used_ids:
            selected.append(s)
            used_ids.add(s["contact_id"])
    for s in best:
        if len(selected) >= MAX_TIPS or n_praise >= MAX_PRAISE_TIPS:
            break
        if s["contact_id"] not in used_ids:
            selected.append(s)
            used_ids.add(s["contact_id"])
            n_praise += 1

    return selected[:MAX_TIPS]


# ---------------------------------------------------------------------------
# Anthropic API call — same client/schema/degrade pattern as
# analysis/reporter.py's generate_llm_report. Called at most once per video.
# ---------------------------------------------------------------------------

TIPS_SCHEMA = {
    "type": "object",
    "properties": {
        "tips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string"},
                    "tip": {"type": "string", "description": "2-3 short sentences in the "
                            "voice of a real coach talking to the player, not a lab report: "
                            "direct, specific, warm but willing to be critical when a shot "
                            "calls for it. No joint angles, degrees, anatomical terms, "
                            "confidence scores, or raw timings/velocities of any kind. Give "
                            "a little of the why, not just the instruction, while staying "
                            "scannable."},
                    "tone": {"type": "string", "enum": ["improve", "praise"]},
                },
                "required": ["contact_id", "tip", "tone"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tips"],
    "additionalProperties": False,
}

SYSTEM = """You are a pickleball coach giving quick, spoken-style tips on
individual shots from one player's session, based on pose-derived shot
mechanics (not video) that a computer-vision pipeline extracted at each
detected contact/swing.

You receive a list of shots, each with a category (footwork, paddle_prep,
contact_point, or follow_through), a deviation_score (negative/positive =
outside the good-form range and needs work; 0 = solid, in-range shot), a
feature_vector of raw pose measurements (joint angles in degrees, 0-1 scores,
seconds), and a low_confidence_metrics list. These raw numbers are internal
signal only — NEVER mention degrees, angles, confidence/certainty scores, or
anatomical terms (no "elbow extension", "shoulder rotation", "knee bend
angle", etc.) in your output. Translate them into plain language a
rec/intermediate pickleball player would actually understand and act on
(e.g. "you're rushing your paddle back" instead of "delayed unit turn", or
"plant your feet before you swing" instead of "insufficient weight
transfer").

You may also receive a player_tendencies object — small, rounded, per-player
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
that player.

IMPORTANT — data gaps are real and marked, not hidden: some feature_vector
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

Write like a real coach talking to the player, not a lab report or a
performance review. Be supportive, but be clear and honest — if a shot is
genuinely off, say so plainly and tell them what to actually do about it
("bend your knees more before you swing," "get your paddle back earlier"),
rather than softening it into vague, hedged, corporate-sounding feedback.
Concrete and direct beats polite and generic.

For each shot given a deviation_score != 0, write one "improve" tip: what to
change and a bit of why it matters, in 2-3 short sentences, concrete,
critical where warranted, and encouraging in tone rather than content. For
each shot with deviation_score == 0, write one "praise" tip: what they're
doing well and why it works, so they know what to keep doing. Return exactly
one tip per shot you were given, each referencing that shot's contact_id so
it can be matched back up. Do not overstate certainty — this is
computer-vision-derived pose data, not a substitute for in-person coaching."""


def generate_llm_tips(shots: list[dict], stats_summary: dict | None = None) -> dict[str, dict]:
    """Returns {contact_id: {tip, tone}}, or {} (with a logged reason) when
    the API key is missing, the call fails, or the response can't be parsed —
    degrades gracefully, same as analysis/reporter.py. Never trusted for
    frame/timestamp — those are attached from our own contact_events data.

    stats_summary (see _summarize_stats) is optional extra context folded
    into this same single API call — never a second Anthropic call — and is
    simply omitted from the prompt input when None/empty rather than
    zero-filled or fabricated."""
    if not shots:
        return {}
    if not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set — skipping coaching tips")
        return {}
    import anthropic

    payload: dict = {"shots": shots}
    if stats_summary:
        payload["player_tendencies"] = stats_summary

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = client.messages.create(
            model=ANTHROPIC_REPORT_MODEL,
            max_tokens=4000,
            thinking={"type": "enabled", "budget_tokens": 1500},
            system=SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": TIPS_SCHEMA}},
            messages=[{"role": "user", "content": db.dumps(payload)}],
        )
    except anthropic.RateLimitError:
        log.error("Anthropic rate limit — try generating the coaching report again shortly")
        return {}
    except anthropic.AuthenticationError:
        log.error("Anthropic API key invalid — check .env")
        return {}
    except anthropic.APIStatusError as exc:
        log.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        return {}
    except anthropic.APIConnectionError:
        log.error("network error reaching the Anthropic API")
        return {}

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = db.loads(text, {})
    except Exception:  # noqa: BLE001
        log.error("LLM returned unparseable JSON")
        return {}

    out: dict[str, dict] = {}
    for t in data.get("tips") or []:
        cid, tip, tone = t.get("contact_id"), t.get("tip"), t.get("tone")
        if cid and tip and tone in ("improve", "praise"):
            out[cid] = {"tip": tip, "tone": tone}
    return out


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
                  llm_fn=None, stats_summary: dict | None = None) -> CoachingReport:
    llm_fn = llm_fn or generate_llm_tips
    candidates = select_tip_candidates(vectors_meta)
    # llm_fn is the one Anthropic call for this video. Injected/mocked llm_fns
    # in tests may only accept the original (shots) signature -- fall back to
    # that rather than requiring every test double to accept stats_summary.
    try:
        tip_text = llm_fn(candidates, stats_summary)
    except TypeError:
        tip_text = llm_fn(candidates)
    # Player attribution is our own data (contact_events.player_id), not
    # trusted from the model's echoed output — attach it here regardless of
    # what the LLM response contains.
    tips = [
        CoachingTip(contact_id=c["contact_id"], frame=c["frame_idx"], timestamp_s=c["timestamp_s"],
                    category=c["category"], tip=tip_text[c["contact_id"]]["tip"],
                    tone=tip_text[c["contact_id"]]["tone"],
                    player_id=c["player_id"], player_label=c["player_label"])
        for c in candidates if c["contact_id"] in tip_text
    ]
    return CoachingReport(video_id=video_id, tips=tips)


def exclude_serves(contacts: list[dict]) -> list[dict]:
    """Serves are excluded from the coaching narrative/worst-best selection
    entirely — filtered once here, at the earliest point the contact list is
    assembled, rather than downstream in feature-vector computation or
    worst/best selection. A rally that's all serves (or a video with zero
    non-serve contacts overall) just yields fewer/no contacts here; the rest
    of the pipeline already degrades an empty contact/vector list to an empty
    tip list (see select_tip_candidates), never an error or serve fallback."""
    return [c for c in contacts if not c.get("is_serve")]


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


def generate_coaching_report(video_id: str, rally_source: str | None = None,
                              overlay_source: str | None = None) -> CoachingReport:
    """Only builds feature vectors for contacts belonging to a rally
    readiness.get_analyzable_rallies() currently clears (Pass 2 done, overlay
    coverage still matches the rally's current frame range) — a stale/partial
    rally's contacts are excluded from compute_feature_vector."""
    # rally_source can be None by default (see contact_detect.py's matching
    # fix) — resolve overlay_source the same way here so the read_range calls
    # below aren't left with an unresolved None.
    overlay_source = overlay_source or rally_source or _default_overlay_source(video_id)
    rallies = get_analyzable_rallies(video_id, rally_source=rally_source, overlay_source=overlay_source)
    ready_rally_ids = {r["id"] for r in rallies}
    with db.connect() as con:
        vrow = con.execute("SELECT fps, match_type FROM videos WHERE id=?", (video_id,)).fetchone()
        contacts = db.rows_to_dicts(con.execute(
            "SELECT * FROM contact_events WHERE video_id=? ORDER BY frame_idx",
            (video_id,)).fetchall())
    contacts = [c for c in contacts if c["rally_id"] in ready_rally_ids]
    contacts = exclude_serves(contacts)
    contacts = merge_singles_players(contacts, vrow["match_type"] if vrow else None)
    fps = float(vrow["fps"]) if vrow and vrow["fps"] else 30.0

    pre_frames, post_frames = int(PRE_WINDOW_S * fps), int(POST_WINDOW_S * fps)
    vectors_meta: list[tuple[ShotFeatureVector, dict]] = []
    for c in contacts:
        entries = overlays.read_range(video_id, overlay_source,
                                       max(0, c["frame_idx"] - pre_frames),
                                       c["frame_idx"] + post_frames)
        vec = compute_feature_vector(c, entries, fps)
        if vec is not None:
            vectors_meta.append((vec, c))

    stats_summary = None
    try:
        from . import report_stats  # lazy: report_stats imports from this module
        stats_summary = _summarize_stats(
            report_stats.get_report_stats(video_id, rally_source=rally_source, overlay_source=overlay_source))
    except Exception:  # noqa: BLE001 — stats are an add-on; never block the report
        log.warning("report_stats unavailable for %s — generating coaching report without it", video_id, exc_info=True)

    report = build_report(video_id, vectors_meta, stats_summary=stats_summary)
    save_report(report)
    return report


def save_report(report: CoachingReport) -> None:
    # stale_reason explicitly NULL: this report was just built by the
    # currently-running (confidence-aware) pipeline, so it's current by
    # definition — never inherit staleness from whatever row it replaces.
    with db.connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO coaching_reports (video_id, tips_json, created_at, stale_reason)"
            " VALUES (?,?,?,NULL)",
            (report.video_id, db.dumps([asdict(t) for t in report.tips]), db.now()))


def get_report(video_id: str) -> CoachingReport | None:
    with db.connect() as con:
        row = con.execute("SELECT * FROM coaching_reports WHERE video_id=?", (video_id,)).fetchone()
    if row is None:
        return None
    return CoachingReport(
        video_id=row["video_id"],
        tips=[CoachingTip(**t) for t in db.loads(row["tips_json"], [])],
        stale_reason=row["stale_reason"])


if __name__ == "__main__":
    # ponytail self-check: a shot with every metric mid-range must land at
    # deviation_score 0.0 and not be flagged; select_tip_candidates must cap
    # total output at MAX_TIPS even when far more shots clear the bar.
    good = ShotFeatureVector(
        contact_id="c1", category="footwork",
        knee_bend_deg=_neutral(KNEE_BEND_GOOD_RANGE_DEG),
        shoulder_rotation_deg=_neutral(SHOULDER_ROTATION_GOOD_RANGE_DEG),
        elbow_extension_deg=_neutral(ELBOW_EXTENSION_GOOD_RANGE_DEG),
        weight_transfer_score=_neutral(WEIGHT_TRANSFER_GOOD_RANGE),
        contact_height_m=0.0,
        follow_through_completion_score=_neutral(FOLLOW_THROUGH_GOOD_RANGE),
        split_step_delay_s=_neutral(SPLIT_STEP_GOOD_RANGE_S),
        deviation_score=0.0)
    cat, dev = _classify(good.knee_bend_deg, good.shoulder_rotation_deg, good.elbow_extension_deg,
                          good.weight_transfer_score, good.contact_height_m,
                          good.follow_through_completion_score, good.split_step_delay_s)
    assert dev == 0.0, dev

    bad_vectors_meta = [
        (ShotFeatureVector(contact_id=f"bad{i}", category="footwork",
                            knee_bend_deg=0.0, shoulder_rotation_deg=0.0, elbow_extension_deg=0.0,
                            weight_transfer_score=0.0, contact_height_m=0.0,
                            follow_through_completion_score=0.0, split_step_delay_s=0.0,
                            deviation_score=-5.0),
         {"contact_id": f"bad{i}", "frame_idx": i, "timestamp_s": float(i), "rally_id": "r",
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

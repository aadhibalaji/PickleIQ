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
from .contracts import CoachingReport, ShotFeatureVector
from .readiness import get_analyzable_rallies

log = logging.getLogger("rallyanalysis.coaching")

CATEGORIES = ["footwork", "paddle_prep", "contact_point", "follow_through"]

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


# ---------------------------------------------------------------------------
# Pure pose-window helpers (no DB) — unit-tested directly on synthetic overlays.
# ---------------------------------------------------------------------------

def _kp_xy(pdict: dict, idx: int) -> tuple[float, float] | None:
    kp = pdict.get("kp")
    if not kp or len(kp) <= idx:
        return None
    pt = kp[idx]
    if not pt or pt[0] <= 0 or pt[1] <= 0:
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

    Individual metrics that lack pose support fall back to the midpoint of
    their own reference range (deviation_score 0 for that metric) rather than
    crashing or fabricating a flag — see the per-field comments below.
    """
    frames = _player_frames(entries, contact["player_id"])
    if not frames:
        return None

    contact_frame = contact["frame_idx"]
    pre_frames = [fp for fp in frames if fp[0] <= contact_frame]
    post_frames = [fp for fp in frames if fp[0] >= contact_frame]
    hitting_side = _hitting_side(frames)

    knee_bend_deg = _nearest_valid(frames, contact_frame, _knee_angle)
    if knee_bend_deg is None:
        knee_bend_deg = _neutral(KNEE_BEND_GOOD_RANGE_DEG)

    shoulder_rotation_deg = _nearest_valid(frames, contact_frame, _shoulder_rotation)
    if shoulder_rotation_deg is None:
        shoulder_rotation_deg = _neutral(SHOULDER_ROTATION_GOOD_RANGE_DEG)

    elbow_extension_deg = _nearest_valid(frames, contact_frame, lambda p: _elbow_angle(p, hitting_side))
    if elbow_extension_deg is None:
        elbow_extension_deg = _neutral(ELBOW_EXTENSION_GOOD_RANGE_DEG)

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
        weight_transfer_score = _neutral(WEIGHT_TRANSFER_GOOD_RANGE)

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
    if box_height and box_height > 1e-6:
        follow_through_completion_score = max(0.0, min(1.0, path / (box_height * FOLLOW_THROUGH_NORM)))
    else:
        follow_through_completion_score = _neutral(FOLLOW_THROUGH_GOOD_RANGE)

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
        split_step_delay_s = _neutral(SPLIT_STEP_GOOD_RANGE_S)

    category, deviation_score = _classify(
        knee_bend_deg, shoulder_rotation_deg, elbow_extension_deg,
        weight_transfer_score, contact_height_m, follow_through_completion_score,
        split_step_delay_s)

    return ShotFeatureVector(
        contact_id=contact["contact_id"], category=category,
        knee_bend_deg=knee_bend_deg, shoulder_rotation_deg=shoulder_rotation_deg,
        elbow_extension_deg=elbow_extension_deg, weight_transfer_score=weight_transfer_score,
        contact_height_m=contact_height_m,
        follow_through_completion_score=follow_through_completion_score,
        split_step_delay_s=split_step_delay_s, deviation_score=deviation_score)


def _norm_dev(value: float, rng: tuple[float, float]) -> float:
    """Signed deviation from [lo, hi], expressed in units of range-width so
    metrics with different units (degrees, seconds, 0-1 scores) are
    comparable. 0 = within range."""
    lo, hi = rng
    width = hi - lo
    if value < lo:
        return (value - lo) / width
    if value > hi:
        return (value - hi) / width
    return 0.0


def _classify(knee_bend_deg, shoulder_rotation_deg, elbow_extension_deg,
              weight_transfer_score, contact_height_m, follow_through_completion_score,
              split_step_delay_s) -> tuple[str, float]:
    """category = whichever of the four categories has the single worst
    (largest-magnitude) normalized deviation among its component metrics;
    deviation_score is that category's signed deviation. A shot with every
    metric in range gets an arbitrary-but-stable category (max of all-zero
    deviations) and deviation_score 0.0 (i.e. not flagged — see
    compute_category_ratios)."""
    per_category = {
        "footwork": [_norm_dev(split_step_delay_s, SPLIT_STEP_GOOD_RANGE_S),
                     _norm_dev(knee_bend_deg, KNEE_BEND_GOOD_RANGE_DEG)],
        "paddle_prep": [_norm_dev(shoulder_rotation_deg, SHOULDER_ROTATION_GOOD_RANGE_DEG)],
        "contact_point": [_norm_dev(elbow_extension_deg, ELBOW_EXTENSION_GOOD_RANGE_DEG),
                           _norm_dev(weight_transfer_score, WEIGHT_TRANSFER_GOOD_RANGE),
                           _norm_dev(contact_height_m, CONTACT_HEIGHT_GOOD_RANGE_M)],
        "follow_through": [_norm_dev(follow_through_completion_score, FOLLOW_THROUGH_GOOD_RANGE)],
    }
    category_dev = {cat: max(devs, key=abs) for cat, devs in per_category.items()}
    category = max(category_dev, key=lambda c: abs(category_dev[c]))
    return category, category_dev[category]


# ---------------------------------------------------------------------------
# Aggregation + adaptive worst/best selection — pure, unit-tested directly.
# ---------------------------------------------------------------------------

def compute_category_ratios(vectors: list[ShotFeatureVector], active_rally_seconds: float) -> dict:
    flagged = {cat: 0 for cat in CATEGORIES}
    for vec in vectors:
        if vec.deviation_score != 0.0:
            flagged[vec.category] += 1
    return {
        cat: {
            "flagged_count": flagged[cat],
            "active_rally_seconds": active_rally_seconds,
            "ratio": (flagged[cat] / active_rally_seconds) if active_rally_seconds > 0 else 0.0,
        }
        for cat in CATEGORIES
    }


def _shot_summary(vec: ShotFeatureVector, contact: dict) -> dict:
    return {
        "contact_id": vec.contact_id,
        "category": vec.category,
        "deviation_score": vec.deviation_score,
        "timestamp_s": contact.get("timestamp_s"),
        "rally_id": contact.get("rally_id"),
        "player_id": contact.get("player_id"),
        "team": contact.get("team"),
        "shot_side": contact.get("shot_side"),
        "is_serve": bool(contact.get("is_serve")),
        "confidence": contact.get("confidence"),
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


# ---------------------------------------------------------------------------
# Anthropic API call — same client/schema/degrade pattern as
# analysis/reporter.py's generate_llm_report. Called at most once per video.
# ---------------------------------------------------------------------------

COACHING_SCHEMA = {
    "type": "object",
    "properties": {
        "narrative": {
            "type": "string",
            "description": "Coaching narrative in markdown: specific, actionable advice "
                           "per flagged category (grounded in the worst_shots evidence), "
                           "and explicit reinforcement of what's already working, grounded "
                           "in the best_shots evidence.",
        },
    },
    "required": ["narrative"],
    "additionalProperties": False,
}

SYSTEM = """You are a pickleball coach reviewing one player's session, based on
pose-derived shot mechanics (not video) that a computer-vision pipeline
extracted at each detected contact/swing.

You receive: category_ratios (flagged shots per minute of active rally play,
per category — footwork, paddle_prep, contact_point, follow_through),
worst_shots (the shots whose mechanics deviated most from reference ranges,
with full feature vectors and rally context), and best_shots (in-range,
high-confidence shots showing good mechanics).

Give specific, actionable coaching advice for each category that shows
meaningful flagged-ratio, citing the concrete metric(s) that were off (e.g.
"knee_bend_deg was X, outside the Y-Z reference range") rather than generic
tips. Then explicitly call out what's working well, grounded in the
best_shots data, so the player knows what to keep doing. Keep it concrete and
encouraging, not a wall of hedging — but do not overstate certainty: this is
computer-vision-derived pose data, not a substitute for in-person coaching."""


def generate_llm_narrative(payload: dict) -> str | None:
    """Returns the narrative markdown, or None (with a logged reason) when
    the API key is missing or the call fails — degrades gracefully, same as
    analysis/reporter.py."""
    if not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set — skipping coaching narrative")
        return None
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = client.messages.create(
            model=ANTHROPIC_REPORT_MODEL,
            max_tokens=4000,
            thinking={"type": "enabled", "budget_tokens": 1500},
            system=SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": COACHING_SCHEMA}},
            messages=[{"role": "user", "content": db.dumps(payload)}],
        )
    except anthropic.RateLimitError:
        log.error("Anthropic rate limit — try generating the coaching report again shortly")
        return None
    except anthropic.AuthenticationError:
        log.error("Anthropic API key invalid — check .env")
        return None
    except anthropic.APIStatusError as exc:
        log.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        return None
    except anthropic.APIConnectionError:
        log.error("network error reaching the Anthropic API")
        return None

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = db.loads(text, {})
    except Exception:  # noqa: BLE001
        log.error("LLM returned unparseable JSON")
        return None
    return data.get("narrative")


# ---------------------------------------------------------------------------
# Orchestration — the only DB-touching layer. build_report is the pure core
# (video_id/vectors_meta -> CoachingReport, LLM call injectable) so tests can
# exercise the ratio math and adaptive selection without a DB or anthropic.
# ---------------------------------------------------------------------------

def build_report(video_id: str, vectors_meta: list[tuple[ShotFeatureVector, dict]],
                  active_rally_seconds: float, llm_fn=None) -> CoachingReport:
    llm_fn = llm_fn or generate_llm_narrative
    vectors = [vec for vec, _ in vectors_meta]
    ratios = compute_category_ratios(vectors, active_rally_seconds)
    worst, best = select_worst_best(vectors_meta)
    payload = {"category_ratios": ratios, "worst_shots": worst, "best_shots": best}
    narrative = llm_fn(payload)  # the one Anthropic call for this video
    return CoachingReport(video_id=video_id, category_ratios=ratios,
                           worst_shots=worst, best_shots=best, narrative=narrative or "")


def generate_coaching_report(video_id: str, rally_source: str = "pass2",
                              overlay_source: str | None = None) -> CoachingReport:
    """Only builds feature vectors for contacts belonging to a rally
    readiness.get_analyzable_rallies() currently clears (Pass 2 done, overlay
    coverage still matches the rally's current frame range) — a stale/partial
    rally is excluded from both the active-rally-seconds denominator and the
    contacts fed into compute_feature_vector."""
    overlay_source = overlay_source or rally_source
    rallies = get_analyzable_rallies(video_id, rally_source=rally_source, overlay_source=overlay_source)
    ready_rally_ids = {r["id"] for r in rallies}
    with db.connect() as con:
        vrow = con.execute("SELECT fps FROM videos WHERE id=?", (video_id,)).fetchone()
        contacts = db.rows_to_dicts(con.execute(
            "SELECT * FROM contact_events WHERE video_id=? ORDER BY frame_idx",
            (video_id,)).fetchall())
    contacts = [c for c in contacts if c["rally_id"] in ready_rally_ids]
    fps = float(vrow["fps"]) if vrow and vrow["fps"] else 30.0
    active_rally_seconds = sum((r["end_frame"] - r["start_frame"]) / fps for r in rallies)

    pre_frames, post_frames = int(PRE_WINDOW_S * fps), int(POST_WINDOW_S * fps)
    vectors_meta: list[tuple[ShotFeatureVector, dict]] = []
    for c in contacts:
        entries = overlays.read_range(video_id, overlay_source,
                                       max(0, c["frame_idx"] - pre_frames),
                                       c["frame_idx"] + post_frames)
        vec = compute_feature_vector(c, entries, fps)
        if vec is not None:
            vectors_meta.append((vec, c))

    report = build_report(video_id, vectors_meta, active_rally_seconds)
    save_report(report)
    return report


def save_report(report: CoachingReport) -> None:
    with db.connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO coaching_reports"
            " (video_id, category_ratios, worst_shots, best_shots, narrative, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (report.video_id, db.dumps(report.category_ratios), db.dumps(report.worst_shots),
             db.dumps(report.best_shots), report.narrative, db.now()))


def get_report(video_id: str) -> CoachingReport | None:
    with db.connect() as con:
        row = con.execute("SELECT * FROM coaching_reports WHERE video_id=?", (video_id,)).fetchone()
    if row is None:
        return None
    return CoachingReport(
        video_id=row["video_id"],
        category_ratios=db.loads(row["category_ratios"], {}),
        worst_shots=db.loads(row["worst_shots"], []),
        best_shots=db.loads(row["best_shots"], []),
        narrative=row["narrative"])


if __name__ == "__main__":
    # ponytail self-check: a shot with every metric mid-range must land at
    # deviation_score 0.0 and not be flagged; ratio math must divide by
    # active_rally_seconds, not shot count.
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
    ratios = compute_category_ratios([good], active_rally_seconds=10.0)
    assert all(r["flagged_count"] == 0 for r in ratios.values())
    print("coaching self-check OK")

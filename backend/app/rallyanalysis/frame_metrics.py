"""Continuous frame-by-frame metric tracking — the primary time-specific
detection mechanism (2026-08-05 redesign, replacing the old per-contact
Tukey-fence pipeline as the operative one — see coaching.py's
_build_vectors_meta). Rather than scoring metrics only in a narrow window
around each detected swing, metrics with a genuine per-frame physical
meaning are computed on EVERY overlay frame of a rally. Each frame's
per-metric deviations from an ABSOLUTE reference target (not a personal
baseline — a continuous stream doesn't have an obvious personal-baseline
unit the way "one shot" does; these targets are the same ones already
validated as coaching.REFERENCE_BANDS / positioning.BANDS) are summed into
one per-frame signal; a run of consecutive frames where that sum crosses
the threshold is a SPIKE — the frame-level equivalent of the old per-shot
"flagged" moment, represented downstream as a "span" pseudo-contact exactly
like positioning.py's span moments (same mechanism, same keyframe/LLM
handling — see coaching._shot_summary's kind=="span" branch).

Not every metric has a defensible per-frame absolute target — this is a
real, documented limitation, not an oversight:
  - form: knee_bend_deg is a pure single-frame pose reading with an existing
    validated target (coaching.REFERENCE_BANDS) and is scored CONTINUOUSLY
    — "stay bent and athletic" is good advice at any moment, mid-swing or
    just standing waiting for the next ball.
    elbow_extension_deg is NOT continuous (2026-08-06 revision) — "reach out
    and extend" is only a real coaching standard AT THE MOMENT OF CONTACT; a
    bent elbow between shots is correct ready position (especially at the
    net), not a flaw. Scoring it every frame — as this module originally
    did, taking whichever arm read more cramped that instant — meant a
    normal compact ready stance, held for most of a rally, constantly
    misread as "jammed," which is why the contact_point category flagged
    far more than footwork ever did on real sessions. It is now
    CONTACT-ANCHORED: evaluated only at each player's own real detected
    contact events, using the ACTUAL hitting arm (determined from wrist
    motion in a small window around that contact — see _hitting_side), not
    a worse-of-both-arms guess a random mid-rally frame has no way to
    resolve. See _contact_elbow_readings.
    shoulder_rotation_deg has no defensible universal minimum (a dink
    legitimately keeps the shoulders square) so it is NOT scored here.
    weight_transfer_score / follow_through_completion_score /
    split_step_delay_s are inherently defined RELATIVE TO A SWING (movement
    into/out of a specific contact) and have no meaningful value between
    swings, so they stay out of the continuous model entirely.
  - positioning: kitchen_distance_m, transition_zone_occupancy,
    lateral_offset_m, and partner_depth_stagger_m all have a genuine
    per-frame value and an existing absolute target (positioning.BANDS).
    return_depth_m is meaningful only exactly at the return of serve and is
    dropped for the same reason as the form swing-relative metrics above;
    kitchen_approach_after_return_s' intent is now implicitly captured by
    SUSTAINED kitchen_distance_m spikes during the same rally phase (once a
    rally is "established" — see established_from_frame), so it is not
    tracked as a separate metric.

Pure module (no DB, no cv2) — mirrors positioning.py's pure/orchestration
split; imports only contracts/shot_context/cv.court/positioning (never
coaching.py, keeping coaching.py the sole top-level orchestrator — same
import direction positioning.py/ball_context.py already use).
"""
from __future__ import annotations

import math

from ..cv.court import COURT_W, KITCHEN, NET_Y
from .positioning import (CENTER_X, MIN_SHOT_INDEX_FOR_KITCHEN, SAMPLE_GAP_BREAK_S,
                           _abs_dev, _own_kitchen_distance)
from .shot_context import ZONE_MIDCOURT, zone_for_court_y

# COCO-17 keypoint indices (same convention as coaching.py/contact_detect.py).
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

KP_CONF_MIN = 0.5  # mirrors coaching.KP_CONF_MIN


def _kp_xy(pdict: dict, idx: int) -> tuple[float, float] | None:
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


def _elbow_angle_side(pdict: dict, side: str) -> float | None:
    shoulder_i, elbow_i, wrist_i = (L_SHOULDER, L_ELBOW, L_WRIST) if side == "L" else (R_SHOULDER, R_ELBOW, R_WRIST)
    shoulder, elbow, wrist = _kp_xy(pdict, shoulder_i), _kp_xy(pdict, elbow_i), _kp_xy(pdict, wrist_i)
    if shoulder and elbow and wrist:
        return _angle_deg(shoulder, elbow, wrist)
    return None


def _hitting_side(frames_by_index: dict[int, dict]) -> str:
    """'L' or 'R' — whichever wrist moved more across the given frames (a
    small window around one contact). Same "faster wrist = hitting arm"
    heuristic contact_detect.py already uses for contact DETECTION;
    duplicated here rather than imported from coaching.py, which this
    module must not depend on (see the module docstring)."""
    totals = {"L": 0.0, "R": 0.0}
    prev = {"L": None, "R": None}
    for f in sorted(frames_by_index):
        pdict = frames_by_index[f]
        for side, idx in (("L", L_WRIST), ("R", R_WRIST)):
            pt = _kp_xy(pdict, idx)
            if pt and prev[side]:
                totals[side] += math.hypot(pt[0] - prev[side][0], pt[1] - prev[side][1])
            if pt:
                prev[side] = pt
    return "L" if totals["L"] > totals["R"] else "R"


# ---------------------------------------------------------------------------
# Absolute per-frame targets — (lo, hi, scale), 0 inside the band, signed
# excess/scale outside (via positioning._abs_dev). Reused verbatim from
# coaching.REFERENCE_BANDS / positioning.BANDS wherever a target already
# existed and was validated there; see the module docstring for exclusions.
# ---------------------------------------------------------------------------
FORM_TARGETS = {
    "knee_bend_deg": (0.0, 168.0, 6.0),
    "elbow_extension_deg": (65.0, 180.0, 15.0),
}
POSITIONING_TARGETS = {
    "kitchen_distance_m": (-KITCHEN, 2.0, 1.0),
    "transition_zone_occupancy": (0.0, 0.0, 1.0),
    "lateral_offset_m": (0.0, 1.5, 0.75),
    "partner_depth_stagger_m": (0.0, 2.5, 1.0),
}
_SINGLES_POSITIONING_OVERRIDES = {"kitchen_distance_m": (-KITCHEN, 3.0, 1.0)}

# metric -> coaching category, reusing the EXACT taxonomy already shipped
# (coaching.CATEGORY_METRICS / positioning.CATEGORY_METRICS) so the LLM
# prompt/UI need no new category vocabulary.
FORM_CATEGORY = {"knee_bend_deg": "footwork", "elbow_extension_deg": "contact_point"}
POSITIONING_CATEGORY = {
    "kitchen_distance_m": "net_position", "transition_zone_occupancy": "transition",
    "lateral_offset_m": "court_coverage", "partner_depth_stagger_m": "team_spacing",
}

# ---------------------------------------------------------------------------
# Per-metric flag thresholds (2026-08-05 revision) — how many `scale` units a
# metric's OWN deviation must cross, INDEPENDENTLY, before a frame counts as
# "hot" for it. Replaces the original design where a shared bar was compared
# against the SUM of every metric's deviation: that let several only-
# moderately-off metrics add up to a flag with no single metric actually
# being bad enough to justify one, diluting "why was this flagged" into
# something too vague to coach. Detection is now per-metric so a spike
# always has ONE clear, nameable cause, and each metric can be tuned
# independently as real coaching feedback on these values comes in.
#
# Deliberately set above coaching.WORST_DEVIATION_THRESHOLD (1.0, still used
# as the severity-language floor and by select_worst_best) — keep every
# value here >= that floor, or a produced spike's deviation_score could stop
# reading as "flagged" downstream.
#
# 2026-08-05: calibrated from the 90th percentile of the real |deviation|
# distribution across every pass2-done clip in the DB at the time (12 clips,
# see calibrate_thresholds.py — re-run it as more clips accumulate and
# revisit these). That run caught two real problems with the original flat
# 1.5-everywhere guess, not just noise-trimming:
#   - kitchen_distance_m at 1.5 flagged 48% of ALL established-phase frames
#     (p90 of the real distribution is 4.38) — explains why net_position
#     alone dominated every positioning report before this fix.
#   - transition_zone_occupancy's deviation can only ever be exactly 0.0 or
#     1.0 (see positioning_frame_values) — a threshold of 1.5 was
#     UNREACHABLE, so this metric never fired a single spike; fixed to 1.0.
# elbow_extension_deg's calibrated 1.0 was measured against the OLD
# continuous (every-frame, worse-of-both-arms) population and is now STALE:
# 2026-08-06 moved elbow to contact-anchored scoring (see
# _contact_elbow_readings), a completely different population (one reading
# per real contact, correct hitting arm only) — re-run calibrate_thresholds.py
# for this metric once enough contact-anchored data exists; 1.0 is a
# placeholder in the meantime, not a recalibrated value.
# ---------------------------------------------------------------------------
FORM_METRIC_THRESHOLDS = {"knee_bend_deg": 1.5, "elbow_extension_deg": 1.0}

# Window (each side, seconds) used to determine which arm is doing the
# hitting at one contact — see _hitting_side / _contact_elbow_readings.
ELBOW_HITTING_SIDE_WINDOW_S = 0.3
POSITIONING_METRIC_THRESHOLDS = {
    "kitchen_distance_m": 4.5, "transition_zone_occupancy": 1.0,
    "lateral_offset_m": 2.0, "partner_depth_stagger_m": 2.2,
}

GAP_BREAK_S = SAMPLE_GAP_BREAK_S  # unseen time never bridges a spike run


def targets_and_category(focus: str) -> tuple[dict, dict]:
    if focus == "positioning":
        return POSITIONING_TARGETS, POSITIONING_CATEGORY
    return FORM_TARGETS, FORM_CATEGORY


def metric_thresholds(focus: str) -> dict:
    return POSITIONING_METRIC_THRESHOLDS if focus == "positioning" else FORM_METRIC_THRESHOLDS


def _band_for(metric: str, targets: dict, match_type: str | None) -> tuple[float, float, float]:
    if targets is POSITIONING_TARGETS and match_type == "singles" and metric in _SINGLES_POSITIONING_OVERRIDES:
        return _SINGLES_POSITIONING_OVERRIDES[metric]
    return targets[metric]


def established_from_frame(contacts: list[dict]) -> int | None:
    """The frame at which this rally reaches MIN_SHOT_INDEX_FOR_KITCHEN
    (serve/return/3rd/4th shot done) — None if the rally never gets that
    far, in which case kitchen_distance_m/transition_zone_occupancy are
    never scored for it (mirrors positioning's per-contact gate: short
    rallies never had these judged either)."""
    for c in sorted(contacts, key=lambda c: c["frame_idx"]):
        if c.get("rally_shot_index") == MIN_SHOT_INDEX_FOR_KITCHEN:
            return c["frame_idx"]
    return None


def form_frame_values(pdict: dict) -> dict:
    # elbow_extension_deg is intentionally NEVER set here — it's contact-
    # anchored now (see _contact_elbow_readings), injected by
    # build_frame_series only at real contact frames, not computed per
    # frame. Always None by default, never a fabricated worse-of-both guess.
    return {"knee_bend_deg": _knee_angle(pdict), "elbow_extension_deg": None}


def _contact_elbow_readings(entries: list[dict], contacts: list[dict], fps: float) -> dict:
    """{(player_id, frame_idx): elbow_deg} — one reading per REAL contact
    event (contacts here is already serve-excluded, singles-merged — the
    same list _build_vectors_meta scopes to this rally), at that contact's
    OWN frame, using whichever arm's wrist moved more in a small window
    around it — the genuine hitting arm, not a worse-of-both guess. A
    contact with no usable pose at its own instant is simply absent, never
    fabricated."""
    by_frame = {e["f"]: e for e in entries}
    window_frames = max(1, round(ELBOW_HITTING_SIDE_WINDOW_S * fps))
    out: dict = {}
    for c in contacts:
        pid, cf = c.get("player_id"), c.get("frame_idx")
        if pid is None or cf is None:
            continue
        frames_by_index = {}
        for f in range(cf - window_frames, cf + window_frames + 1):
            e = by_frame.get(f)
            if not e:
                continue
            for p in e.get("players", []):
                if p.get("id") == pid:
                    frames_by_index[f] = p
                    break
        if cf not in frames_by_index:
            continue  # no pose at the contact instant itself -- nothing to read
        side = _hitting_side(frames_by_index)
        angle = _elbow_angle_side(frames_by_index[cf], side)
        if angle is not None:
            out[(pid, cf)] = angle
    return out


def positioning_frame_values(cx: float, cy: float, team: str | None, established: bool,
                              partner_cy: float | None, match_type: str | None) -> dict:
    kitchen = _own_kitchen_distance(cy, team) if team in ("near", "far") else None
    midcourt = 1.0 if zone_for_court_y(cy) == ZONE_MIDCOURT else 0.0
    return {
        "kitchen_distance_m": kitchen if (established and kitchen is not None) else None,
        "transition_zone_occupancy": midcourt if established else None,
        "lateral_offset_m": abs(cx - CENTER_X) if match_type == "singles" else None,
        "partner_depth_stagger_m": abs(cy - partner_cy) if partner_cy is not None else None,
    }


def frame_deviations(values: dict, targets: dict, match_type: str | None) -> dict:
    return {m: _abs_dev(values.get(m), _band_for(m, targets, match_type)) for m in targets}


def frame_sum(devs: dict) -> float:
    """Sum of |deviation| across every metric with a real reading this
    frame — kept as a DISPLAY/diagnostic figure ("how far off, combined,
    at this instant") but no longer the detection criterion; see
    frame_max_excess for what actually decides a spike now."""
    return sum(abs(d) for d in devs.values() if d is not None)


def frame_max_excess(devs: dict, thresholds: dict) -> float:
    """The largest (|deviation| / that metric's OWN threshold) across every
    metric with both a real reading and a defined threshold this frame — 0
    when nothing has a reading. 1.0 exactly at the point some metric first
    clears its own bar; >1.0 once it has. This, not frame_sum, is the
    per-frame decision quantity detect_spikes groups into spikes, and the
    number the frame-by-frame chart's threshold line is drawn against."""
    ratios = [abs(d) / thresholds[m] for m, d in devs.items()
              if d is not None and thresholds.get(m)]
    return max(ratios) if ratios else 0.0


def _metric_excess(devs: dict, thresholds: dict) -> dict:
    """{metric: |deviation|/threshold} for metrics that INDIVIDUALLY clear
    their own bar this frame — empty when nothing does (frame is not hot).
    Distinct from frame_max_excess (a single number, always defined,
    continuous below 1.0 too): this is the discrete "which metrics fired"
    view detect_spikes needs to pick a dominant metric."""
    return {m: abs(d) / thresholds[m] for m, d in devs.items()
            if d is not None and thresholds.get(m) and abs(d) >= thresholds[m]}


def build_frame_series(entries: list[dict], contacts: list[dict], fps: float,
                        match_type: str | None, focus: str,
                        exclude_frame_ranges: list[tuple[int, int]] = ()) -> dict[int, list[dict]]:
    """{player_id: [{frame, values, devs}, ...]} for one rally's overlay
    entries, in frame order — the continuous series build_spikes and
    average_context both consume. exclude_frame_ranges (inclusive
    [start,end] pairs, e.g. serve swing windows) are skipped entirely —
    those frames never enter the series at all, so they can neither trigger
    a spike nor pull a session average toward serve technique. `contacts`
    (already serve-excluded, singles-merged, scoped to this rally) drives
    positioning's established-phase gate AND, for form, exactly which
    frames get an elbow_extension_deg reading at all (see
    _contact_elbow_readings — contact-anchored, not continuous)."""
    targets, _cat = targets_and_category(focus)
    est_frame = established_from_frame(contacts) if focus == "positioning" else None
    elbow_readings = _contact_elbow_readings(entries, contacts, fps) if focus == "form" else {}

    out: dict[int, list[dict]] = {}
    for e in entries:
        f = e["f"]
        if any(lo <= f <= hi for lo, hi in exclude_frame_ranges):
            continue
        players = e.get("players", [])
        by_pid = {p.get("id"): p for p in players}
        for pid, p in by_pid.items():
            if focus == "positioning":
                cxy = p.get("cxy")
                if not cxy:
                    continue
                team = p.get("team")
                partner_cy = None
                if match_type == "doubles" and team:
                    for other_pid, other in by_pid.items():
                        if other_pid != pid and other.get("team") == team and other.get("cxy"):
                            partner_cy = other["cxy"][1]
                            break
                established = est_frame is not None and f >= est_frame
                values = positioning_frame_values(cxy[0], cxy[1], team, established,
                                                   partner_cy, match_type)
            else:
                values = form_frame_values(p)
                if (pid, f) in elbow_readings:
                    values = {**values, "elbow_extension_deg": elbow_readings[(pid, f)]}
            devs = frame_deviations(values, targets, match_type)
            out.setdefault(pid, []).append({"frame": f, "values": values, "devs": devs})
    return out


def detect_spikes(frame_series: list[dict], thresholds: dict, fps: float,
                   category_map: dict) -> list[dict]:
    """Groups consecutive frames (from ONE player's build_frame_series
    output for ONE rally) where at least one metric INDEPENDENTLY clears
    ITS OWN threshold into spikes — the frame-level equivalent of a
    "flagged" moment. A frame with two metrics each only moderately off
    (neither individually over its own bar) is NOT hot, even if their
    combined magnitude would have crossed the old shared-sum bar — see the
    module-level comment above FORM_METRIC_THRESHOLDS for why. Gaps up to
    GAP_BREAK_S bridge (matches positioning._runs' unseen-time-never-
    fabricated rule); anything longer starts a new spike.

    thresholds: {metric: own_threshold} (see metric_thresholds). One dict
    per spike: {start_frame, end_frame, peak_frame, peak_sum, duration_s,
    category, dominant_metric, devs_at_peak, values_at_peak} — peak_sum is
    frame_sum at the peak frame, kept for display only (not the trigger);
    the peak frame and dominant metric are chosen by frame_max_excess/
    _metric_excess (independent per-metric detection)."""
    max_gap = GAP_BREAK_S * fps
    runs: list[list[dict]] = []
    current: list[dict] = []
    for fs in frame_series:
        excess = _metric_excess(fs["devs"], thresholds)
        if excess:
            if current and fs["frame"] - current[-1]["frame"] > max_gap:
                runs.append(current)
                current = []
            current.append({**fs, "excess": excess})
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)

    spikes = []
    for run in runs:
        peak = max(run, key=lambda r: max(r["excess"].values()))
        dominant_metric = max(peak["excess"], key=lambda m: peak["excess"][m])
        spikes.append({
            "start_frame": run[0]["frame"], "end_frame": run[-1]["frame"],
            "peak_frame": peak["frame"], "peak_sum": frame_sum(peak["devs"]),
            "duration_s": (run[-1]["frame"] - run[0]["frame"]) / fps if fps > 0 else 0.0,
            "category": category_map[dominant_metric], "dominant_metric": dominant_metric,
            "devs_at_peak": peak["devs"], "values_at_peak": peak["values"],
        })
    return spikes


def average_context(frame_series: list[dict], focus: str, match_type: str | None) -> dict:
    """{category: {avg_value, deviation, flagged_fraction,
    flagged_avg_deviation, impact_score, n_frames, metric}} for ONE
    player's ENTIRE session (concatenate frame_series across every rally
    before calling this) — the aggregate behind general feedback.

    avg_value/deviation are the plain session-wide average and its
    band-deviation — kept for display, but a metric that's genuinely
    flagged only a fraction of the time gets diluted back toward the
    acceptable range by all the normal frames surrounding it (verified on
    a real clip: 300 detected spikes, EVERY category's plain deviation
    still read 0.00 — the average alone is too weak a signal to pick a
    category from).

    impact_score = flagged_fraction * |flagged_avg_deviation| is the fix:
    flagged_fraction (0-1) is how much of the session this metric spent
    over ITS OWN threshold (metric_thresholds — the same bar detect_spikes
    uses); flagged_avg_deviation is the average band-deviation across ONLY
    those flagged frames (how bad it is when it happens, not diluted by
    the calm majority). A metric that's rarely flagged but severe when it
    is, and one that's flagged often but mildly, can both score
    meaningfully above zero — impact_score is what strongest_negative/
    positive_category are chosen by, not the plain average. Categories
    with zero valid readings are simply absent — never zero-filled."""
    targets, category_map = targets_and_category(focus)
    thresholds = metric_thresholds(focus)
    by_metric_values: dict[str, list[float]] = {}
    by_metric_flagged_devs: dict[str, list[float]] = {}
    for fs in frame_series:
        for m, v in fs["values"].items():
            if v is not None:
                by_metric_values.setdefault(m, []).append(v)
        for m, d in fs["devs"].items():
            if d is not None and thresholds.get(m) and abs(d) >= thresholds[m]:
                by_metric_flagged_devs.setdefault(m, []).append(d)

    by_category: dict = {}
    for metric, vals in by_metric_values.items():
        avg = sum(vals) / len(vals)
        dev = _abs_dev(avg, _band_for(metric, targets, match_type))
        flagged = by_metric_flagged_devs.get(metric, [])
        flagged_fraction = len(flagged) / len(vals)
        flagged_avg_dev = sum(flagged) / len(flagged) if flagged else None
        impact = flagged_fraction * abs(flagged_avg_dev) if flagged_avg_dev is not None else 0.0
        cat = category_map[metric]
        entry = {"avg_value": avg, "deviation": dev, "flagged_fraction": flagged_fraction,
                  "flagged_avg_deviation": flagged_avg_dev, "impact_score": impact,
                  "n_frames": len(vals), "metric": metric}
        cur = by_category.get(cat)
        if cur is None or impact > cur["impact_score"]:
            by_category[cat] = entry
    return by_category


if __name__ == "__main__":
    # ponytail self-check: target math, independent per-metric spike
    # grouping (incl. gap-breaking and the multi-metric-pileup fix),
    # dominant-metric attribution, average-vs-target with impact_score. No DB.
    assert _abs_dev(0.0, FORM_TARGETS["knee_bend_deg"]) == 0.0
    assert _abs_dev(180.0, FORM_TARGETS["knee_bend_deg"]) == 2.0  # (180-168)/6

    # 5 consecutive frames where knee alone clears ITS OWN bar -> one spike.
    series = [{"frame": f, "values": {"knee_bend_deg": 180.0, "elbow_extension_deg": None},
               "devs": {"knee_bend_deg": 2.0, "elbow_extension_deg": None}} for f in range(100, 105)]
    spikes = detect_spikes(series, FORM_METRIC_THRESHOLDS, fps=30.0, category_map=FORM_CATEGORY)
    assert len(spikes) == 1
    assert spikes[0]["start_frame"] == 100 and spikes[0]["end_frame"] == 104
    assert spikes[0]["dominant_metric"] == "knee_bend_deg"
    assert spikes[0]["category"] == "footwork"

    # THE FIX: two metrics each individually under their own threshold, whose
    # SUM would have crossed the old shared bar, must NOT spike anymore.
    knee_dev, elbow_dev = 0.9, 0.9  # both < FORM_METRIC_THRESHOLDS (1.5), sum = 1.8
    assert knee_dev < FORM_METRIC_THRESHOLDS["knee_bend_deg"]
    assert elbow_dev < FORM_METRIC_THRESHOLDS["elbow_extension_deg"]
    moderate = [{"frame": f, "values": {"knee_bend_deg": 173.4, "elbow_extension_deg": 51.5},
                 "devs": {"knee_bend_deg": knee_dev, "elbow_extension_deg": -elbow_dev}}
                for f in range(0, 5)]
    assert detect_spikes(moderate, FORM_METRIC_THRESHOLDS, fps=30.0, category_map=FORM_CATEGORY) == []

    # A gap longer than GAP_BREAK_S splits one run into two spikes.
    gap_frames = int(GAP_BREAK_S * 30.0) + 5
    series2 = series + [{"frame": 104 + gap_frames, "values": {"knee_bend_deg": 180.0, "elbow_extension_deg": None},
                          "devs": {"knee_bend_deg": 2.0, "elbow_extension_deg": None}}]
    spikes2 = detect_spikes(series2, FORM_METRIC_THRESHOLDS, fps=30.0, category_map=FORM_CATEGORY)
    assert len(spikes2) == 2

    # Below-threshold frames never spike.
    clean = [{"frame": f, "values": {"knee_bend_deg": 130.0, "elbow_extension_deg": None},
              "devs": {"knee_bend_deg": 0.0, "elbow_extension_deg": None}} for f in range(0, 10)]
    assert detect_spikes(clean, FORM_METRIC_THRESHOLDS, fps=30.0, category_map=FORM_CATEGORY) == []

    # Average-vs-target: a player averaging 180 on knee EVERY frame (always
    # locked straight) is the easy case -- both the plain average and the
    # flagged-only view agree it's bad.
    ctx = average_context(series, focus="form", match_type=None)
    assert ctx["footwork"]["avg_value"] == 180.0
    assert ctx["footwork"]["deviation"] == 2.0
    assert ctx["footwork"]["flagged_fraction"] == 1.0
    assert ctx["footwork"]["impact_score"] > 0
    assert ctx["footwork"]["n_frames"] == 5

    # THE DILUTION CASE: a metric flagged only 10% of the session (locked
    # straight 10 frames) buried in 90 normal frames -- the plain average
    # stays comfortably in-band (near-zero deviation), but impact_score must
    # still surface it as a real problem via flagged_fraction/flagged_avg.
    diluted = ([{"frame": f, "values": {"knee_bend_deg": 130.0, "elbow_extension_deg": None},
                 "devs": {"knee_bend_deg": 0.0, "elbow_extension_deg": None}} for f in range(90)]
               + [{"frame": f, "values": {"knee_bend_deg": 180.0, "elbow_extension_deg": None},
                   "devs": {"knee_bend_deg": 2.0, "elbow_extension_deg": None}} for f in range(90, 100)])
    diluted_ctx = average_context(diluted, focus="form", match_type=None)
    assert diluted_ctx["footwork"]["deviation"] == 0.0            # plain average: looks fine
    assert diluted_ctx["footwork"]["flagged_fraction"] == 0.10    # impact view: 10% of the time...
    assert diluted_ctx["footwork"]["flagged_avg_deviation"] == 2.0  # ...and clearly bad when it happens
    assert diluted_ctx["footwork"]["impact_score"] > 0.15         # genuinely nonzero, unlike plain deviation

    # Positioning: established-phase gating, singles-only lateral offset.
    assert established_from_frame([{"frame_idx": 10, "rally_shot_index": 5}]) == 10
    assert established_from_frame([{"frame_idx": 10, "rally_shot_index": 2}]) is None
    vals = positioning_frame_values(cx=CENTER_X + 2.0, cy=NET_Y + 5.0, team="near",
                                     established=True, partner_cy=None, match_type="doubles")
    assert vals["lateral_offset_m"] is None  # doubles: no solo-centering target
    vals_s = positioning_frame_values(cx=CENTER_X + 2.0, cy=NET_Y + 5.0, team="near",
                                       established=True, partner_cy=None, match_type="singles")
    assert vals_s["lateral_offset_m"] == 2.0
    print("frame_metrics self-check OK")

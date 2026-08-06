"""Positioning-focus coaching metrics: where players stand and move on the
court, judged against ABSOLUTE court-geometry bands — the court is a fixed
size (cv/court.py), so unlike body mechanics there IS a validated external
standard to compare against (the per-player-relative rationale at the top of
coaching.py deliberately does not apply here).

Pure module (no DB, no cv2, no LLM) — mirrors coaching.py's pure/orchestration
split. Inputs are contact_events dicts (already rally-annotated, serve-
excluded, singles-merged by generate_coaching_report) and per-rally position
samples (shot_context.position_samples over overlay cxy — court metres from
the homography, available every pass2 frame, independent of keypoint
confidence and ball tracking).

Two kinds of evaluated moment, both flowing through the SAME selection /
salience / keyframes / LLM pipeline as form shots (see coaching.build_report):
  - contact-anchored: metrics computed on a real contact event (its dict is
    reused as the meta/anchor).
  - span moments: synthetic pseudo-contacts for problems that live BETWEEN
    contacts (dwelling in no-man's-land, failing to come in behind a return).
    contact_id "pos_<rally>_<pid>_<metric>_<frame>" (TEXT everywhere
    downstream, so tips / tip_feedback / UI seek work unchanged), frame_idx /
    timestamp_s at the most instructive moment of the span, kind="span" and a
    "span" payload dict so keyframes.py and the LLM prompt can treat them as
    stretches of movement rather than swings.
"""
from __future__ import annotations

from ..cv.court import COURT_L, COURT_W, KITCHEN, NET_Y
from .contracts import PositioningFeatureVector
from .shot_context import ZONE_MIDCOURT, zone_for_court_y

CENTER_X = COURT_W / 2.0

METRIC_FIELDS = (
    "kitchen_distance_at_contact_m", "return_depth_m", "transition_dwell_s",
    "kitchen_approach_after_return_s", "recovery_offset_m", "partner_depth_stagger_m",
)

# category -> the PositioningFeatureVector metric fields that drive it — the
# positioning counterpart of coaching.CATEGORY_METRICS, consumed by the same
# (parameterized) selection functions.
CATEGORY_METRICS = {
    "net_position": ("kitchen_distance_at_contact_m", "kitchen_approach_after_return_s"),
    "transition": ("transition_dwell_s",),
    "court_coverage": ("recovery_offset_m", "return_depth_m"),
    "team_spacing": ("partner_depth_stagger_m",),
}

# ---------------------------------------------------------------------------
# Absolute bands: (lo, hi, scale). A value inside [lo, hi] is fine (dev 0);
# outside, deviation = signed excess / scale — the same "0 inside, signed
# excess width outside" shape as coaching._rel_dev, so
# WORST_DEVIATION_THRESHOLD, salience and the SYSTEM severity bands all read
# identically across the two focuses. Scales are chosen so ~1 unit ≈ "a
# coach would mention it" and ~2.5+ ≈ "the thing to talk about".
#
# Band justifications (court: NVZ line 2.13 m from net, baseline 6.71 m):
#   kitchen_distance_at_contact_m  metres BEHIND own NVZ line at contact,
#       gated to rally_shot_index >= 5 (serve/return/3rd/4th are legitimately
#       deep). Post-3rd-shot standard is being established AT the line;
#       baseline-camping at shot 5+ is ~4.6 m -> dev ~2.6. Singles hi widens
#       to 3.0 (singles legitimately holds deeper court). Negative = inside
#       the kitchen — never flagged low (lo = -KITCHEN is the physical floor).
#   return_depth_m  metres inside own baseline at the return (shot 2).
#       Returns are contacted 0-2 m inside the baseline; >=3.5 m in means the
#       returner camped in midcourt awaiting the serve. Generous because a
#       short serve legitimately pulls the returner in — the LLM's frame
#       check is the second filter.
#   transition_dwell_s  longest continuous stay in the midcourt band while
#       visible. The band is ~2.3 m deep; crossing at ~1.5 m/s takes ~1.5 s;
#       2.5 s allows one split-step pause — beyond that is camping in
#       no-man's-land.
#   kitchen_approach_after_return_s  seconds from the return contact to being
#       established near own kitchen line (returning team only — the serving
#       team CORRECTLY stays back for the third shot). Baseline->NVZ is
#       ~4.6 m at 2.5-3 m/s ≈ 2-3 s; "return and run" says beat the third
#       shot (~3 s). Never-arriving is capped (NEVER_ARRIVED_CAP_S) so the
#       dev tops out ~2.5 rather than growing with rally length.
#   recovery_offset_m  (singles only) lateral offset from the center line at
#       the opponent's NEXT contact, after this player was pulled wide.
#       Singles recovery = back toward center before the opponent strikes;
#       parked at the sideline (~3 m off-center) -> dev 2. Doubles/unset is
#       None — a doubles player's "home" x is unknowable without side
#       assignment.
#   partner_depth_stagger_m  (doubles only) largest |Δcourt_y| between
#       partners HELD for >= ~1 s. Partners move up/back together; one at
#       the NVZ + one at the baseline = ~4.6 m -> dev ~2 (the classic
#       one-up-one-back hole); 2.5 m tolerates normal transition staggering.
#       A transient poach split never flags (rolling-minimum window).
# ---------------------------------------------------------------------------
BANDS = {
    "kitchen_distance_at_contact_m": (-KITCHEN, 2.0, 1.0),
    "return_depth_m": (0.0, 3.5, 1.0),
    "transition_dwell_s": (0.0, 2.5, 1.0),
    "kitchen_approach_after_return_s": (0.0, 3.5, 1.0),
    "recovery_offset_m": (0.0, 1.5, 0.75),
    "partner_depth_stagger_m": (0.0, 2.5, 1.0),
}
_SINGLES_BAND_OVERRIDES = {
    "kitchen_distance_at_contact_m": (-KITCHEN, 3.0, 1.0),
}

MIN_SHOT_INDEX_FOR_KITCHEN = 5   # shots 1-4 (serve/return/3rd/4th) are legitimately deep
RETURN_SHOT_INDEX = 2            # rally_shot_index of the return of serve
KITCHEN_ESTABLISHED_TOLERANCE_M = 0.5  # "at the line" = within this of own NVZ band
APPROACH_ANCHOR_S = 3.5          # anchor the approach moment where they SHOULD already be in
APPROACH_MIN_RALLY_AFTER_RETURN_S = 4.0  # shorter post-return rallies can't fault the approach
NEVER_ARRIVED_CAP_S = 6.0
WIDE_CONTACT_OFFSET_M = 1.8      # |court_x - center| beyond this = pulled wide
RECOVERY_SAMPLE_TOLERANCE_FRAMES = 5
SAMPLE_GAP_BREAK_S = 0.5         # unseen time is never bridged in span runs
MIN_SPAN_SAMPLES = 15            # a player needs this many position samples in a
                                  # rally before span metrics mean anything (also
                                  # filters single-frame phantom track ids)
STAGGER_HOLD_S = 1.0
MIN_BOTH_VISIBLE_FRAMES = 30     # partner stagger needs both partners visible this
                                  # long (graceful degrade for the occluded far team)


def band_for(metric: str, match_type: str | None) -> tuple[float, float, float]:
    if match_type == "singles" and metric in _SINGLES_BAND_OVERRIDES:
        return _SINGLES_BAND_OVERRIDES[metric]
    return BANDS[metric]


def _abs_dev(value: float | None, band: tuple[float, float, float]) -> float | None:
    """Signed deviation beyond the absolute band, in `scale` units — 0 inside
    [lo, hi], (value-hi)/scale above, (value-lo)/scale below (negative). The
    positioning sibling of coaching._rel_dev; kept separate because _rel_dev
    bakes in Tukey-fence widening (FENCE_IQR_MULTIPLIER) that an absolute
    band must not get."""
    if value is None:
        return None
    lo, hi, scale = band
    if value > hi:
        return (value - hi) / scale
    if value < lo:
        return (value - lo) / scale
    return 0.0


def _soft_dev(value: float | None, band: tuple[float, float, float]) -> float | None:
    """Continuous signed distance from the band's midpoint in scale units —
    does NOT clamp to 0 inside the band. Fill-to-limit ranking only (see
    contracts.ShotFeatureVector.soft_deviation)."""
    if value is None:
        return None
    lo, hi, scale = band
    return (value - (lo + hi) / 2.0) / scale


def _own_kitchen_distance(court_y: float, team: str) -> float:
    """Metres behind own NVZ line (negative = inside the kitchen). near team's
    line is at NET_Y + KITCHEN (y grows toward the near baseline), far team's
    at NET_Y - KITCHEN."""
    if team == "near":
        return court_y - (NET_Y + KITCHEN)
    return (NET_Y - KITCHEN) - court_y


def _reached_own_kitchen(court_y: float, team: str) -> bool:
    return _own_kitchen_distance(court_y, team) <= KITCHEN_ESTABLISHED_TOLERANCE_M


def _majority_team(samples: list[tuple[int, float, float, str | None]]) -> str | None:
    counts = {"near": 0, "far": 0}
    for _f, _x, _y, team in samples:
        if team in counts:
            counts[team] += 1
    if counts["near"] == counts["far"] == 0:
        return None
    return "near" if counts["near"] >= counts["far"] else "far"


# ---------------------------------------------------------------------------
# Contact-anchored metrics
# ---------------------------------------------------------------------------

def contact_metrics(contact: dict, match_type: str | None) -> dict:
    """The 6 positioning metric values for one contact event — span-only
    metrics are always None here; contact metrics are None when gated out
    (wrong rally_shot_index) or when the contact has no court position/team.
    Never returns None itself: a contact with nothing applicable simply has
    all-None metrics and gets dropped at vector build (same "nothing to
    score" degrade as coaching)."""
    court_y = contact.get("court_y")
    team = contact.get("team")
    shot_index = contact.get("rally_shot_index")

    kitchen_distance = None
    if (court_y is not None and team in ("near", "far")
            and shot_index is not None and shot_index >= MIN_SHOT_INDEX_FOR_KITCHEN):
        kitchen_distance = _own_kitchen_distance(court_y, team)

    return_depth = None
    if (court_y is not None and team in ("near", "far")
            and shot_index == RETURN_SHOT_INDEX):
        return_depth = (COURT_L - court_y) if team == "near" else court_y

    return {
        "kitchen_distance_at_contact_m": kitchen_distance,
        "return_depth_m": return_depth,
        "transition_dwell_s": None,
        "kitchen_approach_after_return_s": None,
        "recovery_offset_m": None,
        "partner_depth_stagger_m": None,
    }


# ---------------------------------------------------------------------------
# Span moments — synthetic pseudo-contacts for between-contact problems.
# ---------------------------------------------------------------------------

def _pseudo_contact(video_id: str, rally_id: str, pid, metric: str, anchor_frame: int,
                     fps: float, team: str | None, court_xy: tuple | None,
                     value: float, span_start_f: int, span_end_f: int,
                     confidence: float) -> dict:
    return {
        "contact_id": f"pos_{rally_id}_{pid}_{metric}_{anchor_frame}",
        "video_id": video_id,
        "rally_id": rally_id,
        "frame_idx": int(anchor_frame),
        "timestamp_s": anchor_frame / fps if fps > 0 else 0.0,
        "player_id": pid,
        "team": team,
        "court_x": court_xy[0] if court_xy else None,
        "court_y": court_xy[1] if court_xy else None,
        "shot_side": None,
        "hand_speed_peak": None,
        # A span moment's confidence is position-sample coverage, not a wrist
        # peak — how much of the span we actually saw.
        "confidence": confidence,
        "is_serve": False,
        "kind": "span",
        "span": {"metric": metric, "value": value,
                  "start_s": span_start_f / fps if fps > 0 else 0.0,
                  "end_s": span_end_f / fps if fps > 0 else 0.0},
    }


def _runs(samples: list[tuple[int, float, float, str | None]], fps: float,
           predicate) -> list[list[tuple[int, float, float, str | None]]]:
    """Consecutive runs of samples where predicate(sample) holds, broken by
    non-matching samples AND by sample gaps > SAMPLE_GAP_BREAK_S (unseen time
    is never bridged — a player who vanished mid-run gets two short runs, not
    one long fabricated one)."""
    max_gap = SAMPLE_GAP_BREAK_S * fps
    runs: list[list] = []
    current: list = []
    prev_f = None
    for s in samples:
        f = s[0]
        if predicate(s) and (prev_f is None or not current or f - current[-1][0] <= max_gap):
            current.append(s)
        elif predicate(s):
            if current:
                runs.append(current)
            current = [s]
        else:
            if current:
                runs.append(current)
            current = []
        prev_f = f
    if current:
        runs.append(current)
    return runs


def _dwell_moment(video_id: str, rally: dict, pid, samples: list, fps: float,
                   coverage: float) -> dict | None:
    """M3 transition_dwell_s: longest continuous midcourt stay. Emitted even
    when in-band (dev 0.0) — one per (player, rally) — so clean transitions
    feed praise exemplars and the general-feedback aggregates, mirroring how
    every form contact is scored, not only bad ones."""
    if len(samples) < MIN_SPAN_SAMPLES:
        return None
    runs = _runs(samples, fps, lambda s: zone_for_court_y(s[2]) == ZONE_MIDCOURT)
    if runs:
        longest = max(runs, key=lambda r: r[-1][0] - r[0][0])
        dwell_s = (longest[-1][0] - longest[0][0]) / fps
        anchor = longest[len(longest) // 2]
        span_f = (longest[0][0], longest[-1][0])
    else:
        dwell_s = 0.0
        anchor = samples[len(samples) // 2]
        span_f = (anchor[0], anchor[0])
    return _pseudo_contact(
        video_id, rally["id"], pid, "transition_dwell_s", anchor[0], fps,
        _majority_team(samples), (anchor[1], anchor[2]), dwell_s,
        span_f[0], span_f[1], coverage)


def _approach_moments(video_id: str, rally: dict, samples_by_pid: dict,
                       rally_contacts: list[dict], fps: float,
                       coverage_by_pid: dict) -> list[dict]:
    """M4 kitchen_approach_after_return_s for every visible returning-team
    player. Serving team excluded by design (staying back for the third shot
    is correct play)."""
    ret = next((c for c in rally_contacts
                if c.get("rally_shot_index") == RETURN_SHOT_INDEX
                and c.get("team") in ("near", "far")), None)
    if ret is None:
        return []
    return_frame = ret["frame_idx"]
    returning_team = ret["team"]
    rally_end = rally["end_frame"]
    if (rally_end - return_frame) / fps < APPROACH_MIN_RALLY_AFTER_RETURN_S:
        return []  # too little post-return rally to fault anyone's approach

    out = []
    for pid, samples in samples_by_pid.items():
        if len(samples) < MIN_SPAN_SAMPLES or _majority_team(samples) != returning_team:
            continue
        post = [s for s in samples if s[0] >= return_frame]
        if not post:
            continue
        reach = next((s for s in post if _reached_own_kitchen(s[2], returning_team)), None)
        if reach is not None:
            value = (reach[0] - return_frame) / fps
        else:
            value = min((post[-1][0] - return_frame) / fps, NEVER_ARRIVED_CAP_S)
        anchor_frame = min(rally_end, return_frame + round(APPROACH_ANCHOR_S * fps))
        anchor = min(post, key=lambda s: abs(s[0] - anchor_frame))
        out.append(_pseudo_contact(
            video_id, rally["id"], pid, "kitchen_approach_after_return_s",
            anchor_frame, fps, returning_team, (anchor[1], anchor[2]), value,
            return_frame, anchor_frame, coverage_by_pid.get(pid, 0.0)))
    return out


def _recovery_moment(video_id: str, rally: dict, pid, samples: list,
                      rally_contacts: list[dict], fps: float, match_type: str | None,
                      coverage: float) -> dict | None:
    """M5 recovery_offset_m (singles only): after this player was pulled wide
    on their own contact, how far off-center they still were when the
    opponent next struck. Worst trigger per (player, rally)."""
    if match_type != "singles" or len(samples) < MIN_SPAN_SAMPLES:
        return None
    by_frame = sorted(rally_contacts, key=lambda c: c["frame_idx"])
    worst = None  # (value, opp_frame, sample)
    for i, c in enumerate(by_frame):
        if c.get("player_id") != pid or c.get("court_x") is None:
            continue
        if abs(c["court_x"] - CENTER_X) <= WIDE_CONTACT_OFFSET_M:
            continue
        opp = next((o for o in by_frame[i + 1:]
                    if o.get("team") not in (None, c.get("team"))), None)
        if opp is None:
            continue
        near_sample = min(samples, key=lambda s: abs(s[0] - opp["frame_idx"]))
        if abs(near_sample[0] - opp["frame_idx"]) > RECOVERY_SAMPLE_TOLERANCE_FRAMES:
            continue
        value = abs(near_sample[1] - CENTER_X)
        if worst is None or value > worst[0]:
            worst = (value, opp["frame_idx"], near_sample, c["frame_idx"])
    if worst is None:
        return None
    value, opp_frame, sample, wide_frame = worst
    return _pseudo_contact(
        video_id, rally["id"], pid, "recovery_offset_m", opp_frame, fps,
        _majority_team(samples), (sample[1], sample[2]), value,
        wide_frame, opp_frame, coverage)


def _stagger_moments(video_id: str, rally: dict, samples_by_pid: dict, fps: float,
                      match_type: str | None, coverage_by_pid: dict) -> list[dict]:
    """M6 partner_depth_stagger_m (doubles only): per team, the largest
    depth split between partners HELD >= STAGGER_HOLD_S. Attributed to the
    BACK player ("move up with your partner" is the coachable instruction).
    Requires both partners visible together >= MIN_BOTH_VISIBLE_FRAMES — the
    explicit graceful degrade for the camera-occluded far team."""
    if match_type != "doubles":
        return []
    teams: dict[str, list] = {}
    for pid, samples in samples_by_pid.items():
        if len(samples) < MIN_SPAN_SAMPLES:
            continue
        team = _majority_team(samples)
        if team:
            teams.setdefault(team, []).append(pid)

    out = []
    for team, pids in teams.items():
        if len(pids) < 2:
            continue
        # The two best-covered ids on this team (3+ can only be tracker
        # phantoms — identity slots cap at 2 per team).
        a, b = sorted(pids, key=lambda p: -len(samples_by_pid[p]))[:2]
        y_a = {f: y for f, _x, y, _t in samples_by_pid[a]}
        y_b = {f: y for f, _x, y, _t in samples_by_pid[b]}
        both = sorted(f for f in y_a if f in y_b)
        if len(both) < MIN_BOTH_VISIBLE_FRAMES:
            continue

        hold = STAGGER_HOLD_S * fps
        max_gap = SAMPLE_GAP_BREAK_S * fps
        best = None  # (held_min_stagger, anchor_frame)
        i = 0
        for j in range(len(both)):
            if j > 0 and both[j] - both[j - 1] > max_gap:
                i = j  # unseen time breaks the window — never bridged
            while both[j] - both[i] > hold:
                i += 1
            if both[j] - both[i] >= 0.9 * hold:
                window = both[i:j + 1]
                window_min = min(abs(y_a[f] - y_b[f]) for f in window)
                anchor = window[len(window) // 2]
                if best is None or window_min > best[0]:
                    best = (window_min, anchor, window[0], window[-1])
        if best is None:
            continue
        value, anchor, w0, w1 = best
        # Back player = farther from the net at the anchor frame.
        back = a if abs(y_a[anchor] - NET_Y) >= abs(y_b[anchor] - NET_Y) else b
        back_y = y_a[anchor] if back == a else y_b[anchor]
        back_x = next((x for f, x, _y, _t in samples_by_pid[back] if f == anchor), None)
        out.append(_pseudo_contact(
            video_id, rally["id"], back, "partner_depth_stagger_m", anchor, fps,
            team, (back_x, back_y) if back_x is not None else None, value,
            w0, w1, coverage_by_pid.get(back, 0.0)))
    return out


def span_moments(video_id: str, rally: dict, samples_by_pid: dict,
                  rally_contacts: list[dict], fps: float,
                  match_type: str | None) -> list[dict]:
    """Every span pseudo-contact for one rally. samples_by_pid: shot_context.
    position_samples over this rally's overlay entries (singles-folded by the
    caller — see coaching.singles_pid_map)."""
    n_frames = max(1, rally["end_frame"] - rally["start_frame"] + 1)
    coverage = {pid: min(1.0, len(s) / n_frames) for pid, s in samples_by_pid.items()}

    moments: list[dict] = []
    for pid, samples in samples_by_pid.items():
        m = _dwell_moment(video_id, rally, pid, samples, fps, coverage.get(pid, 0.0))
        if m:
            moments.append(m)
        m = _recovery_moment(video_id, rally, pid, samples, rally_contacts, fps,
                              match_type, coverage.get(pid, 0.0))
        if m:
            moments.append(m)
    moments.extend(_approach_moments(video_id, rally, samples_by_pid, rally_contacts,
                                      fps, coverage))
    moments.extend(_stagger_moments(video_id, rally, samples_by_pid, fps,
                                     match_type, coverage))
    return moments


# ---------------------------------------------------------------------------
# Scoring: raw metric dict -> PositioningFeatureVector
# ---------------------------------------------------------------------------

def _score(raw: dict, match_type: str | None) -> PositioningFeatureVector | None:
    """Category assignment mirrors coaching._classify: the category whose
    worst real (non-None) deviation is largest wins; all-None categories can
    never win; (nothing real anywhere) -> None, moment dropped."""
    devs = {f: _abs_dev(raw[f], band_for(f, match_type)) for f in METRIC_FIELDS}
    per_category = {}
    for cat, fields in CATEGORY_METRICS.items():
        real = [devs[f] for f in fields if devs[f] is not None]
        if real:
            per_category[cat] = max(real, key=abs)
    if not per_category:
        return None
    category = max(per_category, key=lambda c: abs(per_category[c]))

    softs = [_soft_dev(raw[f], band_for(f, match_type)) for f in METRIC_FIELDS]
    softs = [s for s in softs if s is not None]
    return PositioningFeatureVector(
        contact_id=raw["contact_id"], category=category,
        kitchen_distance_at_contact_m=raw["kitchen_distance_at_contact_m"],
        return_depth_m=raw["return_depth_m"],
        transition_dwell_s=raw["transition_dwell_s"],
        kitchen_approach_after_return_s=raw["kitchen_approach_after_return_s"],
        recovery_offset_m=raw["recovery_offset_m"],
        partner_depth_stagger_m=raw["partner_depth_stagger_m"],
        deviation_score=per_category[category],
        # Every None metric field, honestly listed: not read on this moment
        # (or structurally not applicable to it) — the LLM's null discipline
        # keys off this list.
        low_confidence_metrics=[f for f in METRIC_FIELDS if raw[f] is None],
        deviating_metrics=[f for f, d in devs.items() if d is not None and d != 0.0],
        soft_deviation=max(softs, key=abs) if softs else None,
        all_metric_devs=dict(devs))


def build_vectors(contacts: list[dict], rallies: list[dict],
                   samples_by_rally: dict, fps: float, match_type: str | None,
                   video_id: str) -> list[tuple[PositioningFeatureVector, dict]]:
    """The positioning counterpart of generate_coaching_report's form loop:
    (vector, meta_contact) pairs for every scoreable contact and span moment,
    ready for coaching.select_tip_candidates / build_report. contacts must
    already be rally-annotated + serve-excluded + singles-merged;
    samples_by_rally: {rally_id: position_samples dict, singles-folded}."""
    contacts_by_rally: dict = {}
    for c in contacts:
        contacts_by_rally.setdefault(c.get("rally_id"), []).append(c)

    vectors_meta: list[tuple[PositioningFeatureVector, dict]] = []
    for c in contacts:
        raw = {**contact_metrics(c, match_type), "contact_id": c["contact_id"]}
        vec = _score(raw, match_type)
        if vec is not None:
            vectors_meta.append((vec, c))

    for r in rallies:
        samples_by_pid = samples_by_rally.get(r["id"]) or {}
        rally_contacts = contacts_by_rally.get(r["id"]) or []
        for m in span_moments(video_id, r, samples_by_pid, rally_contacts, fps, match_type):
            raw = {f: None for f in METRIC_FIELDS}
            raw[m["span"]["metric"]] = m["span"]["value"]
            raw["contact_id"] = m["contact_id"]
            vec = _score(raw, match_type)
            if vec is not None:
                vectors_meta.append((vec, m))
    return vectors_meta


if __name__ == "__main__":
    # ponytail self-check: band math and gates, no DB.
    assert _abs_dev(0.0, BANDS["kitchen_distance_at_contact_m"]) == 0.0   # at the line
    assert _abs_dev(4.57, BANDS["kitchen_distance_at_contact_m"]) > 2.5   # baseline camp
    assert _abs_dev(-1.0, BANDS["kitchen_distance_at_contact_m"]) == 0.0  # inside kitchen: fine
    assert _abs_dev(None, BANDS["return_depth_m"]) is None
    assert band_for("kitchen_distance_at_contact_m", "singles")[1] == 3.0
    assert band_for("kitchen_distance_at_contact_m", "doubles")[1] == 2.0
    assert _soft_dev(2.0, (0.0, 2.0, 1.0)) == 1.0  # continuous inside band

    near_shot5 = {"court_y": NET_Y + KITCHEN, "team": "near", "rally_shot_index": 5}
    assert contact_metrics(near_shot5, None)["kitchen_distance_at_contact_m"] == 0.0
    early = {"court_y": NET_Y + KITCHEN, "team": "near", "rally_shot_index": 3}
    assert contact_metrics(early, None)["kitchen_distance_at_contact_m"] is None
    ret = {"court_y": COURT_L - 1.0, "team": "near", "rally_shot_index": 2}
    assert contact_metrics(ret, None)["return_depth_m"] == 1.0

    # Dwell: 3 s parked in midcourt -> flagged; anchored mid-run.
    mid_y = NET_Y + KITCHEN + 1.0  # midcourt, near side
    samples = [(f, 3.0, mid_y, "near") for f in range(0, 91, 3)]
    rally = {"id": "r1", "start_frame": 0, "end_frame": 90}
    m = _dwell_moment("v", rally, 1, samples, 30.0, 1.0)
    assert m is not None and m["span"]["value"] == 3.0 and m["kind"] == "span"
    assert m["contact_id"].startswith("pos_r1_1_transition_dwell_s_")
    vec = _score({**{f: None for f in METRIC_FIELDS},
                  "transition_dwell_s": 3.0, "contact_id": m["contact_id"]}, None)
    assert vec.category == "transition" and vec.deviation_score == 0.5
    assert "transition_dwell_s" in vec.deviating_metrics

    # Gap > 0.5 s breaks a run — two 1.5 s runs never merge into a 4.5 s flag.
    gap_samples = ([(f, 3.0, mid_y, "near") for f in range(0, 46, 3)]
                   + [(f, 3.0, mid_y, "near") for f in range(90, 136, 3)])
    m = _dwell_moment("v", {"id": "r2", "start_frame": 0, "end_frame": 135}, 1,
                       gap_samples, 30.0, 1.0)
    assert m["span"]["value"] == 1.5, m["span"]["value"]

    # Recovery: singles-only gate.
    rc = [{"player_id": 1, "team": "near", "court_x": CENTER_X + 2.5, "frame_idx": 10},
          {"player_id": 2, "team": "far", "court_x": CENTER_X, "frame_idx": 40}]
    wide_samples = [(f, CENTER_X + 2.5, NET_Y + 3, "near") for f in range(0, 60, 2)]
    assert _recovery_moment("v", rally, 1, wide_samples, rc, 30.0, "doubles", 1.0) is None
    m = _recovery_moment("v", rally, 1, wide_samples, rc, 30.0, "singles", 1.0)
    assert m is not None and abs(m["span"]["value"] - 2.5) < 1e-9 and m["frame_idx"] == 40

    # Stagger: doubles-only; one-up-one-back held for the whole rally flags the back player.
    a = [(f, 2.0, NET_Y + KITCHEN, "near") for f in range(0, 90)]
    b = [(f, 4.0, COURT_L - 0.5, "near") for f in range(0, 90)]
    ms = _stagger_moments("v", rally, {1: a, 2: b}, 30.0, "doubles", {1: 1.0, 2: 1.0})
    assert len(ms) == 1 and ms[0]["player_id"] == 2  # back player owns the tip
    assert ms[0]["span"]["value"] > 4.0
    assert _stagger_moments("v", rally, {1: a, 2: b}, 30.0, "singles", {}) == []
    print("positioning self-check OK")

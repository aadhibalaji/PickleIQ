"""Shot-type classification and rally-outcome context (Tier 2 of the
coaching-signal work): pure, in-memory annotation of contact_events dicts —
never written back to the DB, same in-memory-only pattern as
coaching.merge_singles_players.

Why: pooling every contact into one per-player metric distribution poisons
the personal baselines the whole scoring system rests on — a soft kitchen
exchange's short follow-through gets flagged as a "worst shot" purely
because the same player also hit drives. Stratifying the baseline by a
coarse shot type (court zone x swing tempo) compares like with like, and
rally context (was this the shot the rally ended on?) gives the coaching
call salience information no per-shot pose metric can.

Shot types are deliberately coarse and pose-only (no ball-track dependency,
same rule as contact_detect.py): court zone from the homography-projected
contact position, tempo from the contact's wrist-speed peak RELATIVE TO THAT
PLAYER'S OWN session median (pixel speeds are camera-distance dependent, so
a cross-player absolute cutoff would misclassify the far side). They are
stratification keys, not stroke labels — "kitchen_soft" means "one of this
player's slower swings taken in the kitchen", not a verified dink.

Also home to the canonical court-depth zone helper (moved from
report_stats.py, which now imports it from here — this module must not
import report_stats: report_stats imports coaching, which imports this).
"""
from __future__ import annotations

from ..cv.court import KITCHEN, NET_Y

# ---------------------------------------------------------------------------
# Court-depth zones — derived from the same court constants used everywhere
# else (cv/court.py). The half-court from net to baseline is NET_Y metres;
# KITCHEN of that (nearest the net) is the non-volley zone. The remaining
# distance out to the baseline splits into two equal-width bands: "midcourt"
# (closer to the kitchen line) and "backcourt" (closer to the baseline).
# ---------------------------------------------------------------------------
ZONE_KITCHEN, ZONE_MIDCOURT, ZONE_BACKCOURT = "kitchen", "midcourt", "backcourt"
_REMAINING_HALF = NET_Y - KITCHEN          # baseline-to-kitchen-line distance
_MID_BACK_CUTOFF = KITCHEN + _REMAINING_HALF / 2.0  # distance-from-net cutoff

TEMPO_SOFT, TEMPO_FAST = "soft", "fast"

# A player needs at least this many contacts with a real hand_speed_peak
# this session before their median is a meaningful soft/fast split point —
# below it, the tempo half of the type is omitted (zone-only type) rather
# than derived from a 2-3 point "distribution".
MIN_TEMPO_SAMPLES = 4


def zone_for_court_y(court_y: float) -> str:
    dist_from_net = abs(court_y - NET_Y)
    if dist_from_net <= KITCHEN:
        return ZONE_KITCHEN
    if dist_from_net <= _MID_BACK_CUTOFF:
        return ZONE_MIDCOURT
    return ZONE_BACKCOURT


def position_samples(entries: list[dict]) -> dict[int, list[tuple[int, float, float, str | None]]]:
    """player_id -> sorted [(frame, court_x, court_y, team), ...] for every
    overlay frame carrying a real cxy for that player. Players with zero
    samples in this range are simply absent from the returned dict — never
    zero-filled. (Moved from report_stats.py — shared with positioning.py,
    which must not import report_stats: report_stats imports coaching, which
    imports positioning.)"""
    by_pid: dict[int, list] = {}
    for e in entries:
        for p in e.get("players", []):
            cxy = p.get("cxy")
            if not cxy:
                continue
            pid = p.get("id")
            by_pid.setdefault(pid, []).append(
                (e["f"], float(cxy[0]), float(cxy[1]), p.get("team")))
    for pid in by_pid:
        by_pid[pid].sort(key=lambda t: t[0])
    return by_pid


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def classify_shot_types(contacts: list[dict],
                         tempo_override: dict[str, str] | None = None) -> dict[str, str]:
    """{contact_id: shot_type} for every contact that can be typed at all.
    shot_type = "<zone>_<tempo>" ("kitchen_soft", "backcourt_fast", ...) when
    this player has enough speed readings for a tempo split, else just
    "<zone>". A contact with no court position is absent from the result
    (can't be zoned -> can't be typed -> its metrics fall back to the
    player-wide baseline downstream, flagged as mixed_baseline_metrics).

    Tempo is per-player-relative by construction: fast = at/above THIS
    PLAYER's own median hand_speed_peak this session. Run AFTER
    merge_singles_players so phantom-identity contacts contribute to the
    right player's median.

    tempo_override ({contact_id: 'soft'|'fast'}, see ball_context.py):
    MEASURED tempo from actual ball speed — always beats the wrist-speed
    inference for that contact, and types a contact even when the player
    lacks MIN_TEMPO_SAMPLES wrist readings. Absent/None entries change
    nothing."""
    tempo_override = tempo_override or {}
    speeds_by_pid: dict = {}
    for c in contacts:
        sp = c.get("hand_speed_peak")
        if sp is not None:
            speeds_by_pid.setdefault(c["player_id"], []).append(float(sp))
    median_by_pid = {pid: _median(v) for pid, v in speeds_by_pid.items()
                     if len(v) >= MIN_TEMPO_SAMPLES}

    out: dict[str, str] = {}
    for c in contacts:
        if c.get("court_y") is None:
            continue
        zone = zone_for_court_y(c["court_y"])
        tempo = tempo_override.get(c["contact_id"])
        if tempo not in (TEMPO_SOFT, TEMPO_FAST):
            med = median_by_pid.get(c["player_id"])
            sp = c.get("hand_speed_peak")
            tempo = (TEMPO_FAST if float(sp) >= med else TEMPO_SOFT) \
                if med is not None and sp is not None else None
        out[c["contact_id"]] = f"{zone}_{tempo}" if tempo else zone
    return out


def annotate_rally_context(contacts: list[dict], rallies: list[dict],
                            fps: float) -> list[dict]:
    """Returns NEW contact dicts (inputs untouched) with rally-outcome
    context attached: rally_shot_index (1-based position of this contact in
    its rally's detected-contact sequence), rally_shot_count,
    is_rally_ending (this is the LAST detected contact of the rally — the
    rally ended on or right after it; whether that was a winner or an error
    is unknowable from pose alone and downstream consumers must not assume
    either), and seconds_until_rally_end (contact frame -> rally end_frame).

    Call BEFORE exclude_serves so index/count describe the real rally
    (serve = shot 1), not the serve-filtered remainder. A contact whose
    rally_id isn't in `rallies` (not currently analyzable) passes through
    without context fields rather than being dropped — dropping is the
    readiness filter's job, not this function's."""
    rally_by_id = {r["id"]: r for r in rallies}
    by_rally: dict = {}
    for c in contacts:
        if c.get("rally_id") in rally_by_id:
            by_rally.setdefault(c["rally_id"], []).append(c)

    context: dict[str, dict] = {}
    for rally_id, group in by_rally.items():
        group = sorted(group, key=lambda c: c["frame_idx"])
        end_frame = rally_by_id[rally_id]["end_frame"]
        for i, c in enumerate(group):
            context[c["contact_id"]] = {
                "rally_shot_index": i + 1,
                "rally_shot_count": len(group),
                "is_rally_ending": i == len(group) - 1,
                "seconds_until_rally_end": max(0.0, (end_frame - c["frame_idx"]) / fps)
                                            if fps > 0 else None,
            }
    return [{**c, **context.get(c["contact_id"], {})} for c in contacts]


if __name__ == "__main__":
    # ponytail self-check: zone cutoffs, per-player-relative tempo split,
    # rally indexing/ending flags. No DB.
    assert zone_for_court_y(NET_Y) == ZONE_KITCHEN
    assert zone_for_court_y(0.0) == ZONE_BACKCOURT

    contacts = (
        [{"contact_id": f"a{i}", "player_id": 1, "court_y": NET_Y, "hand_speed_peak": 100.0 + i}
         for i in range(4)]
        + [{"contact_id": "slow", "player_id": 1, "court_y": 0.0, "hand_speed_peak": 10.0},
           {"contact_id": "sparse", "player_id": 2, "court_y": 0.0, "hand_speed_peak": 999.0},
           {"contact_id": "nopos", "player_id": 1, "court_y": None, "hand_speed_peak": 50.0}])
    types = classify_shot_types(contacts)
    assert types["a3"].endswith("_fast")           # above own median
    assert types["slow"] == "backcourt_soft"       # below own median
    assert types["sparse"] == "backcourt"          # player 2: too few speeds for tempo
    assert "nopos" not in types                    # no court position -> untyped

    rallies = [{"id": "r1", "start_frame": 0, "end_frame": 300}]
    ann = annotate_rally_context(
        [{"contact_id": "s1", "rally_id": "r1", "frame_idx": 30},
         {"contact_id": "s2", "rally_id": "r1", "frame_idx": 240},
         {"contact_id": "orphan", "rally_id": "gone", "frame_idx": 5}],
        rallies, fps=30.0)
    by_id = {c["contact_id"]: c for c in ann}
    assert by_id["s1"]["rally_shot_index"] == 1 and not by_id["s1"]["is_rally_ending"]
    assert by_id["s2"]["is_rally_ending"] and by_id["s2"]["rally_shot_count"] == 2
    assert by_id["s2"]["seconds_until_rally_end"] == 2.0
    assert "rally_shot_index" not in by_id["orphan"]
    print("shot_context self-check OK")

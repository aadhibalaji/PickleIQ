"""Rule-based rally highlight tagging (Wave 3): classifies each rally into
zero or more tags purely from metrics/position data Wave 1/2 already compute
— no new detection. Every threshold below is a named constant so a coach can
retune without spelunking through the logic.

Read-only against rally_labels, overlays, contact_events, court geometry
(same read pattern position_metrics.py / shot_metrics.py already use).
Writes nothing — highlights are computed on demand, not persisted."""
from __future__ import annotations

from .. import db
from ..cv.court import COURT_W
from ..cv.overlays import read_range
from .position_metrics import _default_overlay_source, _in_kitchen
from .readiness import get_analyzable_rallies
from .shot_metrics import bucket_rally_length

# --- Firefight thresholds ---------------------------------------------------
# Shot rate (contacts per second of rally) above which the exchange counts as
# fast enough to be a "firefight" candidate.
FIREFIGHT_MIN_SHOTS_PER_S = 1.0
# Fraction of the rally's tracked frames a player must spend in their own
# kitchen (NVZ) to count as "near the net for a sustained stretch".
FIREFIGHT_NET_FRAC_MIN = 0.5

# --- Poach threshold ---------------------------------------------------------
# Metres past the team's own centerline a player's court_x must clear before
# a crossing counts as a real poach rather than pose-jitter noise.
POACH_CROSS_MARGIN_M = 0.3

# Defense in depth against upstream identity fragmentation (cv/identity.py /
# cv/pass2.py): re-ID will never be perfect, so even after the pass2.py fix
# that stops writing capacity-rejected noise detections into overlays, a
# genuine appearance/position re-acquisition hiccup can still hand a couple of
# frames to the wrong/transient id. A team must have TWO ids each tracked for
# at least this fraction of the rally's frames before they're treated as two
# real, distinct players eligible for a poach check — a one-or-two-frame
# phantom id should never register a crossing. Tunable.
MIN_TEAM_PRESENCE_FRAC = 0.15

TAG_LONG_RALLY = "Long Rally"
TAG_FIREFIGHT = "Firefight"
TAG_POACH = "Poach"


# ---------------------------------------------------------------------------
# Pure, DB-free core (unit-tested directly on synthetic overlay/metric data)
# ---------------------------------------------------------------------------

def aggregate_rally_positions(entries: list[dict]) -> tuple[dict[int, float], dict[str, dict[int, list[float]]]]:
    """entries: overlay frame dicts for one rally, any order. Returns
    (near_net_frac {player_id: fraction of tracked frames in own-side kitchen},
    team_x {team: {player_id: [court_x, ...]}})."""
    kitchen_n: dict[int, int] = {}
    total_n: dict[int, int] = {}
    team_x: dict[str, dict[int, list[float]]] = {"near": {}, "far": {}}
    for e in entries:
        for p in e.get("players", []):
            pid, cxy, team = p.get("id"), p.get("cxy"), p.get("team")
            if pid is None or cxy is None or team not in ("near", "far"):
                continue
            total_n[pid] = total_n.get(pid, 0) + 1
            if _in_kitchen(team, cxy):
                kitchen_n[pid] = kitchen_n.get(pid, 0) + 1
            team_x[team].setdefault(pid, []).append(float(cxy[0]))
    near_net_frac = {pid: kitchen_n.get(pid, 0) / total_n[pid] for pid in total_n}
    # Drop ids that weren't tracked for a big enough share of the rally before
    # they can count as a "real" second player for team_has_poach — see
    # MIN_TEAM_PRESENCE_FRAC. near_net_frac is left alone (Firefight already
    # requires >=50% net presence, which is a stronger gate on its own).
    rally_frames = len(entries) or 1
    for team in team_x:
        team_x[team] = {pid: xs for pid, xs in team_x[team].items()
                         if total_n.get(pid, 0) / rally_frames >= MIN_TEAM_PRESENCE_FRAC}
    return near_net_frac, team_x


def team_has_poach(player_xs: dict[int, list[float]]) -> bool:
    """player_xs: one team's {player_id: [court_x, ...]} for a rally. True if
    a player's x crossed past their partner's home side (by mean position)
    beyond POACH_CROSS_MARGIN_M at any point. Needs exactly 2 players with
    positions and distinguishable home sides — otherwise can't tell, so False."""
    means = {pid: sum(xs) / len(xs) for pid, xs in player_xs.items() if xs}
    if len(means) != 2:
        return False
    center = COURT_W / 2.0
    (pid_a, mean_a), (pid_b, mean_b) = means.items()
    if (mean_a < center) == (mean_b < center):
        return False  # both nominally on the same side, no partner split to poach across
    left_pid, right_pid = (pid_a, pid_b) if mean_a < mean_b else (pid_b, pid_a)
    if any(x > center + POACH_CROSS_MARGIN_M for x in player_xs[left_pid]):
        return True
    if any(x < center - POACH_CROSS_MARGIN_M for x in player_xs[right_pid]):
        return True
    return False


def classify_rally_tags(rally_length_bucket: str, shot_count: int, duration_s: float,
                         near_net_frac: dict[int, float],
                         team_x: dict[str, dict[int, list[float]]]) -> list[str]:
    """Pure classification core. duration_s: rally wall-clock length (seconds).
    near_net_frac / team_x: see aggregate_rally_positions. Returns tags in a
    stable order (Long Rally, Firefight, Poach), each at most once."""
    tags: list[str] = []

    if rally_length_bucket == "long":
        tags.append(TAG_LONG_RALLY)

    shot_rate = (shot_count / duration_s) if duration_s > 0 else 0.0
    if shot_rate >= FIREFIGHT_MIN_SHOTS_PER_S:
        both_teams_present = bool(team_x.get("near")) and bool(team_x.get("far"))
        both_near_net = all(
            any(near_net_frac.get(pid, 0.0) >= FIREFIGHT_NET_FRAC_MIN for pid in pids)
            for pids in team_x.values() if pids
        )
        if both_teams_present and both_near_net:
            tags.append(TAG_FIREFIGHT)

    if any(team_has_poach(pids_x) for pids_x in team_x.values()):
        tags.append(TAG_POACH)

    return tags


# ------------------------------------------------------------------------- DB

def compute_video_highlights(video_id: str, overlay_source: str | None = None) -> list[dict]:
    """Returns [{rally_id, tag, timestamp_s, video_id}], one row per
    (rally, tag) pair — zero or more rows per rally."""
    with db.connect() as con:
        vrow = con.execute("SELECT fps FROM videos WHERE id=?", (video_id,)).fetchone()
        contact_rows = db.rows_to_dicts(con.execute(
            "SELECT rally_id, COUNT(*) AS n FROM contact_events WHERE video_id=? GROUP BY rally_id",
            (video_id,)).fetchall())
    fps = float(vrow["fps"]) if vrow and vrow["fps"] else 30.0
    shot_counts = {r["rally_id"]: r["n"] for r in contact_rows}

    overlay_source = overlay_source or _default_overlay_source(video_id)
    rallies = get_analyzable_rallies(video_id, overlay_source=overlay_source)

    rows: list[dict] = []
    for rally in rallies:
        rally_id = rally["id"]
        entries = read_range(video_id, overlay_source, rally["start_frame"], rally["end_frame"])
        near_net_frac, team_x = aggregate_rally_positions(entries)

        shot_count = shot_counts.get(rally_id, 0)
        duration_s = (rally["end_frame"] - rally["start_frame"]) / fps
        bucket = bucket_rally_length(shot_count)
        timestamp_s = rally["start_frame"] / fps

        for tag in classify_rally_tags(bucket, shot_count, duration_s, near_net_frac, team_x):
            rows.append({"rally_id": rally_id, "tag": tag, "timestamp_s": timestamp_s,
                         "video_id": video_id})

    return rows


if __name__ == "__main__":
    # ponytail self-check: one tag per rule, on synthetic data.
    assert classify_rally_tags("long", 3, 10.0, {}, {}) == [TAG_LONG_RALLY]
    assert classify_rally_tags("short", 0, 0.0, {}, {}) == []

    firefight_near_net = {1: 0.9, 2: 0.9}
    firefight_team_x = {"near": {1: [3.0, 3.0]}, "far": {2: [3.0, 3.0]}}
    assert classify_rally_tags("short", 20, 10.0, firefight_near_net, firefight_team_x) == [TAG_FIREFIGHT]
    # same shot rate but nobody sustained net position -> no Firefight
    assert classify_rally_tags("short", 20, 10.0, {1: 0.1, 2: 0.1}, firefight_team_x) == []

    poach_team_x = {"near": {1: [1.0, 1.0, 5.0], 2: [5.0, 5.0, 5.0]}, "far": {}}
    assert classify_rally_tags("short", 0, 1.0, {}, poach_team_x) == [TAG_POACH]
    no_poach_team_x = {"near": {1: [1.0, 1.0], 2: [5.0, 5.0]}, "far": {}}
    assert classify_rally_tags("short", 0, 1.0, {}, no_poach_team_x) == []

    # MIN_TEAM_PRESENCE_FRAC gate (aggregate_rally_positions), via synthetic
    # overlay entries: id 1 is a dominant, long-lived near player who never
    # crosses; id 99 is a one-frame phantom that DOES cross. 10-frame rally.
    def _frame(f, near, far=()):
        return {"f": f, "players": [{"id": pid, "cxy": [x, 6.0], "team": "near"}
                                      for pid, x in near]
                          + [{"id": pid, "cxy": [x, 6.0], "team": "far"}
                             for pid, x in far]}
    phantom_entries = [_frame(i, [(1, 1.0)]) for i in range(9)] + [_frame(9, [(1, 1.0), (99, 5.0)])]
    near_net_frac, team_x = aggregate_rally_positions(phantom_entries)
    assert not team_has_poach(team_x["near"]), "one-frame phantom id must not trigger a poach"

    # Two genuinely sustained ids that cross should still trigger Poach.
    sustained_entries = ([_frame(i, [(1, 1.0), (2, 5.0)]) for i in range(8)]
                          + [_frame(8, [(1, 5.0), (2, 1.0)]), _frame(9, [(1, 5.0), (2, 5.0)])])
    _, team_x2 = aggregate_rally_positions(sustained_entries)
    assert team_has_poach(team_x2["near"]), "two sustained ids crossing should still trigger a poach"

    print("highlights self-check OK")

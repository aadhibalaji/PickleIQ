"""Shot-based rally metrics (Wave 2): shot side distribution and rally length
buckets, computed purely from ContactEvent rows already persisted by
contact_detect.py. No pose/overlay access needed here — this module only
reads the contact_events table (and rally_labels indirectly via rally_id,
already stamped onto each ContactEvent at detection time).

Read-only against contact_events; writes nothing (Wave 1's detect endpoint
owns writing to that table)."""
from __future__ import annotations

from .. import db
from .contracts import ContactEvent, RallyMetrics

# Rally length bucket thresholds (inclusive upper bounds on shot_count).
# short: 1-5, medium: 6-10, long: 11+
RALLY_SHORT_MAX_SHOTS = 5
RALLY_MEDIUM_MAX_SHOTS = 10


# ---------------------------------------------------------------------------
# Pure, DB-free core (unit-tested directly on synthetic ContactEvent lists)
# ---------------------------------------------------------------------------

def bucket_rally_length(shot_count: int) -> str:
    """short (1-5) / medium (6-10) / long (11+). shot_count<=0 also buckets
    as 'short' — there's no separate 'empty' bucket in the contract."""
    if shot_count <= RALLY_SHORT_MAX_SHOTS:
        return "short"
    if shot_count <= RALLY_MEDIUM_MAX_SHOTS:
        return "medium"
    return "long"


def shot_side_pct(events: list[ContactEvent]) -> dict[int, dict[str, float]]:
    """Per player, % of contacts on 'left' vs 'right' (fractions 0-1 of that
    player's own contacts in `events`, matching Wave 1's dominance-share
    convention of plain 0-1 floats rather than 0-100)."""
    counts: dict[int, dict[str, int]] = {}
    for e in events:
        side = e.shot_side
        if side not in ("left", "right"):
            continue
        c = counts.setdefault(e.player_id, {"left": 0, "right": 0})
        c[side] += 1
    return {
        pid: {side: (n / total if total else 0.0) for side, n in c.items()}
        for pid, c in counts.items()
        for total in (c["left"] + c["right"],)
    }


def compute_rally_shot_metrics(rally_id: str, video_id: str,
                                events: list[ContactEvent]) -> RallyMetrics:
    """events: ContactEvents for exactly one rally (any order)."""
    shot_count = len(events)
    return RallyMetrics(
        rally_id=rally_id, video_id=video_id,
        shot_side_pct=shot_side_pct(events),
        shot_count=shot_count,
        rally_length_bucket=bucket_rally_length(shot_count),
    )


# ------------------------------------------------------------------------- DB

def _row_to_event(row: dict) -> ContactEvent:
    return ContactEvent(
        contact_id=row["contact_id"], video_id=row["video_id"], rally_id=row["rally_id"],
        frame_idx=row["frame_idx"], timestamp_s=row["timestamp_s"], player_id=row["player_id"],
        team=row["team"], court_x=row["court_x"], court_y=row["court_y"],
        shot_side=row["shot_side"], hand_speed_peak=row["hand_speed_peak"],
        confidence=row["confidence"], is_serve=bool(row["is_serve"]),
    )


def compute_video_shot_metrics(video_id: str) -> dict:
    """Video-level shot metrics: per-rally RallyMetrics (shot_side_pct,
    shot_count, rally_length_bucket) plus a per-player video-level rollup.

    Returns {video_id, rally_metrics: [RallyMetrics], players:
    {player_id: {team, shot_count, shot_side_pct: {left, right}}}}."""
    with db.connect() as con:
        rows = db.rows_to_dicts(con.execute(
            "SELECT * FROM contact_events WHERE video_id=? ORDER BY rally_id, frame_idx",
            (video_id,)).fetchall())

    by_rally: dict[str, list[ContactEvent]] = {}
    for row in rows:
        by_rally.setdefault(row["rally_id"], []).append(_row_to_event(row))

    rally_metrics = [compute_rally_shot_metrics(rally_id, video_id, events)
                      for rally_id, events in by_rally.items()]

    all_events = [e for events in by_rally.values() for e in events]
    player_team = {e.player_id: e.team for e in all_events}
    player_counts: dict[int, int] = {}
    for e in all_events:
        player_counts[e.player_id] = player_counts.get(e.player_id, 0) + 1
    player_sides = shot_side_pct(all_events)

    players = {
        pid: {
            "team": team,
            "shot_count": player_counts.get(pid, 0),
            "shot_side_pct": player_sides.get(pid, {"left": 0.0, "right": 0.0}),
        }
        for pid, team in player_team.items()
    }

    return {"video_id": video_id, "rally_metrics": rally_metrics, "players": players}


if __name__ == "__main__":
    # ponytail self-check: bucket boundaries and side-% math.
    def _ev(pid, side, rally="r1"):
        return ContactEvent(contact_id="c", video_id="v", rally_id=rally, frame_idx=0,
                             timestamp_s=0.0, player_id=pid, team="near", court_x=0.0,
                             court_y=0.0, shot_side=side, hand_speed_peak=0.0,
                             confidence=1.0, is_serve=False)

    assert bucket_rally_length(1) == "short"
    assert bucket_rally_length(5) == "short"
    assert bucket_rally_length(6) == "medium"
    assert bucket_rally_length(10) == "medium"
    assert bucket_rally_length(11) == "long"

    events = [_ev(1, "left"), _ev(1, "left"), _ev(1, "right")]
    pct = shot_side_pct(events)
    assert pct[1]["left"] == 2 / 3
    assert pct[1]["right"] == 1 / 3

    rm = compute_rally_shot_metrics("r1", "v1", events)
    assert rm.shot_count == 3
    assert rm.rally_length_bucket == "short"
    print("shot_metrics self-check OK")

"""Aggregate report stats (Wave 2/3 add-on): three extra per-rally / per-video
numbers derived from data the pipeline already produces, for the coaching
report's charts. No new CV/pose processing — pure aggregation over overlay
`cxy` (homography-projected court position, NOT keypoint-confidence-gated —
see coaching.py's KP_CONF_MIN note, which does not apply here) and
contact_events.

Read-only against rally_labels, overlays, contact_events. Writes nothing.
Restricted to rallies readiness.get_analyzable_rallies() currently clears,
same as coaching.py.
"""
from __future__ import annotations

from .. import db
from ..cv import overlays
from ..cv.court import KITCHEN, NET_Y
from .coaching import merge_singles_players
from .readiness import default_overlay_source, get_analyzable_rallies

# Court-depth zones and position_samples moved to shot_context.py (now shared
# with the coaching shot-type classifier and positioning.py) — re-exported
# here under the original names so existing imports/tests keep working
# unchanged.
from .shot_context import (ZONE_BACKCOURT, ZONE_KITCHEN, ZONE_MIDCOURT,  # noqa: F401
                            position_samples as _position_samples,
                            zone_for_court_y as _zone_for_court_y)


# ---------------------------------------------------------------------------
# Stat 1 + 2: position-derived, from overlay cxy. Pure, unit-tested directly.
# ---------------------------------------------------------------------------


def forward_momentum_series(samples: list[tuple[int, float, float, str | None]],
                             fps: float) -> list[dict]:
    """[{frame, court_y, velocity_y}] — a simple finite difference between
    consecutive position samples (velocity_y in court-metres/second, signed:
    positive = increasing court_y). The first sample in a rally has no prior
    point to difference against, so its velocity_y is None (never 0-filled —
    a flat start would be indistinguishable from "we don't know")."""
    out = []
    prev = None
    for f, _x, y, _team in samples:
        velocity_y = None
        if prev is not None:
            pf, py = prev
            dt = (f - pf) / fps
            if dt > 0:
                velocity_y = (y - py) / dt
        out.append({"frame": f, "court_y": y, "velocity_y": velocity_y})
        prev = (f, y)
    return out


def kitchen_arrival_s(samples: list[tuple[int, float, float, str | None]],
                       rally_start_frame: int, fps: float) -> float | None:
    """Seconds from rally start until this player first reaches the kitchen
    (non-volley zone) on their own side of the net, or None if they never do
    (or have no position samples at all — see caller). "Reached" is defined
    directly from the court constants: the kitchen zone spans
    [NET_Y - KITCHEN, NET_Y + KITCHEN]; a 'near'-team player has reached it
    once court_y <= NET_Y + KITCHEN (moving in from the near baseline), a
    'far'-team player once court_y >= NET_Y - KITCHEN. If team is missing on
    every sample, it's inferred once from whichever side of the net the
    player's first sample sits on."""
    if not samples:
        return None
    team = next((t for _f, _x, _y, t in samples if t in ("near", "far")), None)
    if team is None:
        team = "near" if samples[0][2] >= NET_Y else "far"

    for f, _x, y, _t in samples:
        reached = y <= NET_Y + KITCHEN if team == "near" else y >= NET_Y - KITCHEN
        if reached:
            return max(0.0, (f - rally_start_frame) / fps)
    return None


# ---------------------------------------------------------------------------
# Stat 3: contact_events aggregation. Pure, unit-tested directly.
# ---------------------------------------------------------------------------

def shot_zone_counts(contacts: list[dict]) -> dict:
    """{player_id: {shot_side: {zone: {"serve": n, "non_serve": n}}}}.
    Contacts with a null court_x/court_y are excluded (can't be zoned);
    nothing else is excluded — serves are counted, broken out by is_serve
    rather than dropped, so nothing is silently discarded."""
    out: dict = {}
    for c in contacts:
        if c.get("court_x") is None or c.get("court_y") is None:
            continue
        pid = c["player_id"]
        side = c.get("shot_side")
        zone = _zone_for_court_y(c["court_y"])
        bucket = out.setdefault(pid, {}).setdefault(side, {}).setdefault(
            zone, {"serve": 0, "non_serve": 0})
        bucket["serve" if c.get("is_serve") else "non_serve"] += 1
    return out


# ---------------------------------------------------------------------------
# Orchestration — the only DB-touching layer, same pattern as coaching.py.
# ---------------------------------------------------------------------------

def get_report_stats(video_id: str, rally_source: str | None = None,
                      overlay_source: str | None = None) -> dict:
    overlay_source = overlay_source or rally_source or default_overlay_source(video_id)
    rallies = get_analyzable_rallies(video_id, rally_source=rally_source, overlay_source=overlay_source)
    ready_rally_ids = {r["id"] for r in rallies}

    with db.connect() as con:
        vrow = con.execute("SELECT fps, match_type FROM videos WHERE id=?", (video_id,)).fetchone()
        contacts = db.rows_to_dicts(con.execute(
            "SELECT * FROM contact_events WHERE video_id=? ORDER BY frame_idx",
            (video_id,)).fetchall())
    fps = float(vrow["fps"]) if vrow and vrow["fps"] else 30.0
    contacts = [c for c in contacts if c["rally_id"] in ready_rally_ids]
    contacts = merge_singles_players(contacts, vrow["match_type"] if vrow else None)

    rallies_out: dict = {}
    for r in rallies:
        entries = overlays.read_range(video_id, overlay_source, r["start_frame"], r["end_frame"])
        by_pid = _position_samples(entries)

        forward_momentum = {str(pid): forward_momentum_series(samples, fps)
                             for pid, samples in by_pid.items()}
        kitchen_arrival = {str(pid): kitchen_arrival_s(samples, r["start_frame"], fps)
                            for pid, samples in by_pid.items()}

        rallies_out[r["id"]] = {
            "start_frame": r["start_frame"],
            "end_frame": r["end_frame"],
            "forward_momentum": forward_momentum,
            "kitchen_arrival_s": kitchen_arrival,
        }

    shot_counts = {str(pid): sides for pid, sides in shot_zone_counts(contacts).items()}

    return {"video_id": video_id, "rallies": rallies_out, "shot_counts": shot_counts}


if __name__ == "__main__":
    # ponytail self-check: velocity finite-diff, kitchen-arrival crossing, zone
    # bucketing, null-omission on empty samples. No DB.
    samples = [(0, 3.0, 12.0, "near"), (30, 3.0, 9.0, "near"), (60, 3.0, 6.5, "near")]
    series = forward_momentum_series(samples, fps=30.0)
    assert series[0]["velocity_y"] is None
    assert abs(series[1]["velocity_y"] - (9.0 - 12.0) / 1.0) < 1e-9
    assert abs(series[2]["velocity_y"] - (6.5 - 9.0) / 1.0) < 1e-9

    arrival = kitchen_arrival_s(samples, rally_start_frame=0, fps=30.0)
    assert arrival == 2.0, arrival  # frame 60 is first with court_y <= NET_Y + KITCHEN (8.8392)

    never = kitchen_arrival_s([(0, 3.0, 12.0, "near")], rally_start_frame=0, fps=30.0)
    assert never is None

    assert kitchen_arrival_s([], rally_start_frame=0, fps=30.0) is None

    assert _zone_for_court_y(NET_Y) == ZONE_KITCHEN
    assert _zone_for_court_y(0.0) == ZONE_BACKCOURT       # far baseline
    assert _zone_for_court_y(NET_Y - KITCHEN - 0.01) == ZONE_MIDCOURT

    contacts = [
        {"player_id": 1, "shot_side": "left", "court_x": 1.0, "court_y": NET_Y, "is_serve": False},
        {"player_id": 1, "shot_side": "left", "court_x": 1.0, "court_y": 0.0, "is_serve": True},
        {"player_id": 1, "shot_side": "right", "court_x": None, "court_y": None, "is_serve": False},
    ]
    counts = shot_zone_counts(contacts)
    assert counts[1]["left"][ZONE_KITCHEN]["non_serve"] == 1
    assert counts[1]["left"][ZONE_BACKCOURT]["serve"] == 1
    assert "right" not in counts[1]  # null-position contact excluded, nothing else fabricated
    print("report_stats self-check OK")

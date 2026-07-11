"""Position-based rally metrics (Wave 1): Kitchen Arrival % and Forward
Pressure, computed purely from per-frame player court position already
stored in overlays (cxy, in court metres). No dependency on contact/shot
detection — this module only needs overlays + rally_labels + court geometry.

Read-only against cv/*, refine/*, rally_labels, overlays and calibration
data: every DB touch below is a SELECT. Nothing here writes to those tables.

Serving team is NOT recorded anywhere (rally_labels has no such column —
confirmed by reading db.py's schema and contracts.py's own note), so it is
DERIVED per-rally from cv.players.is_serve_formation applied to the rally's
opening frames. This is a HEURISTIC, not ground truth: is_serve_formation()
only confirms *a* valid serve stance exists on court, not which side is
serving (it checks both teams with the same symmetric rule). See
infer_serving_team() below for the (best-effort) tie-break.
"""
from __future__ import annotations

from .. import db
from ..cv import court
from ..cv.overlays import read_range
from ..cv.players import PlayerObs, is_serve_formation
from .contracts import RallyMetrics

# How far into a rally we scan for a serve-formation snapshot before giving
# up on inferring the serving team for that rally (~4s at 30fps).
SERVE_SCAN_FRAMES = 120


# --------------------------------------------------------------------- pure
# geometry / aggregation helpers — no DB, unit-testable with synthetic data.

def _in_kitchen(team: str, cxy) -> bool:
    """cxy = (x, y) court metres. Kitchen (NVZ) is the KITCHEN-metre strip on
    each side of the net, full court width."""
    x, y = cxy
    if not (0.0 <= x <= court.COURT_W):
        return False
    if team == "near":
        return court.NET_Y <= y <= court.NET_Y + court.KITCHEN
    return court.NET_Y - court.KITCHEN <= y <= court.NET_Y


def _depth_from_baseline(team: str, y: float) -> float:
    """Net-ward distance from the player's own baseline (0 at the baseline,
    growing toward/through the net), clamped to the court length."""
    d = (court.COURT_L - y) if team == "near" else y
    return max(0.0, min(court.COURT_L, d))


def _players_obs(players: list[dict]) -> list[PlayerObs]:
    """Overlay 'players' entries -> PlayerObs, dropping any without a court
    position (cxy) or team — is_serve_formation needs both."""
    obs = []
    for p in players:
        cxy, team = p.get("cxy"), p.get("team")
        if cxy is None or team not in ("near", "far"):
            continue
        obs.append(PlayerObs(track_id=p.get("id") or -1, box=(0.0, 0.0, 0.0, 0.0),
                              foot_xy=(0.0, 0.0), court_xy=(float(cxy[0]), float(cxy[1])),
                              team=team, slot=p.get("id")))
    return obs


def infer_serving_team(entries: list[dict]) -> str | None:
    """entries: overlay frame dicts (already trimmed to the scan window),
    in frame order. Returns 'near' | 'far' | None.

    Heuristic: at the first frame where is_serve_formation() holds, the
    serving side is guessed as whichever team's deepest player sits furthest
    behind their own kitchen line — the serve rules force the server to
    stand behind the baseline, so at the moment of serve that's usually the
    single deepest player on the court. Not ground truth."""
    for e in entries:
        obs = _players_obs(e.get("players", []))
        if not is_serve_formation(obs):
            continue

        def depth(o: PlayerObs) -> float:
            y = o.court_xy[1]
            return (y - (court.NET_Y + court.KITCHEN)) if o.team == "near" \
                else ((court.NET_Y - court.KITCHEN) - y)

        return max(obs, key=depth).team
    return None


def aggregate_frames(entries: list[dict]) -> tuple[dict[int, bool], dict[int, float], dict[int, str]]:
    """entries: overlay frame dicts for one rally. Returns
    (kitchen_arrival {player_id: bool}, forward_pressure {player_id: avg depth m},
    teams {player_id: 'near'|'far'})."""
    kitchen: dict[int, bool] = {}
    depth_sum: dict[int, float] = {}
    depth_n: dict[int, int] = {}
    teams: dict[int, str] = {}
    for e in entries:
        for p in e.get("players", []):
            pid, cxy, team = p.get("id"), p.get("cxy"), p.get("team")
            if pid is None or cxy is None or team not in ("near", "far"):
                continue
            teams[pid] = team
            kitchen.setdefault(pid, False)
            if _in_kitchen(team, cxy):
                kitchen[pid] = True
            depth_sum[pid] = depth_sum.get(pid, 0.0) + _depth_from_baseline(team, cxy[1])
            depth_n[pid] = depth_n.get(pid, 0) + 1
    pressure = {pid: depth_sum[pid] / depth_n[pid] for pid in depth_n}
    return kitchen, pressure, teams


# ------------------------------------------------------------------------- DB

def _default_overlay_source(video_id: str) -> str:
    """Pass 2 overlays carry stable identity slots; prefer them, fall back to
    Pass 1 spot-check overlays if Pass 2 hasn't run."""
    with db.connect() as con:
        row = con.execute(
            "SELECT 1 FROM overlays WHERE video_id=? AND source='pass2' LIMIT 1",
            (video_id,)).fetchone()
    return "pass2" if row else "pass1"


def _get_rallies(video_id: str) -> list[dict]:
    """Human labels win when present (ground truth), else Pass 2, else
    Pass 1 — same override order the rest of the app treats as canonical
    (see api/annotate.py get_rallies)."""
    with db.connect() as con:
        rows = db.rows_to_dicts(con.execute(
            "SELECT id, source, start_frame, end_frame FROM rally_labels WHERE video_id=?",
            (video_id,)).fetchall())
    by_source: dict[str, list[dict]] = {"human": [], "pass2": [], "pass1": []}
    for r in rows:
        if r["source"] in by_source:
            by_source[r["source"]].append(r)
    for src in ("human", "pass2", "pass1"):
        if by_source[src]:
            return sorted(by_source[src], key=lambda r: r["start_frame"])
    return []


def compute_rally_metrics(video_id: str, rally_id: str, overlay_source: str,
                           start_frame: int, end_frame: int) -> RallyMetrics:
    entries = read_range(video_id, overlay_source, start_frame, end_frame)
    kitchen, pressure, _teams = aggregate_frames(entries)
    return RallyMetrics(rally_id=rally_id, video_id=video_id,
                         kitchen_arrival=kitchen, forward_pressure=pressure)


def _aggregate_rallies(rallies: list[tuple[str, list[dict]]]) -> dict:
    """Pure aggregation core, shared by compute_video_metrics (DB-backed) and
    the unit tests (synthetic entries). rallies: [(rally_id, entries)].
    Returns {rally_metrics: [RallyMetrics], players: {...}} — see
    compute_video_metrics for the players-dict shape.

    NOT gated by a min-presence filter (unlike highlights.team_has_poach):
    pass2.py no longer writes capacity-rejected noise detections into
    overlays at all (see cv/pass2.py's players-list filter), which removes
    most of the phantom-id source this would guard against. What's left here
    aggregates at RALLY granularity (kitchen_arrival_pct/forward_pressure are
    denominated in whole rallies, via kitchen_den/pressure_n below), so a
    frame-level phantom id already can't inflate a percentage — at worst it
    adds one low-signal player-id entry with kitchen_serving_rallies=1 /
    forward_pressure_rallies=1, which is visibly a near-empty sample size
    rather than a misleading one. A tighter, frame-level gate would need this
    function to also thread each rally's total frame count through (it
    currently only sees per-player frame counts), which is a bigger plumbing
    change than this fix warrants — flagged as a follow-up if reviewers spot
    single-rally phantom ids in practice."""
    kitchen_num: dict[int, int] = {}
    kitchen_den: dict[int, int] = {}
    pressure_sum: dict[int, float] = {}
    pressure_n: dict[int, int] = {}
    player_team: dict[int, str] = {}
    rally_metrics: list[RallyMetrics] = []

    for rally_id, entries in rallies:
        kitchen, pressure, teams = aggregate_frames(entries)
        rally_metrics.append(RallyMetrics(rally_id=rally_id, video_id="",
                                           kitchen_arrival=kitchen, forward_pressure=pressure))
        player_team.update(teams)
        for pid, val in pressure.items():
            pressure_sum[pid] = pressure_sum.get(pid, 0.0) + val
            pressure_n[pid] = pressure_n.get(pid, 0) + 1

        start = min((e.get("f", 0) for e in entries), default=0)
        scan_end = start + SERVE_SCAN_FRAMES
        serving_team = infer_serving_team([e for e in entries if e.get("f", start) <= scan_end])
        if serving_team is not None:
            for pid, team in teams.items():
                if team == serving_team:
                    kitchen_den[pid] = kitchen_den.get(pid, 0) + 1
                    if kitchen.get(pid):
                        kitchen_num[pid] = kitchen_num.get(pid, 0) + 1

    players = {
        pid: {
            "team": team,
            "kitchen_arrival_pct": (kitchen_num.get(pid, 0) / kitchen_den[pid])
                                    if kitchen_den.get(pid) else None,
            "kitchen_serving_rallies": kitchen_den.get(pid, 0),
            "forward_pressure_avg_m": (pressure_sum[pid] / pressure_n[pid])
                                       if pressure_n.get(pid) else None,
            "forward_pressure_rallies": pressure_n.get(pid, 0),
        }
        for pid, team in player_team.items()
    }
    return {"rally_metrics": rally_metrics, "players": players}


def compute_video_metrics(video_id: str, overlay_source: str | None = None) -> dict:
    """Video-level aggregate for every player seen across the video's
    rallies. Returns {video_id, overlay_source, rally_metrics: [RallyMetrics],
    players: {player_id: {team, kitchen_arrival_pct, kitchen_serving_rallies,
    forward_pressure_avg_m, forward_pressure_rallies}}}.

    kitchen_arrival_pct denominator is rallies where the player's OWN team
    was the (heuristically) inferred server; forward_pressure is averaged
    across every rally the player appears in, serving or not.

    Only rallies readiness.get_analyzable_rallies() clears (Pass 2 done, and
    its overlay data still covers the rally's current frame range) are
    included — imported locally to avoid a circular import (readiness.py
    itself reuses _get_rallies/_default_overlay_source from this module)."""
    from .readiness import get_analyzable_rallies

    overlay_source = overlay_source or _default_overlay_source(video_id)
    rallies = get_analyzable_rallies(video_id, overlay_source=overlay_source)
    rally_entries = [(r["id"], read_range(video_id, overlay_source, r["start_frame"], r["end_frame"]))
                      for r in rallies]
    result = _aggregate_rallies(rally_entries)
    for rm in result["rally_metrics"]:
        rm.video_id = video_id
    return {"video_id": video_id, "overlay_source": overlay_source, **result}


if __name__ == "__main__":
    # ponytail self-check: a player camped in the kitchen every rally must
    # show up as 100% arrival when their team served every rally.
    entries = [
        {"f": 0, "players": [
            {"id": 1, "team": "near", "cxy": [3.0, 12.5]},   # deep near player (server)
            {"id": 2, "team": "far", "cxy": [3.0, 6.0]},     # far player up near kitchen
        ]},
        {"f": 5, "players": [
            {"id": 1, "team": "near", "cxy": [3.0, 7.0]},    # near player advances into kitchen
            {"id": 2, "team": "far", "cxy": [3.0, 1.0]},
        ]},
    ]
    kitchen, pressure, teams = aggregate_frames(entries)
    assert kitchen[1] is True and kitchen[2] is False
    assert teams == {1: "near", 2: "far"}
    print("position_metrics self-check OK")

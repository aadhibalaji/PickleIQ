"""Synthetic tracking-data generators for the refinement tests.

Builds :class:`TrackingData` directly (no DB, no vision models) with scripted
scenes: a legal serve, a rally with rhythmic net crossings, dead time with a
carried ball, a warm-up toss, a mid-court drill. Court frame matches the
repo: x in [0, COURT_W], y in [0, COURT_L], net at NET_Y.

The generators simulate the detector's player deadzone: ball samples within
``deadzone_m`` of any player are hidden (NaN), exactly like the real
detector losing the ball at paddle contact.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..io_loaders import COURT_L, COURT_W, KITCHEN, NET_Y, TrackingData

FPS = 30.0
DEADZONE_M = 0.9

PosFn = Callable[[float], tuple[float, float]]


def blank(duration_s: float, fps: float = FPS, f0: int = 0) -> TrackingData:
    """All-present, empty tracking span."""
    n = int(round(duration_s * fps))
    return TrackingData(
        fps=fps, f0=f0,
        present=np.ones(n, bool),
        ball_xy=np.full((n, 2), np.nan),
        ball_conf=np.zeros(n),
        ball_interp=np.zeros(n, bool),
        ball_height=np.full(n, np.nan),
        players={},
        pspeed=np.zeros(n),
        bounce=np.zeros(n, bool),
        hit=np.zeros(n, bool),
        evt_xy=np.full((n, 2), np.nan),
        video_id="synthetic",
    )


def _times(td: TrackingData) -> np.ndarray:
    return (td.f0 + np.arange(td.n)) / td.fps


def set_player(td: TrackingData, slot: int, pos: PosFn | tuple[float, float],
               t0: float = 0.0, t1: float = np.inf) -> None:
    """Place a player along `pos` (static tuple or t->xy) for t in [t0, t1]."""
    arr = td.players.setdefault(slot, np.full((td.n, 2), np.nan))
    t = _times(td)
    m = (t >= t0) & (t <= t1)
    if callable(pos):
        for i in np.flatnonzero(m):
            arr[i] = pos(t[i])
    else:
        arr[m] = pos


def set_ball(td: TrackingData, pos: PosFn, t0: float, t1: float,
             deadzone: bool = True, conf: float = 0.6) -> None:
    """Trace the ball along `pos` for t in [t0, t1]; hidden inside player
    deadzones when `deadzone` (mimics the detector's suppression)."""
    t = _times(td)
    for i in np.flatnonzero((t >= t0) & (t <= t1)):
        xy = np.asarray(pos(t[i]), dtype=float)
        if deadzone and any(
                np.isfinite(p[i][0]) and np.hypot(*(xy - p[i])) < DEADZONE_M
                for p in td.players.values()):
            continue
        td.ball_xy[i] = xy
        td.ball_conf[i] = conf


def mark_bounce(td: TrackingData, t: float, xy: tuple[float, float]) -> None:
    i = int(np.clip(round(t * td.fps) - td.f0, 0, td.n - 1))
    td.bounce[i] = True
    td.evt_xy[i] = xy


def _lerp(a: tuple[float, float], b: tuple[float, float],
          t0: float, t1: float) -> PosFn:
    ax, ay = a
    bx, by = b
    def fn(t: float) -> tuple[float, float]:
        u = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
        return (ax + u * (bx - ax), ay + u * (by - ay))
    return fn


def add_doubles_players(td: TrackingData, jitter: float = 0.0,
                        seed: int = 1) -> None:
    """Four players in a legal serve-receive formation: server deep right on
    the near end, diagonal receiver deep left on the far end, partners up."""
    rng = np.random.default_rng(seed)
    base = {1: (4.5, COURT_L + 0.4),   # server: behind near baseline, right
            2: (1.5, COURT_L - 2.6),   # server's partner
            3: (1.6, -0.4),            # receiver: behind far baseline, left
            4: (4.5, NET_Y - 2.3)}     # receiver's partner
    for slot, (x, y) in base.items():
        if jitter > 0:
            wob = rng.normal(0, jitter, (td.n, 2))
            arr = td.players.setdefault(slot, np.full((td.n, 2), np.nan))
            arr[:] = np.asarray([x, y]) + np.cumsum(wob, axis=0) * 0.01
        else:
            set_player(td, slot, (x, y))


def scripted_serve(td: TrackingData, t_contact: float,
                   server_xy: tuple[float, float] = (4.5, COURT_L + 0.2),
                   speed: float = 10.0) -> float:
    """A legal near-end serve: the ball is hidden in the server's deadzone,
    appears at the deadzone boundary at `t_contact` with serve speed, flies
    cross-court, and bounces beyond the far kitchen. Returns the bounce time."""
    target = (1.5, 1.8)                          # deep in the diagonal half
    dist = float(np.hypot(target[0] - server_xy[0], target[1] - server_xy[1]))
    t_land = t_contact + dist / speed
    # ball sits with the server pre-serve (inside deadzone -> auto-hidden)
    set_ball(td, lambda t: server_xy, max(0.0, t_contact - 2.0), t_contact)
    set_ball(td, _lerp(server_xy, target, t_contact, t_land),
             t_contact, t_land)
    mark_bounce(td, t_land, target)
    return t_land


def scripted_rally(td: TrackingData, t0: float, t1: float,
                   shot_s: float = 1.1, speed_spike: bool = True) -> None:
    """Rhythmic cross-net rally between deep positions; players get speed
    spikes near each crossing so reactivity correlates."""
    near = (3.5, COURT_L - 2.0)
    far = (2.5, 2.0)
    t = t0
    a, b = near, far
    while t < t1:
        te = min(t + shot_s, t1)
        set_ball(td, _lerp(a, b, t, te), t, te)
        mark_bounce(td, te - 0.08, b)
        a, b = b, a
        t = te
    if speed_spike:
        # jiggle every player around crossings (~mid-shot) for reactivity
        tt = _times(td)
        for slot, arr in td.players.items():
            phase = (tt - t0) % shot_s
            m = (tt >= t0) & (tt <= t1) & (np.abs(phase - shot_s / 2) < 0.2)
            wob = 0.06 * np.sin(tt[m] * 37.0 + slot)
            arr[m] += np.column_stack([wob, wob])


def scripted_dead_carry(td: TrackingData, t0: float, t1: float,
                        slot: int = 1) -> None:
    """Dead time: a player wanders slowly, carrying the ball (ball stays in
    their deadzone -> mostly invisible, occasionally visible right next to
    them)."""
    arr = td.players[slot]
    tt = _times(td)
    m = (tt >= t0) & (tt <= t1)
    drift = 0.4 * np.sin((tt[m] - t0) * 0.8)
    arr[m, 0] += drift
    arr[m, 1] -= 0.5 * (tt[m] - t0) / max(t1 - t0, 1e-9)
    # ball pops out of the deadzone occasionally, right beside the player
    for i in np.flatnonzero(m)[::15]:
        td.ball_xy[i] = arr[i] + np.array([1.0, 0.2])
        td.ball_conf[i] = 0.5


# ---------------------------------------------------------------------------
# full scenes
# ---------------------------------------------------------------------------

def serve_rally_scene(pre_s: float = 3.0, rally_s: float = 6.0,
                      tail_s: float = 3.0) -> tuple[TrackingData, float, float]:
    """Legal serve at t=pre_s, rally, then dead tail. Returns
    (td, t_contact, t_last_bounce)."""
    td = blank(pre_s + rally_s + tail_s + 2.0)
    add_doubles_players(td, jitter=0.4)
    t_contact = pre_s
    t_land = scripted_serve(td, t_contact)
    scripted_rally(td, t_land + 0.15, t_contact + rally_s)
    # last meaningful event = final rally bounce; then everyone idles
    return td, t_contact, t_contact + rally_s - 0.08


def toss_scene(duration_s: float = 10.0) -> TrackingData:
    """Warm-up: two NEAR-side players lob the ball to each other mid-court.
    Never crosses the net, no serve formation."""
    td = blank(duration_s)
    a, b = (2.0, COURT_L - 3.0), (4.2, COURT_L - 4.5)
    set_player(td, 1, a)
    set_player(td, 2, b)
    set_player(td, 3, (2.8, NET_Y - 1.2))    # far players milling mid-court
    set_player(td, 4, (3.6, NET_Y - 2.0))
    t, p, q = 1.0, a, b
    while t < duration_s - 1.5:
        te = t + 1.4                          # slow lob, ~2.5 m/s
        set_ball(td, _lerp(p, q, t, te), t, te)
        p, q = q, p
        t = te
    return td


def drill_scene(duration_s: float = 10.0) -> TrackingData:
    """Mid-court dinking drill: two players AT THE KITCHEN exchange over the
    net. Crossings exist but no baseline serve signature."""
    td = blank(duration_s)
    a = (3.0, NET_Y + 1.6)                    # near kitchen line
    b = (3.2, NET_Y - 1.6)                    # far kitchen line
    set_player(td, 1, a)
    set_player(td, 2, b)
    t, p, q = 1.0, (3.0, NET_Y + 1.1), (3.2, NET_Y - 1.1)
    while t < duration_s - 1.0:
        te = t + 0.8
        set_ball(td, _lerp(p, q, t, te), t, te)
        p, q = q, p
        t = te
    return td


def two_rally_scene(gap_s: float = 5.0) -> tuple[TrackingData, list[tuple[float, float]]]:
    """One long span holding TWO rallies separated by a dead carry gap.
    Returns (td, [(start1, end1), (start2, end2)]) approximate rally spans."""
    r1 = (3.0, 9.0)
    r2 = (9.0 + gap_s, 15.0 + gap_s)
    td = blank(r2[1] + 3.0)
    add_doubles_players(td, jitter=0.4)
    t_land1 = scripted_serve(td, r1[0])
    scripted_rally(td, t_land1 + 0.15, r1[1])
    scripted_dead_carry(td, r1[1] + 0.3, r2[0] - 0.5)
    t_land2 = scripted_serve(td, r2[0], server_xy=(4.4, COURT_L + 0.3))
    scripted_rally(td, t_land2 + 0.15, r2[1])
    return td, [r1, r2]

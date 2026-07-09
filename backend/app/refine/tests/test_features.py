"""Feature correctness on hand-constructed sequences (spec test 1)."""
from __future__ import annotations

import numpy as np

from ..config import RefineConfig
from ..features import compute_features, detect_bounces, formation_score
from ..io_loaders import COURT_L, NET_Y
from .synth import add_doubles_players, blank, set_ball, set_player, toss_scene

CFG = RefineConfig()


def test_net_crossing_detected():
    td = blank(4.0)
    # straight far -> near flight across the net over 1 s
    set_ball(td, lambda t: (3.0, 4.0 + 5.0 * (t - 1.0)), 1.0, 2.0)
    feats = compute_features(td, CFG)
    assert len(feats.crossings_t) == 1
    # crossing happens where y passes NET_Y: 4 + 5u = 6.7056 -> u ~= 0.54
    assert abs(feats.crossings_t[0] - 1.54) < 0.15
    assert feats.crossings_dir[0] == 1.0          # far -> near


def test_net_crossing_bridges_short_gap_not_long():
    def make(hide_s: float):
        td = blank(6.0)
        set_ball(td, lambda t: (3.0, 3.0 + 2.0 * (t - 1.0)), 1.0, 4.5)
        # hide a stretch straddling the net (y=NET_Y at t ~= 2.85)
        h0, h1 = 2.85 - hide_s / 2, 2.85 + hide_s / 2
        tt = (td.f0 + np.arange(td.n)) / td.fps
        td.ball_xy[(tt >= h0) & (tt <= h1)] = np.nan
        return compute_features(td, CFG)

    assert len(make(0.8).crossings_t) == 1        # gap < cross_max_gap_s
    assert len(make(2.0).crossings_t) == 0        # too long to trust


def test_bounce_from_height_vertex():
    td = blank(4.0)
    set_ball(td, lambda t: (3.0, 3.0 + 0.5 * t), 0.5, 3.5)
    tt = (td.f0 + np.arange(td.n)) / td.fps
    td.ball_height[:] = np.abs(tt - 2.0) * 1.5    # V-shape, vertex at t=2
    speed = np.zeros(td.n)
    mask = detect_bounces(td, CFG, speed)
    assert mask.any()
    assert abs(tt[np.flatnonzero(mask)[0]] - 2.0) < 0.2


def test_formation_triggers_only_in_serve_setup():
    # legal doubles serve-receive formation -> high score
    td = blank(3.0)
    add_doubles_players(td)
    speeds = {s: np.zeros(td.n) for s in td.players}
    good = formation_score(td.players, speeds, CFG, td.n)
    assert good.mean() > 0.5

    # warm-up toss positions (everyone mid-court) -> low score
    toss = toss_scene(4.0)
    speeds = {s: np.zeros(toss.n) for s in toss.players}
    bad = formation_score(toss.players, speeds, CFG, toss.n)
    assert bad.mean() < 0.3
    assert good.mean() > bad.mean() + 0.3


def test_deadzone_proximity_keeps_accumulating():
    """Ball disappearing adjacent to a player counts as near that player."""
    td = blank(6.0)
    set_player(td, 1, (3.0, COURT_L - 1.0))
    # ball visible right next to the player for 1 s, then swallowed by the
    # deadzone for 3 s
    set_ball(td, lambda t: (3.6, COURT_L - 1.0), 1.0, 2.0, deadzone=False)
    feats = compute_features(td, CFG)
    near = feats.col("near_player_time_n")
    i_mid = np.searchsorted(feats.t, 4.0)         # 2 s into the invisible span
    assert near[i_mid] > near[np.searchsorted(feats.t, 1.9)]
    assert near[i_mid] >= 2.5 / 4.0 - 0.15        # ~3 s accumulated (norm /4 s)


def test_ball_in_court_and_speed():
    td = blank(4.0)
    set_ball(td, lambda t: (3.0, 2.0 + 6.0 * (t - 1.0)), 1.0, 2.5)
    feats = compute_features(td, CFG)
    m = (feats.t > 1.3) & (feats.t < 2.3)
    assert feats.col("ball_in_court")[m].mean() > 0.9
    # 6 m/s scripted; normalized by 12
    spd = feats.col("ball_speed_n")[m]
    assert abs(spd.mean() - 0.5) < 0.15

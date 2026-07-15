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
    good, good_pose = formation_score(td.players, speeds, CFG, td.n)
    assert good.mean() > 0.5
    assert (good_pose == 0).all()  # no player_kp/H in this synthetic scene

    # warm-up toss positions (everyone mid-court) -> low score
    toss = toss_scene(4.0)
    speeds = {s: np.zeros(toss.n) for s in toss.players}
    bad, _bad_pose = formation_score(toss.players, speeds, CFG, toss.n)
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


def test_facing_away_retreat_fires_on_sustained_baseline_retreat():
    """A near-side player who stands still then steadily retreats toward
    their own baseline (y increasing, away from NET_Y) for several seconds
    should trip the signal once the retreat has sustained past the
    baseline_retreat_hold_s duration guard."""
    td = blank(6.0)
    start_y = NET_Y + 1.0

    def pos(t):
        y = start_y + 1.5 * max(0.0, t - 1.0)   # stationary 1s, then 1.5 m/s retreat
        return (3.0, min(y, COURT_L - 0.1))

    set_player(td, 1, pos)
    feats = compute_features(td, CFG)
    col = feats.col("facing_away_retreat")
    i_early = np.searchsorted(feats.t, 0.5)     # before any retreat motion
    i_late = np.searchsorted(feats.t, 4.5)      # well past the hold duration
    assert col[i_early] == 0.0
    assert col[i_late] >= 0.5                   # no pose kp -> motion-only credit


def test_facing_away_retreat_ignores_brief_mid_rally_step_back():
    """A player jittering back and forth (recovering to ready position after
    a shot, then moving up again) never sustains retreat for
    baseline_retreat_hold_s, so the signal must stay near zero — this is the
    false-positive case the duration guard exists to reject."""
    td = blank(4.0)

    def pos(t):
        phase = t % 0.6
        y = NET_Y + 1.0 + (0.6 if phase < 0.2 else 0.0)   # quick step back, step up
        return (3.0, y)

    set_player(td, 1, pos)
    feats = compute_features(td, CFG)
    col = feats.col("facing_away_retreat")
    assert col.max() < 0.5


def test_ball_in_court_and_speed():
    td = blank(4.0)
    set_ball(td, lambda t: (3.0, 2.0 + 6.0 * (t - 1.0)), 1.0, 2.5)
    feats = compute_features(td, CFG)
    m = (feats.t > 1.3) & (feats.t < 2.3)
    assert feats.col("ball_in_court")[m].mean() > 0.9
    # 6 m/s scripted; normalized by 12
    spd = feats.col("ball_speed_n")[m]
    assert abs(spd.mean() - 0.5) < 0.15

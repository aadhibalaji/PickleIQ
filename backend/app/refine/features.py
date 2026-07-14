"""Stage A — per-timestep feature extraction inside one analysis span.

Everything is computed at the native tracking frame rate (nan-aware, pure
numpy), then strided down to the feature timeline (~15 Hz). Alongside the
classifier feature columns, the :class:`Features` object keeps the discrete
event series (net crossings, bounces, ball appearances) that Stage D's serve
gate and end-snapping read directly.

Deadzone awareness: the ball detector suppresses/loses the ball adjacent to
players, so "ball track disappeared within ``deadzone_m`` of player X" is
treated as *near player X* — proximity time keeps accumulating through the
invisible stretch instead of resetting.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


@contextmanager
def _quiet():
    """Silence numpy's all-NaN / empty-slice RuntimeWarnings — all-NaN windows
    are an expected, handled case here (ball invisible, player untracked)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

from .config import RefineConfig
from .io_loaders import COURT_L, COURT_W, NET_Y, TrackingData, project_px

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6   # COCO keypoint indices (cv/players.py)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Nan-aware running median, window k (odd)."""
    if k <= 1 or len(x) == 0:
        return x
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    with _quiet():
        return np.nanmedian(windows, axis=1)


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last w samples (inclusive), edge-shortened."""
    if w <= 1 or len(x) == 0:
        return x.astype(float)
    c = np.cumsum(np.insert(np.nan_to_num(x, nan=0.0), 0, 0.0))
    cnt = np.cumsum(np.insert((~np.isnan(x)).astype(float), 0, 0.0))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w + 1)
    n = cnt[i + 1] - cnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (c[i + 1] - c[lo]) / n, 0.0)


def _rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return x.astype(float)
    pad = np.full(w - 1, -np.inf)
    padded = np.concatenate([pad, np.nan_to_num(x, nan=-np.inf)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, w)
    out = windows.max(axis=1)
    return np.where(np.isfinite(out), out, 0.0)


def _speed(xy: np.ndarray, fps: float, k: int, max_mps: float) -> np.ndarray:
    """Finite-difference speed magnitude over k frames; NaN where either
    endpoint is missing; implausible (> max_mps) values dropped as noise."""
    n = len(xy)
    v = np.full(n, np.nan)
    if n > k:
        d = np.linalg.norm(xy[k:] - xy[:-k], axis=1)
        v[k:] = d * fps / k
    v[v > max_mps] = np.nan
    return v


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass
class Features:
    """Feature timeline + event series for one analysis span."""
    fps: float                      # native tracking rate
    t: np.ndarray                   # [T] seconds (absolute clip time)
    frames: np.ndarray              # [T] frame indices
    cols: dict[str, np.ndarray]     # named per-timestep feature columns
    names: list[str]                # column order for matrix()
    # discrete events (absolute seconds, at native resolution)
    crossings_t: np.ndarray         # net-crossing times
    crossings_dir: np.ndarray       # -1 near->far, +1 far->near
    bounces_t: np.ndarray
    bounces_xy: np.ndarray          # [K,2] court m
    appear_t: np.ndarray            # ball (re)appearance after a gap
    appear_xy: np.ndarray           # [K,2] court m at appearance
    appear_speed: np.ndarray        # initial speed m/s just after appearance
    # native-rate series the serve gate reads directly
    native_t: np.ndarray
    formation_native: np.ndarray
    formation_held: np.ndarray      # rolling mean over formation_hold_s
    ball_speed_native: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ball_visible_native: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    player_speed: dict[int, np.ndarray] = field(default_factory=dict)
    td: TrackingData | None = None

    def matrix(self) -> np.ndarray:
        return np.column_stack([self.cols[c] for c in self.names])

    def col(self, name: str) -> np.ndarray:
        return self.cols[name]


# ---------------------------------------------------------------------------
# event detection
# ---------------------------------------------------------------------------

def _detect_crossings(t: np.ndarray, side: np.ndarray, visible: np.ndarray,
                      max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Sign flips of the ball's net side between visible samples that are at
    most max_gap_s apart. Returns (times, directions)."""
    idx = np.flatnonzero(visible & (side != 0))
    times, dirs = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        if side[a] != side[b] and (t[b] - t[a]) <= max_gap_s:
            times.append(0.5 * (t[a] + t[b]))
            dirs.append(1.0 if side[b] > side[a] else -1.0)
    return np.asarray(times), np.asarray(dirs)


def detect_bounces(td: TrackingData, cfg: RefineConfig,
                   speed: np.ndarray) -> np.ndarray:
    """Bounce mask at native rate. Prefers the overlay's evt bounces (arc
    junction classification); falls back to a vertical-velocity sign flip when
    height is available; last resort is a sharp speed drop while low."""
    if td.bounce.any():
        return td.bounce.copy()
    n = td.n
    out = np.zeros(n, bool)
    h = td.ball_height
    if np.isfinite(h).sum() >= 6:
        vh = np.full(n, np.nan)
        vh[1:] = np.diff(h)
        for i in range(2, n - 2):
            if (np.isfinite(vh[i - 1 : i + 2]).all()
                    and vh[i] <= 0 <= vh[i + 1]
                    and np.isfinite(h[i]) and h[i] < 0.5):
                out[i] = True
        if out.any():
            return out
    # speed-drop fallback: fast -> slow while visible (direction change proxy)
    vis = np.isfinite(td.ball_xy[:, 0])
    for i in range(2, n):
        if (vis[i] and np.isfinite(speed[i]) and np.isfinite(speed[i - 2])
                and speed[i - 2] > 3.0
                and speed[i] < speed[i - 2] * cfg.bounce_speed_drop):
            out[i] = True
    return out


def _detect_appearances(td: TrackingData, cfg: RefineConfig,
                        t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ball (re)appearances: a REAL observation (not arc-interpolated) after
    >= serve_min_gap_s with no observation. Initial speed is measured over the
    next few observed samples — the serve-contact signature."""
    observed = np.isfinite(td.ball_xy[:, 0]) & ~td.ball_interp
    idx = np.flatnonzero(observed)
    gap = int(round(cfg.serve_min_gap_s * td.fps))
    times, xys, speeds = [], [], []
    prev = -10**9
    for k, i in enumerate(idx):
        if i - prev >= max(1, gap):
            # initial speed: first following observation within ~0.3 s
            spd = 0.0
            for j in idx[k + 1 : k + 10]:
                if (j - i) <= td.fps * 0.35:
                    spd = float(np.linalg.norm(td.ball_xy[j] - td.ball_xy[i])
                                * td.fps / (j - i))
                    break
            times.append(t[i])
            xys.append(td.ball_xy[i])
            speeds.append(min(spd, cfg.max_speed_mps))
        prev = i
    return (np.asarray(times), np.asarray(xys).reshape(-1, 2),
            np.asarray(speeds))


# ---------------------------------------------------------------------------
# formation score
# ---------------------------------------------------------------------------

def _soft_behind(y: np.ndarray, baseline: float, sign: float, m: float) -> np.ndarray:
    """1.0 at/behind the baseline, ramping to 0 by m metres inside the court.
    sign=+1 for the far baseline (behind = smaller y), -1 for near."""
    inside = sign * (y - baseline)          # metres inside the court
    return np.clip(1.0 - inside / m, 0.0, 1.0)


def formation_score(players_xy: dict[int, np.ndarray],
                    speeds: dict[int, np.ndarray],
                    cfg: RefineConfig, n: int) -> np.ndarray:
    """Soft serve-formation score in [0,1] per native frame.

    Best over the two server-end hypotheses of:
    server stationary at/behind a baseline in one service half x diagonal
    opponent deep in the mirrored half behind the opposite baseline x (in
    doubles) remaining players roughly stationary. Works with 2 or 4 players.
    """
    slots = list(players_xy)
    if not slots:
        return np.zeros(n)
    X = np.stack([players_xy[s][:, 0] for s in slots])       # [P,N]
    Y = np.stack([players_xy[s][:, 1] for s in slots])
    V = np.stack([np.nan_to_num(speeds[s], nan=1e9) for s in slots])
    stationary = np.clip(1.0 - V / (2 * cfg.stationary_mps), 0.0, 1.0)
    m = cfg.behind_baseline_m
    behind_far = np.where(np.isfinite(Y), _soft_behind(Y, 0.0, 1.0, m), 0.0)
    behind_near = np.where(np.isfinite(Y), _soft_behind(Y, COURT_L, -1.0, m), 0.0)
    left = np.where(np.isfinite(X), X < COURT_W / 2, False)

    best = np.zeros(n)
    for behind_s, behind_r in ((behind_far, behind_near),
                               (behind_near, behind_far)):
        # stationarity discounts rather than vetoes — pose jitter inflates
        # apparent speed, especially for the tiny far-court player.
        server_q = behind_s * (0.35 + 0.65 * stationary)       # [P,N]
        si = np.argmax(server_q, axis=0)                       # server per frame
        cols = np.arange(n)
        s_score = server_q[si, cols]
        s_left = left[si, cols]
        # diagonal receiver: opposite baseline, mirrored half
        recv_q = behind_r * np.where(left != s_left[None, :], 1.0, 0.35)
        recv_q[si, cols] = 0.0                                 # not the server
        r_score = recv_q.max(axis=0) if len(slots) > 1 else np.zeros(n)
        # partners (doubles): everyone else roughly stationary
        if len(slots) > 2:
            others = 0.35 + 0.65 * stationary
            others[si, cols] = np.nan
            ri = np.argmax(recv_q, axis=0)
            others[ri, cols] = np.nan
            with _quiet():
                p_score = np.nan_to_num(np.nanmean(others, axis=0), nan=1.0)
            score = 0.45 * s_score + 0.35 * r_score + 0.20 * p_score
        else:
            score = 0.55 * s_score + 0.45 * r_score
        best = np.maximum(best, score)
    return best


def facing_away_retreat(td: TrackingData, cfg: RefineConfig, fps: float) -> np.ndarray:
    """Player-motion "point is probably over" signal in [0,1] per native frame:
    a player has sustained motion back toward their OWN baseline (away from
    the net), optionally boosted when their shoulder line also faces away
    from the net. Two guards keep this from firing on ordinary in-rally
    repositioning (e.g. recovering to ready position after a shot, which also
    involves a brief backward step):
      - the motion must be net-Y speed > baseline_retreat_speed_mps (not
        noise/jitter), AND
      - it must hold for >= baseline_retreat_ratio of the trailing
        baseline_retreat_hold_s window (a single-frame glance or a quick
        step back mid-point doesn't sustain long enough to cross this).
    Orientation is a soft multiplier on top, not a separate hard condition:
    the shoulder-line "chest normal" is inherently ambiguous by ~180 degrees
    from two keypoints alone, so it boosts confidence when available rather
    than gating the signal outright.
    """
    n = td.n
    out = np.zeros(n)
    hold_n = max(1, int(round(cfg.baseline_retreat_hold_s * fps)))
    for s, xy in td.players.items():
        y = xy[:, 1]
        valid = np.isfinite(y)
        away_dir = np.where(y > NET_Y, 1.0, -1.0)   # own-baseline direction
        vy = np.full(n, np.nan)
        vy[1:] = (y[1:] - y[:-1]) * fps
        retreating = (valid & np.isfinite(vy)
                      & (vy * away_dir > cfg.baseline_retreat_speed_mps))
        sustained = (_rolling_mean(retreating.astype(float), hold_n)
                     >= cfg.baseline_retreat_ratio)

        facing = np.zeros(n)
        kp = td.player_kp.get(s)
        if kp is not None and td.H is not None:
            ls, rs = kp[:, LEFT_SHOULDER, :], kp[:, RIGHT_SHOULDER, :]
            okkp = np.isfinite(ls[:, 0]) & np.isfinite(rs[:, 0]) & valid
            idx = np.flatnonzero(okkp)
            if len(idx):
                mid = (ls + rs) / 2.0
                # perpendicular to the shoulder line in image space: an
                # approximate chest-normal / facing direction. COCO's
                # left/right shoulder indices are anatomical (the person's
                # own left/right), so this flips consistently when a player
                # turns their back to the camera.
                shoulder_vec = ls - rs
                normal_px = np.stack([shoulder_vec[:, 1], -shoulder_vec[:, 0]], axis=1)
                p1c = project_px(td.H, mid[idx])
                p2c = project_px(td.H, mid[idx] + normal_px[idx])
                fv = p2c - p1c
                norm = np.linalg.norm(fv, axis=1)
                norm[norm < 1e-6] = 1.0
                fv_unit = fv / norm[:, None]
                away = np.stack([np.zeros(len(idx)), away_dir[idx]], axis=1)
                dot = np.sum(fv_unit * away, axis=1)
                facing[idx] = np.clip(dot, 0.0, 1.0)

        combined = sustained.astype(float) * (0.5 + 0.5 * facing)
        out = np.maximum(out, combined)
    return out


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def compute_features(td: TrackingData, cfg: RefineConfig) -> Features:
    """Full Stage A over one TrackingData span."""
    n, fps = td.n, td.fps
    t = (td.f0 + np.arange(n)) / fps
    visible = np.isfinite(td.ball_xy[:, 0])

    # ---- ball kinematics -------------------------------------------------
    speed_raw = _speed(td.ball_xy, fps, cfg.speed_diff_frames, cfg.max_speed_mps)
    speed = _median_filter(speed_raw, cfg.speed_median_k)
    speed_f = np.nan_to_num(speed, nan=0.0)

    side = np.where(visible, np.sign(td.ball_xy[:, 1] - NET_Y), 0.0)
    crossings_t, crossings_dir = _detect_crossings(t, side, visible,
                                                   cfg.cross_max_gap_s)
    # time since last crossing / count in trailing window
    time_since_cross = np.full(n, cfg.time_cap_s)
    crossings_3s = np.zeros(n)
    if len(crossings_t):
        ci = np.searchsorted(crossings_t, t, side="right")
        has = ci > 0
        time_since_cross[has] = np.minimum(
            t[has] - crossings_t[ci[has] - 1], cfg.time_cap_s)
        lo = np.searchsorted(crossings_t, t - cfg.crossings_window_s, side="left")
        crossings_3s = (ci - lo).astype(float)

    bounce = detect_bounces(td, cfg, speed_f)
    bounce_idx = np.flatnonzero(bounce)
    bounces_t = t[bounce_idx]
    bxy = np.where(np.isfinite(td.evt_xy[bounce_idx]).all(axis=1, keepdims=True),
                   td.evt_xy[bounce_idx], td.ball_xy[bounce_idx]) \
        if len(bounce_idx) else np.zeros((0, 2))
    # bounces on the current side since the last crossing
    bounces_since_cross = np.zeros(n)
    ev = sorted([(ct, "c") for ct in crossings_t] + [(bt, "b") for bt in bounces_t])
    cnt, k = 0.0, 0
    for i in range(n):
        while k < len(ev) and ev[k][0] <= t[i]:
            cnt = 0.0 if ev[k][1] == "c" else cnt + 1.0
            k += 1
        bounces_since_cross[i] = cnt

    in_court = visible & (td.ball_xy[:, 0] >= -cfg.court_margin_m) \
        & (td.ball_xy[:, 0] <= COURT_W + cfg.court_margin_m) \
        & (td.ball_xy[:, 1] >= -cfg.court_margin_m) \
        & (td.ball_xy[:, 1] <= COURT_L + cfg.court_margin_m)
    time_since_in = np.full(n, cfg.time_cap_s)
    last = -np.inf
    for i in range(n):
        if in_court[i]:
            last = t[i]
        time_since_in[i] = min(t[i] - last, cfg.time_cap_s)

    frac_visible = _rolling_mean(visible.astype(float),
                                 max(1, int(cfg.visible_window_s * fps)))

    # ---- players ----------------------------------------------------------
    pspeeds = {s: _median_filter(
        _speed(xy, fps, cfg.speed_diff_frames, 12.0), cfg.speed_median_k)
        for s, xy in td.players.items()}
    if pspeeds:
        P = np.stack(list(pspeeds.values()))
        with _quiet():
            mean_ps = np.nan_to_num(np.nanmean(P, axis=0), nan=0.0)
            max_ps = np.nan_to_num(np.nanmax(P, axis=0), nan=0.0)
    else:
        mean_ps = td.pspeed.copy()
        max_ps = td.pspeed.copy()

    present_cnt = np.zeros(n)
    ready_cnt = np.zeros(n)
    for s, xy in td.players.items():
        ok = np.isfinite(xy[:, 0])
        present_cnt += ok
        rm = cfg.ready_margin_m
        ready_cnt += (ok & (xy[:, 0] >= -rm) & (xy[:, 0] <= COURT_W + rm)
                      & (xy[:, 1] >= -rm) & (xy[:, 1] <= COURT_L + rm))
    with np.errstate(invalid="ignore", divide="ignore"):
        ready_frac = np.where(present_cnt > 0, ready_cnt / present_cnt, 0.0)

    formation = formation_score(td.players, pspeeds, cfg, n)
    formation_held = _rolling_mean(formation,
                                   max(1, int(cfg.formation_hold_s * fps)))

    # ---- ball <-> player proximity (deadzone-aware) ------------------------
    dist_np = np.full(n, 10.0)
    time_near = np.zeros(n)
    near_slot: int | None = None
    near_since = 0.0
    slots = list(td.players)
    for i in range(n):
        if visible[i]:
            best_s, best_d = None, np.inf
            for s in slots:
                p = td.players[s][i]
                if np.isfinite(p[0]):
                    d = float(np.hypot(*(td.ball_xy[i] - p)))
                    if d < best_d:
                        best_s, best_d = s, d
            if best_d < np.inf:
                dist_np[i] = min(best_d, 10.0)
            if best_d < cfg.near_player_m:
                if near_slot != best_s:
                    near_slot, near_since = best_s, t[i]
            else:
                near_slot = None
        else:
            if near_slot is None:
                # disappeared adjacent to someone? attribute the deadzone.
                j = i - 1
                while j >= 0 and not visible[j] and t[i] - t[j] < 0.5:
                    j -= 1
                if j >= 0 and visible[j]:
                    for s in slots:
                        p = td.players[s][j]
                        if (np.isfinite(p[0]) and
                                np.hypot(*(td.ball_xy[j] - p)) < cfg.deadzone_m):
                            near_slot, near_since = s, t[j]
                            break
            if near_slot is not None:
                dist_np[i] = 0.5 * cfg.near_player_m   # "inside the deadzone"
        time_near[i] = min(t[i] - near_since, cfg.time_cap_s) if near_slot is not None else 0.0

    # ---- reactivity ---------------------------------------------------------
    reactivity = np.zeros(n)
    if len(crossings_t):
        spike = max_ps > cfg.reactivity_spike_mps
        spike_t = t[spike]
        for i in range(n):
            lo = t[i] - cfg.reactivity_window_s
            sel = crossings_t[(crossings_t > lo) & (crossings_t <= t[i])]
            if len(sel):
                matched = sum(
                    bool(len(spike_t) and
                         np.min(np.abs(spike_t - ct)) <= cfg.reactivity_match_s)
                    for ct in sel)
                reactivity[i] = matched / len(sel)

    appear_t, appear_xy, appear_speed = _detect_appearances(td, cfg, t)

    facing_retreat = facing_away_retreat(td, cfg, fps)

    # ---- assemble columns @ native rate, then stride to feature_hz ---------
    cap = cfg.time_cap_s
    native_cols: dict[str, np.ndarray] = {
        "cross_recency": np.exp(-time_since_cross / cfg.cross_recency_tau_s),
        "crossings_3s": np.minimum(crossings_3s, 3.0) / 3.0,
        "ball_speed_n": np.minimum(speed_f, cfg.max_speed_mps) / cfg.ball_speed_norm_mps,
        "ball_visible": visible.astype(float),
        "frac_visible_1s": frac_visible,
        "net_side": side,
        "bounce_event": bounce.astype(float),
        "bounces_since_cross": np.minimum(bounces_since_cross, 3.0) / 3.0,
        "ball_in_court": in_court.astype(float),
        "out_of_court_time_n": time_since_in / cap,
        "dist_ball_player_n": np.minimum(dist_np, 5.0) / 5.0,
        "near_player_time_n": np.minimum(time_near, 4.0) / 4.0,
        "mean_pspeed_n": np.minimum(mean_ps, 3.0) / 3.0,
        "max_pspeed_n": np.minimum(max_ps, 5.0) / 5.0,
        "ready_frac": ready_frac,
        "formation_valid": formation,
        "reactivity": reactivity,
        "facing_away_retreat": facing_retreat,
    }
    hist = max(1, int(cfg.history_s * fps))
    for name in ("cross_recency", "ball_speed_n", "frac_visible_1s",
                 "max_pspeed_n", "formation_valid"):
        native_cols[f"{name}_hist"] = _rolling_mean(native_cols[name], hist)
    native_cols["ball_speed_n_histmax"] = _rolling_max(
        native_cols["ball_speed_n"], hist)

    step = max(1, int(round(fps / cfg.feature_hz)))
    sel = np.arange(0, n, step)
    cols = {k: v[sel].astype(float) for k, v in native_cols.items()}
    names = sorted(cols)

    return Features(
        fps=fps, t=t[sel], frames=td.f0 + sel, cols=cols, names=names,
        crossings_t=crossings_t, crossings_dir=crossings_dir,
        bounces_t=bounces_t, bounces_xy=np.asarray(bxy).reshape(-1, 2),
        appear_t=appear_t, appear_xy=appear_xy, appear_speed=appear_speed,
        native_t=t, formation_native=formation, formation_held=formation_held,
        ball_speed_native=speed_f, ball_visible_native=visible,
        player_speed=pspeeds, td=td,
    )

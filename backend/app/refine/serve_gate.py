"""Stage D — serve gate + boundary snapping for one candidate segment.

A real rally begins with a serve; warm-up tossing, drills and carried balls
don't. The gate scores four phases and combines them into ``serve_gate_score``
in [0,1]:

1. **pre-serve** — serve formation held for >= formation_hold_s shortly
   before/around the segment start;
2. **contact** — the ball track APPEARS (the detector's player deadzone hides
   the toss) adjacent to a stationary at/behind-baseline player with high
   initial speed, or an equivalent speed onset if it stayed visible. This is
   the refined-start anchor;
3. **flight** — a net crossing 0.4-1.5 s after contact (booster only: a serve
   faulted into the net is still a serve attempt);
4. **landing** — first bounce beyond the kitchen on the receiver's diagonal
   half (booster only).

Also provides the end snap: walk back from the falling edge to the last
meaningful ball event (bounce / crossing / high-speed moment) + post_roll.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RefineConfig
from .features import Features
from .io_loaders import COURT_L, COURT_W, KITCHEN, NET_Y


@dataclass
class ServeGateResult:
    score: float
    contact_t: float | None            # serve-contact anchor (seconds) or None
    components: dict = field(default_factory=dict)


def _server_fit(feats: Features, t: float, xy: np.ndarray,
                cfg: RefineConfig) -> tuple[float, float | None]:
    """How well `xy` at time `t` sits adjacent to a stationary at/behind-
    baseline player. Returns (fit in [0,1], server court y or None)."""
    td = feats.td
    i = int(np.clip(np.searchsorted(feats.native_t, t), 0, len(feats.native_t) - 1))
    best, best_y = 0.0, None
    for slot, pxy in td.players.items():
        p = pxy[i]
        if not np.isfinite(p[0]):
            # tracking gap — use the nearest valid sample within ~0.5 s
            ok = np.flatnonzero(np.isfinite(pxy[:, 0]))
            if not len(ok):
                continue
            j = ok[np.argmin(np.abs(ok - i))]
            if abs(j - i) > td.fps * 0.5:
                continue
            p = pxy[j]
        # behind either baseline (soft over behind_baseline_m into the court)
        m = cfg.behind_baseline_m
        behind = max(np.clip(1 - p[1] / m, 0, 1), np.clip(1 - (COURT_L - p[1]) / m, 0, 1))
        near = np.clip(1 - max(0.0, float(np.hypot(*(xy - p))) - 0.3)
                       / cfg.serve_contact_near_m, 0, 1)
        spd = feats.player_speed.get(slot)
        v = spd[i] if spd is not None and np.isfinite(spd[i]) else 0.0
        still = np.clip(1 - v / (2 * cfg.stationary_mps), 0, 1)
        fit = behind * near * (0.5 + 0.5 * still)
        if fit > best:
            best, best_y = float(fit), float(p[1])
    return best, best_y


def _contact_candidates(feats: Features, t0: float, t1: float,
                        cfg: RefineConfig) -> list[tuple[float, np.ndarray, float]]:
    """(time, court xy, initial speed) of plausible serve contacts in [t0,t1]:
    ball appearances after a deadzone gap, plus speed onsets while visible."""
    out = []
    for t, xy, spd in zip(feats.appear_t, feats.appear_xy, feats.appear_speed):
        if t0 <= t <= t1 and spd >= cfg.serve_contact_speed_mps:
            out.append((float(t), xy, float(spd)))
    # speed onset: slow/idle -> serve speed within a few frames
    v, vis, tt = feats.ball_speed_native, feats.ball_visible_native, feats.native_t
    k = max(1, int(feats.fps * 0.15))
    sel = np.flatnonzero((tt >= t0) & (tt <= t1) & vis
                         & (v >= cfg.serve_contact_speed_mps))
    appear_ts = set(np.round(feats.appear_t, 2))
    last = -np.inf
    for i in sel:
        before = v[max(0, i - k)]
        if (before < cfg.serve_contact_speed_mps * 0.5 or not vis[max(0, i - k)]) \
                and tt[i] - last > 0.5 and round(tt[i], 2) not in appear_ts:
            out.append((float(tt[i]), feats.td.ball_xy[i], float(v[i])))
            last = tt[i]
    return sorted(out)


def score_serve(feats: Features, seg_start: float, seg_end: float,
                cfg: RefineConfig,
                search_from: float | None = None) -> ServeGateResult:
    """Serve-gate score for a Stage C segment [seg_start, seg_end] seconds.

    `search_from` extends the lookback to the Pass 1 window start for the
    window's first segment — the serve sits near the window start, and the
    smoothed rising edge can lag it by more than the default lookback.
    """
    t0 = seg_start - cfg.serve_lookback_s
    if search_from is not None:
        t0 = min(t0, search_from)
    t1 = min(seg_start + cfg.serve_search_s, seg_end)

    # -- phase 1: pre-serve formation ------------------------------------
    m = (feats.native_t >= t0) & (feats.native_t <= t1)
    pre = float(feats.formation_held[m].max()) if m.any() else 0.0

    # -- phase 2: contact ---------------------------------------------------
    best = None                      # (combined, t, server_y, xy)
    for t, xy, spd in _contact_candidates(feats, t0, t1, cfg):
        fit, server_y = _server_fit(feats, t, xy, cfg)
        speed_q = np.clip(spd / (2 * cfg.serve_contact_speed_mps), 0, 1)
        q = fit * (0.4 + 0.6 * speed_q)
        if fit > 0.15 and (best is None or q > best[0]):
            best = (float(q), t, server_y, xy)
    contact = best[0] if best else 0.0
    contact_t = best[1] if best else None
    server_y = best[2] if best else None

    # -- phase 3: flight (booster) ------------------------------------------
    anchor = contact_t if contact_t is not None else seg_start
    flight = 0.0
    for ct, cd in zip(feats.crossings_t, feats.crossings_dir):
        if cfg.serve_flight_min_s <= ct - anchor <= cfg.serve_flight_max_s:
            flight = 0.8
            if server_y is not None:
                # serve flies away from the server's end
                expected = 1.0 if server_y < NET_Y else -1.0
                if cd == expected:
                    flight = 1.0
            break

    # -- phase 4: landing (booster) --------------------------------------------
    landing = 0.0
    if server_y is not None and len(feats.bounces_t):
        for bt, bxy in zip(feats.bounces_t, feats.bounces_xy):
            if not (anchor < bt <= anchor + 2.5) or not np.isfinite(bxy[1]):
                continue
            beyond = bxy[1] > NET_Y + KITCHEN if server_y < NET_Y \
                else bxy[1] < NET_Y - KITCHEN
            in_x = -0.5 <= bxy[0] <= COURT_W + 0.5
            landing = 1.0 if (beyond and in_x) else 0.3
            break

    # -- rally evidence: sustained crossing rate inside the segment ----------
    # Discounted by play DEPTH: a kitchen dinking drill also crosses the net
    # rhythmically, but its ball never leaves the kitchen corridor, while a
    # real (served) rally spends plenty of time beyond it.
    n_cross = int(np.sum((feats.crossings_t >= seg_start)
                         & (feats.crossings_t <= seg_end)))
    rate10 = n_cross / max(seg_end - seg_start, 1e-6) * 10.0
    rally_ev = float(np.clip(rate10 / cfg.rally_evidence_x10s, 0, 1))
    seg_m = ((feats.native_t >= seg_start) & (feats.native_t <= seg_end)
             & feats.ball_visible_native)
    if seg_m.any():
        y = feats.td.ball_xy[seg_m, 1]
        deep_frac = float(np.mean(np.abs(y - NET_Y) > KITCHEN))
        rally_ev *= float(np.clip(deep_frac / 0.3, 0, 1))

    comps = {"pre_serve": pre, "contact": contact, "flight": flight,
             "landing": landing, "rally_evidence": rally_ev}
    w = cfg.serve_weights
    score = sum(w[k] * comps[k] for k in w) / sum(w.values())
    return ServeGateResult(score=float(score), contact_t=contact_t,
                           components=comps)


def snap_end(feats: Features, seg_start: float, seg_end: float,
             cfg: RefineConfig) -> tuple[float, str]:
    """Walk back from the falling edge to the last meaningful ball event.
    Returns (event time, kind) — post_roll is applied by the caller."""
    cands: list[tuple[float, str]] = []
    b = feats.bounces_t[(feats.bounces_t >= seg_start) & (feats.bounces_t <= seg_end)]
    if len(b):
        cands.append((float(b[-1]), "bounce"))
    c = feats.crossings_t[(feats.crossings_t >= seg_start)
                          & (feats.crossings_t <= seg_end)]
    if len(c):
        cands.append((float(c[-1]), "crossing"))
    m = ((feats.native_t >= seg_start) & (feats.native_t <= seg_end)
         & (feats.ball_speed_native >= cfg.end_event_speed_mps))
    if m.any():
        cands.append((float(feats.native_t[np.flatnonzero(m)[-1]]), "fast_ball"))
    if not cands:
        return seg_end, "falling_edge"
    return max(cands)

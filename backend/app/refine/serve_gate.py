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
    # hard gate — see score_serve: flight_or_landing_ok is always required;
    # ALSO needs EITHER contact_ok OR pre_serve_compensate_ok (not both floors
    # at once — a weak/occluded contact read no longer vetoes an otherwise
    # well-evidenced serve on its own).
    flight_or_landing_ok: bool = True
    flight_directional: bool = False       # diagnostic: flight == 1.0 (direction confirmed, not just a crossing found)
    contact_ok: bool = True
    contact_near_floor_ok: bool = False    # contact >= serve_contact_near_floor_min (gates via the directional-flight OR branch)
    pre_serve_ok: bool = True              # diagnostic: pre >= serve_pre_serve_min (informational, not gating)
    pre_serve_compensate_ok: bool = False  # pre >= serve_pre_serve_compensate_min (gates via the OR branch)
    pass_path: str | None = None           # "contact" | "formation_compensation" | "directional_flight_compensation" | None (rejected)
    # diagnostic only: fraction of the pre-serve window with real
    # shoulder-pose data behind `components["pre_serve"]`, vs. the neutral
    # 0.5 orientation fallback
    pre_serve_pose_coverage: float = 0.0


def _point_box_dist_px(px: np.ndarray, box: np.ndarray) -> float:
    """Distance (px) from a point to an axis-aligned box (0 if inside)."""
    dx = max(box[0] - px[0], 0.0, px[0] - box[2])
    dy = max(box[1] - px[1], 0.0, px[1] - box[3])
    return float(np.hypot(dx, dy))


def _ball_height_at(feats: Features, t: float) -> float | None:
    """Reconstructed ball height (metres) at time `t`, or None if unavailable
    (no ballistic 3D fit / not finite there) — callers should treat None as
    "can't verify" and not penalize."""
    td = feats.td
    h_arr = getattr(td, "ball_height", None)
    if h_arr is None:
        return None
    i = int(np.clip(np.searchsorted(feats.native_t, t), 0, len(feats.native_t) - 1))
    h = h_arr[i]
    return float(h) if np.isfinite(h) else None


def _server_fit(feats: Features, t: float, xy: np.ndarray,
                cfg: RefineConfig) -> tuple[float, float | None]:
    """How well the ball at time `t` sits adjacent to a stationary at/behind-
    baseline player. Proximity is measured in IMAGE PIXELS (ball pixel -> player
    box) when the pixel track is available — the ball's court position is
    height-corrupted at the serve, so a court-metre distance there is garbage;
    the "behind baseline" test still uses the reliable player court position.
    Falls back to court-metre distance (`xy`) when no pixel track is present
    (synthetic inputs). Returns (fit in [0,1], server court y or None)."""
    td = feats.td
    i = int(np.clip(np.searchsorted(feats.native_t, t), 0, len(feats.native_t) - 1))
    # Use pixel proximity ONLY when the ball is airborne — that is exactly when
    # the court projection is unreliable. When it's low, the court-metre distance
    # is both accurate and more discriminating (a fixed pixel radius is more
    # permissive and costs precision on clips the court metric already handles).
    h_i = td.ball_height[i] if (getattr(td, "ball_height", None) is not None
                                and np.isfinite(td.ball_height[i])) else None
    airborne = h_i is not None and h_i > cfg.serve_airborne_h_m
    ball_px = (td.ball_px[i] if (cfg.serve_contact_use_pixels and airborne
                                 and getattr(td, "ball_px", None) is not None
                                 and np.isfinite(td.ball_px[i]).all()) else None)
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
        box = td.player_box.get(slot) if hasattr(td, "player_box") else None
        if ball_px is not None and box is not None and np.isfinite(box[i]).all():
            d_px = _point_box_dist_px(ball_px, box[i])
            near = np.clip(1 - d_px / cfg.serve_contact_near_px, 0, 1)
        else:
            near = np.clip(1 - max(0.0, float(np.hypot(*(xy - p))) - 0.3)
                           / cfg.serve_contact_near_m, 0, 1)
        spd = feats.player_speed.get(slot)
        v = spd[i] if spd is not None and np.isfinite(spd[i]) else 0.0
        still = np.clip(1 - v / (2 * cfg.stationary_mps), 0, 1)
        fit = behind * near * (0.5 + 0.5 * still)
        if fit > best:
            best, best_y = float(fit), float(p[1])
    return best, best_y


def _formation_window(feats: Features, t0: float, t1: float,
                      cfg: RefineConfig) -> tuple[float, float] | None:
    """Time range in [t0,t1] correlated with pre-serve formation, used to
    constrain the contact-candidate search so contact can't latch onto a
    stray high-speed ball event that has nothing to do with THIS point (e.g.
    a hard shot from the tail of the PRIOR point, sitting before this
    point's server ever got into formation). Anchors on the FIRST moment
    formation becomes strong in [t0,t1] and only trims the window from that
    side — deliberately does NOT also clamp the right/late side, since a
    real rally's players often stay in generic behind-baseline positions well
    past contact (formation_score has no "point started" signal beyond
    court position/stationarity), and over-trimming there would exclude
    legitimate late-detected contacts. Returns None if formation was never
    strong enough in [t0,t1] to localize anything — callers should widen
    back to the full [t0,t1] then (nothing to localize against, not "no
    serve here")."""
    m = (feats.native_t >= t0) & (feats.native_t <= t1)
    if not m.any():
        return None
    idx = np.flatnonzero(m)
    strong = feats.formation_held[idx] >= cfg.serve_pre_serve_min
    if not strong.any():
        return None
    first_t = float(feats.native_t[idx[np.flatnonzero(strong)[0]]])
    return max(t0, first_t - 0.5), t1


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


def _ballistic_contact(feats: Features, seg_start: float, t1: float,
                       cfg: RefineConfig) -> tuple[float, float, float | None, np.ndarray] | None:
    """Recover the serve contact the tracker never saw. The ball is untrackable
    at the strike (small, fused with the body), so its first detection is already
    mid-flight; but the fitted gravity arc extrapolates BACKWARD to where the ball
    came from. Walk the serve's arc back frame-by-frame and score proximity to a
    stationary at/behind-baseline player (in correct 3D court metres). Returns
    (fit, contact_t, server_y, xy) or None."""
    td = feats.td
    flights = getattr(td, "ball_flights", None)
    if not flights:
        return None
    # The serve's outgoing arc = the earliest flight near the segment start. Note
    # the serve begins BEFORE the smoothed segment edge (the rising edge lags the
    # real start), so the search window opens serve_lookback_s early, not at it.
    cand = None
    for fl in flights:
        t_fs = fl.f_start / td.fps
        if seg_start - cfg.serve_lookback_s <= t_fs <= t1 \
                and (cand is None or fl.f_start < cand.f_start):
            cand = fl
    if cand is None:
        return None
    # The serve signature in the extrapolated arc: it ORIGINATES low (a struck
    # ball leaves the paddle near the ground) and at/behind a baseline. Player
    # proximity is a soft bonus, not a gate — the extrapolated x carries some
    # uncertainty and the server stands off to a side.
    m = cfg.behind_baseline_m
    best = None
    for back in range(1, int(cfg.serve_flight_max_s * td.fps) + 1):
        f = cand.f_start - back
        if f < 0:
            break
        t = f / td.fps
        if t < seg_start - cfg.serve_lookback_s:
            break
        x, y, h = cand.eval(f)
        if not (-0.2 <= h <= 1.2):                  # contact is low; skip airborne
            continue
        if not (-1.0 <= x <= COURT_W + 1.0):        # off the court sideways
            continue
        behind = max(np.clip(1 - y / m, 0, 1), np.clip(1 - (COURT_L - y) / m, 0, 1))
        if behind < 0.5:                            # not at/behind a baseline
            continue
        low = np.clip(1 - abs(h - 0.4) / 0.9, 0, 1)  # ideal strike ~0.4 m
        fit, server_y = _server_fit(feats, t, (x, y), cfg)
        prox = np.clip(fit / 0.3, 0, 1)
        score = behind * (0.55 + 0.30 * low + 0.15 * prox)
        if server_y is None:                        # infer serving end from origin
            server_y = 0.0 if y < COURT_L / 2 else COURT_L
        if best is None or score > best[0]:
            best = (float(score), t, float(server_y), np.array([x, y]))
    return best


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
    # diagnostic only (not used in scoring): how much of that formation score
    # came from real shoulder-pose data vs. the neutral 0.5 orientation
    # fallback — a `pre` value built mostly on the fallback isn't actually
    # vetoing badly-oriented players, see formation_score's docstring.
    pre_pose_coverage = (float(feats.formation_pose_native[m].mean())
                         if m.any() and len(feats.formation_pose_native) else 0.0)

    # -- phase 2: contact ---------------------------------------------------
    # Search only the window correlated with strong formation being held
    # (falls back to the full [t0,t1] lookback if formation was never
    # strong there) — otherwise a stray high-speed ball event anywhere in the
    # lookback window (a hard shot from the prior point's tail) can outscore
    # the real, weaker-looking serve contact.
    fw = _formation_window(feats, t0, t1, cfg)
    ct0, ct1 = fw if fw is not None else (t0, t1)
    best = None                      # (combined, t, server_y, xy)
    for t, xy, spd in _contact_candidates(feats, ct0, ct1, cfg):
        fit, server_y = _server_fit(feats, t, xy, cfg)
        speed_q = np.clip(spd / (2 * cfg.serve_contact_speed_mps), 0, 1)
        q = fit * (0.4 + 0.6 * speed_q)
        if fit > 0.15 and (best is None or q > best[0]):
            best = (float(q), t, server_y, xy)
    contact = best[0] if best else 0.0
    contact_t = best[1] if best else None
    server_y = best[2] if best else None

    # Fallback: the strike is untrackable (ball fused with the body), so no
    # detected candidate sits at the server. Recover it from the serve arc,
    # extrapolated back to where the ball originated. Slightly discounted vs a
    # directly-observed contact since it is inferred, not seen.
    if best is None:
        bc = _ballistic_contact(feats, seg_start, t1, cfg)
        if bc is not None:
            contact = 0.85 * bc[0]
            contact_t, server_y = bc[1], bc[2]

    # -- phase 3: flight ------------------------------------------------------
    # A confirmed net crossing shortly after contact, at height, is a hard
    # requirement (see flight_or_landing_ok below) — contact + rally_evidence
    # alone let non-serve activity (a rolling ball being fetched, a carried
    # ball) through purely on ambient rhythm. Height check: a served ball
    # crossing the net is genuinely airborne; a ball rolling on the ground
    # tracks near h=0, so require the ball above serve_flight_min_height_m at
    # the crossing when height data is available (ballistic 3D reconstruction
    # off/failed => no height signal => don't penalize, can't verify).
    anchor = contact_t if contact_t is not None else seg_start
    flight = 0.0
    for ct, cd in zip(feats.crossings_t, feats.crossings_dir):
        if cfg.serve_flight_min_s <= ct - anchor <= cfg.serve_flight_max_s:
            h = _ball_height_at(feats, ct)
            if h is not None and h < cfg.serve_flight_min_height_m:
                continue          # ground-level "crossing" — not a real flight, keep scanning
            flight = 0.8
            if server_y is not None:
                # serve flies away from the server's end
                expected = 1.0 if server_y < NET_Y else -1.0
                if cd == expected:
                    flight = 1.0
            break

    # -- phase 4: landing -------------------------------------------------------
    # landing=1.0 means a bounce CONFIRMED beyond the kitchen on the
    # receiver's diagonal half (strong evidence); landing=0.3 means some
    # nearby bounce existed but wasn't clearly beyond the kitchen (weak/
    # ambiguous — NOT sufficient on its own for the hard gate below, since
    # this is exactly what a rolling-ball false positive can produce).
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
    # Hard gate:
    #  1. flight_or_landing_ok is ALWAYS required — EITHER a confirmed
    #     beyond-kitchen landing OR a DIRECTIONALLY-CONFIRMED flight crossing
    #     (flight == 1.0: the crossing matched the expected direction away
    #     from the server's side, not merely "some crossing was found" —
    #     flight == 0.8 is undirected and, on its own, is exactly what a
    #     stray/ambient net crossing unrelated to this point can produce).
    #     Rules out contact + rally_evidence carrying a false positive on
    #     ambient rhythm alone (the module docstring's original confirmed
    #     false-positive case), and closes a second one where undirected
    #     flight (0.8) plus marginal pre_serve/contact still slipped through
    #     (confirmed false positive: flight=0.8, landing=0, pre_serve=0.40,
    #     contact=0.68).
    #  2. PLUS one of: contact_ok, pre_serve_compensate_ok, or (near-floor
    #     contact + directionally-confirmed flight + base pre_serve floor).
    #     Ball-near-player detection is often weak/occluded right at contact
    #     even for real serves, so a strict AND let one noisy metric veto an
    #     otherwise well-evidenced serve (two confirmed false negatives:
    #     pre_serve 0.62/0.36 with flight/landing confirmed, dropped only
    #     because contact read 0.20/0.28, just under the 0.3 floor). Strong
    #     formation + confirmed flight/landing can compensate for a weak
    #     contact read; separately, a directionally-confirmed flight crossing
    #     is itself strong enough evidence that a near-floor (not weak) contact
    #     read plus a merely-base pre_serve shouldn't kill the serve (confirmed
    #     false negative: flight=1.0, landing=0, pre_serve=0.361, contact=0.2815
    #     — missed the 0.3 contact floor by a small margin).
    flight_directional = flight >= 1.0
    flight_or_landing_ok = landing >= 1.0 or flight_directional
    pre_serve_ok = pre >= cfg.serve_pre_serve_min                    # diagnostic only
    contact_ok = contact >= cfg.serve_contact_min
    contact_near_floor_ok = contact >= cfg.serve_contact_near_floor_min
    pre_serve_compensate_ok = pre >= cfg.serve_pre_serve_compensate_min
    pass_path = None
    if flight_or_landing_ok and contact_ok:
        pass_path = "contact"
    elif flight_or_landing_ok and pre_serve_compensate_ok:
        pass_path = "formation_compensation"
    elif flight_directional and contact_near_floor_ok and pre_serve_ok:
        pass_path = "directional_flight_compensation"
    gate_ok = pass_path is not None
    if not gate_ok:
        score = 0.0
    return ServeGateResult(score=float(score), contact_t=contact_t,
                           components=comps,
                           flight_or_landing_ok=flight_or_landing_ok,
                           flight_directional=flight_directional,
                           contact_ok=contact_ok,
                           contact_near_floor_ok=contact_near_floor_ok,
                           pre_serve_ok=pre_serve_ok,
                           pre_serve_compensate_ok=pre_serve_compensate_ok,
                           pass_path=pass_path,
                           pre_serve_pose_coverage=pre_pose_coverage)


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

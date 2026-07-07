"""Rally start / end decision logic (Pass 2).

Start: serve-receive formation + first fast ball motion originating behind a
baseline.

End: the frame the ball becomes UNPLAYABLE — a terminal ball event (bounce
out, double bounce, net fail), confirmed (never timed) by player behaviour.

Hard rules encoded here:
- The recorded end frame is the terminal-event frame, not when the ball
  stops or players reset.
- Low ball speed / low player motion NEVER ends a rally on its own (protects
  dink exchanges).
- Ball-track dropout while players are still moving actively is a tracking
  gap, not an end.

Out of scope in v1 (stated in the README, not faked): faults with no
distinctive ball trajectory — foot faults, kitchen violations, service-motion
faults.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import court
from .bounce import BallEvent, BallWorld


@dataclass
class RallyDecision:
    start_frame: int | None
    end_frame: int | None
    end_event: dict | None = None      # terminal event details for overlays
    confirmed: bool = False
    notes: list[str] = field(default_factory=list)
    start_source: str = ""             # 'serve' | 'fast_ball' | 'coarse_gate'
    end_estimate: int | None = None    # best end guess when no terminal event fired
    end_estimate_source: str = ""      # 'last_cross' | 'player_reset'


def _dist_outside_court(x: float, y: float) -> float:
    """Metres a court-plane point lies outside the playing rectangle (0 if
    inside). Used to reject phantom 'out' bounces whose reconstructed
    position is implausibly far past the lines."""
    dx = max(0.0, -x, x - court.COURT_W)
    dy = max(0.0, -y, y - court.COURT_L)
    return float((dx * dx + dy * dy) ** 0.5)


def _find_start(
    world: list[BallWorld],
    serve_formation_frames: list[int],
    params: dict,
    fps: float,
    window_start: int,
    require_serve: bool,
    model=None,
) -> tuple[int | None, str]:
    """First fast ball motion that originates behind either baseline, with a
    serve-formation sighting shortly before it. Returns (frame, source):
      'serve'       — confirmed serve (trustworthy for boundary learning),
      'fast_ball'   — a real ball motion but no confirmed serve,
      'coarse_gate' — window-edge fallback (no ball evidence).
    If require_serve is set, only a confirmed serve is returned, else (None, "")
    — this is how the multi-rally loop learns there are no more rallies here.

    `model` (a trained serve_start model) is threaded from the registry/Pass 2
    for a future increment; it is None today, so the heuristic below runs."""
    if not world:
        return (None, "")
    speed_th = params.get("serve_speed_mps", 4.0) / fps  # metres per frame
    behind = params.get("serve_behind_margin_m", 1.5)
    look_back = int(fps * params.get("serve_formation_lookback_s", 4.0))

    xs = np.array([w.court_x for w in world])
    ys = np.array([w.court_y for w in world])
    v = np.hypot(np.gradient(xs), np.gradient(ys))
    for i in range(1, len(world)):
        w = world[i]
        if v[i] < speed_th or not w.observed:
            continue
        near_serve_zone = w.court_y > court.COURT_L - behind or w.court_y < behind
        if not near_serve_zone:
            continue
        has_formation = bool(serve_formation_frames) and any(
            0 <= w.frame - f <= look_back for f in serve_formation_frames)
        if serve_formation_frames and not has_formation:
            continue
        # A SUBSEQUENT rally in the same window must be backed by a real
        # serve-formation sighting, not merely a fast ball behind the line
        # (which could be players resetting between points). This is what keeps
        # multi-rally splitting from inventing false rallies out of dead time.
        if require_serve and not has_formation:
            continue
        return (w.frame, "serve")
    if require_serve:
        return (None, "")
    # Fallback: first sustained fast ball motion in the window at all.
    for i in range(1, len(world)):
        if v[i] >= speed_th and world[i].observed:
            return (world[i].frame, "fast_ball")
    return (int(window_start), "coarse_gate")


def detect_start(
    world: list[BallWorld],
    serve_formation_frames: list[int],
    params: dict,
    fps: float,
    window_start: int,
    model=None,
) -> int | None:
    """Single start with the window-edge fallback (backward-compatible wrapper).
    Multi-rally Pass 2 uses _find_start via detect_rallies."""
    return _find_start(world, serve_formation_frames, params, fps, window_start,
                       require_serve=False, model=model)[0]


def detect_end(
    events: list[BallEvent],
    world: list[BallWorld],
    player_motion: dict[int, float],   # frame -> aggregate player speed (m/s)
    params: dict,
    fps: float,
    start_frame: int,
) -> RallyDecision:
    decision = RallyDecision(start_frame=start_frame, end_frame=None)
    net_low = params.get("net_fail_height_m", 0.95)
    confirm_win = int(fps * params.get("confirm_window_s", 2.5))
    active_th = params.get("player_active_mps", 1.2)

    # A rally cannot end before it has meaningfully begun — terminal events
    # inside the minimum rally duration are tracker noise, not endings.
    min_len = int(fps * params.get("p1_min_rally_s", 2.0))
    evs = [e for e in events if e.frame > start_frame + min_len]

    def confirmed_by_players(frame: int) -> bool:
        """Confirmation/veto layer — lagging and noisy, so it can only
        approve or veto a terminal event, never set the clock."""
        speeds = [s for f, s in player_motion.items() if frame < f <= frame + confirm_win]
        if not speeds:
            return True  # no data -> do not veto
        # Players breaking from athletic movement after the event = confirm.
        return float(np.mean(speeds)) < active_th * 1.5

    def finish(frame: int, event: BallEvent | None, why: str) -> RallyDecision:
        # Single choke-point guard for EVERY terminal path (bounce_out /
        # double_bounce / net_fail): a terminal event within the minimum rally
        # duration is tracker noise, not a real ending. The far-side ball
        # reconstruction occasionally fires a spurious bounce/net_fail moments
        # after the serve; without this, that phantom becomes a sub-second
        # "rally" that displaces the real one (observed: a 0.5 s net_fail blip
        # dropped a genuine rally and made Pass 2 score worse than the Pass 1
        # coarse window it is meant to refine). Rejecting it here — end_frame
        # stays None, so the caller keeps scanning or falls back to coarse_gate
        # — guarantees Pass 2 is never worse than Pass 1 on this failure mode.
        if frame - start_frame < min_len:
            decision.notes.append(
                f"terminal event '{why}' at frame {frame} ignored: shorter than "
                f"minimum rally ({params.get('p1_min_rally_s', 2.0):.1f} s) — treated as noise")
            return decision
        decision.end_frame = frame
        decision.end_event = None if event is None else {
            "kind": event.kind, "frame": event.frame, "why": why,
            "court_x": round(event.court_x, 3), "court_y": round(event.court_y, 3),
            "in_court": event.in_court, "side": event.side,
        }
        decision.confirmed = confirmed_by_players(frame)
        if not decision.confirmed:
            decision.notes.append(
                f"terminal event '{why}' at frame {frame} vetoed: players still active")
            decision.end_frame = None
            decision.end_event = None
        return decision

    # --- primary terminal events, in temporal order -----------------------
    last_bounce: BallEvent | None = None
    for e in evs:
        if e.kind == "hit":
            last_bounce = None  # paddle contact resets the double-bounce count
            continue
        # bounce
        if not e.in_court:
            # An "out" call is high-stakes and the far-side ball reconstruction
            # is the least reliable signal in the system, so guard it hard —
            # these three checks are exactly what stops clearly-in balls from
            # being marked out:
            #  1. must be a REAL detection at this frame, not a Kalman-bridged
            #     phantom (bridged points invent positions).
            #  2. height must have been reconstructed (no height => we can't
            #     know it actually bounced on the ground vs flew past).
            #  3. must land in a tight band just outside a line. A real out
            #     ball lands within ~out_max_dist_m of the line; a "bounce"
            #     several metres beyond the court is size->depth noise.
            out_max = params.get("out_max_dist_m", 1.5)
            outside = _dist_outside_court(e.court_x, e.court_y)
            if not e.observed or not e.height_known or outside > out_max:
                decision.notes.append(
                    f"discarded weak out-signal at frame {e.frame} "
                    f"(observed={e.observed}, height_known={e.height_known}, "
                    f"{outside:.1f}m outside)")
                continue
            # A ball crossing the sideline in the air is NOT out — this only
            # fires on an actual bounce outside the polygon, untouched.
            d = finish(e.frame, e, "bounce_out")
            if d.end_frame is not None:
                return d
            continue
        if last_bounce is not None and last_bounce.side == e.side:
            d = finish(e.frame, e, "double_bounce")
            if d.end_frame is not None:
                return d
        last_bounce = e

    # --- net fail: ball reaches the net line low and dies / fails to cross -
    by_frame = {w.frame: w for w in world}
    frames = sorted(by_frame)
    for i, f in enumerate(frames[:-3]):
        w = by_frame[f]
        # Mirror the bounce/double-bounce pre-filter (evs above): skip the
        # first min_len frames so net_fail can't fire before the rally has
        # meaningfully begun. finish() enforces the same invariant, but
        # bounding the scan here avoids building fake events that get rejected.
        if w.frame <= start_frame + min_len or np.isnan(w.height):
            continue
        at_net = abs(w.court_y - court.NET_Y) < 0.6 and w.height < net_low
        if not at_net:
            continue
        after = [by_frame[g] for g in frames[i + 1:i + 1 + int(fps * 0.5)]]
        if not after:
            continue
        going = np.sign(w.court_y - court.NET_Y)  # which side it approached from
        crossed = any(np.sign(a.court_y - court.NET_Y) != going and
                      abs(a.court_y - court.NET_Y) > 0.3 for a in after)
        if not crossed:
            fake = BallEvent(frame=w.frame, kind="bounce", court_x=w.court_x,
                             court_y=w.court_y, height=w.height, in_court=True,
                             side=court.side_of_net(w.court_y), near_player=False)
            d = finish(w.frame, fake, "net_fail")
            if d.end_frame is not None:
                return d

    # --- no terminal event found ------------------------------------------
    # The far-side terminal bounce is the least reliable signal in the system,
    # so ~all rallies reach here and Pass 2 otherwise falls back to the padded
    # Pass 1 window edge — which trails the true end by several seconds. Produce
    # a much closer estimate WITHOUT needing the far ball:
    #   1. last net crossing — in pickleball the point ends just after the last
    #      shot goes over the net; that crossing rides on the near-side ball,
    #      which tracks well (unlike the far-side terminal bounce). End ~= last
    #      crossing + the ball's short flight-to-land time.
    #   2. player reset — if no crossing was tracked, the last frame players were
    #      at rally intensity, + a short settle, marks the point breaking up.
    # This is only a boundary ESTIMATE (end_source stays non-terminal), so the
    # honest 'no terminal event' semantics are preserved for the learning harness.
    last_cross = _last_net_crossing(world, start_frame)
    if last_cross is not None:
        decision.end_estimate = int(min(last_cross + int(fps * 0.6),
                                        world[-1].frame if world else last_cross))
        decision.end_estimate_source = "last_cross"
    else:
        active = [f for f, s in player_motion.items()
                  if f > start_frame + min_len and s > active_th]
        if active:
            decision.end_estimate = int(min(max(active) + int(fps * 0.3),
                                            world[-1].frame if world else max(active)))
            decision.end_estimate_source = "player_reset"

    # Track dropout with players still moving = tracking gap, never an end.
    if world:
        tail = world[-int(fps):]
        tail_unobserved = sum(1 for w in tail if not w.observed) > len(tail) * 0.6
        recent_speed = [s for f, s in player_motion.items() if f >= world[-1].frame - fps]
        if tail_unobserved and recent_speed and float(np.mean(recent_speed)) > active_th:
            decision.notes.append("ball track dropout while players active — tracking gap, no end declared")
    return decision


def _last_net_crossing(world: list[BallWorld], start_frame: int,
                       band: float = 0.5) -> int | None:
    """Frame of the LAST clean net crossing after start_frame (hysteresis `band`
    metres to ignore jitter at the net), or None if the ball never crossed. The
    point ends just after the last shot goes over, so this + flight time is a
    good end estimate that doesn't depend on the far-side terminal bounce."""
    committed = None
    last = None
    for w in world:
        if w.frame < start_frame or not w.observed:
            continue
        dy = w.court_y - court.NET_Y
        if abs(dy) < band:
            continue
        side = 1 if dy > 0 else -1
        if committed is None:
            committed = side
        elif side != committed:
            last = w.frame
            committed = side
    return last


def _was_played(world: list[BallWorld], start_frame: int, band: float = 0.5) -> bool:
    """Was this candidate an actual played point rather than a between-point
    pass? The RELIABLE signal is whether the ball crossed the net after the
    serve (a hysteresis `band` in metres ignores jitter near the net):
      0 crossings  -> the ball never went over the net (a ball tossed/carried to
                      a partner, or noise) -> NOT a rally.
      >=1 crossing -> kept. A real rally with poor far-side tracking can lose its
                      return crossing, and dropping it would be a MISS (worse
                      than a stray extra).
    'hit' events are deliberately NOT used here: near a held/stationary ball they
    are body/paddle noise (observed: 13 fake hits with zero net crossings), which
    is exactly what let a caught ball survive as a serve. Cleanly separating a
    1-crossing pass from a 1-crossing rally needs trustworthy ball tracking
    (fine-tuned model)."""
    committed = None
    crossings = 0
    for w in world:
        if w.frame < start_frame or not w.observed:
            continue
        dy = w.court_y - court.NET_Y
        if abs(dy) < band:
            continue
        side = 1 if dy > 0 else -1
        if committed is None:
            committed = side            # the serving side
        elif side != committed:
            crossings += 1              # a clear net crossing
            committed = side
    return crossings >= 1


def detect_rallies(
    events: list[BallEvent],
    world: list[BallWorld],
    serve_formation_frames: list[int],
    player_motion: dict[int, float],
    params: dict,
    fps: float,
    window_start: int,
    max_rallies: int = 4,
    start_model=None,
) -> list[RallyDecision]:
    """Emit EVERY rally inside a Pass 1 window, not just one: find a serve, find
    its terminal end, then search for the next serve AFTER that end, and repeat.

    This makes Pass 2 robust to Pass 1 UNDER-segmentation (two rallies that
    never went 'dead' between them, so they arrived as one window). The first
    rally may fall back to the window edge; every subsequent rally requires a
    CONFIRMED serve, so dead time after a rally never becomes a phantom rally.
    The caller applies the coarse-gate end to the final decision when it has no
    terminal event (only the last decision in a window can lack an end)."""
    reject_passes = params.get("p2_reject_passes", 1.0) > 0.5
    decisions: list[RallyDecision] = []
    cursor = window_start
    for idx in range(max_rallies):
        sub_world = [w for w in world if w.frame >= cursor]
        if len(sub_world) < 3:
            break
        start, ssrc = _find_start(sub_world, serve_formation_frames, params, fps,
                                  cursor, require_serve=idx > 0, model=start_model)
        if start is None:
            break
        sub_events = [e for e in events if e.frame > start]
        decision = detect_end(sub_events, sub_world, player_motion, params, fps, start)
        decision.start_source = ssrc
        # A candidate is a real rally only if the ball was actually played: it
        # ended in a fault (a terminal event) OR the ball crossed the net. A
        # ball that never crosses the net (tossed/carried to a partner, or noise)
        # is a between-point PASS, not a serve — drop it and stop this window.
        if (reject_passes and decision.end_frame is None
                and not _was_played(sub_world, start)):
            break
        decisions.append(decision)
        if decision.end_frame is None:
            break                        # last rally; caller sets the coarse-gate end
        cursor = decision.end_frame + 1
    return decisions

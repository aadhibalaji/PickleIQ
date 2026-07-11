"""Trajectory-consistency ball selection (Pass 2).

Motion — NOT appearance — is the primary defense against foot/shoe false
positives. In a single static frame a white shoe and a pickleball look alike:
small, round-ish, bright. Over a few frames they are nothing alike. A ball in
flight follows a smooth, fast, gravity-consistent arc; a foot oscillates
slowly, tethered to a body, and never sustains ball-like speed.

`TrajectoryBallSelector` takes the raw motion-blob candidates for one frame and
picks the one that best CONTINUES the recent ball trajectory, rejecting blobs
that don't fit. Two suppression signals stack on top of the trajectory fit:

  * player boxes / foot keypoints (strategy 3): a candidate sitting on a
    player's box or right on a foot keypoint is suspect. It is trusted only if
    it is moving with ball-like speed — i.e. it just left a paddle (a hit) or
    is a ball passing over the player mid-flight — otherwise it is a foot and
    is dropped.
  * cold-start guard: with no established trajectory yet, we refuse to seed the
    track from a blob on a player/foot, so a stationary shoe can never become
    the ball on frame one.

Frames where nothing plausible is found return None; the downstream Kalman
smoother bridges short dropouts, so a couple of missed frames near a hit are
harmless.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .ball_coarse import on_player_box


@dataclass
class _Pt:
    frame: int
    x: float
    y: float


def _on_player(x: float, y: float,
               boxes: list[tuple[float, float, float, float]],
               feet: list[tuple[float, float]], foot_r: float) -> bool:
    """True if (x,y) sits on a player box (expanded outward by the shared margins
    in ball_coarse) or within foot_r of a foot/knee keypoint."""
    if on_player_box(x, y, boxes):
        return True
    r2 = foot_r * foot_r
    for fx, fy in feet:
        if (x - fx) ** 2 + (y - fy) ** 2 < r2:
            return True
    return False


class TrajectoryBallSelector:
    def __init__(self, fps: float, gate_px: float = 45.0, accel_px: float = 6.0,
                 foot_px: float = 35.0, hit_speed_px: float = 18.0,
                 acq_px: float = 90.0, seed_min_px: float = 4.0,
                 max_gap_s: float = 0.5, seed_on_player: bool = False,
                 suppress_on_player: bool = True,
                 lock_dist_px: float = 30.0, lock_rel_vel_px: float = 4.0,
                 lock_frames: int = 3):
        self.fps = fps
        self.gate = float(gate_px)          # residual radius around the velocity prediction
        self.accel = float(accel_px)        # extra radius per frame^2 of gap
        self.foot = float(foot_px)          # foot-keypoint suppression radius
        self.hit_speed = float(hit_speed_px)  # px/frame that overrides foot suppression
        self.acq_px = float(acq_px)         # acquisition radius while velocity is unknown
        self.seed_min = float(seed_min_px)  # min px/frame a cold-start seed must move
        # Allow cold-start acquisition to seed from a blob ON a player, provided
        # it still passes the two-frame ball-speed motion match. Recovers the
        # SERVE ball (struck at the server, so on their box + the first ball of
        # the rally = cold start) that the off-player-only pool otherwise hides
        # until it is already mid-court. A stationary foot fails the motion match.
        self.seed_on_player = bool(seed_on_player)
        # When False, candidates on a player box / foot are NOT suppressed in the
        # established-trajectory path (select()) or the cold-start pool. The weak
        # classical detector's few blobs cluster near players on some clips, so
        # suppression can reject the entire track — loosen it to recover a
        # (noisier) track when a stronger detector isn't available. Independent of
        # seed_on_player above: that one *permits* an on-player cold-start seed
        # (still gated on hit_speed); this one *disables* on-player suppression
        # altogether.
        self.suppress_on_player = bool(suppress_on_player)
        # Limb-lock guard: a candidate that stays within lock_dist of a wrist AND
        # shares its frame-to-frame velocity (constant offset / ~zero relative
        # velocity) for lock_frames consecutive frames is a limb/paddle blob, not
        # an airborne ball, and is dropped — regardless of how it was originally
        # acquired (cold-start seed or established-trajectory pick). A ball merely
        # close to a paddle for one contact frame never accumulates the run, so
        # real hits (including a serve seeded on-player via seed_on_player) survive.
        self.lock_dist = float(lock_dist_px)
        self.lock_rel_vel = float(lock_rel_vel_px)
        self.lock_frames = int(lock_frames)
        self._wrist_prev: list = []
        self._wrist_prev_f: int = -10**9
        self._lock_run: int = 0
        self.max_gap = max(1, int(fps * max_gap_s))
        self.hist: deque[_Pt] = deque(maxlen=5)
        # Cold-start acquisition state: the PREVIOUS frame's off-player candidate
        # pool. A track is seeded only when a candidate this frame continues one
        # from the previous frame at a ball-like speed (see _cold_start) — a
        # two-frame motion match, so a stationary/near-stationary shoe never
        # seeds even when it is the roundest blob on screen.
        self._cold_prev: list = []
        self._cold_prev_f: int = -10**9

    def _predict(self, frame: int):
        """(predicted_xy, gate_radius) from the recent track, or (None, None)
        if the track is empty / gone stale (a fresh cold start)."""
        if not self.hist:
            return None, None
        last = self.hist[-1]
        gap = frame - last.frame
        if gap > self.max_gap:
            self.hist.clear()
            return None, None
        if len(self.hist) >= 2:
            # Velocity known: predict forward, gate tightly around it (residual
            # + a little accel slack). A foot metres off the arc is excluded.
            p0, p1 = self.hist[-2], self.hist[-1]
            dtp = max(1, p1.frame - p0.frame)
            vx, vy = (p1.x - p0.x) / dtp, (p1.y - p0.y) / dtp
            pred = (last.x + vx * gap, last.y + vy * gap)
            gate = self.gate + self.accel * gap * gap
        else:
            # Only one point so far: velocity unknown, so we can't predict where
            # the ball went — gate the FULL plausible per-frame ball travel
            # around the last position to acquire the second point (a fast serve
            # can move ~2-3x the residual gate between frames).
            pred = (last.x, last.y)
            gate = self.acq_px * gap
        return pred, gate

    def _fast_from_last(self, frame: int, x: float, y: float) -> bool:
        if not self.hist:
            return False
        last = self.hist[-1]
        dt = max(1, frame - last.frame)
        return math.hypot(x - last.x, y - last.y) / dt >= self.hit_speed

    def _limb_locked(self, frame: int, x: float, y: float, wrists: list) -> bool:
        """True once (x,y) has moved in LOCKSTEP with a nearby wrist for
        lock_frames consecutive frames. Reads the PREVIOUS frame's wrists
        (self._wrist_prev) — select() updates them after. Maintains the run
        counter so a single close-contact frame doesn't trip it."""
        lockstep = False
        if wrists and self.hist:
            wx, wy = min(wrists, key=lambda w: (x - w[0]) ** 2 + (y - w[1]) ** 2)
            if math.hypot(x - wx, y - wy) <= self.lock_dist:
                last = self.hist[-1]
                gap = max(1, frame - last.frame)
                vcx, vcy = (x - last.x) / gap, (y - last.y) / gap  # ball velocity
                prev, pf = self._wrist_prev, self._wrist_prev_f
                if prev and 0 < frame - pf <= self.max_gap:
                    pwx, pwy = min(prev, key=lambda w: (wx - w[0]) ** 2 + (wy - w[1]) ** 2)
                    gw = max(1, frame - pf)
                    vwx, vwy = (wx - pwx) / gw, (wy - pwy) / gw     # wrist velocity
                    if math.hypot(vcx - vwx, vcy - vwy) <= self.lock_rel_vel:
                        lockstep = True
        self._lock_run = self._lock_run + 1 if lockstep else 0
        return self._lock_run >= self.lock_frames

    def select(self, frame: int, cands, boxes, feet, wrists=None):
        """Pick the best ball candidate for this frame (or None). `cands` items
        expose .x, .y, .diameter_px, .score (circularity). `wrists` are player
        wrist pixels for the limb-lock guard (None/empty = guard never trips)."""
        wrists = wrists or []
        try:
            pred, gate = self._predict(frame)
            if pred is None:
                # No established trajectory: acquire only via confirmed motion,
                # never by picking the roundest single-frame blob (seeds on shoes).
                return self._cold_start(frame, cands, boxes, feet)
            best, best_s = None, -1e18
            for c in cands:
                on_player = (self.suppress_on_player
                             and _on_player(c.x, c.y, boxes, feet, self.foot))
                # A blob on a player/foot is only the ball if it's moving fast (a
                # hit or a fly-over). A slow blob on a foot IS a foot — skip it.
                if on_player and not self._fast_from_last(frame, c.x, c.y):
                    continue
                dist = math.hypot(c.x - pred[0], c.y - pred[1])
                if dist > gate:                       # off the predicted arc
                    continue
                s = (gate - dist)                     # trajectory fit dominates
                if s > best_s:
                    best, best_s = c, s
            if best is None:
                self._lock_run = 0
                return None
            if self._limb_locked(frame, best.x, best.y, wrists):
                # Sustained lockstep with a wrist: a limb blob. Drop it and let
                # the Kalman smoother bridge / cold-start reacquire the real ball.
                return None
            self.hist.append(_Pt(frame, best.x, best.y))
            self._reset_cold()
            return best
        finally:
            # Always advance the wrist history one frame (every return path).
            self._wrist_prev, self._wrist_prev_f = wrists, frame

    def _reset_cold(self) -> None:
        self._cold_prev = []
        self._cold_prev_f = -10**9

    def _cold_start(self, frame: int, cands, boxes, feet):
        """Acquire a new track only from a candidate showing ball-like MOTION
        between two consecutive frames. A stationary shoe is rounder than a
        motion-blurred ball, so ranking by circularity is a trap — instead we
        look for a mutually-nearest, ball-speed displacement between the previous
        frame's candidate pool and this frame's (same idea as Pass 1's
        _match_fast). Feet/shorts blobs barely move frame-to-frame and are
        rejected; the ball, moving fast, is the one that matches. Returns the
        confirming candidate on the promotion frame, else None."""
        # Normally the acquisition pool excludes on-player blobs entirely. With
        # seed_on_player we keep them — the two-frame motion match below still
        # rejects a stationary foot, and an on-player seed is additionally held
        # to hit_speed (a served ball is fast; a shuffling foot is not). Likewise
        # if suppress_on_player is disabled (weak-detector fallback) we keep them,
        # with no extra hit_speed bar in that case.
        if self.seed_on_player or not self.suppress_on_player:
            pool = list(cands)
        else:
            pool = [c for c in cands
                    if not _on_player(c.x, c.y, boxes, feet, self.foot)]
        prev, pf = self._cold_prev, self._cold_prev_f
        # Remember this frame's pool for the next frame's match.
        self._cold_prev, self._cold_prev_f = pool, frame
        gap = frame - pf
        if not pool or not prev or not (0 < gap <= self.max_gap):
            return None
        lo, hi = self.seed_min * gap, self.acq_px * gap
        best = None
        best_speed = -1.0
        for ci, c in enumerate(pool):
            # Nearest previous candidate to c.
            dp = [math.hypot(c.x - p.x, c.y - p.y) for p in prev]
            pi = min(range(len(prev)), key=lambda i: dp[i])
            d = dp[pi]
            if not (lo <= d <= hi):               # not a ball-like step
                continue
            p = prev[pi]
            # Mutual-nearest: c must also be the nearest current candidate to p,
            # so we don't pair a foot to a far-off blob by coincidence.
            back = [math.hypot(p.x - c2.x, p.y - c2.y) for c2 in pool]
            if min(range(len(pool)), key=lambda i: back[i]) != ci:
                continue
            speed = d / gap
            # An on-player seed must be clearly ball-fast (a hit), not a shuffling
            # foot — a stricter bar than the off-player seed_min floor.
            if self.seed_on_player and speed < self.hit_speed \
                    and _on_player(c.x, c.y, boxes, feet, self.foot):
                continue
            if speed > best_speed:                # the ball is the fast one
                best_speed, best = speed, (p, c)
        if best is None:
            return None
        p, c = best
        self.hist.clear()
        self.hist.append(_Pt(pf, p.x, p.y))
        self.hist.append(_Pt(frame, c.x, c.y))
        self._reset_cold()
        return c

    def near_track(self, frame: int, x: float, y: float,
                   radius_mult: float = 1.0) -> bool:
        """Whether (x, y) is close enough to the recent track to be a plausible
        ball continuation. Used to gate the classical-only 'last resort' pick so
        it can only FILL near the established trajectory — never paint a random
        round noise blob on the far side of the court when the track is cold."""
        if not self.hist:
            return False
        last = self.hist[-1]
        gap = frame - last.frame
        if gap <= 0 or gap > self.max_gap:
            return False
        return math.hypot(x - last.x, y - last.y) <= self.acq_px * gap * radius_mult

    def seed(self, frame: int, x: float, y: float) -> None:
        """Feed a ball position from another source (manual/external/TrackNet)
        into the trajectory history so continuity holds if sources mix."""
        if self.hist and frame - self.hist[-1].frame > self.max_gap:
            self.hist.clear()
        self.hist.append(_Pt(frame, x, y))
        self._reset_cold()


def _demo() -> None:
    """Self-check for the reconciled cold-start + limb-lock guards:
      1. limb-lock: a candidate glued to a moving wrist is excluded after
         lock_frames; one that only brushes past a wrist once is kept.
      2. seed_on_player + limb-lock compose: a genuine fast serve seeded
         on-player at cold start keeps tracking normally, but if the SAME
         track later degenerates into riding a wrist at constant offset,
         the limb-lock guard (not the cold-start check, which only runs once)
         still catches and drops it.
    """
    from types import SimpleNamespace
    C = lambda x, y: SimpleNamespace(x=float(x), y=float(y), diameter_px=6.0, score=1.0)

    # Case 1: blob rides a wrist with a constant +5px offset -> excluded.
    sel = TrajectoryBallSelector(60.0, lock_frames=3)
    sel.seed(0, 100.0, 100.0)
    sel.seed(1, 105.0, 100.0)          # establish velocity (+5/frame)
    excluded = False
    for f in range(2, 8):
        wx = 100.0 + 5.0 * f           # wrist moving +5/frame
        cand = C(wx + 5.0, 100.0)      # ball glued at constant offset from wrist
        out = sel.select(f, [cand], [], [], wrists=[(wx, 100.0)])
        if out is None:
            excluded = True
            break
    assert excluded, "limb-locked blob was not excluded"

    # Case 2: a real ball that merely crosses a wrist for ONE frame is kept.
    sel2 = TrajectoryBallSelector(60.0, lock_frames=3)
    sel2.seed(0, 0.0, 0.0)
    sel2.seed(1, 20.0, 0.0)            # fast ball, +20/frame
    kept = 0
    for f in range(2, 6):
        cand = C(20.0 * f, 0.0)
        wx = 20.0 * f if f == 3 else 500.0   # wrist only coincides at f==3
        out = sel2.select(f, [cand], [], [], wrists=[(wx, 0.0)])
        if out is not None:
            kept += 1
    assert kept >= 3, f"real ball wrongly excluded (kept={kept})"

    # Case 3: rohan's serve recovery + aadhi's limb-lock composing. Cold start
    # with seed_on_player permits a fast blob ON the server's box to seed the
    # track (the serve); once seeded, if that same blob then degenerates into
    # riding a wrist at a constant offset, the limb-lock guard drops it even
    # though it passed cold-start fine.
    sel3 = TrajectoryBallSelector(60.0, seed_on_player=True, hit_speed_px=18.0,
                                   lock_frames=3)
    server_box = (90.0, 90.0, 130.0, 130.0)   # ball starts inside this box
    # Frame 0: candidate sitting on the server's body (no track yet -> ignored,
    # just fills _cold_prev).
    sel3.select(0, [C(100.0, 100.0)], [server_box], [])
    # Frame 1: same spot fast-forward +25px/frame -> a genuine serve hit,
    # on-player but fast enough to seed via seed_on_player.
    out1 = sel3.select(1, [C(125.0, 100.0)], [server_box], [])
    assert out1 is not None, "fast serve on server's body was not seeded"
    # Frames 2-4: ball continues normally, off player, straight line -> survives.
    out2 = sel3.select(2, [C(150.0, 100.0)], [], [])
    assert out2 is not None, "genuine serve ball wrongly dropped after seeding"
    # Now degrade: the same track starts riding a wrist at a constant offset
    # for lock_frames consecutive frames -> must be rejected by the lock guard.
    locked_out = False
    last_x = 150.0
    for i, f in enumerate(range(3, 8)):
        wx = last_x + 5.0 * (i + 1)
        cand = C(wx + 5.0, 100.0)
        out = sel3.select(f, [cand], [], [], wrists=[(wx, 100.0)])
        if out is None:
            locked_out = True
            break
    assert locked_out, "wrist-riding blob after a seeded serve was not caught by limb-lock"

    print("ball_track limb-lock + seed_on_player demo OK")


if __name__ == "__main__":
    _demo()

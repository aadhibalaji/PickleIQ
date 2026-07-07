"""Semantic filtering layer over raw ball-detector candidates (Pass 2).

Phase 1 of the ball-tracking improvement plan: no new model, no training —
physics plus the (now reliable) player tracks used as semantic context. The
layer is detector-agnostic: it filters the classical motion detector today and
applies unchanged to a TrackNet-family detector later (Phase 2).

Four stages. The first two run ONLINE (per frame, before trajectory
selection); the last two run BATCH (per Pass 2 window, after selection):

1. Velocity gate — reject candidates implying an impossible jump from the
   last confirmed ball position. The max plausible pixel displacement per
   frame is derived from the homography (the local px->metre scale at the
   candidate) and a generous max ball speed (65 mph + headroom), so the gate
   tightens automatically in the far court, where each pixel spans more
   metres and a real ball therefore moves FEWER pixels per frame.

2. Player-box motion logic — the ball is legitimately inside a player box at
   every paddle contact, so boxes are never blanket-suppressed. Instead:
     * candidates in the bottom fraction of a box are feet/shoes -> reject;
     * candidates that sit at the same box-relative position for 3+ frames
       move WITH the player (shoe, hat, held paddle) -> reject;
     * everything else (transits, hit-moment blobs) passes through to the
       trajectory stages, which keep it only if it fits a ball-like arc.

3. Arc segmentation + RANSAC — the per-frame selected track is split into
   per-shot arcs at sharp direction changes and time gaps; each arc gets a
   RANSAC quadratic fit (image-space projectile); off-arc points are dropped
   as outliers and missed frames are filled from the fit (interpolated=True).

4. Hit/bounce classification — a direction change between temporally
   adjacent arcs at/next to a player box is a hit; in open court a bounce; a
   tiny arc fragment with no arc continuity on either side is noise and is
   discarded outright.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import court

MPH_TO_MPS = 0.44704


def _scale_m_per_px(H: np.ndarray, x: float, y: float) -> float:
    """Local metres-per-pixel at image point (x, y), via the homography
    Jacobian (finite differences). Returns the SMALLEST of the two axis
    scales — the generous choice for a speed gate, since it lower-bounds the
    real-world distance a pixel displacement implies."""
    pts = np.array([[x, y], [x + 1.0, y], [x, y + 1.0]], dtype=np.float64)
    c = court.img_to_court(H, pts)
    sx = math.hypot(c[1, 0] - c[0, 0], c[1, 1] - c[0, 1])
    sy = math.hypot(c[2, 0] - c[0, 0], c[2, 1] - c[0, 1])
    return max(1e-6, min(sx, sy))


@dataclass
class RejectStats:
    """Per-window counters of what stage 1-2 rejected (harness/log output)."""
    velocity: int = 0
    foot_zone: int = 0
    persistent: int = 0
    seen: int = 0

    def total(self) -> int:
        return self.velocity + self.foot_zone + self.persistent


class BallFrameFilter:
    """Online per-frame filter: velocity gate + player-box motion logic.

    Call filter() with the frame's candidates and player (key, box) pairs,
    then confirm() with whatever ball position the pipeline finally accepted
    for the frame (any source) so the velocity gate tracks the real ball.
    """

    def __init__(self, H: np.ndarray, fps: float,
                 max_speed_mph: float = 65.0, headroom: float = 1.6,
                 persist_frames: int = 3, persist_match_px: float = 12.0,
                 foot_zone_frac: float = 0.25, max_gap_s: float = 0.5):
        self.H = H
        self.fps = fps
        self.vmax_mps = float(max_speed_mph) * MPH_TO_MPS * float(headroom)
        self.persist_frames = max(2, int(persist_frames))
        self.match_px = float(persist_match_px)
        self.foot_frac = float(foot_zone_frac)
        self.max_gap = max(1, int(fps * max_gap_s))
        self._last: tuple[int, float, float] | None = None
        # Persistence tracks: player key -> [ [rel_x, rel_y, hits, last_frame] ]
        # rel_* are offsets from the player's box CENTRE, so the track follows
        # the player as the box moves — a shoe keeps a stable box-relative
        # position for many frames; a ball transiting the box does not.
        self._tracks: dict[object, list[list[float]]] = {}
        self.stats = RejectStats()

    def confirm(self, frame: int, x: float, y: float) -> None:
        """Feed back the ball position the pipeline accepted for `frame`."""
        self._last = (frame, float(x), float(y))

    def _velocity_ok(self, frame: int, x: float, y: float) -> bool:
        if self._last is None:
            return True
        lf, lx, ly = self._last
        gap = frame - lf
        if gap <= 0 or gap > self.max_gap:
            return True  # track stale — no reference position to gate against
        dist_px = math.hypot(x - lx, y - ly)
        if dist_px < 2.0:
            return True
        # Scale sampled at the segment midpoint: in this axial view the
        # px->metre scale changes a lot between near and far court.
        s = _scale_m_per_px(self.H, (x + lx) / 2.0, (y + ly) / 2.0)
        return dist_px * s <= self.vmax_mps * gap / self.fps

    def _persistent(self, frame: int, key: object,
                    box: tuple[float, float, float, float],
                    x: float, y: float) -> bool:
        """Match the candidate against this player's box-relative persistence
        tracks; True once the same relative spot has been hit persist_frames
        times. The first (persist_frames - 1) sightings pass through — a real
        hit-moment ball is only in the box for a frame or two anyway."""
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        rx, ry = x - cx, y - cy
        tracks = self._tracks.setdefault(key, [])
        # Age out tracks not refreshed for 2 frames (blob flicker tolerance).
        tracks[:] = [t for t in tracks if frame - t[3] <= 2]
        for t in tracks:
            if math.hypot(rx - t[0], ry - t[1]) <= self.match_px:
                t[0], t[1] = rx, ry
                if t[3] < frame:
                    t[2] += 1
                    t[3] = frame
                return t[2] >= self.persist_frames
        tracks.append([rx, ry, 1, frame])
        return False

    def filter(self, frame: int, cands: list,
               players: list[tuple[object, tuple[float, float, float, float]]]
               ) -> list:
        """Return the candidates that survive stages 1-2. `cands` items expose
        .x and .y; `players` are (stable key, box) pairs for this frame."""
        kept = []
        for c in cands:
            self.stats.seen += 1
            hit_box = None
            for key, (x1, y1, x2, y2) in players:
                if x1 <= c.x <= x2 and y1 <= c.y <= y2:
                    hit_box = (key, (x1, y1, x2, y2))
                    break
            if hit_box is not None:
                key, (x1, y1, x2, y2) = hit_box
                if c.y > y2 - self.foot_frac * (y2 - y1):
                    self.stats.foot_zone += 1
                    continue
                if self._persistent(frame, key, (x1, y1, x2, y2), c.x, c.y):
                    self.stats.persistent += 1
                    continue
            if not self._velocity_ok(frame, c.x, c.y):
                self.stats.velocity += 1
                continue
            kept.append(c)
        return kept


# ---------------------------------------------------------------------------
# Stage 3: per-shot arc segmentation + RANSAC fit + interpolation
# ---------------------------------------------------------------------------

@dataclass
class ArcPoint:
    frame: int
    x: float
    y: float
    diameter_px: float | None = None
    interpolated: bool = False


@dataclass
class Arc:
    """One fitted shot arc: quadratic image-space trajectory x(t), y(t) with
    t = frame - f0 (a projectile is quadratic in time; perspective bends the
    curve a little, which the RANSAC tolerance absorbs over a single shot)."""
    f0: int
    f1: int
    cx: np.ndarray  # np.polyfit coeffs, highest power first
    cy: np.ndarray
    n_inliers: int = 0

    def xy_at(self, f: int | float) -> tuple[float, float]:
        t = float(f) - self.f0
        return float(np.polyval(self.cx, t)), float(np.polyval(self.cy, t))

    def vel_at(self, f: int | float) -> tuple[float, float]:
        t = float(f) - self.f0
        return (float(np.polyval(np.polyder(self.cx), t)),
                float(np.polyval(np.polyder(self.cy), t)))


@dataclass
class RefinedTrack:
    points: list[ArcPoint]          # surviving REAL observations, frame-sorted
    interpolated: list[ArcPoint]    # fit-filled misses (interpolated=True)
    arcs: list[Arc]                 # fitted arcs, frame-sorted
    n_outliers: int = 0             # real points dropped by the RANSAC fits
    n_fragments: int = 0            # isolated arclet points discarded as noise


def _segment(obs: list[tuple], gap_frames: int, split_deg: float,
             min_speed_px: float) -> list[list[tuple]]:
    """Split (frame, x, y, diam) points into per-shot segments at time gaps
    and sharp direction changes. The turning point closes the outgoing
    segment AND opens the next one, so both arcs stay anchored at the hit."""
    if not obs:
        return []
    cos_split = math.cos(math.radians(split_deg))
    segs: list[list[tuple]] = [[obs[0]]]
    for i in range(1, len(obs)):
        prev, cur = obs[i - 1], obs[i]
        if cur[0] - prev[0] > gap_frames:
            segs.append([cur])
            continue
        seg = segs[-1]
        if len(seg) >= 2:
            p0 = seg[-2]
            dt1 = max(1, prev[0] - p0[0])
            dt2 = max(1, cur[0] - prev[0])
            v1 = ((prev[1] - p0[1]) / dt1, (prev[2] - p0[2]) / dt1)
            v2 = ((cur[1] - prev[1]) / dt2, (cur[2] - prev[2]) / dt2)
            s1 = math.hypot(*v1)
            s2 = math.hypot(*v2)
            if s1 >= min_speed_px and s2 >= min_speed_px:
                cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (s1 * s2)
                if cosang < cos_split:  # sharp turn at `prev` -> new shot
                    segs.append([prev, cur])
                    continue
        seg.append(cur)
    return segs


def _ransac_quadratic(seg: list[tuple], tol_px: float,
                      rng: np.random.Generator, iters: int = 60
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC quadratic fit of one segment. Returns (cx, cy, inlier_mask)."""
    t = np.array([p[0] - seg[0][0] for p in seg], dtype=np.float64)
    xs = np.array([p[1] for p in seg], dtype=np.float64)
    ys = np.array([p[2] for p in seg], dtype=np.float64)
    n = len(seg)
    deg = 2 if n >= 3 else 1

    def residuals(cx, cy):
        return np.hypot(xs - np.polyval(cx, t), ys - np.polyval(cy, t))

    if n <= 4:
        cx = np.polyfit(t, xs, deg)
        cy = np.polyfit(t, ys, deg)
        return cx, cy, residuals(cx, cy) <= tol_px

    best_mask = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        try:
            cx = np.polyfit(t[idx], xs[idx], 2)
            cy = np.polyfit(t[idx], ys[idx], 2)
        except np.linalg.LinAlgError:
            continue
        mask = residuals(cx, cy) <= tol_px
        cnt = int(mask.sum())
        if cnt > best_count:
            best_count, best_mask = cnt, mask
        if cnt == n:
            break
    if best_mask is None or best_mask.sum() < 3:
        cx = np.polyfit(t, xs, deg)
        cy = np.polyfit(t, ys, deg)
        return cx, cy, residuals(cx, cy) <= tol_px
    # Refit on the consensus set, then re-score all points against it.
    cx = np.polyfit(t[best_mask], xs[best_mask], 2)
    cy = np.polyfit(t[best_mask], ys[best_mask], 2)
    return cx, cy, residuals(cx, cy) <= tol_px


def refine_track(obs: list[tuple], *, split_deg: float = 55.0,
                 min_speed_px: float = 1.5, gap_frames: int = 10,
                 tol_px: float = 6.0, min_arc_pts: int = 4,
                 join_gap: int = 8, seed: int = 12345) -> RefinedTrack:
    """Stage 3 over one window's selected track. `obs` is the frame-sorted
    (frame, x, y, diameter_px|None) list the per-frame selector produced.

    Segments below min_arc_pts points survive only if they have arc
    continuity — an accepted arc within join_gap frames on either side (the
    stub of a shot the detector barely caught near a hit); isolated fragments
    are discarded as noise. Fitted arcs drop their RANSAC outliers and fill
    every missed integer frame inside their span from the fit."""
    out = RefinedTrack(points=[], interpolated=[], arcs=[])
    if not obs:
        return out
    rng = np.random.default_rng(seed)
    segs = _segment(obs, gap_frames, split_deg, min_speed_px)

    accepted: list[tuple[list[tuple], Arc]] = []
    fragments: list[list[tuple]] = []
    for seg in segs:
        if len(seg) < max(3, min_arc_pts):
            fragments.append(seg)
            continue
        _, _, mask = _ransac_quadratic(seg, tol_px, rng)
        inliers = [p for p, m in zip(seg, mask) if m]
        out.n_outliers += len(seg) - len(inliers)
        if len(inliers) < min_arc_pts:
            fragments.append(inliers if inliers else seg)
            continue
        # Final fit on the inliers with t=0 anchored at the FIRST INLIER, the
        # origin xy_at/vel_at evaluate against (the RANSAC fit's origin was
        # the segment start, which an outlier may have occupied).
        f0 = inliers[0][0]
        ti = np.array([p[0] - f0 for p in inliers], dtype=np.float64)
        arc = Arc(f0=f0, f1=inliers[-1][0],
                  cx=np.polyfit(ti, [p[1] for p in inliers], 2),
                  cy=np.polyfit(ti, [p[2] for p in inliers], 2),
                  n_inliers=len(inliers))
        accepted.append((inliers, arc))

    accepted.sort(key=lambda ia: ia[1].f0)
    arcs = [a for _, a in accepted]

    # Fragments keep their points only with temporal continuity to a real arc.
    for seg in fragments:
        if not seg:
            continue
        s0, s1 = seg[0][0], seg[-1][0]
        near_arc = any(0 <= s0 - a.f1 <= join_gap or 0 <= a.f0 - s1 <= join_gap
                       for a in arcs)
        if near_arc:
            out.points.extend(ArcPoint(p[0], p[1], p[2], p[3]) for p in seg)
        else:
            out.n_fragments += len(seg)

    for inliers, arc in accepted:
        have = {p[0] for p in inliers}
        out.points.extend(ArcPoint(p[0], p[1], p[2], p[3]) for p in inliers)
        diam_f = [p[0] for p in inliers if p[3]]
        diam_v = [p[3] for p in inliers if p[3]]
        for f in range(arc.f0 + 1, arc.f1):
            if f in have:
                continue
            x, y = arc.xy_at(f)
            d = float(np.interp(f, diam_f, diam_v)) if len(diam_v) >= 2 else None
            out.interpolated.append(ArcPoint(f, x, y, d, interpolated=True))

    out.points.sort(key=lambda p: p.frame)
    out.interpolated.sort(key=lambda p: p.frame)
    out.arcs = arcs
    return out


# ---------------------------------------------------------------------------
# Stage 4: hit/bounce classification from arc boundaries + player boxes
# ---------------------------------------------------------------------------

@dataclass
class ArcEvent:
    frame: int
    kind: str            # 'hit' | 'bounce'
    x: float             # image px at the direction change
    y: float
    angle_deg: float     # turn magnitude between the two arcs
    near_player: bool


def _near_box(x: float, y: float,
              boxes: list[tuple[float, float, float, float]],
              frac: float) -> bool:
    for x1, y1, x2, y2 in boxes:
        mx = frac * (x2 - x1)
        my = frac * (y2 - y1)
        if x1 - mx <= x <= x2 + mx and y1 - my <= y <= y2 + my:
            return True
    return False


def classify_arc_events(arcs: list[Arc],
                        boxes_by_frame: dict[int, list],
                        *, join_gap: int = 8, min_angle_deg: float = 25.0,
                        adjacent_frac: float = 0.30) -> list[ArcEvent]:
    """One event per adjacent arc pair whose junction is a real direction
    change: at/near a player box = hit (paddle contact); in open court =
    bounce. Junctions further apart than join_gap are track restarts, not
    events; turns shallower than min_angle_deg are segmentation artifacts of
    a single continuing shot."""
    events: list[ArcEvent] = []
    for a, b in zip(arcs, arcs[1:]):
        gap = b.f0 - a.f1
        if gap > join_gap or gap < 0:
            continue
        v1 = a.vel_at(a.f1)
        v2 = b.vel_at(b.f0)
        s1, s2 = math.hypot(*v1), math.hypot(*v2)
        if s1 < 1e-6 or s2 < 1e-6:
            continue
        cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (s1 * s2)))
        ang = math.degrees(math.acos(cosang))
        if ang < min_angle_deg:
            continue
        ax, ay = a.xy_at(a.f1)
        bx, by = b.xy_at(b.f0)
        ex, ey = (ax + bx) / 2.0, (ay + by) / 2.0
        ef = int(round((a.f1 + b.f0) / 2.0))
        near = any(_near_box(ex, ey, boxes_by_frame.get(f, []), adjacent_frac)
                   for f in range(a.f1, b.f0 + 1))
        events.append(ArcEvent(frame=ef, kind="hit" if near else "bounce",
                               x=ex, y=ey, angle_deg=ang, near_player=near))
    return events

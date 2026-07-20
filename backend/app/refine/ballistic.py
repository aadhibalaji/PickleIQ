"""Ballistic 3D reconstruction of the ball from its pixel track.

A single camera can't place a ball in 3D from one frame — the pixel is only a
ray, and the two per-frame cues are both unreliable for this footage: the
ground-plane projection assumes height=0 (wrong whenever the ball is airborne,
e.g. a serve toss projects metres behind the baseline), and the apparent-size
depth is far too noisy for the small far-court ball.

Physics resolves the ambiguity. Between a hit and the next bounce the ball is in
free fall, so its world position is a parabola ``P(t) = P0 + V0 t + ½ g t²``.
Fitting that parabola (6 unknowns) so it PROJECTS onto the observed pixel run
recovers the full 3D trajectory — court x/y AND height — from the accurate pixel
track alone, with gravity removing the single-view ambiguity. One good fit fixes
every downstream ball check at once (net crossings, contact, bounces, depth),
because they all read ``ball_xy`` / ``ball_height``.

Used by ``io_loaders.load_tracking`` to overwrite the ground-plane projection on
flight segments that fit well; segments too short or that fit poorly keep the
plane projection (exact when the ball is low anyway).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..cv.court import CameraModel

log = logging.getLogger("refine.ballistic")

G = np.array([0.0, 0.0, -9.81])   # m/s², world z up


@dataclass
class Flight:
    """A fitted gravity arc for one ball flight. Evaluable at ANY frame (incl.
    before the first tracked one), which is how the serve gate recovers the
    untrackable strike: extrapolate this arc back to the server."""
    f_start: float           # absolute frame index of the fit's t=0
    fps: float
    P0: np.ndarray           # world position at f_start [3]
    V0: np.ndarray           # world velocity at f_start [3] (m/s)
    f_lo: int                # tracked-frame span [f_lo, f_hi] the fit was built on
    f_hi: int

    def eval(self, frame: float) -> np.ndarray:
        """Court (x, y, height) at an absolute frame index."""
        t = (frame - self.f_start) / self.fps
        return self.P0 + self.V0 * t + 0.5 * G * t * t


def _project(P: np.ndarray, cam: CameraModel) -> np.ndarray:
    """World 3D point -> image pixel (perspective)."""
    pc = cam.R @ P + cam.t
    uv = cam.K @ pc
    return uv[:2] / uv[2]


def fit_flight(frames: np.ndarray, uv: np.ndarray, cam: CameraModel, fps: float,
               ground_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fit one ballistic arc to a pixel run. Returns (court_xy[M,2], height[M],
    rms_px) or None if the solve fails."""
    k = (frames - frames[0]) / fps

    def resid(p):
        P0, V0 = p[:3], p[3:]
        out = []
        for kk, (u, v) in zip(k, uv):
            pu, pv = _project(P0 + V0 * kk + 0.5 * G * kk * kk, cam)
            out += [pu - u, pv - v]
        return out

    # Init: x/y from the ground projection of the first pixel (right ballpark),
    # a typical mid-flight height, and a downcourt velocity guess.
    init = [ground_xy[0], ground_xy[1], 1.5, 0.0, 4.0, 0.0]
    try:
        sol = least_squares(resid, init, loss="soft_l1", f_scale=5.0, max_nfev=120)
    except Exception:                                    # pragma: no cover
        return None
    r = np.asarray(resid(sol.x))
    rms = float(np.sqrt(np.mean(r * r))) if len(r) else np.inf
    P0, V0 = sol.x[:3], sol.x[3:]
    traj = np.array([P0 + V0 * kk + 0.5 * G * kk * kk for kk in k])
    return traj[:, :2], traj[:, 2], rms, P0, V0


def _segments(finite: np.ndarray, bounce: np.ndarray, min_len: int) -> list[np.ndarray]:
    """Index runs of consecutive tracked frames, split at bounces (a bounce ends
    one parabola and starts the next)."""
    segs, cur = [], []
    for i in range(len(finite)):
        if finite[i]:
            cur.append(i)
            if bounce[i]:
                segs.append(cur); cur = []
        elif cur:
            segs.append(cur); cur = []
    if cur:
        segs.append(cur)
    return [np.asarray(s) for s in segs if len(s) >= min_len]


def reconstruct(td, cam: CameraModel, max_rms_px: float = 12.0,
                min_len: int = 5) -> int:
    """In place: overwrite ``td.ball_xy`` / ``td.ball_height`` with the ballistic
    3D solution on every flight segment that fits within ``max_rms_px``. Segments
    that are too short or fit poorly keep the ground-plane projection (which is
    exact when the ball is low). Returns the number of segments replaced."""
    if getattr(td, "ball_px", None) is None:
        return 0
    finite = np.isfinite(td.ball_px[:, 0])
    replaced = 0
    for idx in _segments(finite, td.bounce, min_len):
        frames = (td.f0 + idx).astype(float)
        gxy = td.ball_xy[idx[0]]
        if not np.isfinite(gxy).all():
            gxy = np.array([3.0, 6.7])          # court centre fallback
        res = fit_flight(frames, td.ball_px[idx], cam, td.fps, gxy)
        if res is None:
            continue
        xy, h, rms, P0, V0 = res
        if rms <= max_rms_px:
            td.ball_xy[idx] = xy
            td.ball_height[idx] = h
            td.ball_flights.append(Flight(f_start=float(frames[0]), fps=td.fps,
                                          P0=P0, V0=V0,
                                          f_lo=int(frames[0]), f_hi=int(frames[-1])))
            replaced += 1
    return replaced

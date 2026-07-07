"""Constant-acceleration Kalman smoothing for the 2D ball track, with short
gap bridging. Operates on sparse observations (frame, x, y) and produces a
dense smoothed track with velocity, plus a per-frame 'observed' flag so the
rally logic can tell real dropouts from bridged gaps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float
    vx: float
    vy: float
    observed: bool
    diameter_px: float | None = None


def smooth_track(
    obs: list[tuple[int, float, float, float | None]],
    max_gap: int = 12,
    dt: float = 1.0,
    process_noise: float = 1.0,
    meas_noise: float = 4.0,
    gate2: float = 0.0,
    max_reject: int = 3,
) -> list[TrackPoint]:
    """obs: sorted (frame, x, y, diameter_px|None). Gaps <= max_gap frames are
    bridged by prediction; larger gaps split the output (track restarts).

    Outlier rejection (gate2 > 0): each measurement's innovation is scored by
    Mahalanobis^2 distance from the Kalman prediction. A ball obeys smooth
    physics, so a measurement that lands far outside the predicted uncertainty
    (a foot/noise blob, or a mis-association) is REJECTED — the filter predicts
    through it (unobserved point) instead of being yanked off the trajectory.
    This is the 'remove irregular ball patterns' filter.

    Real hits/bounces are genuine sharp maneuvers, not noise, so they must NOT
    be permanently suppressed: after `max_reject` CONSECUTIVE rejects the motion
    model is deemed wrong (a real direction change) and the filter restarts on
    the newest measurement. One-off outliers are dropped; sustained new
    trajectories are re-acquired."""
    if not obs:
        return []
    # State [x, y, vx, vy, ax, ay]
    F = np.eye(6)
    F[0, 2] = F[1, 3] = dt
    F[2, 4] = F[3, 5] = dt
    F[0, 4] = F[1, 5] = 0.5 * dt * dt
    Hm = np.zeros((2, 6)); Hm[0, 0] = Hm[1, 1] = 1.0
    Q = np.eye(6) * process_noise
    Rm = np.eye(2) * meas_noise

    out: list[TrackPoint] = []
    x = np.array([obs[0][1], obs[0][2], 0, 0, 0, 0], dtype=float)
    P = np.eye(6) * 100.0
    out.append(TrackPoint(obs[0][0], x[0], x[1], 0.0, 0.0, True, obs[0][3]))
    prev_frame = obs[0][0]
    reject_run = 0

    for frame, mx, my, diam in obs[1:]:
        gap = frame - prev_frame
        if gap > max_gap:
            # Restart: too long to bridge honestly.
            x = np.array([mx, my, 0, 0, 0, 0], dtype=float)
            P = np.eye(6) * 100.0
            out.append(TrackPoint(frame, mx, my, 0.0, 0.0, True, diam))
            prev_frame = frame
            reject_run = 0
            continue
        for step in range(gap):
            x = F @ x
            P = F @ P @ F.T + Q
            f_i = prev_frame + step + 1
            if f_i < frame:
                out.append(TrackPoint(f_i, x[0], x[1], x[2], x[3], False))
        # Innovation at `frame`.
        y_res = np.array([mx, my]) - Hm @ x
        S = Hm @ P @ Hm.T + Rm
        Sinv = np.linalg.inv(S)
        if gate2 > 0.0 and float(y_res @ Sinv @ y_res) > gate2:
            reject_run += 1
            if reject_run >= max_reject:
                # Sustained deviation = a real maneuver (hit/bounce), not noise.
                # Re-acquire the ball at the current measurement.
                x = np.array([mx, my, 0, 0, 0, 0], dtype=float)
                P = np.eye(6) * 100.0
                out.append(TrackPoint(frame, mx, my, 0.0, 0.0, True, diam))
                prev_frame = frame
                reject_run = 0
                continue
            # Outlier: predict through it (do not let it pull the estimate).
            out.append(TrackPoint(frame, x[0], x[1], x[2], x[3], False))
            prev_frame = frame
            continue
        # Accepted measurement update.
        reject_run = 0
        K = P @ Hm.T @ Sinv
        x = x + K @ y_res
        P = (np.eye(6) - K @ Hm) @ P
        out.append(TrackPoint(frame, x[0], x[1], x[2], x[3], True, diam))
        prev_frame = frame
    return out

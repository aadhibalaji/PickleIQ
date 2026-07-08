"""Classical fast-moving-small-blob ball detector.

Two jobs:
1. Pass 1 coarse gate at low resolution / low frame rate: "is a fast
   ball-sized object moving?" — cheap and runs over 100% of footage.
2. Full-resolution fallback ball tracker for Pass 2 while TrackNet weights
   are not yet installed (see ball_tracknet.py). It is a real detector
   (three-frame differencing + blob geometry), clearly weaker than a trained
   TrackNet on far-baseline balls.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Player-box expansion margins for foot/shoe/paddle suppression, as fractions of
# box width/height. The bottom margin is the largest — that's where shoes and
# their shadows sit just past the box edge. Shared by the classical detector
# (ball_coarse) AND the trajectory selector (ball_track) so the two never drift.
BOX_MARGIN_X = 0.06
BOX_MARGIN_TOP = 0.03
BOX_MARGIN_BOTTOM = 0.12


def on_player_box(x: float, y: float,
                  boxes: list[tuple[float, float, float, float]]) -> bool:
    """True if (x, y) sits inside any player box expanded outward by the shared
    margins (largest below, where shoes live)."""
    for bx1, by1, bx2, by2 in boxes:
        w = bx2 - bx1
        h = by2 - by1
        if (bx1 - BOX_MARGIN_X * w < x < bx2 + BOX_MARGIN_X * w
                and by1 - BOX_MARGIN_TOP * h < y < by2 + BOX_MARGIN_BOTTOM * h):
            return True
    return False


@dataclass
class BallCandidate:
    x: float
    y: float
    diameter_px: float
    score: float


class MotionBallDetector:
    """Three-frame differencing keeps only objects moving between BOTH frame
    pairs — static court lines and slow players are suppressed."""

    def __init__(self, min_area: float = 2.0, max_area: float = 400.0,
                 sat_min: float = 0.0, val_min: float = 0.0,
                 hue_lo: float = 0.0, hue_hi: float = 180.0,
                 motion_thresh: int = 12, min_circularity: float = 0.35):
        self.prev: np.ndarray | None = None
        self.prev2: np.ndarray | None = None
        self._prev_bgr: np.ndarray | None = None  # colour of the MIDDLE frame
        self.min_area = min_area
        self.max_area = max_area
        self.motion_thresh = int(motion_thresh)
        self.min_circularity = float(min_circularity)
        # Optional appearance gate (OpenCV HSV: H 0-180, S/V 0-255). A pickleball
        # is bright and saturated; a dark shorts hem or a broad-spectrum white
        # shoe often is not. DISABLED by default (sat_min=val_min=0, full hue
        # range) because ball colour varies by clip — set these params once you
        # know the ball colour to cheaply kill off-colour foot/short blobs.
        self.sat_min = float(sat_min)
        self.val_min = float(val_min)
        self.hue_lo = float(hue_lo)
        self.hue_hi = float(hue_hi)
        self._color_gate = self.sat_min > 0 or self.val_min > 0 \
            or self.hue_lo > 0 or self.hue_hi < 180

    def reset(self) -> None:
        self.prev = self.prev2 = None
        self._prev_bgr = None

    def _color_ok(self, hsv: np.ndarray, x: float, y: float, r: float) -> bool:
        """Whether the blob's colour passes the appearance gate. Sampled from the
        blob CORE (the brightest pixels in a small patch) rather than a padded-box
        mean, so the dark background around a few-pixel ball doesn't dilute the
        reading. Hue is checked with wraparound (hue_lo may exceed hue_hi to span
        the red/orange seam)."""
        h, w = hsv.shape[:2]
        rr = max(1, int(r))
        x0, x1 = max(0, int(x) - rr), min(w, int(x) + rr + 1)
        y0, y1 = max(0, int(y) - rr), min(h, int(y) + rr + 1)
        if x0 >= x1 or y0 >= y1:
            return True
        patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
        vmax = float(patch[:, 2].max())
        core = patch[patch[:, 2] >= 0.7 * vmax]     # the bright object, not its halo
        m = core.mean(axis=0)
        hue, sat, val = float(m[0]), float(m[1]), float(m[2])
        if sat < self.sat_min or val < self.val_min:
            return False
        if self.hue_lo <= self.hue_hi:
            return self.hue_lo <= hue <= self.hue_hi
        return hue >= self.hue_lo or hue <= self.hue_hi  # wraparound

    def update(self, frame_bgr: np.ndarray,
               player_boxes: list[tuple[float, float, float, float]] | None = None,
               ) -> list[BallCandidate]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Three-frame differencing localises a moving blob to the MIDDLE frame's
        # position (the overlap of the two diffs), which is the PREVIOUS frame's
        # colour, not the current one — a fast ball has already moved on in the
        # current frame. Sample the appearance gate from that middle frame.
        mid_bgr = self._prev_bgr
        self._prev_bgr = frame_bgr
        if self.prev is None or self.prev2 is None:
            self.prev2, self.prev = self.prev, gray
            return []
        hsv = (cv2.cvtColor(mid_bgr, cv2.COLOR_BGR2HSV)
               if self._color_gate and mid_bgr is not None else None)
        d1 = cv2.absdiff(gray, self.prev)
        d2 = cv2.absdiff(self.prev, self.prev2)
        t = self.motion_thresh
        motion = cv2.bitwise_and(cv2.threshold(d1, t, 255, cv2.THRESH_BINARY)[1],
                                 cv2.threshold(d2, t, 255, cv2.THRESH_BINARY)[1])
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        self.prev2, self.prev = self.prev, gray

        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[BallCandidate] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r <= 0:
                continue
            circularity = area / (np.pi * r * r)
            if circularity < self.min_circularity:
                continue
            # Suppress limbs / paddle / SHOE false positives: ignore any blob
            # inside a player's box (expanded outward by the shared margins — see
            # on_player_box). Previously only the inner CORE was excluded, so
            # shoes at the bottom edge and paddles at the sides survived and got
            # tracked as the ball, then projected to the sideline and mislabelled
            # "bounce OUT". The ball is only suppressed when directly over a
            # player — where it's a hit passing through, which the Kalman tracker
            # bridges over a few frames.
            if player_boxes and on_player_box(x, y, player_boxes):
                continue
            # Appearance gate (no-op unless configured): drop off-colour blobs.
            if hsv is not None and not self._color_ok(hsv, x, y, r):
                continue
            out.append(BallCandidate(float(x), float(y), float(2 * r), float(circularity)))
        out.sort(key=lambda b: -b.score)
        return out[:5]

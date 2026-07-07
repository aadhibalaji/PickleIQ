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


@dataclass
class BallCandidate:
    x: float
    y: float
    diameter_px: float
    score: float


class MotionBallDetector:
    """Three-frame differencing keeps only objects moving between BOTH frame
    pairs — static court lines and slow players are suppressed."""

    def __init__(self, min_area: float = 2.0, max_area: float = 400.0):
        self.prev: np.ndarray | None = None
        self.prev2: np.ndarray | None = None
        self.min_area = min_area
        self.max_area = max_area

    def reset(self) -> None:
        self.prev = self.prev2 = None

    def update(self, frame_bgr: np.ndarray,
               player_boxes: list[tuple[float, float, float, float]] | None = None,
               ) -> list[BallCandidate]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev is None or self.prev2 is None:
            self.prev2, self.prev = self.prev, gray
            return []
        d1 = cv2.absdiff(gray, self.prev)
        d2 = cv2.absdiff(self.prev, self.prev2)
        motion = cv2.bitwise_and(cv2.threshold(d1, 12, 255, cv2.THRESH_BINARY)[1],
                                 cv2.threshold(d2, 12, 255, cv2.THRESH_BINARY)[1])
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
            if circularity < 0.35:
                continue
            # Suppress limbs / paddle / SHOE false positives: ignore any blob
            # inside a player's box, expanded slightly outward to also catch a
            # foot, shadow, or paddle just past the box edge (bottom margin is
            # largest — that's where shoes live). Previously only the inner CORE
            # was excluded, so shoes at the bottom edge and paddles at the sides
            # survived and got tracked as the ball, then projected to the
            # sideline and mislabelled "bounce OUT". The ball is only suppressed
            # when directly over a player — where it's a hit passing through,
            # which the Kalman tracker bridges over a few frames.
            if player_boxes and any(
                bx1 - 0.06 * (bx2 - bx1) < x < bx2 + 0.06 * (bx2 - bx1)
                and by1 - 0.03 * (by2 - by1) < y < by2 + 0.12 * (by2 - by1)
                for bx1, by1, bx2, by2 in player_boxes
            ):
                continue
            out.append(BallCandidate(float(x), float(y), float(2 * r), float(circularity)))
        out.sort(key=lambda b: -b.score)
        return out[:5]

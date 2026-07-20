"""Image-space player-detection exclusion zones.

Regions of the FIXED camera frame where pose detections are always false — the
neighbouring courts, spectators, and fence-line people in the background corners.
A detection whose box CENTRE falls inside any zone is dropped (see
players.PoseTracker.update), which is cheaper and safer than masking the frame
before YOLO (no edge artefacts). The zones are also drawn on the overlay so they
are visible (including black-background mode).

Zones are normalized polygons — each vertex is (x, y) as a FRACTION of frame
width/height — so they are resolution-independent and shared across every clip
from this camera (the calibration is global for the same reason). Tuned from a
user-marked frame on 2026-07-07; edit the polygons here if the framing changes.
"""
from __future__ import annotations

# Each zone is a list of (x, y) vertices in 0..1 of frame width/height.
# Two symmetric right-triangles in the top corners (right angle at the corner,
# hypotenuse cutting across) — they cover the background / fence-line people in
# the upper-left and upper-right while leaving the whole court region clear.
# Tuned from a user-marked frame on 2026-07-07; mirror-symmetric about x=0.5.
# Lowered 2026-07-07 (round 3): pulled the top-edge (center) vertices in toward
# x=0.5 (0.40->0.44 left, 0.60->0.56 right) and dropped the side vertices lower
# (0.56->0.62), which swings the hypotenuse to cover more of the background wedge.
# The top intercepts stay symmetric about x=0.5 and clear of the real far player
# at x~0.54 (sits in the 0.44..0.56 gap) so it is NOT excluded.
EXCLUSION_ZONES: list[list[tuple[float, float]]] = [
    # Top-left corner triangle.
    [(0.0, 0.0), (0.44, 0.0), (0.0, 0.62)],
    # Top-right corner triangle (mirror of the above).
    [(1.0, 0.0), (0.56, 0.0), (1.0, 0.62)],
]


def _point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon on normalized coords."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def in_excluded(x_px: float, y_px: float, w: int, h: int) -> bool:
    """True if pixel (x_px, y_px) lies inside any exclusion zone."""
    if not w or not h:
        return False
    nx, ny = x_px / w, y_px / h
    return any(_point_in_poly(nx, ny, z) for z in EXCLUSION_ZONES)


def zones_as_lists() -> list[list[list[float]]]:
    """JSON-friendly copy for the API / overlay drawing."""
    return [[[x, y] for (x, y) in z] for z in EXCLUSION_ZONES]

"""Court model, homography and single-view geometry.

World frame (metres): pickleball court is 20 ft x 44 ft (6.096 x 13.4112 m).
Origin = FAR-left court corner as seen from the camera; x to the right
(0..6.096), y toward the camera (0 = far baseline, 13.4112 = near baseline),
z up. Net at y = 6.7056; kitchen (NVZ) lines 2.1336 m either side of it.

The homography H maps image pixels -> court plane (z=0) metres. Because the
camera is fixed, calibration is one-per-camera-setup and reused across clips.
From H we also self-calibrate the focal length (orthogonality of the rotation
columns), which unlocks ball height reconstruction from apparent size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

COURT_W = 6.096       # metres (20 ft)
COURT_L = 13.4112     # metres (44 ft)
NET_Y = COURT_L / 2   # 6.7056
KITCHEN = 2.1336      # 7 ft
BALL_DIAMETER = 0.074  # metres (~2.9 in pickleball)

# Named reference points a user can click; any >= 4 non-collinear fit H.
REFERENCE_POINTS: dict[str, tuple[float, float]] = {
    "far_left_corner": (0.0, 0.0),
    "far_right_corner": (COURT_W, 0.0),
    "near_left_corner": (0.0, COURT_L),
    "near_right_corner": (COURT_W, COURT_L),
    "far_kitchen_left": (0.0, NET_Y - KITCHEN),
    "far_kitchen_right": (COURT_W, NET_Y - KITCHEN),
    "near_kitchen_left": (0.0, NET_Y + KITCHEN),
    "near_kitchen_right": (COURT_W, NET_Y + KITCHEN),
    "net_left": (0.0, NET_Y),
    "net_right": (COURT_W, NET_Y),
    "far_center_baseline": (COURT_W / 2, 0.0),
    "near_center_baseline": (COURT_W / 2, COURT_L),
}


def fit_homography(named_pixels: list[dict]) -> np.ndarray:
    """named_pixels: [{name, x, y}] with name in REFERENCE_POINTS.
    Returns 3x3 H mapping image px -> court metres."""
    src, dst = [], []
    for p in named_pixels:
        if p["name"] not in REFERENCE_POINTS:
            raise ValueError(f"unknown reference point {p['name']!r}")
        src.append([p["x"], p["y"]])
        dst.append(REFERENCE_POINTS[p["name"]])
    if len(src) < 4:
        raise ValueError("need at least 4 reference points")
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 0.05)
    if H is None:
        raise ValueError("homography fit failed — check the clicked points")
    return H


def img_to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts (N,2) pixels -> (N,2) court metres."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def court_to_img(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return img_to_court(np.linalg.inv(H), pts)


def in_court(xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Court-plane point(s) inside the playing rectangle (lines count as in)."""
    xy = np.atleast_2d(xy)
    return (
        (xy[:, 0] >= -margin) & (xy[:, 0] <= COURT_W + margin)
        & (xy[:, 1] >= -margin) & (xy[:, 1] <= COURT_L + margin)
    )


def side_of_net(y_court: float) -> str:
    return "far" if y_court < NET_Y else "near"


@dataclass
class CameraModel:
    """Full single-view camera recovered from the plane homography.

    H maps image->world; internally we decompose H_wi = inv(H) = K [r1 r2 t].
    Focal length is self-calibrated from the rotation-column constraints
    (principal point assumed at image centre — fine for consumer cameras).
    """

    H: np.ndarray          # image -> court (3x3)
    K: np.ndarray          # intrinsics
    R: np.ndarray          # world -> camera rotation
    t: np.ndarray          # world -> camera translation
    focal_px: float

    @staticmethod
    def from_homography(H: np.ndarray, img_w: int, img_h: int) -> "CameraModel":
        cx, cy = img_w / 2.0, img_h / 2.0
        Hwi = np.linalg.inv(H)
        Hwi = Hwi / Hwi[2, 2]
        h1, h2 = Hwi[:, 0], Hwi[:, 1]

        # With K = diag(f,f,1)+(cx,cy): K^-1 h = [(hx-cx*hz)/f, (hy-cy*hz)/f, hz]
        a1, b1, c1 = h1[0] - cx * h1[2], h1[1] - cy * h1[2], h1[2]
        a2, b2, c2 = h2[0] - cx * h2[2], h2[1] - cy * h2[2], h2[2]
        f_sq = []
        if abs(c1 * c2) > 1e-12:
            v = -(a1 * a2 + b1 * b2) / (c1 * c2)          # r1 . r2 = 0
            if v > 0:
                f_sq.append(v)
        denom = c1 * c1 - c2 * c2
        if abs(denom) > 1e-12:
            v = (a2 * a2 + b2 * b2 - a1 * a1 - b1 * b1) / denom  # |r1| = |r2|
            if v > 0:
                f_sq.append(v)
        # Fallback: a generic ~60 deg horizontal FOV if constraints degenerate.
        f = math.sqrt(np.mean(f_sq)) if f_sq else img_w * 0.9

        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        Kinv = np.linalg.inv(K)
        r1 = Kinv @ h1
        lam = np.linalg.norm(r1)
        r1, r2 = r1 / lam, (Kinv @ h2) / lam
        r3 = np.cross(r1, r2)
        R = np.column_stack([r1, r2, r3])
        # Orthonormalise (SVD projection to nearest rotation).
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        t = (Kinv @ Hwi[:, 2]) / lam
        # Camera must sit above the court plane (z up in world frame).
        cam_center = -R.T @ t
        if cam_center[2] < 0:
            R[:, 2] *= -1  # flip handedness choice
            cam_center = -R.T @ t
        return CameraModel(H=H, K=K, R=R, t=t, focal_px=float(f))

    def ball_world_from_size(self, u: float, v: float, diam_px: float) -> np.ndarray:
        """3D ball centre from pixel position + apparent diameter.
        Small-object model: camera-frame depth Z ~= f * D_real / d_px."""
        if diam_px <= 0:
            raise ValueError("diameter must be positive")
        Z = self.focal_px * BALL_DIAMETER / diam_px
        ray = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        p_cam = Z * ray  # ray has z=1, so this puts the point at depth Z
        return self.R.T @ (p_cam - self.t)  # world (x, y, height)


def _sample_mask(mask: np.ndarray, p0: np.ndarray, p1: np.ndarray, n: int = 60) -> float:
    """Fraction of sampled points along segment p0->p1 that land on white
    court-line paint (with a small vertical tolerance)."""
    h, w = mask.shape
    hits = total = 0
    for t in np.linspace(0.05, 0.95, n):
        x = int(round(p0[0] + t * (p1[0] - p0[0])))
        y = int(round(p0[1] + t * (p1[1] - p0[1])))
        if 0 <= x < w and 2 <= y < h - 2:
            total += 1
            if mask[y - 2:y + 3, x].max() > 0:
                hits += 1
    return hits / total if total else 0.0


def suggest_corners(frame: np.ndarray) -> list[dict]:
    """Best-effort auto-suggestion of the 4 outer corners from court line
    paint. The UI always lets the user drag/correct these; this just saves
    clicks. Returns [] when no confident quadrilateral is found.

    Neighbouring courts confuse a naive "highest horizontal line = far
    baseline" pick, so every candidate far baseline is verified by fitting a
    trial homography and scoring how well the PROJECTED net and kitchen
    lines land on white paint — only the true far baseline makes the whole
    template line up."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Court lines are bright and low-saturation regardless of paint colour.
    mask = cv2.inRange(hsv, (0, 0, 170), (180, 80, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=120,
                            minLineLength=frame.shape[1] // 8, maxLineGap=30)
    if lines is None:
        return []
    horiz, vert = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        (horiz if ang < 30 or ang > 150 else vert).append((x1, y1, x2, y2))
    if len(horiz) < 2 or len(vert) < 2:
        return []

    def line_params(seg):
        x1, y1, x2, y2 = seg
        return np.cross([x1, y1, 1.0], [x2, y2, 1.0])

    def intersect(l1, l2):
        p = np.cross(l1, l2)
        if abs(p[2]) < 1e-9:
            return None
        return np.array([p[0] / p[2], p[1] / p[2]])

    h_img, w_img = frame.shape[:2]
    # Near baseline = lowest horizontal; sidelines = extreme slanted lines.
    horiz.sort(key=lambda s: (s[1] + s[3]) / 2)
    near_l = line_params(horiz[-1])
    vert.sort(key=lambda s: (s[0] + s[2]) / 2)
    left_l, right_l = line_params(vert[0]), line_params(vert[-1])
    near_y = (horiz[-1][1] + horiz[-1][3]) / 2

    # Deduplicate candidate far baselines by mean y.
    cand_ys, cands = [], []
    for seg in horiz[:-1]:
        y = (seg[1] + seg[3]) / 2
        if y > near_y - 40:
            continue
        if any(abs(y - cy) < 12 for cy in cand_ys):
            continue
        cand_ys.append(y)
        cands.append(line_params(seg))

    template = np.float32([
        REFERENCE_POINTS["far_left_corner"], REFERENCE_POINTS["far_right_corner"],
        REFERENCE_POINTS["near_left_corner"], REFERENCE_POINTS["near_right_corner"],
    ])
    check_lines = [  # court-frame lines that must land on paint
        [(0.0, NET_Y - KITCHEN), (COURT_W, NET_Y - KITCHEN)],
        [(0.0, NET_Y + KITCHEN), (COURT_W, NET_Y + KITCHEN)],
        [(COURT_W / 2, NET_Y + KITCHEN), (COURT_W / 2, COURT_L)],
    ]

    best, best_score = None, 0.0
    for far_l in cands:
        pts = [intersect(far_l, left_l), intersect(far_l, right_l),
               intersect(near_l, left_l), intersect(near_l, right_l)]
        if any(p is None or not (-w_img * 0.2 <= p[0] <= w_img * 1.2
                                 and -h_img * 0.2 <= p[1] <= h_img * 1.2) for p in pts):
            continue
        src = np.float32([p for p in pts])
        try:
            Hwi = cv2.getPerspectiveTransform(template, src)  # court -> image
        except cv2.error:
            continue
        score = 0.0
        for a, b in check_lines:
            proj = cv2.perspectiveTransform(np.float32([[a, b]]).reshape(-1, 1, 2), Hwi).reshape(2, 2)
            score += _sample_mask(mask, proj[0], proj[1])
        score /= len(check_lines)
        if score > best_score:
            best_score, best = score, pts

    if best is None or best_score < 0.35:
        return []
    names = ["far_left_corner", "far_right_corner", "near_left_corner", "near_right_corner"]
    return [{"name": n, "x": float(p[0]), "y": float(p[1])} for n, p in zip(names, best)]

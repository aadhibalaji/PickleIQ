"""Pure-function tests for trajectory.py: confidence formula (RMS + depth),
player attribution, and null/omission handling on synthetic ball tracks. No
DB -- same pattern as test_report_stats.py / test_readiness.py."""
from __future__ import annotations

import cv2
import numpy as np

from ...cv.court import COURT_L, COURT_W, CameraModel
from ...refine import ballistic
from ...refine.io_loaders import TrackingData
from ..trajectory import (ATTRIBUTION_WINDOW_FRAMES, _attribute_player,
                          point_confidence, rally_trajectory_points)

FPS = 30.0


def _camera() -> CameraModel:
    """Same synthetic axial-camera homography contact_detect.py/bounce.py's
    own self-checks use: far baseline narrow near image top, near baseline
    wide near image bottom."""
    img_pts = np.float32([[400, 200], [880, 200], [1180, 700], [100, 700]])
    court_pts = np.float32([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])
    H, _ = cv2.findHomography(img_pts, court_pts)
    return CameraModel.from_homography(H, 1280, 720)


def _blank_td(n: int, f0: int = 0) -> TrackingData:
    return TrackingData(
        fps=FPS, f0=f0,
        present=np.ones(n, bool),
        ball_xy=np.full((n, 2), np.nan),
        ball_conf=np.zeros(n),
        ball_interp=np.zeros(n, bool),
        ball_height=np.full(n, np.nan),
        players={},
        pspeed=np.zeros(n),
        bounce=np.zeros(n, bool),
        hit=np.zeros(n, bool),
        evt_xy=np.full((n, 2), np.nan),
        video_id="synthetic",
        ball_px=np.full((n, 2), np.nan),
    )


def _script_arc(td: TrackingData, cam: CameraModel, idx: range,
                P0: np.ndarray, V0: np.ndarray) -> None:
    """Project a known ballistic arc into td.ball_px/ball_xy over `idx` (a
    plausible in-court parabola), mirroring what a real fitted+observed
    flight segment would look like."""
    for i in idx:
        t = (i - idx[0]) / td.fps
        P = P0 + V0 * t + 0.5 * ballistic.G * t * t
        uv = ballistic._project(P, cam)
        td.ball_px[i] = uv
        td.ball_xy[i] = P[:2]   # ground-projection stand-in, only used as fit init


# ------------------------------------------------------------- confidence

def test_point_confidence_perfect_fit_near_camera():
    # rms=0, y at the near baseline (y=COURT_L) -> full confidence.
    assert point_confidence(rms=0.0, y=COURT_L, max_rms_px=12.0) == 1.0


def test_point_confidence_decreases_with_rms():
    near = point_confidence(rms=1.0, y=COURT_L, max_rms_px=12.0)
    far = point_confidence(rms=20.0, y=COURT_L, max_rms_px=12.0)
    assert 0.0 <= far < near <= 1.0


def test_point_confidence_far_court_is_discounted_but_not_zeroed():
    near_baseline = point_confidence(rms=2.0, y=COURT_L, max_rms_px=12.0)
    far_baseline = point_confidence(rms=2.0, y=0.0, max_rms_px=12.0)
    assert far_baseline < near_baseline
    assert far_baseline > 0.0   # floored, never fabricated to exactly 0


# ------------------------------------------------------------- attribution

def test_attribute_player_nearest_preceding_contact():
    contacts = [{"frame_idx": 90, "player_id": 1, "is_serve": False},
                {"frame_idx": 100, "player_id": 2, "is_serve": False}]
    assert _attribute_player(105, contacts) == 2
    assert _attribute_player(95, contacts) == 1


def test_attribute_player_none_when_too_far():
    contacts = [{"frame_idx": 0, "player_id": 1, "is_serve": False}]
    assert _attribute_player(0 + ATTRIBUTION_WINDOW_FRAMES + 1, contacts) is None


def test_attribute_player_none_when_no_contacts():
    assert _attribute_player(50, []) is None


# ------------------------------------------------------------- end to end

def test_rally_trajectory_points_reconstructs_known_arc_with_attribution():
    cam = _camera()
    n = 20
    td = _blank_td(n)
    P0 = np.array([3.0, 2.0, 0.3])
    V0 = np.array([0.0, 4.0, 5.0])
    _script_arc(td, cam, range(n), P0, V0)
    contacts = [{"frame_idx": -1, "player_id": 7, "is_serve": True}]

    points = rally_trajectory_points(td, cam, contacts, max_rms_px=12.0, min_len=5)

    assert points, "expected the synthetic arc to reconstruct"
    for p in points:
        assert p["height_m"] is not None
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["player_id"] == 7
        assert p["is_serve"] is True
    # Peak height should be in the right ballpark for this launch (real lob,
    # not near-zero ground-plane noise).
    assert max(p["height_m"] for p in points) > 0.5


def test_rally_trajectory_points_omits_short_segment():
    cam = _camera()
    td = _blank_td(10)
    P0 = np.array([3.0, 2.0, 0.3])
    V0 = np.array([0.0, 4.0, 5.0])
    _script_arc(td, cam, range(3), P0, V0)  # shorter than min_len
    assert rally_trajectory_points(td, cam, [], min_len=5) == []


def test_rally_trajectory_points_no_ball_px_returns_empty():
    td = _blank_td(10)
    td.ball_px = None
    assert rally_trajectory_points(td, _camera(), []) == []


if __name__ == "__main__":
    test_point_confidence_perfect_fit_near_camera()
    test_point_confidence_decreases_with_rms()
    test_point_confidence_far_court_is_discounted_but_not_zeroed()
    test_attribute_player_nearest_preceding_contact()
    test_attribute_player_none_when_too_far()
    test_attribute_player_none_when_no_contacts()
    test_rally_trajectory_points_reconstructs_known_arc_with_attribution()
    test_rally_trajectory_points_omits_short_segment()
    test_rally_trajectory_points_no_ball_px_returns_empty()
    print("trajectory self-check OK")

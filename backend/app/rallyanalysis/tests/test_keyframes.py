"""Pure-function tests for keyframe extraction geometry/lookup — no video
file, no DB, no cv2 decode: extract_for_candidates' IO path degrades on its
own (unopenable video -> None) and is exercised end-to-end manually; what
needs pinning here is the crop math and the box-lookup tolerance rules."""
from __future__ import annotations

from ..keyframes import BOX_SEARCH_TOLERANCE_FRAMES, _crop_rect, _nearest_box, extract_for_candidates


# ------------------------------------------------------------- crop geometry

def test_crop_rect_expands_around_box_and_stays_inside_frame():
    rect = _crop_rect((100.0, 100.0, 200.0, 300.0), 1920, 1080)
    x0, y0, x1, y1 = rect
    assert 0 <= x0 < x1 <= 1920
    assert 0 <= y0 < y1 <= 1080
    assert x0 < 100 and x1 > 200      # margin actually expands beyond the box
    assert y0 < 100 and y1 > 300


def test_crop_rect_clamps_offscreen_box_without_inverting():
    x0, y0, x1, y1 = _crop_rect((-50.0, -50.0, 10.0, 10.0), 1920, 1080)
    assert x0 == 0 and y0 == 0
    assert x1 > x0 and y1 > y0


def test_crop_rect_degenerate_box_still_yields_valid_rect():
    x0, y0, x1, y1 = _crop_rect((500.0, 500.0, 500.0, 500.0), 1920, 1080)
    assert x1 > x0 and y1 > y0


# ------------------------------------------------------------ box lookup

def _entries():
    return [
        {"f": 90, "players": [{"id": 1, "box": [1, 2, 3, 4]}, {"id": 2, "box": [9, 9, 9, 9]}]},
        {"f": 100, "players": [{"id": 1, "box": [5, 6, 7, 8]}]},
    ]


def test_nearest_box_picks_closest_frame_for_the_right_player():
    assert _nearest_box(_entries(), 1, 91) == (1, 2, 3, 4)
    assert _nearest_box(_entries(), 1, 99) == (5, 6, 7, 8)
    assert _nearest_box(_entries(), 2, 91) == (9, 9, 9, 9)


def test_nearest_box_rejects_boxes_beyond_tolerance():
    # A box from far elsewhere in the rally would crop the wrong patch of
    # court -- better to fall back to the (downscaled) full frame.
    far = 100 + BOX_SEARCH_TOLERANCE_FRAMES + 1
    assert _nearest_box(_entries(), 1, far) is None


def test_nearest_box_none_for_unseen_player():
    assert _nearest_box(_entries(), 42, 95) is None


# ------------------------------------------------------- degrade behavior

def test_extract_for_candidates_unopenable_video_returns_none():
    out = extract_for_candidates(
        "Z:/definitely/not/a/video.mp4",
        [{"contact_id": "c1", "frame_idx": 100, "player_id": 1}],
        "vid1", "pass2", fps=30.0, pre_window_s=0.6, post_window_s=0.45)
    assert out is None


def test_extract_for_candidates_empty_candidates_returns_none():
    assert extract_for_candidates("x.mp4", [], "vid1", "pass2",
                                   fps=30.0, pre_window_s=0.6, post_window_s=0.45) is None

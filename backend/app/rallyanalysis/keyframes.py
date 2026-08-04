"""Keyframe extraction for the coaching report's vision context: for each
tip-candidate shot, pull a small set of JPEG crops (prep / contact /
follow-through) of the hitting player from the original video, so the one
coaching LLM call can ground its critique in what the player actually did
instead of coaching blind from seven noisy pose numbers.

Read-only against the video file and overlays. Never writes anything —
frames are extracted fresh per report generation and sent straight to the
LLM call, not persisted (the video file is the source of truth; caching
24-odd JPEGs per report into the DB buys nothing).

Degrades to "no frames" (None / missing contact ids) on any failure — a
missing/moved video file, an unreadable frame, a player box the overlays
never captured — and the coaching call proceeds exactly as it did before
vision context existed. Frames are an upgrade, never a dependency.
"""
from __future__ import annotations

import base64
import logging

import cv2

from ..cv import overlays

log = logging.getLogger("rallyanalysis.keyframes")

# What the LLM sees of each shot: moments chosen to span one swing.
# Offsets stay inside coaching.py's PRE_WINDOW_S/POST_WINDOW_S so the overlay
# rows the pipeline already reads for metrics also cover every keyframe's
# player-box lookup.
FRAME_OFFSETS_S = (
    ("prep", -0.4),
    ("contact", 0.0),
    ("follow_through", 0.35),
)

# Crop = player box expanded by this factor on every side — enough court
# context to read the stance/court position, not so much that a far-side
# player shrinks back into a full-frame speck (the whole reason we crop).
CROP_MARGIN = 1.6

# Longest output edge in px. 512 keeps a 3-frame x MAX_TIPS report well under
# a megabyte of image payload while leaving limbs clearly readable.
MAX_EDGE_PX = 512

JPEG_QUALITY = 80

# How far (frames) from the target frame we'll accept an overlay player box
# for cropping before falling back to the full frame downscaled.
BOX_SEARCH_TOLERANCE_FRAMES = 12


def _crop_rect(box: tuple[float, float, float, float], frame_w: int, frame_h: int,
                margin: float = CROP_MARGIN) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) ints, clamped to the frame: the player box expanded
    by `margin` on each side of its center. Never inverts/empties — a
    degenerate input box still yields at least a 1px rect inside the frame."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w = max(1.0, (x1 - x0) / 2.0) * (1.0 + margin)
    half_h = max(1.0, (y1 - y0) / 2.0) * (1.0 + margin)
    rx0 = max(0, int(cx - half_w))
    ry0 = max(0, int(cy - half_h))
    rx1 = min(frame_w, int(cx + half_w))
    ry1 = min(frame_h, int(cy + half_h))
    if rx1 <= rx0:
        rx1 = min(frame_w, rx0 + 1)
    if ry1 <= ry0:
        ry1 = min(frame_h, ry0 + 1)
    return rx0, ry0, rx1, ry1


def _nearest_box(entries: list[dict], player_id, target_frame: int,
                  tolerance: int = BOX_SEARCH_TOLERANCE_FRAMES) -> tuple | None:
    """This player's box from the overlay row nearest target_frame (ties
    favour earlier), or None if the nearest one is further than `tolerance`
    frames away — a box from half a second elsewhere would crop the wrong
    patch of court."""
    best = None
    for e in entries:
        for p in e.get("players", []):
            if p.get("id") == player_id and p.get("box"):
                d = abs(e["f"] - target_frame)
                if best is None or d < best[0] or (d == best[0] and e["f"] < best[1]):
                    best = (d, e["f"], tuple(p["box"]))
                break
    if best is None or best[0] > tolerance:
        return None
    return best[2]


def _encode_jpeg_b64(img) -> str | None:
    h, w = img.shape[:2]
    scale = MAX_EDGE_PX / max(w, h)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def extract_for_candidates(video_path: str, candidates: list[dict], video_id: str,
                            overlay_source: str, fps: float,
                            pre_window_s: float, post_window_s: float) -> dict[str, list[dict]] | None:
    """{contact_id: [{label, jpeg_b64}, ...]} for each tip candidate — called
    ONLY on the <=MAX_TIPS selected shots (see coaching.build_report's
    frames_fn), never every contact in the video. Returns None when the video
    can't be opened at all; individual unreadable frames/shots are simply
    absent rather than failing the report.

    Per-keyframe seek (CAP_PROP_POS_FRAMES) instead of a sequential decode:
    ~24 seeks per report is cheap, and a long-GOP seek landing a frame or two
    off is irrelevant here — any frame within a few hundredths of a second of
    the target moment shows the same phase of the swing."""
    if not candidates or fps <= 0:
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            log.warning("keyframes: cannot open video %s — coaching report proceeds without frames",
                        video_path)
            return None
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # (target_frame, contact_id, label, box|None) sorted by frame so the
        # capture only ever seeks forward.
        pre_f, post_f = int(pre_window_s * fps), int(post_window_s * fps)
        jobs = []
        for c in candidates:
            contact_frame = c["frame_idx"]
            entries = overlays.read_range(video_id, overlay_source,
                                           max(0, contact_frame - pre_f), contact_frame + post_f)
            for label, off_s in FRAME_OFFSETS_S:
                target = max(0, contact_frame + int(round(off_s * fps)))
                box = _nearest_box(entries, c["player_id"], target)
                jobs.append((target, c["contact_id"], label, box))
        jobs.sort(key=lambda j: j[0])

        out: dict[str, list[dict]] = {}
        for target, cid, label, box in jobs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if box is not None:
                x0, y0, x1, y1 = _crop_rect(box, frame_w, frame_h)
                frame = frame[y0:y1, x0:x1]
            b64 = _encode_jpeg_b64(frame)
            if b64 is None:
                continue
            out.setdefault(cid, []).append({"label": label, "jpeg_b64": b64})
        return out or None
    finally:
        cap.release()


if __name__ == "__main__":
    # ponytail self-check: crop rect clamps to the frame, never inverts, and
    # expands around the box center; nearest-box respects the tolerance.
    rect = _crop_rect((100.0, 100.0, 200.0, 300.0), 1920, 1080)
    assert 0 <= rect[0] < rect[2] <= 1920 and 0 <= rect[1] < rect[3] <= 1080
    assert rect[0] < 100 and rect[2] > 200  # expanded beyond the box

    edge = _crop_rect((-50.0, -50.0, 10.0, 10.0), 1920, 1080)
    assert edge[0] == 0 and edge[1] == 0 and edge[2] > 0 and edge[3] > 0

    entries = [{"f": 90, "players": [{"id": 1, "box": [1, 2, 3, 4]}]},
               {"f": 200, "players": [{"id": 1, "box": [5, 6, 7, 8]}]}]
    assert _nearest_box(entries, 1, 92) == (1, 2, 3, 4)
    assert _nearest_box(entries, 1, 150) is None      # nearest is 50 frames off -> too far
    assert _nearest_box(entries, 2, 92) is None       # player never seen
    print("keyframes self-check OK")

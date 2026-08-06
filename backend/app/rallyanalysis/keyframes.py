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

# Span moments (positioning focus, candidate dicts carrying kind == "span" —
# see positioning.span_moments) are stretches of movement, not swings: wider
# offsets to show the movement (or lack of it) around the anchor, and NO crop
# — a box-tight crop would hide exactly the evidence (where on the court the
# player is standing). The full frame is downscaled instead, with the
# player's box outlined so the LLM knows who the moment is about.
SPAN_FRAME_OFFSETS_S = (
    ("before", -1.0),
    ("moment", 0.0),
    ("after", 1.0),
)
SPAN_WINDOW_S = 1.0  # overlay read window (must cover the span offsets above)
SPAN_MARK_COLOR = (0, 200, 255)  # BGR outline for the flagged player
SPAN_MARK_THICKNESS = 3

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


def _plan_candidate_jobs(c: dict, entries: list[dict], fps: float) -> list[tuple]:
    """(target_frame, contact_id, label, crop_box|None, mark_box|None) for one
    candidate — pure, so the offset/crop policy is testable on synthetic
    entries. Contact-anchored candidates get swing offsets + a crop box; span
    candidates get the wider span offsets, no crop, and a mark box (outline)
    when the overlays have one near the target."""
    is_span = c.get("kind") == "span"
    offsets = SPAN_FRAME_OFFSETS_S if is_span else FRAME_OFFSETS_S
    jobs = []
    for label, off_s in offsets:
        target = max(0, c["frame_idx"] + int(round(off_s * fps)))
        box = _nearest_box(entries, c["player_id"], target)
        crop_box, mark_box = (None, box) if is_span else (box, None)
        jobs.append((target, c["contact_id"], label, crop_box, mark_box))
    return jobs


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

        # (target_frame, contact_id, label, crop_box, mark_box) sorted by
        # frame so the capture only ever seeks forward. Span candidates need
        # a wider overlay window than contact ones (their offsets reach ±1s).
        pre_f, post_f = int(pre_window_s * fps), int(post_window_s * fps)
        span_f = int(SPAN_WINDOW_S * fps)
        jobs = []
        for c in candidates:
            contact_frame = c["frame_idx"]
            lo, hi = ((span_f, span_f) if c.get("kind") == "span" else (pre_f, post_f))
            entries = overlays.read_range(video_id, overlay_source,
                                           max(0, contact_frame - lo), contact_frame + hi)
            jobs.extend(_plan_candidate_jobs(c, entries, fps))
        jobs.sort(key=lambda j: j[0])

        out: dict[str, list[dict]] = {}
        for target, cid, label, crop_box, mark_box in jobs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if crop_box is not None:
                x0, y0, x1, y1 = _crop_rect(crop_box, frame_w, frame_h)
                frame = frame[y0:y1, x0:x1]
            elif mark_box is not None:
                frame = frame.copy()
                x0, y0, x1, y1 = (int(v) for v in mark_box)
                cv2.rectangle(frame, (x0, y0), (x1, y1),
                               SPAN_MARK_COLOR, SPAN_MARK_THICKNESS)
            b64 = _encode_jpeg_b64(frame)
            if b64 is None:
                continue
            out.setdefault(cid, []).append({"label": label, "jpeg_b64": b64})
        return out or None
    finally:
        cap.release()


# Tighter than CROP_MARGIN (1.6, used for swing-context crops that need
# court context) — an identity photo just needs to clearly show the person,
# not their surroundings.
IDENTITY_CROP_MARGIN = 0.5


def _best_box_per_player(entries: list[dict]) -> dict[int, tuple[float, int, tuple]]:
    """{player_id: (area, frame, box)} — each player's LARGEST box across
    `entries` (closest to camera / most face-on), not their first sighting,
    which is often small, far, or mid-stride."""
    best: dict[int, tuple[float, int, tuple]] = {}
    for e in entries:
        for p in e.get("players", []):
            pid, box = p.get("id"), p.get("box")
            if pid is None or not box:
                continue
            x0, y0, x1, y1 = box
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if pid not in best or area > best[pid][0]:
                best[pid] = (area, e["f"], tuple(box))
    return best


def extract_identity_photos(video_path: str, video_id: str, overlay_source: str,
                             frame_count: int) -> dict[int, str] | None:
    """{player_id: jpeg_b64} — one representative photo per player who
    appears in the overlays, so the coaching UI can show "this is who Player
    1 is" instead of an unlabeled seat number (see _best_box_per_player for
    the selection rule). Computed fresh per report generation — a couple of
    frames, cheap next to extract_for_candidates' per-tip extraction — never
    persisted as its own artifact; the coaching report row is the cache.
    Degrades to None on any failure, same contract as extract_for_candidates."""
    entries = overlays.read_range(video_id, overlay_source, 0, max(0, frame_count))
    best = _best_box_per_player(entries)
    if not best:
        return None

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            log.warning("identity photos: cannot open video %s — proceeding without them",
                        video_path)
            return None
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out: dict[int, str] = {}
        for pid, (_, frame_idx, box) in sorted(best.items(), key=lambda kv: kv[1][1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            x0, y0, x1, y1 = _crop_rect(box, frame_w, frame_h, margin=IDENTITY_CROP_MARGIN)
            b64 = _encode_jpeg_b64(frame[y0:y1, x0:x1])
            if b64 is not None:
                out[pid] = b64
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

    # Contact candidates get swing offsets + crop box; span candidates get
    # span offsets, no crop, and a mark box instead.
    contact_jobs = _plan_candidate_jobs({"contact_id": "c1", "frame_idx": 90, "player_id": 1},
                                         entries, fps=30.0)
    assert [j[2] for j in contact_jobs] == ["prep", "contact", "follow_through"]
    assert contact_jobs[1][3] == (1, 2, 3, 4) and contact_jobs[1][4] is None
    span_jobs = _plan_candidate_jobs({"contact_id": "pos_r_1_m_90", "frame_idx": 90,
                                       "player_id": 1, "kind": "span"}, entries, fps=30.0)
    assert [j[2] for j in span_jobs] == ["before", "moment", "after"]
    assert span_jobs[1][3] is None and span_jobs[1][4] == (1, 2, 3, 4)

    # _best_box_per_player: picks the LARGEST box per player, not the first
    # or last sighting; ignores entries with no id/box.
    identity_entries = [
        {"f": 10, "players": [{"id": 1, "box": [0, 0, 50, 50]}]},   # area 2500
        {"f": 20, "players": [{"id": 1, "box": [0, 0, 100, 100]}]},  # area 10000 -> wins
        {"f": 30, "players": [{"id": 1, "box": [0, 0, 60, 60]}]},
        {"f": 15, "players": [{"id": 2, "box": None}, {"id": None, "box": [0, 0, 999, 999]}]},
    ]
    best = _best_box_per_player(identity_entries)
    assert set(best) == {1}
    assert best[1] == (10000.0, 20, (0, 0, 100, 100))
    assert _best_box_per_player([]) == {}
    print("keyframes self-check OK")

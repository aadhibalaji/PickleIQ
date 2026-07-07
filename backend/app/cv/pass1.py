"""Pass 1 — the coarse gate. Runs on 100% of the footage, cheaply, at low
resolution and reduced frame rate. Answers only "is a rally roughly
happening here?" from low-cost signals:

  1. NET CROSSINGS (primary, needs calibration): a fast ball-sized blob
     inside the court region whose track crosses the net line. This is the
     signature of live play — validated on real footage where raw motion and
     "any moving blob" fail completely (neighbouring courts in view, players
     bouncing balls between points).
  2. motion energy vs an adaptive ambient baseline (fallback when no court
     calibration exists yet, and always logged for the trainable gate),
  3. sparse serve-formation spot checks (pose model every ~2 s).

Active<->dead flips (with hysteresis) become rally windows. Pass 1 is now the
SOLE source of rally start/end: it pads each window by p1_pad_s per side and
merges windows that touch, emitting generous (recall-favouring) boundaries.
Pass 2 no longer refines them — it only produces the accurate player/ball
detection overlay inside these windows.

Per-sample features are logged into the overlays, so the rule can later be
replaced by a small trainable classifier over the same features (see
LearnedParams / train scripts).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .. import db
from ..jobs import JobCancelled, JobHandle
from . import court
from . import overlays
from .ball_coarse import MotionBallDetector
from .players import PoseTracker, is_serve_formation
from .video_io import iter_frames, probe

log = logging.getLogger("cv.pass1")


def _scaled_homography(H, scale: float):
    if H is None:
        return None
    S = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    return H @ S


def _court_mask(Hs, shape) -> np.ndarray:
    """Court region (+ small margin) at low-res scale — excludes neighbouring
    courts, spectators, and walkways from all Pass 1 signals."""
    m = np.zeros(shape, np.uint8)
    pts = court.court_to_img(Hs, np.array([
        (-1.5, -2.0), (court.COURT_W + 1.5, -2.0),
        (court.COURT_W + 1.5, court.COURT_L + 2.5), (-1.5, court.COURT_L + 2.5),
    ]))
    cv2.fillPoly(m, [np.clip(pts, [0, 0], [shape[1] - 1, shape[0] - 1]).astype(np.int32)], 255)
    return m


def _match_fast(cands, prev_cands, step: int, smin: float, smax: float = 45.0):
    """Mutually-nearest, size-consistent blob match with ball-like speed.
    Returns (candidate, prev_candidate) or (None, None)."""
    if not cands or not prev_cands:
        return None, None
    best, best_prev, best_sp = None, None, 0.0
    for ci, c in enumerate(cands):
        dists = [np.hypot(c.x - p.x, c.y - p.y) for p in prev_cands]
        pi = int(np.argmin(dists))
        back = [np.hypot(prev_cands[pi].x - c2.x, prev_cands[pi].y - c2.y) for c2 in cands]
        if int(np.argmin(back)) != ci:
            continue
        p = prev_cands[pi]
        if abs(c.diameter_px - p.diameter_px) > 0.6 * max(c.diameter_px, p.diameter_px):
            continue
        sp = dists[pi] / step
        if smin <= sp <= smax and sp > best_sp:
            best_sp, best, best_prev = sp, c, p
    return best, best_prev


def run_pass1(video_id: str, params: dict, H, handle: JobHandle,
              source: str = "pass1") -> list[tuple[int, int]]:
    with db.connect() as con:
        vrow = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
    path = vrow["path"]
    meta = probe(path)
    fps = meta.fps
    step = int(params["p1_frame_step"])
    scale = float(params["p1_scale"])
    sampled_fps = fps / step

    # Pose runs on the FULL-res frame (small far-side players vanish in the 0.4x
    # motion frame), sparse spot checks only, so track_sequence=False.
    pose = PoseTracker(H, track_sequence=False,
                       conf=float(params.get("pose_conf", 0.25)),
                       imgsz=int(params.get("p1_pose_imgsz", 1280)))
    ball_det = MotionBallDetector(min_area=1.5, max_area=200.0)

    overlays.clear(video_id, source)
    with db.connect() as con:
        con.execute("DELETE FROM rally_labels WHERE video_id=? AND source=?", (video_id, source))

    on_n = max(1, int(params["p1_on_s"] * sampled_fps))
    off_n = max(1, int(params["p1_off_s"] * sampled_fps))
    ball_recent_n = int(params["p1_ball_recent_s"] * fps)
    spot_every = int(params["p1_spot_check_s"] * fps)
    # Pass 1 is now the SOLE source of rally boundaries — Pass 2 only refines
    # detection inside these windows and no longer pads/merges them. So Pass 1
    # emits GENEROUS windows itself: pad each side and merge windows that touch,
    # favouring recall (a rally is never clipped or split) over tight edges.
    pad = int(params["p1_pad_s"] * fps)
    merge_gap = int(params["p1_merge_gap_s"] * fps)
    end_settle = int(params.get("p1_end_settle_s", 0.6) * fps)
    last_frame = max(0, meta.frame_count - 1)

    prev_gray = None
    last_ball_frame = -10**9
    last_spot = -10**9
    active = False
    streak = 0
    rally_start = None
    # Adaptive ambient-motion baseline: absolute frame-diff levels vary with
    # lighting/scene, so the gate triggers on motion RELATIVE to a slow
    # baseline (tracks down quickly, up very slowly so rallies don't lift it).
    baseline = None
    motion_ratio = float(params.get("p1_motion_ratio", 1.6))
    motion_floor = float(params.get("p1_motion_floor", 0.12))
    speed_min = float(params.get("p1_ball_speed_px", 3.5))   # px/frame @ low-res
    prev_cands: list | None = None
    Hs = _scaled_homography(H, scale)
    roi = None          # court region mask, built from the first frame
    last_cross = -10**9  # last frame a fast blob crossed the net line
    # In live play the ball crosses the net every ~1-2 s; a crossing seen
    # within this window keeps the gate open (validated ~2 s on real footage —
    # mean IoU 0.52 vs 0.05 for the motion-only gate). Now its own learnable
    # param: it used to be hard-floored at 2.0 s, which prevented learning a
    # shorter window to stop adjacent rallies bridging into one candidate.
    cross_window_n = int(params["p1_cross_window_s"] * fps)
    rallies: list[tuple[int, int]] = []
    entries: list[dict] = []
    flushed_chunk = -1
    # Player boxes from the most recent spot check, scaled to the low-res ball
    # frame, so the ball detector can suppress foot/shoe blobs (feet look like a
    # small bright ball in a single frame). Stale by up to p1_spot_check_s, but
    # players stay in roughly the same court zone that briefly — a rough mask
    # beats none, and Pass 1's primary gate (net crossings) is unaffected.
    last_boxes_scaled: list[tuple[float, float, float, float]] = []
    last_row: dict | None = None  # most recent emitted rally, for pad-merge

    def close_rally(start_f: int, end_f: int) -> None:
        nonlocal last_row
        # Drop sub-rally blips on their RAW (unpadded) length — noise, not play.
        if (end_f - start_f) / fps < params["p1_min_rally_s"]:
            return
        # TIGHT end: a point ends just after the last shot crosses the net (the
        # ball's short flight-to-land = end_settle), NOT when the net-crossing
        # hysteresis finally lapses — that flip is ~p1_cross_window_s + p1_off_s
        # (~4 s) later and lands the end well past the real one. `last_cross`
        # holds the last crossing of the rally now closing. Fall back to the raw
        # flip when no crossing was tracked (uncalibrated / motion-only gate).
        if last_cross >= start_f:
            end_tight = min(end_f, last_cross + end_settle)
            end_src = "last_cross"
        else:
            end_tight = end_f
            end_src = "hysteresis"
        # Pad the START for lead-in and to drive the touch/merge test; the END is
        # left TIGHT (Pass 2 pads its own overlay window off this boundary). Then
        # either extend the previous window if this one touches it, or emit new.
        s = max(0, start_f - pad)
        e = min(last_frame, max(end_tight, s + 1))
        meta_json = db.dumps({"end_source": end_src, "raw_end": int(end_f),
                              "last_cross": int(last_cross) if end_src == "last_cross" else None})
        if last_row is not None and s <= last_row["end"] + merge_gap:
            new_e = max(e, last_row["end"])
            if new_e != last_row["end"]:
                last_row["end"] = new_e
                with db.connect() as con:
                    con.execute("UPDATE rally_labels SET end_frame=? WHERE id=?",
                                (new_e, last_row["id"]))
                if rallies:
                    rallies[-1] = (last_row["start"], new_e)
            return
        rid = db.new_id()
        with db.connect() as con:  # stream results so the UI updates live
            con.execute(
                "INSERT INTO rally_labels (id, video_id, source, start_frame, end_frame, meta)"
                " VALUES (?,?,?,?,?,?)",
                (rid, video_id, source, s, e, meta_json),
            )
        last_row = {"id": rid, "start": s, "end": e}
        rallies.append((s, e))

    for f, frame, full in iter_frames(path, step=step, scale=scale, with_full=True):
        if handle.cancelled():
            raise JobCancelled()
        if roi is None and Hs is not None:
            roi = _court_mask(Hs, frame.shape[:2])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            # Motion restricted to this court — neighbouring courts and
            # spectators must not drive the gate.
            motion = float(diff[roi > 0].mean()) if roi is not None else float(diff.mean())
        else:
            motion = 0.0
        prev_gray = gray

        cands = ball_det.update(frame, last_boxes_scaled or None)
        if roi is not None:
            cands = [c for c in cands
                     if roi[min(int(c.y), roi.shape[0] - 1), min(int(c.x), roi.shape[1] - 1)] > 0]
        fast, fast_prev = _match_fast(cands, prev_cands, step, speed_min)
        prev_cands = cands
        crossed = False
        if fast is not None:
            last_ball_frame = f
            if Hs is not None and fast_prev is not None:
                cys = court.img_to_court(Hs, np.array(
                    [[fast.x, fast.y], [fast_prev.x, fast_prev.y]]))[:, 1]
                # The live-play signature: the fast blob crossed the net line.
                if (cys[0] - court.NET_Y) * (cys[1] - court.NET_Y) < 0:
                    crossed = True
                    last_cross = f

        entry: dict = {"f": f, "me": round(motion, 2)}
        if fast is not None:
            entry["ball"] = [round(fast.x / scale, 1), round(fast.y / scale, 1),
                             round(fast.diameter_px / scale, 1)]
        if crossed:
            entry["cross"] = 1

        if f - last_spot >= spot_every:
            last_spot = f
            obs = pose.update(full)  # full-res so far-side players are detectable
            sf = is_serve_formation(obs)
            entry["sf"] = sf
            # Boxes are already full-res (pose ran on `full`); overlays expect
            # full-res coords, so no scale division here. Cache the boxes scaled
            # down to the ball frame for foot suppression.
            last_boxes_scaled = [tuple(v * scale for v in o.box) for o in obs]
            entry["players"] = [
                {"id": o.track_id,
                 "box": [round(v, 1) for v in o.box],
                 "cxy": [round(o.court_xy[0], 2), round(o.court_xy[1], 2)] if o.court_xy else None,
                 "team": o.team,
                 "kp": [[round(x, 1), round(y, 1)] for x, y in o.keypoints]
                       if o.keypoints else None}
                for o in obs
            ]
        entries.append(entry)

        # ---- rolling active/dead with hysteresis --------------------------
        if baseline is None:
            baseline = max(motion, 0.05)
        elif not active:
            # Only learn the ambient level during dead time — rallies must
            # never lift the baseline (protects clips that are mostly play).
            alpha = 0.05 if motion < baseline else 0.01
            baseline += alpha * (motion - baseline)
        thresh = max(motion_floor, baseline * motion_ratio)
        entry["mt"] = round(thresh, 2)  # logged so the gate is auditable
        if Hs is not None:
            # Primary gate: live play = the ball keeps crossing the net.
            signal = (f - last_cross) <= cross_window_n
        else:
            # No calibration yet: fall back to motion + fast-blob recency.
            signal = motion > thresh and (f - last_ball_frame) <= ball_recent_n
        if signal != active:
            streak += 1
            if (not active and streak >= on_n) or (active and streak >= off_n):
                flip_at = f - streak * step  # activity actually changed back here
                if not active:
                    rally_start = max(0, flip_at)
                else:
                    if rally_start is not None:
                        close_rally(rally_start, flip_at)
                    rally_start = None
                active = not active
                streak = 0
        else:
            streak = 0

        # Flush completed overlay chunks so the tab streams while running.
        ci = overlays.chunk_index(f)
        if ci > flushed_chunk + 1:
            done = [e for e in entries if overlays.chunk_index(e["f"]) <= flushed_chunk + 1]
            if done:
                overlays.write_entries(video_id, source, done)
                entries = [e for e in entries if overlays.chunk_index(e["f"]) > flushed_chunk + 1]
            flushed_chunk = ci - 1
            # update() also raises if cancelled — a long Pass 1 yields the
            # worker promptly so shorter queued jobs can run.
            handle.update(progress=f / max(meta.frame_count, 1),
                          message=f"Pass 1: {len(rallies)} rallies so far")

    if active and rally_start is not None:
        close_rally(rally_start, meta.frame_count - 1)
    if entries:
        overlays.write_entries(video_id, source, entries)
    handle.update(progress=1.0, message=f"Pass 1 done: {len(rallies)} candidate rallies")
    log.info("pass1 %s: %d candidate rallies", video_id, len(rallies))
    return rallies

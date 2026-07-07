"""Pass 2 — precise DETECTION / VISUALIZATION inside the Pass 1 rally windows
only (~20% of the clip), at full frame rate.

Pass 2 no longer computes rally boundaries. Pass 1 is now the SOLE source of
rally start/end (see run_pass1); Pass 2's only job is to produce an accurate
per-frame overlay of player location and ball position within each Pass 1
window: full-rate player tracking, full-rate ball tracking (TrackNet if
weights are installed, otherwise the classical fallback), Kalman smoothing,
3D height reconstruction, and bounce/hit markers — all purely for display.

The rally start/end decision code (rally_logic.detect_rallies, terminal-event
end detection, learned start/end offsets, serve-vs-pass rejection, and the
trained rally_end/serve_start model hooks) is intentionally left in the tree
but is NOT called here — it can be switched back on later. Per-frame feature
traces are still embedded in the pass2 overlay entries; they remain training
data for a future learned gate/bounce model.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .. import db
from ..config import PASS2_BALL_ENABLED
from ..jobs import JobHandle
from . import overlays
from . import court
from .ball_coarse import MotionBallDetector
from .ball_track import TrajectoryBallSelector
from .ball_tracknet import TrackNetBall
from .bounce import detect_events, reconstruct_world
from .court import CameraModel
from .kalman import smooth_track
from .players import PoseTracker, is_serve_formation, is_serve_receive_formation
from .video_io import iter_frames, probe

log = logging.getLogger("cv.pass2")


def _in_court_region(H, x: float, y: float, mx: float = 2.5, my: float = 3.0) -> bool:
    """Reject ball blobs whose court-plane projection is well outside THIS court
    — neighbouring courts and spectators are several metres beyond the generous
    margins here, while airborne / just-out balls stay within. Pass 1 masks the
    court region; Pass 2 previously did not, so once the player-box exclusion
    removed the shoe blobs, the detector latched onto neighbouring-court motion
    and produced a flood of phantom 'bounce OUT' events."""
    cxy = court.img_to_court(H, np.array([[float(x), float(y)]], dtype=np.float64))[0]
    return (-mx <= cxy[0] <= court.COURT_W + mx) and (-my <= cxy[1] <= court.COURT_L + my)


def _foot_points(obs) -> list[tuple[float, float]]:
    """Ankle + knee keypoints (COCO 13-16) of every tracked player — the pixel
    locations where shoe/foot false-positive blobs cluster. Fed to the ball
    trajectory selector so a blob sitting on a foot is trusted only if it shows
    ball-like speed (a hit), never when it is stationary (a foot)."""
    pts: list[tuple[float, float]] = []
    for o in obs:
        kp = o.keypoints
        if not kp or len(kp) < 17:
            continue
        for idx in (13, 14, 15, 16):
            x, y = kp[idx][0], kp[idx][1]
            if x > 0 and y > 0:
                pts.append((float(x), float(y)))
    return pts


def run_pass2(video_id: str, params: dict, H, cam: CameraModel | None,
              handle: JobHandle, source: str = "pass2",
              p1_source: str = "pass1") -> list[dict]:
    if H is None or cam is None:
        raise RuntimeError(
            "Pass 2 needs a court calibration (homography). Calibrate the "
            "court corners in the labeling tab, then hit Reprocess.")

    with db.connect() as con:
        vrow = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        p1 = con.execute(
            "SELECT start_frame, end_frame FROM rally_labels"
            " WHERE video_id=? AND source=? ORDER BY start_frame",
            (video_id, p1_source),
        ).fetchall()
    path = vrow["path"]
    meta = probe(path)
    fps = meta.fps
    pad = int(params["p1_pad_s"] * fps)
    merge_gap = int(params["p1_merge_gap_s"] * fps)
    last_frame = meta.frame_count - 1

    # Build the region Pass 2 processes from the Pass 1 windows. This ONLY
    # defines where detection runs — it produces no boundary output (Pass 1 is
    # the sole rally source and already pads/merges its own windows).
    # 1. Merge adjacent Pass 1 windows when the raw gap between them is small.
    raw: list[list[int]] = []
    for r in p1:
        s, e = r["start_frame"], r["end_frame"]
        if raw and s - raw[-1][1] <= merge_gap:
            raw[-1][1] = max(raw[-1][1], e)
        else:
            raw.append([s, e])

    # 2. Pad each merged window, clamping the pad to at most half the gap to each
    #    neighbour so two processing windows never overlap.
    windows: list[tuple[int, int, int, int]] = []  # ws, we, raw_start, raw_end
    for i, (s, e) in enumerate(raw):
        left_gap = s - raw[i - 1][1] if i > 0 else 10**9
        right_gap = raw[i + 1][0] - e if i < len(raw) - 1 else 10**9
        ws = max(0, s - min(pad, max(0, left_gap // 2)))
        we = min(last_frame, e + min(pad, max(0, right_gap // 2)))
        windows.append((ws, we, s, e))

    # Ball detection/tracking is a switchable stage. OFF right now (PASS2_BALL=0)
    # so Pass 2 is player-detection only — faster, and safe because the ball
    # outputs are display-only and drive no rally boundaries. All the ball setup
    # and per-frame/per-window ball work below is gated on this flag.
    detect_ball = PASS2_BALL_ENABLED

    overlays.clear(video_id, source)
    # Pass 2 no longer emits its own rally boundaries — clear any stale ones so
    # Pass 1 stays the single source of truth for rally start/end.
    with db.connect() as con:
        con.execute("DELETE FROM rally_labels WHERE video_id=? AND source=?", (video_id, source))

    tracknet = TrackNetBall() if detect_ball else None
    corrections: dict[int, tuple[float, float]] = {}
    ext_track: dict[int, list] = {}
    if detect_ball:
        # User spot-corrections override the tracker at those frames (clip-local).
        with db.connect() as con:
            corr_rows = con.execute(
                "SELECT frame, x, y FROM ball_corrections WHERE video_id=?", (video_id,)
            ).fetchall()
            trk = con.execute("SELECT payload FROM ball_tracks WHERE video_id=?",
                              (video_id,)).fetchone()
        corrections = {r["frame"]: (r["x"], r["y"]) for r in corr_rows}
        # Dense per-frame ball track from an EXTERNAL tracker (MTI / temporal
        # integration, uploaded via POST /videos/{id}/ball_track). Preferred over
        # the classical detector; manual corrections still win over it. Values are
        # [x, y, diameter|null, confidence] in full-res pixels.
        ext_track = {int(k): v for k, v in db.loads(trk["payload"], {}).items()} if trk else {}
        if ext_track:
            log.info("pass2 %s: using external ball track (%d frames)", video_id, len(ext_track))

    total_frames = sum(we - ws + 1 for ws, we, _, _ in windows) or 1
    done_frames = 0

    # One PoseTracker for the whole run: track_sequence=True (consecutive frames),
    # higher imgsz for far-side players, OpenVINO on the Arc GPU. new_sequence()
    # resets ByteTrack per window WITHOUT recompiling the model (~5 s each on GPU);
    # a fresh instance per window would isolate the same way but pay that recompile.
    pose = PoseTracker(H, track_sequence=True,
                       conf=float(params.get("pose_conf", 0.25)),
                       imgsz=int(params.get("pose_imgsz", 1920)))
    for wi, (ws, we, _raw_ws, _raw_we) in enumerate(windows):
        pose.new_sequence()
        if detect_ball:
            fallback = MotionBallDetector(
                min_area=2.0, max_area=600.0,
                sat_min=float(params.get("ball_sat_min", 0.0)),
                val_min=float(params.get("ball_val_min", 0.0)),
                hue_lo=float(params.get("ball_hue_lo", 0.0)),
                hue_hi=float(params.get("ball_hue_hi", 180.0)))
            # Motion-primary ball selection (see ball_track.py): the raw detector is
            # left PERMISSIVE (no player-box exclusion) so a ball passing over/leaving
            # a player still reaches the selector, which then keeps it only if it fits
            # a plausible ball trajectory and rejects slow foot/shoe blobs.
            selector = TrajectoryBallSelector(
                fps,
                gate_px=float(params.get("ball_gate_px", 45.0)),
                accel_px=float(params.get("ball_accel_px", 6.0)),
                foot_px=float(params.get("ball_foot_suppress_px", 35.0)),
                hit_speed_px=float(params.get("ball_hit_speed_px", 18.0)),
                acq_px=float(params.get("ball_acq_px", 90.0)),
                seed_min_px=float(params.get("ball_seed_min_px", 4.0)),
            )
        # Serve-start gate: proper serve-receive (server + diagonal receiver both
        # behind their baselines) when require_serve_receive is on, else the
        # looser 'one up, rest back' check. NOTE this is stricter and depends on
        # the far receiver being detected — hence the pose imgsz bump above.
        if params.get("require_serve_receive", 1.0) > 0.5:
            margin = float(params.get("serve_baseline_margin_m", 0.75))
            serve_ok = lambda obs: is_serve_receive_formation(obs, margin)
        else:
            serve_ok = is_serve_formation
        ball_obs: list[tuple[int, float, float, float | None]] = []
        players_by_frame: dict[int, list[tuple[float, float]]] = {}
        player_speed: dict[int, float] = {}     # frame -> mean player speed m/s
        serve_frames: list[int] = []
        prev_courts: dict[int, tuple[float, float]] = {}
        frame_buf: list[np.ndarray] = []
        entries: dict[int, dict] = {}

        for f, frame in iter_frames(path, start_frame=ws, end_frame=we + 1):
            obs = pose.update(frame)
            courts = [o.court_xy for o in obs if o.court_xy]
            players_by_frame[f] = courts
            speeds = []
            for o in obs:
                if o.court_xy and o.track_id in prev_courts:
                    px, py = prev_courts[o.track_id]
                    speeds.append(math.hypot(o.court_xy[0] - px, o.court_xy[1] - py) * fps)
            prev_courts = {o.track_id: o.court_xy for o in obs if o.court_xy}
            player_speed[f] = float(np.mean(speeds)) if speeds else 0.0

            if f % max(1, int(fps * 0.5)) == 0 and serve_ok(obs):
                serve_frames.append(f)
                pose.assign_numbers_at_serve(obs)

            # ---- ball observation ----------------------------------------
            # Priority: manual correction > external track > TrackNet > classical.
            # Disabled while focusing on player detection (PASS2_BALL=0): Pass 2's
            # ball outputs are display-only and feed no rally boundaries.
            det = None
            if detect_ball:
                if f in corrections:
                    x, y = corrections[f]
                    det = (x, y, None, 1.0)
                elif f in ext_track:
                    ex = ext_track[f]
                    det = (ex[0], ex[1], ex[2], ex[3] if ex[3] is not None else 0.9)
                elif tracknet.available:
                    frame_buf.append(frame)
                    if len(frame_buf) > 3:
                        frame_buf.pop(0)
                    if len(frame_buf) == 3:
                        r = tracknet.infer_triplet(frame_buf)
                        if r:
                            det = r
                else:
                    # Permissive detector (no box exclusion) + trajectory selector.
                    cands = [cd for cd in fallback.update(frame)
                             if _in_court_region(H, cd.x, cd.y)]
                    boxes = [o.box for o in obs]
                    feet = _foot_points(obs)
                    c = selector.select(f, cands, boxes, feet)
                    if c:
                        det = (c.x, c.y, c.diameter_px, c.score)
                if det:
                    ball_obs.append((f, det[0], det[1], det[2]))
                    # The classical path updates the selector inside select(); for
                    # any other source seed the trajectory history so continuity
                    # holds if ball sources mix mid-window.
                    classical = not (f in corrections or f in ext_track or tracknet.available)
                    if not classical:
                        selector.seed(f, det[0], det[1])

            entries[f] = {
                "f": f,
                "players": [
                    {"id": o.track_id, "num": o.number,
                     "box": [round(v, 1) for v in o.box],
                     "cxy": [round(o.court_xy[0], 2), round(o.court_xy[1], 2)] if o.court_xy else None,
                     "team": o.team,
                     "kp": [[round(x, 1), round(y, 1)] for x, y in o.keypoints]
                           if o.keypoints else None}
                    for o in obs
                ],
                "pspeed": round(player_speed[f], 2),
            }
            if det:
                entries[f]["ball"] = [round(det[0], 1), round(det[1], 1),
                                      round(det[2], 1) if det[2] else None,
                                      round(det[3], 2)]
            done_frames += 1
            if done_frames % 100 == 0:
                # update() also raises if the job was cancelled — keeps a
                # long Pass 2 from blocking the queue after a cancel.
                handle.update(progress=done_frames / total_frames,
                              message=f"Pass 2: window {wi + 1}/{len(windows)}")

        # ---- window-level ball reasoning (VISUALIZATION ONLY) -------------
        # No rally boundaries are computed here anymore. We only refine the
        # per-frame detection overlay: smooth the ball track, reconstruct its
        # court position + height, and mark bounce/hit events for display. The
        # rally-decision code (rally_logic.detect_rallies, learned start/end
        # offsets, serve-vs-pass rejection) is deliberately not called. Skipped
        # entirely when ball detection is off (PASS2_BALL=0).
        if detect_ball:
            track = smooth_track(ball_obs, max_gap=int(fps * 0.4),
                                 gate2=float(params.get("ball_kalman_gate2", 13.8)))
            world = reconstruct_world(track, cam)
            for w in world:  # feature trace: attach reconstruction to entries
                e = entries.get(w.frame)
                if e is not None:
                    e["world"] = [round(w.court_x, 2), round(w.court_y, 2),
                                  None if math.isnan(w.height) else round(w.height, 3)]
                    e["obs"] = w.observed
            # Bounce/hit markers are kept purely as visual annotations (model=None:
            # the trained rally_end model no longer drives anything in this pass).
            events = detect_events(world, players_by_frame, params, fps,
                                   model=None, entries=entries)
            for ev in events:
                e = entries.get(ev.frame)
                if e is not None:
                    e["evt"] = {"kind": ev.kind, "in": ev.in_court, "side": ev.side,
                                "cx": round(ev.court_x, 2), "cy": round(ev.court_y, 2)}
        overlays.write_entries(video_id, source, list(entries.values()))

    handle.update(progress=1.0,
                  message=f"Pass 2 done: detection overlays for {len(windows)} window(s)")
    log.info("pass2 %s: detection overlays for %d windows", video_id, len(windows))
    return []

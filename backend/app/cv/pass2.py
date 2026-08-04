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
from types import SimpleNamespace

import numpy as np

from .. import db, results_export, storage_s3
from ..config import PASS2_BALL_ENABLED, STORAGE_BACKEND
from ..jobs import JobHandle
from . import overlays
from . import court
from .ball_coarse import MotionBallDetector, on_player_box
from .ball_filter import BallFrameFilter, classify_arc_events, refine_track
from .ball_track import TrajectoryBallSelector
from .ball_tracknet import TrackNetBall
from .exclusions import in_excluded
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


def _wrist_points(obs) -> list[tuple[float, float]]:
    """Wrist keypoints (COCO 9,10) of every tracked player — the closest proxy
    to a paddle/contact point (no paddle keypoint exists in COCO-17). Fed to
    the ball selector's limb-lock guard so a candidate moving in lockstep with
    a wrist (a limb, not an airborne ball) is excluded."""
    pts: list[tuple[float, float]] = []
    for o in obs:
        kp = o.keypoints
        if not kp or len(kp) < 17:
            continue
        for idx in (9, 10):
            x, y = kp[idx][0], kp[idx][1]
            if x > 0 and y > 0:
                pts.append((float(x), float(y)))
    return pts


def _upload_window_incremental(video_id: str, source: str, window_index: int,
                                start_frame: int, end_frame: int,
                                total_windows: int) -> None:
    """S3 mode only: as soon as one Pass 2 window's overlays are flushed
    (just above), ship a scoped snapshot to S3 as results/{video_id}/
    window_{n}.json — live visibility for the local app (see
    api/media.py's poll-cloud-windows) without waiting for the whole chain
    to finish. Never fails the pass: a transient S3 hiccup here must not
    cost a user their Pass 2 run, and the chain-level export in
    pipeline._finish_chain still captures everything at the end regardless."""
    try:
        payload = results_export.export_window_results(video_id, source, window_index,
                                                          start_frame, end_frame,
                                                          total_windows=total_windows)
        storage_s3.upload_window_results(video_id, window_index, payload)
    except Exception:
        log.exception("incremental window %d S3 export failed for %s (non-fatal)",
                      window_index, video_id)


def run_pass2(video_id: str, params: dict, H, cam: CameraModel | None,
              handle: JobHandle, source: str = "pass2",
              p1_source: str = "pass1",
              on_window=None,
              windows_override: list[tuple[int, int]] | None = None) -> list[dict]:
    """``on_window(last_covered_frame)`` fires after each processing window's
    overlays land — the pipeline hooks the incremental refiner here so Pass 2
    results stream into the UI per window instead of arriving all at the end.

    ``windows_override``: when given, Pass 2 runs over exactly these
    (start_frame, end_frame) ranges instead of querying rally_labels — used
    for a user-specified coaching selection that doesn't come from a Pass 1
    window at all. Bypasses the merge/pad logic below entirely (that logic
    exists to stitch adjacent Pass 1 windows together; a single caller-given
    range needs neither). Leaves default (None) behaviour byte-for-byte
    unchanged."""
    if H is None or cam is None:
        raise RuntimeError(
            "Pass 2 needs a court calibration (homography). Calibrate the "
            "court corners in the labeling tab, then hit Reprocess.")
    if cam.focal_degenerate:
        # Degenerate focal self-calibration = every ball height this run is
        # built on a guessed focal. Surface it in the job, not just the log.
        log.warning("pass2 %s: focal self-calibration was DEGENERATE — ball "
                    "heights are unreliable; re-click the court corners.", video_id)
        handle.update(message="WARNING: degenerate court calibration — ball "
                              "heights unreliable (re-click court corners)")

    with db.connect() as con:
        vrow = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        p1 = [] if windows_override is not None else con.execute(
            "SELECT start_frame, end_frame FROM rally_labels"
            " WHERE video_id=? AND source=? ORDER BY start_frame",
            (video_id, p1_source),
        ).fetchall()
    path = vrow["path"]
    meta = probe(path)
    fps = meta.fps
    last_frame = meta.frame_count - 1

    if windows_override is not None:
        # Caller-supplied range(s) — no Pass 1 windows to merge/pad. Clamp to
        # the clip's real bounds, same as the merge/pad path below does.
        windows: list[tuple[int, int, int, int]] = [
            (max(0, s), min(last_frame, e), max(0, s), min(last_frame, e))
            for s, e in windows_override]
    else:
        pad = int(params["p1_pad_s"] * fps)
        merge_gap = int(params["p1_merge_gap_s"] * fps)

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
        #    neighbour so two processing windows never overlap. The (gap-1)//2 clamp
        #    matters: with gap//2 on both sides an EVEN gap made adjacent windows
        #    share one frame (we_prev == ws_next, both inclusive), so that frame was
        #    processed twice and its overlay entry written by both windows.
        windows = []  # ws, we, raw_start, raw_end
        for i, (s, e) in enumerate(raw):
            left_gap = s - raw[i - 1][1] if i > 0 else 10**9
            right_gap = raw[i + 1][0] - e if i < len(raw) - 1 else 10**9
            ws = max(0, s - min(pad, max(0, (left_gap - 1) // 2)))
            we = min(last_frame, e + min(pad, max(0, right_gap // 2)))
            windows.append((ws, we, s, e))

    # Ball detection/tracking is a switchable stage. OFF right now (PASS2_BALL=0)
    # so Pass 2 is player-detection only — faster, and safe because the ball
    # outputs are display-only and drive no rally boundaries. All the ball setup
    # and per-frame/per-window ball work below is gated on this flag.
    detect_ball = PASS2_BALL_ENABLED
    # Phase-1 semantic filtering layer (ball_filter.py) on top of whichever
    # ball source runs: velocity gate + player-box motion logic per frame,
    # then per-window arc RANSAC + hit/bounce classification. ON by default;
    # ball_filter_enabled=0 falls back to the previous heuristic unchanged.
    filter_on = detect_ball and params.get("ball_filter_enabled", 1.0) > 0.5

    overlays.clear(video_id, source)
    # Pass 2 no longer emits its own rally boundaries — clear any stale ones so
    # Pass 1 stays the single source of truth for rally start/end.
    with db.connect() as con:
        con.execute("DELETE FROM rally_labels WHERE video_id=? AND source=?", (video_id, source))

    tracknet = TrackNetBall(
        conf_threshold=float(params.get("tracknet_conf", 0.5))) if detect_ball else None
    use_tracknet = bool(tracknet and tracknet.available)
    # Court-region margins for the ball blob filter (were hardcoded 2.5/3.0).
    ball_mx = float(params.get("ball_court_margin_x_m", 2.5))
    ball_my = float(params.get("ball_court_margin_y_m", 3.0))
    # Opt-in candidate suppressors (see params.py for defaults/rationale):
    #  - box suppression: drop any classical CANDIDATE inside a player box before
    #    the trajectory selector ever sees it. Default OFF — it would strip the
    #    on-player blobs ball_seed_on_player needs to recover a cold-start serve.
    #  - exclusion zones: the same image-space deadzone triangles used for pose
    #    (neighbouring courts / background), applied to ball candidates too.
    ball_suppress_in_box = params.get("ball_suppress_in_box", 0.0) > 0.5
    ball_use_exclusions = params.get("ball_exclusion_zones", 1.0) > 0.5
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
    implied_heights: list[float] = []  # calibration cross-check samples
    for wi, (ws, we, _raw_ws, _raw_we) in enumerate(windows):
        pose.new_sequence()
        if detect_ball:
            fallback = MotionBallDetector(
                min_area=2.0, max_area=600.0,
                sat_min=float(params.get("ball_sat_min", 0.0)),
                val_min=float(params.get("ball_val_min", 0.0)),
                hue_lo=float(params.get("ball_hue_lo", 0.0)),
                hue_hi=float(params.get("ball_hue_hi", 180.0)),
                motion_thresh=int(params.get("ball_motion_thresh", 12.0)),
                min_circularity=float(params.get("ball_min_circularity", 0.35)))
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
                seed_on_player=bool(params.get("ball_seed_on_player", False)),
                suppress_on_player=params.get("ball_suppress_on_player", 1.0) > 0.5,
                lock_dist_px=float(params.get("limb_lock_dist_px", 30.0)),
                lock_rel_vel_px=float(params.get("limb_lock_rel_vel_px", 4.0)),
                lock_frames=int(params.get("limb_lock_frames", 3)),
            )
            # Bootstrap from Pass 1's coarse ball hits (sparse but real motion) so
            # the trajectory selector isn't cold at the very start of the window.
            n_seeded = 0
            for e in overlays.read_range(video_id, "pass1", ws, we):
                if "ball" in e and len(e["ball"]) >= 2:
                    selector.seed(e["f"], float(e["ball"][0]), float(e["ball"][1]))
                    n_seeded += 1
            if n_seeded:
                log.info("pass2 %s w%d: seeded ball track from %d Pass 1 points",
                         video_id, wi + 1, n_seeded)
            frame_filter = BallFrameFilter(
                H, fps,
                max_speed_mph=float(params.get("ball_max_speed_mph", 65.0)),
                headroom=float(params.get("ball_speed_headroom", 1.6)),
                persist_frames=int(params.get("ball_box_persist_frames", 3)),
                persist_match_px=float(params.get("ball_box_persist_px", 12.0)),
                foot_zone_frac=float(params.get("ball_foot_zone_frac", 0.25)),
            ) if filter_on else None
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
        boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
        player_speed: dict[int, float] = {}     # frame -> mean player speed m/s
        serve_frames: list[int] = []
        prev_courts: dict[int, tuple[float, float]] = {}
        frame_buf: list[np.ndarray] = []
        entries: dict[int, dict] = {}

        for f, frame in iter_frames(path, start_frame=ws, end_frame=we + 1):
            obs = pose.update(frame)
            courts = [o.court_xy for o in obs if o.court_xy]
            players_by_frame[f] = courts
            if detect_ball:
                boxes_by_frame[f] = [o.box for o in obs]
            # Key speed continuity on the stable identity slot (falls back to the
            # ByteTrack id) so an id-switch doesn't register as a teleport.
            def _key(o):
                return o.slot if o.slot is not None else o.track_id
            speeds = []
            for o in obs:
                if o.court_xy and _key(o) in prev_courts:
                    px, py = prev_courts[_key(o)]
                    speeds.append(math.hypot(o.court_xy[0] - px, o.court_xy[1] - py) * fps)
            prev_courts = {_key(o): o.court_xy for o in obs if o.court_xy}
            player_speed[f] = float(np.mean(speeds)) if speeds else 0.0

            if f % max(1, int(fps * 0.5)) == 0:
                if serve_ok(obs):
                    serve_frames.append(f)
                    pose.assign_numbers_at_serve(obs)
                # Calibration cross-check sample: implied standing-player height
                # from foot court position + head pixel. Aggregated after the
                # run — a bad-but-not-degenerate calibration (wrong self-
                # calibrated focal from imperfect corner clicks) shifts the
                # whole distribution out of the plausible human band.
                for o in obs:
                    if o.court_xy is None:
                        continue
                    hgt, resid = court.implied_person_height(
                        cam, o.court_xy, ((o.box[0] + o.box[2]) / 2, o.box[1]))
                    if resid < 0.5 and not math.isnan(hgt):
                        implied_heights.append(hgt)

            # ---- ball observation ----------------------------------------
            # Priority: manual correction > external track > TrackNet > classical.
            # Disabled while focusing on player detection (PASS2_BALL=0): Pass 2's
            # ball outputs are display-only and feed no rally boundaries.
            det = None
            if detect_ball:
                h_px, w_px = frame.shape[:2]
                # Keep the ACTIVE lower-priority detector's temporal state warm on
                # EVERY frame, even ones covered by a correction or the external
                # track. Both detectors difference consecutive frames (TrackNet:
                # 3-frame buffer; classical: 3-frame diff); skipping covered
                # frames left them holding non-consecutive frames and produced
                # garbage on the first frames after every coverage gap.
                raw_cands = None
                if use_tracknet:
                    frame_buf.append(frame)
                    if len(frame_buf) > 3:
                        frame_buf.pop(0)
                else:
                    raw_cands = fallback.update(frame)
                if f in corrections:
                    x, y = corrections[f]
                    det = (x, y, None, 1.0)
                elif f in ext_track:
                    ex = ext_track[f]
                    det = (ex[0], ex[1], ex[2], ex[3] if ex[3] is not None else 0.9)
                elif use_tracknet:
                    if len(frame_buf) == 3:
                        r = tracknet.infer_triplet(frame_buf)
                        # Court-region gate, same margins as the classical branch
                        # below — TrackNet was only gated by the classical path's
                        # filter, so it could output neighbouring-court/spectator
                        # points the classical path would have dropped.
                        if r and _in_court_region(H, r[0], r[1], ball_mx, ball_my):
                            # Diameter is deliberately dropped: TrackNet's heatmap
                            # blob extent reflects its fixed training label sigma,
                            # NOT the ball's physical apparent size, so feeding it
                            # to size->depth produces confidently wrong heights.
                            # Position + conf only; height honestly degrades to
                            # plane projection instead.
                            det = (r[0], r[1], None, r[3])
                else:
                    # Permissive detector (no box exclusion) + trajectory selector.
                    # Drop blobs in the fixed camera's exclusion zones (the red
                    # corner triangles — background / fence-line motion) BEFORE the
                    # selector sees them, so a deadzone blob can never seed or
                    # pollute the ball trajectory. Same zones that gate players
                    # (see cv/exclusions.py). ball_suppress_in_box additionally
                    # drops candidates on a player box (default off — see params.py).
                    boxes = [o.box for o in obs]
                    cands = [
                        cd for cd in raw_cands
                        if _in_court_region(H, cd.x, cd.y, ball_mx, ball_my)
                        and not (ball_use_exclusions
                                 and in_excluded(cd.x, cd.y, w_px, h_px))
                        and not (ball_suppress_in_box
                                 and on_player_box(cd.x, cd.y, boxes))]
                    if frame_filter is not None:
                        # Stage 1-2 of the filtering layer: velocity gate +
                        # player-box motion logic (foot zone / persistent blobs).
                        cands = frame_filter.filter(
                            f, cands, [(_key(o), o.box) for o in obs])
                    feet = _foot_points(obs)
                    c = selector.select(f, cands, boxes, feet, _wrist_points(obs))
                    if c is None and cands and selector.near_track(f, cands[0].x, cands[0].y):
                        # Last resort for classical-only: fill with the best blob
                        # so the overlay isn't blank across a brief miss. Gated by
                        # near_track so it only fills NEAR the established
                        # trajectory — never paints a random round noise blob
                        # anywhere in the court when the ball is momentarily lost.
                        c = cands[0]
                        selector.seed(f, c.x, c.y)
                    if c:
                        det = (c.x, c.y, c.diameter_px, c.score)
                # Exclusion zones apply to ANY non-manual detector: an external /
                # TrackNet position landing in a red corner triangle is dropped
                # just like the classical candidates were above (the classical
                # path already filtered at the candidate level). Manual
                # corrections stay authoritative.
                if (ball_use_exclusions and det is not None and f not in corrections
                        and in_excluded(det[0], det[1], w_px, h_px)):
                    det = None
                # The filtering layer applies to ANY detector: external/TrackNet
                # dets pass the same stage 1-2 gates the classical candidates
                # did above. Manual corrections stay authoritative.
                if (det is not None and frame_filter is not None
                        and f not in corrections
                        and (f in ext_track or use_tracknet)):
                    cand = SimpleNamespace(x=det[0], y=det[1])
                    if not frame_filter.filter(f, [cand], [(_key(o), o.box) for o in obs]):
                        det = None
                if det:
                    ball_obs.append((f, det[0], det[1], det[2]))
                    if frame_filter is not None:
                        # Whatever source produced the position, it is the
                        # velocity-gate reference for the next frames.
                        frame_filter.confirm(f, det[0], det[1])
                    # The classical path updates the selector inside select(); for
                    # any other source seed the trajectory history so continuity
                    # holds if ball sources mix mid-window.
                    classical = not (f in corrections or f in ext_track or use_tracknet)
                    if not classical:
                        selector.seed(f, det[0], det[1])

            entries[f] = {
                "f": f,
                "players": [
                    {"id": o.slot if o.slot is not None else o.track_id, "num": o.number,
                     "box": [round(v, 1) for v in o.box],
                     "cxy": [round(o.court_xy[0], 2), round(o.court_xy[1], 2)] if o.court_xy else None,
                     "team": o.team,
                     "kp": [[round(x, 1), round(y, 1)] for x, y in o.keypoints]
                           if o.keypoints else None,
                     # Parallel to "kp" (same COCO-17 indices) — YOLO's raw
                     # per-keypoint confidence, previously discarded. Additive:
                     # "kp" itself is unchanged, so every existing reader of it
                     # keeps working untouched.
                     "kp_conf": [round(c, 3) for c in o.kp_conf]
                                if o.kp_conf else None}
                    for o in obs
                    # identity.py caps real identity slots at MAX_PER_TEAM (2) per
                    # team for the whole match, and force-binds every detection to
                    # an existing slot once both are filled — so a team-tagged obs
                    # that still has slot=None here is NOT a genuine re-ID miss, it
                    # is a detection identity.py deliberately rejected as noise (a
                    # 3rd+ same-team box in one frame: false positive / bystander
                    # inside the court margin). Falling back to its raw track_id
                    # used to fabricate a stable-looking id for that noise, which
                    # fragmented one physical player into extra phantom ids across
                    # window/track-id churn. Drop it instead. Untamed obs (team is
                    # None — off-court projection) still fall back to track_id, as
                    # before, purely for display; they never carry a team so they
                    # can't pollute team-based analytics either way.
                    if o.slot is not None or o.team is None
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
            # Stage 3-4 of the filtering layer (batch, per window): per-shot arc
            # RANSAC that drops off-arc outliers and fills missed frames along
            # the fit, then hit/bounce classification from arc junctions +
            # player boxes. Replaces the height-heuristic events when on.
            arc_events = None
            if filter_on and ball_obs:
                refined = refine_track(
                    ball_obs,
                    split_deg=float(params.get("ball_arc_split_deg", 55.0)),
                    min_speed_px=float(params.get("ball_arc_min_speed_px", 1.5)),
                    gap_frames=int(params.get("ball_arc_gap_frames", 10)),
                    tol_px=float(params.get("ball_arc_tol_px", 6.0)),
                    min_arc_pts=int(params.get("ball_arc_min_pts", 4)),
                    join_gap=int(params.get("ball_arc_join_gap_frames", 8)),
                    split_dv_px=float(params.get("ball_arc_split_dv_px", 4.0)),
                    claim_tol_mult=float(params.get("ball_arc_claim_tol_mult", 2.5)),
                    max_interp_gap=int(params.get("ball_arc_max_interp_gap", 6)))
                kept = {p.frame for p in refined.points}
                # Only prune points AGAINST a fitted arc — with zero arcs in
                # this window (too few/sparse real detections ever cluster
                # into ball_arc_min_pts within ball_arc_gap_frames, common for
                # the far-baseline ball) there is no trajectory to judge
                # "off-arc" against, so every already-picked per-frame
                # detection would otherwise get wiped here, turning a sparse-
                # but-real ball track into a silent, indistinguishable-from-
                # never-detected blank for the whole window (verified: an
                # EC2 clip with 13-50 raw candidates/window and 0 arcs in
                # EVERY window ended up with 0/7413 overlay ball entries).
                if refined.arcs:
                    for bf, _, _, _ in ball_obs:      # picks the fits rejected
                        if bf not in kept and bf in entries:
                            entries[bf].pop("ball", None)
                for p in refined.interpolated:    # fit-filled misses
                    e = entries.get(p.frame)
                    if e is not None:
                        e["ball"] = [round(p.x, 1), round(p.y, 1),
                                     round(p.diameter_px, 1) if p.diameter_px else None,
                                     0.0]
                        e["interp"] = True
                ball_obs = [(p.frame, p.x, p.y, p.diameter_px)
                            for p in refined.points]
                arc_events = classify_arc_events(
                    refined.arcs, boxes_by_frame,
                    join_gap=int(params.get("ball_arc_join_gap_frames", 8)),
                    min_angle_deg=float(params.get("ball_event_min_angle_deg", 25.0)),
                    adjacent_frac=float(params.get("ball_hit_adjacent_frac", 0.30)),
                    vertex_min_px=float(params.get("ball_bounce_vertex_min_px", 4.0)),
                    plausibility=params.get("ball_bounce_plausibility", 1.0) > 0.5,
                    vy_frames=int(params.get("ball_bounce_vy_frames", 3)),
                    vy_min_px=float(params.get("ball_bounce_vy_min_px", 0.5)),
                    vy_trust_span=int(params.get("ball_bounce_vy_trust_span", 7)),
                    merge_frames=int(params.get("ball_bounce_merge_frames", 6)))
                if frame_filter is not None:
                    st = frame_filter.stats
                    log.info(
                        "pass2 %s w%d ball filter: %d/%d cands rejected "
                        "(vel %d, foot %d, persist %d); %d arcs, %d outliers, "
                        "%d noise pts, %d interpolated, %d events",
                        video_id, wi + 1, st.total(), st.seen, st.velocity,
                        st.foot_zone, st.persistent, len(refined.arcs),
                        refined.n_outliers, refined.n_fragments,
                        len(refined.interpolated), len(arc_events))
            track = smooth_track(ball_obs, max_gap=int(fps * 0.4),
                                 gate2=float(params.get("ball_kalman_gate2", 13.8)))
            world = reconstruct_world(
                track, cam,
                height_sigma_max=float(params.get("ball_height_max_sigma_m", 0.35)))
            for w in world:  # feature trace: attach reconstruction to entries
                e = entries.get(w.frame)
                if e is not None:
                    e["world"] = [round(w.court_x, 2), round(w.court_y, 2),
                                  None if math.isnan(w.height) else round(w.height, 3)]
                    e["obs"] = w.observed
            if arc_events is not None:
                # Filtering layer ON: events come from arc junctions + player
                # boxes (plan stage 4). Court position prefers the smoothed
                # world track at the event frame; else project the junction
                # point through the homography (exact for bounces — on floor).
                wbf = {w.frame: w for w in world}
                for ev in arc_events:
                    e = entries.get(ev.frame)
                    if e is None:
                        continue
                    w = wbf.get(ev.frame)
                    if w is not None:
                        ecx, ecy = w.court_x, w.court_y
                    else:
                        cxy = court.img_to_court(H, np.array([[ev.x, ev.y]]))[0]
                        ecx, ecy = float(cxy[0]), float(cxy[1])
                    e["evt"] = {"kind": ev.kind,
                                "in": bool(court.in_court(np.array([[ecx, ecy]]))[0]),
                                "side": court.side_of_net(ecy),
                                "cx": round(ecx, 2), "cy": round(ecy, 2)}
            else:
                # Legacy heuristic (ball_filter_enabled=0): height-reversal
                # bounce/hit detection. Kept purely as visual annotations
                # (model=None: no trained model drives anything in this pass).
                # H + observed pixels let BOUNCE positions come from the plane
                # homography (exact on the floor) instead of noisy size->depth.
                ball_px = {bf: (bx, by) for bf, bx, by, _ in ball_obs}
                events = detect_events(world, players_by_frame, params, fps,
                                       model=None, entries=entries, H=H,
                                       ball_px=ball_px)
                for ev in events:
                    e = entries.get(ev.frame)
                    if e is not None:
                        e["evt"] = {"kind": ev.kind, "in": ev.in_court, "side": ev.side,
                                    "cx": round(ev.court_x, 2), "cy": round(ev.court_y, 2)}
        overlays.write_entries(video_id, source, list(entries.values()))
        if STORAGE_BACKEND == "s3":
            _upload_window_incremental(video_id, source, wi, ws, we, len(windows))
        if on_window is not None:
            on_window(we)

    # Calibration cross-check verdict: standing players must reconstruct to
    # plausible human heights. Outside the band = the homography+focal pair is
    # wrong for this clip (bad corner clicks), and every ball height with it.
    calib_note = ""
    if len(implied_heights) >= 20:
        med_h = float(np.median(implied_heights))
        if not (1.2 <= med_h <= 2.2):
            calib_note = (f" — WARNING: implied player height {med_h:.2f} m is "
                          f"outside the human band; calibration is suspect, "
                          f"re-click the court corners")
            log.warning("pass2 %s: calibration cross-check FAILED — median "
                        "implied player height %.2f m (n=%d). Ball heights and "
                        "far-court positions are unreliable for this clip.",
                        video_id, med_h, len(implied_heights))
        else:
            log.info("pass2 %s: calibration cross-check ok (median implied "
                     "player height %.2f m, n=%d)", video_id, med_h,
                     len(implied_heights))

    handle.update(progress=1.0,
                  message=f"Pass 2 done: detection overlays for {len(windows)} "
                          f"window(s){calib_note}")
    log.info("pass2 %s: detection overlays for %d windows", video_id, len(windows))
    return []

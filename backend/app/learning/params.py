"""LearnedParams: the versioned, GLOBAL tunable state that makes accuracy
compound. Every upload loads the latest version before Pass 1 runs, so all
prior lessons apply to every future clip automatically.

Routing rule (critical): method-level feedback updates these shared params
(global). Clip-local oddities (bumped camera that day, one-frame ball fix)
must NOT touch this baseline — they live on the clip row instead.
"""
from __future__ import annotations

from typing import Any

from .. import db

DEFAULTS: dict[str, Any] = {
    # ---- Pass 1 coarse gate ------------------------------------------------
    "p1_frame_step": 3,            # sample every Nth frame
    "p1_scale": 0.4,               # spatial downscale for motion/ball signals
    "p1_motion_ratio": 1.6,        # motion must exceed ambient baseline x this
    "p1_motion_floor": 0.12,       # absolute floor on the adaptive threshold
    "p1_ball_recent_s": 1.2,       # fast ball seen within this = ball activity
    "p1_ball_speed_px": 3.5,       # min blob speed (px/frame @ p1_scale) to count as the ball
    "p1_on_s": 0.7,                # sustained activity to flip dead->active
    "p1_off_s": 2.0,               # sustained quiet to flip active->dead
    "p1_pad_s": 1.5,               # padding handed to Pass 2 (brief: ~1-2 s)
    "p1_merge_gap_s": 0.5,         # merge adjacent Pass 1 windows only if raw gap <= this
    "p1_cross_window_s": 2.0,      # net-cross recency that keeps the calibrated gate open
    "p1_min_rally_s": 2.0,         # discard blips shorter than a real rally
    "p1_spot_check_s": 2.0,        # serve-formation pose spot-check interval
    "p1_end_settle_s": 0.6,        # rally end = last net crossing + this settle (flight-to-land)
                                   # instead of the hysteresis flip (~cross_window+off later)
    "p1_cross_max_gap_s": 1.0,     # max gap between the two straddling side-observations for a
                                   # net crossing to register (a dropout longer than this drops
                                   # the crossing, feeding the close/tight-end decisions)
    "p1_net_margin_m": 0.5,        # court-metre deadzone around the net for side classification
    "p1_cross_cluster_gap_s": 2.0, # trailing crossings farther apart than this are trimmed off
                                   # the tight rally end (close_rally)
    # ---- Pose detection (players) -----------------------------------------
    "pose_conf": 0.25,             # YOLO-pose confidence floor (lower = catch far players)
    "pose_imgsz": 1920,            # Pass 2 pose inference size. Runs on the Arc GPU via
                                   # OpenVINO (see players._openvino_pose_dir), which makes
                                   # a big input cheap. MEASURED on "First 3 mins" (GPU):
                                   # far-player recall 1280->35%, 1536->87%, 1920->100% at
                                   # 43/55/74 ms/frame (vs 193 ms for the OLD torch-on-CPU
                                   # path at 1280/35%). The 2nd far player in doubles is
                                   # OCCLUDED in this axial view (rear hides behind front) —
                                   # nano@2560 and yolo11s@1920 both still find 0, so it's a
                                   # camera-geometry limit, not resolution; 1920 is the knee.
    "p1_pose_imgsz": 1280,         # Pass 1 pose size (sparse spot checks; can be large)
    # ---- Pass 2 ball trajectory selection ---------------------------------
    "ball_gate_px": 45.0,          # residual radius around the velocity prediction
    "ball_accel_px": 6.0,          # extra gate radius per frame^2 of track gap
    "ball_acq_px": 90.0,           # acquisition radius while ball velocity is unknown
    "ball_seed_min_px": 4.0,       # min px/frame a cold-start candidate must MOVE before it
                                   # can seed a track (rejects stationary shoe/short blobs)
    "ball_motion_thresh": 3.0,     # three-frame diff threshold (0-255). 12 is fine for
                                   # Pass 1's low-res subsample; full-res 60 fps clips often
                                   # need ~3-5 or the classical detector sees zero motion.
    "ball_min_circularity": 0.30,  # blob roundness floor (area / pi*r^2)
    "ball_suppress_in_box": 0.0,   # 1 = drop ball CANDIDATES inside a detected player box
                                   # (expanded by shoe/paddle margins) before the trajectory
                                   # selector ever sees them. Default OFF: this pre-filter
                                   # would strip the on-player blobs seed_on_player relies on
                                   # to recover a serve struck from the server's own box at
                                   # cold start (see ball_seed_on_player) — turn on only if
                                   # you don't need that recovery and want harder box muting.
    "ball_exclusion_zones": 1.0,   # apply the pose image-space deadzone triangles
                                   # (exclusions.py) to ball candidates too
    "ball_foot_suppress_px": 35.0, # radius around foot keypoints to suppress blobs
    "ball_hit_speed_px": 18.0,     # px/frame a near-player blob must move to count as a hit
    "ball_suppress_on_player": 1.0,  # 1 = drop ball blobs on a player box/foot unless moving
                                   # fast; 0 = keep them (recovers a noisier track when the
                                   # classical detector's blobs cluster near players).
                                   # Independent of ball_seed_on_player: that one permits an
                                   # on-player COLD-START seed; this one gates the established-
                                   # trajectory path.
    "ball_seed_on_player": 0.0,    # 1 = allow cold-start acquisition to seed from a blob ON a
                                   # player (still gated on hit_speed) — recovers the SERVE
                                   # ball, struck at the server, which cold start otherwise hides
    # Pose-aware limb-lock ball exclusion (cv/ball_track.py): a candidate that
    # rides a wrist at ~constant offset (same velocity) for limb_lock_frames
    # consecutive frames is a limb/paddle blob, not the ball, and is dropped —
    # regardless of how it was acquired. Catches what a single-frame hit-speed
    # check at acquisition can't (e.g. a limb blob that MATCHED hit_speed once).
    "limb_lock_dist_px": 30.0,     # a candidate this close to a wrist is a lock candidate
    "limb_lock_rel_vel_px": 4.0,   # max |cand - wrist| displacement diff = moving in lockstep
    "limb_lock_frames": 3,         # consecutive lockstep frames before a candidate is excluded
    # Classical-fallback appearance gate (OpenCV HSV; H 0-180, S/V 0-255). OFF by
    # default (sat/val floors 0, full hue range) because ball colour varies per
    # clip — set once you know the ball colour to cheaply drop off-colour blobs.
    "ball_sat_min": 0.0,           # min mean saturation of a ball blob
    "ball_val_min": 0.0,           # min mean brightness/value of a ball blob
    "ball_hue_lo": 0.0,            # ball hue window low (may exceed hi to wrap the red seam)
    "ball_hue_hi": 180.0,          # ball hue window high
    "ball_kalman_gate2": 13.8,     # Mahalanobis^2 gate: a measurement further than this
                                   # from the Kalman prediction is a physically-irregular
                                   # jump (foot/noise) and is rejected, not smoothed in
    "ball_court_margin_x_m": 2.5,  # court-region ball filter: metres beyond the sidelines
    "ball_court_margin_y_m": 3.0,  # ... and beyond the baselines a blob may project
    "tracknet_conf": 0.5,          # TrackNet heatmap peak confidence floor (was hardcoded)
    # Size->depth honesty gate (cv/bounce.py reconstruct_world height_sigma_max): a
    # point whose HEIGHT 1-sigma for +-1px of diameter error exceeds this gets
    # height=NaN + plane-projected court position instead of the noisy 3D estimate.
    # Measured on the stored calibration: ~0.12 m/px at the near baseline vs
    # ~0.57 m/px at the far baseline (<5 px ball) — far heights are structurally
    # unreliable and must be reported as unknown, not confidently wrong.
    "ball_height_max_sigma_m": 0.35,
    # ---- Pass 2 ball filtering layer (cv/ball_filter.py) -------------------
    "ball_filter_enabled": 1.0,    # 0 = fall back to the pre-filter heuristic path
    "ball_max_speed_mph": 65.0,    # generous max pickleball speed for the velocity gate
    "ball_speed_headroom": 1.6,    # gate slack (ball height/perspective aren't modelled)
    "ball_box_persist_frames": 3,  # frames at the same box-relative spot = shoe/hat/paddle
    "ball_box_persist_px": 12.0,   # box-relative match radius for that persistence check
    "ball_foot_zone_frac": 0.25,   # bottom fraction of a player box = feet, always reject
    "ball_arc_split_deg": 55.0,    # direction change that splits the track into arcs
    "ball_arc_split_dv_px": 4.0,   # min velocity-vector change (px/frame) for a split —
                                   # angle alone shreds slow/noisy far-court shots
    "ball_arc_min_speed_px": 1.5,  # min px/frame both sides of a split must move
    # Arc-stage values are set for the TRACKNET detector (the production path
    # since 2026-07-07): its dense track lets the stricter pre-tuning values
    # hold — measured on the held-out clip they keep the same bounce recall
    # with true-FP precision 50% vs 33% against the looser values (min_pts 3,
    # tol 9, gap/join 15, claim 3.0) that the classical fallback's short
    # detection bursts needed. If you must run the classical detector as the
    # primary again, those looser values recover its coverage.
    "ball_arc_gap_frames": 10,     # time gap that always splits arcs
    "ball_arc_tol_px": 6.0,        # RANSAC inlier tolerance around the arc fit
    "ball_arc_min_pts": 4,         # arcs smaller than this are fragments (noise unless
                                   # an adjacent arc extrapolates to them); <5-point
                                   # arcs additionally need ball-like travel
    "ball_arc_join_gap_frames": 8,  # max junction gap for events / fragment continuity
    "ball_arc_claim_tol_mult": 2.5,  # rescue tolerance (x tol_px) for off-arc points
                                   # against an adjacent arc's extrapolation
    "ball_arc_max_interp_gap": 6,  # longest hole (frames) the arc fit may fill —
                                   # long fills measured 286px median error vs GT marks
    "ball_event_min_angle_deg": 25.0,  # junction turns shallower than this = same shot
    "ball_hit_adjacent_frac": 0.30,    # box expansion for 'at/near a player' = hit
    "ball_bounce_vertex_min_px": 4.0,  # min descend+ascend depth of a within-arc
                                   # y-maximum to count as a swallowed bounce
    "ball_bounce_plausibility": 1.0,  # bounce junctions must arrive descending and
                                   # leave ascending (kills ghost-turn FPs); 0 = off
    "ball_bounce_vy_frames": 3,    # frames each side of a junction to average vy over
    "ball_bounce_vy_min_px": 0.5,  # min |vy| (px/frame) for the descend/ascend test
    "ball_bounce_vy_trust_span": 7,  # arc frames needed before its vy estimate may
                                   # VETO a bounce (short stubs abstain — their
                                   # fitted vy wrongly rejected 2 real bounces)
    "ball_bounce_merge_frames": 6, # bounce events closer than this merge into one
    # ---- Rally start: serve-receive formation ------------------------------
    "require_serve_receive": 1.0,  # 1 = a rally may only start from a proper serve-receive
                                   # formation (server + diagonal receiver both behind their
                                   # baselines); 0 = looser 'one up rest back' check
    "serve_baseline_margin_m": 0.75,  # metres inside a baseline still counted as 'behind' it
    # ---- Pass 2 ball/bounce ------------------------------------------------
    "bounce_max_height_m": 0.20,
    "hit_min_height_m": 0.45,
    "player_radius_m": 1.3,
    "min_event_gap_s": 0.12,
    "net_fail_height_m": 0.95,
    "out_max_dist_m": 1.5,         # max metres past a line an out-bounce may land (else = noise)
    # ---- Rally start/end ---------------------------------------------------
    "serve_speed_mps": 4.0,
    "serve_behind_margin_m": 1.5,
    "serve_formation_lookback_s": 4.0,
    "confirm_window_s": 2.5,
    "player_active_mps": 1.2,
    "p2_reject_passes": 1.0,       # 1=drop candidates that cross the net once and are
                                   # caught/held (a between-point pass, not a real serve)
    # Learned systematic boundary corrections (frames, applied to Pass 2
    # output). These are the primary compounding knobs the report updates:
    # e.g. "rally end lands consistently 4 frames late" -> end_offset -4.
    "start_offset_frames": 0,
    "end_offset_frames": 0,
}

# Bounds keep report-driven updates from ever pushing a param somewhere
# destructive (LLM suggestions are clamped through this too).
BOUNDS: dict[str, tuple[float, float]] = {
    "p1_frame_step": (1, 6), "p1_scale": (0.2, 1.0),
    "p1_motion_ratio": (1.1, 4.0), "p1_motion_floor": (0.03, 2.0),
    "p1_ball_recent_s": (0.3, 4.0), "p1_ball_speed_px": (1.0, 12.0),
    "p1_on_s": (0.2, 3.0), "p1_off_s": (0.5, 6.0), "p1_pad_s": (0.5, 3.0),
    "p1_merge_gap_s": (0.0, 2.0), "p1_cross_window_s": (0.5, 4.0),
    "p1_min_rally_s": (1.0, 5.0), "p1_spot_check_s": (0.5, 5.0),
    "p1_end_settle_s": (0.0, 2.5),
    "p1_cross_max_gap_s": (0.3, 3.0), "p1_net_margin_m": (0.0, 1.5),
    "p1_cross_cluster_gap_s": (0.5, 4.0),
    "pose_conf": (0.05, 0.6), "pose_imgsz": (640, 1920), "p1_pose_imgsz": (640, 1920),
    "ball_gate_px": (15.0, 120.0), "ball_accel_px": (0.0, 30.0),
    "ball_acq_px": (40.0, 200.0), "ball_seed_min_px": (1.0, 20.0),
    "ball_motion_thresh": (2.0, 20.0), "ball_min_circularity": (0.15, 0.60),
    "ball_suppress_in_box": (0.0, 1.0), "ball_exclusion_zones": (0.0, 1.0),
    "ball_sat_min": (0.0, 255.0), "ball_val_min": (0.0, 255.0),
    "ball_hue_lo": (0.0, 180.0), "ball_hue_hi": (0.0, 180.0),
    "ball_foot_suppress_px": (10.0, 80.0), "ball_hit_speed_px": (6.0, 50.0),
    "ball_suppress_on_player": (0.0, 1.0), "ball_seed_on_player": (0.0, 1.0),
    "limb_lock_dist_px": (5.0, 120.0), "limb_lock_rel_vel_px": (0.5, 30.0),
    "limb_lock_frames": (2, 8),
    "ball_kalman_gate2": (5.0, 60.0),
    "ball_court_margin_x_m": (0.5, 8.0), "ball_court_margin_y_m": (0.5, 8.0),
    "tracknet_conf": (0.1, 0.9),
    "ball_height_max_sigma_m": (0.05, 2.0),
    "ball_filter_enabled": (0.0, 1.0),
    "ball_max_speed_mph": (30.0, 120.0), "ball_speed_headroom": (1.0, 3.0),
    "ball_box_persist_frames": (2, 10), "ball_box_persist_px": (4.0, 40.0),
    "ball_foot_zone_frac": (0.0, 0.5),
    "ball_arc_split_deg": (20.0, 90.0), "ball_arc_split_dv_px": (1.0, 20.0),
    "ball_arc_min_speed_px": (0.5, 10.0),
    "ball_arc_gap_frames": (3, 30), "ball_arc_tol_px": (2.0, 20.0),
    "ball_arc_min_pts": (3, 10), "ball_arc_join_gap_frames": (2, 20),
    "ball_arc_claim_tol_mult": (1.0, 5.0), "ball_arc_max_interp_gap": (1, 30),
    "ball_event_min_angle_deg": (10.0, 90.0), "ball_hit_adjacent_frac": (0.0, 1.0),
    "ball_bounce_vertex_min_px": (1.0, 30.0),
    "ball_bounce_plausibility": (0.0, 1.0), "ball_bounce_vy_frames": (1, 10),
    "ball_bounce_vy_min_px": (0.0, 5.0), "ball_bounce_vy_trust_span": (3, 20),
    "ball_bounce_merge_frames": (2, 15),
    "require_serve_receive": (0.0, 1.0), "serve_baseline_margin_m": (0.0, 2.0),
    "bounce_max_height_m": (0.05, 0.5), "hit_min_height_m": (0.2, 1.2),
    "player_radius_m": (0.5, 2.5), "min_event_gap_s": (0.05, 0.5),
    "net_fail_height_m": (0.6, 1.3), "out_max_dist_m": (0.3, 4.0),
    "serve_speed_mps": (1.5, 10.0), "serve_behind_margin_m": (0.5, 4.0),
    "serve_formation_lookback_s": (1.0, 8.0), "confirm_window_s": (1.0, 6.0),
    "player_active_mps": (0.4, 3.0), "p2_reject_passes": (0.0, 1.0),
    "start_offset_frames": (-90, 90), "end_offset_frames": (-90, 90),
}


def ensure_default_version() -> None:
    with db.connect() as con:
        row = con.execute("SELECT COUNT(*) c FROM learned_params").fetchone()
        if row["c"] == 0:
            con.execute(
                "INSERT INTO learned_params (params, origin, created_at) VALUES (?,?,?)",
                (db.dumps(DEFAULTS), "default", db.now()),
            )


def get_latest() -> tuple[int, dict[str, Any]]:
    ensure_default_version()
    with db.connect() as con:
        row = con.execute(
            "SELECT version, params FROM learned_params ORDER BY version DESC LIMIT 1"
        ).fetchone()
    params = {**DEFAULTS, **db.loads(row["params"], {})}
    return int(row["version"]), params


def get_version(version: int) -> dict[str, Any]:
    with db.connect() as con:
        row = con.execute(
            "SELECT params FROM learned_params WHERE version=?", (version,)
        ).fetchone()
    if row is None:
        return dict(DEFAULTS)
    return {**DEFAULTS, **db.loads(row["params"], {})}


def clamp(key: str, value: float) -> float:
    lo, hi = BOUNDS.get(key, (float("-inf"), float("inf")))
    return min(max(value, lo), hi)


def apply_updates(updates: dict[str, Any], origin: str) -> int:
    """Create a NEW version = latest + updates (clamped). Returns version."""
    _, current = get_latest()
    merged = dict(current)
    for k, v in updates.items():
        if k not in DEFAULTS:
            continue  # unknown keys from LLM suggestions are dropped
        if isinstance(DEFAULTS[k], int) and not isinstance(DEFAULTS[k], bool):
            merged[k] = int(round(clamp(k, float(v))))
        else:
            merged[k] = float(clamp(k, float(v)))
    with db.connect() as con:
        cur = con.execute(
            "INSERT INTO learned_params (params, origin, created_at) VALUES (?,?,?)",
            (db.dumps(merged), origin, db.now()),
        )
        return int(cur.lastrowid)

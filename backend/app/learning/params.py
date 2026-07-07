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
    "ball_foot_suppress_px": 35.0, # radius around foot keypoints to suppress blobs
    "ball_hit_speed_px": 18.0,     # px/frame a near-player blob must move to count as a hit
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
    "pose_conf": (0.05, 0.6), "pose_imgsz": (640, 1920), "p1_pose_imgsz": (640, 1920),
    "ball_gate_px": (15.0, 120.0), "ball_accel_px": (0.0, 30.0),
    "ball_acq_px": (40.0, 200.0), "ball_seed_min_px": (1.0, 20.0),
    "ball_sat_min": (0.0, 255.0), "ball_val_min": (0.0, 255.0),
    "ball_hue_lo": (0.0, 180.0), "ball_hue_hi": (0.0, 180.0),
    "ball_foot_suppress_px": (10.0, 80.0), "ball_hit_speed_px": (6.0, 50.0),
    "ball_kalman_gate2": (5.0, 60.0),
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

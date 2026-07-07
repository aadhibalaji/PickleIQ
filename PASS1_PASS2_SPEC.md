# Pass 1 & Pass 2 — Current Detection Spec

Pickleball rally detection. Two-pass CV pipeline. This document reflects the
**current** state of the code (post the detection-quality + learning-loop work).
Hand this to another agent as the source of truth for how detection works.

## Orchestration
- `backend/app/pipeline.py` runs everything. On upload / reprocess it loads the
  latest global `LearnedParams` (`learning/params.py`) **and** the effective
  court calibration (clip-local override else newest global), then runs Pass 1,
  which chains Pass 2. Every run uses the newest params, so accumulated lessons
  apply to every future clip.
- Both passes take the same `params` dict; all thresholds below are keys in it
  and are **globally learnable/tunable** (see "Learnable state").
- Calibration = a homography (image px → court metres) + estimated focal length.
  Pass 2 **requires** it; Pass 1 uses it for its primary signal, else falls back.

---

## PASS 1 — coarse gate (`backend/app/cv/pass1.py :: run_pass1`)

**Goal:** cheaply answer "is a rally roughly happening here?" over 100% of the
footage. Runs at low resolution (`p1_scale`=0.4) and reduced frame rate
(`p1_frame_step`=3). Output = candidate `(start,end)` windows → `rally_labels
source='pass1'`.

**Signals**
1. **Net crossings (PRIMARY, needs calibration).** A fast, ball-sized, size-
   consistent blob is matched frame-to-frame (`_match_fast`), projected to court
   coords, and flagged when its court-Y sign flips across the net line. The gate
   stays open while `(frame - last_cross) <= p1_cross_window_s`.
   *(This window is now its own learnable param — it used to be hard-floored at
   2.0 s, which blocked learning a shorter window.)*
2. **Motion energy vs adaptive baseline (FALLBACK, no calibration).** Mean
   frame-diff **inside the court ROI mask** vs a slow EMA baseline; fires when
   `motion > max(p1_motion_floor, baseline * p1_motion_ratio)` and a fast blob
   was seen within `p1_ball_recent_s`. Baseline adapts only during dead time.
3. **Sparse serve-formation pose spot-checks** every `p1_spot_check_s`
   (`is_serve_formation`) — logged, not part of the active/dead flip.

**Court ROI mask** (`_court_mask`): all Pass 1 signals are restricted to the
court region + margin, so neighbouring courts / spectators can't drive the gate.
*(Note: Pass 1's ball detector does NOT receive player boxes — it's a gate, not
a precise tracker.)*

**Windowing:** active↔dead flips use **hysteresis** (`p1_on_s`/`p1_off_s`
sustained sample counts). On close, windows shorter than `p1_min_rally_s` are
dropped.

**Trace:** per-sample features (`me` motion, `mt` threshold, `ball`, `cross`,
`sf`, players) are written to the `pass1` overlay — training data for a future
learned gate classifier (`backend/scripts/train_gate.py`).

---

## PASS 2 — precise boundaries + events (`backend/app/cv/pass2.py :: run_pass2`)

**Goal:** inside each padded candidate window only (~20% of frames), at full
frame rate, find precise rally start/end and the terminal event. Output =
`rally_labels source='pass2'` + a rich per-frame `pass2` overlay.

**Window building (fixed).** Merge consecutive Pass 1 windows only when the
**raw** gap ≤ `p1_merge_gap_s`, THEN pad by `p1_pad_s`, clamping the pad to at
most half the gap to each neighbour so two padded windows can never bridge two
separate rallies. Raw (unpadded) bounds are carried through for honest fallback
+ the validation harness. *(This was the key fix: padding used to happen before
merging, collapsing adjacent rallies into one.)*

**Per-frame, per window:**
- **Players:** `PoseTracker` (YOLO11-pose + ByteTrack) → court positions +
  per-frame mean player speed. Serve-formation spot-check → `serve_frames`.
- **Ball observation, in priority order:**
  1. user `ball_corrections` for that frame (confidence 1.0);
  2. **TrackNet** triplet **if `TRACKNET_ENABLED` + weights present** — *currently
     OFF by default* (pretrained tennis weights don't transfer to this
     down-the-line view without fine-tuning; ~5× slower, no gain measured);
  3. **classical `MotionBallDetector`** fallback (3-frame differencing + blob
     geometry) with TWO filters: **player-box exclusion** (full box + outward
     margins — kills shoe/paddle false positives) and **court-region filter**
     (`_in_court_region` — rejects blobs whose plane-projection is > ~2.5 m/3 m
     outside the court, i.e. neighbouring courts / spectators).
- **Kalman smoothing** (`smooth_track`, `max_gap` ≈ 0.4 s) bridges short gaps.
- **3D reconstruction** (`reconstruct_world`): apparent ball diameter → height
  above the court plane; homography → court x/y. **This size→depth step is the
  system's biggest noise source for the far-baseline ball** (few px wide).

**Events** (`bounce.py :: detect_events`): candidates = local **height minima**
near the court plane (bounce) + sharp court-depth-velocity reversals (hit).
Classification: reversal at surface level away from a player → **bounce**; at
height or at a player → **hit**; a gravity check rejects noise minima.
**Phantom-out guard:** an out-of-court bounce is only emitted if it's a **real
(observed), height-known** detection landing within `out_max_dist_m` of a line —
otherwise it's boundary noise and is discarded (this killed the false "bounce
OUT" markers). A trained **rally_end model** (model registry) overrides the
heuristic when a version is promoted to `live`.

**Rally decisions** (`rally_logic.py :: detect_rallies`) — **multi-rally**: loops
`serve → terminal end → next serve` within a window, so it emits >1 rally if
Pass 1 under-segmented. The first rally may fall back to the window edge; every
**subsequent** rally requires a confirmed serve-formation (not just a fast ball),
so dead time never becomes a phantom rally.
- **Serve start** (`_find_start`): first fast ball motion (`serve_speed_mps`)
  originating **behind a baseline** (`serve_behind_margin_m`) with a serve-
  formation sighting within `serve_formation_lookback_s`. `start_source` is
  reported honestly as `serve` / `fast_ball` / `coarse_gate`. Hook for a trained
  `serve_start` model is threaded (not yet trained).
- **Rally end** (`detect_end`): terminal event = ball becomes unplayable —
  **bounce_out** (guarded by `out_max_dist_m` + observed + height_known),
  **double_bounce**, or **net_fail**. Each candidate end is confirmed/vetoed by
  `confirmed_by_players` (if players are still moving fast within
  `confirm_window_s`, veto). No terminal event → **coarse_gate** fallback (raw
  window edge). *(On the current footage most ends fall back to coarse_gate —
  the far-ball problem; the trained rally_end model is the intended fix.)*
- **Serve-vs-pass rejection** (`p2_reject_passes`): a coarse-gate candidate that
  was **not returned** — no second net crossing AND no hit after the serve — is
  dropped as a between-point **pass**, not a serve.
- **Learned offset corrections:** `start_offset_frames` / `end_offset_frames`
  are added to the output boundaries (clipped to the window).

**Output meta** per pass2 rally: `end_event`, `confirmed`, `start_source`,
`end_source`, `raw_start`/`raw_end`/`ws`/`we`, `code_version`. Per-frame trace
(players, ball, world x/y/height, pspeed, events) → `pass2` overlay, which feeds
discrepancy reports and model training.

---

## Learnable / trainable state
- **`LearnedParams`** (`learning/params.py`) — 28 global, versioned, clamped
  scalar params (all keys referenced above). Tuned by the discrepancy-report
  loop and the auto-on-label-save learner, validated before commit.
  - Pass 1 gate: `p1_frame_step, p1_scale, p1_motion_ratio, p1_motion_floor,
    p1_ball_recent_s, p1_ball_speed_px, p1_on_s, p1_off_s, p1_pad_s,
    p1_merge_gap_s, p1_cross_window_s, p1_min_rally_s, p1_spot_check_s`
  - Pass 2 ball/bounce: `bounce_max_height_m, hit_min_height_m, player_radius_m,
    min_event_gap_s, net_fail_height_m, out_max_dist_m`
  - Rally start/end: `serve_speed_mps, serve_behind_margin_m,
    serve_formation_lookback_s, confirm_window_s, player_active_mps,
    p2_reject_passes`
  - Learned boundary bias: `start_offset_frames, end_offset_frames`
- **`rally_end` model** (BounceGRU) — model registry (`learning/model_registry.py`),
  shadow→live; used by `detect_events` when promoted. Trained by
  `backend/scripts/train_bounce.py`. Cold-start guard: needs ≥3 labeled clips to
  validate/promote.
- **`serve_start` model** — hook threaded into `detect_start`, not yet built.
- **TrackNet** — off by default; the real far-ball fix is fine-tuning it on the
  user's `ball_corrections`.

## Known limitations / open problems
- **Far-side ball** (size→depth noise) → end detection mostly falls back to
  `coarse_gate`; neither the pretrained TrackNet nor a global offset fixes it —
  the trained `rally_end` model + more labeled clips is the path.
- Classical-detector residual noise can still fabricate a "return", so the pass-
  rejection occasionally keeps a between-point ball movement (false-positive
  rally). The report's gate-tightening learning reduces these over time.
- Fault endings with **no ball signature** (foot faults, kitchen/service-motion
  faults) are out of scope in v1.

## Key files
- `backend/app/cv/pass1.py` — Pass 1 gate
- `backend/app/cv/pass2.py` — Pass 2 orchestration, window building, ball filters
- `backend/app/cv/ball_coarse.py` — classical detector + player-box exclusion
- `backend/app/cv/ball_tracknet.py` — TrackNet (gated off by default)
- `backend/app/cv/bounce.py` — world reconstruction, `detect_events`, phantom-out guard, BounceGRU
- `backend/app/cv/rally_logic.py` — `detect_rallies`, `detect_start`/`_find_start`, `detect_end`, serve-vs-pass
- `backend/app/cv/court.py` — geometry, homography, `in_court`, `side_of_net`
- `backend/app/cv/players.py` — YOLO pose, `is_serve_formation`
- `backend/app/learning/params.py` — the tunable params
- `backend/app/pipeline.py` — orchestration

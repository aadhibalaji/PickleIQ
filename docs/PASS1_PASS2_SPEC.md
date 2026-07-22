# Pass 1 & Pass 2 — Current Detection Spec

Pickleball rally detection. Two-pass CV pipeline. This document reflects the
**current** state of the code (post the detection-quality + learning-loop work).
Hand this to another agent as the source of truth for how detection works.

> **Current focus (2026-07-07): detection accuracy, not rally segmentation.**
> **Pass 1 is the SOLE source of rally start/end** — it emits generous,
> recall-favouring windows (padded + merged). **Pass 2 no longer computes rally
> boundaries**; it only produces the accurate per-frame player/ball detection
> overlay *inside* each Pass 1 window. The rally-decision code (`rally_logic`,
> terminal-event end detection, learned start/end offsets, serve-vs-pass
> rejection, the trained `rally_end`/`serve_start` model hooks) is left in the
> tree but **not called** — it can be switched back on later. The report /
> auto-learn boundary-offset loop is correspondingly **paused** (the offsets are
> applied nowhere, so reports are informational and don't auto-mutate params).

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
(`p1_frame_step`=3). Output = the **authoritative** rally `(start,end)` windows
→ `rally_labels source='pass1'`.

**Boundaries (loosened for recall).** Pass 1 is now the sole boundary source, so
`close_rally` pads each raw window by `p1_pad_s` per side and **merges** it into
the previous window when the padded windows touch (within `p1_merge_gap_s`).
Windows shorter than `p1_min_rally_s` (on their *raw* length) are still dropped
as blips. Net effect: generous windows that fully contain each rally, at the
cost of loose edges and occasional under-segmentation (two close rallies bridged
into one) — an accepted trade-off while the focus is detection, not
segmentation. (This folds the old Pass-2 pad+merge step into Pass 1.)

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

## PASS 2 — detection / visualization only (`backend/app/cv/pass2.py :: run_pass2`)

**Goal:** inside each Pass 1 rally window only (~20% of frames), at full frame
rate, produce an **accurate per-frame detection overlay** of player location and
ball position. **Pass 2 emits NO `rally_labels`** — it clears any stale
`source='pass2'` rows and writes only the rich per-frame `pass2` overlay. Rally
start/end come entirely from Pass 1.

**Window building.** Pass 2 still reconstructs the region to process by merging
consecutive Pass 1 windows (raw gap ≤ `p1_merge_gap_s`) and padding by
`p1_pad_s` — but this only defines *where detection runs*, not any boundary
output. (Pass 1 already pads/merges its emitted windows; this is just the
processing extent.)

**What was removed / disabled here:** `rally_logic.detect_rallies`,
terminal-event end detection, `start_offset_frames`/`end_offset_frames`,
serve-vs-pass rejection, and the `model_registry` live-bundle load. `detect_events`
still runs but with `model=None`, producing bounce/hit markers **purely as
overlay annotations**.

**Per-frame, per window:**
- **Players:** `PoseTracker` (YOLO11-pose + ByteTrack) → court positions +
  per-frame mean player speed. Serve-formation spot-check → `serve_frames`.
- **Ball observation, in priority order:**
  1. user `ball_corrections` for that frame (confidence 1.0);
  2. **TrackNet** triplet **if `TRACKNET_ENABLED` + weights present** — off by
     default in the code (`config.py`'s `TRACKNET_ENABLED` default), but
     **enabled on the EC2 worker since 2026-07-21** (`restart-ec2.sh` sets
     `TRACKNET_ENABLED=1`) now that fine-tuned weights for this camera setup
     exist. Pretrained tennis weights genuinely didn't transfer (~5× slower,
     no gain) — that finding stands — but the fine-tuned checkpoint is a
     different measurement: on the same real clip that showed ~0.5% ball-
     observation with the classical detector, TrackNet gave ~34% (2525/7413
     frames), real arcs forming in every window (vs. 0 previously) and
     `net_crossings_per_10s` up to ~31 where it was 0.0 everywhere before —
     4 of that clip's 13 Pass 1 windows went from rejected to kept `rally`
     verdicts on that ball evidence alone, no Stage B/C thresholds touched.
     Weights sync from S3 the same way calibration does — see
     `storage_s3.upload_tracknet_weights`/`download_tracknet_weights` and
     `cloud_ingest._sync_tracknet_weights`. Re-run
     `backend/scripts/upload_tracknet_weights.py` to ship an updated
     checkpoint; the EC2 worker picks it up on its next ~20s poll tick;
  3. **classical `MotionBallDetector`** run **PERMISSIVE** (no player-box
     exclusion) + a **`TrajectoryBallSelector`** (`ball_track.py`): the selector
     scores each raw blob by fit to the recent ball trajectory and applies the
     velocity-aware foot/shoe suppression (a blob on a foot is kept only if it
     moves ≥ `ball_hit_speed_px`). The **court-region filter** (`_in_court_region`
     — rejects plane-projections > ~2.5 m/3 m outside the court) still applies.
- **Arc-fit noise filter** (`ball_filter.refine_track`, Stage 3-4, batch per
  window): segments the window's per-frame selected picks at time gaps
  (`ball_arc_gap_frames`) and sharp turns, RANSAC-fits a quadratic to each
  segment of `≥ ball_arc_min_pts=4` points, and drops points that end up on no
  fitted arc as noise. **2026-07-20 fix:** the caller (`pass2.py`) used to
  prune every non-arc point unconditionally, including when the window fit
  ZERO arcs at all (real detections too few/scattered — at ~60fps,
  `ball_arc_gap_frames=10` is under 0.2s, so a genuinely sparse far-baseline
  track easily never clusters 4 points that close together) — that silently
  wiped every real per-frame detection in the window, turning "sparse but
  real" into "no ball ever" with no trace. Verified on a real EC2 clip: every
  one of 13 windows logged `0 arcs`, and the overlay had **0/7413** entries
  with any ball position despite 13-50 raw candidates logged per window.
  Downstream this reads to refine's Stage B as a fixed, content-independent
  low `P(live)` ceiling — the actual cause of a "same p_live_max across
  rallies of very different length" report (see
  `PASS1_PASS2_RALLY_BOUNDARIES.md` for the refine-side symptom). Fixed by
  only pruning off-arc points when `refined.arcs` is non-empty — with zero
  arcs there is no fitted trajectory to judge "off-arc" against, so the raw
  per-frame picks are left in place instead.
- **Kalman smoothing** (`smooth_track`, `max_gap` ≈ 0.4 s, `ball_kalman_gate2`
  outlier rejection) bridges short gaps and rejects physically-irregular jumps.
- **3D reconstruction** (`reconstruct_world`): apparent ball diameter → height
  above the court plane; homography → court x/y. **This size→depth step is the
  system's biggest noise source for the far-baseline ball** (few px wide).
- **Player numbering:** a serve-formation spot-check still calls
  `assign_numbers_at_serve` so players get stable 1–4 numbers in the overlay
  (visualization only — it drives no rally decision).

**Bounce/hit markers** (`bounce.py :: detect_events`, `model=None`): local
**height minima** near the court plane (bounce) + sharp court-depth-velocity
reversals (hit), with the phantom-out guard (an out-of-court bounce is emitted
only for a **real, height-known** detection within `out_max_dist_m` of a line).
These are written to the overlay **purely as visual annotations** — they no
longer drive any rally end. The trained `rally_end` model is **not** loaded.

**No rally decisions.** `rally_logic.detect_rallies`, `detect_start`/`detect_end`,
serve-vs-pass rejection, and the `start_offset_frames`/`end_offset_frames`
corrections are **not called** (the code remains in the tree, disabled). Pass 2
writes no `rally_labels`.

**Output** per window: the per-frame `pass2` overlay only (players, ball, world
x/y/height, pspeed, bounce/hit events). This still feeds discrepancy-report
feature traces and future model training; it just carries no boundary decision.

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
  - Rally start/end **(currently INERT — the rally-decision code that reads
    these isn't called):** `serve_speed_mps, serve_behind_margin_m,
    serve_formation_lookback_s, confirm_window_s, player_active_mps,
    p2_reject_passes`
  - Learned boundary bias **(currently INERT — applied nowhere while Pass 2 does
    not refine boundaries):** `start_offset_frames, end_offset_frames`
  - **Active recall levers now:** `p1_pad_s`, `p1_merge_gap_s`, `p1_min_rally_s`,
    `p1_cross_window_s` shape how loose/generous the Pass 1 windows are.
- **`rally_end` model** (BounceGRU) — model registry (`learning/model_registry.py`),
  shadow→live; used by `detect_events` when promoted. Trained by
  `backend/scripts/train_bounce.py`. Cold-start guard: needs ≥3 labeled clips to
  validate/promote.
- **`serve_start` model** — hook threaded into `detect_start`, not yet built.
- **TrackNet** — off by default in code; **on for the EC2 worker** since
  2026-07-21 now that a fine-tuned checkpoint exists (see the Pass 2 ball-
  observation section above). Fine-tuning on the user's `ball_corrections` /
  `FeatureTraces` was the anticipated path and is what actually happened.

## Known limitations / open problems
- **Far-side ball** (size→depth noise) → end detection mostly falls back to
  `coarse_gate`; the pretrained TrackNet didn't fix this, but the **fine-tuned**
  checkpoint measurably does (see above) — a global offset still doesn't.
- Classical-detector residual noise can still fabricate a "return", so the pass-
  rejection occasionally keeps a between-point ball movement (false-positive
  rally). The report's gate-tightening learning reduces these over time.
- Fault endings with **no ball signature** (foot faults, kitchen/service-motion
  faults) are out of scope in v1.

## Key files
- `backend/app/cv/pass1.py` — Pass 1 gate
- `backend/app/cv/pass2.py` — Pass 2 orchestration, window building, ball filters
- `backend/app/cv/ball_coarse.py` — classical detector + player-box exclusion
- `backend/app/cv/ball_tracknet.py` — TrackNet (off by default in code; on for
  the EC2 worker via `TRACKNET_ENABLED=1` — see `scripts/restart-ec2.sh`)
- `backend/app/cv/bounce.py` — world reconstruction, `detect_events`, phantom-out guard, BounceGRU
- `backend/app/cv/rally_logic.py` — `detect_rallies`, `detect_start`/`_find_start`, `detect_end`, serve-vs-pass
- `backend/app/cv/court.py` — geometry, homography, `in_court`, `side_of_net`
- `backend/app/cv/players.py` — YOLO pose, `is_serve_formation`
- `backend/app/learning/params.py` — the tunable params
- `backend/app/pipeline.py` — orchestration

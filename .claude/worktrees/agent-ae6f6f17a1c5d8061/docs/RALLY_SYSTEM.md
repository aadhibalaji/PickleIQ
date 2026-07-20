# Rally Segmentation System — Technical Reference

_Last updated: 2026-07-09_

This document describes the full rally-analysis pipeline for the pickleball CV system:
how Pass 1 and Pass 2 work, the player/ball detection stack, the manual annotation
tools, how training data is stored, and how that data compounds back into the models.

---

## 1. High-level architecture

The system is a **two-pass computer-vision pipeline** backed by a single SQLite
database (`data/app.db`). Nothing is written back to the source video — all output is
JSON metadata (overlays, labels) that the frontend renders on demand.

```
              ┌─────────────── pipeline.py (orchestration) ───────────────┐
              │                                                            │
   video ──►  Pass 1 (cheap, 100% of footage)  ──► rally_labels (source='pass1')
              │   coarse "is a rally happening?"        = authoritative boundaries
              │                                                            │
              └─► Pass 2 (full-rate, only inside Pass 1 windows) ──► overlays (source='pass2')
                     precise per-frame players + ball          = display / future training data
                                                                            │
                                                                └─► refine/<id>.json (advisory)
```

**Key design decision (2026-07-07 pivot):** Pass 1 is the **sole source of rally
start/end boundaries**. Pass 2 was reduced to producing per-frame detection and
visualization overlays *inside* the windows Pass 1 already decided. Much of the
historical rally-decision logic still lives in Pass 2's imports but is deliberately
**not called** — the per-frame feature traces it emits are retained as future training
data.

**Orchestration** (`backend/app/pipeline.py`):
- `start_processing(video_id)` — loads the current global learned params + effective
  calibration, submits Pass 1, then **auto-chains Pass 2** if Pass 1 was not cancelled.
- `effective_calibration(video_id)` — uses the clip's own calibration if present,
  otherwise falls back to the newest `scope='global'` calibration. This is how each new
  clip inherits accumulated calibration improvements.
- `_refine_after_pass2()` — runs the JSON-only refiner (`data/refine/<id>.json`);
  wrapped in try/except so it never fails the pass and never writes `rally_labels`.

---

## 2. Pass 1 — coarse rally gate (`backend/app/cv/pass1.py`)

Runs on **100% of footage**, cheaply: low resolution (`p1_scale`) and reduced frame
rate (`p1_frame_step`). It answers only "is a rally roughly happening?" and its output
windows are the definitive rally boundaries.

**Entry point:**
```python
def run_pass1(video_id, params, H, handle, source="pass1") -> list[tuple[int, int]]
```
Returns a list of `(start_frame, end_frame)` rally windows.

### Signals (in priority order)
1. **Net crossings (primary, needs calibration):** a ball-sized fast blob inside the
   court region whose track crosses the net line — the live-play signature.
2. **Motion energy vs. an adaptive ambient baseline** — fallback when no homography
   exists; always logged.
3. **Sparse serve-formation pose spot checks** — pose run every `p1_spot_check_s`.

### Court masking
`_court_mask()` fills a court polygon (with margins) so all Pass 1 signals ignore
neighbouring courts, spectators and walkways. When uncalibrated, fixed-camera exclusion
zones (`exclusions.in_excluded`) are used instead.

### Hysteresis gate
- **Adaptive baseline** learns ambient motion only while *not* active, so rallies never
  lift it. `thresh = max(motion_floor, baseline * motion_ratio)`.
- **Signal (calibrated):** `(frame - last_cross) <= cross_window_n` — a net crossing
  within the window keeps the gate open. `p1_cross_window_s` is a learnable param
  (validated ~2 s; mean IoU 0.52 vs. 0.05 for motion-only).
- **Signal (uncalibrated):** motion above threshold with a recent ball.
- **Streak counters** `on_n` / `off_n` flip the `active` state after enough consecutive
  agreeing frames, avoiding flicker.

### `close_rally()` — the actual segmentation writer
- Drops blips shorter than `p1_min_rally_s` (measured on raw, unpadded length).
- **Tight end:** if a crossing occurred, `end = min(end, last_cross + end_settle)`
  (avoids ending ~4 s late); otherwise falls back to the hysteresis flip.
- **Pads the START only** (`start - p1_pad_s`); the end is left tight (Pass 2 pads its
  own overlay window).
- **Merge-on-touch:** if a new rally starts within `p1_merge_gap_s` of the previous
  one, it **extends** the previous row instead of emitting a new one — favouring recall
  (never split a rally).
- Writes rows to `rally_labels` live so the UI updates while processing runs. `meta`
  JSON records `{end_source, raw_end, last_cross}`.

### Overlay output (`source='pass1'`)
Coarse per-sample features: motion energy, threshold, optional ball `[x,y,diam]`,
crossing flag, and on spot-check frames the serve-formation flag + player entries.

---

## 3. Pass 2 — precise detection inside windows (`backend/app/cv/pass2.py`)

Runs at **full frame rate** but only inside Pass 1 windows (~20% of the clip).
**Computes no rally boundaries.** It produces accurate per-frame player + ball overlays
for display and future training.

**Entry point:**
```python
def run_pass2(video_id, params, H, cam, handle, source="pass2", p1_source="pass1") -> list[dict]
```
Raises `RuntimeError` if `H` or `cam` is `None` (Pass 2 requires calibration). Its
product is the overlays, not a return value.

### Window construction from Pass 1
1. Read Pass 1 rally labels.
2. **Merge** adjacent windows when the raw gap ≤ `p1_merge_gap_s`.
3. **Pad** each merged window by `p1_pad_s`, clamped to at most half the gap to each
   neighbour so processing windows never overlap.
4. Delete any stale `pass2` rally-label rows (enforcing Pass 1 as single source of
   truth).

### Per-frame processing
- **Pose:** one `PoseTracker(track_sequence=True, imgsz=1920)` for the whole run;
  `new_sequence()` resets ByteTrack per window **without** recompiling the OpenVINO
  model (~5 s saved per window).
- **Player speed** computed on a stable identity key (`slot` or `track_id`) so an
  id-switch is not read as a teleport; stored per frame in m/s.
- **Serve gate:** roughly every 0.5 s, checks serve/receive formation and calls
  `assign_numbers_at_serve()` to anchor official player numbers.
- **Ball** (only when enabled — see §5): per-frame source priority
  **manual correction > external track > TrackNet > classical selector**.

### Window-level ball reasoning (visualization only)
- `refine_track()` — per-shot arc RANSAC that drops off-arc outliers and interpolates
  short holes.
- `classify_arc_events()` — hit/bounce events from arc junctions + player boxes.
- `smooth_track()` — Kalman smoothing.
- `reconstruct_world()` — attaches `[court_x, court_y, height]` per entry via the camera
  model.
- Events written to each entry as `evt = {kind, in, side, cx, cy}`.

### Overlay output (`source='pass2'`)
Dense per-frame `players[...]` (id, number, box, court xy, team, keypoints), player
speed, optional `ball [x,y,diam,conf]`, world reconstruction, and bounce/hit events.

### Dormant code
`rally_logic.detect_rallies`, learned start/end offsets, serve-vs-pass rejection, and
trained `rally_end` / `serve_start` hooks are imported but **intentionally not called**.
Feature traces are still embedded for future training.

---

## 4. Player detection & identity

### Model stack (`backend/app/cv/players.py`)
- **Ultralytics YOLO11-pose** (`yolo11n-pose.pt`, COCO 17-keypoint) + built-in
  **ByteTrack** for persistent per-frame track IDs.
- **GPU path:** the model is exported to **OpenVINO FP16** at a fixed `imgsz` and run on
  the **Intel Arc GPU** (`intel:gpu`). The torch-XPU path is avoided (it throws inside
  ultralytics `.track()`), so OpenVINO is the production GPU route. Export runs at most
  once per image size, cached under `data/models/pose/<imgsz>/`.
- A **fresh YOLO model per `PoseTracker`** — ByteTrack state lives on the predictor, so
  a shared model would corrupt track state across windows/jobs.
- **Synthetic IDs:** detections whose ByteTrack id is `None` (tentative/lost tracks) are
  kept with negative synthetic ids — this fixed the "far players never detected" and
  "detection stopped at frame ~2900" bugs.
- Pass 1 uses `track_sequence=False` at full-res (far-side players vanish at low res);
  Pass 2 uses `track_sequence=True, imgsz=1920`.

`PlayerObs` output: `track_id`, `box`, `foot_xy`, `court_xy` (metres via homography),
`keypoints`, `team` (`near`/`far`), `number`, `slot`.

### Match-stable identity / re-ID (`backend/app/cv/identity.py`)
ByteTrack ids are appearance-blind and break on crossings/occlusions, so player numbers
used to switch. This layer keeps **persistent identity slots** (≤2 per team) that
survive id-switches and window resets.

- **Tools:** classical **HSV torso colour histogram** (16×16 Hue×Sat, Value dropped for
  exposure invariance; runs on CPU) + court-space distance. No deep ReID model.
- **Cost:** `court_dist_m / POS_SCALE + APP_W · bhattacharyya(hist, hist)`.
- Optimal injective matching by brute-forcing permutations (N ≤ 2 per team).
- `assign_numbers_at_serve()` anchors numbers left→right per side at serve frames and
  binds each number to its persistent slot so it rides with the body.

### Court geometry (`backend/app/cv/court.py`)
`COURT_W=6.096`, `COURT_L=13.4112`, `NET_Y=6.7056`, `KITCHEN=2.1336` (metres);
`side_of_net(y)` returns `far` if `y < NET_Y` else `near`.

### Exclusion zones (`backend/app/cv/exclusions.py`)
Two normalized right-triangles in the top corners reject fixed-camera artifacts;
resolution-independent, also drawn on overlays.

### Device selection (`backend/app/cv/device.py`)
`torch_device()` probes **Intel XPU → CUDA → CPU** (overridable via `CV_DEVICE`). Heavy
models reach the Arc GPU via OpenVINO rather than torch-XPU.

---

## 5. Ball detection

Four cooperating modules, with a strict source priority in Pass 2:
**manual correction > external dense track > TrackNet > classical fallback.**

### 5.1 TrackNetV2 (`backend/app/cv/ball_tracknet.py`) — primary when weights present
- VGG-style U-Net mirroring NTT **WASB-SBDT** (3 RGB frames → 3 heatmaps), so official
  WASB checkpoints load directly. Input 512×288.
- **Disabled by code default** (`TRACKNET_ENABLED=0`) **but enabled in this deployment
  via `.env` (`TRACKNET_ENABLED=1`)**; weights present at `data/models/tracknet_v2.pt`
  (WASB tennis checkpoint), with a pre-compiled OpenVINO FP16 IR at
  `data/models/tracknet_ov/`. This is what produces the accurate per-frame ball track
  you see rendered in Pass 2.
- **GPU path:** torch → OpenVINO FP16 IR, compiled for the Arc GPU; CPU fallback.
- Returns `(x, y, diameter_px, conf)` in original pixels or `None` below threshold.

### 5.2 Classical `MotionBallDetector` (`backend/app/cv/ball_coarse.py`) — fallback + Pass 1 gate
- **Three-frame differencing** keeps only objects moving between both frame pairs;
  contours filtered by area + circularity (≥0.35). Returns top-5 candidates by score.
- Optional HSV appearance gate (disabled by default).
- Player-box suppression expands boxes (largest margin at the bottom, for shoes). In
  Pass 2 the fallback runs **permissive** (no box exclusion) so hit/flyover balls reach
  the selector.

### 5.3 `TrajectoryBallSelector` (`backend/app/cv/ball_track.py`) — per-frame motion selection
Motion, not appearance, is the primary defense against shoe/foot false positives. Picks
the candidate that best **continues the recent trajectory** (linear velocity prediction
+ adaptive gate). A candidate on a player/foot is only accepted if moving fast enough
(hit speed). Cold-starts a track only from a two-frame mutual-nearest ball-speed match,
so a stationary shoe never seeds. `seed()` injects external/TrackNet/manual positions
for cross-source continuity.

### 5.4 `BallFrameFilter` + arc refinement (`backend/app/cv/ball_filter.py`) — semantic layer
Detector-agnostic; uses physics + player tracks as context.
- **Stage 1 — velocity gate:** rejects candidates implying an impossible pixel jump;
  max speed converted px→m via the homography Jacobian (tightens in the far court).
- **Stage 2 — player-box motion logic:** rejects candidates in the foot zone of a box or
  those holding the same box-relative position for several frames.
- **Stage 3 — `refine_track()`:** segments the track into per-shot arcs at time gaps and
  sharp direction changes, then RANSAC quadratic (projectile) fit per arc; off-arc
  points get one rescue chance then are dropped; short holes are interpolated from the
  fit.
- **Stage 4 — `classify_arc_events()`:** direction change near a player box → **hit**,
  else **bounce** (with a plausibility gate on vertical velocity); within-arc y-vertex
  bounces detected in open court.

### Pass 2 ball toggle
Ball detection in Pass 2 is display-only and controlled by `PASS2_BALL`. The **code
default is off** (`"0"`), but this deployment's `.env` sets **`PASS2_BALL=1`**, so ball
tracking is live. Rally boundaries always come solely from Pass 1 regardless of this
toggle.

---

## 6. Manual annotation tools

There are **two independent manual-annotation surfaces**, feeding two independent
improvement loops.

### 6.1 Manual rally segmentation (`frontend/src/LabelingPanel.jsx`)
A strict alternating state machine (Start → End → Start …) so malformed pairs are
impossible.

**Labeling definition:**
- **START** = the serve (players set, first ball motion).
- **END** = the exact frame the ball becomes **unplayable** (bounces out, second bounce,
  or dies at the net) — *not* when the ball stops rolling, *not* when players reset.

Controls are buttons ("▶ Start rally @ f…", "■ End rally @ f…"), with per-row
seek/re-set/delete and a "Save N labels" action.

**Backend** (`backend/app/api/annotate.py`):
- `PUT /videos/{video_id}/labels/human` — replaces all `source='human'` rows for the
  clip and **submits a background `learn` job** (`autolearn.run_learn`) that closes the
  learning loop.
- `GET /videos/{video_id}/rallies` — returns `{human, pass1, pass2}` grouped by source.

### 6.2 Manual ball ground-truth marking (`frontend/src/Workspace.jsx`)
Toggled by the **"🎯 Mark ground truth"** button. Click the ball on the canvas to pause
and capture the frame + pixel; a key then classifies the mark. The `ball_corrections`
table doubles as the ball-event GT store.

**Keyboard shortcuts:**

| Key | Context | Action |
|-----|---------|--------|
| `←` / `→` | in gtMode | step ∓1 frame (pauses) |
| `Shift+←` / `Shift+→` | in gtMode | step ±10 frames |
| `B` | after click | classify as **bounce** → in/out step |
| `H` | after click | save as **hit** |
| `N` | after click | save as **position** only |
| `Enter` | in/out step | save bounce with current in/out choice |
| `X` | in/out step / edit | flip in ↔ out |
| `Esc` | any | cancel / reset |
| click a diamond | idle | select existing mark (edit) |
| `B`/`H`/`N` | edit | retype selected mark |
| `D` / `Delete` | edit | delete selected mark |
| click (mark selected) | edit | move mark to clicked point + current frame |

**Server-side projection is the single source of truth:** court coordinates and the
in/out suggestion are computed at save time via the clip's homography. A `disagree` flag
is set when the human's in/out call differs from the homography suggestion (flags likely
calibration error near lines).

**Backend endpoints:**
- `POST /videos/{video_id}/ball_corrections` — event types `bounce | hit | position`; a
  bounce requires `in_out ∈ {in, out}`.
- `PUT` / `DELETE /ball_corrections/{id}`.
- `GET /videos/{video_id}/project_point?x=&y=` — live in/out suggestion before saving.
- `POST /videos/{video_id}/ball_tracks` — distinct **external dense ball track** store
  used by Pass 2 as a preferred detector input.

---

## 7. How training data is stored

### SQLite tables (`backend/app/db.py`, `data/app.db`)
- **`rally_labels`** — `(id, video_id, source, start_frame, end_frame, meta)`. Human
  labels are `source='human'`; Pass 1/Pass 2 rows use their own source. The temporal
  rally ground truth.
- **`ball_corrections`** — ball-event GT: `frame, x, y, event_type, court_x, court_y,
  in_out, in_out_suggested, disagree, scope`. GT columns are added idempotently at
  startup; legacy NULL `event_type` = `position`.
- **`ball_tracks`** — one external dense track per clip; payload `{frame: [x, y,
  diameter|null, confidence]}` in full-res px.
- **`learned_params`** — versioned global tunables `(version, params JSON, origin,
  created_at)`.
- **`param_eval`** — scoreboard snapshots (pooled start/end error, match rate, counts,
  applied flag).
- **`pending_changes`** — medium-risk param proposals queued for human approval.
- **`model_registry`** — versioned weight models `(version, kind, path, status, metrics,
  parent_version, trained_on, …)`; `kind ∈ {rally_end, serve_start, tracknet}`; `status
  ∈ {shadow, live, rejected, archived}`. Weights at `data/models/<kind>/v<version>.pt`.
- **`overlays`** — per-frame players/ball/events JSON, chunked (900 frames/chunk).
- Plus `calibrations`, `discrepancies`, `reports`, `user_notes`.

### File-based artifacts
- **Serve dataset** (`data/refine/labels/`): `<id>.serves.json` (every rally start is a
  serve-contact timestamp), `<id>.rallies.json`, `serve_gate/<id>.json` (span-level
  serve / non-serve). Built by `backend/app/refine/serve_labels.py` (CLI:
  `backend/scripts/build_serve_dataset.py`).
- **XGBoost live/dead model**: `data/models/refine/xgb.json` + `.meta.json` sidecar
  recording feature-name order.
- **Ball GT git export**: `data/app.db` is gitignored, so
  `python -m backend.scripts.ball_gt_io export` dumps `ball_corrections` to the tracked
  `reference_data/ball_gt.json`; `import` upserts by id.

---

## 8. How training data improves the models

Three distinct learning mechanisms, all fed by the same human labels.

### A. Analytic boundary-offset auto-learner (`backend/app/learning/`)
Triggered synchronously on every human-label save (no CV runs — it re-scores **stored**
Pass 2 raw boundaries, so it's millisecond-fast).

1. Debounce via a hash of the human spans.
2. `compare(human, ai, fps)` (`analysis/metrics.py`) does greedy IoU matching and
   classifies each discrepancy as `boundary_drift` / `missed` / `extra`, with signed
   start/end errors and `clean` flags (boundary came from a real detected serve /
   confirmed terminal event, not the coarse-gate fallback).
3. `updates.compute_param_updates` emits risk-tagged proposals:
   - **LOW risk** — `start_offset_frames` / `end_offset_frames` from the median of clean,
     capped pairs (damped, ≤5 frames/report).
   - **MEDIUM risk** — Pass 1 gate nudges (`p1_motion_ratio`, `p1_off_s`,
     `p1_min_rally_s`) from signed missed−extra pressure.
4. **Guardrail** (`learning/validate.py`): a low-risk offset is committed globally only
   if it improves pooled mean-abs error across **all** labeled clips without regressing
   match rate. Accepted → new `learned_params` version. Rejected → recorded, not applied.
5. Medium-risk proposals are **never auto-applied** — queued in `pending_changes` for
   human approval (which runs the expensive full-rerun harness).
6. Every run records a `param_eval` snapshot.

`learned_params` are **global and compounding**: loaded before every future clip's Pass 1,
so each accepted lesson applies to all subsequent clips (values clamped to `BOUNDS`).

### B. XGBoost live/dead rally classifier (`backend/app/refine/`)
A per-timestep P(live) classifier used by the Pass 2 refiner.
- **Features** (`refine/features.py`): ~24 nan-aware columns at ~15 Hz — cross recency,
  crossings/3 s, ball speed, fraction visible, net side, bounce event, ball-player
  distance, formation validity, reactivity, plus rolling variants.
- **Labels** (`refine/labels.py`): human `rally_labels` expanded to a per-timestep 0/1
  vector aligned to the feature timeline.
- **Training** (`refine/train.py`): `objective=binary:logistic`, `max_depth 4,
  n_estimators 200, lr 0.08`. Split is **always by clip** (never by timestep — adjacent
  timesteps leak); `holdout_clips` frozen out. Falls back to pure-numpy L2 logistic
  regression when xgboost isn't importable. A `RulesClassifier` weighted-sigmoid baseline
  is the permanent fallback.
- **Post-processing** (`refine/postprocess.py`): hysteresis → live-ratio window → merge
  short gaps → drop short segments. **Serve gate** (`refine/serve_gate.py`) scores
  formation/contact/flight/landing phases and snaps the end to the last meaningful ball
  event.
- **Eval** (`refine/evaluate.py`): segment precision/recall/F1 at IoU > 0.5, boundary
  error (median + p90 s), rejection accuracy; `sweep()` grid-searches post-processing
  knobs on non-holdout clips. Last sweep (2026-07-08, 5 labeled clips): F1 0.627,
  precision 0.778, rejection acc 1.0.

### C. Torch rally-end / bounce model (`backend/scripts/train_bounce.py`, `learning/dataset.py`)
A learned temporal model (`kind='rally_end'`) trained by weak supervision:
- Each human rally END frame is a terminal-bounce positive; mid-rally heuristic hits are
  pseudo-labelled. Uses the **same featurizer as inference** (no train/serve skew);
  augmentation = temporal jitter + feature noise.
- Training registers a new **`shadow`** model version. Promotion to `live` requires
  `--promote` **and** ≥3 labeled clips (cold-start overfit guard), gated by
  `validate.evaluate_model` (must not lose matches and must lower mean-abs end error).
  Live models compound — Pass 2 loads the latest live version per kind.

### The loop, end to end
```
Human marks rallies (LabelingPanel) ──PUT /labels/human──► rally_labels(source=human)
   │  (synchronous background job)
   ├─► serve dataset files (serve_labels.py)
   ├─► refine XGBoost live/dead (train.py, eval by clip)
   ├─► torch rally-end model (train_bounce.py → model_registry, shadow→live)
   └─► autolearn.run_learn:
         compare(human, pass2) → discrepancies → compute_param_updates
           ├ LOW  → validate across all clips → apply → new learned_params (global, compounds)
           └ MED  → pending_changes (human approval → full-rerun harness)
         record param_eval snapshot

Human marks ball events (Workspace 🎯) ──► ball_corrections (GT: court_xy, in/out vs suggested)
   └─► ground truth for ball-tracking / bounce in-out evaluation; git-shared via reference_data/ball_gt.json
```

The two annotation systems are deliberately decoupled. **Global vs. clip-local scoping**
is a recurring routing rule: method-level lessons update the shared, compounding
`learned_params` baseline, while clip-local oddities (a bumped camera, a one-frame ball
fix) stay on the clip row and never move the global baseline.

---

## 9. Key file map

| Area | Files |
|------|-------|
| Orchestration | `backend/app/pipeline.py` |
| Pass 1 / Pass 2 | `backend/app/cv/pass1.py`, `backend/app/cv/pass2.py` |
| Players / identity | `backend/app/cv/players.py`, `identity.py`, `court.py`, `exclusions.py`, `device.py` |
| Ball detection | `ball_tracknet.py`, `ball_coarse.py`, `ball_track.py`, `ball_filter.py` |
| Overlays storage | `backend/app/cv/overlays.py` |
| Manual rally UI | `frontend/src/LabelingPanel.jsx` |
| Manual ball GT UI | `frontend/src/Workspace.jsx` |
| Labeling API | `backend/app/api/annotate.py` |
| Refiner (XGBoost) | `backend/app/refine/{features,labels,classifiers,train,postprocess,serve_gate,evaluate,config}.py` |
| Serve dataset | `backend/app/refine/serve_labels.py`, `backend/scripts/build_serve_dataset.py` |
| Auto-learner | `backend/app/learning/{autolearn,updates,validate,params,dataset,model_registry}.py` |
| Metrics / comparison | `backend/app/analysis/metrics.py` |
| Model training CLI | `backend/scripts/train_bounce.py` |
| Ball GT git export | `backend/scripts/ball_gt_io.py` |
| Schema | `backend/app/db.py` |

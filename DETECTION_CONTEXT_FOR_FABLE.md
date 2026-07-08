# Player & Ball Detection — Architecture Context for Review / Rebuild

**Audience:** an evaluating agent (Fable) asked to *audit the architecture at every
level* and, where warranted, *rebuild* the player-detection and ball-detection
subsystems so the computer vision is sound and works well.

**What this document is:** a self-contained, code-accurate briefing on how player
and ball detection currently work in `CV-Training-Tool`, the constraints they run
under, the known failure modes, and the exact questions we want you to answer.
Read this first; then read the cited source files. All line numbers are current as
of this writing.

> **Scope discipline (important).** This is a *pickleball rally-analysis* tool with a
> **fixed, single, down-the-line camera**. Every design choice below is downstream of
> that geometry and of an **Intel Arc iGPU + OpenVINO** deployment target. Do not
> propose a rebuild that ignores these two constraints (e.g. "just use a
> multi-camera calibrated rig" or "run a big transformer on CUDA") — evaluate and
> improve *within* them, and call out explicitly if you believe a constraint should
> change and why.

---

## 0. The task we want you to perform

1. **Audit the architecture at every level** — from frame I/O → detection →
   tracking → identity → geometry/3D → event detection → persistence. For each
   layer state: is the responsibility correct, is the interface clean, is the
   algorithm appropriate for this camera geometry, and where is it fragile?
2. **Judge whether the CV will "work well"** for the stated goal: accurate
   per-frame *player location* and *ball position* inside rally windows, on this
   fixed axial camera, with far-baseline objects only a few pixels wide.
3. **Rebuild where the architecture is unsound**, preserving the hard constraints
   (§6) and the invariants (§7). Prefer surgical, well-justified changes over a
   from-scratch rewrite; if you do recommend a rewrite of a subsystem, justify it
   against the constraints and give a migration path.
4. **Deliverable:** a written architecture assessment (per-layer verdicts +
   prioritized issues), followed by concrete code changes / a rebuild proposal.

Success criteria and evaluation questions are collected in §8.

---

## 1. System overview (just enough to place the detectors)

- **Domain:** pickleball rally detection + training feedback. Upload a
  fixed-camera video → run a two-pass CV pipeline → human labels ground truth →
  compare/report → global parameters + models compound over time.
- **Backend:** Python 3.12, FastAPI + uvicorn on port **8420**
  (`backend/app/main.py`). Single-threaded background `JobRunner`
  (`backend/app/jobs.py`).
- **Frontend:** React 18 + Vite (`frontend/`), draws overlays *on top of* the
  original video — the video is never modified.
- **Storage:** SQLite (`data/app.db`, WAL). Detections are stored as chunked JSON
  "overlays", not burned into pixels.
- **Two passes** (`backend/app/pipeline.py` chains them):

| | **Pass 1** (`cv/pass1.py`) | **Pass 2** (`cv/pass2.py`) |
|---|---|---|
| Question | "Is a rally roughly happening?" | "Where exactly are the players and ball?" |
| Coverage | 100% of frames, subsampled | only inside Pass 1 windows (~20%), full FPS |
| Resolution | `p1_scale=0.4`, every 3rd frame | full res, every frame |
| Output | authoritative rally `(start,end)` → `rally_labels source='pass1'` + `pass1` overlay | per-frame player/ball overlay → `pass2` overlay only |
| Calibration | optional (net-cross primary, motion fallback) | **required** (raises without homography) |

> **Current state you must respect (see `PASS1_PASS2_SPEC.md`):** Pass 1 is the
> **sole** source of rally boundaries. **Pass 2 emits no rally boundaries** — it
> only produces the detection overlay. The rally-decision code (`cv/rally_logic.py`,
> terminal-event end detection, serve-vs-pass rejection, learned start/end offsets,
> the `rally_end`/`serve_start` model hooks) is **left in the tree but not called**.
> Several `LearnedParams` are therefore currently **inert** (flagged in §5). Your
> audit should focus on the **live detection path**, but note any coupling to the
> dormant code.

---

## 2. Player detection — deep dive

**Primary file:** `backend/app/cv/players.py`
**Supporting:** `backend/app/cv/identity.py`, `backend/app/cv/exclusions.py`,
`backend/app/cv/court.py`, `backend/app/cv/device.py`

### 2.1 Model & tracker
- **Detector/pose:** Ultralytics **YOLO11-pose**, default weights
  `yolo11n-pose.pt` (nano; `config.py:33`). `classes=[0]` (person only).
- **Tracker:** **ByteTrack** (`tracker="bytetrack.yaml"`), persistent IDs in Pass 2.
- **Accel:** exports an **OpenVINO FP16** model per `imgsz` to
  `data/models/pose/{imgsz}/` and runs it on the **Intel Arc iGPU** (~6× faster
  than the torch/CPU path; Ultralytics' torch path can't drive `device='xpu'` for
  `.track()`). Falls back to torch (XPU→CUDA→CPU) if OpenVINO/GPU is unavailable.

```103:129:backend/app/cv/players.py
    def __init__(self, H: np.ndarray | None, track_sequence: bool = True,
                 conf: float = 0.25, imgsz: int = 640,
                 model_name: str | None = None):
        ...
        ov_dir = _openvino_pose_dir(self._imgsz) if model_name is None else None
        if ov_dir:
            from ultralytics import YOLO
            self._model = YOLO(ov_dir)
            self._device = "intel:gpu"
        else:
            self._model = _build_model(model_name or POSE_MODEL)
            self._device = ultralytics_device()
```

### 2.2 Critical design decisions (do NOT silently discard these)
- **Per-tracker model instance.** Each `PoseTracker` builds its **own** YOLO
  object. ByteTrack state lives on the model's predictor; two trackers sharing one
  model corrupt each other's track state — the documented bug that once wedged
  tracking mid-clip and made detection stop returning players
  (`players.py:27-37`). `new_sequence()` (`players.py:204-218`) re-inits ByteTrack
  per Pass 2 window via `persist=False` **without** recompiling the OpenVINO model
  (~5 s each on GPU).
- **`id=None` detections are kept, not dropped.** ByteTrack returns `id=None` for
  tentative/unconfirmed tracks. The old code returned `[]` here, discarding visible
  players — root cause of "far-side players never detected" and "detection stopped
  at ~frame 2900". They are now kept with synthetic negative ids
  (`players.py:153-164`).
- **Foot point** = mean of ankle keypoints (COCO 15–16) when confident, else
  box bottom-center (`players.py:175-181`). Projected to court metres via the
  homography; team = `side_of_net`.
- **Exclusion zones** (`exclusions.py`): hardcoded, normalized right-triangles in
  the top corners to drop background/neighbouring-court people. Box-center test.
  **These are hand-tuned to one camera framing (2026-07-07)** and must be re-tuned
  if framing changes — a clear fragility.

```137:147:backend/app/cv/players.py
        kwargs = dict(persist=persist, tracker="bytetrack.yaml", classes=[0],
                      conf=self._conf, imgsz=self._imgsz, verbose=False)
        try:
            res = model.track(frame, device=self._device if self._device_ok else "cpu", **kwargs)
        except Exception:
            if self._device_ok and self._device not in ("cpu",):
                log.warning("tracker failed on %s, falling back to CPU", self._device)
                self._device_ok = False
                res = model.track(frame, device="cpu", **kwargs)
```

### 2.3 Match-stable identity (`identity.py`)
ByteTrack ids churn on crossings/occlusions, which made displayed player **numbers**
jump. `PlayerIdentityTracker` keeps ≤2 persistent slots per team, assigning each
detection by a cost = court-distance + HSV-torso-histogram Bhattacharyya distance
(`identity.py:146-154`). Slots persist across windows; `new_window()` forgets only
positions (players walk between rallies) so re-acquisition is appearance-first.
Official 1–4 numbers are re-anchored left→right at each serve
(`assign_numbers_at_serve`).

### 2.4 Serve-formation gates (used for numbering + the dormant rally-start logic)
- `is_serve_formation` (`players.py:227-249`): "one up, rest back" per side.
- `is_serve_receive_formation` (`players.py:252-276`): stricter — server behind one
  baseline + diagonal receiver behind the opposite baseline. Gated by
  `require_serve_receive` (default on). Depends on the far receiver being detected,
  which motivates the high Pass 2 `pose_imgsz`.

### 2.5 Tuning knobs (in `LearnedParams`)
`pose_conf=0.25`, `pose_imgsz=1920` (Pass 2), `p1_pose_imgsz=1280` (Pass 1 spot
checks). See the measured recall/latency note at `params.py:34-42`:
far-player recall 1280→35%, 1536→87%, **1920→100%** at 43/55/74 ms/frame on the
Arc GPU. **The 2nd far player in doubles is often physically occluded** (rear hides
behind front in this axial view) — nano@2560 and yolo11s@1920 both still find 0, so
it is called a **camera-geometry limit, not a resolution/model-capacity limit.**

### 2.6 Known weaknesses to evaluate
- Far-side occlusion of the 2nd doubles player (geometry). Is there any signal
  (temporal, appearance, motion) that could recover it, or is the "accept it" call
  correct?
- Hardcoded exclusion zones — is a geometry-derived mask (from the homography /
  court ROI, as Pass 1 already builds via `_court_mask`) a better architecture than
  hand-tuned image-space triangles?
- Identity re-ID is a coarse HS histogram. Sufficient for ≤4 known players on a
  static camera, or a fragility on same-colored kits / lighting shifts?
- Synthetic-negative-id detections have no temporal association — do they pollute
  speed continuity or identity assignment?

---

## 3. Ball detection — deep dive (the current accuracy ceiling)

**Files:** `cv/ball_coarse.py` (classical detector), `cv/ball_track.py`
(trajectory selector), `cv/ball_tracknet.py` (TrackNetV2), `cv/ball_filter.py`
(optional semantic filter), `cv/kalman.py` (smoothing), `cv/bounce.py` (3D +
events). Orchestrated in `cv/pass2.py`.

### 3.1 Ball-source priority chain (Pass 2, per frame)
`pass2.py:236-289`. Priority:
1. **Manual `ball_corrections`** for that frame (confidence 1.0).
2. **External dense track** uploaded via `POST /api/videos/{id}/ball_track`
   (see `BALL_TRACKER_INTEGRATION.md`) — *recommended integration path*.
3. **TrackNetV2** if `TRACKNET_ENABLED=1` + weights present — **OFF by default**.
4. **Classical fallback:** `MotionBallDetector` (permissive) → `TrajectoryBallSelector`.

```239:268:backend/app/cv/pass2.py
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
                    ...
                    if len(frame_buf) == 3:
                        r = tracknet.infer_triplet(frame_buf)
                        if r:
                            det = r
                else:
                    cands = [cd for cd in fallback.update(frame)
                             if _in_court_region(H, cd.x, cd.y)]
                    ...
                    c = selector.select(f, cands, boxes, feet)
                    if c:
                        det = (c.x, c.y, c.diameter_px, c.score)
```

### 3.2 Classical detector (`ball_coarse.py`)
Three-frame differencing → binary motion mask → contour → area bounds +
`min_circularity` gate → optional HSV appearance gate (off by default, ball color
varies per clip). Middle-frame localization is used for the color sample because a
fast ball has already moved in the current frame. Player-box suppression
(`on_player_box`, shared margins, largest at the bottom where shoes/shadows sit).
Returns top-5 by circularity. In Pass 2 the detector is run **permissive** (no
box exclusion) so a ball passing over a player still reaches the selector; the
`_in_court_region` filter (`pass2.py:44-52`) rejects neighbouring-court motion.

Honest self-assessment in the file header: *"a real detector … clearly weaker than
a trained TrackNet on far-baseline balls."*

### 3.3 Trajectory selector (`ball_track.py`) — motion, not appearance, is the defense
Key idea: in a single frame a white shoe and a pickleball look alike; over a few
frames they don't. `TrajectoryBallSelector`:
- **Velocity-gated prediction** (`_predict`): with ≥2 history points, predict
  forward and gate tightly (`gate_px + accel_px·gap²`); with 1 point, gate a large
  acquisition radius (`acq_px·gap`).
- **Cold start** (`_cold_start`): seed a track only from a **mutually-nearest,
  ball-speed** displacement across two consecutive frames — a stationary shoe
  (roundest blob) can never seed.
- **Foot/box suppression:** a blob on a player box/foot keypoint is kept only if it
  moves ≥ `hit_speed_px` (a hit/fly-over); otherwise dropped. Toggle
  `ball_suppress_on_player` loosens this when the weak detector's blobs cluster near
  players.

```117:142:backend/app/cv/ball_track.py
    def select(self, frame: int, cands, boxes, feet):
        pred, gate = self._predict(frame)
        if pred is None:
            return self._cold_start(frame, cands, boxes, feet)
        best, best_s = None, -1e18
        for c in cands:
            on_player = (self.suppress_on_player
                         and _on_player(c.x, c.y, boxes, feet, self.foot))
            if on_player and not self._fast_from_last(frame, c.x, c.y):
                continue
            dist = math.hypot(c.x - pred[0], c.y - pred[1])
            if dist > gate:
                continue
            s = (gate - dist)
            if s > best_s:
                best, best_s = c, s
        ...
```

### 3.4 TrackNetV2 (`ball_tracknet.py`) — the "real" model, currently off
- Architecture byte-for-byte mirrors NTT WASB-SBDT TrackNetV2 (VGG U-Net, 3 RGB
  frames → 3 heatmaps, 512×288 input) so their pretrained tennis/badminton
  checkpoints load directly. Download via `backend/scripts/get_tracknet_weights.py`.
- OpenVINO FP16 GPU path preferred; torch fallback. `conf_threshold=0.5`
  (hardcoded, **not** a LearnedParam — flag this). Diameter estimated from the
  half-max heatmap blob extent.
- **Off by default** (`TRACKNET_ENABLED=0`, `config.py:35-41`): measured that the
  pretrained tennis/badminton weights **don't transfer** to this down-the-line
  pickleball view without fine-tuning, and it's ~5× slower with no measured gain.
  The stated real fix is **fine-tuning on the user's `ball_corrections`.**

### 3.5 Post-detection pipeline (window-level)
1. Optional `BallFrameFilter` + arc-RANSAC (`ball_filter.py`) — **off by default**
   (`ball_filter_enabled=0`); its arc params were tuned for TrackNet's dense track,
   not the classical fallback's short bursts (`params.py:80-96`).
2. **Kalman `smooth_track`** (`kalman.py`): bridge gaps ≤ ~0.4 s, reject
   physically-irregular jumps via Mahalanobis gate `ball_kalman_gate2=13.8`.
3. **3D reconstruction** (`bounce.py::reconstruct_world` + `court.py::
   CameraModel.ball_world_from_size`): apparent diameter → camera-depth
   `Z = f·D_real/d_px` (pickleball D = 74 mm) → world (court_x, court_y, height).
   **This size→depth step is explicitly called the system's biggest noise source
   for the far-baseline ball (few px wide).**
4. **Bounce/hit markers** (`bounce.py::detect_events`, currently `model=None`):
   height-minima (bounce) + court-depth-velocity reversals (hit), with a
   phantom-out guard. **Visual annotations only right now** (drive no rally
   decision). A learned `BounceGRU` (`bounce.py:273-304`) exists and is preferred
   once trained.

### 3.6 Known weaknesses to evaluate
- **Far-baseline ball:** few pixels wide → noisy diameter → noisy depth/height →
  weak bounce detection. This is the stated ceiling. Is the size→depth model the
  right approach for this camera, or should height come from a different cue (e.g.
  learned temporal model, court-line/shadow geometry, or accepting 2D-only for far
  balls)?
- **Detector quality:** classical detector fabricates false "returns" from
  shoe/short blobs despite the trajectory logic. Is the priority chain
  (manual > external > TrackNet > classical) the right architecture, and is the
  classical fallback worth keeping vs. requiring an external/finetuned track?
- **Hardcoded `conf_threshold=0.5`** in TrackNet is outside the learnable params.
- **`ball_filter_enabled=0`** by default means the arc-RANSAC event layer is
  effectively dead in the shipping config — is that the right call?
- **Mixed-source continuity:** the selector is `seed()`-ed when a non-classical
  source produces a detection (`pass2.py:284-289`) so trajectory history holds if
  sources mix mid-window. Verify this is coherent.

---

## 4. Shared geometry & 3D (evaluate as its own layer)

**File:** `backend/app/cv/court.py`

- World frame in metres, origin at the **far-left** court corner; court
  6.096 × 13.4112 m; net at y=6.7056; kitchen ±2.1336 m.
- **Homography** `H` (image px → court plane z=0) from ≥4 named clicked reference
  points (`fit_homography`, RANSAC). Fixed camera → one calibration per setup,
  reused across clips. `suggest_corners` auto-proposes the 4 outer corners by
  Hough lines + verifying projected net/kitchen lines land on paint.
- **Self-calibrated focal length** from the homography (orthogonality of rotation
  columns; principal point assumed at image center). Unlocks ball-height-from-size.
  **Degenerate-fit fallback** `f = img_w·0.9` with a logged warning — height
  reconstruction is then unreliable (`court.py:122-135`).
- `CameraModel.ball_world_from_size` (`court.py:154-162`) is the small-object depth
  model feeding all 3D.

**Evaluate:** Is a single-view, size-based depth model the right foundation, given
the far-ball noise? Is the focal self-calibration robust to imperfect corner
clicks? Is the fixed origin/orientation assumption (`far-left`, camera above plane
handedness flip at `court.py:147-151`) safe?

---

## 5. Data flow, storage, parameters

- **Frame I/O:** `cv/video_io.py::iter_frames` — sequential grab/retrieve (no
  per-frame seeking). Pass 1 uses `step`/`scale`/`with_full=True`; Pass 2 iterates
  each window at full FPS.
- **Overlays:** `cv/overlays.py` — chunked JSON (900 frames/chunk), streamed to the
  UI as passes run. Pass 2 entry fields: `f, players[], ball, world, pspeed, evt,
  obs, interp`.
- **DB:** `backend/app/db.py`. Relevant tables: `videos, jobs, rally_labels
  (human|pass1|pass2), overlays, calibrations, learned_params, ball_corrections,
  ball_tracks, reports, discrepancies, model_registry`.
- **LearnedParams:** `backend/app/learning/params.py` — versioned global dict, every
  value **clamped** by `BOUNDS`. Loaded fresh on every upload so lessons compound.
  **Currently-inert params** (rally-decision code not called): `serve_speed_mps,
  serve_behind_margin_m, serve_formation_lookback_s, confirm_window_s,
  player_active_mps, p2_reject_passes, start_offset_frames, end_offset_frames`
  (`params.py:121-133`, and see the spec). Live ball/pose params are all in §2.5/§3.

---

## 6. Hard constraints (must hold in any rebuild)

1. **Camera:** single fixed camera, looking **down the length** of the court
   (axial view). Bounces have no clean 2D "V"; depth and height both project onto
   the image's vertical axis. Far objects are tiny.
2. **Compute target:** consumer box with an **Intel Arc iGPU**. Production path is
   **OpenVINO FP16 on the GPU** for both YOLO-pose and TrackNet; eager torch-XPU is
   slow and Ultralytics rejects `device='xpu'` for `.track()`. CPU is the last
   resort. No assumption of NVIDIA CUDA at deploy time.
3. **Overlay, don't burn:** detections are stored as JSON overlays keyed by frame;
   the original video is never edited. Coordinates are **full-resolution image
   pixels, origin top-left** (see `BALL_TRACKER_INTEGRATION.md`).
4. **Global-vs-clip routing:** method-level improvements update the **global**
   `LearnedParams` (clamped, versioned); clip-specific oddities live on the clip
   row (e.g. `ball_corrections`, clip-local calibration). Do not push clip-local
   noise into global state.
5. **Pass 1 owns rally boundaries; Pass 2 owns detection** (current spec). If you
   want to re-enable Pass 2 rally decisions, treat it as an explicit, separately
   justified change — not an incidental side effect of a detector rebuild.

---

## 7. Invariants that exist because they fixed real bugs (don't regress these)

- **One YOLO model per tracker instance** (ByteTrack state isolation).
- **Keep `id=None` detections** (don't drop unconfirmed tracks).
- **Cold-start ball seeding requires two-frame ball-speed motion match** (a
  stationary shoe must never seed the ball track).
- **Permissive Pass 2 detector + `_in_court_region` filter** (dropping box blobs at
  the detector level made it latch onto neighbouring-court motion → phantom
  "bounce OUT").
- **Phantom-out guard** for out-of-court bounces (must be a real, height-known
  detection within `out_max_dist_m` of a line).
- **Identity slots persist across windows; only positions are forgotten.**

---

## 8. What we want from you (evaluation questions + success criteria)

### 8.1 Per-layer verdicts (answer each)
1. **Frame I/O / windowing:** is sequential decode + Pass-1-window reconstruction
   in `pass2.py:94-114` correct and efficient? Any off-by-one / boundary risks?
2. **Player detector+tracker:** is YOLO11-pose-nano + ByteTrack + per-instance
   model + OpenVINO the right stack here? Is the `imgsz=1920` recall/latency
   trade-off defensible? Is the exclusion-zone approach sound or should it be
   geometry-derived?
3. **Player identity:** is the HS-histogram + court-distance re-ID adequate, or a
   fragility? Any better lightweight approach within the CPU/no-extra-model
   constraint?
4. **Ball detector:** is the priority chain right? Is the classical fallback worth
   keeping? Should TrackNet fine-tuning be the mandated path, and is the loading /
   OpenVINO / thresholding architecture sound?
5. **Ball trajectory/selection:** is motion-primary selection + cold-start +
   foot-suppression robust? Where does it break?
6. **Geometry/3D:** is single-view size→depth the right foundation for height on
   this camera? Alternatives given the far-ball noise?
7. **Event detection:** is the heuristic (`detect_events`) sound as a
   *visualization-only* layer, and is the `BounceGRU` featurization
   (`bounce.py:206-231`) train/serve-skew-free?
8. **Persistence/params:** is the clamped-global-params + overlay design sound? Any
   param that should exist but doesn't (e.g. TrackNet `conf_threshold`)?

### 8.2 Cross-cutting
- Where is the architecture **coupled to the dormant rally-decision code** in a way
  that will bite when it's re-enabled?
- Any **train/serve skew**, **hardcoded magic numbers outside `LearnedParams`**, or
  **silent fallbacks** that mask failures?
- Is there a **simpler** architecture that meets the constraints, or unnecessary
  complexity that should be removed?

### 8.3 Definition of "works well"
- Player location: correct court position + stable 1–4 numbers for all *visible*
  players throughout each rally window (accepting the geometry-occluded 2nd far
  player).
- Ball position: continuous, low-jitter track through rallies with far-baseline
  behavior explicitly characterized (not silently wrong). Bounce/hit markers
  plausible enough to be useful overlays.
- No regressions against the invariants in §7.

### 8.4 Output format we want
1. **Per-layer assessment** (§8.1) — verdict + top risks each.
2. **Prioritized issue list** (severity × effort).
3. **Rebuild proposal** — surgical changes first; if a subsystem warrants a
   rewrite, justify it against §6 and give a migration path and how to validate
   (the `backend/scripts/eval_matrix.py` / `eval_ball_filter.py` harnesses and the
   stored `ball_corrections` / `reference_data/ball_gt.json` are available as
   ground truth).

---

## 9. File map (where to look)

| Concern | Files |
|---|---|
| Pass orchestration | `backend/app/pipeline.py`, `cv/pass1.py`, `cv/pass2.py` |
| Player detection | `cv/players.py`, `cv/identity.py`, `cv/exclusions.py`, `cv/device.py` |
| Ball — classical | `cv/ball_coarse.py`, `cv/ball_track.py` |
| Ball — TrackNet | `cv/ball_tracknet.py`, `scripts/get_tracknet_weights.py` |
| Ball — filter/smoothing/3D/events | `cv/ball_filter.py`, `cv/kalman.py`, `cv/bounce.py` |
| Geometry / camera | `cv/court.py` |
| Frame I/O + overlays | `cv/video_io.py`, `cv/overlays.py` |
| Params / learning | `learning/params.py`, `learning/updates.py`, `learning/autolearn.py`, `learning/model_registry.py` |
| Storage / API | `app/db.py`, `app/api/*.py` |
| Dormant rally logic | `cv/rally_logic.py` (present, not called) |
| Specs (read these too) | `PASS1_PASS2_SPEC.md`, `BALL_TRACKER_INTEGRATION.md`, `README.md`, `LEARNING_TRACKED_VALUES.md` |
| Eval harnesses / GT | `scripts/eval_matrix.py`, `scripts/eval_ball_filter.py`, `scripts/ball_gt_io.py`, `reference_data/ball_gt.json` |

---

## 10. TL;DR of current known problems (your starting hypotheses)

1. **Far-baseline ball is the accuracy ceiling** — few px wide → noisy size→depth →
   weak height/bounce. Pretrained TrackNet doesn't transfer; fine-tuning or an
   external dense track is the stated fix.
2. **Classical detector is honestly weak** and can fabricate false returns from
   feet/shoes; the trajectory selector mitigates but doesn't cure it.
3. **2nd far doubles player is geometry-occluded** — likely unfixable by
   resolution/model alone.
4. **Hardcoded, camera-specific values** (exclusion zones, TrackNet threshold)
   sit outside the learnable/global system.
5. **The strong ball-filter/arc-event layer is off by default** and tuned for a
   detector (TrackNet) that is also off by default — the shipping path is the
   weakest combination.
6. **Detection is decoupled from rally decisions** (Pass 2 emits no boundaries);
   several params and the whole `rally_logic` module are dormant — evaluate the
   coupling before anything is re-enabled.

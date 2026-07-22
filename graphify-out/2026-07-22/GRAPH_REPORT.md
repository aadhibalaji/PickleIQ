# Graph Report - CV-Training-Tool  (2026-07-20)

## Corpus Check
- 122 files · ~144,368 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1401 nodes · 3082 edges · 60 communities (56 shown, 4 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 61 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0cc26519`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- contact_detect.py
- TrackingData
- coaching.py
- eval_video
- pass1.py
- validate.py
- storage_s3.py
- Features
- annotate.py
- io_loaders.py
- TrajectoryBallSelector
- connect
- loads
- Workspace.jsx
- RefineConfig
- model_registry.py
- PlayerIdentityTracker
- postprocess
- app/config.py
- CameraModel
- db.py
- analyze.py
- 2. PASS 2 (refine module) — keep / start-later / end-sooner
- media.py
- jobs.py
- refine_window
- test_serve_labels.py
- frontend/package.json
- BallWorld
- Reporting & Improvement — Tracked & Adjusted Values
- params.py
- dumps
- court.py
- bounce.py
- package.json
- pass2.py
- dataset.py
- Audit: Does manual labeling actually improve the models?
- Rally Analysis & Coaching Feedback — Multi-Agent Build Prompts
- build_dataset
- refine.py
- Ball Detection — In-Depth Guide
- Codebase Context — PickleCV / CV-Training-Tool
- features.py
- eval_matrix.py
- Session Handoff — Labeled-eval workflow for the Pass 2 serve gate (2026-07-15)
- manifest.json
- restart-ec2.sh
- restart-local.sh
- Pickleball AI Rally-Detection System
- Hardware profiles (`HW_PROFILE`)
- Connecting an external ball tracker (e.g. MTI + temporal integration)
- Pass 1 & Pass 2 — Current Detection Spec
- CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `connect()` - 105 edges
2. `RefineConfig` - 71 edges
3. `run_pass2()` - 37 edges
4. `TrackingData` - 34 edges
5. `compute_features()` - 31 edges
6. `Features` - 28 edges
7. `now()` - 25 edges
8. `loads()` - 25 edges
9. `dumps()` - 25 edges
10. `refine_window()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `get_overlays()` --calls--> `read_range()`  [EXTRACTED]
  backend/app/api/analyze.py → backend/app/cv/overlays.py
- `delete_window_grade()` --calls--> `connect()`  [EXTRACTED]
  backend/app/api/annotate.py → backend/app/db.py
- `delete_ball_correction()` --calls--> `connect()`  [EXTRACTED]
  backend/app/api/annotate.py → backend/app/db.py
- `get_ball_track_info()` --calls--> `connect()`  [EXTRACTED]
  backend/app/api/annotate.py → backend/app/db.py
- `delete_ball_track()` --calls--> `connect()`  [EXTRACTED]
  backend/app/api/annotate.py → backend/app/db.py

## Import Cycles
- None detected.

## Communities (60 total, 4 thin omitted)

### Community 0 - "contact_detect.py"
Cohesion: 0.05
Nodes (77): detect_contacts(), generate_coaching_report(), get_coaching_report(), list_contact_events(), Rally Analysis & Coaching endpoints. Read-only against rally_labels, overlays, b, Runs pose-only contact detection for every rally of `source` on this     video a, Per-rally Pass 2 readiness, so the frontend can show a 'waiting on     Pass 2' s, Triggers coaching report generation (one Anthropic API call) and     persists it (+69 more)

### Community 1 - "TrackingData"
Cohesion: 0.09
Nodes (51): compute_features(), detect_bounces(), formation_score(), _median_filter(), _quiet(), Bounce mask at native rate. Prefers the overlay's evt bounces (arc     junction, Soft serve-formation score in [0,1] per native frame.      Best over the two ser, Silence numpy's all-NaN / empty-slice RuntimeWarnings — all-NaN windows     are (+43 more)

### Community 2 - "coaching.py"
Cohesion: 0.08
Nodes (53): _angle_deg(), build_report(), _classify(), compute_feature_vector(), _elbow_angle(), generate_coaching_report(), get_report(), _hitting_side() (+45 more)

### Community 3 - "eval_video"
Cohesion: 0.05
Nodes (52): Arc, ArcEvent, ArcPoint, BallFrameFilter, classify_arc_events(), _near_box(), ndarray, _ransac_quadratic() (+44 more)

### Community 4 - "pass1.py"
Cohesion: 0.09
Nodes (27): in_excluded(), _point_in_poly(), Image-space player-detection exclusion zones.  Regions of the FIXED camera frame, Ray-casting point-in-polygon on normalized coords., True if pixel (x_px, y_px) lies inside any exclusion zone., _match_fast(), Pass 1 — the coarse gate. Runs on 100% of the footage, cheaply, at low resolutio, Mutually-nearest, size-consistent blob match with ball-like speed.     Returns ( (+19 more)

### Community 5 - "validate.py"
Cohesion: 0.13
Nodes (28): Rally, clear_override(), set_override(), _accept_gate(), _accept_model(), _accept_offset(), analytic_metrics(), _apply_offsets() (+20 more)

### Community 6 - "storage_s3.py"
Cohesion: 0.06
Nodes (65): poll_cloud_windows(), Lightweight incremental pull for live Pass 2 visibility while cloud     processi, _ingest_video(), _known_video_ids(), _loop(), EC2 worker side of the hybrid handoff: discovers videos a RUN_PROCESSING=0 UI-on, Adds video_id to this instance's own DB WITHOUT downloading it or     starting p, Downloads the video (needed locally for Pass 1/2 regardless — see     storage_s3 (+57 more)

### Community 7 - "Features"
Cohesion: 0.08
Nodes (26): _NumpyLogistic, ndarray, Path, Minimal L2 logistic regression (fallback when xgboost is missing)., Learned P(live). XGBoost when available, numpy logistic otherwise.      The save, XGBoostClassifier, Features, Feature timeline + event series for one analysis span. (+18 more)

### Community 8 - "annotate.py"
Cohesion: 0.05
Nodes (60): _attach_grades(), BallCorrectionIn, BallCorrectionUpdate, BallTrackIn, CalibrationIn, delete_ball_correction(), delete_ball_track(), delete_note() (+52 more)

### Community 9 - "io_loaders.py"
Cohesion: 0.08
Nodes (38): Path, Score labeled serve / non-serve spans and report gate accuracy.     Each JSON fi, validate_serve_gate(), clips_with_human_labels(), load_homography(), load_pass1_windows(), load_tracking(), load_video() (+30 more)

### Community 10 - "TrajectoryBallSelector"
Cohesion: 0.09
Nodes (21): BallCandidate, MotionBallDetector, on_player_box(), ndarray, Classical fast-moving-small-blob ball detector.  Two jobs: 1. Pass 1 coarse gate, True if (x, y) sits inside any player box expanded outward by the shared     mar, Three-frame differencing keeps only objects moving between BOTH frame     pairs, Whether the blob's colour passes the appearance gate. Sampled from the         b (+13 more)

### Community 11 - "connect"
Cohesion: 0.14
Nodes (29): reject_change(), add_ball_correction(), add_note(), add_window_grade(), put_ball_track(), put_human_labels(), save_calibration(), upload_video() (+21 more)

### Community 12 - "loads"
Cohesion: 0.13
Nodes (16): approve_change(), learning_scoreboard(), list_pending_changes(), list_reports(), Gate changes awaiting human approval (plus recently decided ones)., Approve a queued gate change: apply it immediately as a new global param     ver, The 'is it getting better?' trend: mean absolute start/end boundary error     (f, get_calibration() (+8 more)

### Community 13 - "Workspace.jsx"
Cohesion: 0.11
Nodes (18): api, j(), App(), CalibrationPanel(), LabelingPanel(), CoachingVideo(), formatCategory(), formatTime() (+10 more)

### Community 14 - "RefineConfig"
Cohesion: 0.11
Nodes (33): LiveClassifier, make_classifier(), Stage B — live/dead classification, two interchangeable backends.  Both backends, Backend per config; a missing/broken model artifact falls back to     rules with, Common interface: per-timestep P(live) from a Features object., Return a copy with the given fields replaced (used by sweeps)., RefineConfig, EvalResult (+25 more)

### Community 15 - "model_registry.py"
Cohesion: 0.09
Nodes (26): learning_curve(), list_models(), promote_model(), Trained-model registry endpoints: list versions, promote/reject, and the learnin, Held-out end/start boundary error per model version against the number of     la, reject_model(), build_bounce_model(), load_bounce_model() (+18 more)

### Community 16 - "PlayerIdentityTracker"
Cohesion: 0.15
Nodes (11): _Identity, PlayerIdentityTracker, ndarray, Match-stable player identity via appearance + court-position re-ID.  ByteTrack (, Start a fresh Pass 2 window. Keep identities (id/number/appearance)         but, Set o.slot and o.number on each obs that has a team + court position.         Ob, Minimum-cost injective matching of dets <-> idents. N is tiny (<=2 per         t, Anchor official numbers left->right per side (near 1,2 / far 3,4;         single (+3 more)

### Community 17 - "postprocess"
Cohesion: 0.16
Nodes (21): hysteresis(), live_ratio_filter(), merge_segments(), postprocess(), ndarray, Stage C — temporal post-processing of the per-timestep P(live).  Order matters a, Half-open [start, end) in feature-timeline sample indices., Two-threshold state machine: LIVE when p>enter, back to DEAD only when     p<exi (+13 more)

### Community 18 - "app/config.py"
Cohesion: 0.08
Nodes (33): Central configuration. Loads .env from the repo root., _build_model(), load_state(), _openvino_infer(), ndarray, Path, TrackNetV2 heatmap ball tracker — the real per-frame model for Pass 2 windows., Load a checkpoint and normalise its state dict (WASB stores     {'model_state_di (+25 more)

### Community 19 - "CameraModel"
Cohesion: 0.12
Nodes (19): CameraModel, implied_person_height(), 3D ball centre from pixel position + apparent diameter.         Small-object mod, 1-sigma HEIGHT uncertainty of ball_world_from_size for a diameter         error, Height of a person standing at `foot_court_xy` (court metres, from the     homog, Full single-view camera recovered from the plane homography.      H maps image->, fit_flight(), Flight (+11 more)

### Community 20 - "db.py"
Cohesion: 0.14
Nodes (19): chunk_index(), clear(), Overlay metadata persistence. All annotation the frontend draws is stored as JSO, Entries must each carry an 'f' frame field. Merges with any chunk     already st, write_entries(), SQLite persistence. One connection per call site (SQLite + WAL handles concurren, describe_chain_status(), export_video_results() (+11 more)

### Community 21 - "analyze.py"
Cohesion: 0.10
Nodes (29): compare(), Human-vs-AI rally comparison: greedy IoU matching, temporal IoU, and separate st, Greedy best-IoU matching (rallies rarely overlap, so greedy ≈ optimal).     Retu, temporal_iou(), generate_llm_report(), Discrepancy feedback report via the Anthropic API.  The LLM is used ONLY here —, Returns (report_markdown, clamped param_suggestions). Degrades to     (None, []), apply_suggestions() (+21 more)

### Community 22 - "2. PASS 2 (refine module) — keep / start-later / end-sooner"
Cohesion: 0.06
Nodes (32): 0. Two meanings of "Pass 2" — do not conflate, 1.1 Per-sample gate signal, 1.2 Start/end via hysteresis, 1.3 `close_rally` — the emitted boundaries, 1. PASS 1 — how a rally starts and ends, 2.1 Stage B — P(live) per timestep (`feature_hz = 15`), 2.1a `facing_away_retreat` — player-motion signal (2026-07-14), 2.1b `no_tracking_data` — missing pass2 overlay, not dead play (2026-07-20) (+24 more)

### Community 23 - "media.py"
Cohesion: 0.10
Nodes (17): get_video(), list_videos(), BaseModel, Upload, video streaming (range requests), job status, reprocess., Per-clip metadata that isn't produced by a pass — currently just     match_type,, Manual trigger (UI button) — checks S3 for this video's exported     cloud-proce, reprocess(), stream_video() (+9 more)

### Community 24 - "jobs.py"
Cohesion: 0.10
Nodes (11): Job, JobCancelled, JobHandle, JobRunner, Background job runner.  Processing is offline/asynchronous: uploads return immed, job_id -> number of jobs ahead of it in the queue (0 = next up)., DB-backed, independent of self._q's own in-memory state. submit()         writes, Raised inside a job when it has been asked to stop. (+3 more)

### Community 25 - "refine_window"
Cohesion: 0.16
Nodes (23): Hand-written weighted rules squashed through a sigmoid.      Positive evidence:, RulesClassifier, Pass1Window, One Pass 1 candidate rally window (frames, inclusive)., Run stages A-D over one candidate window's tracking span., _record(), refine_window(), Legal serve at t=pre_s, rally, then dead tail. Returns     (td, t_contact, t_las (+15 more)

### Community 26 - "test_serve_labels.py"
Cohesion: 0.10
Nodes (32): preview_serve_dataset(), Build the model-ready serve dataset from this clip's human rally marks     WITHO, Emit the per-clip serve dataset files (serve-events sidecar, rally file,     ser, write_serve_dataset(), load_label_file(), Path, Rally spans from a CSV or JSON label file (seconds)., build_dataset() (+24 more)

### Community 27 - "frontend/package.json"
Cohesion: 0.11
Nodes (18): dependencies, react, react-dom, devDependencies, vite, @vitejs/plugin-react, name, private (+10 more)

### Community 28 - "BallWorld"
Cohesion: 0.19
Nodes (20): BallEvent, BallWorld, _events_from_model(), Run the trained per-frame bounce/hit classifier and lift its non-zero     predic, side_of_net(), detect_end(), detect_rallies(), detect_start() (+12 more)

### Community 29 - "Reporting & Improvement — Tracked & Adjusted Values"
Cohesion: 0.11
Nodes (18): 1a. Pass 1 coarse-gate parameters, 1b. Pass 2 ball / bounce parameters, 1c. Rally start / end parameters, 1d. Learned boundary offsets — the primary compounding knobs, 4a. Per-rally discrepancy fields (`discrepancies` table + report payload), 4b. Per-clip summary metrics (`reports.summary` + `compare` output), 4c. Scoreboard trend (`param_eval` table → `GET /api/learning/scoreboard`), Cost / model config (from [config.py](backend/app/config.py)) (+10 more)

### Community 30 - "params.py"
Cohesion: 0.36
Nodes (8): apply_updates(), clamp(), ensure_default_version(), get_latest(), get_version(), Any, LearnedParams: the versioned, GLOBAL tunable state that makes accuracy compound., Create a NEW version = latest + updates (clamped). Returns version.

### Community 31 - "dumps"
Cohesion: 0.19
Nodes (20): get_params(), list_ball_corrections(), list_notes(), chat(), ChatMessage, ChatRequest, _dispatch(), BaseModel (+12 more)

### Community 32 - "court.py"
Cohesion: 0.17
Nodes (14): court_to_img(), fit_homography(), in_court(), ndarray, Court model, homography and single-view geometry.  World frame (metres): pickleb, Fraction of sampled points along segment p0->p1 that land on white     court-lin, Best-effort auto-suggestion of the 4 outer corners from court line     paint. Th, named_pixels: [{name, x, y}] with name in REFERENCE_POINTS.     Returns 3x3 H ma (+6 more)

### Community 33 - "bounce.py"
Cohesion: 0.17
Nodes (17): _demo(), detect_events(), _median_filter(), ndarray, Height reconstruction + bounce-vs-hit classification (the axial-view hard part)., Find trajectory events and classify bounce vs hit.      Candidate events = local, Per-frame feature vector aligned with BOUNCE_MODEL_FEATURES:     [x, y, vx, vy,, Self-check for the height-uncertainty gate + bounce plane-projection:       1. r (+9 more)

### Community 34 - "package.json"
Cohesion: 0.14
Nodes (13): concurrently, description, devDependencies, concurrently, name, private, scripts, build (+5 more)

### Community 35 - "pass2.py"
Cohesion: 0.15
Nodes (17): suggest_corners(), _foot_points(), _in_court_region(), Pass 2 — precise DETECTION / VISUALIZATION inside the Pass 1 rally windows only, ``on_window(last_covered_frame)`` fires after each processing window's     overl, Reject ball blobs whose court-plane projection is well outside THIS court     —, Ankle + knee keypoints (COCO 13-16) of every tracked player — the pixel     loca, Wrist keypoints (COCO 9,10) of every tracked player — the closest proxy     to a (+9 more)

### Community 36 - "dataset.py"
Cohesion: 0.28
Nodes (8): _add(), build_end_dataset(), _labeled_videos(), ndarray, Training-dataset assembly from stored human labels + Pass 2 feature traces.  Tur, Return (windows, labels, meta). windows: list of (Ti, F) arrays; labels:     lis, Pad variable-length windows to a fixed T with IGNORE labels (masked in     the l, to_padded()

### Community 37 - "Audit: Does manual labeling actually improve the models?"
Cohesion: 0.12
Nodes (16): 1.1 Capture, 1.2 Storage, 1.3 Consumers — the whole list (repo-wide grep confirmed only 3), 1.4 The gap (ball), 2.1 Capture, 2.2 Storage, 2.3 Training flows (both are manual CLI), 2.4 Feedback loops (does more labeling ⇒ better model?) (+8 more)

### Community 38 - "Rally Analysis & Coaching Feedback — Multi-Agent Build Prompts"
Cohesion: 0.12
Nodes (16): 0. Hard constraints (apply to every agent, every wave), How to run this (one-time, no manual copy-pasting between waves), Orchestration instructions (Claude Code: read and execute this section), Prompt 0 — Foundation & Contracts (run first, solo), Prompt 1 — Contact/Swing Event Detection, Prompt 2 — Position-Based Rally Metrics, Prompt 3 — Frontend Tab Build-Out, Prompt 4 — Shot-Based Metrics (+8 more)

### Community 39 - "build_dataset"
Cohesion: 0.25
Nodes (9): main(), Build the model-ready SERVE dataset from accumulated human rally marks.  Every h, resolve_clips(), build_dataset(), fit_logistic(), main(), ndarray, Train the small Pass 1 active/dead gate classifier from accumulated data.  The P (+1 more)

### Community 40 - "refine.py"
Cohesion: 0.12
Nodes (22): Path, Repo-root resolution, shared by config.py / hw_profile.py / refine/*.  Computed, repo_root(), Single config for the whole refinement pipeline.  Every window, threshold and we, save_config(), _load_gt(), plot_clip(), Path (+14 more)

### Community 41 - "Ball Detection — In-Depth Guide"
Cohesion: 0.12
Nodes (15): 10. File map, 1. Two passes, two very different ball jobs, 2. The raw detector (`cv/ball_coarse.py`), 3. TrackNet — the real per-frame model (`cv/ball_tracknet.py`), 4. Ball source priority in Pass 2, 5. The semantic filtering layer (`cv/ball_filter.py`), 6. The trajectory selector (`cv/ball_track.py`), 7. Smoothing & 3D reconstruction (+7 more)

### Community 42 - "Codebase Context — PickleCV / CV-Training-Tool"
Cohesion: 0.12
Nodes (15): APIs (`/api` prefix), Categories / flags, Codebase Context — PickleCV / CV-Training-Tool, DB, Env / hardware (this machine), First actions for a new session, Flow, Git / working tree (this snapshot) (+7 more)

### Community 43 - "features.py"
Cohesion: 0.15
Nodes (17): _detect_appearances(), _detect_crossings(), facing_away_retreat(), ndarray, Stage A — per-timestep feature extraction inside one analysis span.  Everything, Sign flips of the ball's net side between visible samples that are at     most m, Ball (re)appearances: a REAL observation (not arc-interpolated) after     >= ser, 1.0 at/behind the baseline, ramping to 0 by m metres inside the court.     sign= (+9 more)

### Community 44 - "eval_matrix.py"
Cohesion: 0.60
Nodes (4): main(), Held-out evaluation matrix for the ball pipeline.  Runs eval_ball_filter.py over, run_config(), summarize()

### Community 45 - "Session Handoff — Labeled-eval workflow for the Pass 2 serve gate (2026-07-15)"
Cohesion: 0.13
Nodes (14): `backend/app/refine/evaluate.py`, `backend/app/refine/recommend.py`, `backend/app/refine/refine.py`, Code changes this session (all still in the working tree), Current benchmark (5 labeled clips, current live config — NOT changed), `data/refine/config.override.yaml`, Key files, Key lesson (read this before trusting any future sweep number) (+6 more)

### Community 55 - "Pickleball AI Rally-Detection System"
Cohesion: 0.18
Nodes (10): Architecture, Configuration, First-run flow, In-app assistant (💬 Assistant tab), Known behavior on this machine (Intel Arc 140V), Learning is global by default, Pickleball AI Rally-Detection System, Quick start (+2 more)

### Community 56 - "Hardware profiles (`HW_PROFILE`)"
Cohesion: 0.20
Nodes (9): Adding a new profile, Every knob and what it plugs into, Hardware profiles (`HW_PROFILE`), Known interactions, `p1_scale` / `p1_frame_step` — the one subtlety, Precedence rules, Quick start, Verifying (+1 more)

### Community 57 - "Connecting an external ball tracker (e.g. MTI + temporal integration)"
Cohesion: 0.22
Nodes (8): After uploading, Ball-source priority in Pass 2, Connecting an external ball tracker (e.g. MTI + temporal integration), Coordinate + frame conventions (important), Endpoint, Minimal Python example, Tighter (in-process) alternative, Why this helps (ties the whole system together)

### Community 58 - "Pass 1 & Pass 2 — Current Detection Spec"
Cohesion: 0.25
Nodes (7): Key files, Known limitations / open problems, Learnable / trainable state, Orchestration, PASS 1 — coarse gate (`backend/app/cv/pass1.py :: run_pass1`), Pass 1 & Pass 2 — Current Detection Spec, PASS 2 — detection / visualization only (`backend/app/cv/pass2.py :: run_pass2`)

## Knowledge Gaps
- **158 isolated node(s):** `schema_version`, `clips`, `name`, `private`, `version` (+153 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `connect()` connect `connect` to `contact_detect.py`, `coaching.py`, `pass2.py`, `pass1.py`, `dataset.py`, `storage_s3.py`, `validate.py`, `annotate.py`, `io_loaders.py`, `refine.py`, `loads`, `model_registry.py`, `db.py`, `analyze.py`, `media.py`, `params.py`, `dumps`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `RefineConfig` connect `RefineConfig` to `TrackingData`, `Features`, `refine.py`, `io_loaders.py`, `annotate.py`, `features.py`, `build_dataset`, `postprocess`, `refine_window`, `test_serve_labels.py`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `run_pass2()` connect `pass2.py` to `court.py`, `bounce.py`, `contact_detect.py`, `eval_video`, `pass1.py`, `validate.py`, `annotate.py`, `TrajectoryBallSelector`, `connect`, `loads`, `app/config.py`, `CameraModel`, `db.py`, `jobs.py`, `BallWorld`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `RefineConfig` (e.g. with `LiveClassifier` and `_NumpyLogistic`) actually correct?**
  _`RefineConfig` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `TrackingData` (e.g. with `Features` and `IncrementalRefiner`) actually correct?**
  _`TrackingData` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `schema_version`, `clips`, `name` to the rest of the system?**
  _158 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `contact_detect.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05088919288645691 - nodes in this community are weakly interconnected._
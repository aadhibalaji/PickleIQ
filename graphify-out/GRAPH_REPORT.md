# Graph Report - CV-Training-Tool  (2026-08-04)

## Corpus Check
- 145 files · ~178,346 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1763 nodes · 3994 edges · 90 communities (85 shown, 5 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 66 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2f5b867f`
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
- cloud_ingest.py
- ball_filter.py
- detect_contacts_for_video
- pipeline.py
- io_loaders.py
- overlays.py
- rally_analysis.py
- MotionBallDetector
- _scale_m_per_px
- autolearn.py
- refine/config.py
- BallFrameFilter
- export_manual_tracking.py
- test_ball_filter.py
- deploy.sh
- ndarray
- media.py
- _finalize_vector
- test_progress_sync.py
- test_train_eval.py
- _robust_fence
- results_exist
- ShotFeatureVector
- _export_video
- poll_cloud_windows
- download_to_scratch
- project_court
- start_processing

## God Nodes (most connected - your core abstractions)
1. `connect()` - 131 edges
2. `RefineConfig` - 73 edges
3. `TrackingData` - 39 edges
4. `run_pass2()` - 38 edges
5. `now()` - 34 edges
6. `compute_features()` - 32 edges
7. `dumps()` - 29 edges
8. `Features` - 29 edges
9. `rows_to_dicts()` - 27 edges
10. `loads()` - 26 edges

## Surprising Connections (you probably didn't know these)
- `delete_window_grade()` --calls--> `connect()`  [EXTRACTED]
  backend/app/api/annotate.py → backend/app/db.py
- `delete_ball_correction()` --calls--> `connect()`  [EXTRACTED]
  backend/app/api/annotate.py → backend/app/db.py
- `get_ball_track_info()` --calls--> `connect()`  [EXTRACTED]
  backend/app/api/annotate.py → backend/app/db.py
- `delete_ball_track()` --calls--> `connect()`  [EXTRACTED]
  backend/app/api/annotate.py → backend/app/db.py
- `delete_note()` --calls--> `connect()`  [EXTRACTED]
  backend/app/api/annotate.py → backend/app/db.py

## Import Cycles
- 3-file cycle: `backend/app/cv/pass1.py -> backend/app/jobs.py -> backend/app/pipeline.py -> backend/app/cv/pass1.py`
- 3-file cycle: `backend/app/cv/pass2.py -> backend/app/jobs.py -> backend/app/pipeline.py -> backend/app/cv/pass2.py`

## Communities (90 total, 5 thin omitted)

### Community 0 - "contact_detect.py"
Cohesion: 0.05
Nodes (74): get_overlays(), detect_contacts(), process_selection(), Per-rally Pass 2 readiness, so the frontend can show a 'waiting on     Pass 2' s, Enqueues Pass 2 for one selection. If it's a Pass 1-accepted rally and     the n, Runs pose-only contact detection for every rally of `source` on this     video a, readiness(), read_range() (+66 more)

### Community 1 - "TrackingData"
Cohesion: 0.08
Nodes (60): compute_features(), detect_bounces(), formation_score(), _median_filter(), _quiet(), Bounce mask at native rate. Prefers the overlay's evt bounces (arc     junction, Soft serve-formation score in [0,1] per native frame.      Best over the two ser, Silence numpy's all-NaN / empty-slice RuntimeWarnings — all-NaN windows     are (+52 more)

### Community 2 - "coaching.py"
Cohesion: 0.11
Nodes (31): _angle_deg(), _elbow_angle(), _extract_raw_metrics(), generate_coaching_report(), get_report(), _hitting_side(), _knee_angle(), _kp_xy() (+23 more)

### Community 3 - "eval_video"
Cohesion: 0.12
Nodes (25): _attribute_bounce_miss(), _attribute_mark(), _box_dist(), _court_lines_img(), eval_video(), fmt_clip(), gap_stats(), in_foot_zone() (+17 more)

### Community 4 - "pass1.py"
Cohesion: 0.10
Nodes (21): in_excluded(), _point_in_poly(), Image-space player-detection exclusion zones.  Regions of the FIXED camera frame, Ray-casting point-in-polygon on normalized coords., True if pixel (x_px, y_px) lies inside any exclusion zone., Pass 2 — precise DETECTION / VISUALIZATION inside the Pass 1 rally windows only, _build_model(), is_serve_formation() (+13 more)

### Community 5 - "validate.py"
Cohesion: 0.07
Nodes (47): compare(), Rally, Human-vs-AI rally comparison: greedy IoU matching, temporal IoU, and separate st, Greedy best-IoU matching (rallies rarely overlap, so greedy ≈ optimal).     Retu, temporal_iou(), detect_debug(), Read-only diagnostics for why a clip matches (or doesn't) its human     labels., labels_hash() (+39 more)

### Community 6 - "storage_s3.py"
Cohesion: 0.11
Nodes (30): calibration_key(), claim_pass2_request(), download_calibration(), download_pass2_request(), download_refine_config(), download_self_stop_state(), download_window_results(), _get_json() (+22 more)

### Community 7 - "Features"
Cohesion: 0.16
Nodes (6): _NumpyLogistic, ndarray, Path, Minimal L2 logistic regression (fallback when xgboost is missing)., Learned P(live). XGBoost when available, numpy logistic otherwise.      The save, XGBoostClassifier

### Community 8 - "annotate.py"
Cohesion: 0.07
Nodes (41): _attach_grades(), BallCorrectionIn, BallCorrectionUpdate, BallTrackIn, CalibrationIn, delete_ball_correction(), delete_ball_track(), delete_note() (+33 more)

### Community 9 - "io_loaders.py"
Cohesion: 0.10
Nodes (31): grade_recommendation(), Sweep serve-gate thresholds against the USER GRADES and recommend the one     wi, Best-effort per-window refine hook for run_pass2: refines each Pass 1     window, Rally refinement over the fresh Pass 2 tracking data (pass2/refine     module)., _refine_after_pass2(), _window_refiner(), Stage B — live/dead classification, two interchangeable backends.  Both backends, load_config() (+23 more)

### Community 10 - "TrajectoryBallSelector"
Cohesion: 0.15
Nodes (13): _demo(), _on_player(), _Pt, Trajectory-consistency ball selection (Pass 2).  Motion — NOT appearance — is th, (predicted_xy, gate_radius) from the recent track, or (None, None)         if th, True once (x,y) has moved in LOCKSTEP with a nearby wrist for         lock_frame, Pick the best ball candidate for this frame (or None). `cands` items         exp, Acquire a new track only from a candidate showing ball-like MOTION         betwe (+5 more)

### Community 11 - "connect"
Cohesion: 0.11
Nodes (40): add_ball_correction(), add_note(), add_window_grade(), save_calibration(), upload_video(), create_selections(), process_selections(), process_single_rally() (+32 more)

### Community 12 - "loads"
Cohesion: 0.36
Nodes (8): apply_updates(), clamp(), ensure_default_version(), get_latest(), get_version(), Any, LearnedParams: the versioned, GLOBAL tunable state that makes accuracy compound., Create a NEW version = latest + updates (clamped). Returns version.

### Community 13 - "Workspace.jsx"
Cohesion: 0.08
Nodes (32): api, j(), App(), CalibrationPanel(), Home(), LabelingPanel(), CoachingSelections(), fmt() (+24 more)

### Community 14 - "RefineConfig"
Cohesion: 0.12
Nodes (18): LiveClassifier, make_classifier(), Backend per config; a missing/broken model artifact falls back to     rules with, Common interface: per-timestep P(live) from a Features object., EvalResult, evaluate(), evaluate_clip(), main() (+10 more)

### Community 15 - "model_registry.py"
Cohesion: 0.13
Nodes (20): learning_curve(), list_models(), promote_model(), Trained-model registry endpoints: list versions, promote/reject, and the learnin, Held-out end/start boundary error per model version against the number of     la, reject_model(), history(), latest_live() (+12 more)

### Community 16 - "PlayerIdentityTracker"
Cohesion: 0.14
Nodes (15): _Identity, PlayerIdentityTracker, ndarray, Match-stable player identity via appearance + court-position re-ID.  ByteTrack (, Start a fresh Pass 2 window. Keep identities (id/number/appearance)         but, Set o.slot and o.number on each obs that has a team + court position.         Ob, Minimum-cost injective matching of dets <-> idents. N is tiny (<=2 per         t, Anchor official numbers left->right per side (near 1,2 / far 3,4;         single (+7 more)

### Community 17 - "postprocess"
Cohesion: 0.16
Nodes (20): hysteresis(), live_ratio_filter(), merge_segments(), postprocess(), ndarray, Half-open [start, end) in feature-timeline sample indices., Two-threshold state machine: LIVE when p>enter, back to DEAD only when     p<exi, Trailing-window majority: in-rally iff >= ratio of the last `window`     samples (+12 more)

### Community 18 - "app/config.py"
Cohesion: 0.08
Nodes (33): _build_model(), load_state(), _openvino_infer(), ndarray, Path, TrackNetV2 heatmap ball tracker — the real per-frame model for Pass 2 windows., Load a checkpoint and normalise its state dict (WASB stores     {'model_state_di, Convert the torch model to OpenVINO FP16 (once, IR cached next to the     weight (+25 more)

### Community 19 - "CameraModel"
Cohesion: 0.07
Nodes (47): get_trajectories(), Per-rally ball trajectory samples for the 3D trajectory view:     {video_id, ral, CameraModel, 1-sigma HEIGHT uncertainty of ball_world_from_size for a diameter         error, Full single-view camera recovered from the plane homography.      H maps image->, _blank_td(), _camera(), ndarray (+39 more)

### Community 20 - "db.py"
Cohesion: 0.24
Nodes (11): describe_chain_status(), export_video_results(), export_window_results(), _job_statuses(), merge_window_results(), Any, Export/import contract for the hybrid architecture: the cloud (EC2) side seriali, Human summary for the sync response, e.g. 'Pass 1: 4 rallies, Pass 2:     error (+3 more)

### Community 21 - "analyze.py"
Cohesion: 0.11
Nodes (28): generate_llm_report(), Returns (report_markdown, clamped param_suggestions). Degrades to     (None, []), apply_suggestions(), ApplySuggestions, approve_change(), generate_report(), get_params(), learning_scoreboard() (+20 more)

### Community 22 - "2. PASS 2 (refine module) — keep / start-later / end-sooner"
Cohesion: 0.06
Nodes (32): 0. Two meanings of "Pass 2" — do not conflate, 1.1 Per-sample gate signal, 1.2 Start/end via hysteresis, 1.3 `close_rally` — the emitted boundaries, 1. PASS 1 — how a rally starts and ends, 2.1 Stage B — P(live) per timestep (`feature_hz = 15`), 2.1a `facing_away_retreat` — player-motion signal (2026-07-14), 2.1b `no_tracking_data` — missing pass2 overlay, not dead play (2026-07-20) (+24 more)

### Community 23 - "media.py"
Cohesion: 0.11
Nodes (15): Discrepancy feedback report via the Anthropic API.  The LLM is used ONLY here —, Central configuration. Loads .env from the repo root., SQLite persistence. One connection per call site (SQLite + WAL handles concurren, Background job runner.  Processing is offline/asynchronous: uploads return immed, FastAPI entry point. Run with:     uvicorn app.main:app --port 8420, load_anthropic_key(), Fetch ANTHROPIC_API_KEY from AWS Secrets Manager when nothing has already set it, _job_row() (+7 more)

### Community 24 - "jobs.py"
Cohesion: 0.16
Nodes (6): Job, JobRunner, job_id -> number of jobs ahead of it in the queue (0 = next up)., Runs job.fn and writes its terminal DB status. Split out of _loop         so it, Pass 2 full chain (pass2 → contacts) or a selection-range Pass 2         reachin, DB-backed, independent of self._q's own in-memory state. submit()         writes

### Community 25 - "refine_window"
Cohesion: 0.15
Nodes (27): Return a copy with the given fields replaced (used by sweeps)., RefineConfig, Features, Feature timeline + event series for one analysis span., _ball_height_at(), _ballistic_contact(), _contact_candidates(), _flight_and_landing() (+19 more)

### Community 26 - "test_serve_labels.py"
Cohesion: 0.10
Nodes (35): load_video(), fps / frame_count / dimensions / filename for one clip., load_ground_truth(), load_label_file(), Path, Stage E (labels) — ground-truth loading + per-timestep label expansion.  Two GT, Human rally spans from the DB, in seconds., Rally spans from a CSV or JSON label file (seconds). (+27 more)

### Community 27 - "frontend/package.json"
Cohesion: 0.11
Nodes (18): dependencies, react, react-dom, devDependencies, vite, @vitejs/plugin-react, name, private (+10 more)

### Community 28 - "BallWorld"
Cohesion: 0.07
Nodes (50): BallEvent, BallWorld, build_bounce_model(), _demo(), detect_events(), _events_from_model(), load_bounce_model(), _median_filter() (+42 more)

### Community 29 - "Reporting & Improvement — Tracked & Adjusted Values"
Cohesion: 0.11
Nodes (18): 1a. Pass 1 coarse-gate parameters, 1b. Pass 2 ball / bounce parameters, 1c. Rally start / end parameters, 1d. Learned boundary offsets — the primary compounding knobs, 4a. Per-rally discrepancy fields (`discrepancies` table + report payload), 4b. Per-clip summary metrics (`reports.summary` + `compare` output), 4c. Scoreboard trend (`param_eval` table → `GET /api/learning/scoreboard`), Cost / model config (from [config.py](backend/app/config.py)) (+10 more)

### Community 30 - "params.py"
Cohesion: 0.08
Nodes (29): _classify(), exclude_serves(), generate_llm_tips(), _llm_shot_payload(), Returns ({contact_id: {tip, tone}}, [{player_label, theme}, ...]), or     ({}, [, Compacts report_stats.get_report_stats()'s per-rally time-series output     into, Serves are excluded from the coaching narrative/worst-best selection     entirel, devs: {metric_field_name: relative_deviation_or_None} for one shot     (see _rel (+21 more)

### Community 31 - "dumps"
Cohesion: 0.30
Nodes (14): chat(), ChatMessage, ChatRequest, _dispatch(), BaseModel, In-app assistant: a Claude conversation with full context of the current clip th, _state_snapshot(), _tool_add_note() (+6 more)

### Community 32 - "court.py"
Cohesion: 0.06
Nodes (50): fit_homography(), implied_person_height(), ndarray, Court model, homography and single-view geometry.  World frame (metres): pickleb, 3D ball centre from pixel position + apparent diameter.         Small-object mod, Height of a person standing at `foot_court_xy` (court metres, from the     homog, Fraction of sampled points along segment p0->p1 that land on white     court-lin, Best-effort auto-suggestion of the 4 outer corners from court line     paint. Th (+42 more)

### Community 33 - "bounce.py"
Cohesion: 0.15
Nodes (14): img_to_court(), pts (N,2) pixels -> (N,2) court metres., obs: sorted (frame, x, y, diameter_px|None). Gaps <= max_gap frames are     brid, smooth_track(), _foot_points(), _in_court_region(), ``on_window(last_covered_frame)`` fires after each processing window's     overl, Reject ball blobs whose court-plane projection is well outside THIS court     — (+6 more)

### Community 34 - "package.json"
Cohesion: 0.14
Nodes (13): concurrently, description, devDependencies, concurrently, name, private, scripts, build (+5 more)

### Community 35 - "pass2.py"
Cohesion: 0.15
Nodes (17): chunk_index(), clear(), Overlay metadata persistence. All annotation the frontend draws is stored as JSO, Entries must each carry an 'f' frame field. Merges with any chunk     already st, write_entries(), _match_fast(), Pass 1 — the coarse gate. Runs on 100% of the footage, cheaply, at low resolutio, Mutually-nearest, size-consistent blob match with ball-like speed.     Returns ( (+9 more)

### Community 36 - "dataset.py"
Cohesion: 0.14
Nodes (24): get_self_stop(), Current state of the EC2 self-stop-on-empty-queue toggle (see     self_stop.py), arm_after_pass2_chain(), _awaiting_pass2_state(), get_enabled(), _pass2_terminal(), _pending_pass2_s3_requests(), Stop this EC2 instance once work is done, so GPU time isn't billed while idle be (+16 more)

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
Cohesion: 0.10
Nodes (27): The refiner's per-window records, self-healing when the JSON is missing     but, _refine_records(), _load_gt(), plot_clip(), Path, Stage F — diagnostics: the one-look timeline plot + confidence logs.  ``plot_cli, One JSON line per record: verdict + confidence, for review queues., Render the full-clip refinement timeline to PNG. `ground_truth` is a     list of (+19 more)

### Community 41 - "Ball Detection — In-Depth Guide"
Cohesion: 0.12
Nodes (15): 10. File map, 1. Two passes, two very different ball jobs, 2. The raw detector (`cv/ball_coarse.py`), 3. TrackNet — the real per-frame model (`cv/ball_tracknet.py`), 4. Ball source priority in Pass 2, 5. The semantic filtering layer (`cv/ball_filter.py`), 6. The trajectory selector (`cv/ball_track.py`), 7. Smoothing & 3D reconstruction (+7 more)

### Community 42 - "Codebase Context — PickleCV / CV-Training-Tool"
Cohesion: 0.12
Nodes (15): APIs (`/api` prefix), Categories / flags, Codebase Context — PickleCV / CV-Training-Tool, DB, Env / hardware (this machine), First actions for a new session, Flow, Git / working tree (this snapshot) (+7 more)

### Community 43 - "features.py"
Cohesion: 0.16
Nodes (30): build_report(), Adaptive severity selection — NOT a fixed top-N. worst_shots: shots     whose |d, Bounds select_worst_best's adaptive output to at most `limit` shots —     worst, select_tip_candidates(), select_worst_best(), _contact(), Build a ShotFeatureVector whose fields don't matter for the tests     below — on, test_best_shots_requires_in_range_and_high_confidence() (+22 more)

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

### Community 60 - "cloud_ingest.py"
Cohesion: 0.15
Nodes (20): _ingest_pass2_request(), _ingest_video(), _known_video_ids(), _loop(), EC2 worker side of the hybrid handoff: discovers videos a RUN_PROCESSING=0 UI-on, Mirrors a fine-tuned TrackNet checkpoint from S3 into this instance's     own da, Mirrors the applied refine config.override.yaml (serve_gate_min etc.,     see ap, Mirrors the self-stop toggle from S3 into this instance's own local     state (s (+12 more)

### Community 61 - "ball_filter.py"
Cohesion: 0.21
Nodes (13): Arc, ArcEvent, ArcPoint, classify_arc_events(), _near_box(), Semantic filtering layer over raw ball-detector candidates (Pass 2).  Phase 1 of, One fitted shot arc: quadratic image-space trajectory x(t), y(t) with     t = fr, Split (frame, x, y, diam) points into per-shot segments at time gaps     and sha (+5 more)

### Community 62 - "detect_contacts_for_video"
Cohesion: 0.19
Nodes (18): Hand-written weighted rules squashed through a sigmoid.      Positive evidence:, RulesClassifier, Pass1Window, One Pass 1 candidate rally window (frames, inclusive)., IncrementalRefiner, Run stages A-D over one candidate window's tracking span., Refine Pass 1 windows one at a time, as soon as Pass 2 has written the     overl, _record() (+10 more)

### Community 63 - "pipeline.py"
Cohesion: 0.15
Nodes (12): JobCancelled, JobHandle, Raised inside a job when it has been asked to stop., Passed to job functions so they can report progress and check for     cancellati, _build_do_pass2(), _build_do_pass2_range(), _build_do_pass2_selections(), Processing orchestration. Every upload loads the CURRENT global LearnedParams an (+4 more)

### Community 64 - "io_loaders.py"
Cohesion: 0.24
Nodes (10): Path, Score labeled serve / non-serve spans and report gate accuracy.     Each JSON fi, validate_serve_gate(), load_homography(), load_tracking(), project_px(), ndarray, Effective calibration: clip-local override else newest global.     (Mirrors pipe (+2 more)

### Community 65 - "overlays.py"
Cohesion: 0.19
Nodes (18): _crop_rect(), _encode_jpeg_b64(), extract_for_candidates(), _nearest_box(), Keyframe extraction for the coaching report's vision context: for each tip-candi, {contact_id: [{label, jpeg_b64}, ...]} for each tip candidate — called     ONLY, (x0, y0, x1, y1) ints, clamped to the frame: the player box expanded     by `mar, This player's box from the overlay row nearest target_frame (ties     favour ear (+10 more)

### Community 66 - "rally_analysis.py"
Cohesion: 0.13
Nodes (15): generate_coaching_report(), get_coaching_report(), get_report_stats(), get_tip_feedback(), list_contact_events(), pass1_candidates(), BaseModel, Rally Analysis & Coaching endpoints. Read-only against rally_labels, overlays, b (+7 more)

### Community 67 - "MotionBallDetector"
Cohesion: 0.21
Nodes (8): BallCandidate, MotionBallDetector, on_player_box(), ndarray, Classical fast-moving-small-blob ball detector.  Two jobs: 1. Pass 1 coarse gate, True if (x, y) sits inside any player box expanded outward by the shared     mar, Three-frame differencing keeps only objects moving between BOTH frame     pairs, Whether the blob's colour passes the appearance gate. Sampled from the         b

### Community 68 - "_scale_m_per_px"
Cohesion: 0.18
Nodes (9): ndarray, _ransac_quadratic(), RANSAC quadratic fit of one segment. Returns (cx, cy, inlier_mask)., Local metres-per-pixel at image point (x, y), via the homography     Jacobian (f, Per-window counters of what stage 1-2 rejected (harness/log output)., RejectStats, _scale_m_per_px(), physics_violations() (+1 more)

### Community 69 - "autolearn.py"
Cohesion: 0.16
Nodes (19): _client(), delete_progress(), delete_video_data(), download_tracknet_weights(), generate_presigned_url(), list_pass2_request_video_ids(), list_video_keys(), Removes the live progress snapshot once Pass 1 finishes. Without     this, a pol (+11 more)

### Community 70 - "refine/config.py"
Cohesion: 0.40
Nodes (4): Hardware-profile bundle: SPEED knobs tuned per machine, not for cross-machine re, Path, Repo-root resolution, shared by config.py / hw_profile.py / refine/*.  Computed, repo_root()

### Community 71 - "BallFrameFilter"
Cohesion: 0.28
Nodes (5): BallFrameFilter, Feed back the ball position the pipeline accepted for `frame`., Match the candidate against this player's box-relative persistence         track, Return the candidates that survive stages 1-2. `cands` items expose         .x a, Online per-frame filter: velocity gate + player-box motion logic.      Call filt

### Community 72 - "export_manual_tracking.py"
Cohesion: 0.36
Nodes (8): _ball_marks(), _clip_calibration(), _human_labels(), main(), _notes(), Export all manual (human-entered) tracking data to JSON, one bundle per clip, wi, The calibration in effect for this clip: its clip-local override if set,     els, _safe_name()

### Community 73 - "test_ball_filter.py"
Cohesion: 0.47
Nodes (5): _prune(), Regression test for the pass2.py overlay-pruning bug: when refine_track finds ze, The fixed pass2.py snippet: only prune off-arc points when at least     one arc, test_real_arc_still_prunes_off_arc_noise(), test_zero_arcs_keeps_sparse_real_detections()

### Community 77 - "ndarray"
Cohesion: 0.12
Nodes (16): _detect_appearances(), _detect_crossings(), facing_away_retreat(), ndarray, Sign flips of the ball's net side between visible samples that are at     most m, Ball (re)appearances: a REAL observation (not arc-interpolated) after     >= ser, 1.0 at/behind the baseline, ramping to 0 by m metres inside the court.     sign=, Per-native-frame dot of a player's shoulder-line normal (approximate     body-fa (+8 more)

### Community 78 - "media.py"
Cohesion: 0.15
Nodes (14): get_video(), list_videos(), BaseModel, Upload, video streaming (range requests), job status, reprocess., Per-clip metadata that isn't produced by a pass — currently just     match_type,, reprocess(), SelfStopIn, set_self_stop() (+6 more)

### Community 79 - "_finalize_vector"
Cohesion: 0.28
Nodes (16): compute_feature_vector(), _finalize_vector(), raw: one _extract_raw_metrics() result. fences: this shot's player's     ALL-sho, Single-shot convenience wrapper around _extract_raw_metrics +     _finalize_vect, _kp17(), _pose_contact(), _pose_frame(), 17 COCO keypoints with hip/knee/ankle/shoulder/elbow/wrist placed to     give _a (+8 more)

### Community 80 - "test_progress_sync.py"
Cohesion: 0.16
Nodes (14): Manual trigger (UI button) — checks S3 for this video's exported     cloud-proce, sync_from_cloud(), mirror_cloud_chain_status(), mirror_cloud_progress(), RUN_PROCESSING=0 only — called from poll-cloud-windows when it finds a     live, RUN_PROCESSING=0 only — called from sync-from-cloud once a full     result.json, merge_rally_labels(), merge_video_results() (+6 more)

### Community 81 - "test_train_eval.py"
Cohesion: 0.19
Nodes (14): iou(), match_segments(), Greedy best-IoU matching; each GT rally is matched at most once., expand_labels(), ndarray, Per-timestep 0/1 live labels for a feature timeline `t` (seconds)., Training split + eval-harness unit behaviour (spec test 7)., test_eval_result_metrics() (+6 more)

### Community 82 - "_robust_fence"
Cohesion: 0.17
Nodes (13): _player_fences_by_metric(), (q1, q3, iqr) over `values` — this player's own non-null readings for     one me, Signed deviation from this player's own typical spread for this     metric, this, {player_id: {metric_field_name: fence_or_None}} — one fence per     player per m, {(player_id, shot_type): {metric_field_name: fence_or_None}} — the     same fenc, _rel_dev(), _robust_fence(), _stratum_fences_by_metric() (+5 more)

### Community 83 - "results_exist"
Cohesion: 0.17
Nodes (12): delete_video(), _export_results(), _finish_chain(), End of the automatic pass1->pass2->contacts chain. Coaching reports     run late, cleanup_scratch(), download_results(), True if a final result.json already exists for video_id. Used by     cloud_inges, Uploads a video's exported pipeline results (see results_export.py) to     S3 as (+4 more)

### Community 84 - "ShotFeatureVector"
Cohesion: 0.27
Nodes (10): player_label(), How strongly this shot deserves a tip slot — |deviation| plus a bonus     when t, At most ONE praise candidate per player, at most MAX_PRAISE_TIPS     total — cho, Per-player whole-session aggregates for the LLM prompt, built from     EVERY sco, _salience(), _select_praise_exemplars(), _session_context(), _shot_summary() (+2 more)

### Community 85 - "_export_video"
Cohesion: 0.39
Nodes (8): _export_video(), _labeled_frames(), main(), Path, Export ball_corrections + video frames into a TrackNet fine-tuning dataset.  Bui, Resolve a real local file for this video — DB `path` may be a stale     containe, frame -> (x, y) in ORIGINAL video pixels, one mark per frame (last     write win, _video_path()

### Community 86 - "poll_cloud_windows"
Cohesion: 0.25
Nodes (8): poll_cloud_windows(), Lightweight incremental pull for live Pass 2 visibility while cloud     processi, download_progress(), list_window_objects(), progress_key(), Live single-object progress snapshot for the currently-running Pass 1     stage, [{'index': n, 'etag': ...}] for every window_N.json currently in S3     for this, upload_progress()

### Community 87 - "download_to_scratch"
Cohesion: 0.38
Nodes (7): _ensure_local_video(), S3 mode only: pass1.py/pass2.py read vrow["path"] as a real local file     (they, download_to_scratch(), Path, Returns a local path holding this video's bytes, downloading from S3     only if, scratch_dir(), scratch_path()

### Community 88 - "project_court"
Cohesion: 0.33
Nodes (6): project_court(), Court template lines projected to image px — the frontend draws these     as the, court_to_img(), _court_mask(), ndarray, Court region (+ small margin) at low-res scale — excludes neighbouring     court

### Community 89 - "start_processing"
Cohesion: 0.40
Nodes (6): Overlay the active hardware profile's p1_scale/p1_frame_step (if set)     onto a, Kick off Pass 1 immediately (background). Pass 2 no longer auto-runs     after i, Re-run with the CURRENT params/calibration (used by Regenerate and     after cal, reprocess(), start_processing(), _with_hw_overrides()

## Knowledge Gaps
- **160 isolated node(s):** `schema_version`, `clips`, `name`, `private`, `version` (+155 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `connect()` connect `connect` to `contact_detect.py`, `coaching.py`, `validate.py`, `annotate.py`, `io_loaders.py`, `loads`, `model_registry.py`, `CameraModel`, `db.py`, `analyze.py`, `media.py`, `test_serve_labels.py`, `BallWorld`, `dumps`, `bounce.py`, `pass2.py`, `dataset.py`, `refine.py`, `cloud_ingest.py`, `io_loaders.py`, `rally_analysis.py`, `media.py`, `test_progress_sync.py`, `results_exist`, `poll_cloud_windows`, `download_to_scratch`, `project_court`, `start_processing`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **Why does `run_pass2()` connect `bounce.py` to `court.py`, `contact_detect.py`, `MotionBallDetector`, `pass1.py`, `pass2.py`, `validate.py`, `BallFrameFilter`, `TrajectoryBallSelector`, `connect`, `app/config.py`, `CameraModel`, `analyze.py`, `BallWorld`, `ball_filter.py`, `pipeline.py`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Why does `RefineConfig` connect `refine_window` to `io_loaders.py`, `TrackingData`, `Features`, `refine.py`, `io_loaders.py`, `build_dataset`, `ndarray`, `RefineConfig`, `postprocess`, `test_serve_labels.py`, `detect_contacts_for_video`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `RefineConfig` (e.g. with `LiveClassifier` and `_NumpyLogistic`) actually correct?**
  _`RefineConfig` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `TrackingData` (e.g. with `Features` and `IncrementalRefiner`) actually correct?**
  _`TrackingData` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `schema_version`, `clips`, `name` to the rest of the system?**
  _160 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `contact_detect.py` be split into smaller, more focused modules?**
  _Cohesion score 0.054746835443037975 - nodes in this community are weakly interconnected._
# Graph Report - CV-Training-Tool  (2026-07-22)

## Corpus Check
- 128 files · ~148,766 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1447 nodes · 3199 edges · 77 communities (72 shown, 5 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 61 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `daf558ce`
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

## God Nodes (most connected - your core abstractions)
1. `connect()` - 113 edges
2. `RefineConfig` - 73 edges
3. `run_pass2()` - 37 edges
4. `TrackingData` - 34 edges
5. `compute_features()` - 32 edges
6. `Features` - 29 edges
7. `now()` - 27 edges
8. `dumps()` - 26 edges
9. `loads()` - 25 edges
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

## Communities (77 total, 5 thin omitted)

### Community 0 - "contact_detect.py"
Cohesion: 0.11
Nodes (32): is_serve_formation(), PlayerObs, Call once per rally at the serve frame. Re-anchors the official         left->ri, Sparse spot-check used by Pass 1 and the rally-start logic.     Serve-receive: e, _check_serve_formation(), detect_contacts_for_rally(), extract_wrist_track(), find_peaks() (+24 more)

### Community 1 - "TrackingData"
Cohesion: 0.05
Nodes (93): Hand-written weighted rules squashed through a sigmoid.      Positive evidence:, RulesClassifier, compute_features(), _detect_appearances(), detect_bounces(), _detect_crossings(), facing_away_retreat(), formation_score() (+85 more)

### Community 2 - "coaching.py"
Cohesion: 0.11
Nodes (33): _angle_deg(), build_report(), compute_feature_vector(), _elbow_angle(), generate_coaching_report(), get_report(), _hitting_side(), _knee_angle() (+25 more)

### Community 3 - "eval_video"
Cohesion: 0.12
Nodes (25): _attribute_bounce_miss(), _attribute_mark(), _box_dist(), _court_lines_img(), eval_video(), fmt_clip(), gap_stats(), in_foot_zone() (+17 more)

### Community 4 - "pass1.py"
Cohesion: 0.09
Nodes (24): on_player_box(), True if (x, y) sits inside any player box expanded outward by the shared     mar, _foot_points(), _in_court_region(), Pass 2 — precise DETECTION / VISUALIZATION inside the Pass 1 rally windows only, ``on_window(last_covered_frame)`` fires after each processing window's     overl, Reject ball blobs whose court-plane projection is well outside THIS court     —, Ankle + knee keypoints (COCO 13-16) of every tracked player — the pixel     loca (+16 more)

### Community 5 - "validate.py"
Cohesion: 0.11
Nodes (31): compare(), Rally, Human-vs-AI rally comparison: greedy IoU matching, temporal IoU, and separate st, Greedy best-IoU matching (rallies rarely overlap, so greedy ≈ optimal).     Retu, temporal_iou(), clear_override(), set_override(), _accept_gate() (+23 more)

### Community 6 - "storage_s3.py"
Cohesion: 0.06
Nodes (65): delete_video(), poll_cloud_windows(), Lightweight incremental pull for live Pass 2 visibility while cloud     processi, stream_video(), _ensure_local_video(), _export_results(), _finish_chain(), S3 mode only: pass1.py/pass2.py read vrow["path"] as a real local file     (they (+57 more)

### Community 7 - "Features"
Cohesion: 0.13
Nodes (7): _NumpyLogistic, ndarray, Path, Stage B — live/dead classification, two interchangeable backends.  Both backends, Minimal L2 logistic regression (fallback when xgboost is missing)., Learned P(live). XGBoost when available, numpy logistic otherwise.      The save, XGBoostClassifier

### Community 8 - "annotate.py"
Cohesion: 0.06
Nodes (47): add_ball_correction(), add_note(), add_window_grade(), _attach_grades(), BallCorrectionIn, BallCorrectionUpdate, BallTrackIn, CalibrationIn (+39 more)

### Community 9 - "io_loaders.py"
Cohesion: 0.08
Nodes (34): The refiner's per-window records, self-healing when the JSON is missing     but, _refine_records(), Best-effort per-window refine hook for run_pass2: refines each Pass 1     window, _window_refiner(), load_config(), Defaults <- YAML file (if given) <- keyword overrides., Pass-2 rally refinement.  Consumes (a) Pass 1 candidate windows (``rally_labels, clips_with_human_labels() (+26 more)

### Community 10 - "TrajectoryBallSelector"
Cohesion: 0.15
Nodes (13): _demo(), _on_player(), _Pt, Trajectory-consistency ball selection (Pass 2).  Motion — NOT appearance — is th, (predicted_xy, gate_radius) from the recent track, or (None, None)         if th, True once (x,y) has moved in LOCKSTEP with a nearby wrist for         lock_frame, Pick the best ball candidate for this frame (or None). `cands` items         exp, Acquire a new track only from a candidate showing ball-like MOTION         betwe (+5 more)

### Community 11 - "connect"
Cohesion: 0.10
Nodes (35): approve_change(), Approve a queued gate change: apply it immediately as a new global param     ver, reprocess(), upload_video(), clear(), connect(), init_db(), now() (+27 more)

### Community 12 - "loads"
Cohesion: 0.14
Nodes (19): list_reports(), get_calibration(), get_rallies(), project_court(), Court template lines projected to image px — the frontend draws these     as the, loads(), Any, apply_updates() (+11 more)

### Community 13 - "Workspace.jsx"
Cohesion: 0.11
Nodes (18): api, j(), App(), CalibrationPanel(), LabelingPanel(), CoachingVideo(), formatCategory(), formatTime() (+10 more)

### Community 14 - "RefineConfig"
Cohesion: 0.10
Nodes (35): LiveClassifier, make_classifier(), Backend per config; a missing/broken model artifact falls back to     rules with, Common interface: per-timestep P(live) from a Features object., EvalResult, evaluate(), evaluate_clip(), iou() (+27 more)

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
Cohesion: 0.13
Nodes (18): _build_model(), load_state(), _openvino_infer(), ndarray, Path, TrackNetV2 heatmap ball tracker — the real per-frame model for Pass 2 windows., Load a checkpoint and normalise its state dict (WASB stores     {'model_state_di, Convert the torch model to OpenVINO FP16 (once, IR cached next to the     weight (+10 more)

### Community 19 - "CameraModel"
Cohesion: 0.19
Nodes (13): fit_flight(), Flight, _project(), ndarray, Ballistic 3D reconstruction of the ball from its pixel track.  A single camera c, Index runs of consecutive tracked frames, split at bounces (a bounce ends     on, In place: overwrite ``td.ball_xy`` / ``td.ball_height`` with the ballistic     3, A fitted gravity arc for one ball flight. Evaluable at ANY frame (incl.     befo (+5 more)

### Community 20 - "db.py"
Cohesion: 0.16
Nodes (17): Manual trigger (UI button) — checks S3 for this video's exported     cloud-proce, sync_from_cloud(), mirror_cloud_chain_status(), RUN_PROCESSING=0 only — called from sync-from-cloud once a full     result.json, describe_chain_status(), export_video_results(), export_window_results(), _job_statuses() (+9 more)

### Community 21 - "analyze.py"
Cohesion: 0.10
Nodes (30): generate_llm_report(), Returns (report_markdown, clamped param_suggestions). Degrades to     (None, []), apply_suggestions(), ApplySuggestions, detect_debug(), generate_report(), get_overlays(), get_params() (+22 more)

### Community 22 - "2. PASS 2 (refine module) — keep / start-later / end-sooner"
Cohesion: 0.06
Nodes (32): 0. Two meanings of "Pass 2" — do not conflate, 1.1 Per-sample gate signal, 1.2 Start/end via hysteresis, 1.3 `close_rally` — the emitted boundaries, 1. PASS 1 — how a rally starts and ends, 2.1 Stage B — P(live) per timestep (`feature_hz = 15`), 2.1a `facing_away_retreat` — player-motion signal (2026-07-14), 2.1b `no_tracking_data` — missing pass2 overlay, not dead play (2026-07-20) (+24 more)

### Community 23 - "media.py"
Cohesion: 0.09
Nodes (17): Discrepancy feedback report via the Anthropic API.  The LLM is used ONLY here —, get_video(), BaseModel, Upload, video streaming (range requests), job status, reprocess., Per-clip metadata that isn't produced by a pass — currently just     match_type,, update_video(), VideoUpdateIn, Central configuration. Loads .env from the repo root. (+9 more)

### Community 24 - "jobs.py"
Cohesion: 0.14
Nodes (6): Job, JobHandle, JobRunner, job_id -> number of jobs ahead of it in the queue (0 = next up)., DB-backed, independent of self._q's own in-memory state. submit()         writes, Passed to job functions so they can report progress and check for     cancellati

### Community 25 - "refine_window"
Cohesion: 0.12
Nodes (32): Return a copy with the given fields replaced (used by sweeps)., RefineConfig, Features, Feature timeline + event series for one analysis span., _gate_checks(), IncrementalRefiner, Orchestrator: Pass 1 candidate windows in -> refinement records out.  Per candid, Refine Pass 1 windows one at a time, as soon as Pass 2 has written the     overl (+24 more)

### Community 26 - "test_serve_labels.py"
Cohesion: 0.12
Nodes (28): load_label_file(), Path, Rally spans from a CSV or JSON label file (seconds)., build_dataset(), build_sidecar(), expand_serve_labels(), _labels_dir(), ndarray (+20 more)

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
Cohesion: 0.24
Nodes (21): _covers_rally(), get_analyzable_rallies(), _get_rallies(), {rally_id: {"ready": bool, "reason": "pass2_not_done"|"pass2_running"|     "over, Human labels win when present (ground truth), else Pass 2, else     Pass 1 — sam, True if entries is non-empty and its frame coverage reaches within     `toleranc, Rallies (id, start_frame, end_frame, ...) that are safe to analyze     right now, readiness_report() (+13 more)

### Community 31 - "dumps"
Cohesion: 0.30
Nodes (14): chat(), ChatMessage, ChatRequest, _dispatch(), BaseModel, In-app assistant: a Claude conversation with full context of the current clip th, _state_snapshot(), _tool_add_note() (+6 more)

### Community 32 - "court.py"
Cohesion: 0.13
Nodes (19): CameraModel, court_to_img(), fit_homography(), img_to_court(), implied_person_height(), in_court(), ndarray, Court model, homography and single-view geometry.  World frame (metres): pickleb (+11 more)

### Community 33 - "bounce.py"
Cohesion: 0.22
Nodes (13): _demo(), detect_events(), _median_filter(), ndarray, Height reconstruction + bounce-vs-hit classification (the axial-view hard part)., Find trajectory events and classify bounce vs hit.      Candidate events = local, Self-check for the height-uncertainty gate + bounce plane-projection:       1. r, Per-frame world position/height. Diameter is the noisy input, so it is     media (+5 more)

### Community 34 - "package.json"
Cohesion: 0.14
Nodes (13): concurrently, description, devDependencies, concurrently, name, private, scripts, build (+5 more)

### Community 35 - "pass2.py"
Cohesion: 0.11
Nodes (23): in_excluded(), _point_in_poly(), Image-space player-detection exclusion zones.  Regions of the FIXED camera frame, Ray-casting point-in-polygon on normalized coords., True if pixel (x_px, y_px) lies inside any exclusion zone., _court_mask(), _match_fast(), ndarray (+15 more)

### Community 36 - "dataset.py"
Cohesion: 0.24
Nodes (10): Per-frame feature vector aligned with BOUNCE_MODEL_FEATURES:     [x, y, vx, vy,, trace_to_features(), _add(), build_end_dataset(), _labeled_videos(), ndarray, Training-dataset assembly from stored human labels + Pass 2 feature traces.  Tur, Return (windows, labels, meta). windows: list of (Ti, F) arrays; labels:     lis (+2 more)

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
Cohesion: 0.29
Nodes (9): _load_gt(), plot_clip(), Path, Stage F — diagnostics: the one-look timeline plot + confidence logs.  ``plot_cli, One JSON line per record: verdict + confidence, for review queues., Render the full-clip refinement timeline to PNG. `ground_truth` is a     list of, write_confidence_log(), Everything diagnostics needs to draw one window's timeline. (+1 more)

### Community 41 - "Ball Detection — In-Depth Guide"
Cohesion: 0.12
Nodes (15): 10. File map, 1. Two passes, two very different ball jobs, 2. The raw detector (`cv/ball_coarse.py`), 3. TrackNet — the real per-frame model (`cv/ball_tracknet.py`), 4. Ball source priority in Pass 2, 5. The semantic filtering layer (`cv/ball_filter.py`), 6. The trajectory selector (`cv/ball_track.py`), 7. Smoothing & 3D reconstruction (+7 more)

### Community 42 - "Codebase Context — PickleCV / CV-Training-Tool"
Cohesion: 0.12
Nodes (15): APIs (`/api` prefix), Categories / flags, Codebase Context — PickleCV / CV-Training-Tool, DB, Env / hardware (this machine), First actions for a new session, Flow, Git / working tree (this snapshot) (+7 more)

### Community 43 - "features.py"
Cohesion: 0.20
Nodes (20): _classify(), _neutral(), _norm_dev(), Signed deviation from [lo, hi], expressed in units of range-width so     metrics, category = whichever of the four categories has the single worst     (largest-ma, Adaptive severity selection — NOT a fixed top-N. worst_shots: shots     whose |d, select_worst_best(), _contact() (+12 more)

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
Cohesion: 0.17
Nodes (17): _ingest_video(), _known_video_ids(), _loop(), EC2 worker side of the hybrid handoff: discovers videos a RUN_PROCESSING=0 UI-on, Mirrors a fine-tuned TrackNet checkpoint from S3 into this instance's     own da, Mirrors the applied refine config.override.yaml (serve_gate_min etc.,     see ap, Adds video_id to this instance's own DB WITHOUT downloading it or     starting p, Downloads the video (needed locally for Pass 1/2 regardless — see     storage_s3 (+9 more)

### Community 61 - "ball_filter.py"
Cohesion: 0.21
Nodes (13): Arc, ArcEvent, ArcPoint, classify_arc_events(), _near_box(), Semantic filtering layer over raw ball-detector candidates (Pass 2).  Phase 1 of, One fitted shot arc: quadratic image-space trajectory x(t), y(t) with     t = fr, Split (frame, x, y, diam) points into per-shot segments at time gaps     and sha (+5 more)

### Community 62 - "detect_contacts_for_video"
Cohesion: 0.18
Nodes (13): detect_contacts(), Runs pose-only contact detection for every rally of `source` on this     video a, _detect_contacts_after_pass2(), Contact detection over the fresh Pass 2 overlays. Readiness-gated (see     rally, detect_contacts_for_video(), Only rallies readiness.get_analyzable_rallies() clears (Pass 2 done,     and its, Replaces this video's contact_events (own table — re-running detection     is id, save_events() (+5 more)

### Community 63 - "pipeline.py"
Cohesion: 0.18
Nodes (14): active_profile(), Any, Resolved profile dict, or None if HW_PROFILE is unset (current .env     behavior, _build_do_pass2(), Processing orchestration. Every upload loads the CURRENT global LearnedParams an, Overlay the active hardware profile's p1_scale/p1_frame_step (if set)     onto a, Rally refinement over the fresh Pass 2 tracking data (pass2/refine     module)., Shared by start_processing (chained after Pass 1) and reprocess     (standalone (+6 more)

### Community 64 - "io_loaders.py"
Cohesion: 0.19
Nodes (13): Path, Score labeled serve / non-serve spans and report gate accuracy.     Each JSON fi, validate_serve_gate(), load_homography(), load_tracking(), project_px(), ndarray, Loaders adapting the repo's real data formats to refinement inputs.  Formats (ve (+5 more)

### Community 65 - "overlays.py"
Cohesion: 0.23
Nodes (10): chunk_index(), Overlay metadata persistence. All annotation the frontend draws is stored as JSO, Entries must each carry an 'f' frame field. Merges with any chunk     already st, read_range(), write_entries(), _latest_pass2_job_status(), _pass2_ready_reason(), Pass 2 readiness gating (patch): nothing else in this package checked that Pass (+2 more)

### Community 66 - "rally_analysis.py"
Cohesion: 0.22
Nodes (9): generate_coaching_report(), get_coaching_report(), list_contact_events(), Rally Analysis & Coaching endpoints. Read-only against rally_labels, overlays, b, Per-rally Pass 2 readiness, so the frontend can show a 'waiting on     Pass 2' s, Triggers coaching report generation (one Anthropic API call) and     persists it, Returns the cached report, or null if none has been generated yet —     regenera, readiness() (+1 more)

### Community 67 - "MotionBallDetector"
Cohesion: 0.24
Nodes (6): BallCandidate, MotionBallDetector, ndarray, Classical fast-moving-small-blob ball detector.  Two jobs: 1. Pass 1 coarse gate, Three-frame differencing keeps only objects moving between BOTH frame     pairs, Whether the blob's colour passes the appearance gate. Sampled from the         b

### Community 68 - "_scale_m_per_px"
Cohesion: 0.18
Nodes (9): ndarray, _ransac_quadratic(), RANSAC quadratic fit of one segment. Returns (cx, cy, inlier_mask)., Local metres-per-pixel at image point (x, y), via the homography     Jacobian (f, Per-window counters of what stage 1-2 rejected (harness/log output)., RejectStats, _scale_m_per_px(), physics_violations() (+1 more)

### Community 69 - "autolearn.py"
Cohesion: 0.31
Nodes (8): new_id(), labels_hash(), _load_human_ai(), _queue_pending(), The autonomous learning step, run in the background whenever the user saves corr, Compute and safely apply learning from the current human labels., record_eval(), run_learn()

### Community 70 - "refine/config.py"
Cohesion: 0.24
Nodes (7): Hardware-profile bundle: SPEED knobs tuned per machine, not for cross-machine re, Path, Repo-root resolution, shared by config.py / hw_profile.py / refine/*.  Computed, repo_root(), Path, Single config for the whole refinement pipeline.  Every window, threshold and we, save_config()

### Community 71 - "BallFrameFilter"
Cohesion: 0.28
Nodes (5): BallFrameFilter, Feed back the ball position the pipeline accepted for `frame`., Match the candidate against this player's box-relative persistence         track, Return the candidates that survive stages 1-2. `cands` items expose         .x a, Online per-frame filter: velocity gate + player-box motion logic.      Call filt

### Community 72 - "export_manual_tracking.py"
Cohesion: 0.36
Nodes (8): _ball_marks(), _clip_calibration(), _human_labels(), main(), _notes(), Export all manual (human-entered) tracking data to JSON, one bundle per clip, wi, The calibration in effect for this clip: its clip-local override if set,     els, _safe_name()

### Community 73 - "test_ball_filter.py"
Cohesion: 0.47
Nodes (5): _prune(), Regression test for the pass2.py overlay-pruning bug: when refine_track finds ze, The fixed pass2.py snippet: only prune off-arc points when at least     one arc, test_real_arc_still_prunes_off_arc_noise(), test_zero_arcs_keeps_sparse_real_detections()

## Knowledge Gaps
- **159 isolated node(s):** `schema_version`, `clips`, `name`, `private`, `version` (+154 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `connect()` connect `connect` to `coaching.py`, `pass1.py`, `validate.py`, `storage_s3.py`, `annotate.py`, `io_loaders.py`, `loads`, `RefineConfig`, `model_registry.py`, `db.py`, `analyze.py`, `media.py`, `params.py`, `dumps`, `pass2.py`, `dataset.py`, `cloud_ingest.py`, `detect_contacts_for_video`, `pipeline.py`, `io_loaders.py`, `overlays.py`, `rally_analysis.py`, `autolearn.py`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `RefineConfig` connect `refine_window` to `io_loaders.py`, `TrackingData`, `refine/config.py`, `Features`, `refine.py`, `io_loaders.py`, `build_dataset`, `RefineConfig`, `postprocess`, `test_serve_labels.py`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `run_pass2()` connect `pass1.py` to `court.py`, `bounce.py`, `overlays.py`, `MotionBallDetector`, `pass2.py`, `contact_detect.py`, `validate.py`, `BallFrameFilter`, `TrajectoryBallSelector`, `connect`, `loads`, `app/config.py`, `jobs.py`, `BallWorld`, `ball_filter.py`, `pipeline.py`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `RefineConfig` (e.g. with `LiveClassifier` and `_NumpyLogistic`) actually correct?**
  _`RefineConfig` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `TrackingData` (e.g. with `Features` and `IncrementalRefiner`) actually correct?**
  _`TrackingData` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `schema_version`, `clips`, `name` to the rest of the system?**
  _159 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `contact_detect.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11092436974789915 - nodes in this community are weakly interconnected._
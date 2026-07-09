# Audit: Does manual labeling actually improve the models?

**Date:** 2026-07-09
**Author:** Claude (code-verified audit, read-only)
**Purpose:** Hand this to another agent to (a) independently verify each claim against the code, (b) confirm or refute the gaps, and (c) produce a fix plan. Every claim below has a `file:line` anchor — verify by reading, don't trust.

---

## TL;DR verdict

The user does two kinds of manual work in the labeling/viewer UI. They pay off very differently:

- **Manual ball marks** (B=hit, H=hit, N=position, bounce + in/out): **essentially wasted for model improvement.** Nothing trains on them; no detector/filter parameters are tuned from them. They only (1) produce offline eval metrics and (2) patch a single clip's pixel position.
- **Manual rally labels** (serve = rally start, unplayable = rally end): **partially pay off.** One automatic loop genuinely works (boundary-offset params). One model (XGBoost refiner) benefits only when manually retrained. The flagship neural model (`rally_end` GRU) is trained + registered + promotable but **never loaded by the pipeline** — so training it has zero runtime effect today. The serve model doesn't exist yet.

The single highest-leverage fix is wiring the trained `rally_end` model into Pass 2 (it's currently hard-coded to `model=None`).

---

## Part 1 — Manual ball marks

### 1.1 Capture
- Marking UI lives in `frontend/src/Workspace.jsx`, not `LabelingPanel.jsx` (that panel is rally help text only).
  - `gtMode` toggle state: `Workspace.jsx:29`; button "Mark ground truth" `:233`.
  - Canvas click captures `{frame,x,y}` as `pending`, calls `api.projectPoint` for live in/out suggestion: `onCanvasClick` `Workspace.jsx:116-146` (projectPoint `:137`).
  - Keyboard state machine `Workspace.jsx:150-194`:
    - **B** → bounce, then in/out step; **Enter** saves, **X** flips in/out (`:167-177`)
    - **H** → `saveMark('hit')` (`:170`)
    - **N** → `saveMark('position')` (`:171`)
    - Edit existing mark: B/H/N retype, X flip, D/Delete delete (`:178-189`)
  - `saveMark` POSTs `{frame,x,y,event_type,in_out}` via `api.addBallCorrection`: `Workspace.jsx:92-104`; API client `frontend/src/api.js:55-64`.
- Backend API `backend/app/api/annotate.py`:
  - `POST /videos/{id}/ball_corrections` `:309-335` — validates `event_type ∈ {bounce,hit,position}`, bounces require `in_out`; computes court projection + in/out suggestion + `disagree` flag via `_project_point`/`_gt_fields` `:247-275`. Marks hardcoded `scope="clip_local"` `:284`.
  - `GET .../project_point` `:287-293` (live suggestion); `PUT` `:346-379`; `DELETE` `:382-386`.
  - Separate `POST /videos/{id}/ball_track` `:399-417` accepts an EXTERNAL dense tracker into a different `ball_tracks` table — not manual marks.

### 1.2 Storage
- SQLite table `ball_corrections` in `data/app.db`. Base schema `backend/app/db.py:105-112`; GT columns added idempotently `:167-179` (`event_type, court_x, court_y, in_out, in_out_suggested, disagree`). Comment at `annotate.py:236-242` says it "doubles as the GROUND-TRUTH mark store."
- Git-shareable export/import: `backend/scripts/ball_gt_io.py` → `reference_data/ball_gt.json` (`:21`, `:24-47`).

### 1.3 Consumers — the whole list (repo-wide grep confirmed only 3)
1. **Pass 2 override** `backend/app/cv/pass2.py:139-145` — reads only `frame,x,y`; substitutes pixel position for that frame in that one clip. **Ignores `event_type`, `in_out`, `court_*`.** A manual patch, not learning.
2. **Eval harness** `backend/scripts/eval_ball_filter.py` — `load_marks` `:150-171`, `pick_videos` `:64-99`; metrics (pos err, bounce P/R, in/out acc) `:595-629`; `--diagnose` failure attribution. **Read-only report. Never writes params or models.**
3. **Export tool** `ball_gt_io.py`.

### 1.4 The gap (ball)
- **No training consumes ball GT.** grep of `backend/app/learning/` for `ball_corrections`/`ball_*` → no matches in autolearn/updates/validate/dataset (only `params.py`, which merely *defines* ball defaults).
- TrackNet ball model is a **frozen external checkpoint** `data/models/wasb_tracknetv2_tennis.pth.tar` (`eval_ball_filter.py:827-829`), never fine-tuned in-repo.
- Ball filter/detector are pure classes with no DB/GT/learning access: `ball_filter.py` `BallFrameFilter.__init__:80`; `ball_track.py` `TrajectoryBallSelector.__init__:57`.
- Their params come from `learning/params.py:44-101` (`ball_*` keys), injected by `pass2.py:166-172` and `eval_ball_filter.py:416-466`.
- **Ball params are tuned BY HAND:** `backend/scripts/eval_matrix.py:35-53` runs configs via `--set KEY=VAL`; human reads the table and hand-edits `DEFAULTS` in `params.py`. The only programmatic param writer `apply_updates()` `params.py:206-222` is called by autolearn (boundary offsets) and train_gate (`p1_motion_ratio`) — **neither passes any `ball_*` key.** Comment `params.py:73-79` documents arc values were set by hand.

**Net:** ball GT drives measurement + one-clip position-patching only. No self-improvement loop exists (unlike rally boundaries).

---

## Part 2 — Manual rally labels

### 2.1 Capture
- `frontend/src/LabelingPanel.jsx` — strict alternating Start/End machine (`clickStart`/`clickEnd` `:25-34`):
  - START = the serve (players set, first ball motion) `:10, 61`
  - END = frame ball becomes unplayable (out / twice / dies at net — NOT when it stops rolling or players reset) `:11-13, 61-63`
  - `adjust(i, field)` `:36-42`, `remove` `:43`, save via `api.putHumanLabels` `:49`.
- `frontend/src/Workspace.jsx` — hosts panel in Setup tab `:319-327`; `rallies` state `{human,pass1,pass2}` `:16`; `refined` advisory boundaries `:18`, rendered `:335-337`.
- Backend `annotate.py`:
  - `PUT /videos/{id}/labels/human` (`put_human_labels` `:67-95`) — validates end>start, **deletes all `source='human'` rows and re-inserts** the full set. Human labels are always a full replace.

### 2.2 Storage
- SQLite `rally_labels` (`id, video_id, source, start_frame, end_frame, meta`); sources `human|pass1|pass2` (`annotate.py:33-39, 77-81`). Stored in frames; fps converts to seconds (`labels.py:24-33`).
- Pass 2 refined boundaries → JSON file (not DB): `resolve_path(RefineConfig().out_dir)/<video_id>.json`, read by `get_refined` (`annotate.py:42-64`). "Advisory until the refiner is validated."
- `backend/app/refine/`:
  - `labels.py` — `load_ground_truth(video_id)` reads human rally spans `:24-33`; `expand_labels` → per-timestep live vector `:60-66`.
  - `features.py` `compute_features` — feature timeline for train + inference.
  - `classifiers.py` — `LiveClassifier`; `RulesClassifier` (no training) + `XGBoostClassifier` (native, else `_NumpyLogistic` fallback `:57-161`); `make_classifier` loads `cfg.model_path`, **falls back to rules on any failure** `:164-173`. Artifact `data/models/refine/xgb.json` + `xgb.meta.json`.
  - `serve_gate.py` — `score_serve` Stage-D heuristic. **Pure heuristic, no learned weights.**
  - `serve_labels.py` / `build_serve_dataset.py` — see 2.5.
  - `train.py` — see 2.3A.

### 2.3 Training flows (both are manual CLI)

**A. Refine live/dead classifier — `refine/train.py` (XGBoost)**
- CLI `python -m backend.app.refine.train` `:118-126`.
- `build_clip_dataset` `:36-61`: per clip with human labels, per Pass 1 window, tracking overlays → `compute_features` → X = motion/tracking features, **y = `expand_labels(feats.t, load_ground_truth(...))`** (human spans rasterized to live/dead).
- Split by clip `:74-85`; holdout frozen. Saves `data/models/refine/xgb.json` `:113`.
- **Is loaded** by `make_classifier` when `cfg.classifier=="xgboost"`. Does NOT use overlay bounce/hit as labels.

**B. `rally_end` bounce/hit GRU — `scripts/train_bounce.py`**
- CLI `python backend/scripts/train_bounce.py [--epochs] [--promote]`.
- `learning/dataset.py:build_end_dataset:51-87`: per human rally —
  - END window: frames within ±2 of human rally end → class 1 (real weak label) `:67-73`
  - MID window: frames the heuristic already flagged `kind=="hit"` from Pass 2 overlays → class 2 (pseudo) `:76-82`
  - Features via `trace_to_features` over Pass 2 overlay traces (`overlays.read_range(vid,"pass2",...)`, `bounce.py:206-231`) + augmentation.
- Trains on human rally-END frames + heuristic hit pseudo-labels. **Does NOT use `ball_corrections`.**
- Model `BounceGRU` `bounce.py:273-290`; registered SHADOW `train_bounce.py:68-72`; promote needs `--promote` AND `≥ MIN_PROMOTE_CLIPS=3` `:29, 74-91`.

### 2.4 Feedback loops (does more labeling ⇒ better model?)

**(a) Automatic on every save — tunes PARAMS only.** `put_human_labels` queues background `learn` (`annotate.py:83-95`) → `learning/autolearn.run_learn:78-126`:
- Compares human vs Pass 2 (`analysis.metrics.compare`), computes start/end offset proposals (`updates.compute_param_updates`).
- LOW-risk offsets validated across all clips, committed globally only if aggregate accuracy improves (`validate.evaluate_offset_change:109-117`) → new `learned_params` version.
- MEDIUM-risk gate changes queued in `pending_changes` for human approval `:119, 59-75`.
- **This is the real "gets better as you label" loop — but only numeric boundary offsets, not ML weights.**

**(b) Model registry / promotion — `learning/model_registry.py`.** Append-only, kinds `("rally_end","serve_start","tracknet")` `:16`. `register`→shadow, `promote`→live, `reject`, `latest_live`, `load_live`/`load_live_bundle` `:95-123`. Promotion triggered manually via `train_bounce.py --promote` or `POST /models/{version}/promote` (`api/models.py:36-40`). Learning-curve endpoint `api/models.py:17-33`.

**(c) CRITICAL DISCONNECT — `rally_end` model is never used by the pipeline.**
- `load_live`/`load_live_bundle` are **defined but never called** anywhere in `backend/app` (grep: only the definitions match).
- Pass 2 hard-codes `model=None`: `pass2.py:420-422` — comment "model=None: no trained model drives anything in this pass."
- `rally_logic.py:344, 363` — `start_model=None` default, no caller passes a live model.
- `_events_from_model` path `bounce.py:234-270` is only reachable if a model is passed in — pipeline never does.
- `validate.evaluate_model:239-257` sets an override then calls `full_rerun_metrics`, but the rerun still goes through `model=None` → **validation scores model == heuristic (inert).**

### 2.5 Serve detection — `serve_labels.py`
- GT: every human rally-START = a serve-contact timestamp `:1-27, 48`; source = human rally labels via `load_ground_truth` `:158`. No separate serve labeling.
- `build_dataset:149-183`: `build_sidecar` serve times `:46-57`; `expand_serve_labels` per-timestep target with ±`serve_buffer_s` masked `:60-85`; `to_gate_spans` span-level serve/non-serve `:88-111`. Negatives implicit; times in seconds.
- API `GET/POST /videos/{id}/serve_dataset` (`annotate.py:98-119`); frontend `ServeDatasetPanel` `Workspace.jsx:347-380`; CLI `scripts/build_serve_dataset.py`.
- **Nothing trains on serve labels.** Registry reserves `serve_start` but `_build` only knows `rally_end` — "serve_start / tracknet builders are added when those models ship" (`model_registry.py:88-89`). No serve trainer exists; `serve_gate.py` is heuristic. Serve labels are consumed only by `evaluate.validate_serve_gate` (heuristic validator).

---

## Part 3 — Summary table

| Mechanism | Trigger | Inputs | Improves over time? |
|---|---|---|---|
| Boundary-offset params (`autolearn`) | **Auto** on every label save | human vs Pass 2 spans | ✅ Yes — auto-validated & committed if aggregate accuracy improves (`autolearn.py:109-117`) |
| Refine XGBoost live/dead (`refine/train.py`) | Manual CLI | human spans + tracking features | ✅ When re-run; **is** loaded by refiner |
| `rally_end` GRU (`train_bounce.py`) | Manual CLI + `--promote` | human rally-END + heuristic hit pseudo-labels | ❌ **Registry/promote exist but NOT loaded by Pass 2** (`pass2.py:421-422` `model=None`; `load_live` never called) → no runtime effect |
| `serve_start` detector | — | serve dataset from rally-starts | ❌ No model exists; dataset only |
| Ball detector/filter (TrackNet + `ball_*` params) | Manual hand-edit of `params.py` | — (marks not used) | ❌ Ball GT used only for eval + one-clip position patch |

---

## Part 4 — Proposed fixes (for the next agent to verify, refine, and plan)

**P0 — Fix the dead wire (highest leverage, contained).** Make Pass 2 load `model_registry.load_live("rally_end")` and pass it into `detect_events` instead of `model=None` (`pass2.py:420-422`). Also fix `validate.evaluate_model` (`validate.py:239-257`) so validation actually routes the model through the rerun (currently inert). Recommend a flag to A/B heuristic vs. model on held-out clips. *Until this lands, all `train_bounce.py --promote` work is inert and its validation is meaningless.*

**P1 — Let ball marks tune something automatically.**
- Cheap: auto-tune `ball_*` params from the eval harness (grid/coordinate search against GT) instead of hand-editing `params.py`. Metric fn + `--set` override plumbing already exist (`eval_ball_filter.py`, `eval_matrix.py`).
- Real: use bounce/in-out marks to calibrate/fine-tune the TrackNet post-filter (arc/bounce classifier) — where in/out accuracy lives.

**P2 — Stop discarding semantics in Pass 2's ball override.** `pass2.py:141` reads only `x,y` and ignores `event_type`/`in_out`. Feed bounce/hit marks as high-confidence event anchors, not just position patches.

**P3 — Trust-but-verify the autolearn loop.** It's the one thing improving automatically, but only moves offsets and auto-commits on aggregate improvement. Confirm the `param_eval` scoreboard is reviewed — a subset regression can hide behind an aggregate gain.

**P4 — Serve model, or stop labeling for it.** Either build the `serve_start` trainer (dataset + gate heuristic are ready) or acknowledge the serve dataset is speculative so labeling effort isn't spent on a non-existent consumer.

---

## Part 5 — Verification checklist for the reviewing agent

1. Confirm `load_live`/`load_live_bundle` have **no callers** in `backend/app` (grep). If false, P0 changes.
2. Confirm `pass2.py:420-422` passes `model=None` and no other Pass 2 path loads a live `rally_end` model.
3. Confirm `apply_updates` callers never pass a `ball_*` key (autolearn, train_gate).
4. Confirm no training reads `ball_corrections` (grep `backend/app/learning/`, `backend/app/refine/`, all `scripts/`).
5. Confirm `serve_start` has no trainer and `_build` only handles `rally_end`.
6. Confirm the refine XGBoost path IS actually reachable (is `cfg.classifier=="xgboost"` the live config, or does it fall back to rules?). **Open question — verify which classifier the refiner runs in production.**
7. Confirm whether `refined` boundaries are shown-only or fed back anywhere downstream.

> Note: line numbers are from 2026-07-09 working tree (branch `rohan-dev`, with uncommitted changes to annotate.py, pass1.py, pass2.py, pipeline.py, Workspace.jsx, api.js and untracked refine/, build_serve_dataset.py, LabelingPanel.jsx). Re-anchor if the tree has moved.

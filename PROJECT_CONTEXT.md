# PickleCV / CV-Training-Tool — Full Project Context

**Snapshot date:** 2026-07-10  
**Purpose:** Single handoff document for the codebase as it stands. Prefer this plus the linked specs over the older README where they disagree.

**Repo root:** `c:\Users\balaj\pickleCV\CV-Training-Tool` (git lives here; workspace parent `pickleCV` only holds this app plus editor tooling).  
**Package name:** `pickleball-rally-system` v0.1.0

---

## 1. What this product is

A **local** video-analysis app for **fixed-camera pickleball** footage (down-the-line / axial view). It:

1. Detects rally windows (serve → ball unplayable) via a two-pass CV pipeline  
2. Lets you calibrate the court, label ground truth, and mark ball events  
3. Compares AI vs human, explains discrepancies via Anthropic (optional)  
4. Saves lessons as **global** versioned parameters so every future clip starts smarter  

**Hard constraints**

- Single fixed camera (not multi-angle / broadcast)  
- Offline, asynchronous processing (single-worker job queue)  
- Overlays drawn from **metadata** over the **original video** — never cut, copy, or re-encode  
- LLM used only for reports / in-app assistant — **never** in the per-frame vision loop  

---

## 2. Current product focus (important — overrides older README)

As of ~2026-07-07 (`PASS1_PASS2_SPEC.md`):

> **Detection accuracy, not rally segmentation.**  
> **Pass 1 is the SOLE source of rally start/end.**  
> **Pass 2 no longer computes rally boundaries** — it only produces accurate per-frame player/ball overlays inside Pass 1 windows.

Implications:

- `rally_logic.py`, terminal-event end detection, learned `start_offset_frames` / `end_offset_frames`, serve-vs-pass rejection, and live `rally_end` / `serve_start` model hooks remain in the tree but are **not called** from Pass 2  
- Bounce/hit markers still run as **overlay annotations only**  
- Auto-learn boundary-offset loop is effectively **paused** (offsets applied nowhere)  
- Pass 1 windows are **recall-generous** (pad + merge); loose edges / occasional under-segmentation are accepted trade-offs  

Canonical detection truth: [`PASS1_PASS2_SPEC.md`](PASS1_PASS2_SPEC.md)

---

## 3. Tech stack

| Layer | Stack |
|-------|--------|
| Frontend | React 18 + Vite 5 (`frontend/`), port **5173**, proxies `/api` → backend |
| Backend | Python **3.12**, FastAPI + uvicorn, port **8420** |
| CV / ML | OpenCV, PyTorch (XPU / CUDA / CPU), Ultralytics **YOLO11-pose + ByteTrack**, TrackNetV2 (optional), optional OpenVINO (Intel), BounceGRU (trainable) |
| DB | SQLite WAL (`data/app.db`) |
| LLM | Anthropic API (reports + chat assistant) |
| Orchestration | Root `npm` + `concurrently`; backend venv at `backend/.venv` |

---

## 4. Directory map

```
pickleCV/
├── .claude/ .vscode/          # editor tooling only
├── package-lock.json          # empty root lockfile
└── CV-Training-Tool/          # ← THE APP (git repo)
    ├── README.md              # setup + architecture (partially stale vs Pass1/2 spec)
    ├── PROJECT_CONTEXT.md     # this file
    ├── PASS1_PASS2_SPEC.md    # current detection truth
    ├── LEARNING_TRACKED_VALUES.md
    ├── DETECTION_CONTEXT_FOR_FABLE.md
    ├── DETECTION_AUDIT.md
    ├── BALL_TRACKER_INTEGRATION.md
    ├── BALL_AUDIT_HANDOFF.md  # 2026-07-09 ball-audit session state
    ├── package.json           # npm run dev / setup:backend*
    ├── .env / .env.example
    ├── yolo11n-pose.pt        # default pose weights at repo root
    ├── backend/
    │   ├── requirements.txt       # Intel Arc XPU torch index
    │   ├── requirements-cuda.txt  # NVIDIA (torch installed separately)
    │   ├── app/
    │   │   ├── main.py            # FastAPI entry
    │   │   ├── config.py          # env + paths
    │   │   ├── db.py              # schema + helpers
    │   │   ├── pipeline.py        # Pass1→Pass2 orchestration
    │   │   ├── jobs.py            # single-worker async queue
    │   │   ├── api/               # HTTP routers
    │   │   ├── cv/                # vision pipeline
    │   │   ├── learning/          # params, autolearn, models
    │   │   └── analysis/          # metrics + Anthropic reporter
    │   └── scripts/               # train/eval/import utilities
    ├── frontend/
    │   └── src/                   # App, Workspace, VideoPlayer, …
    ├── data/                      # gitignored: app.db, videos/, models/
    └── reference_data/
        └── manual_tracking.json   # portable GT export (schema_version 1)
```

---

## 5. How to run

```powershell
cd CV-Training-Tool
copy .env.example .env          # set ANTHROPIC_API_KEY, CV_DEVICE, etc.
npm install
npm run setup:backend:cuda      # NVIDIA — or setup:backend for Intel XPU
npm run dev                     # backend :8420 + frontend :5173
# open http://localhost:5173
```

Separate: `npm run dev:backend` / `npm run dev:frontend`.  
Production-ish: `npm run build` then backend serves `frontend/dist` at `:8420`.

### Env vars (`.env.example`)

| Variable | Role |
|----------|------|
| `ANTHROPIC_API_KEY` | Reports + assistant; rest works without it |
| `ANTHROPIC_MODEL` | Chat/assistant model (e.g. `claude-opus-4-8`) |
| `ANTHROPIC_REPORT_MODEL` | Cheaper report model (default in `config.py`) |
| `CV_DEVICE` | `auto` \| `xpu` \| `cuda` \| `cpu` |
| `TRACKNET_WEIGHTS` | Explicit weights path (wins if set) |
| `TRACKNET_ENABLED` | Opt-in TrackNet (`0` default — tennis weights transfer poorly) |
| `PASS2_BALL` | `1` default; `0` = player-only Pass 2 |
| `POSE_MODEL` | Default `yolo11n-pose.pt` |
| `DATA_DIR` | Storage root (default `./data`) |

Default TrackNet path when enabled: `data/models/tracknet_v2.pt`  
Download helper: `backend/scripts/get_tracknet_weights.py --sport tennis`

### GPU install notes

| Hardware | Install | `.env` |
|----------|---------|--------|
| NVIDIA | `setup:backend:cuda` / `requirements-cuda.txt` + cu126 torch | `CV_DEVICE=cuda` |
| Intel Arc | `setup:backend` / `requirements.txt` (XPU index) | `CV_DEVICE=auto` or `xpu` |
| CPU | either + CPU torch | `CV_DEVICE=cpu` |

Do **not** mix XPU and CUDA requirement files — wrong torch build.

---

## 6. Pipeline architecture

```
Upload
  └─► jobs.pass1 → run_pass1 (100% footage, cheap, low-res / subsampled)
                    → rally_labels source=pass1  ← AUTHORITATIVE boundaries
                    → pass1 overlay traces
         └─ chains ►
       jobs.pass2 → run_pass2 (only inside Pass 1 windows ≈20% of frames)
                    → players (YOLO11-pose + ByteTrack)
                    → ball (priority chain below)
                    → Kalman / height / bounce-hit markers (overlay only)
                    → pass2 overlay
                    → NO rally_labels
```

Orchestration: `backend/app/pipeline.py` loads latest `LearnedParams` + effective calibration (clip-local override else newest global), then Pass 1 → Pass 2.  
Jobs: `backend/app/jobs.py` — **single worker**, cancel / supersede, progress for UI polling.

### Pass 1 — coarse gate (`cv/pass1.py`)

- Goal: “is a rally roughly happening?” over full clip  
- Scale `p1_scale` (0.4), step `p1_frame_step` (3)  
- Signals: **net crossings** (primary, needs calibration) → adaptive **motion energy** fallback → sparse serve-formation spot-checks (logged only)  
- Court ROI mask so spectators / neighbour courts don’t drive the gate  
- Hysteresis (`p1_on_s` / `p1_off_s`); pad (`p1_pad_s`) + merge (`p1_merge_gap_s`); drop blips `< p1_min_rally_s`  
- Traces feed future gate classifier (`scripts/train_gate.py`)

### Pass 2 — detection only (`cv/pass2.py`)

- Full-rate inside Pass 1 windows only  
- **Ball observation priority:**
  1. Manual `ball_corrections` (confidence 1.0)  
  2. External track via `POST …/ball_track` (`ball_tracks` table) — see `BALL_TRACKER_INTEGRATION.md`  
  3. **TrackNet** if `TRACKNET_ENABLED` + weights (`ball_tracknet.py`)  
  4. Classical `MotionBallDetector` + `TrajectoryBallSelector` (`ball_coarse.py`, `ball_track.py`)  
- Optional post-filter: `ball_filter.py` (velocity / foot / arc-RANSAC; `ball_filter_enabled` default **off**)  
- Kalman smooth + short gap fill; size→depth→height reconstruction; bounce/hit markers for display  
- Court: 6.096 × 13.4112 m; ball Ø 0.074 m (`court.py`)

### Honest capability status

| Area | Status |
|------|--------|
| Upload, jobs, overlays, calibration, labeling APIs | Works |
| Player pose / track / numbering | Works (YOLO weights auto-download) |
| Pass 1 coarse gate | Works |
| Classical ball | Works but weak on far-baseline (few px) |
| TrackNet | Integrated; needs weights + **fine-tune** for this camera; off by default |
| BounceGRU / rally_end | Trainable via `train_bounce.py`; not driving boundaries currently |
| Audio hit onsets + fusion | Works (overlay confidence only); mono audio, cross-court bleed rejected by visual corroboration (`AUDIO_POSE_FUSION_SPEC.md`) |
| Pose-aware limb-lock ball exclusion | Works (drops ball candidates moving in lockstep with a wrist) |
| Serve marker (AI-inferred `evt.kind='serve'`) | Works (overlay + timeline `S`); NOT a rally-boundary signal |
| Faults with no ball signature | Out of scope v1 |

Biggest physics noise source: **apparent size → depth** for the far-court ball (`DETECTION_AUDIT.md`), partly offset by depth-independent audio onset fusion (`AUDIO_POSE_FUSION_SPEC.md`).

---

## 7. Backend modules

### Entry & core

| File | Role |
|------|------|
| `app/main.py` | FastAPI, CORS, `/api` routers, optional `frontend/dist`, startup init |
| `app/config.py` | Env + paths |
| `app/db.py` | Schema + helpers |
| `app/pipeline.py` | Pass1→Pass2 |
| `app/jobs.py` | Single-worker queue |

### API (`/api`)

| Module | Responsibility |
|--------|----------------|
| `api/media.py` | Upload/list/get videos, stream, reprocess, cancel jobs |
| `api/annotate.py` | Human rally labels, calibration, ball corrections/GT, external `ball_track`, notes |
| `api/analyze.py` | Overlays, detect_debug, reports, pending_changes, scoreboard, params |
| `api/chat.py` | In-app Claude assistant (bounded tools) |
| `api/models.py` | Model registry list/promote/reject, learning curve |
| `main.py` | `GET /api/health` |

**Assistant tools (cannot edit CV source):** `get_feature_trace`, `list_rallies`, `compare_to_human`, `apply_params` (clamped), `reprocess`, `add_note`

### CV (`app/cv/`)

| File | Role |
|------|------|
| `pass1.py` / `pass2.py` | Two-pass orchestration |
| `ball_tracknet.py` | TrackNetV2 U-Net (WASB-compatible); optional OpenVINO |
| `ball_coarse.py` | Three-frame differencing + blob geometry |
| `ball_track.py` | Trajectory-consistent blob selection + foot suppression + limb-lock guard |
| `ball_filter.py` | Optional velocity/foot/arc-RANSAC layer |
| `audio_events.py` | ffmpeg→bandpass→librosa onset extraction (`audio_events` table); `AUDIO_POSE_FUSION_SPEC.md` |
| `players.py` | PoseTracker, serve formation, OpenVINO export cache |
| `court.py` | Homography, `CameraModel`, court geometry |
| `bounce.py` | Size→depth height, `detect_events`, `fuse_audio_events`, BounceGRU |
| `kalman.py` | Constant-acceleration smooth + gap bridge |
| `rally_logic.py` | Start/end/terminal events — **present, unused by Pass 2** |
| `identity.py` | Appearance + court re-ID |
| `exclusions.py` | Image-space dead zones |
| `overlays.py` | Chunked JSON overlay persistence |
| `device.py` | XPU → CUDA → CPU selection |
| `video_io.py` | OpenCV probe / frame iteration |

### Learning (`app/learning/`)

| File | Role |
|------|------|
| `params.py` | `DEFAULTS` / `BOUNDS` — single source of truth for tunables |
| `updates.py` | Deterministic param proposals (risk-tagged) |
| `autolearn.py` | Learn on label-save |
| `validate.py` | Harness before committing offsets/models |
| `model_registry.py` | `shadow → live → archived` for `rally_end` / `serve_start` / `tracknet` |
| `dataset.py` | Training dataset helpers |

### Analysis

| File | Role |
|------|------|
| `analysis/metrics.py` | Temporal IoU, boundary errors |
| `analysis/reporter.py` | Anthropic discrepancy report |

### Scripts (`backend/scripts/`)

| Script | Role |
|--------|------|
| `get_tracknet_weights.py` | Download WASB tennis/badminton → `data/models/tracknet_v2.pt` |
| `train_bounce.py` | BounceGRU → model registry (`--promote`, needs ≥3 labeled clips) |
| `train_gate.py` | Logistic gate from Pass 1 traces (`--apply`) |
| `eval_ball_filter.py` / `eval_matrix.py` | Offline ball quality vs GT |
| `ball_gt_io.py` / `import_manual_tracking.py` | Portable GT import/export |

---

## 8. Frontend

**Entry:** `main.jsx` → `App.jsx` → `Workspace.jsx`

| Component | Role |
|-----------|------|
| `App.jsx` | Clip list + upload; opens workspace |
| `Workspace.jsx` | Tabs: **Setup**, **Pass 1 rallies**, **Pass 2 detection**; job polling; ground-truth ball marking (B/H/N + in/out) |
| `VideoPlayer.jsx` | Original stream + metadata overlays |
| `Timeline.jsx` | Frame scrubber with tracks/marks |
| `CalibrationPanel.jsx` | Court corners / auto-suggest |
| `api.js` | Client for media, labels, reports, chat, models, scoreboard |

**Note:** README still mentions LabelingPanel / ReportPanel / NotesPanel / Assistant tabs. Current UI is leaner (Setup + Pass tabs + GT marking). Backend APIs for reports/chat/params/scoreboard **still exist** and are wired in `api.js`.

---

## 9. Data model (SQLite `data/app.db`)

| Table | Purpose |
|-------|---------|
| `videos` | Clip metadata, optional clip-local `calibration_id`, params version snapshot |
| `jobs` | `pass1` / `pass2` status, progress |
| `rally_labels` | `human` \| `pass1` \| `pass2` intervals + meta JSON |
| `overlays` | Chunked per-frame JSON (`pass1` / `pass2`) |
| `calibrations` | `global` \| `clip_local` corners, H, focal |
| `learned_params` | Append-only versioned tunables |
| `user_notes` | Typed/voice notes, global or clip-local |
| `discrepancies` / `reports` | Human vs AI comparison + LLM output |
| `ball_corrections` | Spot fixes + GT events (`event_type`, `in_out`, court xy, …) |
| `ball_tracks` | One external tracker payload per video |
| `pending_changes` | Medium-risk param proposals awaiting approve/reject |
| `param_eval` | Scoreboard snapshots across param versions |
| `model_registry` | Weight versions by kind/status |

**Storage philosophy:** persist params / labels / traces — **regenerate visuals from metadata**.

---

## 10. Learning system

Detail: [`LEARNING_TRACKED_VALUES.md`](LEARNING_TRACKED_VALUES.md)

- **Global by default:** method-level lessons → new `LearnedParams` version; every upload loads latest  
- **Clip-local:** bumped camera calibration, single-frame ball corrections — never touch baseline  
- Params always **clamped to bounds**; must exist in both `DEFAULTS` and `BOUNDS`  
- Three update paths: auto offsets / human-gated gate / LLM suggestions (risk-routed)  
- Model registry: `shadow → live → archived`  

**Active recall levers now (Pass 1):** `p1_pad_s`, `p1_merge_gap_s`, `p1_min_rally_s`, `p1_cross_window_s`, motion/ball gate knobs  

**Currently inert (code present, not applied):** `start_offset_frames`, `end_offset_frames`, serve/rally-end scalar params that only `rally_logic` reads  

Core Pass 1 defaults (see `params.py` for full set — has grown beyond the original “27”):

| Param | Default | Role |
|-------|---------|------|
| `p1_frame_step` | 3 | Subsample |
| `p1_scale` | 0.4 | Downscale |
| `p1_motion_ratio` | 1.6 | Motion vs ambient |
| `p1_pad_s` | 1.5 | Window padding |
| `p1_merge_gap_s` | 0.5 | Merge adjacent windows |
| `p1_cross_window_s` | 2.0 | Net-cross keep-alive |
| `p1_min_rally_s` | 2.0 | Drop blips |

---

## 11. Architectural decisions

1. **Two-pass cost model** — cheap Pass 1 over 100%; expensive Pass 2 only in active windows  
2. **Metadata overlays, not rendered video** — original file is the only media asset  
3. **Global compounding learning** — append-only params + optional weight registry  
4. **Clamped, risk-routed updates** — deterministic math first; LLM explains/suggests  
5. **Calibration global by default** — one camera setup; clip-local if bumped  
6. **Pluggable ball source** — corrections > external > TrackNet > classical  
7. **Single-worker jobs** — laptop-friendly GPU contention  
8. **Device flexibility** — Intel XPU/OpenVINO vs NVIDIA CUDA  
9. **Current pivot** — detection quality over fine rally segmentation  
10. **Assistant boundary** — can tune/reprocess/note; cannot edit CV source (writes handoffs)

---

## 12. Reference data & related docs

| Path | Contents |
|------|----------|
| `PASS1_PASS2_SPEC.md` | **Current** detection behavior |
| `LEARNING_TRACKED_VALUES.md` | Every tracked/adjusted value + update paths |
| `DETECTION_CONTEXT_FOR_FABLE.md` | Architecture briefing for CV audit |
| `DETECTION_AUDIT.md` | Far-court size→depth failure analysis |
| `BALL_TRACKER_INTEGRATION.md` | External tracker upload API + coords |
| `BALL_AUDIT_HANDOFF.md` | 2026-07-09 session: what’s fixed vs missing, local DB notes |
| `reference_data/manual_tracking.json` | schema_version 1 GT (video `760f1e0e…` / “Second 3 mins.mp4”) |
| `README.md` | Setup + older full-architecture narrative (partially stale) |

---

## 13. Known open problems (as of 2026-07)

- Far-side ball: size→depth noise; pretrained TrackNet does not transfer well to this axial pickleball view — fine-tune on `ball_corrections` is the long-term path (training path may still be incomplete; see `BALL_AUDIT_HANDOFF.md`)  
- Classical residual noise can fabricate returns / false-positive activity  
- Hit-gap work from a prior Fable session may be **absent** on some machines (never committed) — do not assume it exists  
- GT clips in `reference_data` may not match local DB video IDs without `--map` import  
- Fault endings with no ball signature (foot faults, kitchen, service-motion) out of scope  

---

## 14. Quick file index (absolute)

```
c:\Users\balaj\pickleCV\CV-Training-Tool\README.md
c:\Users\balaj\pickleCV\CV-Training-Tool\PROJECT_CONTEXT.md
c:\Users\balaj\pickleCV\CV-Training-Tool\PASS1_PASS2_SPEC.md
c:\Users\balaj\pickleCV\CV-Training-Tool\AUDIO_POSE_FUSION_SPEC.md
c:\Users\balaj\pickleCV\CV-Training-Tool\LEARNING_TRACKED_VALUES.md
c:\Users\balaj\pickleCV\CV-Training-Tool\.env.example
c:\Users\balaj\pickleCV\CV-Training-Tool\backend\app\main.py
c:\Users\balaj\pickleCV\CV-Training-Tool\backend\app\config.py
c:\Users\balaj\pickleCV\CV-Training-Tool\backend\app\db.py
c:\Users\balaj\pickleCV\CV-Training-Tool\backend\app\pipeline.py
c:\Users\balaj\pickleCV\CV-Training-Tool\backend\app\cv\pass1.py
c:\Users\balaj\pickleCV\CV-Training-Tool\backend\app\cv\pass2.py
c:\Users\balaj\pickleCV\CV-Training-Tool\backend\app\cv\ball_tracknet.py
c:\Users\balaj\pickleCV\CV-Training-Tool\backend\app\learning\params.py
c:\Users\balaj\pickleCV\CV-Training-Tool\frontend\src\App.jsx
c:\Users\balaj\pickleCV\CV-Training-Tool\frontend\src\Workspace.jsx
c:\Users\balaj\pickleCV\CV-Training-Tool\reference_data\manual_tracking.json
c:\Users\balaj\pickleCV\CV-Training-Tool\data\app.db
c:\Users\balaj\pickleCV\CV-Training-Tool\data\models\tracknet_v2.pt
```

---

## 15. One-paragraph summary for a new agent

**CV-Training-Tool** is a FastAPI + React local app that finds pickleball rallies in fixed-camera video via a cheap Pass 1 gate (authoritative windows) and a fine Pass 2 detector (players + ball overlays only). State lives in SQLite; overlays are metadata on the original file. Global `LearnedParams` and an optional model registry compound lessons across clips. TrackNet and BounceGRU exist but need weights/fine-tuning; classical ball is the default fallback. Current priority is **per-frame detection quality**, not terminal-event rally ends — `rally_logic` is parked. Start from `PASS1_PASS2_SPEC.md` and this file; treat README architecture diagrams as historical where they conflict.

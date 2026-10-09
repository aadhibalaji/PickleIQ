# PickleCV

**Pickleball rally detection, player and ball tracking, and pose-based coaching from a single fixed camera.**

PickleCV takes ordinary down-the-line pickleball footage and works out where the rallies are. It tracks every player's pose and the ball inside those rallies, finds each swing, and writes a coaching report grounded in measured biomechanics. When it gets something wrong, you correct it. That correction becomes ground truth, and the global parameters every future clip starts from get better.

> Package: `pickleball-rally-system` · Backend: Python 3.12 / FastAPI · Frontend: React 18 / Vite · CV: YOLO11-pose + ByteTrack, TrackNetV2, OpenCV · LLM: Anthropic (reports only, never in the vision loop)

---

## Table of contents

- [What it does](#what-it-does)
- [Design principles](#design-principles)
- [How it works](#how-it-works)
- [Quick start (local)](#quick-start-local)
- [Cloud deployment (AWS)](#cloud-deployment-aws)
- [Configuration](#configuration)
- [Using the app](#using-the-app)
- [Coaching reports and data integrity](#coaching-reports-and-data-integrity)
- [The learning system](#the-learning-system)
- [Project layout](#project-layout)
- [API overview](#api-overview)
- [Data model](#data-model)
- [Scripts](#scripts)
- [Status: what works and what doesn't yet](#status-what-works-and-what-doesnt-yet)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Further reading](#further-reading)

---

## What it does

| Stage | Output |
|---|---|
| **Rally detection** | Start/end windows for every rally in the clip (serve to dead ball) |
| **Player tracking** | Per-frame 17-point pose for every player, with stable identity slots |
| **Ball tracking** | Per-frame ball position, Kalman-smoothed, with bounce/hit markers and height reconstruction |
| **Contact detection** | Every swing, found from wrist-speed peaks, with player, team, court position, shot side and serve flag |
| **Rally analysis** | Kitchen arrival, forward pressure, shot side/length buckets and highlights (Long Rally, Firefight, Poach) |
| **Coaching report** | Per-category ratios, best and worst shots, and one written narrative from Anthropic |
| **Discrepancy report** | AI vs human labels compared, with an LLM explanation of why they differ |

All visuals are **overlays drawn from metadata on top of the original video**. PickleCV never cuts, copies or re-encodes your footage.

---

## Design principles

1. **One fixed camera.** The system is built for axial (down-the-line) footage from a single fixed position. It does not handle broadcast or multi-angle video.
2. **Cheap pass first, expensive pass only where it matters.** Pass 1 scans 100% of the footage at low cost. Pass 2 runs full-rate detection only inside Pass 1's windows, which is roughly 20% of frames.
3. **Metadata, not media.** The original video is the only media asset. Everything else is regenerated from stored params, labels and traces.
4. **Lessons compound globally.** Method-level improvements become a new versioned `LearnedParams` row that every future upload loads. Clip-specific fixes, such as a bumped camera, stay local.
5. **Deterministic first, LLM second.** Parameter updates are clamped, bounded math. The LLM explains and suggests; it never sits in the per-frame loop.
6. **Never fabricate.** If a measurement isn't trustworthy, it is reported as missing, not guessed. See [Coaching reports and data integrity](#coaching-reports-and-data-integrity).

---

## How it works

```
upload
  │
  ▼
Pass 1  ── 100% of footage, downscaled + subsampled ──►  rally_labels (source=pass1)   ← AUTHORITATIVE boundaries
  │         net crossings → motion-energy fallback,
  │         court-ROI masked, hysteresis + pad + merge
  ▼  (windows only)
Pass 2  ── full-rate pose ± ball ─────────────────────►  overlays (source=pass2)        ← display + training data
  │         YOLO11-pose + ByteTrack, ball priority chain,                                  emits NO boundaries
  │         Kalman smoothing, height, bounce/hit markers
  ▼
refine/ ── keep / drop / start-later / end-sooner ────►  data/refine/<id>.json          ← ADVISORY only
  │
  ▼  (chained job: "contacts")
contacts ── wrist-speed peaks from pose ─────────────►  contact_events
  │
  ▼  (on demand: "Generate report")
coaching ── feature vectors → ratios + best/worst ───►  coaching_reports               ← one Anthropic call
```

### Pass 1: the coarse gate (`cv/pass1.py`)

Pass 1 answers one question: *is a rally roughly happening?* It runs at `p1_scale=0.4` and every `p1_frame_step=3` frames. Its primary signal is **net crossings**, which needs court calibration. Where those aren't available it falls back to adaptive **motion energy** inside a court ROI mask, so spectators and neighbouring courts don't trigger it. Hysteresis smooths the signal. Windows are then padded, merged and filtered for blips.

Pass 1 deliberately favours recall. It would rather produce a loose window, or occasionally merge two rallies, than miss one.

### Pass 2: detection only (`cv/pass2.py`)

Pass 2 runs full-rate inside Pass 1 windows and produces per-frame players and ball. It does **not** decide rally boundaries. Ball observations come from the first available source in this order:

1. Manual `ball_corrections` (confidence 1.0)
2. An external tracker uploaded via `POST …/ball_track` (see `docs/BALL_TRACKER_INTEGRATION.md`)
3. **TrackNetV2**, if `TRACKNET_ENABLED=1` and weights are present
4. Classical three-frame differencing with trajectory-consistent blob selection (`ball_coarse.py`, `ball_track.py`)

After that, observations go through an optional velocity/foot/arc-RANSAC filter, then Kalman smoothing with short gap fill, size→depth→height reconstruction and bounce/hit markers. The markers are overlay annotations only.

### Refine (`backend/app/refine/`)

A post-Pass-2 layer that grades each window as keep, drop, start later or end sooner. It uses ballistic 3D reconstruction, a serve gate and classifiers over Pass 1/2 features. Its output is **advisory JSON** and it does not write `rally_labels`.

### Rally Analysis & Coaching (`backend/app/rallyanalysis/`)

This layer is read-only with respect to CV, refine, labels and overlays. It writes only `contact_events` and `coaching_reports`. Contacts are detected after Pass 2 finishes, in their own chained job. The coaching report is generated on demand.

---

## Quick start (local)

### Prerequisites

- Python **3.12**
- Node.js + npm
- A GPU is recommended: NVIDIA (CUDA) or Intel Arc (XPU/OpenVINO). CPU works, but slowly.
- An Anthropic API key. This is optional; everything except reports, the assistant and coaching narratives works without it.

### Install and run

```powershell
cd CV-Training-Tool
copy .env.example .env          # then fill in ANTHROPIC_API_KEY, CV_DEVICE / HW_PROFILE
npm install

# pick ONE backend setup for your hardware
npm run setup:backend:cuda      # NVIDIA
npm run setup:backend           # Intel Arc (XPU)

npm run dev                     # backend :8420 + frontend :5173 (dev.ps1 frees both ports first)
```

Open **http://localhost:5173**.

| Command | What it does |
|---|---|
| `npm run dev` | Backend + frontend together |
| `npm run dev:backend` | FastAPI on `:8420` only |
| `npm run dev:frontend` | Vite on `:5173` only (proxies `/api` → `:8420`) |
| `npm run build` | Builds `frontend/dist`, which the backend then serves at `:8420` |

The backend venv lives at `backend/.venv` (`backend/.venv/Scripts/python.exe` on Windows).

> ⚠️ **Don't mix requirement files.** `requirements.txt` pulls Intel XPU torch builds and `requirements-cuda.txt` targets NVIDIA. Installing both gives you the wrong torch.

YOLO11-pose weights (`yolo11n-pose.pt`) download automatically. TrackNet weights are optional:

```powershell
backend\.venv\Scripts\python.exe backend\scripts\get_tracknet_weights.py --sport tennis
# → data/models/tracknet_v2.pt
```

The pretrained weights are for tennis and transfer poorly to this camera view. That's why TrackNet is off by default.

---

## Cloud deployment (AWS)

For GPU processing without a GPU laptop, PickleCV runs as **two containers with separate jobs**:

```
Your machine                          AWS
┌────────────────────┐   upload     ┌──────────────────────┐
│ Local container    │ ───────────► │ S3                   │
│ • serves the UI    │              │ videos, calibration, │
│ • RUN_PROCESSING=0 │ ◄─────────── │ progress + results   │
│ • polls S3         │   results    └──────────┬───────────┘
└────────────────────┘                         │ ObjectCreated
                                               ▼
                                     ┌──────────────────────┐
                                     │ Lambda               │ starts the worker
                                     │ (picklecv-start-ec2) │ if it's stopped
                                     └──────────┬───────────┘
                                                ▼
                                     ┌──────────────────────┐
                                     │ EC2 GPU worker       │
                                     │ (g4dn.xlarge, T4)    │
                                     │ • polls S3 every 20s │
                                     │ • Pass1→2→contacts→  │
                                     │   coaching           │
                                     │ • streams progress   │
                                     │ • self-stops on idle │
                                     └──────────────────────┘
```

**Why the split?** An earlier single-instance design ran the UI and the GPU worker on one EC2 box. When the queue emptied and the instance stopped itself to save money, the whole app disappeared with it, including the screens you'd need to fix the problem, such as court calibration. Now self-stop only ever affects the GPU worker.

### Infrastructure pieces

| Component | Role |
|---|---|
| **S3** (`picklecv-videos-dev`) | Durable store for videos, calibration, progress and results. Block Public Access on, SSE-S3. |
| **ECR** (`picklecv-pipeline`) | One Docker image (CUDA + PyTorch + app + YOLO weights) |
| **Secrets Manager** (`picklecv/anthropic-api-key`) | Fetched at startup by `secrets_bootstrap.py` via the instance role |
| **Lambda** (`picklecv-start-ec2`) | On S3 upload, starts stopped instances tagged `Project=PickleCV` |
| **EC2 worker** | GPU processing. Stops itself via `self_stop.py` when the queue is empty. |

### IAM, scoped to the minimum

- **EC2 role:** S3 get/put on the bucket, read on the secret, ECR pull on the repo (plus the unscoped `ecr:GetAuthorizationToken`), and `ec2:StopInstances` conditioned on `Project=PickleCV`
- **Lambda role:** `ec2:StartInstances` conditioned on the tag, `ec2:DescribeInstances`, and basic logging

### Build and deploy

```bash
docker build --platform linux/amd64 -t picklecv-pipeline .
docker tag picklecv-pipeline:latest <account>.dkr.ecr.us-east-2.amazonaws.com/picklecv-pipeline:latest
docker push <account>.dkr.ecr.us-east-2.amazonaws.com/picklecv-pipeline:latest

scripts/restart-local.sh   # on your machine
restart-ec2.sh             # on the EC2 instance
```

### Gotchas that have actually bitten us

- **IMDS hop limit.** Set `HttpPutResponseHopLimit=2` on the instance. Docker's bridge network adds a hop, and with the default of 1, metadata calls (self-stop, credentials) fail from inside the container.
- **`.env` isn't in the image.** It's left out on purpose to keep secrets out. Every local default, especially **`PASS2_BALL=1`**, must be passed explicitly to the deployed container. A missing `PASS2_BALL` silently disables ball tracking.
- **No Elastic IP.** The worker's public IP changes on every start. You only need it for SSH debugging.
- **Calibration sync timing.** Calibration syncs to the worker on the same 20-second poll as videos. After changing calibration, wait a few seconds before uploading.

---

## Configuration

Copy `.env.example` → `.env`. **Explicit env vars always override the hardware profile.**

### Core

| Variable | Default | Purpose |
|---|---|---|
| `HW_PROFILE` | *(unset)* | Bundle of speed knobs per machine, e.g. `cuda` (NVIDIA, ball on) or `arc140v` (Intel Arc, OpenVINO on, ball off). Defined in `backend/app/hw_profiles.json`; add a profile there without code changes. See `docs/HW_PROFILES.md`. |
| `CV_DEVICE` | `auto` | `auto` \| `cuda` \| `xpu` \| `cpu` (auto tries XPU → CUDA → CPU) |
| `PASS2_BALL` | `1` | `0` = player-only Pass 2. Must be `1` for the refine serve gate. |
| `TRACKNET_ENABLED` | `0` | Opt-in TrackNet ball detector |
| `TRACKNET_WEIGHTS` | `data/models/tracknet_v2.pt` | Explicit weights path (wins if set) |
| `POSE_MODEL` | `yolo11n-pose.pt` | Pose weights |
| `DATA_DIR` | `./data` | Storage root |

### Anthropic

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Reports, assistant and coaching narrative. Fetched from Secrets Manager in the cloud. |
| `ANTHROPIC_MODEL` | Chat/assistant model |
| `ANTHROPIC_REPORT_MODEL` | Cheaper model for discrepancy + coaching reports |

### Cloud

| Variable | Purpose |
|---|---|
| `AWS_REGION` | Region for all boto3 clients (`us-east-2`) |
| `STORAGE_BACKEND` | `local` \| `s3` |
| `S3_VIDEO_BUCKET` | Bucket name |
| `ANTHROPIC_SECRET_ID` | Secrets Manager ID for the key |
| `RUN_PROCESSING` | `0` on the local UI container, so it never runs Pass 1/2 itself |

Detection tunables (`p1_scale`, `p1_pad_s`, `p1_cross_window_s`, etc.) are **not** env vars. They live in `backend/app/learning/params.py` as versioned, bounded `LearnedParams`.

---

## Using the app

The workspace has four tabs, used left to right:

1. **Setup.** Calibrate the court by clicking the four corners or accepting the auto-suggestion. Calibration is global by default; mark it clip-local if the camera was bumped. Without calibration, Pass 1 loses net-crossing detection and Pass 2 can't run.
2. **Pass 1 rallies.** Review detected windows on the timeline. Use 👍/👎 to grade keep/drop, which feeds the refine recommendation loop.
3. **Pass 2 detection.** Watch pose and ball overlays on the original video. Mark ground-truth ball events with **B** (bounce), **H** (hit) and **N** (none), plus in/out.
4. **Rally Analysis.** Position and shot metrics, highlights, coaching moments, and **Generate report**, which makes the one Anthropic call.

Uploading kicks off the chain automatically: Pass 1 → Pass 2 → contacts. Progress bars poll the job queue, or poll S3 in cloud mode.

---

## Coaching reports and data integrity

Coaching makes claims about someone's body, so it is held to a strict rule: **a metric the system can't measure reliably is reported as missing. It is never estimated, averaged or defaulted.**

**Per-contact metrics:** knee bend, shoulder rotation, elbow extension, weight transfer, contact-height proxy, follow-through and split-step delay.

**Buckets:** `footwork` · `paddle_prep` · `contact_point` · `follow_through`

**How confidence gating works:**

- `players.py` stores YOLO's per-keypoint confidence (`kp_conf`) alongside position (`kp`).
- In `ShotFeatureVector` (`contracts.py`), every keypoint-derived field is `float | None`. A field is `None` when a supporting keypoint falls below `KP_CONF_MIN = 0.5`. Each shot's `low_confidence_metrics` lists which fields are missing.
- The Anthropic prompt is told explicitly never to make a claim about a null field.
- `footwork` and `paddle_prep` (knees, hips, shoulders) are the most reliable. `contact_point` and `follow_through` depend on wrists and elbows, the weakest joints, so expect fewer tips there. That is by design, not a bug.

**Caveats:**

- `KP_CONF_MIN=0.5` is a conservative global starting point, not a calibrated per-joint value. Two clips disagreed on which joint is weakest.
- `contact_height_m` is `court_y − NET_Y`, a ground-plane proxy, not true height. It comes from the contact's court position, not keypoints.
- Biomechanical ranges are coach-tunable defaults and haven't been validated by a coach yet.
- Overlays processed **before 2026-07-25** have no `kp_conf`, so they fall back to presence-only gating. Coaching reports generated before the integrity fix may contain fabricated neutral values. **Reprocess and regenerate before trusting them.**

---

## The learning system

Every correction can make the next clip better.

- **Global by default.** Method-level lessons produce a new append-only `LearnedParams` version, and every upload loads the latest.
- **Clip-local when appropriate.** A bumped camera's calibration or a single-frame ball fix never touches the baseline.
- **Always clamped.** Every tunable must exist in both `DEFAULTS` and `BOUNDS` in `params.py`.
- **Risk-routed updates.** Low-risk deterministic changes apply automatically. Medium-risk proposals land in `pending_changes` for approval. LLM suggestions are routed by risk.
- **Model registry.** Weights for `rally_end`, `serve_start` and `tracknet` move through `shadow → live → archived`. Promote or reject them via `/api/models`.
- **Scoreboard.** `param_eval` snapshots quality across param versions, so you can see whether a change helped.

The in-app **assistant** can read traces, list rallies, compare to human labels, apply clamped params, reprocess and add notes. It **cannot edit CV source code**.

Full inventory of tracked values: `docs/LEARNING_TRACKED_VALUES.md`.

---

## Project layout

```
CV-Training-Tool/
├── Dockerfile                  single image: API + CV pipeline
├── package.json                npm run dev / setup:backend*
├── dev.ps1                     frees :5173/:8420, starts both
├── .env.example
├── backend/
│   ├── requirements.txt        Intel XPU torch
│   ├── requirements-cuda.txt   NVIDIA (pinned torch + torchvision)
│   ├── app/
│   │   ├── main.py             FastAPI entry
│   │   ├── config.py · paths.py · db.py
│   │   ├── pipeline.py         Pass1 → Pass2 → contacts chain
│   │   ├── jobs.py             single-worker job queue
│   │   ├── hw_profile.py · hw_profiles.json
│   │   ├── secrets_bootstrap.py · storage_s3.py · self_stop.py   (cloud)
│   │   ├── api/                media, annotate, analyze, chat, models, rally_analysis
│   │   ├── cv/                 pass1, pass2, players, ball_*, court, kalman,
│   │   │                       bounce, identity, exclusions, overlays, rally_logic
│   │   ├── refine/             ballistic, serve_gate, classifiers, features,
│   │   │                       postprocess, recommend, evaluate
│   │   ├── rallyanalysis/      contact_detect, coaching, contracts, readiness,
│   │   │                       position_metrics, shot_metrics, highlights
│   │   ├── learning/           params, updates, autolearn, validate, model_registry
│   │   └── analysis/           metrics, reporter (discrepancy report)
│   └── scripts/                train/eval/import utilities
├── frontend/src/               App, Workspace, VideoPlayer, Timeline,
│                               CalibrationPanel, LabelingPanel, RallyAnalysis/
├── docs/                       specs and design notes (see below)
├── reference_data/
│   └── manual_tracking.json    portable ground-truth export (schema_version 1)
└── data/                       gitignored: app.db, videos/, models/, refine/
```

---

## API overview

All routes are under `/api`. Health check: `GET /api/health`.

| Router | Responsibility |
|---|---|
| `media` | Upload, list, stream, reprocess, cancel jobs |
| `annotate` | Human rally labels, calibration, ball corrections/GT, external `ball_track`, notes |
| `analyze` | Overlays, detect-debug, discrepancy reports, pending changes, scoreboard, params |
| `chat` | In-app Claude assistant (bounded tools) |
| `models` | Model registry list/promote/reject, learning curve |
| `rally_analysis` | Contacts, position/shot metrics, highlights, readiness, coaching report |

**Rally Analysis endpoints:**

| Method | Path | Behaviour |
|---|---|---|
| `POST` | `/rally-analysis/{id}/detect-contacts` | Run and replace `contact_events` |
| `GET` | `/rally-analysis/{id}/contact-events` | List contacts |
| `GET` | `/rally-analysis/{id}/position-metrics` | Kitchen arrival / forward pressure |
| `GET` | `/rally-analysis/{id}/shot-metrics` | Shot side / length buckets |
| `GET` | `/rally-analysis/{id}/highlights` | Rule-based tags |
| `GET` | `/rally-analysis/{id}/readiness` | Per-rally Pass 2 readiness |
| `POST` | `/rally-analysis/{id}/coaching-report` | Generate and persist (one Anthropic call) |
| `GET` | `/rally-analysis/{id}/coaching-report` | Cached report or `{report: null}` |

---

## Data model

SQLite in WAL mode at `data/app.db`. The guiding rule: **persist params, labels and traces; regenerate visuals.**

| Table | Purpose |
|---|---|
| `videos` | Clip metadata, optional clip-local calibration, params snapshot, `s3_key` |
| `jobs` | `pass1` / `pass2` / `contacts` status and progress |
| `rally_labels` | `human` \| `pass1` intervals plus meta |
| `overlays` | Chunked per-frame JSON (`pass1` / `pass2`) |
| `calibrations` | Global or clip-local corners, homography, focal |
| `learned_params` | Append-only versioned tunables |
| `ball_corrections` | Spot fixes and GT events (type, in/out, court xy) |
| `ball_tracks` | External tracker payload per video |
| `contact_events` | Per-swing rows |
| `coaching_reports` | One per video: ratios, best/worst, narrative |
| `discrepancies` / `reports` | Human vs AI comparison and LLM output |
| `pending_changes` · `param_eval` · `model_registry` · `user_notes` | Learning-loop bookkeeping |

---

## Scripts

Run from `backend/` with the venv Python.

| Script | Purpose |
|---|---|
| `python -m app.refine.evaluate` | Reproduce the refine benchmark |
| `scripts/get_tracknet_weights.py --sport tennis` | Download WASB TrackNet weights |
| `scripts/train_bounce.py [--promote]` | Train BounceGRU into the registry (needs ≥3 labeled clips) |
| `scripts/train_gate.py [--apply]` | Fit the Pass 1 logistic gate from traces |
| `scripts/eval_ball_filter.py` / `eval_matrix.py` | Offline ball quality against GT |
| `scripts/import_manual_tracking.py [--map]` | Import portable GT. Use `--map` when local video IDs differ. |

---

## Status: what works and what doesn't yet

**Solid**

- Upload → async Pass 1/2 → labels → calibration → compare → LLM report → global params
- Court homography and focal self-calibration
- YOLO11-pose + ByteTrack with stable identity slots
- Pass 1 coarse gate and the refine keep/drop grading loop
- Ballistic 3D + serve gate (benchmark F1 ≈ 0.70 on 7 labeled clips)
- Rally Analysis: contacts, metrics, highlights, confidence-gated coaching reports
- Two-tier AWS deployment with automatic start/stop

**Soft or gated**

- **Far-baseline ball tracking.** The far ball is a few pixels across, and size→depth is the largest physics noise source. TrackNet needs fine-tuning on this camera.
- **BounceGRU** and the **learned Pass 1 gate** are trainable but not yet driving anything.
- **Coaching ranges** haven't been validated by a coach, and height is a ground-plane proxy.
- **Faults with no ball signature** (foot faults, kitchen, service motion) are out of scope for v1.

**Known issues**

- **Merged rallies.** With `p1_cross_window_s=3.0`, false negatives are near zero, but adjacent rallies merge, and refine can't split them yet. Root cause: P(live) overweights `cross_recency` and treats static ready posture as live, while player motion (the real dead-ball signal) is underweighted.
- **"Re-run Pass 1"** doesn't work in the cloud setup. Re-upload the video instead.
- Pass 1 progress can sit at **99%** briefly after finishing. This is cosmetic.
- One background job was seen to **hang for ~22.8 hours with no output and no error**. Keep an eye on long-running jobs.
- `queue.Queue.empty()` isn't atomic with `submit()`. That's fine for a single user, but it's a gap if concurrency is added.

---

## Roadmap

1. **Rally-end detection and splitting merged windows.** These are the same missing capability: detecting when live play stops. The plan is to reweight P(live) toward players stopping, de-weight `cross_recency` and static ready posture, and work in `refine/classifiers.py`, `features.py` and `postprocess.py`.
2. **User-driven segmentation and correction** (designed, not yet built). Users correct Pass 1 windows and draw arbitrary segments for coaching. Those corrections write into existing label tables, so they double as free ground truth. Pass 2 needs to accept arbitrary frame ranges, and per-shot confidence gating still applies inside user-selected clips.
3. **Richer coaching reports.** Shot counts by type and zone, forward momentum over time, a court-relative shot heatmap, and a multi-section narrative, all without loosening the no-fabrication rule.
4. **TrackNet fine-tuning** on accumulated `ball_corrections`.
5. Validate coaching ranges with a coach, cache ballistic reconstruction for sweeps, and truncate ballistic arcs at height = 0.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Pass 2 fails right after Pass 1 | No court calibration. Calibrate in **Setup** and reprocess. |
| No ball overlay in the cloud | `PASS2_BALL` wasn't passed to `docker run` |
| `torchvision::nms` error | torch and torchvision versions don't match. Use the pinned `requirements-cuda.txt`. |
| Self-stop or credential calls fail inside Docker | IMDS hop limit is still 1. Raise it to 2. |
| ECR pull fails at boot | EC2 role is missing `ecr:GetAuthorizationToken`. Check `cloud-init-output.log`. |
| Wrong or very slow device | Mixed XPU/CUDA requirements, or `CV_DEVICE`/`HW_PROFILE` mismatch |
| GT import finds no matching video | Local DB IDs differ. Use `import_manual_tracking.py --map`. |
| Coaching tips look generic or suspiciously neutral | The report predates the integrity fix. Reprocess the video and regenerate. |

---

## Further reading

| Doc | Contents |
|---|---|
| `docs/PASS1_PASS2_RALLY_BOUNDARIES.md` | Source of truth for boundary logic. Update it in the same change if you touch boundaries. |
| `docs/PASS1_PASS2_SPEC.md` | Broader pipeline spec |
| `docs/RALLY_SYSTEM.md` | Rally Analysis & Coaching design |
| `docs/LEARNING_TRACKED_VALUES.md` | Every tracked value and how it's updated |
| `docs/BALL_DETECTION.md` · `docs/BALL_TRACKER_INTEGRATION.md` | Ball tracking and the external tracker API |
| `docs/DETECTION_AUDIT.md` | Far-court size→depth failure analysis |
| `docs/HW_PROFILES.md` | Hardware profiles |

---

<sub>PickleCV runs entirely on your hardware or your own AWS account. Video never leaves your control, and the LLM only ever sees summarized metrics, never frames.</sub>

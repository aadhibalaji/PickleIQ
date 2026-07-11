# Pickleball AI Rally-Detection System

A local video-analysis app that detects rallies (serve → the frame the ball becomes
unplayable) in fixed-camera pickleball footage, lets you label the ground truth,
compares the two, explains discrepancies with the Anthropic API, and **saves the
lessons globally** so every future clip starts from an improved baseline.

- **Backend:** Python 3.12, FastAPI, OpenCV, PyTorch (Intel Arc XPU / CUDA / CPU),
  Ultralytics YOLO11-pose + ByteTrack. All processing is offline and asynchronous.
- **Frontend:** React + Vite. All overlays are rendered from stored metadata over
  the original video — no video is ever cut, edited, or copied.
- **LLM:** Anthropic API, used only for the discrepancy feedback report — never in
  the per-frame vision loop.

---

## Quick start

```powershell
# 1. Configure (one time)
copy .env.example .env        # then paste your Anthropic API key into .env
npm install                   # one time, at the repo root (installs concurrently)

# 2. Backend Python env (one time) — pick YOUR GPU path:
npm run setup:backend:cuda    # NVIDIA GPU (recommended on CUDA machines)
# OR
npm run setup:backend         # Intel Arc (original dev machine)

# 3. Run everything (backend :8420 + frontend :5173) in one terminal
npm run dev                   # open http://localhost:5173
```

### Backend setup (detail)

**NVIDIA GPU** — use the CUDA PyTorch wheels, not the Intel XPU index:

```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-cuda.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Pick the `cu12x` index that matches `nvidia-smi` — see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/). Set
`CV_DEVICE=cuda` in `.env`.

**Intel Arc** — production path uses OpenVINO on the iGPU:

```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Set `CV_DEVICE=auto` or `xpu` in `.env`. Optional: `pip install openvino` for the
fast pose/TrackNet path on Intel GPU.

`npm run dev` launches the FastAPI backend (from `backend\.venv`) and the Vite
frontend together. To run them separately: `npm run dev:backend` and
`npm run dev:frontend`.

Production-ish alternative: `npm run build` once, then the backend serves the
built frontend itself at http://127.0.0.1:8420.

### In-app assistant (💬 Assistant tab)
A chat with the same Claude model, embedded in the UI with full context of the
current clip. It can read the per-frame feature traces to diagnose a specific
moment you point at, score Pass 2 against your labels, tune the global
(clamped) parameters, trigger reprocessing, and save notes. It **cannot** edit
the CV source code — when a fix needs source edits it writes a precise handoff
note instead of pretending. Uses `ANTHROPIC_API_KEY` (same key as the report).
The tool surface is bounded and safe: `get_feature_trace`, `list_rallies`,
`compare_to_human`, `apply_params` (clamped), `reprocess`, `add_note`.

### First-run flow
1. **Upload a clip.** Pass 1 starts immediately in the background; go straight to
   the *Label rallies* tab and start marking ground truth while it runs.
2. **Calibrate the court** (one time per camera setup): in the labeling tab, click
   "Auto-detect corners" or click ≥4 named reference points on a paused frame.
   Saved globally — every future clip reuses it. Pass 2 requires this.
3. **Label rallies** with the Start/End buttons (strict alternating state machine),
   review/adjust the pairs, add notes about problems you see (typed or voice).
4. **Compare** your labels vs Pass 1/Pass 2 tracks on the shared timeline.
5. **Generate report** — computes temporal IoU + start/end boundary errors,
   classifies each discrepancy (boundary drift vs missed/extra), applies
   deterministic global parameter corrections, and asks Claude to explain the
   discrepancies and suggest fixes (suggestions are clamped and only applied when
   you accept them).
6. **Regenerate** re-runs detection with the current global params/calibration —
   overlays are regenerated from metadata, never stored as new video.

---

## Architecture

```
upload ─► Pass 1 (coarse gate, 100% of footage, cheap)
             motion energy + coarse ball blobs + sparse serve-formation checks
             gate is ADAPTIVE: motion must exceed a slow ambient baseline by
             p1_motion_ratio (transfers across lighting/scenes; the ratio is
             the learnable knob)
             └─► active/dead windows, padded ±1.5 s
                    └─► Pass 2 (fine, only inside windows ≈20% of clip)
                          full-rate player tracking (YOLO11-pose + ByteTrack)
                          full-rate ball tracking (TrackNet | classical fallback)
                          Kalman smoothing + gap bridging
                          apparent size → depth → height above court plane
                          bounce vs hit (height minima + gravity + player proximity)
                          rally start (serve formation + first fast ball behind baseline)
                          rally end   (terminal event: bounce-out / double bounce / net fail,
                                       confirmed — never timed — by player behaviour)
```

Key end-logic rules (in `backend/app/cv/rally_logic.py`):
- The recorded end is the **terminal-event frame**, not when the ball stops or
  players reset.
- Low ball speed / low player motion **never** ends a rally on its own (protects
  dink exchanges).
- Ball-track dropout while players are still active is treated as a tracking gap,
  not an end.
- A ball crossing the sideline **in the air** is not out — only a bounce outside
  the court polygon fires the out signal (which is why bounce detection is the
  prerequisite for out calls).

### Learning is global by default
Feedback updates the shared model/parameters, not the clip's output:
- **Global (default):** method-level lessons — boundary-offset corrections,
  Pass 1 gate tuning — become a new versioned `LearnedParams` row. Every new
  upload loads the latest version before Pass 1 runs.
- **Clip-local:** one-off oddities (bumped camera → clip-local calibration,
  single-frame ball spot-corrections) attach to that clip only and never touch
  the baseline.

Storage stays light: parameters, labels, discrepancy records, notes, and
per-frame feature traces are persisted (SQLite, `data/app.db`); visuals are
regenerated on demand from that metadata.

---

## What works immediately vs what needs weights/fine-tuning (honest status)

**Works immediately**
- Full app: upload → background processing → two results tabs → labeling state
  machine → calibration → comparison → LLM report → compound learning loop.
- Court homography + focal self-calibration (verified to 0% focal error on
  synthetic projections).
- Player detection / pose / tracking / team + serve-time numbering
  (YOLO11-pose weights auto-download on first run).
- Pass 1 coarse gate and Pass 2 terminal-event logic.

**Real, but needs weights + fine-tuning — the hard 20%**
- **Ball tracking** — a full TrackNetV2 implementation is integrated
  (`backend/app/cv/ball_tracknet.py`, `TODO(weights)`). Until you download
  public TrackNet weights and set `TRACKNET_WEIGHTS` in `.env`, Pass 2 uses a
  real classical fallback (three-frame differencing + blob geometry) that is
  honestly weaker, especially on the far-baseline ball (a few pixels wide).
  Public TrackNetV2 tennis/badminton checkpoints load directly when layer
  shapes match; fine-tune on your own clips via ball spot-corrections.
- **Bounce detection** — the axial camera view is the genuinely hard part.
  The physics reconstruction (size→depth→height, bounce = height minimum near
  zero away from players) runs today; the preferred learned temporal model
  (`BounceGRU`, `TODO(fine-tune)`) trains from your accumulated labels via
  `backend/scripts/train_bounce.py` and is picked up automatically once
  `data/models/bounce_gru.pt` exists. 60 fps footage materially improves
  bounce localization.
- **Pass 1 gate classifier** — the thresholded gate logs its own features;
  `backend/scripts/train_gate.py` fits a classifier over them once you've
  labeled a few clips and can write the learned threshold into the global
  params (`--apply`).

**Out of scope for v1 (not faked):** faults with no distinctive ball trajectory —
foot faults, kitchen violations, service-motion faults. These are only catchable
via the lagging "players stopped and reset" signal with fuzzy timing.

---

## Configuration

`.env` (copy from `.env.example`):

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Feedback report only. Everything else works without it. |
| `ANTHROPIC_MODEL` | Report model (default `claude-opus-4-8`). |
| `CV_DEVICE` | `auto` (XPU → CUDA → CPU) / `xpu` / `cuda` / `cpu`. |
| `TRACKNET_WEIGHTS` | Path to TrackNet weights; unset = classical fallback. |
| `DATA_DIR` | Storage root (default `./data`). |

### GPU / device notes

| Hardware | Install file | `.env` | Inference path |
|---|---|---|---|
| **NVIDIA GPU** | `requirements-cuda.txt` + CUDA torch wheels | `CV_DEVICE=cuda` | Ultralytics + TrackNet on CUDA |
| **Intel Arc** | `requirements.txt` (XPU wheels) | `CV_DEVICE=auto` or `xpu` | OpenVINO on iGPU (optional `pip install openvino`) |
| **CPU only** | either file, torch CPU wheels | `CV_DEVICE=cpu` | Slower; same detection quality at default params |

**Intel Arc 140V (original dev machine):**
- PyTorch runs on the Arc GPU via the XPU backend — used by TrackNet and BounceGRU
  when OpenVINO is unavailable.
- Ultralytics rejects `device='xpu'` for YOLO-pose `.track()`; with OpenVINO
  installed, pose runs on the Intel GPU. Without it, pose falls back to CPU.

**NVIDIA:**
- Do **not** install from the XPU index in `requirements.txt` — that can pull
  the wrong torch build. Use `requirements-cuda.txt` instead.
- OpenVINO targets Intel GPU only; on NVIDIA the code uses CUDA torch/Ultralytics
  automatically. Detection quality matches at the same params; the main win is speed.

## Repo map

```
backend/app/cv/        court.py (homography+camera), players.py, ball_tracknet.py,
                       ball_coarse.py, kalman.py, bounce.py, rally_logic.py,
                       pass1.py, pass2.py, overlays.py
backend/app/learning/  params.py (versioned global LearnedParams), updates.py
backend/app/analysis/  metrics.py (IoU + boundary errors), reporter.py (Anthropic)
backend/app/api/       media.py, annotate.py, analyze.py
backend/scripts/       train_bounce.py, train_gate.py
frontend/src/          Workspace, VideoPlayer (metadata overlays), Timeline,
                       LabelingPanel, CalibrationPanel, NotesPanel, ReportPanel
data/                  app.db + uploaded videos + trained models (gitignored)
```

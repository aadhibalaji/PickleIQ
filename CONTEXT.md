# Codebase Context — PickleCV / CV-Training-Tool

_Snapshot: 2026-07-11. For agents and humans starting a session._

Living companions (keep in sync when behavior changes):
- `PASS1_PASS2_RALLY_BOUNDARIES.md` — boundary-logic source of truth (has a maintenance contract)
- `PASS1_PASS2_SPEC.md` — broader pipeline spec
- `docs/RALLY_SYSTEM.md`, `docs/BALL_DETECTION.md` — deeper technical refs
- `SESSION_HANDOFF_2026-07-10.md` — detailed prior-session notes (benchmark tables, file-level diffs)

---

## What this is

Local pickleball video-analysis app: detect rallies (serve → terminal unplayable frame)
in fixed-camera footage, label ground truth, compare, explain discrepancies via Anthropic,
and compound lessons into global params.

| Layer | Stack |
|---|---|
| Backend | Python 3.12, FastAPI `:8420`, OpenCV, PyTorch, Ultralytics YOLO11-pose + ByteTrack, SQLite |
| Frontend | React + Vite `:5173` — overlays from metadata only (never rewrites video) |
| LLM | Anthropic — discrepancy report + in-app Assistant only; never in the per-frame CV loop |

Run: `npm run dev` (via `dev.ps1` — frees 5173/8420 first) → http://localhost:5173  
Venv: `backend/.venv/Scripts/python.exe`

---

## Pipeline (mental model)

```
upload → Pass 1 (100% footage, cheap) → rally_labels source='pass1'  ← AUTHORITATIVE boundaries
            ↓ windows only
         Pass 2 (full-rate pose ± ball) → overlays source='pass2'     ← display / training data
            ↓
         refine/  → data/refine/<id>.json                            ← ADVISORY keep/drop/trim
                                                                      (does NOT write rally_labels)
```

**Do not conflate “Pass 2”:**
- `cv/pass2.py` = detection/visualization only; emits **no** rally boundaries.
- `backend/app/refine/` = post-Pass-2 keep/drop/start-later/end-sooner; JSON-only today.

Key end rules (`cv/rally_logic.py` / refine): end = terminal event frame, not “ball stopped”;
low speed alone never ends a rally (protects dinks); air-out ≠ bounce-out.

---

## Git / working tree (this snapshot)

| | |
|---|---|
| Branch | `rally-analysis` |
| HEAD | `d8eb5fc` — ballistic 3D refinement, recommendation module, rally boundary docs |
| Uncommitted | ByteTrack YAML path cache (`players.py` — ~29% Pass 2 time fix), `dev.ps1` + `npm run dev` wiring, `vite` `strictPort`, `requirements_cuda.txt` |

Recent landed work (committed):
1. Refine module + serve-gate + keep/drop grading UI + recommend sweep
2. Ballistic 3D projection + arc contact recovery (serve `no_serve` false-drop fix)
3. Pass 1 gap-tolerant net-crossing; `p1_cross_window_s=3.0` (FN≈0%, merges adjacent rallies)
4. Ball GT tooling, TrackNet eval matrix, OpenVINO pose path, Pass 2 ball toggle

---

## Env / hardware (this machine)

Per-GPU speed knobs (device, precision, OpenVINO, TrackNet, Pass 1 subsampling)
bundle into one `HW_PROFILE` name — see `backend/app/hw_profiles.json` (add a
profile there for a new machine, no code change) and `backend/app/hw_profile.py`
(schema). This is a SPEED setting, not for cross-machine reproducibility;
unset `HW_PROFILE` keeps the plain `.env` knobs below working unchanged.

From `.env` (do not commit secrets):

| Knob | Current | Meaning |
|---|---|---|
| `CV_DEVICE` | `cuda` | Was historically Arc/XPU; CUDA path + `requirements_cuda.txt` in tree |
| `PASS2_BALL` | `0` | Pass 2 ball OFF — pose-only until set to `1` (needed for refine/serve gate) |
| `TRACKNET_ENABLED` | `1` | On |
| `TRACKNET_WEIGHTS` | `data/models/tracknet_v2.pt` | Present (~45 MB); also `wasb_tracknetv2_tennis.pth.tar` |
| `ANTHROPIC_*` | configured in `.env` | Report + Assistant |

Other facts:
- TrackNet weights **are on disk**; classical fallback only if disabled/missing.
- Refine override YAML `data/refine/config.override.yaml` — **not present** right now (hand-off once had `serve_gate_min: 0.35`).
- SQLite: `data/app.db`. Some Pass 2 overlays may be stale vs current Pass 1 windows.

---

## What works vs what’s soft

**Solid**
- Upload → async Pass 1/2 → label state machine → court calibration → compare → LLM report → global `LearnedParams`
- Court homography / focal self-cal
- YOLO11-pose players + ByteTrack (absolute tracker YAML path pending commit)
- Pass 1 coarse gate + refine keep/drop grading loop (built; little real grading yet)
- Ballistic 3D + `serve_gate_min≈0.35` path (benchmark F1 ~0.70 on 7 labeled clips)

**Soft / gated**
- Ball track quality (far baseline, classical bursts; TrackNet tennis weights need pickleball fine-tune)
- BounceGRU (`TODO(fine-tune)` — trains via `scripts/train_bounce.py` when labels exist)
- Pass 1 learned gate classifier (`scripts/train_gate.py`)
- Foot/kitchen/service-motion faults — out of scope for v1

---

## THE open problem (recommended next work)

**Rally-end detection and splitting merged Pass 1 windows are the same missing capability:
detect when live play stops.**

- `p1_cross_window_s=3.0` → FN≈0% but **merges** adjacent rallies.
- Refine does **not** split merged windows; P(live) stays high in between-point gaps.
- Root cause skew: `cross_recency` overweighted; static `ready_frac` treated as live; **player motion** (the real dead discriminator) underweighted.

**Recommended fix:** reweight P(live) / Stage C toward player-motion stopping; de-weight `cross_recency` and static ready. That sharpens ends **and** enables safe splits under `cw=3.0`.

Stopgap alternative: drop `p1_cross_window_s` toward 2.0 (fewer merges, some window FNs).

Also backlog: grade real clips through keep/drop UI; cache ballistic recon for sweeps; truncate ballistic arcs at height=0.

---

## Repo map (where to look)

```
backend/app/cv/        pass1.py, pass2.py, players.py, ball_*, court.py,
                       bounce.py, rally_logic.py, overlays.py, identity.py
backend/app/refine/    ballistic.py, serve_gate.py, classifiers.py, features.py,
                       postprocess.py, recommend.py, evaluate.py, config.py
backend/app/learning/  params.py (versioned LearnedParams), updates.py, …
backend/app/analysis/  metrics.py, reporter.py
backend/app/api/       media.py, annotate.py (grades), analyze.py, chat.py
backend/scripts/       train_bounce.py, train_gate.py, get_tracknet_weights.py, eval_*
frontend/src/          Workspace, VideoPlayer, Timeline, LabelingPanel,
                       CalibrationPanel, NotesPanel, ReportPanel
data/                  app.db, videos/, models/, refine/   (gitignored payloads)
```

Reproduce refine benchmark:
```powershell
cd backend
.\.venv\Scripts\python.exe -m app.refine.evaluate
```
Reprocess with ball (for serve gate): set `PASS2_BALL=1` in `.env`.

---

## First actions for a new session

1. Read this file + `PASS1_PASS2_RALLY_BOUNDARIES.md` § open end/split problem.
2. Confirm `.env` (`PASS2_BALL`, `CV_DEVICE`, TrackNet path) matches the task.
3. If continuing end-detection: start in `refine/classifiers.py` + `features.py` + `postprocess.py`.
4. If continuing grading loop: use Pass tab 👍/👎 → recommendation → write override YAML.
5. Update `PASS1_PASS2_RALLY_BOUNDARIES.md` in the same change if you touch boundary logic.

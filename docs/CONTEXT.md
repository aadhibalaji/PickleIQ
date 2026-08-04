# Codebase Context — PickleCV / CV-Training-Tool

_Snapshot: 2026-07-13. For agents and humans starting a session._

Living companions (keep in sync when behavior changes):
- `docs/PASS1_PASS2_RALLY_BOUNDARIES.md` — boundary-logic source of truth (has a maintenance contract)
- `docs/PASS1_PASS2_SPEC.md` — broader pipeline spec
- `docs/RALLY_SYSTEM.md` — Rally Analysis & Coaching design/build prompts (as-built is below)
- `docs/BALL_DETECTION.md` / `docs/BALL_TRACKER_INTEGRATION.md` — ball-track refs
- `docs/HW_PROFILES.md` — `HW_PROFILE` operator docs (speed knobs per machine)
- `docs/SESSION_HANDOFF_2026-07-10.md` — detailed prior-session notes (benchmark tables, file-level diffs)

---

## What this is

Local pickleball video-analysis app: detect rallies (serve → terminal unplayable frame)
in fixed-camera footage, label ground truth, compare, explain discrepancies via Anthropic,
compound lessons into global params, and (optionally) produce pose-based coaching feedback.

| Layer | Stack |
|---|---|
| Backend | Python 3.12, FastAPI `:8420`, OpenCV, PyTorch, Ultralytics YOLO11-pose + ByteTrack, SQLite |
| Frontend | React + Vite `:5173` — overlays from metadata only (never rewrites video) |
| LLM | Anthropic — discrepancy report, in-app Assistant, **and coaching narrative**; never in the per-frame CV loop |

UI tabs: **Setup** → **Pass 1 rallies** → **Pass 2 detection** → **Rally Analysis**.

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
            ↓ (chained job kind="contacts")
         contact_detect → contact_events                             ← pose wrist-speed peaks
            ↓ (UI: Generate report)
         coaching → coaching_reports                                 ← flags + 1 Anthropic narrative
```

**Do not conflate “Pass 2”:**
- `cv/pass2.py` = detection/visualization only; emits **no** rally boundaries.
- `backend/app/refine/` = post-Pass-2 keep/drop/start-later/end-sooner; JSON-only today.
- `backend/app/rallyanalysis/` = coaching layer; **read-only** vs CV/refine/labels/overlays;
  writes only `contact_events` and `coaching_reports`.

Key end rules (`cv/rally_logic.py` / refine): end = terminal event frame, not “ball stopped”;
low speed alone never ends a rally (protects dinks); air-out ≠ bounce-out.

---

## Rally Analysis & Coaching

Built package under `backend/app/rallyanalysis/` + tab **Rally Analysis**.
Separate from the discrepancy feedback report (`analysis/reporter.py`).

### Flow
1. Pass 2 finishes → pipeline chains a `contacts` job (must be its own job so Pass 2 is already `done` for the readiness gate).
2. Contact detection: adaptive wrist-speed peaks from Pass 2 pose overlays (not ball/bounce). Min gap ≥0.35s; serve flag via formation. Also available via `POST .../detect-contacts`.
3. User clicks **Generate report** → feature vectors → category ratios + worst/best shots → **one** Anthropic call (`ANTHROPIC_COACHING_MODEL`, default Sonnet — deliberately heavier than the discrepancy report's `ANTHROPIC_REPORT_MODEL`) → persist `coaching_reports` (`tips_json` + `themes_json`).
   The single call also receives `session_context` (per-player whole-session flag counts, from every scored shot, not just the ≤`MAX_TIPS` shown) and returns one per-player session **theme** alongside the per-shot tips. At the LLM boundary only, `contact_height_m` is renamed `distance_from_net_m` (see caveat below) — DB/UI keep the internal name.
   **Vision context** (`rallyanalysis/keyframes.py`): the same call carries JPEG crops of the hitting player (prep/contact/follow-through, cropped via overlay boxes, ≤512px) for each selected tip candidate only — extracted fresh from `videos.path`, never persisted. Frames are the model's primary evidence; the pose metrics corroborate. Missing/unopenable video degrades to the frameless call (frames are an upgrade, never a dependency).

### Categories / flags
Per contact: knee bend, shoulder rotation, elbow extension, weight transfer, contact-height proxy, follow-through, split-step delay.  
Buckets: `footwork`, `paddle_prep`, `contact_point`, `follow_through`.  
Worst/best are adaptive (deviation thresholds), not fixed top-N.  
**Caveat:** `contact_height_m` is `court_y - NET_Y` (ground-plane proxy), not true height — ranges are coach-tunable defaults.

### Player attribution + singles/doubles
Every tip carries `player_id`/`player_label` (attached in code from `contact_events.player_id`, never trusted from
the LLM). Tip budget (`MAX_TIPS`) splits as evenly as practical across the distinct players present.
`videos.match_type` (`'singles'|'doubles'|NULL`, set via `PATCH /videos/{id}`) drives a stopgap in
`coaching.merge_singles_players`: on a singles clip with >2 distinct `contact_events` player ids, extras are folded
into whichever of the top-2 (by contact count) they're nearest to in time — in-memory only, not written back to
`contact_events`. Doesn't fix the underlying identity mis-tracking (`cv/identity.py` / pose tracking assigning more
identities than real players) — that's separate CV-tracking work.
Shot-side/length metrics (`shot_metrics.py`) and position/highlights (`position_metrics.py`/`highlights.py`) have
been removed from this feature — coaching tips are now the only Rally Analysis output.

### APIs (`/api` prefix)

| Method | Path | Behavior |
|---|---|---|
| GET | `/rally-analysis/health` | Liveness |
| POST | `/rally-analysis/{id}/detect-contacts` | Run + replace `contact_events` |
| GET | `/rally-analysis/{id}/contact-events` | List contacts |
| GET | `/rally-analysis/{id}/readiness` | Per-rally Pass 2 readiness |
| POST | `/rally-analysis/{id}/coaching-report` | Generate + persist (1 Anthropic call); returns `{video_id, tips, players, player_themes}` |
| GET | `/rally-analysis/{id}/coaching-report` | Cached report or `{report: null}` |
| PATCH | `/videos/{id}` | Set per-clip metadata (currently `match_type`) |

UI: `frontend/src/RallyAnalysis/RallyAnalysisPanel.jsx` — singles/doubles toggle, Generate/Regenerate, a
dedicated pin-on-video coaching view (own `<video>`, scrubber with per-tip pins, fixed feedback panel below,
player filter tabs). Readiness + detect-contacts are API-only today (contacts rely on the Pass 2 chain).

### DB
- `contact_events` — per-swing rows (frame, player, team, court_xy, shot_side, speed peak, serve flag, …)
- `coaching_reports` — one row per `video_id` (JSON tips: contact_id/frame/timestamp_s/category/tip/tone/player_id/player_label)
- `videos.match_type` — nullable `'singles'|'doubles'`, feeds the singles player-merge stopgap above

---

## Git / working tree (this snapshot)

| | |
|---|---|
| Branch | `gpuConfig` |
| Recent HEAD trail | `0f44205` gpu toggle; `52dbf11` merged ball + rally segmenting + rally analysis; identity / readiness / coaching wiring on `rally-analysis-features` |
| Doc layout | Root markdown companions moved under `docs/` (uncommitted rename on this branch) |

Recent landed work (conceptually):
1. Rally Analysis & Coaching (contacts after Pass 2, metrics tab, coaching report)
2. HW profiles (`HW_PROFILE` + `hw_profiles.json` / `docs/HW_PROFILES.md`)
3. Refine module + serve-gate + keep/drop grading + recommend sweep
4. Ballistic 3D + Pass 1 gap-tolerant net-crossing (`p1_cross_window_s=3.0`)
5. Ball GT tooling, TrackNet eval, OpenVINO pose path, Pass 2 ball toggle
6. `model_registry` + `api/models.py` (shadow/live promote for `rally_end` / `serve_start` / `tracknet`)

---

## Env / hardware (this machine)

Per-GPU speed knobs (device, precision, OpenVINO, TrackNet, Pass 1 subsampling,
Pass 2 ball) bundle into one `HW_PROFILE` name — see `docs/HW_PROFILES.md`,
`backend/app/hw_profiles.json` (add a profile there for a new machine, no code change),
and `backend/app/hw_profile.py` (schema). This is a **speed** setting, not for
cross-machine reproducibility. **Explicit env always wins over profile.**
Unset `HW_PROFILE` keeps the plain `.env` knobs working unchanged.

Current profiles: `cuda` (NVIDIA — ball on) · `arc140v` (Intel Arc — OpenVINO on, ball off).

From `.env` / config (do not commit secrets):

| Knob | Role |
|---|---|
| `HW_PROFILE` | e.g. `cuda` — see HW_PROFILES.md |
| `CV_DEVICE` | `auto` \| `cuda` \| `xpu` \| `cpu` |
| `PASS2_BALL` | `1` needed for refine/serve gate ball path; profile may enable |
| `TRACKNET_ENABLED` / `TRACKNET_WEIGHTS` | Opt-in; tennis weights need pickleball fine-tune |
| `POSE_MODEL` | Default `yolo11n-pose.pt` |
| `ANTHROPIC_API_KEY` | Report + Assistant + coaching narrative |
| `ANTHROPIC_MODEL` | Heavier / chat-oriented default |
| `ANTHROPIC_REPORT_MODEL` | Discrepancy + coaching narrative (default Haiku) |

Other facts:
- TrackNet weights live under `data/models/`; classical fallback if disabled/missing.
- Refine override YAML `data/refine/config.override.yaml` — optional (hand-off once used `serve_gate_min: 0.35`).
- SQLite: `data/app.db`. Some Pass 2 overlays may be stale vs current Pass 1 windows.
- Job kinds: `pass1` \| `pass2` \| `contacts` (contacts chained after Pass 2).

---

## What works vs what’s soft

**Solid**
- Upload → async Pass 1/2 → label state machine → court calibration → compare → LLM report → global `LearnedParams`
- Court homography / focal self-cal
- YOLO11-pose players + ByteTrack (stable identity slots; capacity-rejected dets not written to overlays)
- Pass 1 coarse gate + refine keep/drop grading loop
- Ballistic 3D + `serve_gate_min≈0.35` path (benchmark F1 ~0.70 on 7 labeled clips)
- Rally Analysis tab: contact detect after Pass 2, on-demand per-player coaching report (pin-on-video UI)

**Soft / gated**
- Ball track quality (far baseline, classical bursts; TrackNet tennis weights need pickleball fine-tune)
- BounceGRU (`TODO(fine-tune)` — trains via `scripts/train_bounce.py` when labels exist)
- Pass 1 learned gate classifier (`scripts/train_gate.py`)
- Coaching biomechanics ranges — unvalidated coach-tunable defaults; height is a plane proxy
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

Also backlog: grade real clips through keep/drop UI; cache ballistic recon for sweeps; truncate ballistic arcs at height=0; validate coaching ranges with a coach.

---

## Repo map (where to look)

```
backend/app/cv/             pass1.py, pass2.py, players.py, ball_*, court.py,
                            bounce.py, rally_logic.py, overlays.py, identity.py
backend/app/refine/         ballistic.py, serve_gate.py, classifiers.py, features.py,
                            postprocess.py, recommend.py, evaluate.py, config.py
backend/app/rallyanalysis/  contact_detect.py, coaching.py, readiness.py, contracts.py
backend/app/learning/       params.py (versioned LearnedParams), updates.py, …
backend/app/analysis/       metrics.py, reporter.py   ← discrepancy report (≠ coaching)
backend/app/api/            media.py, annotate.py, analyze.py, chat.py,
                            rally_analysis.py, models.py
backend/app/hw_profile.py + hw_profiles.json
backend/scripts/            train_bounce.py, train_gate.py, get_tracknet_weights.py, eval_*
frontend/src/               Workspace (tabs), VideoPlayer, Timeline, LabelingPanel,
                            CalibrationPanel, RallyAnalysis/*
data/                       app.db, videos/, models/, refine/   (gitignored payloads)
docs/                       CONTEXT, HW_PROFILES, RALLY_SYSTEM, boundary specs, …
```

Reproduce refine benchmark:
```powershell
cd backend
.\.venv\Scripts\python.exe -m app.refine.evaluate
```
Reprocess with ball (for serve gate): set `PASS2_BALL=1` or use `HW_PROFILE=cuda`.

---

## First actions for a new session

1. Read this file + `docs/PASS1_PASS2_RALLY_BOUNDARIES.md` § open end/split problem.
2. Confirm `.env` / `HW_PROFILE` (`PASS2_BALL`, `CV_DEVICE`, TrackNet path) matches the task.
3. If continuing end-detection: start in `refine/classifiers.py` + `features.py` + `postprocess.py`.
4. If continuing grading loop: use Pass tab 👍/👎 → recommendation → write override YAML.
5. If coaching: need Pass 2 done (contacts job) → Rally Analysis tab → Generate report; code in `rallyanalysis/`.
6. Update `docs/PASS1_PASS2_RALLY_BOUNDARIES.md` in the same change if you touch boundary logic.

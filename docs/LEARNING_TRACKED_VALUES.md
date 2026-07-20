# Reporting & Improvement — Tracked & Adjusted Values

This document lists **every value the reporting/self-improvement feature tracks (measures) and adjusts (learns)**, where each lives in code, how it changes, and what guards it.

The system has two halves:

- **Adjusted values** — the tunable state that makes accuracy *compound* across clips. Two forms: **learned parameters** (`learned_params` table, defined in [params.py](backend/app/learning/params.py)) and **learned models** (`model_registry` table, [model_registry.py](backend/app/learning/model_registry.py)).
- **Tracked values** — the measurements the system computes to decide *whether* to adjust and to show the "is it getting better?" trend (`param_eval`, `reports`, `discrepancies` tables).

---

## Part 1 — Adjusted values: learned parameters

These are the **27 global tunable parameters**. Every upload loads the latest version before Pass 1 runs, so all prior lessons apply to every future clip. Defaults + bounds are the single source of truth in [params.py](backend/app/learning/params.py) (`DEFAULTS` / `BOUNDS`). A param must be in **both** dicts or `apply_updates` drops it.

Each parameter is created as a **new version** (`learned_params` is append-only); values are always **clamped to bounds** before commit, no matter the source.

### 1a. Pass 1 coarse-gate parameters

| Parameter | Default | Bounds | What it controls |
|---|---|---|---|
| `p1_frame_step` | 3 | 1 – 6 | Sample every Nth frame in the coarse pass |
| `p1_scale` | 0.4 | 0.2 – 1.0 | Spatial downscale for motion/ball signals |
| `p1_motion_ratio` | 1.6 | 1.1 – 4.0 | Motion must exceed ambient baseline by this factor |
| `p1_motion_floor` | 0.12 | 0.03 – 2.0 | Absolute floor on the adaptive motion threshold |
| `p1_ball_recent_s` | 1.2 | 0.3 – 4.0 | Fast ball seen within this window = ball activity |
| `p1_ball_speed_px` | 3.5 | 1.0 – 12.0 | Min blob speed (px/frame @ p1_scale) to count as the ball |
| `p1_on_s` | 0.7 | 0.2 – 3.0 | Sustained activity needed to flip dead→active |
| `p1_off_s` | 2.0 | 0.5 – 6.0 | Sustained quiet needed to flip active→dead |
| `p1_pad_s` | 1.5 | 0.5 – 3.0 | Padding handed to Pass 2 around each window |
| `p1_merge_gap_s` | 0.5 | 0.0 – 2.0 | Merge adjacent Pass 1 windows only if raw gap ≤ this |
| `p1_cross_window_s` | 2.0 | 0.5 – 4.0 | Net-cross recency that keeps the calibrated gate open |
| `p1_min_rally_s` | 2.0 | 1.0 – 5.0 | Discard blips shorter than a real rally |
| `p1_spot_check_s` | 2.0 | 0.5 – 5.0 | Serve-formation pose spot-check interval |

### 1b. Pass 2 ball / bounce parameters

| Parameter | Default | Bounds | What it controls |
|---|---|---|---|
| `bounce_max_height_m` | 0.20 | 0.05 – 0.5 | Max reconstructed height for a bounce (height minima) |
| `hit_min_height_m` | 0.45 | 0.2 – 1.2 | Min height to classify a contact as a hit |
| `player_radius_m` | 1.3 | 0.5 – 2.5 | Player exclusion radius for ball association |
| `min_event_gap_s` | 0.12 | 0.05 – 0.5 | Minimum spacing between detected events |
| `net_fail_height_m` | 0.95 | 0.6 – 1.3 | Height threshold for a net-fail terminal event |
| `out_max_dist_m` | 1.5 | 0.3 – 4.0 | Max metres past a line an out-bounce may land (else = noise) |

### 1c. Rally start / end parameters

| Parameter | Default | Bounds | What it controls |
|---|---|---|---|
| `serve_speed_mps` | 4.0 | 1.5 – 10.0 | Min ball speed to qualify a serve |
| `serve_behind_margin_m` | 1.5 | 0.5 – 4.0 | How far behind the baseline the serve must originate |
| `serve_formation_lookback_s` | 4.0 | 1.0 – 8.0 | Lookback window for serve-formation confirmation |
| `confirm_window_s` | 2.5 | 1.0 – 6.0 | Window to confirm a candidate rally |
| `player_active_mps` | 1.2 | 0.4 – 3.0 | Player speed that counts as "active" |
| `p2_reject_passes` | 1.0 | 0.0 – 1.0 | 1 = drop single-net-cross carried/caught passes (not real serves) |

### 1d. Learned boundary offsets — the primary compounding knobs

| Parameter | Default | Bounds | What it controls |
|---|---|---|---|
| `start_offset_frames` | 0 | −90 – 90 | Systematic correction added to every Pass 2 rally **start** |
| `end_offset_frames` | 0 | −90 – 90 | Systematic correction added to every Pass 2 rally **end** |

> These two are the knobs the report adjusts most. Example: if rally ends land consistently 4 frames late, `end_offset_frames` moves to −4. They are **decoupled** (a clip with a clean start but messy ends still compounds the start offset).

---

## Part 2 — How each parameter is adjusted (the three paths)

The proposal engine is [updates.py](backend/app/learning/updates.py) (`compute_param_updates`). Every proposal is **risk-tagged**, and the risk decides the path.

### Path A — LOW risk: boundary offsets (auto-applied, harness-validated)
- **Which params:** `start_offset_frames`, `end_offset_frames`.
- **How computed:** median of the *clean* matched-pair boundary errors, **half-median damped** (`step = round(-median/2)`), then **capped at ±5 frames per report** (`_OFFSET_STEP_CAP`) so repeated reports converge instead of oscillating. Only "clean" pairs count — a start from a real detected serve (`clean_start`), an end from a confirmed terminal event (`clean_end`) — and only if the error is within ±45 frames (`_OFFSET_LEARN_CAP`); coarse-gate fallbacks are excluded. Errors under 3 frames (~100 ms @30fps) are ignored.
- **Auto-apply gate:** on label-save, [autolearn.py](backend/app/learning/autolearn.py) runs the **analytic harness** ([validate.py](backend/app/learning/validate.py) → `evaluate_offset_change`). The change commits globally **only if aggregate mean-abs error improves across ALL labeled clips** and match_rate doesn't drop (`_EPS_FRAMES = 0.5`, `_EPS_RATE = 0.01`). A non-targeted metric may not regress (a start change can't hurt end error). One noisy clip can never move the baseline alone.
- **Manual "Generate report" button:** applies the low-risk offsets directly (the on-save path additionally validates against the held-out set first).

### Path B — MEDIUM risk: Pass 1 gate changes (never auto-applied, human-gated)
- **Which params:** driven by signed rally-count pressure (`missed − extra`):
  - Net **≥ +2** (too many missed): `p1_motion_ratio` × 0.9 (loosen gate), `p1_off_s` × 1.15 (hold rallies together).
  - Net **≤ −2** (too many extra): `p1_motion_ratio` × 1.1 (tighten gate), `p1_min_rally_s` × 1.1 (drop short false positives).
  - No-ops when missed and extra roughly balance.
- **Gate:** queued in `pending_changes` for human approval. Approve/reject via `POST /api/pending_changes/{id}/approve|reject`. Approval applies it as a new version instantly (the human reviewing the rationale IS the gate; the value is already bounds-clamped). A fresh save **supersedes** any still-pending proposal for the same param.

### Path C — LLM suggestions (any param, human-accepted)
- The LLM report ([reporter.py](backend/app/analysis/reporter.py)) may return `param_suggestions` (`{param, value, rationale, scope}`). The LLM only *explains and suggests* — the deterministic math above already ran before it was called.
- Suggestions are filtered to known params and **clamped through `BOUNDS`** on apply. `global` scope suggestions the user accepts (`POST /api/reports/{id}/apply_suggestions`) become a new version; `clip_local` suggestions **never touch the global baseline**.

> **Routing rule (critical):** method-level feedback updates these shared global params. Clip-local oddities (bumped camera, one-frame ball fix) must not touch the baseline.

---

## Part 3 — Adjusted values: learned models

The weight-based analogue of learned params. Registry: [model_registry.py](backend/app/learning/model_registry.py), `model_registry` table. Append-only, `shadow → live → archived`. The live pipeline loads the latest `live` version per kind before Pass 2.

| Model kind | Status | Purpose |
|---|---|---|
| `rally_end` | Trainable now (BounceGRU) | Learn the terminal-event boundary — the weakest link (most ends fall back to coarse_gate) |
| `serve_start` | Reserved | Learn serve-start boundary (builder not yet shipped) |
| `tracknet` | Reserved / gated off | Ball tracker; pretrained weights don't transfer, gated by `TRACKNET_ENABLED` |

Per model the registry tracks: `version`, `kind`, `status`, `metrics` (JSON), `parent_version`, `trained_on`, `origin`, `created_at`, `promoted_at`, weights `path`.

**Promotion gate:** a model stays **shadow** until validated. `evaluate_model` ([validate.py](backend/app/learning/validate.py)) scores it via full pipeline re-run vs the heuristic incumbent; a `rally_end` model is accepted only if it lowers mean-abs **end** error without losing matches. **Cold-start guard:** models can't be validated/promoted until **≥3 labeled clips** exist (a real leave-one-clip-out split).

---

## Part 4 — Tracked values (measurements)

These are computed, stored, and surfaced — they drive the adjust decisions and the trend display. Comparison engine: [metrics.py](backend/app/analysis/metrics.py) (`compare`).

### 4a. Per-rally discrepancy fields (`discrepancies` table + report payload)
Each human-vs-AI matched pair or unmatched rally records:

- `type` — `boundary_drift` (found, boundaries off) | `missed` (human rally absent) | `extra` (AI invented)
- `temporal_iou` — temporal intersection-over-union of the pair
- `start_error`, `end_error` — frames (positive = AI late)
- `start_error_ms`, `end_error_ms` — same in milliseconds
- `start_source`, `end_source` — where the boundary came from (`serve` / `terminal_event` / `coarse_gate` …)
- `confirmed` — terminal event was confirmed
- `clean_start`, `clean_end` — boundary came from a real detection, not the coarse-gate fallback (these decide if the pair is a trustworthy learning signal)
- `feature_trace` — per-frame signal samples (±2 s, 20 samples) around each disputed boundary; attached to `boundary_drift` only. Fields: `f, ball, world, pspeed, evt, obs`.

### 4b. Per-clip summary metrics (`reports.summary` + `compare` output)

| Metric | Meaning |
|---|---|
| `human_rallies` / `ai_rallies` | Counts labeled vs detected |
| `matched` / `missed` / `extra` | Greedy-IoU match outcome |
| `match_rate` | matched ÷ human rallies |
| `mean_iou` | Mean temporal IoU over matched pairs |
| `mean_start_error_frames` / `mean_end_error_frames` | Signed mean boundary error |
| `mean_abs_start_error_frames` / `mean_abs_end_error_frames` | **Absolute** error (the honest "how far off" — signed means cancel out) |
| `median_start_error_frames` / `median_end_error_frames` | Median boundary error |
| `fps` | Clip frame rate (for frame↔ms conversion) |

### 4c. Scoreboard trend (`param_eval` table → `GET /api/learning/scoreboard`)
One snapshot per committed param version — the "is it actually getting better?" trend, pooled across ALL labeled clips (raw per-pair errors pooled, not an average of per-clip averages):

- `version`, `scope` (`baseline` / `auto:label` / …), `origin`
- `mean_abs_start_err`, `mean_abs_end_err` — pooled boundary error in frames
- `match_rate`
- `n_clips`, `n_human`, `n_matched`, `n_extra`
- `applied` — committed (1) vs rejected candidate (0); rejected candidates are kept visible so a refusal is auditable
- `created_at`

Frontend: `ScoreboardPanel.jsx`.

---

## Part 5 — Where to see / trigger each thing

| Action | Endpoint / location |
|---|---|
| View current params + version history | `GET /api/params` |
| Generate discrepancy report (applies low-risk offsets) | `POST /api/videos/{id}/report` |
| List past reports | `GET /api/videos/{id}/reports` |
| Accept LLM param suggestions | `POST /api/reports/{id}/apply_suggestions` |
| List gate changes awaiting approval | `GET /api/pending_changes` |
| Approve / reject a gate change | `POST /api/pending_changes/{id}/approve` \| `/reject` |
| Accuracy trend over versions | `GET /api/learning/scoreboard` |
| Model versions | `GET /api/models*` |
| Detection diagnostics for a clip | `GET /api/videos/{id}/detect_debug` |
| Autonomous learn on label-save | [autolearn.py](backend/app/learning/autolearn.py) `run_learn` |

### Cost / model config (from [config.py](backend/app/config.py))
The learning loop is **token-free** except the LLM report. Report model: `ANTHROPIC_REPORT_MODEL` (default `claude-haiku-4-5-20251001`), `max_tokens` 6000, thinking budget capped at 2000. The deterministic param math runs regardless of the LLM.

---

### Summary count
- **27** learned parameters adjusted (13 Pass 1 + 6 Pass 2 + 6 rally start/end + 2 boundary offsets)
- **2** auto-applied (harness-gated): `start_offset_frames`, `end_offset_frames`
- **4** deterministic human-gated: `p1_motion_ratio`, `p1_off_s`, `p1_min_rally_s` (+ combinations)
- **Any** param adjustable via LLM suggestion (clamped, human-accepted)
- **3** model kinds registered (`rally_end` trainable; `serve_start`, `tracknet` reserved)
- **~20+** per-rally and per-clip metrics tracked to drive and display all of the above

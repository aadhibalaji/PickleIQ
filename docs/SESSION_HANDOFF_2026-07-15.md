# Session Handoff — Labeled-eval workflow for the Pass 2 serve gate (2026-07-15)

Handoff for a fresh agent. Covers building a real labeled-evaluation loop to
replace one-off threshold patching, a data-corruption incident and full reset
encountered along the way, and the resulting (non-)conclusion: **current
`config.override.yaml` settings are validated against a holdout clip and
should not be changed based on anything found this session.**

No runtime pipeline behavior changed (no thresholds were applied from any
sweep) — **`PASS1_PASS2_RALLY_BOUNDARIES.md` does not need updating.** Only
`evaluate.py`, `recommend.py`, `refine.py` (sweep infra) and `data/refine/`
(labels + config) changed.

---

## TL;DR

1. **Built out the labeled-eval loop**: extended both sweep grids
   (`evaluate.py` DEFAULT_GRID, `recommend.py` DEFAULT_GRID) to cover every
   threshold hand-patched in recent sessions (contact floors, pre_serve
   floors, `min_rally_s_early`, `serve_gate_min`).
2. **Fixed a real perf bug**: neither sweep cached Stage A (ballistic 3D
   tracking reconstruction, ~5s/clip) across grid combos — a full sweep would
   have taken ~13 hours. Added `refine_video_fast()` / `_stage_a_windows()`
   caching in `refine.py` (verified byte-identical output to the original).
   Sweeps now run in single-digit minutes.
3. **Discovered `app.db` corruption** in 2 clips' `overlays` rows (binary
   garbage spliced into stored JSON) — root cause per the user: an earlier
   agent session deleted "duplicate" clips by filename instead of by the
   upload timestamps given, then a botched restore corrupted the data.
   `app.db.pre_recovery*` / `recovered_*` files were leftover evidence of that
   incident.
4. **Full reset at the user's request**: wiped every video-derived DB table
   and the video files on disk, keeping only `calibrations`, `learned_params`,
   `model_registry` (see "The reset" below for the exact list).
5. **Re-labeled from scratch**: 5 fresh, genuinely distinct clips — 4 for
   tuning, 1 held out (`doubles #4.mp4`) from the very start (added to
   `holdout_clips` before it was ever labeled).
6. **Fixed a second real bug**: `recommend.py` had **no holdout awareness at
   all** (unlike `evaluate.py`) — it pooled every clip's `window_grades`
   unconditionally, so a graded holdout clip would have silently leaked into
   serve-gate tuning. Added `include_holdout` param + `is_holdout()`
   filtering, matching `evaluate.py`'s existing pattern.
7. **Ran both sweeps on the 4 tuning clips** — both "best" combos beat current
   settings by only noise-level margins (F1 +0.03, grade-accuracy +0.006).
   **Decision: did not apply either.**
8. **Holdout check confirmed the decision was right**: F1 and window-grade
   accuracy on the holdout clip are consistent with the tuning pool (no
   cliff, no overfitting collapse). Current `config.override.yaml` is
   validated as-is.

---

## Key lesson (read this before trusting any future sweep number)

The very first sweep this session ran against 2 "clips" that turned out to be
**duplicate uploads of the same source video** (`Second 3 mins.mp4` uploaded
three times under different `video_id`s). That sweep reported F1 **0.757** —
which evaporated to **~0.48–0.52** once real clip diversity (4-5 genuinely
different videos) was used. **Always check clip diversity (distinct source
content, not just distinct `video_id`/row count) before trusting a sweep
result.** `clips_with_human_labels()` counts rows, not distinct content.

---

## Current benchmark (5 labeled clips, current live config — NOT changed)

| video_id | filename | human labels | pass1 windows | window_grades | role |
|---|---|---|---|---|---|
| `c75ffab0f1b9481584f12932475e63f0` | Second 3 mins.mp4 | 11 | 16 | 16 | tuning |
| `3af6aa7789f14bcf8c53941028e5c274` | clip #1.mp4 | 5 | 6 | 6 | tuning |
| `9d492e1c23b64926918ac7aac43e1c78` | doubles #2.mp4 | 9 | 12 | 12 | tuning |
| `9703fdeb994b49198f648bae382188b7` | doubles #3.mp4 | 9 | 10 | 10 | tuning |
| `e8c401c1dcbd4aa7b8764a1ec8d4f6b4` | doubles #4.mp4 | 11 | 14 | 13 | **holdout** |

Current live config = defaults in `config.py` + `data/refine/config.override.yaml`
(`serve_gate_min: 0.15` — this was already applied from a prior session, not
set by this one; `holdout_clips: [e8c401c1...]` added this session).

**Rally-span F1** (`evaluate.py`, `evaluate_clip`/`evaluate`):

| scope | tp | fp | fn | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 4-clip tuning pool | 15 | 13 | 19 | 0.536 | 0.441 | **0.484** |
| holdout clip alone | 7 | 4 | 4 | 0.636 | 0.636 | **0.636** |
| all 5 pooled (incl. holdout) | 22 | 17 | 23 | 0.564 | 0.489 | **0.524** |

**Window-grade keep/drop accuracy** (`recommend.py`, `score_against_grades`):

| scope | n | tp | fp | fn | tn | accuracy | F1 |
|---|---|---|---|---|---|---|---|
| 4-clip tuning (holdout excluded) | 44 | 25 | 3 | 7 | 9 | 0.773 | 0.833 |
| all 5 (holdout included) | 57 | 33 | 4 | 9 | 11 | 0.772 | 0.835 |

The ~0.001 accuracy delta between tuning-only and holdout-included is about as
clean a generalization signal as you'll get — **not overfit**.

### Sweep results (both within noise — neither applied)

F1 sweep (`evaluate.py --sweep`, 2,304 combos, grid below) best combo:
`enter_thresh=0.6, exit_thresh=0.25, live_ratio=0.7, merge_gap_s=3.0,
min_rally_s=1.5, min_rally_s_early=0.75, serve_gate_min∈[0.25,0.4] (tied)`
→ F1 0.516 (**+0.032** over current 0.484, on 4 tuning clips).

Grade-accuracy sweep (`recommend.py`, 486 combos) best combo:
`serve_contact_min=0.25, serve_contact_near_floor_min=0.2, serve_gate_min=0.4,
serve_pre_serve_compensate_min=0.45, serve_pre_serve_min=0.3`
→ accuracy 0.773 (tied with current), F1 0.839 (**+0.006** over current 0.833).

**Neither gain is worth the risk of overfitting to a 4-5 clip label set.**
Current `config.override.yaml` was left unchanged.

---

## Code changes this session (all still in the working tree)

### `backend/app/refine/refine.py`
- Added `_STAGE_A_CACHE` (module-level dict) + `_stage_a_windows(video_id, cfg,
  p1_source)`: caches Pass-1-windows + their ballistic-reconstructed
  `TrackingData` per `(video_id, use_ballistic_3d, ballistic_max_rms_px,
  window_margin_s, p1_source)`. Stage A doesn't depend on Stage C/D/gate
  knobs, so sweeps over those can reuse it instead of re-reconstructing
  (~5s/clip) on every grid point.
- Added `refine_video_fast(video_id, cfg, classifier, p1_source="pass1")` —
  like `refine_video` but uses the cache. **Verified byte-identical records**
  vs. `refine_video` before trusting any sweep built on it.
- `ponytail:` cache is unbounded/process-lifetime — fine for sweep runs
  (a handful of clips), would leak if pointed at many distinct videos long-term.

### `backend/app/refine/evaluate.py`
- `DEFAULT_GRID` gained `min_rally_s_early: [0.75, 1.0, 1.25, 1.5]`.
- `evaluate_clip()` / `evaluate()` gained a `fast: bool = False` param; when
  `True`, uses `refine_video_fast` instead of `refine_video`. Only `sweep()`
  passes `fast=True` — normal (non-sweep) callers are unaffected.

### `backend/app/refine/recommend.py`
- `DEFAULT_GRID` gained `serve_contact_min`, `serve_contact_near_floor_min`,
  `serve_pre_serve_min`, `serve_pre_serve_compensate_min` (the serve-gate
  component floors hand-patched in prior sessions, now sweepable).
- Switched from `refine_video` to `refine_video_fast` in
  `score_against_grades`.
- `recommend()`'s `"current"` dict now reports **all** swept params' current
  values (was hardcoded to just `serve_gate_min`).
- **Added `include_holdout: bool = False` param to `recommend()`**, filtering
  `grades_by_clip` through `is_holdout()` (imported from `.train`) before
  scoring — this module previously had **no holdout concept at all** and
  pooled every graded clip unconditionally. This is a real bug fix, not just
  an addition: without it, grading a holdout clip's keep/drop calls would
  silently contaminate serve-gate tuning. `grade_recommendation` in
  `annotate.py` inherits the fix for free (calls `recommend()` with defaults).

### `data/refine/config.override.yaml`
- Unchanged value: `serve_gate_min: 0.15` (pre-existing from an earlier
  session — not set or validated by this session beyond confirming it holds
  up on holdout).
- Added: `holdout_clips: [e8c401c1dcbd4aa7b8764a1ec8d4f6b4]` (doubles #4.mp4).

---

## The reset (why there are only 5 clips, not more)

Mid-session, 2 of the 4 then-current clips' `overlays` rows were found
corrupted (raw binary/garbage bytes spliced into stored per-frame JSON,
breaking `json.loads`/`UnicodeDecodeError` on read). Root cause per the user:
an earlier agent session deleted duplicate video uploads by filename instead
of by the upload timestamps it was given, then a restore attempt went wrong.
`data/app.db.pre_recovery(-shm/-wal)`, `data/recovered_3vids.sql`, and
`data/recovered_counts.json` were leftover evidence of that recovery attempt.

At the user's explicit request, this session did a **full reset**:
- Cleared every video-derived table: `videos`, `jobs`, `rally_labels`,
  `overlays`, `window_grades`, `ball_corrections`, `ball_tracks`,
  `contact_events`, `audio_events`, `discrepancies`, `reports`,
  `coaching_reports`, `user_notes`, `pending_changes`, `param_eval`,
  `ball_eval`.
- **Kept**: `calibrations`, `learned_params`, `model_registry` (global
  setup/tuning state, not video-specific).
- Deleted all files under `data/videos/` (~1.8GB) and all per-video artifacts
  under `data/refine/` except `config.override.yaml`.
- Deleted the pre_recovery backup files listed above.
- Also found (and removed) one orphaned video folder
  (`data/videos/760f1e0e95d048fc9ec20ce4c61ccd35`) with no matching `videos`
  row — another leftover of the same incident.

The 5 clips in the benchmark table above were uploaded and labeled fresh
after this reset. All 5 were verified to load cleanly (no corruption) before
any sweep trusted them.

---

## Schema notes worth knowing (a fresh agent will otherwise rediscover these)

- **`rally_labels source='human'` is positive-only ground truth** (real,
  precisely-trimmed rally spans) — it has no "confirmed not a rally" concept.
  Do not insert wide/pass1-window-boundary spans here as a stand-in for a
  keep/drop call, and do not insert a second GT span for a rally that already
  has one (it will double-count in `match_segments`).
- **`window_grades` is the correct table for pass1-window-level keep/drop
  verdicts** (`pass1_start_frame`/`pass1_end_frame`, `human_verdict:
  'keep'|'drop'`, plus an `ai_verdict`/`ai_rejection_reason`/`serve_gate_score`
  snapshot for context). It's matched to current Pass 1 windows by IoU
  (`MATCH_IOU=0.5`), not by `pass1_window_id` (re-minted every Pass 1 run).
- **Two independent sweep tools exist and test different things**:
  `evaluate.py`'s F1 sweep (Stage C/D knobs, needs precise rally-span GT,
  measures boundary accuracy too) vs. `recommend.py`'s grade-accuracy sweep
  (serve-gate-only knobs, needs only keep/drop grades, no boundary signal).
  Both are needed; neither substitutes for the other.
- **`config.override.yaml` is the only thing Pass 2 actually reads at
  runtime** (`pipeline.py` → `load_config(override_path())`). Sweep/recommend
  functions are pure measurement — nothing applies automatically. The
  `POST /refine/config/override` endpoint (`RefineOverrideIn` model in
  `annotate.py`) is the only writable surface, and **it currently only
  accepts `serve_gate_min` and `mean_p_live_min`** — it would need extending
  before any of the newly-swept knobs (`enter_thresh`, `min_rally_s`,
  `serve_contact_min`, etc.) could be applied through the API rather than by
  hand-editing the YAML.

---

## Recommended next steps

1. **Do not apply either sweep's "best" combo.** Both are within noise of
   current settings on a 4-5 clip label set; the holdout check already
   confirms current settings generalize reasonably. Chasing a 0.03 F1 or
   0.006 accuracy bump here is more likely to overfit than to help.
2. **Threshold tuning (Stage C/D/gate) has plateaued** at ~48-52% rally F1
   and ~77% window-grade accuracy. More grid search over the same knobs is
   very unlikely to move these numbers further — this is the actual finding
   of the session, not a null result to brush past.
3. **The next real lever is Stage A/B**: tracking/feature quality
   (`features.py`, `ballistic.py`) or the rules classifier
   (`classifiers.py`), not the postprocessing thresholds swept this session.
   Recommended concrete next step: pull up the current-config false
   positives/negatives across the 5 clips and go through each one's
   per-record `diagnostics` dict (already computed in `refine.py`'s
   `refine_window`, surfaced in the grading UI's "why" expander) to find
   whether each miss is a genuine feature/classifier limitation vs. a
   threshold that's still slightly off for that specific case.
4. **Keep growing the labeled set** as more clips get uploaded — the
   workflow now supports it cleanly: label rally spans + keep/drop on every
   new clip, decide holdout-or-not up front (add to `holdout_clips` in
   `config.override.yaml` *before* labeling it, not after), then re-run:
   ```
   cd backend
   ./.venv/Scripts/python.exe -m app.refine.evaluate --sweep
   # and, in Python:
   from app.refine.config import load_config, override_path
   from app.refine.recommend import recommend, DEFAULT_GRID
   from app.api.annotate import _grades_by_clip
   cfg = load_config(override_path())
   recommend(cfg, _grades_by_clip(None), DEFAULT_GRID)
   ```
   Both are fast now (single-digit minutes) thanks to the Stage A cache.
5. Before trusting any future sweep number: check the clip set is genuinely
   diverse (see "Key lesson" above), and always run the holdout check
   (`evaluate(..., include_holdout=True)` / `recommend(..., include_holdout=True)`)
   before deciding a result is real.

## Key files
- `backend/app/refine/evaluate.py` — rally-span F1 harness + sweep (`DEFAULT_GRID`, `sweep()`, `evaluate()`)
- `backend/app/refine/recommend.py` — window-grade accuracy sweep (`DEFAULT_GRID`, `recommend()`, holdout-aware now)
- `backend/app/refine/refine.py` — `refine_video_fast` / `_stage_a_windows` (the caching fix)
- `backend/app/refine/train.py` — `is_holdout()` (id-or-filename match against `cfg.holdout_clips`)
- `backend/app/refine/labels.py` — `load_ground_truth()` (rally_labels source='human' → GT spans)
- `backend/app/refine/config.py` — `RefineConfig`, `override_path()`, all tunable knobs
- `backend/app/api/annotate.py` — grading UI endpoints (`window_grades`, `grade_recommendation`, `refine/config/override`), `_grades_by_clip()`
- `data/refine/config.override.yaml` — the only file Pass 2 reads at runtime; currently `serve_gate_min: 0.15` + `holdout_clips`

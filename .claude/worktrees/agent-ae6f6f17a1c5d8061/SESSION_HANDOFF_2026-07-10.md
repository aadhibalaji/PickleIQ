# Session Handoff — Pass 2 keep/drop + Pass 1 crossing/segmentation (2026-07-10)

Handoff for a fresh agent. Covers a long session of work on the pickleball rally
pipeline. **Nothing is committed** — all changes are in the working tree. Read
alongside [PASS1_PASS2_RALLY_BOUNDARIES.md](PASS1_PASS2_RALLY_BOUNDARIES.md)
(the living boundary-logic doc, updated this session: §2.4b/2.4c) and
`PASS1_PASS2_SPEC.md` (broader pipeline spec).

> **Maintenance:** if you change any behavior described here, update
> `PASS1_PASS2_RALLY_BOUNDARIES.md` too (it carries a maintenance contract).

---

## TL;DR of what was accomplished

1. **Built a user-facing keep/drop grading loop** for Pass 2 (grade each Pass 1
   window keep/drop → live accuracy + serve-gate recommendation). Fully built &
   verified; **not yet used with real grades** — the user will grade clips next.
2. **Diagnosed & fixed the `no_serve` false-drop** root cause: refine projected
   the ball to court coords assuming height=0, so an **airborne serve ball** landed
   metres off-court and every ball check failed. Fixed with **ballistic 3D**
   (gravity-arc fit to the pixel track) + **arc contact recovery**. Labeled
   benchmark **F1 0.561 → 0.673**.
3. **Fixed `rejection_acc`** by raising `serve_gate_min` 0.15 → **0.35** (the
   ballistic work created the headroom). Best F1 (0.697), rejection restored.
4. **Improved Pass 1 net-crossing detection** (gap-tolerant, once-per-transition)
   — but hit a **fundamental recall-vs-segmentation tradeoff**. Persisted
   `p1_cross_window_s=3.0` (FN=0%) but it **merges adjacent rallies**.
5. **Found the key open problem:** Pass 2 does **NOT split merged windows**, and
   rally-END detection is weak — **both are the same capability** (detecting when
   live play stops). **Recommended next work: player-motion-based dead/end
   detection** — fixes both at once.

---

## Benchmark (7 labeled clips, refine keep/drop, `python -m app.refine.evaluate`)

| stage | F1 | precision | recall | rejection_acc |
|---|---|---|---|---|
| ground-projection baseline | 0.561 | 0.625 | 0.508 | 0.571 |
| + ballistic 3D | 0.643 | 0.679 | 0.610 | 0.429 |
| + arc contact recovery | 0.673 | 0.704 | 0.644 | 0.429 |
| **+ serve_gate_min 0.35 (shipped)** | **0.697** | 0.760 | 0.644 | **0.571** |

`serve_gate_min` sweep (ballistic on): 0.35 is the sweet spot (best F1, rejection
restored, no recall loss). 0.40 over-drops (recall falls). `mean_p_live_min` and
`min_rally_s` were swept — raising them **did not help** (left at 0.25 / 2.5).

---

## Changes by area (all uncommitted, in working tree)

### A. Keep/drop grading loop — BUILT & VERIFIED (30/30 refine tests, frontend builds)
- **`backend/app/db.py`** — new `window_grades` table (keyed on frame span, not the
  re-minted `pass1_window_id`; re-associated by IoU).
- **`backend/app/api/annotate.py`** — `get_refined` rewritten to return ALL windows
  (kept + dropped) with verdict/reason/serve_gate_score, self-heals the refine JSON
  (`?regenerate=1`); new endpoints: `window_grades` POST/GET/DELETE,
  `grade_recommendation`, `refine/config/override`.
- **`backend/app/refine/recommend.py`** — NEW. Grade-scored serve-gate sweep
  (scores `serve_gate_min` candidates against user grades, not GT rally spans).
- **`backend/app/refine/config.py`** — `override_path()` → `data/refine/config.override.yaml`.
- **`backend/app/pipeline.py`** — `_refine_after_pass2` reads `load_config(override_path())`.
- **Frontend** — `api.js` (grade + recommendation methods), `Workspace.jsx`
  (`PassTab` grading table: 👍/👎 per window, live accuracy readout,
  `Recommendation` panel), `Timeline.jsx` (per-span verdict/grade coloring).
- **How it improves:** measure + recommend only, no auto-apply. Grade windows →
  live keep/drop accuracy + a recommended `serve_gate_min` the user applies (writes
  the override YAML). The `dropped_wrong by reason` count is the headline metric.

### B. Ballistic 3D + serve-gate fixes — SHIPPED (default ON)
- **`backend/app/refine/ballistic.py`** — NEW. Fits a gravity arc
  `P0 + V0 t + ½ g t²` to each ball flight so it projects onto the pixel track
  (reproj RMS ~3px), overwrites the ground-plane `ball_xy`/`ball_height` on
  well-fitting segments. `Flight` dataclass stores per-arc params on
  `TrackingData.ball_flights`. Config: `use_ballistic_3d=True`, `ballistic_max_rms_px=12.0`.
- **`backend/app/refine/serve_gate.py`** — `_ballistic_contact()`: extrapolates the
  serve arc BACKWARD to the untracked strike (ball is invisible at contact — fused
  with the body); scores origin-low + behind-baseline; re-anchors flight+landing.
  Also an off-by-default image-space contact path (`serve_contact_use_pixels=False`,
  superseded by ballistic — leave off).
- **`backend/app/refine/io_loaders.py`** — `TrackingData` now carries `ball_px`,
  `player_box`, `ball_flights`; `load_tracking(reconstruct_3d=...)` runs ballistic.
- **Verified:** the example serve (clip `e29bcd56` "Video Project 4", window
  ~3786-4119) went 0.21 → 0.705, contact 0.00 → 0.73.
- **Caveats:** `rejection_acc` dipped (stronger ball signal keeps a few more junk
  windows — the 0.35 threshold restores it). ~5s/clip slower (disable for big
  sweeps — future: cache reconstruction, it's cfg-independent). Arc can over-extend
  slightly past an undetected bounce (minor; truncate at height=0 crossing to fix).

### C. Rejected experiments (kept in tree, OFF)
- **`serve_contact_use_pixels`** (image-space contact) — regressed benchmark; a
  perspective coincidence (airborne mid-court ball overlaps a far player in 2D).
- **`ball_seed_on_player`** (`cv/ball_track.py`, `cv/pass2.py`) — tried re-enabling
  ball detection in the player deadzone to catch the serve strike; **no effect**
  (ball genuinely untrackable at the strike — fused with body). Left off.

### D. Pass 1 crossing detection — IN TREE, tradeoff unresolved
- **`backend/app/cv/pass1.py`** — replaced the consecutive-frame net-crossing rule
  with a **gap-tolerant, once-per-transition** detector (tracks the ball's current
  clear net side with a net-line margin `p1_net_margin_m=0.5`, fires one crossing
  per genuine side change within `p1_cross_max_gap_s=1.0`). Plus a **decoupled end**:
  `close_rally` trims trailing crossings isolated from the rally rhythm
  (`p1_cross_cluster_gap_s=2.0`).
- **Persisted `p1_cross_window_s=3.0`** (learned_params **v15**, was 2.0) to sustain
  the gate across sparse long-rally crossings → FN 0%.
- **`p1_pad_s=1.5` unchanged** — start leeway is well-calibrated (median 0.77s,
  *less* than the pad; do not shrink).

Pass 1 variant results (vs human GT, 7 clips):

| variant | false negatives | end p90 | note |
|---|---|---|---|
| baseline (consecutive) | 3% | 4.76s | original |
| transition, cw=2.0 | 5% | **4.27s** | best ends |
| **transition, cw=3.0 (shipped)** | **0%** | 8.25s | best recall, MERGES rallies |

---

## THE OPEN PROBLEM (recommended next work)

**Rally-end detection and merged-window splitting are the same unsolved capability:
detecting when live play stops.**

- `p1_cross_window_s=3.0` gives FN=0% but **merges** 4 windows (8 rallies) — e.g.
  First 3 mins window 3069-3768 spans two human rallies (3072-3238) & (3405-3710).
- **Pass 2/refine does NOT split them** (verified with fresh Pass 2): the merged
  window comes out as ONE segment / ONE rally.
- **Why:** P(live) stays ~0.76 in the dead gap (needs <0.25 to split/end). Diagnosis:
  - `cross_recency` (classifier weight 3.2, τ=2.5s) dominates and is **refreshed by
    a single between-point ball knock** → stays high.
  - `ready_frac` is treated as **live**, but players standing ready *between* points
    looks the same as pre-serve ready → it *rises* (0.65) in the gap, propping P(live).
  - **Player MOTION is the real discriminator but underweighted**: max player speed
    0.46 (rally) vs 0.23 (gap). Players run during a point, stop between points.

**End-detection functionality DOES exist and is NOT bounce-dependent** (the user
asked specifically):
1. **P(live) decay (Stage C)** — blends ball rhythm (cross recency, speed, visibility,
   out-of-court) AND player signals (player speed, reactivity, ready_frac). No bounce.
2. **`snap_end`** — snaps end to the last of {bounce, **net crossing**, **fast-ball
   moment**}. Bounce is 1 of 3; crossings + fast-ball work without it.

**Recommended fix (one change, fixes both):** make the dead/end signal lean on
**player motion stopping** (players decelerating out of rally movement) and de-weight
`cross_recency` / static `ready_frac`. This would (a) sharpen rally ends and (b) let
Pass 2 split merged windows → making `cw=3.0` actually safe.

**Decision pending for the user:** either fix P(live) dead-detection first (then
cw=3.0 is safe — recommended), OR lower `p1_cross_window_s` back toward 2.0 (fewer
merges, ~2 window-level false negatives) as a stopgap.

---

## Other open items / backlog
- **Grade real clips** through the loop (built, unused) → the recommendation sweep
  then tunes `serve_gate_min` on real data (the 7-clip benchmark is tiny/noisy).
- **Ballistic sweep perf**: cache the reconstruction (cfg-independent) so
  `evaluate.sweep` isn't ~17× slower with ballistic on.
- **Ballistic arc truncation**: stop extrapolating past an undetected bounce
  (truncate at the height=0 crossing) — minor off-court/negative-height artifact.
- **`landing` still 0** on the example serve (receiver-side bounce not in expected
  window) — minor 0.10 booster; serve is kept anyway.

---

## Current state / environment notes
- **Params:** learned_params **v15** (`p1_cross_window_s=3.0`). Override YAML
  `data/refine/config.override.yaml` = `serve_gate_min: 0.35`.
- **DB:** Pass 1 windows are the cw=3.0 (merged) set. Pass 2 overlays: First 3 mins
  clips (`646b708a`, `76a710e8`) freshly reprocessed; **others stale** (built on old
  Pass 1 windows). `5a653ffa` clip#1 has no Pass 2 overlay in the merged range.
- **Pass 2 ball detection is gated by env `PASS2_BALL`** (default 0 = OFF). Set
  `PASS2_BALL=1` to reprocess with ball tracking (needed for refine/serve gate).
- **Runtime:** Intel Arc XPU via OpenVINO. Pass 1 ~1-2 min/clip; Pass 2 ~3-5
  min/clip (1920 pose + ball). venv: `backend/.venv/Scripts/python.exe`.
- Reprocess a script file with `PYTHONPATH="…/backend"` and `PASS2_BALL=1`.

## Key files
- `backend/app/cv/pass1.py` — Pass 1 gate + gap-tolerant crossing + decoupled end
- `backend/app/refine/ballistic.py` — NEW ballistic 3D + arc contact
- `backend/app/refine/serve_gate.py` — serve gate, `_ballistic_contact`, `snap_end`
- `backend/app/refine/classifiers.py` — P(live) RulesClassifier (the dead/live signal to improve)
- `backend/app/refine/features.py` — features incl. player_speed, reactivity, ready_frac
- `backend/app/refine/postprocess.py` — Stage C segmentation (where splitting must happen)
- `backend/app/refine/recommend.py` — NEW grade-scored recommendation
- `backend/app/api/annotate.py` — grading + recommendation endpoints
- `backend/app/refine/config.py` — all refine knobs + `override_path()`
- `frontend/src/Workspace.jsx`, `Timeline.jsx`, `api.js` — grading UI
- `PASS1_PASS2_RALLY_BOUNDARIES.md` — living boundary-logic doc (§2.4b/2.4c updated)

## How to reproduce the key measurements
- Refine keep/drop benchmark: `cd backend && ./.venv/Scripts/python.exe -m app.refine.evaluate`
- serve_gate_min sweep: `evaluate.evaluate(RefineConfig(use_ballistic_3d=True, serve_gate_min=X))`
- Pass 1 vs GT (FN/end error): compare `rally_labels source='pass1'` to `source='human'`
  by IoU (script pattern used this session; `clips_with_human_labels()` for the set).

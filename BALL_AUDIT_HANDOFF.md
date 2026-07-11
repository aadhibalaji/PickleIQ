# Ball Detection Audit — Session Handoff

**Date:** 2026-07-09  
**Purpose:** Carry forward audit findings, measured numbers, code state, and prioritized fixes for a new session.

---

## Executive summary

The pipeline "works better" after local fixes, but meaningful tracking noise remains. Two separate problems were conflated:

1. **Display/infrastructure noise** — Kalman coast painting + ungated TrackNet (mostly fixed in uncommitted `pass2.py` changes; requires Pass 2 reprocess to see).
2. **Detector quality** — Pretrained TrackNet doesn't transfer to this fixed pickleball camera; gates help but can't fix wrong detections. Real long-term fix = fine-tune TrackNet on `ball_corrections` (path doesn't exist yet).

The **hit-gap fix** from a prior Fable session (`find_hit_gap_spans`, `hit_gap_*` params, hit P/R in eval) **does NOT exist on this machine's repo** — never committed/pushed. Do not assume it's present.

---

## Machine / repo state (critical)

- **Repo:** `CV-Training-Tool` (git at `CV-Training-Tool/.git`, branch `aadhi-dev`, HEAD `0e63b3e` "fable audit")
- **Hit-gap code:** Absent everywhere — working tree, `origin/main`, `origin/rohan-dev`, `origin/aadhi-dev`, reflog
- **Local DB:** 3 video rows, all same file `doubles clip.mp4` (MD5 `DD922CB660D94417252102C688609EFE`), IDs:
  - `91ab2dfaef434f3daff9d1582ce8ad83` (most overlays — use this one)
  - `c01d5681ad2b4af1a3b01ab56d5c89a1`
  - `ea5b23bed53943ceaae9b6bac87b7624`
- **GT clips (`760f1e0e…`, `17be6eb5…`):** NOT in local DB. `reference_data/ball_gt.json` is legacy flat-array keyed by exporter's video IDs — **not portable** without `--map`
- **GT vs local footage:** Measured NOT the same video (alignment tests failed). Cannot run acceptance eval on this machine's marks.
- **`.env`:** `TRACKNET_ENABLED=1`, weights at `data/models/tracknet_v2.pt`, `CV_DEVICE=cuda`
- **LearnedParams:** version 1, `ball_filter_enabled=0.0` (arc RANSAC off by default)

---

## What "worked better" actually was (local uncommitted fixes)

These exist in working tree but may not be committed:

| Area | Fix |
|------|-----|
| `ball_coarse.py` | Duplicate-frame dedup (30fps muxed to 60fps — classical was blind every other frame) |
| `ball_track.py` | `near_track()` gates classical "last resort" pick |
| `pass2.py` | Pass-1 seeding for selector; `suppress_on_player` off for classical-only |
| `ball_tracknet.py` | Default weights pickup if `tracknet_v2.pt` exists |
| `pass2.py` (audit session) | TN court gate, TN/ext frame filter always on, linear short-gap fill (no Kalman coast paint) |
| `ball_gt_io.py` | Legacy import + `--map ID=filename` |

---

## Part 1 — Acceptance run (BLOCKED on this machine)

**Prior session's acceptance bar** (never run here):

- Pooled hit recall/precision vs GT hit marks
- Bounce precision (should hold/rise after fake in-gap bounce suppression)
- `player=inside` position-error slice unchanged
- Optional `hit_gap_turn_deg` / `hit_gap_speed_ratio` sweep

**Commands** (on machine WITH GT clips + marks in DB):

```bash
cd backend
.venv/Scripts/python.exe scripts/eval_ball_filter.py --diagnose
.venv/Scripts/python.exe scripts/eval_matrix.py
```

**To import GT locally** (marks still won't match footage unless same video):

```bash
python -m backend.scripts.ball_gt_io import --map 760f1e0e="doubles clip.mp4" --map 17be6eb5="doubles clip.mp4"
```

---

## Part 2 — Measured error sources (doubles clip, stored Pass 2 overlay)

**Setup:** 18 pass1 windows, 9267 frames, TrackNet run, `ball_filter_enabled=0`

### Before fixes (stored overlay)

| Metric | Value |
|--------|-------|
| Trail points | 9147 (98.7% frame coverage) |
| Source mix | 53% `kalman`, 47% `tn` |
| Kalman fill off straight-line bridge | med 94px, p90 605px, max 7125px |
| Physically impossible frame-to-frame jumps | 1737 / 1737 pairs (18.9%) — almost all Kalman coast |
| TN outside court region | 23.6% (gate wasn't wired to TN path) |
| TN foot-zone | 17.4% |
| TN consecutive-pair speed violations (ungated) | 8.4% |
| Corner exclusion triangles beyond court gate | **0** (446/446 already outside court region) |

### After replaying new logic on same stored TN dets (no GPU rerun)

| Metric | Value |
|--------|-------|
| Trail points | 4879 (52.6% coverage) |
| Source mix | 59% `tn`, 41% `interp` (short linear fill) |
| Speed violations | 40/4768 (0.84%) vs 18.9% before |
| After court-region + frame-filter on TN stream | 0% impossible pairs (0/2774) |

### Dominant noise sources (ranked)

1. **Kalman coast painting** (fixed in code — reprocess to see)
2. **Ungated TrackNet** passing court-region + velocity gates (fixed in code)
3. **Pretrained TrackNet FPs** that survive gates (~88% of in-court dets kept)
4. **NOT** corner triangles (redundant with homography)
5. **NOT** far-baseline depth ceiling as primary issue on this clip
6. **Hit-gap misclassification** — can't measure; code doesn't exist here

---

## Proposal verdicts

### A — Confidence-adaptive gating + curvature

**REJECT for now.** Existing gates weren't wired; once wired, 0% impossible pairs on TN stream. Curvature check redundant until post-fix GT shows surviving in-gate FPs. Needs hit-gap deference logic that doesn't exist.

### B — Reuse player exclusion triangles for ball

**REJECT.** 100% subset of `_in_court_region`. Wire homography gate to TN (done).

### C — Is labeled data improving anything?

**YES — audit finding: dead loop.**

| Question | Answer |
|----------|--------|
| `ball_corrections` used for? | Per-frame Pass 2 override, eval harness, viewer CRUD only |
| TrackNet fine-tune path? | **Does not exist** (`model_registry` has `"tracknet"` kind, builder returns None; only `get_tracknet_weights.py`) |
| `ball_gt.json` + eval loop? | Eval only; legacy format broke import until `--map` fix |
| Architecture wrong vs starved? | **Starved/mis-wired** — gates existed but off/unwired; training signal exists but feeds nothing |
| Minimum loop | (1) filename-keyed GT export/import (partial), (2) eval harness as validation gate, (3) `train_tracknet.py` → `model_registry` shadow/promote |

---

## Code changes made (audit session — verify still present)

### `backend/app/cv/pass2.py`

- TrackNet: `_in_court_region` before accepting det
- `BallFrameFilter` always constructed; applied to `tn`/`ext` unconditionally
- Classical path: frame filter still behind `ball_filter_enabled`
- Trail fill: linear interp across gaps ≤ 0.15s (`src: "interp"`), NOT Kalman coast paint
- Kalman `smooth_track` still runs for world/height/events — just not painted to overlay

### `backend/scripts/ball_gt_io.py`

- Legacy flat-array import with `--map <id-prefix>=<filename>`
- Loud warning for unmapped exporter IDs

---

## Viewer semantics (VideoPlayer.jsx)

- **Solid red** (α≈0.95): real detection (`tn`, `cls`, etc.)
- **Faded red** (α≈0.2–0.55): inferred (`interp`, `kalman`, `cls-loose`, `obs:false`)
- **Small white ring**: current ball position — normally **one** at a time
- **Large labeled ring**: bounce/hit event marker (lingers ~45 frames)
- After fresh Pass 2: higher bright-to-faded ratio, sparser trail, more honest gaps — NOT a dot every frame

---

## To see fixes: reprocess checklist

1. Restart backend (`.env` loads at startup; code via `--reload` on `npm run dev`)
2. Reprocess **Pass 2 only** on the video ID you're viewing (`91ab2dfa…` recommended)
3. Wait for job completion
4. Browser refresh, **Pass 2** tab
5. Do NOT need re-upload unless redoing Pass 1

---

## Prioritized action list (value per credit)

| # | Action | Cost | Status |
|---|--------|------|--------|
| 1 | Court-region gate on TN | Trivial | Done in pass2.py |
| 2 | Stop Kalman coast paint; short linear fill | Small | Done in pass2.py |
| 3 | Frame filter on TN/ext always | Small | Done in pass2.py |
| 4 | Legacy GT import `--map` | Small | Done in ball_gt_io.py |
| 5 | **Re-run Pass 2** to materialize fixes | User action | Pending |
| 6 | Raise `tracknet_conf` to 0.65–0.75 | Trivial param | Not done |
| 7 | Try `ball_filter_enabled=1` (arc RANSAC) | Param toggle | Not done |
| 8 | A/B classical-only (`TRACKNET_ENABLED=0`) | Env toggle | Not done |
| 9 | Re-export GT filename-keyed from marks machine | Small | Not done |
| 10 | Push hit-gap branch from other machine OR re-implement | Medium | Not done |
| 11 | `train_tracknet.py` + registry wiring | Expensive | Not done |

---

## Quick tuning knobs

```python
# learned_params (DB) or eval overrides
tracknet_conf = 0.5          # try 0.65–0.75
ball_filter_enabled = 0.0    # try 1.0 for arc RANSAC outlier drop
ball_court_margin_x_m = 2.5
ball_court_margin_y_m = 3.0
ball_kalman_gate2 = 13.8
```

```env
TRACKNET_ENABLED=1
TRACKNET_WEIGHTS=.../data/models/tracknet_v2.pt
```

---

## Key file paths

```
backend/app/cv/pass2.py           # main ball pipeline
backend/app/cv/ball_filter.py     # velocity/foot/persist + arc RANSAC
backend/app/cv/ball_coarse.py     # classical detector + dedup
backend/app/cv/ball_tracknet.py
backend/app/cv/kalman.py
backend/app/learning/params.py      # ball_filter_enabled default 0
backend/scripts/eval_ball_filter.py
backend/scripts/eval_matrix.py
backend/scripts/ball_gt_io.py
reference_data/ball_gt.json       # 445 marks, legacy format
frontend/src/VideoPlayer.jsx      # trail rendering
DETECTION_CONTEXT_FOR_FABLE.md      # original architecture audit
```

---

## Open questions for next session

1. After Pass 2 reprocess: is noise mostly solid-red (detector) or faded (fill)?
2. Does `ball_filter_enabled=1` help or hurt TrackNet trail on this clip?
3. Can hit-gap code be recovered from the other machine and merged?
4. Re-export `ball_gt.json` in filename-keyed format from marks machine?
5. Build `train_tracknet.py` — minimum viable fine-tune loop?

---

## Prior session hit-gap context (NOT in repo — reference only)

Rejected: unconditional in-box suppression, arc prediction through player region.

Survived design: `find_hit_gap_spans()` — entry/exit velocity turn≥60° + speed ratio≥1.8x classifies hit vs passive occlusion.

Planned params: `hit_gap_min_frames`, `hit_gap_turn_deg`, `hit_gap_speed_ratio`.

**None of this is in the current codebase.**

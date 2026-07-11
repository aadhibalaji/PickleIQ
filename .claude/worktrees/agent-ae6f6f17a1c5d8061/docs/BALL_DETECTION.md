# Ball Detection — In-Depth Guide

This document explains how the pickleball ball is detected, tracked, filtered, and
turned into hit/bounce events, and — importantly for anyone **consuming the data** —
what each output field means, where it comes from, and how much to trust it.

> TL;DR for data consumers: the per-frame ball position lives in the **Pass 2 overlay
> entries** under the `ball` key (`[x, y, diameter_px, confidence]`, full-resolution
> image pixels). Bounce/hit events live under the `evt` key with a court-space position
> and an in/out call. Ball tracking is the accuracy ceiling of the whole system; treat
> far-court positions and bounce recall as the weakest links.

---

## 1. Two passes, two very different ball jobs

The system runs two passes over each clip, and the ball means something different in each.

| | Pass 1 (`cv/pass1.py`) | Pass 2 (`cv/pass2.py`) |
|---|---|---|
| Coverage | 100% of the clip, low-res / sub-sampled | Only inside Pass 1 rally windows (~20%), full-res, full frame rate |
| Ball's role | **Cheap signal**: "is a fast ball-sized blob crossing the net?" — used to decide where rallies are | **Precise per-frame position** for display + bounce/hit events |
| Detector | Classical motion blob (`MotionBallDetector`) at low res | TrackNet (if weights) → else classical fallback → optional external track → manual corrections |
| Output | Rally start/end boundaries (`rally_labels`, `source='pass1'`) | Per-frame overlay entries with `ball`, `world`, `evt` |

Pass 1 is the **sole source of rally boundaries**. Pass 2 no longer computes rallies —
its ball outputs are **display / analysis only** and drive no boundary decisions. So a
wrong ball position in Pass 2 is cosmetic; a missed net-crossing in Pass 1 can drop a
whole rally.

---

## 2. The raw detector (`cv/ball_coarse.py`)

Both passes ultimately rest on the same classical detector: **three-frame differencing**.

1. Convert frame to gray, Gaussian blur.
2. Threshold the absolute difference against the previous frame **and** against the
   frame before that, then `AND` them. Only objects moving across **both** frame pairs
   survive — this localizes the blob to the *middle* frame and suppresses static court
   lines and slowly-moving players.
3. Find contours; keep those with area in `[min_area, max_area]` and circularity ≥ 0.35.
4. **Player-box suppression** (`on_player_box`): drop any blob inside a player box
   expanded outward by shared margins (largest at the bottom, `BOX_MARGIN_BOTTOM=0.12`,
   where shoes and shadows live). These margins are shared with the trajectory selector
   so the two never drift.
5. Optional **appearance/colour gate** (`_color_ok`): a pickleball is bright and
   saturated. Sampled from the blob *core* (brightest pixels), with hue wraparound.
   **Disabled by default** (`sat_min=val_min=0`) because ball colour varies per clip;
   set `ball_sat_min` / `ball_val_min` / `ball_hue_lo` / `ball_hue_hi` once you know the
   clip's ball colour to cheaply kill off-colour foot/shorts blobs.
6. Return the top-5 candidates sorted by circularity.

**Key limitation:** the classical detector detects the ball *in bursts* — it needs
motion between three frames, so it loses the ball for many frames around bounces and in
the far court where the ball is only a few pixels wide. This is the single biggest
source of downstream error.

---

## 3. TrackNet — the real per-frame model (`cv/ball_tracknet.py`)

The intended production detector is **TrackNetV2**, a VGG-style U-Net that takes 3 RGB
frames in and emits 3 heatmaps out (one per frame); the ball is the heatmap argmax.

- Architecture byte-for-byte mirrors NTT's WASB-SBDT so their **pretrained
  tennis/badminton checkpoints load directly** (`scripts/get_tracknet_weights.py`).
- Runs on the **Intel Arc GPU via OpenVINO** (FP16 IR cached next to the weights). Eager
  torch-XPU runs this U-Net at ~110 ms/frame; OpenVINO is the production path.
- `infer_triplet(frames)` returns `(x, y, diameter_px, confidence)` in **original frame
  pixels**, or `None` below `conf_threshold` (default 0.5). `last_peak` keeps the raw
  argmax regardless of threshold, so diagnostics can tell a true miss from a
  sub-threshold detection.

**Status:** gated by `TRACKNET_ENABLED` / `TRACKNET_WEIGHTS`. Until weights are
installed, Pass 2 falls back to the classical detector and logs a warning. Tennis
weights transfer the "small fast ball" prior but the fixed down-the-baseline pickleball
view still needs fine-tuning, especially on far-baseline balls. Training data source:
the labeling UI's ball spot-corrections + stored FeatureTraces.

---

## 4. Ball source priority in Pass 2

For each frame, Pass 2 picks the ball position from the **first available** source:

```
manual correction  >  external track  >  TrackNet  >  classical detector
```

- **Manual corrections** (`ball_corrections` table): user spot-fixes in the labeling
  tab, clip-local, always authoritative. Exportable via `scripts/ball_gt_io.py` to
  `reference_data/ball_gt.json` (git-tracked) so a partner can read/re-import them.
- **External track** (`ball_tracks` table): a dense per-frame track from an outside
  tracker (e.g. an MTI / temporal-integration tracker), uploaded via
  `POST /videos/{id}/ball_track`. Values are `[x, y, diameter|null, confidence]` in
  full-res pixels. Preferred over the classical detector.
- **TrackNet** / **classical**: as above.

Whatever the source (except manual corrections), the detection is re-checked against the
**same semantic filter** (stages 1–2 below) before being accepted.

---

## 5. The semantic filtering layer (`cv/ball_filter.py`)

Physics + player context on top of the raw candidates. Detector-agnostic. Four stages:
the first two run **online** (per frame, before trajectory selection); the last two run
**batch** (per window, after selection). Toggle with `ball_filter_enabled`.

### Stage 1 — Velocity gate (`BallFrameFilter._velocity_ok`)
Reject candidates implying an impossible jump from the last confirmed ball position. The
max plausible pixel step is derived from the **homography** (local px→metre scale at the
candidate) and a generous max ball speed (65 mph × 1.6 headroom). Because the far court
has more metres per pixel, the gate **tightens automatically** in the far court, where a
real ball moves fewer pixels per frame.

### Stage 2 — Player-box motion logic (`BallFrameFilter._persistent`)
The ball is legitimately inside a player box at every paddle contact, so boxes are never
blanket-suppressed. Instead:
- candidates in the bottom `foot_zone_frac` (0.25) of a box are **feet/shoes → reject**;
- candidates holding the same **box-relative** position for `persist_frames` (3+) frames
  move *with* the player (shoe, hat, held paddle) → reject;
- everything else (transits, hit-moment blobs) passes through to the trajectory stages.

Per-candidate outcomes are recorded in `last_reasons` (`kept` / `foot_zone` /
`persistent` / `velocity`) for diagnostics, and counted in `RejectStats`.

### Stage 3 — Arc segmentation + RANSAC (`refine_track`)
The per-frame selected track is split into per-shot **arcs** at sharp direction changes
and time gaps (`_segment`). A split needs **both** a big turn angle **and** a big
velocity-vector change (`split_dv_px`) — angle alone shreds slow far-court shots (a few
px of noise → huge spurious angle). Each arc gets a **RANSAC quadratic** fit in image
space (a projectile is quadratic in time; perspective bends it a little, absorbed by the
tolerance). Off-arc points are dropped as outliers; short holes between real points are
**filled from the fit** (`interpolated=True`).

> **Important accuracy note:** interpolation only fills **short** holes
> (`max_interp_gap`, default 6). Filling long holes was measured as the single worst
> error source (median 286 px error vs 19 px for real picks) — the ball often changes
> direction mid-hole and the quadratic bridges the gap with a confident wrong curve. So
> long gaps are left empty rather than guessed.

Off-arc leftovers get one rescue chance: if an adjacent arc extrapolates to them within
a loose tolerance they're kept as a shot stub; otherwise they're discarded as noise.

### Stage 4 — Hit / bounce classification (`classify_arc_events`)
Events come from two signals:
1. **Arc junctions** — a real direction change between adjacent arcs. At/near a player
   box → **hit** (paddle contact); in open court → **bounce**. Bounce junctions also
   pass a **plausibility gate**: a real bounce arrives descending and leaves ascending
   (image y grows downward), so mean vy must be ≥ +`vy_min_px` in and ≤ −`vy_min_px`
   out. A short-stub side (< `vy_trust_span` frames) abstains rather than vetoes.
2. **Within-arc y-vertex bounces** — a shallow bounce the RANSAC tolerance swallowed
   into one quadratic: if the fitted y-curve peaks inside the arc span (a descend-then-
   ascend V), it's a bounce. Open court only.

Nearby duplicate bounces (double-splits of one physical event) are merged, keeping the
larger-angle one.

> **Measured limitation:** bounce **recall** is capped by *track* quality, not this
> logic — the detector loses the ball for 12+ frames at most bounces. That's TrackNet
> (Phase 2) territory. Don't expect every real bounce to appear as an `evt`.

---

## 6. The trajectory selector (`cv/ball_track.py`)

Sits between the raw candidates and the arc stages, in the **classical** path. Its whole
premise: **motion, not appearance, separates a ball from a shoe.** In one frame a white
shoe and a pickleball look identical; over a few frames a ball follows a fast, smooth,
gravity-consistent arc while a foot oscillates slowly, tethered to a body.

`TrajectoryBallSelector.select()` picks the candidate that best **continues the recent
trajectory**:
- With ≥2 history points, it predicts forward by velocity and gates tightly
  (`gate_px` + accel slack). A foot off the arc is excluded.
- With 1 point, velocity is unknown, so it gates the full plausible per-frame travel
  (`acq_px`) to acquire the second point.
- A candidate on a player box / foot keypoint is trusted **only if it's moving at
  `hit_speed_px`** (a hit or a fly-over); a slow blob on a foot *is* a foot → dropped.

**Cold start** (`_cold_start`): with no trajectory yet, the selector refuses to seed from
the roundest single-frame blob (that seeds on shoes). Instead it requires a
**mutually-nearest, ball-speed displacement between two consecutive frames'** off-player
candidate pools. Feet barely move frame-to-frame and are rejected; the fast-moving ball
is the one that matches. `seed()` lets an external/manual/TrackNet position feed the
history so continuity holds when sources mix.

Court-region gate (`_in_court_region` in pass2): candidates whose court-plane projection
falls well outside this court (neighbouring courts, spectators) are dropped before the
selector even sees them.

---

## 7. Smoothing & 3D reconstruction

After selection + filtering, the surviving observations are:
- **Kalman-smoothed** (`smooth_track`, `cv/kalman.py`) which bridges short dropouts
  (`max_gap ≈ 0.4 s`) and gates outliers (`gate2`).
- Reconstructed to **court position + height** (`reconstruct_world`, `cv/bounce.py`)
  using the calibrated `CameraModel`. Attached to each entry as `world`.

---

## 8. Output data — what a consumer actually reads

Pass 2 writes overlay entries (one per frame) via `overlays.write_entries`. Relevant
ball fields on each entry:

| Field | Type | Meaning |
|---|---|---|
| `f` | int | Frame index |
| `ball` | `[x, y, diameter_px\|null, confidence]` | Ball position in **full-res image pixels**. `confidence` is 1.0 for manual, ~0.9 external, TrackNet's heatmap peak, or the classical circularity score. **0.0 means interpolated** (see `interp`). |
| `interp` | bool | Present + true when `ball` was **filled from an arc fit**, not observed. |
| `world` | `[court_x, court_y, height\|null]` | Ball in court metres + reconstructed height (null when unreconstructable). |
| `obs` | bool | Whether `world` came from a real observation vs. a smoother fill. |
| `evt` | object | A bounce/hit event **at this frame** (see below). |

An `evt` object:

```json
{ "kind": "hit" | "bounce",
  "in": true,                 // ball inside court bounds at the event
  "side": -1 | 0 | 1,         // which side of the net
  "cx": 4.12, "cy": 8.90 }    // court-space position (metres)
```

For bounces, `cx/cy` come from the smoothed world track when available, else the arc
junction point projected through the homography (exact for bounces since they're on the
floor). The `in`/`out` call is derived from `court.in_court`.

**Ground-truth marks** (for evaluating any of the above) export to
`reference_data/ball_gt.json` via `python -m backend.scripts.ball_gt_io export`.

---

## 9. How much to trust it (caveats for consumers)

- **Ball tracking is the system's accuracy ceiling.** Player detection is now reliable;
  the ball is the weak link.
- **Far-court positions** are least reliable — the ball is a few pixels wide, the
  classical detector often misses it, and the velocity gate is tightest there.
- **Bounce recall is limited by detector dropouts** around the bounce (12+ frames lost),
  not by the event logic. Missing bounces are expected until TrackNet is fine-tuned.
- **`confidence == 0.0` / `interp == true` = interpolated**, not observed. Treat these
  positions as a smooth guess between real detections; they're never filled across long
  gaps precisely because long-gap fills were the worst error source.
- **Everything in Pass 2 is display/analysis only** — it does not affect rally
  boundaries (those come from Pass 1).
- The **classical detector is a fallback.** For best accuracy install TrackNet weights or
  supply an external track via `POST /videos/{id}/ball_track`.

---

## 10. File map

| File | Responsibility |
|---|---|
| `cv/ball_coarse.py` | Classical three-frame-diff detector + shared player-box margins |
| `cv/ball_tracknet.py` | TrackNetV2 model, OpenVINO/Arc inference |
| `cv/ball_track.py` | Trajectory selector (motion-based ball vs. foot) |
| `cv/ball_filter.py` | Velocity gate, box motion logic, arc RANSAC, hit/bounce classification |
| `cv/pass1.py` | Low-res net-crossing gate → rally boundaries |
| `cv/pass2.py` | Full-res per-frame ball orchestration + overlay output |
| `cv/kalman.py`, `cv/bounce.py` | Smoothing + 3D world reconstruction |
| `scripts/ball_gt_io.py` | Export/import ground-truth ball marks |
| `scripts/eval_ball_filter.py`, `scripts/eval_matrix.py` | Evaluation harnesses |
</content>
</invoke>

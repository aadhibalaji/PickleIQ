# Detection Architecture Audit — Player/Ball Pipeline

**Scope:** audit + rebuild of the live detection path per
`DETECTION_CONTEXT_FOR_FABLE.md`. All numbers marked *measured* were computed
against the **stored calibration** in `data/app.db` (doubles clip, 1920×1080 @
59.94 fps; self-calibrated focal **f = 878 px**, camera centre recovered at
court coordinates **(3.0, 14.9, 3.7) m** — 1.5 m behind the near baseline,
3.7 m up) and against the stored Pass 2 overlays (≈11k frames, ≈8.2k player
boxes). Claims I could not verify without the GT-marked clips ("Second 3
mins", clip #1 — not present in this DB) are flagged as such.

---

## 1. Noise / error propagation (the first-class deliverable)

### 1.1 Ball diameter → depth → height: structurally broken at the far court, fine near

`Z = f·D/d` gives `|dZ/dd| = Z²/(f·D)`. With the real calibration
(f = 878 px, D = 74 mm), for a **±1 px** diameter error:

| court_y | camera dist | apparent diam | ΔZ (depth) | Δheight | Δcourt_y |
|---|---|---|---|---|---|
| 13.4 m (near BL) | 3.3 m | 22.0 px | ±0.13 m | ±0.12 m | ±0.06 m |
| 6.7 m (net) | 8.7 m | 7.8 px | ±1.07 m | ±0.37 m | ±1.01 m |
| 0.0 m (far BL) | 15.2 m | **4.7 px** | **±2.91 m** | **±0.57 m** | **±2.85 m** |

Three consequences:

1. **Far heights are noise dressed as measurement.** The bounce gate is
   `bounce_max_height_m = 0.20`; far-court height 1σ is ~0.57 m/px. The 7-tap
   diameter median + 5-tap height median cut *random* noise by ~2.5×, leaving
   σ ≈ 0.2 m — still the size of the entire gate. Far "bounces" from height
   minima were therefore mostly threshold-crossings of noise. The 2 cm
   descend/ascend "gravity check" (`desc > 0.02`) is ~25× below far-court
   noise and filters nothing there.
2. **It is a *bias*, not just noise, exactly when it matters.** At 60 fps a
   15 m/s ball at the far baseline moves ~16 px/frame while being 4.7 px
   wide; with a typical 1/120 s shutter the blur streak is ~8 px on a 4.7 px
   ball. `minEnclosingCircle` on the streak inflates diameter 2–3×, so depth
   collapses toward the camera by the same factor — a *speed-correlated,
   systematic* error the median filter cannot remove (all frames of a fast
   shot are biased the same way). Circularity ≥ 0.30 does not reject a 2:1
   streak (circ ≈ 0.47).
3. **The 3D court position is worse than the homography for on-ground
   points.** `Δcourt_y ≈ 2.85 m/px` at the far baseline versus the
   `out_max_dist_m = 1.5` phantom-out band means far in/out calls off the 3D
   estimate were near-random. A bounce is *by definition on the plane*, where
   the homography projection is exact — that information was available and
   unused.

**Verdict: size→depth is viable roughly up to the net (d ≥ ~8 px, σ_h ≤
~0.4 m) and structurally non-viable beyond it.** Height for far frames should
come from a different cue (temporal/learned model, or honestly reported
unknown), never from apparent size.

**Implemented:** `CameraModel.height_sigma()` computes the per-point 1σ
height error; `reconstruct_world()` now NaN's the height and uses the plane
projection for court position wherever σ_h exceeds the new learnable
`ball_height_max_sigma_m` (default 0.35 m); the legacy `detect_events` now
takes `H + ball_px` and computes **bounce** positions (and the phantom-out
distance test) from the homography instead of size→depth. The NaN heights
flow into the existing honesty machinery unchanged: bounce candidates are
skipped on NaN, and the phantom-out guard already requires `height_known`.
On a synthetic 20-point track sweeping the court, 13/20 far points correctly
report height-unknown while near points keep exact heights.

### 1.2 Classical detector → selector → Kalman: single bad points are damped; the dangerous case is a fast *sequence*

- **Single false detection:** the Kalman gate (`ball_kalman_gate2 = 13.8`,
  meas σ = 2 px) rejects anything ≳9 px off the prediction after
  convergence; a rejected point is predicted through, not smoothed in.
  **Damped, not amplified** — the design is right. A false point *within*
  9 px of the prediction is accepted but is then also within measurement
  noise of the truth, so harmless. The arc-RANSAC layer (when enabled)
  independently drops isolated off-arc points.
- **The realistic gate-passing sequence is the paddle, not the shoe.** A
  swing moves the paddle tip 10–40 px/frame; it clears the ±6 % box side
  margins for several frames; motion-blurred it is round-ish (circ > 0.3);
  its speed passes `hit_speed_px = 18` and `seed_min_px = 2`; two consecutive
  mutually-nearest sightings satisfy the cold-start invariant. Result: a
  3–6 frame false track. Consequences downstream: after 3 consecutive
  Kalman rejections (`max_reject = 3`) the filter *restarts on the false
  position*; the arc layer will fit a ≥4-point arc if the travel is
  ball-like. Because such tracks live next to a player box they classify as
  phantom **hits** (bounded damage), but one that flies off the box edge can
  make a phantom bounce. This is inherent to a motion-only detector; the
  selector logic itself has no cheap fix that doesn't also kill real
  hit-moment detections. The real mitigation is a detector with appearance
  (fine-tuned TrackNet / external track).
- **Empirical reality check (measured, this clip):** over 9,162 rally-window
  frames the classical detector produced only **431 raw in-court candidates**
  (~5 % of frames) and the selector confirmed **0** — the stored overlay has
  20 ball points in 11k frames. On this footage the classical fallback is
  not "weak", it is effectively **dead** (too few consecutive candidate
  pairs to ever cold-start). See §3.4.

### 1.3 Homography / focal self-calibration: fragile, and bad-but-not-degenerate was undetectable — now cross-checked

Monte-Carlo over the *actual stored 12 clicked points* (300 trials/row,
Gaussian click noise):

| click noise σ | focal p10–p90 (true 878) |
|---|---|
| 1 px | 772 – 1056 (−12 % / +20 %) |
| 2 px | 763 – 1158 |
| 4 px | 777 – 1258 (+43 %) |

A ±20 % focal error scales all far-court depths ±20 % (≈ ±3 m) and heights
≈ ±0.5 m *systematically for the whole clip*, with zero symptoms — the
previously-existing check only caught the fully degenerate case (and the
degenerate fallback `f = img_w·0.9 = 1728 px` is itself ~2× the true 878 px
on this genuinely wide-FOV camera, ~95° horizontal; the "generic 60° FOV"
assumption is far off for this rig).

**A cheap downstream detector exists and works.** Reconstructed
standing-player heights (foot court position from H + head-pixel ray through
K, R) are a focal-sensitive invariant with a known human prior. *Measured on
the stored overlays: median implied height 1.57 m, p10–p90 = 1.35–1.96 m* —
comfortably inside a [1.2, 2.2] m acceptance band with the good calibration;
a 2× focal error pushes the whole distribution far outside it.

**Implemented:** `court.implied_person_height()`; Pass 2 samples it twice a
second on every placed player and, at the end of the run, logs pass/fail and
appends a visible warning to the job message when the median leaves the
band. The degenerate-focal fallback is also no longer log-only:
`CameraModel.focal_degenerate` is set and Pass 2 surfaces it in the job
message at start.

### 1.4 Exclusion zones / far-player occlusion: what is physics and what is a discardable heuristic

**Occlusion of the 2nd far doubles player — genuine information loss,
quantified.** With the camera at (3.0, 14.9, 3.7):

- A near-court player (torso ~0.5 m wide at ~5 m from the camera) shadows a
  ~**1.5 m-wide corridor** at the far baseline (3× angular magnification);
  two near players shadow ~3 m of the 6.1 m court width — ~50 % of the far
  baseline is behind somebody at any moment.
- The far team stacks along the *camera axis* (down-the-line view), so the
  rear far player is fully covered by their partner whenever laterally
  within ~0.5 m of them — which is most of ready position.

Frames where this holds contain **zero pixels** of the rear player: no
resolution or model capacity recovers them, confirming the doc's measured
nano@2560 / yolo11s@1920 result. The "accept it" call is correct **per
frame**. What *is* recoverable is track-level: the player is visible before
and after occlusion, so a constant-velocity coast on the identity slot
(flagged as predicted, like the ball's `interp`) would fill the display gap
honestly. Proposed, not implemented (visualization polish, not detection).

**Exclusion triangles — a fragile heuristic duplicating information the
calibration already has.** The zones encode "people up there are on other
courts" in image space, hand-tuned to one framing (2026-07-07, "round 3").
The homography proves the same thing robustly: a foot projecting > ~4 m
outside a sideline or > ~8 m beyond a baseline cannot be a participant.
**Implemented:** geometry cull in `PoseTracker.update` (margins deliberately
wider than the ±3/±6 m `court_xy` validity band so it can only remove
provably-off-court people). The triangles are retained as a fallback for the
uncalibrated path and any residual cases (e.g. spectators seated within 4 m
of a sideline), but they are no longer load-bearing and can be emptied per
deployment.

### 1.5 Identity re-ID: failure is silent — now surfaced

Cost = court-distance/2.0 + 1.5×Bhattacharyya(HS-hist). Failure modes:

- **Same-colour kits:** the appearance term is ~equal for both slots, so the
  decision rides on position alone; in a crossing, position is exactly what
  is ambiguous → coin-flip swap. **Silent**: once both slots exist the code
  always binds to the best match (`MATCH_MAX` only gates new-slot creation),
  so a wrong number renders identically to a right one.
- **Lighting drift:** the histogram EMA (α = 0.12/frame at 60 fps) adapts in
  well under a second — non-issue for gradual drift; a sudden exposure step
  during an occlusion can flip a re-acquisition, again silently.
- **Fast crossing near the net + dropout:** slot `court_xy` goes stale
  during the dropout; re-acquisition leans on appearance, which is the weak
  signal in the same-kit case.

The information needed to *detect* (not fix) these is already computed and
was thrown away: the cost margin between the chosen slot and the
alternative. **Implemented:** `PlayerObs.id_margin` (margin to the best
alternative slot), written to the overlay as `idm`. Near-zero `idm` frames
are "the number shown may be swapped" — visible in stored data, usable by
the UI, and a future auto-flag for numbering audits. Assignment behaviour is
deliberately unchanged (no eval data here to validate a hysteresis change).

### 1.6 Silent fallbacks that change output statistics — audited

| Fallback | Changes statistics? | Was it surfaced? | Now |
|---|---|---|---|
| OpenVINO→torch/CPU (pose) | No (same weights), 6× slower | log only | unchanged (acceptable: latency-only) |
| Mid-run `.track()` exception→CPU, permanent | No | log only | unchanged (same) |
| TrackNet/external unavailable → classical | **Massively** — det rate collapses (§1.2), diameter statistics differ, conf scale differs | log line at startup only; stored overlay had *no trace* of which detector made each point | **fixed**: per-point `src` field (`corr/ext/tn/cls/interp`) in the overlay + per-window source-mix log |
| Degenerate homography → f = img_w·0.9 | **Yes** — all heights systematically wrong (~2× here) | log only | **fixed**: `focal_degenerate` flag surfaced in the job message + implied-height cross-check catches the non-degenerate-but-wrong case |
| TrackNet diameter from heatmap blob | **Yes and unfixable at the source**: TrackNetV2's blob extent reflects its fixed training-label σ, not the ball's physical size — feeding it to size→depth yields confidently wrong heights | not at all | **fixed**: Pass 2 drops TrackNet diameter (position+conf only); height honestly degrades to plane projection |

### 1.7 Existing-but-unused uncertainty signals (per the "use what exists" rule)

Used now: identity cost margin (→ `idm`), diameter (→ `height_sigma` gate),
homography (→ bounce positions, geometry cull, implied-height check), ball
source (→ `src`). Deliberately *not* added: a new confidence model, Kalman
per-point measurement-noise weighting (plausible, but untestable without the
GT clips — noted as future work), TrackNet sub-threshold peak mining (the
eval harness already surfaces `DETECTOR_SUBTHRESHOLD`; `tracknet_conf` is
now learnable so the loop can act on it).

---

## 2. Per-layer verdicts

### 2.1 Frame I/O / windowing — **keep, one bug fixed, one risk flagged**
Merge/pad reconstruction in `pass2.py` is sound and the inclusive/exclusive
frame arithmetic (`end_frame=we+1` vs inclusive `we`) is consistent.
**Bug (fixed):** with an even gap `g` between merged windows and `pad ≥ g/2`,
`we_prev == ws_next` — one frame processed twice and written by both windows.
Now clamped with `(gap−1)//2` on the left side.
**Risk (flagged, not fixed):** `iter_frames` does one `CAP_PROP_POS_FRAMES`
seek per window despite the "no seeking" docstring. On long-GOP H.264/HEVC,
OpenCV's seek is usually frame-accurate with the FFmpeg backend but is not
guaranteed; a ±1–2 frame slip would silently misalign Pass 2 output with
`ball_corrections` and external tracks (both keyed by decode-order index).
If misalignment is ever observed, decode sequentially from the previous
keyframe or from frame 0 per run. Also `probe()`'s `CAP_PROP_FRAME_COUNT` is
container metadata, not ground truth — benign here (only clamps the last
window).

### 2.2 Player detector + tracker — **keep; exclusion zones superseded by geometry**
YOLO11-pose-nano + ByteTrack + per-instance model + OpenVINO FP16 is the
right stack for this target, and the measured 1920-knee (100 % far recall @
74 ms) is a defensible trade — at ~13.5 fps inference the pipeline is offline
anyway, and the far receiver drives both numbering and the (dormant) serve
gate. Invariants (per-instance model, keep `id=None`) are respected and
well-commented. The hand-tuned triangles were the weak point: **replaced as
the primary mechanism** by a homography-derived cull (foot projection > 4 m
outside a sideline / 8 m beyond a baseline ⇒ drop), which adapts to any
framing; triangles retained as fallback. Synthetic-negative-id detections:
verified they cannot pollute speed continuity (ids never repeat, so no false
pairs; slotted ones key on the slot) — they simply don't contribute until
slotted, which is correct.

### 2.3 Player identity — **keep for ≤4 known players; failure now detectable**
HS-histogram + court distance is the right weight class for this roster and
compute budget; a deep ReID embedding is not justified. Its real defect was
silence, not accuracy — addressed with `idm` (§1.5). If swaps prove common
on same-kit matches, the next cheap step (within budget) is an assignment
hysteresis: keep the previous binding when the margin is below a small
threshold. Not implemented — needs labeled numbering GT to validate.

### 2.4 Ball detector priority chain — **keep the architecture; demote the classical fallback's role in expectations**
Manual > external > TrackNet > classical is the right shape, and the
external-track path is the correct primary integration. Two real bugs found
and fixed in the chain's mixed-source behaviour (the doc asked to verify it):
the **classical detector's 3-frame state and TrackNet's 3-frame buffer went
stale across frames covered by higher-priority sources**, producing garbage
for the first frames after every coverage gap. Both now update every frame.
`tracknet_conf` is now a learnable param (was hardcoded 0.5). TrackNet's
pseudo-diameter no longer feeds size→depth (§1.6).
**On keeping the classical fallback:** measured on the available clip it
contributes ~nothing (0 confirmed picks in 9k frames) while carrying the
paddle-sequence FP risk (§1.2). It is still worth keeping *as a fallback*
(it is the only zero-setup path, and its FPs are largely contained by the
selector/Kalman/arc stack), but no accuracy expectations should rest on it;
the honest posture — now visible via `src` — is "no external/TrackNet track
⇒ this clip effectively has no ball data".
**TrackNet fine-tuning is the right mandated path** for a self-contained
deploy: the WASB architecture mirror is sound, OpenVINO conversion is the
right runtime, and `ball_corrections` is the right training source. (Not
validated here: no weights on this machine.)

### 2.5 Ball trajectory/selection — **keep; concrete break scenario documented**
Motion-primary selection with the two-frame cold-start invariant is the
correct defence given a shape-blind detector, and the invariant genuinely
prevents shoe seeding. Where it breaks: (a) the paddle-swing sequence of
§1.2 — fast, off-box, mutually-nearest for 2+ frames — seeds a false track
legitimately under every gate; (b) sparse candidates (this clip) mean
cold-start never fires at all — a recall failure, not a precision one;
(c) after `max_gap` (0.5 s) the track drops and must re-seed, so a long far-
court dropout costs an additional 2 frames minimum. These are detector-
quality limits; the selector logic itself is sound and was left unchanged.

### 2.6 Geometry / 3D — **keep the camera model; height decoupled from size where size is uninformative**
The homography fit (RANSAC over named points), focal self-calibration, and
small-object depth model are individually correct — verified by exact
round-trip reconstruction against the stored calibration. Two findings:
(a) **det(R) = −1**: the recovered "rotation" is improper, *forced* by the
court frame being left-handed (x right, y toward camera, z up). It is
self-consistent everywhere R is used as an orthogonal map (all current code)
but will silently mirror any future Rodrigues/Euler/quaternion use — now
documented as an invariant in `CameraModel`. (b) The far-court sigma
explosion (§1.1) — handled with the `height_sigma` gate rather than a
rewrite: near-court 3D (serves, kitchen play, near bounces) keeps working;
far-court output honestly reports height-unknown and uses plane projection
for position. The fixed origin/handedness assumptions are safe *given* the
camera-above-plane flip, which the audit exercised.

### 2.7 Event detection — **adequate as visualization-only, now with honest inputs; GRU featurization has two skews to fix before enabling**
The heuristic (`detect_events`) is acceptable for overlay markers once its
inputs stop lying: with far heights NaN'd it can no longer emit noise-driven
far bounces, and bounce positions/in-out now come from the homography
(exact on the floor). The 2 cm gravity check is meaningful only where
heights survive the sigma gate — which is now exactly where it applies.
The arc-junction path (`ball_filter.py`) is the better event engine but is
tuned for a dense track; enabling it with the classical detector's sparse
bursts is documented as counterproductive — the current default (off) is
right until an external/fine-tuned track is the norm.
**BounceGRU featurization** (`trace_to_features`): shared train/infer code
means no *code-path* skew, but (a) pixel coords divided by a constant 1000
make features resolution-dependent (a 4K clip shifts the input distribution),
(b) `vx, vy` are per-*entry* differences not normalized by frame gap, so a
dropout changes velocity scale, and (c) training windows (human labels) vs
inference windows (padded Pass 1) have different boundary context. Fix all
three *before* first training — they're free now and expensive after weights
exist. Not fixed in code here since the model is unbuilt and any change to
the featurizer now is unconstrained by weights.

### 2.8 Persistence / params — **keep; gaps closed**
Chunked-JSON overlays and clamped, versioned global params are the right
design, and the global-vs-clip routing is respected everywhere I checked
(corrections and external tracks stay on the clip; nothing clip-local writes
into `learned_params`). Previously-hardcoded values now learnable:
`tracknet_conf`, `ball_court_margin_x_m` / `ball_court_margin_y_m` (the
`_in_court_region` 2.5/3.0), and the new `ball_height_max_sigma_m`. Still
hardcoded and deliberately left so (constants, not tuning candidates):
detector `min_area`/`max_area` (differ between Pass 1 and Pass 2 —
resolution-dependent by design), Kalman `process/meas_noise`, identity
weights (`POS_SCALE`, `APP_W`, `HIST_ALPHA`), and the geometry-cull margins.
Flagging for completeness: `GEOM_CULL_*` could graduate to params if a
deployment ever needs it.

---

## 3. Cross-cutting checks

- **Coupling to dormant rally code:** the live path feeds `rally_logic`-shaped
  inputs (`world`, `player_speed`, `serve_frames`) that have changed meaning:
  (a) far-court `world` heights are now NaN where unreliable — `detect_end`'s
  net-fail scan (`np.isnan(w.height) → skip`) and the phantom-out guard
  already handle NaN correctly, but `_find_start`'s speed threshold uses
  court-plane gradients that are *less* noisy now (plane projection), a
  silent improvement; (b) `serve_frames` only populates when the
  serve-receive gate passes, which depends on detecting the far receiver —
  in doubles, if the diagonal receiver is the occluded rear player,
  `assign_numbers_at_serve` never fires for that rally and numbers ride on
  stale anchors. That is a **live** issue for numbering, not just a dormant
  one: consider falling back to `is_serve_formation` when the strict gate
  hasn't passed for N seconds. (c) `p1_end_settle_s` tight-end logic means
  Pass 2's padded window can end *before* `rally_logic`'s notion of
  confirm-window — if re-enabled, re-audit `confirm_window_s` against the
  new tight ends.
- **Train/serve skew:** GRU featurization issues in §2.7. The eval harness
  had a real one — it constructed `MotionBallDetector` without
  `motion_thresh`/`min_circularity`, so it evaluated at class defaults
  (12 / 0.35) while production runs learned values (3 / 0.30): the harness
  wasn't measuring what ships (candidate count changed 196→431 on this clip
  when fixed). **Fixed.**
- **Simplification opportunities:** the biggest one is philosophical, not
  structural: the pipeline's complexity is mostly *compensating for detector
  weakness* (selector gates, foot suppression, persistence tracks, arc
  rescue, Kalman restart logic). With a trustworthy dense track (external or
  fine-tuned TrackNet) more than half of that machinery is inert
  pass-through, which is the correct end state — I would not remove it now,
  but I would not extend it either. One concrete removal candidate: the
  HSV appearance gate on the classical detector (off by default, per-clip
  colour makes it clip-local state in global params — a routing smell).

---

## 4. Prioritized issue list (severity × effort)

| # | Issue | Sev | Effort | Status |
|---|---|---|---|---|
| 1 | Far-court size→depth noise/bias drives heights, bounce & in/out calls (§1.1) | Critical | S | **Fixed** (sigma gate + plane-projected bounces) |
| 2 | Bad-but-not-degenerate calibration undetectable; degenerate fallback ~2× off and log-only (§1.3) | High | S | **Fixed** (implied-height cross-check + surfaced flags) |
| 3 | Ball source untracked in overlay — silent fallback invisible in stored data (§1.6) | High | S | **Fixed** (`src` per point + per-window mix log) |
| 4 | Detector state staleness across mixed sources (classical diff state, TrackNet buffer) | High | S | **Fixed** |
| 5 | TrackNet heatmap "diameter" fed to size→depth (label-σ, not physical size) | High | S | **Fixed** (position+conf only) |
| 6 | Eval harness detector-param skew vs production | High | S | **Fixed** |
| 7 | Exclusion triangles fragile to framing; calibration already encodes the answer | Med | S | **Fixed** (geometry cull primary, triangles fallback) |
| 8 | Identity swaps silent (§1.5) | Med | S | **Fixed** (surfaced as `idm`; behaviour unchanged) |
| 9 | `tracknet_conf`, court-region margins hardcoded outside `LearnedParams` | Med | S | **Fixed** |
| 10 | Even-gap 1-frame window overlap (double processing/writes) | Low | S | **Fixed** |
| 11 | `eval_matrix.py` hardcoded Windows venv path | Low | S | **Fixed** (`sys.executable`) |
| 12 | Classical fallback ~zero recall on real 60 fps footage — expectations, not code | High | — | Documented; posture change (§2.4) |
| 13 | GRU featurization: resolution-dependent scale, gap-blind velocity, window-context skew | Med (dormant) | S | Documented — fix before first training |
| 14 | Serve-receive numbering anchor starves when far receiver occluded | Med | M | Documented; proposed fallback |
| 15 | Per-window seek accuracy on long-GOP video (index misalignment risk) | Low (latent) | M | Documented; mitigation stated |
| 16 | Far-player occlusion display gap (recoverable at track level) | Low | M | Proposed (coast + `interp`-style flag) |
| 17 | det(R) = −1 latent trap for future pose math | Low (latent) | S | **Fixed** (documented invariant in `CameraModel`) |

## 5. Rebuild proposal

**No subsystem warrants a rewrite.** Every layer is the right weight class
for the camera/compute constraints; the defects were (a) one geometric
overreach — trusting apparent size where it carries no information — and
(b) systematic silence about degradation. Both are fixable surgically, and
were. The changes preserve every §7 invariant: per-instance models,
`id=None` retention, two-frame cold start, permissive detector +
court-region filter (margins now learnable, defaults identical), the
phantom-out guard (strengthened — homography positions make its distance
test meaningful at the far court), and identity-slot persistence.

**The one strategic decision** confirmed rather than changed: ball accuracy
is capped by the detector, not the pipeline. The priority order for effort
is (1) external dense track integration (already built, now honestly
labelled via `src`), (2) TrackNet fine-tuned on `ball_corrections` at
**crop-level resolution** — the far ball is ~5 px in a 1920 frame, which is
~1.3 px after TrackNet's 512×288 resize; fine-tuning on full-frame resizes
cannot fix the far court, so train/infer on a court-anchored crop of the far
half (the homography gives the crop box for free) — and (3) only then event
logic. 60 fps + faster shutter is the highest-leverage capture-side change
(shrinks the blur bias in §1.1 directly).

**Validation plan** (blocked here: the GT-marked clips
`760f1e0e…`/`17be6eb5…` referenced by `eval_matrix.py` are not in this DB,
and this machine has no Arc GPU/TrackNet weights — stated per the "say so
explicitly" rule):
1. `backend/scripts/eval_matrix.py` on the two GT clips. Expect: `med px` /
   `on-ball %` unchanged (position path untouched for classical), **bounce
   precision up** and far-court `in/out` accuracy up (plane-projected
   bounces + sigma gate kill the noise-driven far events), possibly bounce
   *recall* down at the far court — those recalled events were
   height-noise coincidences; verify against the per-rally FP table.
2. `eval_ball_filter.py --diagnose` per config: the failure-attribution
   categories (`BOUNCE_TRACK_HOLE` vs `BOUNCE_SWALLOWED`) should shift
   toward track holes, confirming the ceiling is detector coverage.
3. Re-run with `--set ball_height_max_sigma_m=999` to A/B the sigma gate in
   isolation (0.35 default vs effectively-off).
4. `ball_gt_io.py import` the shared `reference_data/ball_gt.json` first so
   both machines score the same marks.
5. Calibration cross-check: perturb one stored corner by +6 px, re-run
   Pass 2, confirm the implied-height warning fires; restore.

## 6. Code changes implemented

- `cv/court.py` — `focal_degenerate` flag; `height_sigma()`;
  `implied_person_height()`; det(R) = −1 invariant documented.
- `cv/bounce.py` — `reconstruct_world(height_sigma_max=…)`: NaN heights +
  plane-projected court positions where size is uninformative; NaN-aware
  height smoothing; `detect_events(H=…, ball_px=…)`: homography-exact bounce
  positions and phantom-out distances.
- `cv/pass2.py` — degenerate-focal surfacing; implied-height cross-check with
  end-of-run verdict; window even-gap overlap fix; detector-state continuity
  across mixed sources; `src` per ball point (incl. `interp`, cleared with
  arc-rejected points) + per-window source-mix logging; TrackNet diameter
  dropped; `tracknet_conf` + court-region margins from params; `idm` in
  player entries; sigma gate wired to `reconstruct_world`.
- `cv/players.py` — homography-derived geometry cull (`GEOM_CULL_*`);
  `PlayerObs.id_margin`.
- `cv/identity.py` — assignment-cost margin computed and attached.
- `learning/params.py` — `tracknet_conf`, `ball_court_margin_x_m`,
  `ball_court_margin_y_m`, `ball_height_max_sigma_m` (+ bounds).
- `scripts/eval_ball_filter.py` — production-parity detector construction;
  sigma gate + homography bounce positions mirrored so the harness measures
  what ships.
- `scripts/eval_matrix.py` — interpreter path portability.

Verified: byte-compile + lint clean across all touched files; geometry
round-trip and sigma numbers reproduced against the stored calibration;
identity-margin smoke test; full `eval_ball_filter.py` run end-to-end on the
available clip (exercises the modified reconstruct/event/filter paths).

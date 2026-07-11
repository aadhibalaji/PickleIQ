# Pass 1 & Pass 2 — Rally Boundary Logic

> **Scope of this document.** How rally **start/end boundaries** are decided:
> how Pass 1 opens and closes a rally window, and how the Pass 2 *refinement*
> module decides whether to **keep** a Pass 1 window, **move its start later**,
> or **end it sooner**. This is a focused companion to `PASS1_PASS2_SPEC.md`
> (which covers the broader detection pipeline). Exact params and math are
> quoted below; every threshold named is a config/param key.
>
> ---
>
> ## ⚠️ MAINTENANCE CONTRACT — read this if you are an AI agent or developer
>
> **This file must be kept in sync with the code.** If you change ANY of the
> behavior, thresholds, params, file locations, or decision logic described
> here — or you read this file and notice it already disagrees with the code —
> you **must update this document in the same change**. Specifically:
>
> 1. Update the affected section(s) below with the new logic/numbers.
> 2. Update `Key files & symbols` if a function/file moved or was renamed.
> 3. Bump the `Last verified against code` date at the bottom and add a one-line
>    entry to `Change log`.
> 4. If the change also affects `PASS1_PASS2_SPEC.md`, update that too.
>
> Do not let this file silently rot. A stale boundary-logic doc is worse than
> none — downstream agents trust it as source of truth.

---

## 0. Two meanings of "Pass 2" — do not conflate

- **`cv/pass2.py`** — detection / visualization ONLY. Emits **no** `rally_labels`
  and makes **no** boundary decisions. Per `PASS1_PASS2_SPEC.md`, Pass 1 is the
  sole authoritative boundary source.
- **`backend/app/refine/`** — the *rally refinement* module (added after the
  original spec). This is what decides **keep / start-later / end-sooner** for
  each Pass 1 window. It runs after Pass 2 via
  `pipeline._refine_after_pass2` → `refine.run_refine_json`.
  - **STATUS: JSON-only / advisory.** It writes `data/refine/<video_id>.json`
    (plus a `.segments.jsonl` confidence log), **NOT** `rally_labels`. So its
    keep/trim decisions are currently a **shadow computation** — Pass 1 windows
    remain the boundaries actually used/shown until the refiner is validated and
    promoted. (Eval-first; see the ball-GT / model-gap note in project memory.)

Everything in §2 ("Pass 2 decisions") refers to the **refine module**.

---

## 1. PASS 1 — how a rally starts and ends

File: `backend/app/cv/pass1.py :: run_pass1` / `close_rally`.
Cheap sampling: `p1_frame_step=3`, `p1_scale=0.4`, so `sampled_fps = fps/3`.

### 1.1 Per-sample gate signal
Two mutually exclusive regimes:

- **Calibrated (PRIMARY):**
  `signal = (f - last_cross) <= p1_cross_window_s * fps`
  The gate is "live" as long as a **net crossing** was seen within
  `p1_cross_window_s` seconds. Live play == the ball keeps crossing the net.
- **Uncalibrated (FALLBACK):**
  `signal = motion > max(p1_motion_floor, baseline * p1_motion_ratio)`
  `        AND (f - last_ball_frame) <= p1_ball_recent_s * fps`
  Defaults: `p1_motion_floor=0.12`, `p1_motion_ratio=1.6`. `baseline` is an EMA
  that adapts ONLY during dead time (α=0.05 down, 0.01 up) so a rally never
  lifts it. Motion is measured inside the court ROI mask only.

**Net crossing** (`_match_fast` + court projection): a blob that is
mutually-nearest between consecutive sampled frames, size-consistent
(`|Δdiameter| ≤ 0.6·max(diameter)`), moving at ball-like speed
(`p1_ball_speed_px=3.5` .. `45` px/frame at low-res). Both endpoints projected
to court coords; crossing fires when `(cy_now − NET_Y)·(cy_prev − NET_Y) < 0`
(court-Y sign flips across the net). Sets `last_cross = f`.

### 1.2 Start/end via hysteresis
- Go **active** after `on_n = p1_on_s · sampled_fps` sustained live samples.
- Go **dead** after `off_n = p1_off_s · sampled_fps` sustained dead samples.
- Flip is back-dated: `flip_at = f − streak·step`.
- Rising flip → sets `rally_start`. Falling flip → `close_rally(rally_start, flip_at)`.

### 1.3 `close_rally` — the emitted boundaries
1. **Blip drop:** if raw `(end_f − start_f)/fps < p1_min_rally_s`, discard.
2. **Tight end:** the hysteresis falling edge lands ~`p1_cross_window_s + p1_off_s`
   (~4 s) AFTER the real end, so instead:
   `end_tight = min(end_f, last_cross + p1_end_settle_s·fps)`
   (`p1_end_settle_s` default `0.6 s` = ball flight-to-land). Falls back to the
   raw flip only if no crossing was tracked (uncalibrated / motion gate).
3. **Padded start:** `s = max(0, start_f − p1_pad_s·fps)`. Start padded for
   lead-in; **end left tight**.
4. **Merge:** if `s ≤ last_row.end + p1_merge_gap_s·fps`, extend the previous
   window instead of emitting a new one (two close rallies can bridge into one —
   an accepted recall trade-off).

Net effect: **generous, recall-favouring** windows — padded/merged start,
ball-flight-tight end.

---

## 2. PASS 2 (refine module) — keep / start-later / end-sooner

Per Pass 1 window: load tracking for `window ± window_margin_s=2.0 s`, then four
stages (`refine/refine.py :: refine_window`).

### 2.1 Stage B — P(live) per timestep (`feature_hz = 15`)
Default classifier `rules` (`refine/classifiers.py`):
`P = σ(bias + Σ wᵢ·featureᵢ)`, `rules_bias = −1.6`.
- Positive weights: `cross_recency 3.2` (exp-decayed, τ=`cross_recency_tau_s=2.5 s`),
  `crossings_3s 0.8`, `ball_speed_n 1.2`, `frac_visible_1s 0.6`,
  `reactivity 0.8`, `max_pspeed_n 0.7`, `ready_frac 0.6`.
- Negative weights: `near_player_time_n −1.4` (ball parked on a player = dead),
  `out_of_court_time_n −1.0`.
- An XGBoost backend exists (F1 ~0.52 vs rules ~0.34) but `rules` is the default
  and the permanent fallback if the model artifact is missing/broken.

### 2.2 DECISION 1 — keep the window at all?
Stage C (`refine/postprocess.py`) turns P(live) into segments:
- **Hysteresis:** enter `enter_thresh=0.7`, exit `exit_thresh=0.25`.
- **Trailing live-ratio filter:** a timestep is in-rally only if `≥ live_ratio=0.7`
  of the last `ratio_window_s=2.0 s` were live.
- **Merge** segments with gaps `≤ merge_gap_s=2.0 s`.
- **Drop** segments shorter than `min_rally_s=2.5 s`.

Then per segment, accept gates — ANY failure ⇒ record `verdict="rejected"` with
a reason:
- No segments at all ⇒ reject `low_live_ratio` (if `max P < enter_thresh`) else
  `too_short`.
- `mean_p_live < mean_p_live_min=0.25` ⇒ reject `low_live_ratio`.
- `serve_gate_score < serve_gate_min=0.25` ⇒ reject `no_serve`.
- refined duration `< min_rally_s=2.5 s` after snapping ⇒ reject `too_short`.

A single Pass 1 window can **split** into several rally records (one per
surviving segment) — this un-does Pass 1's under-segmentation/merging.

**Serve gate score** (`refine/serve_gate.py :: score_serve`) = weighted mean of
five components, weights `{pre_serve 0.25, contact 0.35, flight 0.15,
landing 0.10, rally_evidence 0.15}`:
- **pre_serve:** serve formation held ≥ `formation_hold_s=0.8 s` near seg start.
- **contact:** ball appears/accelerates (`≥ serve_contact_speed_mps=4.0`) within
  `serve_contact_near_m=2.0 m` of a stationary at/behind-baseline player. Also
  the start anchor.
- **flight (booster):** net crossing `serve_flight_min_s=0.3` .. `serve_flight_max_s=2.0 s`
  after contact; full credit if it flies away from the server.
- **landing (booster):** first bounce beyond the kitchen on the receiver's
  diagonal.
- **rally_evidence:** crossing rate normalized to `rally_evidence_x10s=5.0`/10 s,
  multiplied by "deep fraction" (fraction of visible ball beyond kitchen / 0.3)
  — discounts kitchen dinking drills.

### 2.3 DECISION 2 — move the start LATER (`refined_start`)
```
anchor = gate.contact_t                         # detected serve contact, if found
       else min(t_start, first_crossing − 0.9)  # back off a crossing by serve-flight time
       else t_start                             # Stage C rising edge
refined_start = max(span_lo, anchor − pre_roll_s=0.5)
```
Because Pass 1 padded the start back by `p1_pad_s`, anchoring to the actual serve
contact (−0.5 s pre-roll) almost always **pulls the start later**, onto the serve.

### 2.4 DECISION 3 — end SOONER (`refined_end`)
`snap_end` walks back from the segment falling edge to the **last meaningful ball
event** inside `[start, seg_end]` — the `max` time among:
- last **bounce**, last **net crossing**, or last **fast-ball** moment
  (`speed ≥ end_event_speed_mps=3.0`).
- If none exist, keeps the falling edge.

Then `refined_end = min(span_hi, event_t + post_roll_s=1.0)`. Since Pass 1's end
already sits near `last_cross + p1_end_settle_s`, this typically trims a little
more to the genuine last ball event + 1 s (or holds when they agree).

### 2.4b Config override + keep/drop grading loop (2026-07-10)
- **Runtime override.** `serve_gate_min` / `mean_p_live_min` (and any
  `RefineConfig` field) can be overridden at runtime by
  `data/refine/config.override.yaml` — the refiner now loads
  `load_config(override_path())` on every run (`pipeline._refine_after_pass2`,
  `annotate.py :: _refine_records`). Missing file ⇒ pure defaults. This file is
  the single writable surface for these knobs (they are NOT in `learned_params`).
- **Grading loop (measure + recommend).** The Pass 2 tab lists every Pass 1
  window with its keep/drop verdict; the user grades 👍keep/👎drop per window
  (`window_grades` table, keyed on the frame span + re-associated by IoU≥0.5,
  since `pass1_window_id` is re-minted each run). Grades drive a live keep/drop
  accuracy readout and a serve-gate **recommendation** (`refine/recommend.py`
  sweeps `serve_gate_min` against the grades). Recommendations are applied
  manually (writes the override YAML); nothing auto-applies. Keep/drop only —
  this loop does not touch boundaries. Endpoints in `api/annotate.py`;
  UI in `frontend/src/Workspace.jsx` (`PassTab`, `Recommendation`).

### 2.4c Ball court-projection caveat + image-space contact (2026-07-10)
- **Root-cause finding.** `refine/io_loaders.load_tracking` derives the ball's
  court position by projecting its image pixels through the homography with a
  **ground-plane (height=0) assumption**. This is wrong whenever the ball is
  airborne — a serve toss at ~3 m projects several metres behind the baseline
  (e.g. court-Y −3.8 to −7.9). The size→depth `world` cx/cy is too noisy to help,
  and the far ball is often *never* near the ground in a window. Consequence: the
  serve gate's ball-position checks (`contact`, net `crossings`, `rally_evidence`)
  fail/misfire on real serves even though the **pixel** track is accurate.
- **Proper fix — ballistic 3D (ON by default).** `refine/ballistic.py` recovers
  the ball's true 3D by fitting a gravity parabola `P0 + V0 t + ½ g t²` to each
  flight so it projects onto the observed pixel run (reproj RMS ~3px), then
  `io_loaders.load_tracking` overwrites the ground-plane `ball_xy`/`ball_height`
  with it on well-fitting segments (`use_ballistic_3d`, `ballistic_max_rms_px`).
  Correct court positions fix net crossings, `rally_evidence`, flight and bounce
  depth at once — no per-check patch. Verified: the dropped serve went 0.21 → 0.42
  (crossings-in-segment 0 → 3, flight 0 → 0.8, rally_evidence 0 → 0.63) and the
  labeled benchmark improved **F1 0.561 → 0.643** (+6 recovered rallies; precision
  up too). Cost ~5s/clip (disable for large offline sweeps); `rejection_acc` dipped
  slightly. `TrackingData` carries `ball_px`+`player_box` to enable this.
- **Arc contact recovery (part of ballistic 3D).** The ball is untrackable at the
  serve strike (small, fused with the body), so its first detection is already
  mid-flight and `contact` reads 0. `serve_gate._ballistic_contact` extrapolates
  the serve's fitted arc BACKWARD to where the ball originated; when that origin
  is low (near the ground) and at/behind a baseline, that's the strike — recovered
  by physics, not detection. Scored on origin-low + behind-baseline (player
  proximity is a soft bonus, since the extrapolated x is uncertain and the server
  stands to a side); the serving end is inferred from the origin baseline, which
  re-anchors flight + landing. `TrackingData.ball_flights` (from ballistic.py)
  carries the per-arc fit. Verified: the example serve went 0.42 → 0.705
  (contact 0 → 0.73), and the labeled benchmark rose further **F1 0.643 → 0.673**
  (precision AND recall up). Tried but rejected: re-enabling ball detection in the
  player deadzone (`ball_seed_on_player`) — the ball simply isn't cleanly
  trackable at the strike, so seeding-on-player changed nothing; left off.
- **Superseded interim — image-space contact (`serve_contact_use_pixels`, default
  False).** Measures serve contact by ball-pixel→player-box proximity. Kept for
  reference but NOT needed once 3D is correct: with true 3D the ball's first
  detection is legitimately mid-court (the toss was in the deadzone), so `contact`
  reads 0 honestly and flight+rally_evidence carry the serve. The pixel-proximity
  reading was a perspective coincidence (an airborne mid-court ball can overlap a
  far player in the image); leave it off.

### 2.5 Output record (per segment)
`{pass1_window_id, pass1_start/end(_frame), verdict, rejection_reason,
refined_start/end(_frame), confidence{mean_p_live, serve_gate_score,
net_crossings_per_10s}, splits[]}` → `data/refine/<video_id>.json`.

---

## One-line summary
Pass 1 = a **net-crossing recency gate** with hysteresis, emitting
padded-start / ball-flight-tight-end windows that merge when close. Pass 2
(refine) = per-window **P(live) → smoothed segments → four accept gates
(live-ratio, mean-P, serve-gate, min-duration)**, then re-anchors the **start to
serve contact − 0.5 s** and the **end to the last real ball event + 1 s** —
currently JSON-only, not yet promoted to `rally_labels`.

---

## Key files & symbols
- `backend/app/cv/pass1.py` — `run_pass1`, `close_rally`, `_match_fast`, `_court_mask`
- `backend/app/cv/pass2.py` — Pass 2 detection/overlay (no boundaries)
- `backend/app/refine/refine.py` — `refine_window`, `refine_video`, `run_refine_json`
- `backend/app/refine/classifiers.py` — `RulesClassifier`, `XGBoostClassifier` (P(live))
- `backend/app/refine/postprocess.py` — Stage C (`hysteresis`, `live_ratio_filter`, segments)
- `backend/app/refine/serve_gate.py` — `score_serve`, `snap_end`
- `backend/app/refine/config.py` — `RefineConfig` (ALL refine thresholds/weights)
- `backend/app/pipeline.py` — `_refine_after_pass2` hook
- `backend/app/learning/params.py` — `LearnedParams` (all Pass 1 `p1_*` params)
- `PASS1_PASS2_SPEC.md` — broader detection-pipeline spec (companion doc)

---

## Change log
- 2026-07-10 — Initial version. Documents Pass 1 net-crossing gate + `close_rally`
  padding/merge/tight-end, and the refine module's keep / start-later /
  end-sooner decisions. Refine is JSON-only/advisory at this date.
- 2026-07-10 — Added §2.4b: runtime `config.override.yaml` for serve-gate knobs
  and the keep/drop grading loop (`window_grades` table, `refine/recommend.py`,
  Pass 2 grading UI). Keep/drop *logic* and defaults unchanged.
- 2026-07-10 — Added §2.4c: diagnosed the airborne-ball court-projection bug that
  drops real serves as `no_serve`; added an off-by-default image-space contact
  path (`serve_contact_use_pixels`) + `ball_px`/`player_box` on `TrackingData`.
- 2026-07-10 — Ballistic 3D (`refine/ballistic.py`, `use_ballistic_3d` **ON by
  default**) supersedes the image-space hack: fits a gravity arc to the pixel
  track for correct ball 3D. Labeled benchmark F1 0.561 → 0.643. This IS a
  keep/drop behaviour change (default flipped on).
- 2026-07-10 — Arc contact recovery (`_ballistic_contact` + `TrackingData.
  ball_flights`): extrapolate the serve arc back to the untracked strike.
  Benchmark F1 0.643 → 0.673 (precision + recall both up). `ball_seed_on_player`
  Pass 2 toggle added but rejected (no benefit — ball untrackable at the strike).

**Last verified against code: 2026-07-10.**

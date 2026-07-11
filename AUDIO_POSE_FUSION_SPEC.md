# Audio × Visual Fusion, Pose-Aware Ball Exclusion & Serve Marking — Spec

Added 2026-07-10. This document is the source of truth for four Pass-2
confidence/overlay features. **All of it is overlay/confidence only.** Pass 1
remains the SOLE source of rally start/end (see `PASS1_PASS2_SPEC.md`); nothing
here writes `rally_labels`, feeds `rally_logic.py`, or changes rally boundaries.
The LLM stays out of the per-frame loop — this is pure DSP/CV/heuristic.

## Why

The visual detector's biggest error is far-court `apparent-size → depth` noise
(`DETECTION_AUDIT.md`). A paddle/ball contact makes a short broadband "pop" that
is **depth-independent**, so audio is a strong orthogonal signal to confirm hits
and recover ones the visual track missed. Separately, the classical detector
sometimes latches onto a limb/paddle blob moving with a player instead of the
ball — a pose-motion correlation fix, independent of audio.

## Feature 1 — Audio event extraction

- `backend/app/cv/audio_events.py`: ffmpeg mono extract → `scipy` Butterworth
  bandpass → `librosa` spectral-flux onset detection → `(time_s, frame_index,
  onset_strength)`. Productionised form of `backend/scripts/audio_hit_detector.py`.
- Persisted in the **`audio_events`** table (`db.py`). `compute_and_store` is
  **idempotent** per `video_id` (skips if rows exist).
- Runs as a cheap `"audio"` job submitted by `pipeline.start_processing`
  **before** Pass 1 (single FIFO worker → done before Pass 2 chains). No
  calibration dependency. ~4% of realtime, not a bottleneck.
- Tunables (`params.py`): `audio_band_low_hz` (1500), `audio_band_high_hz`
  (6000), `audio_onset_delta` (0.4), `audio_onset_wait_s` (0.35).
- Deps added to `requirements.txt`: `scipy`, `librosa`, `soundfile`.

## Feature 2 — Audio × visual fusion

- `bounce.fuse_audio_events(...)`, called from `pass2.py` right after
  `detect_events`, with `audio_events` rows **clipped to the current Pass 1
  window** (implicit window gating — Pass 2 already loops per window).
- Per in-window onset it scores: **onset strength** (normalised), **trajectory
  kink** (curvature of the court-depth velocity `np.gradient(vy)` already used by
  `detect_events`), **paddle proximity** (min px from the ball to a wrist
  keypoint COCO 9/10). `conf = w_onset·onset + w_kink·kink + w_prox·prox`.
- **Matched** to a visual event within `audio_visual_fusion_window_s` →
  corroboration boost: the event's overlay `evt` gets `audio:true` + `conf`.
- **Unmatched** but has a trajectory kink and `conf ≥ audio_visual_min_conf` →
  **surfaced as a new low-confidence `hit`** marker (recall path for the
  far-court weak-visual case). Wrist proximity **boosts** conf but is not
  required — on the far axial camera pose wrists are frequently missing, and
  requiring them killed the far-court recall this feature exists for.
- **Unmatched, low combined confidence** (cross-court bleed / weak onset) →
  dropped. Bleed is rejected by window-gating + the conf threshold, not by a
  hard proximity gate. Audio is never hard-rejected on its own.
- Tunables: `audio_visual_fusion_window_s` (0.12), `audio_paddle_px` (60),
  `audio_w_onset`/`audio_w_kink`/`audio_w_prox` (0.4/0.35/0.25),
  `audio_visual_min_conf` (0.5).
- Overlay `evt` now optionally carries `conf` (0–1), `audio` (bool). Existing
  consumers ignore unknown keys; the frontend uses none of them yet beyond serve.

## Feature 3 — Pose-aware limb-lock ball exclusion

- `TrajectoryBallSelector.select(..., wrists=…)` in `ball_track.py`. Pass 2
  builds `wrists` from `PlayerObs.keypoints[9/10]` (`_wrist_points`).
- A selected candidate is dropped once it has moved in **lockstep** with a nearby
  wrist for `limb_lock_frames` consecutive frames: within `limb_lock_dist_px`
  **and** sharing the wrist's frame-to-frame velocity within
  `limb_lock_rel_vel_px` (constant offset / ~zero relative velocity). A real ball
  merely close to a paddle for one contact frame never accumulates the run, so
  hits survive. Decoupled from audio — `wrists=None` = guard off, unchanged.
- Tunables: `limb_lock_dist_px` (30), `limb_lock_rel_vel_px` (4), `limb_lock_frames` (3).

## Feature 4 — Serve marking

- `pass2._mark_serve`: the first fused **hit** within `serve_window_start_s` of a
  Pass 1 window start is promoted to `evt.kind == "serve"`. Prefers an
  audio-corroborated hit; falls back to the first hit flagged `low_conf`. Stance
  cross-check reuses Pass 2's existing `serve_frames`
  (`is_serve_[receive_]formation`) — no new geometry; when stance isn't
  confirmed the serve is flagged `low_conf`.
- Stored in the **overlay `evt` channel** (`kind:"serve"`), no new table.
- UI: `VideoPlayer.jsx` draws a gold-green ring + `SERVE (AI)` / `SERVE? (AI)`
  label; `Timeline.jsx` renders an `S` tick row fed by a new `serves` prop, which
  `Workspace.jsx` loads from `GET /api/videos/{id}/serves` (reads the pass2
  overlay for `evt.kind=='serve'`). Clearly AI-inferred, distinct from the
  human B/H/N ground-truth channel (`ball_corrections`).
- Tunable: `serve_window_start_s` (1.5).

## Validation

- Self-checks: `python -m app.cv.audio_events` (synthetic click train);
  `python -m app.cv.ball_track` (limb-lock glued-vs-crossing candidate).
- Offline eval: `scripts/eval_audio_fusion.py` — per clip: `audio_all`,
  `audio_inwin`, `evt_total`, `fused`, `serves`, `demoted` (in-window onsets not
  corroborated ≈ cross-court bleep rejected). Cheap precision proxy, no manual GT.
- Regression anchors on `Second 3 mins.mp4`: real hits ~1.87/3.41/158s should
  fuse; ~6.11s adjacent-court bleed should land in `demoted`.

## Known limitations

- **Mono audio on the validated camera** (channels bit-identical) — no
  directional/stereo localization is attempted. Cross-court bleed is rejected by
  *visual corroboration*, not by direction. May differ on other venues' cameras.
- **No paddle keypoint** in COCO-17 — the wrist is a proxy for the contact point,
  so paddle-proximity is approximate.
- **Audio-only surfaced hits are low-confidence** by construction.
- **Fusion + serve run only on the legacy heuristic event path** (`detect_events`,
  the default `ball_arc_enabled=0`). The opt-in arc path (`classify_arc_events`)
  is not audio-fused; it emits `ArcEvent`, not `BallEvent`.
- `demoted` is an approximate proxy (`audio_inwin - fused`), not a matched count.

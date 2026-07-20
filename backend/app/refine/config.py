"""Single config for the whole refinement pipeline.

Every window, threshold and weight used by features / classification /
post-processing / serve gate lives here — nothing is hardcoded in the stage
modules. Load overrides from YAML with :func:`load_config`.

Court frame (see cv/court.py): x in [0, 6.096], y in [0, 13.4112] metres,
net at y = NET_Y = 6.7056. "Net side" throughout = sign(y - NET_Y):
-1 = far half, +1 = near half.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..paths import repo_root

_ROOT = repo_root()


def override_path() -> Path:
    """The single writable YAML that overrides refine defaults. Read on every
    refine run (pipeline + on-demand regen); written by the grade-recommendation
    'Apply'. Missing file => pure defaults (load_config handles that)."""
    return _ROOT / "data" / "refine" / "config.override.yaml"


@dataclass
class RefineConfig:
    # ---- timeline -------------------------------------------------------
    feature_hz: float = 15.0        # feature timeline rate (native rate is ~30)
    window_margin_s: float = 2.0    # analysis margin around each Pass 1 window
    history_s: float = 1.5          # sliding history window for stacked features

    # ---- Stage A: features ---------------------------------------------
    # Ballistic 3D: replace the ground-plane ball projection with a gravity-arc
    # fit to the pixel track, so court x/y AND height are correct even when the
    # ball is airborne (fixes net crossings, contact, bounces at once). See
    # refine/ballistic.py. ON by default — improved the labeled benchmark F1
    # 0.561 -> 0.643 (+6 recovered rallies). ~5s/clip; disable for fast sweeps.
    use_ballistic_3d: bool = True
    ballistic_max_rms_px: float = 12.0   # accept a flight fit under this reproj error
    speed_diff_frames: int = 3      # finite-difference span for ball speed
    speed_median_k: int = 5         # median smoothing kernel (odd)
    visible_window_s: float = 1.0   # frac_visible_last_1s
    cross_max_gap_s: float = 1.2    # max invisible gap across which a side flip counts
    crossings_window_s: float = 3.0
    court_margin_m: float = 0.5     # ball_in_court margin
    near_player_m: float = 1.2      # "ball near player" radius (court metres)
    deadzone_m: float = 1.2         # disappearance within this of a player => near them
    time_cap_s: float = 10.0        # cap for all time_since_* features
    ready_margin_m: float = 1.5     # ready position: inside court+margin, own side
    stationary_mps: float = 1.0     # "stationary" player speed (pose jitter!)
    behind_baseline_m: float = 1.5  # soft ramp: at/behind baseline -> this far in
    formation_hold_s: float = 0.8   # formation must hold this long (serve pre-phase)
    reactivity_window_s: float = 2.0
    reactivity_spike_mps: float = 1.2   # player speed spike threshold
    reactivity_match_s: float = 0.5     # spike within this of a crossing counts
    bounce_speed_drop: float = 0.45     # fallback bounce: speed drops below this frac
    ball_speed_norm_mps: float = 12.0   # normalization for rules classifier
    max_speed_mps: float = 45.0         # physically plausible cap; above = noise

    # Player-motion signal: turning away from the net and retreating toward
    # one's own baseline is real evidence a point just ended (the documented
    # gap — the rules classifier over-leans on cross_recency/ready_frac and
    # under-weights actual player motion). Two parts, both required to avoid
    # misreading a brief in-rally glance/step-back as "point over":
    #   1. sustained court-Y motion away from the net, held for
    #      baseline_retreat_hold_s (hysteresis-style duration guard, same
    #      spirit as p1_on_s/p1_off_s) — a single-frame blip can't fire it;
    #   2. shoulder-line body orientation (from pose keypoints) facing away
    #      from the net, as a confidence booster on top of #1.
    baseline_retreat_hold_s: float = 1.0     # retreat motion must sustain this long
    baseline_retreat_speed_mps: float = 0.5  # min court-Y speed toward own baseline
    baseline_retreat_ratio: float = 0.7      # frac of the hold window that must show it

    # ---- Stage B: classifier ---------------------------------------------
    classifier: str = "rules"       # "rules" | "xgboost"
    model_path: str = "data/models/refine/xgb.json"
    # RulesClassifier weights (feature -> weight); bias chosen so an idle
    # court sits well below 0.5 and rhythmic crossings push well above.
    #
    # 2026-07-10 reweight (merged-window split fix): weights re-derived from
    # measured live-vs-gap separability over the 55 labeled Pass 1 windows
    # (AUC: ball_speed_n 0.88, ball_in_court 0.84, mean_pspeed_n 0.77 vs
    # cross_recency only 0.64 — between-point ball knocks refresh crossings,
    # so cross recency barely separates a rally from its dead gap). Leaning
    # on ball speed / ball-in-court / player motion and dropping the bias to
    # -2.0 lets P(live) fall below exit_thresh between points: merged Pass 1
    # windows now split into their component rallies (benchmark F1
    # 0.700 -> 0.750, recall 0.593 -> 0.661, precision/rejection held).
    rules_bias: float = -2.0
    rules_weights: dict = field(default_factory=lambda: {
        "cross_recency": 1.0,       # exp-decayed time since last net crossing
        "crossings_3s": 0.8,
        "ball_speed_n": 3.0,        # strongest live/dead discriminator
        "ball_in_court": 1.5,
        "frac_visible_1s": 0.6,
        "reactivity": 0.8,
        "mean_pspeed_n": 1.5,       # players run during a point, stop between
        "max_pspeed_n": 0.7,
        "near_player_time_n": -1.4, # ball parked near one player = dead
        "ready_frac": 0.2,
        "out_of_court_time_n": -1.4,
        # 2026-07-14: player-motion signal (see baseline_retreat_* above).
        # Raised -1.0 -> -1.6: now the strongest negative term (above
        # near_player_time_n / out_of_court_time_n at -1.4), since a
        # sustained turn-and-retreat is harder to trigger falsely (needs both
        # baseline_retreat_hold_s sustain AND ratio, orientation is a bonus
        # multiplier) than "ball happens to be near a player" — still well
        # under ball_speed_n (3.0) so live ball evidence isn't overridden.
        "facing_away_retreat": -1.6,
    })
    cross_recency_tau_s: float = 2.5    # decay constant for cross_recency

    # ---- Stage C: temporal post-processing ------------------------------
    # Defaults = best of the 2026-07-08 sweep on the 5 labeled clips
    # (F1 0.627, precision 0.778, rejection acc 1.0). Re-sweep as GT grows.
    enter_thresh: float = 0.7
    exit_thresh: float = 0.25
    live_ratio: float = 0.7
    ratio_window_s: float = 2.0
    merge_gap_s: float = 2.0
    # 2026-07-14: min_rally_s history 2.5 -> 3.0 -> 3.5 -> 2.5 (final value).
    # The 3.0/3.5 bumps were chasing the "too short" penalty as a single
    # duration term, but diagnostics on real drops showed sub-3.5s Pass 1
    # windows with p_live_max=1.0 being killed by the EARLY Stage C filter
    # (postprocess.py) before serve gate / anchor / trim ever ran, plus real
    # rally time getting clamped off the front by the pre_roll_s anchor
    # (refine.py) — i.e. trimming, not actual duration, was the cause. Split
    # into two thresholds: min_rally_s_early (below) kills only obvious noise
    # blips pre-serve-gate, and this value is now the ONLY real accept/reject
    # duration gate, applied post-trim in refine.py. Set to 2.5s as a starting
    # point — a served point can end in one unreturned return and run under
    # 3s. Re-sweep against labeled data once available (evaluate.py
    # DEFAULT_GRID currently tests 1.5/2.5; add 2.0/3.0 and pick by F1).
    min_rally_s: float = 2.5
    # Early Stage C noise-blip filter (postprocess.py only) — deliberately far
    # below min_rally_s so a real rally that got fractured by a brief
    # mid-point confidence dip still reaches serve gate / anchor / trim, where
    # the real length decision belongs. Only meant to drop sub-second flicker
    # (a ball knock, a tracking glitch), not to gate rally length.
    min_rally_s_early: float = 1.25

    # ---- Stage D: serve gate + snapping ----------------------------------
    serve_search_s: float = 3.0     # search window after the segment rising edge
    serve_lookback_s: float = 3.0   # also search this far before the rising edge
                                    # (the smoothed rising edge lags the serve
                                    # by up to ratio_window_s + flight time)
    serve_gate_min: float = 0.25
    serve_contact_speed_mps: float = 4.0    # min initial ball speed at contact
    serve_contact_near_m: float = 2.0       # contact this near the server (court m,
                                            # fallback when no pixel track)
    # Measure serve contact by BALL-PIXEL -> player-BOX proximity (when the ball
    # is airborne) instead of the court-metre distance. Fixes serves the height-
    # corrupted court projection drops (verified: a real serve 0.21 -> 0.75), but
    # image proximity is more permissive and slightly regressed the tiny 7-clip
    # benchmark (-1 window) — off until representative airborne-serve clips are
    # graded into the eval set. Flip on via config.override.yaml to A/B.
    serve_contact_use_pixels: bool = False
    serve_contact_near_px: float = 130.0    # contact this near the server BOX in
                                            # image px, used ONLY when the ball is
                                            # airborne (court pos then unreliable)
    serve_airborne_h_m: float = 0.7         # ball above this height => trust the
                                            # PIXEL proximity over the court-metre
                                            # one (plane projection is only valid
                                            # when the ball is near the ground)
    serve_min_gap_s: float = 0.2    # ball invisible at least this long before "appearance"
    serve_flight_min_s: float = 0.3
    serve_flight_max_s: float = 2.0
    # 2026-07-14: a net crossing at/near ground height (h < this) doesn't
    # count as a serve flight — distinguishes a genuinely airborne served
    # ball from a ball rolling across/near the net line. Only enforced when
    # ball_height data exists (ballistic 3D fit succeeded); no height data =>
    # can't verify, not penalized. Well below serve_airborne_h_m (0.7, a
    # DIFFERENT threshold used for court-projection reliability at contact,
    # not flight confirmation) since a serve's arc can be shallow well before
    # its peak — this only needs to rule out "on the ground".
    serve_flight_min_height_m: float = 0.15
    # 2026-07-14: independent floors alongside flight_or_landing_ok — a net
    # crossing or bounce can happen during ordinary rally play too, not just
    # right after a real serve, so flight/landing evidence alone still isn't
    # sufficient. A real serve needs genuine pre-serve formation AND genuine
    # ball-at-player contact evidence as well (confirmed false positive:
    # contact=0.276, flight=0.8, old gate passed at 0.567). Starting points
    # from the suggested 0.35-0.4 / ~0.3 ranges; re-sweep once labeled data
    # can validate them like min_rally_s.
    serve_pre_serve_min: float = 0.35   # diagnostic floor only, no longer gates alone
    serve_contact_min: float = 0.3
    # 2026-07-14: near-floor contact compensation — a directionally-confirmed
    # flight crossing (flight==1.0, i.e. the crossing matched the expected
    # direction away from the server's side, not just "some crossing found")
    # is strong enough evidence that a contact read just under serve_contact_min
    # shouldn't veto the serve on its own (confirmed false negative: contact
    # 0.2815 vs 0.3 floor, flight=1.0, pre_serve 0.361).
    serve_contact_near_floor_min: float = 0.25
    # 2026-07-14: compensating path — a strong pre-serve formation read PLUS
    # confirmed flight/landing can pass the gate even when contact is weak
    # (occluded/missed ball-near-player detection at the moment of contact is
    # common even for real serves). Set well above serve_pre_serve_min (0.35)
    # since this branch has no contact corroboration at all, so it needs to
    # be more confident on its own. Caveat: pre_serve's shoulder-orientation
    # component is only as trustworthy as pose coverage allows (see
    # pre_serve_pose_coverage) — this compensation path leans harder on
    # pre_serve than the direct-contact path does, so it's more exposed to
    # low pose coverage than the old design was.
    serve_pre_serve_compensate_min: float = 0.55
    # rally_evidence: sustained net-crossing rate inside the segment. Partial
    # stand-in for a serve the sparse ball tracker never saw — a Pass 1 window
    # with rhythmic crossings almost surely began with a serve. Sweepable.
    serve_weights: dict = field(default_factory=lambda: {
        "pre_serve": 0.25, "contact": 0.35, "flight": 0.15, "landing": 0.10,
        "rally_evidence": 0.15})
    rally_evidence_x10s: float = 5.0    # crossings/10 s that count as full evidence
    pre_roll_s: float = 0.5
    post_roll_s: float = 1.0
    end_event_speed_mps: float = 3.0    # "high-speed ball moment" for end snap
    mean_p_live_min: float = 0.25       # below this => rejected "low_live_ratio"

    # ---- Stage E: serve-label dataset (GT construction) -----------------
    # A human rally-START mark IS a serve-contact mark (a rally cannot begin
    # without a serve). The instants immediately around it (pre-serve settle +
    # follow-through) look almost exactly like a serve, so they are EXCLUDED
    # from the negative set rather than labeled negative; only the marked
    # contact is a positive. Everything outside every buffer is a clean
    # negative (implicit — nothing is hand-labeled). See refine/serve_labels.py.
    serve_buffer_s: float = 0.5         # ± exclusion zone around each serve mark
    serve_neg_min_span_s: float = 3.0   # shortest dead-time gap emitted as a
                                        # non-serve span for the gate validator

    # ---- Stage E: training / eval ----------------------------------------
    holdout_clips: list = field(default_factory=list)  # video_ids OR filenames, frozen
    eval_iou: float = 0.5
    xgb_params: dict = field(default_factory=lambda: {
        "max_depth": 4, "n_estimators": 200, "learning_rate": 0.08,
        "subsample": 0.9, "colsample_bytree": 0.9})

    # ---- output -----------------------------------------------------------
    out_dir: str = "data/refine"

    def copy(self, **overrides) -> "RefineConfig":
        """Return a copy with the given fields replaced (used by sweeps)."""
        return dataclasses.replace(self, **overrides)


def load_config(path: str | Path | None = None, **overrides) -> RefineConfig:
    """Defaults <- YAML file (if given) <- keyword overrides."""
    data: dict = {}
    if path is not None and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text()) or {}
    data.update(overrides)
    known = {f.name for f in dataclasses.fields(RefineConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown refine config keys: {sorted(unknown)}")
    return RefineConfig(**data)


def save_config(cfg: RefineConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

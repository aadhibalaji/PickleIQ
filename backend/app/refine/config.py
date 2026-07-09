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


@dataclass
class RefineConfig:
    # ---- timeline -------------------------------------------------------
    feature_hz: float = 15.0        # feature timeline rate (native rate is ~30)
    window_margin_s: float = 2.0    # analysis margin around each Pass 1 window
    history_s: float = 1.5          # sliding history window for stacked features

    # ---- Stage A: features ---------------------------------------------
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

    # ---- Stage B: classifier ---------------------------------------------
    classifier: str = "rules"       # "rules" | "xgboost"
    model_path: str = "data/models/refine/xgb.json"
    # RulesClassifier weights (feature -> weight); bias chosen so an idle
    # court sits well below 0.5 and rhythmic crossings push well above.
    rules_bias: float = -1.6
    rules_weights: dict = field(default_factory=lambda: {
        "cross_recency": 3.2,       # exp-decayed time since last net crossing
        "crossings_3s": 0.8,
        "ball_speed_n": 1.2,
        "frac_visible_1s": 0.6,
        "reactivity": 0.8,
        "max_pspeed_n": 0.7,
        "near_player_time_n": -1.4, # ball parked near one player = dead
        "ready_frac": 0.6,
        "out_of_court_time_n": -1.0,
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
    min_rally_s: float = 2.5

    # ---- Stage D: serve gate + snapping ----------------------------------
    serve_search_s: float = 3.0     # search window after the segment rising edge
    serve_lookback_s: float = 3.0   # also search this far before the rising edge
                                    # (the smoothed rising edge lags the serve
                                    # by up to ratio_window_s + flight time)
    serve_gate_min: float = 0.25
    serve_contact_speed_mps: float = 4.0    # min initial ball speed at contact
    serve_contact_near_m: float = 2.0       # contact must be this near the server
    serve_min_gap_s: float = 0.2    # ball invisible at least this long before "appearance"
    serve_flight_min_s: float = 0.3
    serve_flight_max_s: float = 2.0
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

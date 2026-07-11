"""Stage C — temporal post-processing of the per-timestep P(live).

Order matters and is fixed:
1. hysteresis thresholding (enter high, exit low) — kills flicker;
2. trailing live-ratio window — a timestep counts as in-rally only if most of
   the recent window was LIVE, which bounds boundary lag and absorbs short
   dropouts;
3. merge LIVE segments separated by short gaps;
4. walk each segment start back to the raw hysteresis rising edge (undoes the
   ratio filter's start lag, which otherwise eats short rallies) — AFTER the
   merge, so the walk-back cannot shrink a genuine dead gap below the merge
   threshold and glue two rallies together;
5. drop LIVE segments that are too short to be a rally.

Works purely on sample indices; the caller maps back to times.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RefineConfig


@dataclass
class Segment:
    """Half-open [start, end) in feature-timeline sample indices."""
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


def hysteresis(p: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """Two-threshold state machine: LIVE when p>enter, back to DEAD only when
    p<exit. Returns a bool state array."""
    state = np.zeros(len(p), bool)
    live = False
    for i, v in enumerate(p):
        if not live and v > enter:
            live = True
        elif live and v < exit_:
            live = False
        state[i] = live
    return state


def live_ratio_filter(state: np.ndarray, ratio: float, window: int) -> np.ndarray:
    """Trailing-window majority: in-rally iff >= ratio of the last `window`
    samples are LIVE. Edge-shortened so the span start isn't penalized."""
    if window <= 1:
        return state.copy()
    c = np.cumsum(np.insert(state.astype(float), 0, 0.0))
    i = np.arange(len(state))
    lo = np.maximum(0, i - window + 1)
    frac = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return frac >= ratio


def to_segments(mask: np.ndarray) -> list[Segment]:
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return [Segment(s, e) for s, e in zip(starts, ends)]


def walk_back_starts(segs: list[Segment], hyst: np.ndarray) -> list[Segment]:
    """Extend each segment's start back to the raw hysteresis rising edge.

    The trailing live-ratio filter exists to kill flicker, but it also DELAYS
    every rally start by up to (1-ratio)*window (~1.4 s) — on a short point
    that lag can eat more than half the rally and the min-length filter then
    deletes a real point (seen on the first serve of a clip). The un-filtered
    hysteresis state knows the true rising edge, so walk back to it; ends are
    left to the ratio filter (its trailing window bounds end lag by design).
    """
    out = []
    prev_end = 0
    for s in segs:
        st = s.start
        while st > prev_end and hyst[st - 1]:
            st -= 1
        out.append(Segment(st, s.end))
        prev_end = s.end
    return out


def merge_segments(segs: list[Segment], max_gap: int) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        if out and s.start - out[-1].end <= max_gap:
            out[-1] = Segment(out[-1].start, s.end)
        else:
            out.append(Segment(s.start, s.end))
    return out


def postprocess(p_live: np.ndarray, cfg: RefineConfig,
                hz: float | None = None) -> tuple[list[Segment], np.ndarray]:
    """Full Stage C. Returns (segments, smoothed bool state) — the state array
    is what diagnostics plot alongside raw P(live)."""
    hz = hz or cfg.feature_hz
    hyst = hysteresis(p_live, cfg.enter_thresh, cfg.exit_thresh)
    state = live_ratio_filter(hyst, cfg.live_ratio,
                              max(1, int(round(cfg.ratio_window_s * hz))))
    segs = to_segments(state)
    segs = merge_segments(segs, max(1, int(round(cfg.merge_gap_s * hz))))
    segs = walk_back_starts(segs, hyst)
    min_len = int(round(cfg.min_rally_s * hz))
    segs = [s for s in segs if len(s) >= min_len]
    return segs, state

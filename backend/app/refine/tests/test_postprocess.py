"""Stage C behaviour on synthetic P(live) traces (spec test 2)."""
from __future__ import annotations

import numpy as np

from ..config import RefineConfig
from ..postprocess import hysteresis, postprocess

CFG = RefineConfig()
HZ = CFG.feature_hz


def _p(*spans: tuple[float, float, float], total_s: float) -> np.ndarray:
    """Build a P(live) trace from (t0, t1, value) spans over zeros."""
    p = np.zeros(int(total_s * HZ))
    t = np.arange(len(p)) / HZ
    for t0, t1, v in spans:
        p[(t >= t0) & (t < t1)] = v
    return p


def test_hysteresis_no_flicker_on_noise():
    rng = np.random.default_rng(0)
    # live entry, then noise oscillating BETWEEN the two thresholds
    p = np.concatenate([np.full(10, 0.9),
                        0.5 + 0.15 * rng.uniform(-1, 1, 200)])
    state = hysteresis(p, CFG.enter_thresh, CFG.exit_thresh)
    assert state.all()                       # never drops out mid-noise
    # never entered: same noise from a dead start stays dead
    state2 = hysteresis(p[10:], CFG.enter_thresh, CFG.exit_thresh)
    assert not state2.any()


def test_short_gap_merges():
    # 0.5 s raw dropout. NB the ratio filter's rise lag inflates a raw gap by
    # ~(1 - ratio + ratio) * window in the state timeline, so merge_gap is
    # compared against that inflated gap (stage order is fixed by spec).
    p = _p((1, 5, 1.0), (5.5, 10, 1.0), total_s=12)
    segs, _ = postprocess(p, CFG)
    assert len(segs) == 1


def test_long_gap_stays_split():
    p = _p((1, 5, 1.0), (9, 13, 1.0), total_s=15)   # 4 s gap > merge_gap
    segs, _ = postprocess(p, CFG)
    assert len(segs) == 2


def test_short_blip_dropped():
    p = _p((2, 3.0, 1.0), total_s=8)                # 1 s < min_rally 1.5 s
    segs, _ = postprocess(p, CFG)
    assert segs == []


def test_ratio_window_bounds_end_lag():
    p = _p((1, 6, 1.0), total_s=12)
    segs, state = postprocess(p, CFG)
    assert len(segs) == 1
    end_t = segs[0].end / HZ
    # trailing-ratio flag must clear within (1 - ratio) * window of the drop
    max_lag = (1 - CFG.live_ratio) * CFG.ratio_window_s + 2 / HZ
    assert end_t <= 6 + max_lag

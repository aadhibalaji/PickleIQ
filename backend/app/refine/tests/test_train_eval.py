"""Training split + eval-harness unit behaviour (spec test 7)."""
from __future__ import annotations

import numpy as np
import pytest

from ..evaluate import EvalResult, iou, match_segments
from ..labels import expand_labels
from ..train import split_by_clip


def test_split_is_by_clip_never_both_sides():
    ids = [f"clip{i}" for i in range(7)]
    for seed in range(5):
        train_ids, eval_ids = split_by_clip(ids, 0.3, seed)
        assert train_ids & eval_ids == set()
        assert train_ids | eval_ids == set(ids)
        assert len(eval_ids) >= 1 and len(train_ids) >= 1


def test_split_needs_two_clips():
    with pytest.raises(RuntimeError):
        split_by_clip(["only"], 0.3)


def test_expand_labels_aligns_with_timeline():
    t = np.arange(0, 10, 1 / 15)
    y = expand_labels(t, [(2.0, 4.0), (6.0, 7.0)])
    assert y[(t >= 2.1) & (t <= 3.9)].all()
    assert not y[(t > 4.1) & (t < 5.9)].any()


def test_segment_matching_iou():
    assert iou((0, 10), (0, 10)) == 1.0
    assert iou((0, 10), (5, 15)) == pytest.approx(1 / 3)
    pred = [(0.0, 10.0), (20.0, 30.0), (50.0, 51.0)]
    gt = [(1.0, 10.5), (21.0, 29.0)]
    m = match_segments(pred, gt, 0.5)
    assert sorted(m) == [(0, 0), (1, 1)]


def test_eval_result_metrics():
    r = EvalResult(tp=8, fp=2, fn=2,
                   start_errs=[0.1, -0.2, 0.3], end_errs=[1.0, -1.0, 0.5])
    assert r.precision == pytest.approx(0.8)
    assert r.recall == pytest.approx(0.8)
    b = r.boundary_stats()
    assert b["start_median_s"] == pytest.approx(0.2)
    assert b["end_median_s"] == pytest.approx(1.0)

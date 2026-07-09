"""Serve-label dataset construction — rally-start IS a serve, ±buffer
exclusion, implicit negatives, and the labels.py-native rally file."""
from __future__ import annotations

import json

import numpy as np

from ..config import RefineConfig
from ..labels import load_label_file
from ..serve_labels import (SERVE_SCHEMA_VERSION, build_sidecar,
                            expand_serve_labels, rally_file_items,
                            to_gate_spans)

SPANS = [(10.0, 14.0), (20.0, 26.0), (40.0, 42.0)]   # 3 rallies
DURATION = 60.0
HZ = 15.0


def _t():
    return np.arange(0.0, DURATION, 1.0 / HZ)


def test_serve_event_is_rally_start():
    sc = build_sidecar("vid", SPANS, 30.0, DURATION, RefineConfig())
    assert sc["serve_events"] == [10.0, 20.0, 40.0]          # == the rally starts
    assert sc["rally_spans"] == [[10.0, 14.0], [20.0, 26.0], [40.0, 42.0]]
    assert sc["schema_version"] == SERVE_SCHEMA_VERSION
    assert sc["buffer_s"] == 0.5 and sc["fps"] == 30.0


def test_expand_one_positive_per_serve_always_kept():
    t = _t()
    y, mask = expand_serve_labels(t, [10.0, 20.0, 40.0], 0.5)
    assert int(y.sum()) == 3
    assert mask[y == 1].all()                                # positives never masked
    for s in (10.0, 20.0, 40.0):
        assert y[int(np.argmin(np.abs(t - s)))] == 1.0


def test_buffer_excludes_neighborhood_but_not_positive_or_far():
    t, buf = _t(), 0.5
    y, mask = expand_serve_labels(t, [20.0], buf)
    inside_non_pos = (t >= 20.0 - buf) & (t <= 20.0 + buf) & (y == 0)
    assert inside_non_pos.any() and not mask[inside_non_pos].any()   # excluded, not neg
    far = (t > 20.0 + buf + 0.2) & (t < 40.0)
    assert mask[far].all() and not y[far].any()                     # clean negatives


def test_buffer_size_is_configurable():
    t = _t()
    _, small = expand_serve_labels(t, [20.0], 0.3)
    _, big = expand_serve_labels(t, [20.0], 1.0)
    assert (~big).sum() > (~small).sum()


def test_gate_spans_positives_are_rallies_negatives_are_trimmed_gaps():
    c = RefineConfig()
    items = to_gate_spans("vid", SPANS, DURATION, c)
    serves = [(it["start"], it["end"]) for it in items if it["label"] == "serve"]
    negs = [it for it in items if it["label"] == "non-serve"]
    assert serves == SPANS
    # every negative sits strictly in a gap, clear of each rally by the buffer
    for it in negs:
        for s, e in SPANS:
            assert it["end"] <= s - c.serve_buffer_s + 1e-9 \
                or it["start"] >= e + c.serve_buffer_s - 1e-9


def test_gate_negative_min_span_floor_drops_tiny_gaps():
    spans = [(10.0, 14.0), (14.4, 18.0)]        # gap far below the floor
    c = RefineConfig(serve_neg_min_span_s=3.0)
    negs = [it for it in to_gate_spans("vid", spans, 60.0, c)
            if it["label"] == "non-serve"]
    assert all(not (13.5 <= it["start"] <= 14.4) for it in negs)


def test_rally_file_drops_straight_into_labels_loader(tmp_path):
    p = tmp_path / "vid.rallies.json"
    p.write_text(json.dumps(rally_file_items(SPANS)))
    assert load_label_file(p) == SPANS          # no manual reshaping

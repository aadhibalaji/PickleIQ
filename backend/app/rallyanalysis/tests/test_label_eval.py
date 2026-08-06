"""label_eval.score: the rule that decides whether a detector change helped.

These tests pin the three judgment calls that make the metric meaningful
rather than merely computable:
  - an UNREVIEWED spike is not a false positive (it would otherwise punish
    the detector for how much labeling the expert had time to do);
  - RECALL comes only from freeform spans (spike-sourced marks exist only
    where a spike already fired, so counting them makes recall trivially 1);
  - matching is by overlap, not id, so swept/hypothetical spikes score under
    the identical rule as production ones.

Pure-function tests — no DB, marks are passed in directly."""
from __future__ import annotations

import pytest

from ..label_eval import score, score_by_metric


def _spike(start, end, pid=1, metric="knee_bend_deg"):
    return {"contact_id": f"spk_{pid}_{metric}_{start}", "player_id": pid,
            "start_frame": start, "end_frame": end, "peak_frame": (start + end) // 2,
            "dominant_metric": metric, "category": "footwork"}


def _mark(start, end, verdict, pid=1, source="freeform", spike_id=None):
    return {"player_id": pid, "start_frame": start, "end_frame": end,
            "verdict": verdict, "source": source, "spike_id": spike_id,
            "metric": None, "note": None}


def test_unreviewed_spike_is_unjudged_not_a_false_positive():
    s = score([_spike(10, 20), _spike(100, 110)], [_mark(10, 20, "bad")])
    assert s["tp"] == 1
    assert s["fp"] == 0
    assert s["unjudged"] == 1
    assert s["precision"] == 1.0  # the unreviewed spike must not dilute this


def test_explicit_fine_verdict_is_a_false_positive():
    s = score([_spike(10, 20), _spike(30, 40)],
              [_mark(10, 20, "bad"), _mark(30, 40, "fine")])
    assert (s["tp"], s["fp"], s["unjudged"]) == (1, 1, 0)
    assert s["precision"] == 0.5


def test_recall_counts_only_freeform_bad_spans():
    # Two bad spans exist; the detector found one. Recall must be 0.5 even
    # though a spike-sourced mark also says 'bad' on the one it did find.
    spikes = [_spike(10, 20)]
    marks = [_mark(10, 20, "bad", source="spike", spike_id="spk_x"),
             _mark(10, 20, "bad"), _mark(200, 220, "bad")]
    s = score(spikes, marks)
    assert s["n_bad_spans"] == 2
    assert s["fn"] == 1
    assert s["recall"] == 0.5
    assert [m["start_frame"] for m in s["missed"]] == [200]


def test_spike_sourced_marks_alone_give_no_recall():
    # Every mark sits on a spike, so nothing can be a miss — recall is
    # undefined rather than a falsely perfect 1.0.
    s = score([_spike(10, 20)], [_mark(10, 20, "bad", source="spike", spike_id="spk_x")])
    assert s["n_bad_spans"] == 0
    assert s["recall"] is None
    assert s["precision"] == 1.0  # precision still works from spike labels


def test_marks_for_a_different_player_do_not_match():
    s = score([_spike(10, 20, pid=1)], [_mark(10, 20, "bad", pid=2)])
    assert s["unjudged"] == 1
    assert s["fn"] == 1  # player 2's bad span was genuinely missed


def test_partial_overlap_counts_as_the_same_moment():
    s = score([_spike(10, 20)], [_mark(18, 40, "bad")])
    assert s["tp"] == 1
    assert s["fn"] == 0


def test_adjacent_but_disjoint_spans_do_not_match():
    s = score([_spike(10, 20)], [_mark(21, 30, "bad")])
    assert s["tp"] == 0
    assert s["fn"] == 1


def test_bad_wins_when_several_marks_overlap_one_spike():
    s = score([_spike(10, 40)], [_mark(10, 20, "fine"), _mark(30, 40, "bad")])
    assert s["tp"] == 1
    assert s["fp"] == 0


def test_zero_precision_still_yields_a_number_not_none():
    s = score([_spike(10, 20)], [_mark(10, 20, "fine"), _mark(200, 210, "bad")])
    assert s["precision"] == 0.0
    assert s["recall"] == 0.0
    assert s["f1"] is None  # both zero -> undefined, must not divide by zero


def test_no_marks_leaves_every_rate_undefined():
    s = score([_spike(10, 20)], [])
    assert s["precision"] is None and s["recall"] is None and s["f1"] is None
    assert s["unjudged"] == 1


def test_f1_combines_both_rates():
    # 2 spikes, 1 correct 1 wrong -> precision .5; 2 bad spans, 1 found -> recall .5
    spikes = [_spike(10, 20), _spike(30, 40)]
    marks = [_mark(10, 20, "bad"), _mark(30, 40, "fine"), _mark(200, 210, "bad")]
    s = score(spikes, marks)
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    assert s["f1"] == pytest.approx(0.5)


def test_score_by_metric_attributes_false_positives():
    spikes = [_spike(10, 20, metric="knee_bend_deg"),
              _spike(30, 40, metric="elbow_extension_deg")]
    marks = [_mark(10, 20, "bad"), _mark(30, 40, "fine")]
    by_metric = score_by_metric(spikes, marks)
    assert by_metric["knee_bend_deg"]["precision"] == 1.0
    assert by_metric["elbow_extension_deg"]["precision"] == 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

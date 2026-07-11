"""Synthetic feature vectors -> ratio math and adaptive-threshold selection.
No DB, no anthropic import needed: exercises the pure functions directly
(compute_category_ratios, select_worst_best, build_report with an injected
fake llm_fn), same pattern as position_metrics/contact_detect tests."""
from __future__ import annotations

from ..coaching import (CATEGORIES, _classify, _neutral, build_report,
                         compute_category_ratios, select_worst_best,
                         ELBOW_EXTENSION_GOOD_RANGE_DEG, FOLLOW_THROUGH_GOOD_RANGE,
                         KNEE_BEND_GOOD_RANGE_DEG, SHOULDER_ROTATION_GOOD_RANGE_DEG,
                         SPLIT_STEP_GOOD_RANGE_S, WEIGHT_TRANSFER_GOOD_RANGE)
from ..contracts import ShotFeatureVector


def _vec(contact_id, category, deviation_score):
    """Build a ShotFeatureVector whose fields don't matter for the tests
    below — only category/deviation_score are asserted on — so every other
    field is a neutral mid-range placeholder."""
    return ShotFeatureVector(
        contact_id=contact_id, category=category,
        knee_bend_deg=_neutral(KNEE_BEND_GOOD_RANGE_DEG),
        shoulder_rotation_deg=_neutral(SHOULDER_ROTATION_GOOD_RANGE_DEG),
        elbow_extension_deg=_neutral(ELBOW_EXTENSION_GOOD_RANGE_DEG),
        weight_transfer_score=_neutral(WEIGHT_TRANSFER_GOOD_RANGE),
        contact_height_m=0.0,
        follow_through_completion_score=_neutral(FOLLOW_THROUGH_GOOD_RANGE),
        split_step_delay_s=_neutral(SPLIT_STEP_GOOD_RANGE_S),
        deviation_score=deviation_score)


def _contact(contact_id, confidence=0.8, timestamp_s=1.0, rally_id="r1"):
    return {"contact_id": contact_id, "timestamp_s": timestamp_s, "rally_id": rally_id,
            "player_id": 1, "team": "near", "shot_side": "left", "is_serve": False,
            "confidence": confidence}


# --------------------------------------------------------------------- ratio

def test_category_ratios_divide_by_active_rally_seconds_not_shot_count():
    vectors = [_vec("c1", "footwork", 0.5), _vec("c2", "footwork", -0.3),
               _vec("c3", "footwork", 0.0), _vec("c4", "contact_point", 0.0)]
    ratios = compute_category_ratios(vectors, active_rally_seconds=20.0)
    assert ratios["footwork"]["flagged_count"] == 2  # only the two nonzero deviations
    assert ratios["footwork"]["active_rally_seconds"] == 20.0
    assert ratios["footwork"]["ratio"] == 2 / 20.0
    assert ratios["contact_point"]["flagged_count"] == 0
    assert set(ratios) == set(CATEGORIES)


def test_category_ratios_zero_active_seconds_does_not_divide_by_zero():
    vectors = [_vec("c1", "paddle_prep", 1.0)]
    ratios = compute_category_ratios(vectors, active_rally_seconds=0.0)
    assert ratios["paddle_prep"]["ratio"] == 0.0


# --------------------------------------------------------------- classify()

def test_classify_all_in_range_scores_zero():
    cat, dev = _classify(_neutral(KNEE_BEND_GOOD_RANGE_DEG), _neutral(SHOULDER_ROTATION_GOOD_RANGE_DEG),
                          _neutral(ELBOW_EXTENSION_GOOD_RANGE_DEG), _neutral(WEIGHT_TRANSFER_GOOD_RANGE),
                          0.0, _neutral(FOLLOW_THROUGH_GOOD_RANGE), _neutral(SPLIT_STEP_GOOD_RANGE_S))
    assert dev == 0.0
    assert cat in CATEGORIES


def test_classify_flags_the_worst_offending_category():
    # knee way outside range (footwork), everything else dead-center -> footwork wins.
    cat, dev = _classify(300.0, _neutral(SHOULDER_ROTATION_GOOD_RANGE_DEG),
                          _neutral(ELBOW_EXTENSION_GOOD_RANGE_DEG), _neutral(WEIGHT_TRANSFER_GOOD_RANGE),
                          0.0, _neutral(FOLLOW_THROUGH_GOOD_RANGE), _neutral(SPLIT_STEP_GOOD_RANGE_S))
    assert cat == "footwork"
    assert dev > 0


# --------------------------------------------------------- adaptive selection

def test_worst_shots_adaptive_not_fixed_topn_clean_session():
    # One mild outlier among many perfect shots — adaptive threshold should
    # surface it without us hardcoding "top 3" or similar.
    vectors_meta = [(_vec(f"c{i}", "footwork", 0.0), _contact(f"c{i}")) for i in range(8)]
    vectors_meta.append((_vec("bad", "footwork", 5.0), _contact("bad")))
    worst, _best = select_worst_best(vectors_meta)
    assert any(w["contact_id"] == "bad" for w in worst)
    assert len(worst) < len(vectors_meta)  # didn't just flag everything


def test_worst_shots_empty_when_every_shot_in_range():
    vectors_meta = [(_vec(f"c{i}", "footwork", 0.0), _contact(f"c{i}")) for i in range(5)]
    worst, _best = select_worst_best(vectors_meta)
    assert worst == []


def test_worst_shots_scales_up_for_a_rough_session():
    # Many comparably-bad shots -> adaptive threshold should let most of them
    # through (a rough session surfaces many, not a fixed count).
    vectors_meta = [(_vec(f"c{i}", "contact_point", 2.0 + 0.01 * i), _contact(f"c{i}"))
                     for i in range(10)]
    worst, _best = select_worst_best(vectors_meta)
    assert len(worst) >= 4


def test_best_shots_requires_in_range_and_high_confidence():
    vectors_meta = [
        (_vec("low_conf", "footwork", 0.0), _contact("low_conf", confidence=0.1)),
        (_vec("high_conf", "footwork", 0.0), _contact("high_conf", confidence=0.95)),
        (_vec("flagged", "footwork", 1.0), _contact("flagged", confidence=0.95)),
    ]
    _worst, best = select_worst_best(vectors_meta)
    best_ids = {b["contact_id"] for b in best}
    assert "flagged" not in best_ids          # never picks an out-of-range shot as "best"
    assert "low_conf" not in best_ids          # confidence floor excludes it


# ----------------------------------------------------------- LLM payload wiring

def test_build_report_passes_ratios_and_shots_to_llm_fn_and_calls_it_once():
    calls = []

    def fake_llm(payload):
        calls.append(payload)
        return "mock narrative"

    vectors_meta = [(_vec("good", "footwork", 0.0), _contact("good", confidence=0.9)),
                     (_vec("bad", "contact_point", -3.0), _contact("bad", confidence=0.9))]
    report = build_report("vid1", vectors_meta, active_rally_seconds=15.0, llm_fn=fake_llm)

    assert len(calls) == 1  # exactly one Anthropic call per video
    payload = calls[0]
    assert set(payload) == {"category_ratios", "worst_shots", "best_shots"}
    assert payload["category_ratios"]["contact_point"]["flagged_count"] == 1
    assert payload["category_ratios"]["contact_point"]["active_rally_seconds"] == 15.0
    assert any(s["contact_id"] == "bad" for s in payload["worst_shots"])
    assert any(s["contact_id"] == "good" for s in payload["best_shots"])
    assert report.video_id == "vid1"
    assert report.narrative == "mock narrative"


def test_build_report_degrades_gracefully_when_llm_fn_returns_none():
    vectors_meta = [(_vec("c1", "footwork", 0.0), _contact("c1"))]
    report = build_report("vid1", vectors_meta, active_rally_seconds=5.0, llm_fn=lambda payload: None)
    assert report.narrative == ""  # never crashes, never fabricates a narrative


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    failures = 0
    for name in dir(mod):
        if name.startswith("test_"):
            try:
                getattr(mod, name)()
                print(f"ok  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)

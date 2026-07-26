"""Synthetic feature vectors -> classify()/adaptive-threshold selection and
tip-generation wiring. No DB, no anthropic import needed: exercises the pure
functions directly (_classify, select_worst_best, select_tip_candidates,
build_report with an injected fake llm_fn), same pattern as contact_detect
tests."""
from __future__ import annotations

from ..coaching import (MAX_PRAISE_TIPS, MAX_TIPS, SYSTEM, _classify, _neutral, _norm_dev,
                         _summarize_stats, build_report, compute_feature_vector,
                         exclude_serves, generate_llm_tips, select_tip_candidates,
                         select_worst_best,
                         ELBOW_EXTENSION_GOOD_RANGE_DEG, FOLLOW_THROUGH_GOOD_RANGE,
                         KNEE_BEND_GOOD_RANGE_DEG, SHOULDER_ROTATION_GOOD_RANGE_DEG,
                         SPLIT_STEP_GOOD_RANGE_S, WEIGHT_TRANSFER_GOOD_RANGE)
from ..contracts import ShotFeatureVector

CATEGORIES = ["footwork", "paddle_prep", "contact_point", "follow_through"]


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


def _contact(contact_id, confidence=0.8, timestamp_s=1.0, frame_idx=0, rally_id="r1", is_serve=False):
    return {"contact_id": contact_id, "frame_idx": frame_idx, "timestamp_s": timestamp_s,
            "rally_id": rally_id, "player_id": 1, "team": "near", "shot_side": "left",
            "is_serve": is_serve, "confidence": confidence}


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
    # Timestamps spread well beyond MIN_TIP_GAP_S apart -- this test is about
    # the adaptive severity threshold, not the time-gap filter.
    vectors_meta = [(_vec(f"c{i}", "footwork", 0.0), _contact(f"c{i}", timestamp_s=i * 10.0))
                     for i in range(8)]
    vectors_meta.append((_vec("bad", "footwork", 5.0), _contact("bad", timestamp_s=999.0)))
    worst, _best = select_worst_best(vectors_meta)
    assert any(w["contact_id"] == "bad" for w in worst)
    assert len(worst) < len(vectors_meta)  # didn't just flag everything


def test_worst_shots_empty_when_every_shot_in_range():
    vectors_meta = [(_vec(f"c{i}", "footwork", 0.0), _contact(f"c{i}", timestamp_s=i * 10.0))
                     for i in range(5)]
    worst, _best = select_worst_best(vectors_meta)
    assert worst == []


def test_worst_shots_scales_up_for_a_rough_session():
    # Many comparably-bad shots -> adaptive threshold should let most of them
    # through (a rough session surfaces many, not a fixed count). Timestamps
    # spread well beyond MIN_TIP_GAP_S apart so the time-gap filter (tested
    # separately below) doesn't collapse them.
    vectors_meta = [(_vec(f"c{i}", "contact_point", 2.0 + 0.01 * i), _contact(f"c{i}", timestamp_s=i * 10.0))
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


# --------------------------------------------------- min time-gap suppression

def test_same_player_close_worst_tips_keep_only_more_severe_one():
    # Two worst candidates for the same player 1s apart (well under
    # MIN_TIP_GAP_S) -- only the larger |deviation_score| one should survive.
    vectors_meta = [
        (_vec("mild", "footwork", -1.5), _contact("mild", timestamp_s=10.0)),
        (_vec("severe", "footwork", -4.0), _contact("severe", timestamp_s=11.0)),
    ]
    worst, _best = select_worst_best(vectors_meta)
    assert {w["contact_id"] for w in worst} == {"severe"}


def test_different_players_close_in_time_both_survive():
    # Same timestamps, but different players -- the gap filter must never
    # suppress across players.
    vectors_meta = [
        (_vec("p1", "footwork", -3.0), _contact("p1", timestamp_s=10.0)),
        (_vec("p2", "footwork", -3.0),
         {**_contact("p2", timestamp_s=10.4), "player_id": 2}),
    ]
    worst, _best = select_worst_best(vectors_meta)
    assert {w["contact_id"] for w in worst} == {"p1", "p2"}


def test_same_player_close_tips_suppressed_across_categories_worst_beats_best():
    # A worst tip (footwork) and a best tip (contact_point), same player,
    # 1s apart -- cross-category suppression still applies, and the worst
    # (actionable) tip always outranks the best (praise) tip for the same
    # moment.
    vectors_meta = [
        (_vec("worst_one", "footwork", -2.0), _contact("worst_one", timestamp_s=5.0, confidence=0.9)),
        (_vec("best_one", "contact_point", 0.0), _contact("best_one", timestamp_s=5.5, confidence=0.99)),
    ]
    worst, best = select_worst_best(vectors_meta)
    assert {w["contact_id"] for w in worst} == {"worst_one"}
    assert best == []


# ------------------------------------------------------- tip-candidate cap

def test_select_tip_candidates_caps_total_and_reserves_praise_slots():
    # Many worse-than-threshold shots plus many clean high-confidence
    # shots -- total surfaced must never exceed MAX_TIPS, and praise never
    # crowds out more than MAX_PRAISE_TIPS worth of improve tips.
    # Timestamps spread well beyond MIN_TIP_GAP_S apart -- this test is about
    # the MAX_TIPS/MAX_PRAISE_TIPS cap, not the time-gap filter.
    worst_meta = [(_vec(f"bad{i}", "footwork", -5.0), _contact(f"bad{i}", confidence=0.9, timestamp_s=i * 10.0))
                  for i in range(20)]
    best_meta = [(_vec(f"good{i}", "footwork", 0.0), _contact(f"good{i}", confidence=0.95, timestamp_s=1000.0 + i * 10.0))
                 for i in range(20)]
    candidates = select_tip_candidates(worst_meta + best_meta)
    assert len(candidates) == MAX_TIPS
    n_best = sum(1 for c in candidates if c["deviation_score"] == 0.0)
    assert n_best <= MAX_PRAISE_TIPS


# ----------------------------------------------------------- LLM payload wiring

def test_build_report_passes_candidates_to_llm_fn_and_calls_it_once():
    calls = []

    def fake_llm(shots, stats=None):
        calls.append(shots)
        return {s["contact_id"]: {"tip": "mock tip", "tone": "improve" if s["deviation_score"] else "praise"}
                for s in shots}

    # timestamps well beyond MIN_TIP_GAP_S apart -- this test is about the
    # llm_fn wiring/call-count, not the time-gap filter.
    vectors_meta = [(_vec("good", "footwork", 0.0), _contact("good", confidence=0.9, frame_idx=10, timestamp_s=1.5)),
                     (_vec("bad", "contact_point", -3.0), _contact("bad", confidence=0.9, frame_idx=20, timestamp_s=30.0))]
    report = build_report("vid1", vectors_meta, llm_fn=fake_llm)

    assert len(calls) == 1  # exactly one Anthropic call per video
    contact_ids = {s["contact_id"] for s in calls[0]}
    assert contact_ids == {"good", "bad"}
    assert report.video_id == "vid1"
    tips_by_id = {t.contact_id: t for t in report.tips}
    assert tips_by_id["bad"].tone == "improve"
    assert tips_by_id["bad"].frame == 20
    assert tips_by_id["bad"].timestamp_s == 30.0
    assert tips_by_id["good"].tone == "praise"


def test_build_report_degrades_gracefully_when_llm_fn_returns_empty():
    vectors_meta = [(_vec("c1", "footwork", 0.0), _contact("c1"))]
    report = build_report("vid1", vectors_meta, llm_fn=lambda shots: {})
    assert report.tips == []  # never crashes, never fabricates a tip


# --------------------------------------------------------- serve exclusion

def test_exclude_serves_drops_only_serve_contacts():
    contacts = [{"contact_id": "serve1", "is_serve": True},
                {"contact_id": "rally1", "is_serve": False},
                {"contact_id": "rally2", "is_serve": 0},
                {"contact_id": "serve2", "is_serve": 1}]
    kept = exclude_serves(contacts)
    assert {c["contact_id"] for c in kept} == {"rally1", "rally2"}


def test_exclude_serves_all_serves_yields_empty_list():
    contacts = [{"contact_id": "serve1", "is_serve": True}, {"contact_id": "serve2", "is_serve": 1}]
    assert exclude_serves(contacts) == []


def test_select_tip_candidates_never_surfaces_a_serve_contact():
    # A serve contact that reaches select_tip_candidates (e.g. if the upstream
    # exclude_serves filter were ever skipped) must still never win a worst/
    # best slot -- but the real guarantee is exclude_serves keeping it out of
    # vectors_meta in the first place, which generate_coaching_report relies
    # on. Here we simulate "exclude_serves ran" by building vectors_meta only
    # from non-serve contacts, then assert no is_serve contact sneaks in.
    vectors_meta = [(_vec("serve", "footwork", -5.0), _contact("serve", is_serve=True)),
                    (_vec("rally", "footwork", -5.0), _contact("rally", is_serve=False))]
    non_serve_meta = [(vec, c) for vec, c in vectors_meta if not c["is_serve"]]
    candidates = select_tip_candidates(non_serve_meta)
    assert all(not c["is_serve"] for c in candidates)
    assert {c["contact_id"] for c in candidates} == {"rally"}


def test_build_report_all_serve_input_degrades_to_empty_report_no_error():
    # exclude_serves() upstream would leave vectors_meta empty for an
    # all-serve rally/video -- build_report must produce an empty-but-valid
    # report, not crash or fabricate tips from serve data.
    report = build_report("vid1", [], llm_fn=lambda shots: {})
    assert report.tips == []


# -------------------------------------------------- report_stats summarizing

def test_summarize_stats_none_or_empty_returns_none():
    assert _summarize_stats(None) is None
    assert _summarize_stats({}) is None
    assert _summarize_stats({"rallies": {}, "shot_counts": {}}) is None


def test_summarize_stats_rounds_kitchen_arrival_and_omits_nulls():
    # Player "2" never reaches the kitchen in either rally (None every time)
    # -- must be entirely absent from the summary, never averaged in as 0.
    stats = {
        "rallies": {
            "r1": {"kitchen_arrival_s": {"1": 1.8, "2": None}},
            "r2": {"kitchen_arrival_s": {"1": 2.6, "2": None}},
        },
        "shot_counts": {},
    }
    summary = _summarize_stats(stats)
    tendencies = summary["kitchen_arrival_tendencies"]
    assert "Player 2" not in tendencies
    entry = tendencies["Player 1"]
    assert entry["avg_kitchen_arrival_s_rounded"] == round((1.8 + 2.6) / 2)
    assert isinstance(entry["avg_kitchen_arrival_s_rounded"], int)  # never sub-second
    assert entry["rallies_with_data"] == 2


def test_summarize_stats_passes_through_small_shot_counts():
    stats = {"rallies": {}, "shot_counts": {"1": {"left": {"kitchen": {"serve": 0, "non_serve": 3}}}}}
    summary = _summarize_stats(stats)
    assert summary["shot_distribution_by_zone"]["Player 1"] == {"left": {"kitchen": {"serve": 0, "non_serve": 3}}}
    assert "kitchen_arrival_tendencies" not in summary


# ------------------------------------------------ stats context in the prompt

def test_generate_llm_tips_includes_stats_in_prompt_when_present(monkeypatch):
    captured = {}

    class FakeTextBlock:
        type = "text"
        text = '{"tips": []}'

    class FakeResponse:
        content = [FakeTextBlock()]

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key=None):
            self.messages = FakeMessages()

    import sys
    import types

    fake_anthropic = types.SimpleNamespace(
        Anthropic=FakeClient, RateLimitError=Exception, AuthenticationError=Exception,
        APIStatusError=Exception, APIConnectionError=Exception)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setattr("app.rallyanalysis.coaching.ANTHROPIC_API_KEY", "fake-key")

    stats = {"kitchen_arrival_tendencies": {"Player 1": {"avg_kitchen_arrival_s_rounded": 2, "rallies_with_data": 3}}}
    generate_llm_tips([{"contact_id": "c1"}], stats_summary=stats)
    assert "player_tendencies" in captured["messages"][0]["content"]

    captured.clear()
    generate_llm_tips([{"contact_id": "c1"}], stats_summary=None)
    assert "player_tendencies" not in captured["messages"][0]["content"]


# ------------------------------------------------ silent-neutral fallback fix

def test_norm_dev_none_passes_through_instead_of_zero():
    assert _norm_dev(None, (0.0, 1.0)) is None
    assert _norm_dev(0.5, (0.0, 1.0)) == 0.0


def test_classify_never_picks_a_category_whose_metrics_are_all_excluded():
    # footwork's only two metrics (split_step, knee_bend) are both None (no
    # confident pose support). The old fallback would have defaulted both to
    # their neutral midpoint -> deviation_score 0.0, a fabricated "footwork is
    # fine" reading. footwork must not even be an eligible category here.
    cat, _dev = _classify(None, _neutral(SHOULDER_ROTATION_GOOD_RANGE_DEG),
                           _neutral(ELBOW_EXTENSION_GOOD_RANGE_DEG), _neutral(WEIGHT_TRANSFER_GOOD_RANGE),
                           0.0, _neutral(FOLLOW_THROUGH_GOOD_RANGE), None)
    assert cat != "footwork"


def test_classify_returns_none_when_nothing_real_anywhere():
    cat, dev = _classify(None, None, None, None, None, None, None)
    assert cat is None
    assert dev is None


def _kp17():
    """17 COCO keypoints with hip/knee/ankle/shoulder/elbow/wrist placed to
    give _angle_deg a well-defined (non-degenerate) angle; exact values don't
    matter for these tests, only that a real number comes out."""
    pts = [[100.0 + i, 200.0 + i] for i in range(17)]
    pts[5], pts[6] = [100.0, 50.0], [150.0, 50.0]     # shoulders
    pts[7], pts[8] = [100.0, 100.0], [150.0, 100.0]   # elbows
    pts[9], pts[10] = [150.0, 100.0], [200.0, 100.0]  # wrists
    pts[11], pts[12] = [100.0, 100.0], [150.0, 100.0]  # hips
    pts[13], pts[14] = [100.0, 150.0], [150.0, 150.0]  # knees
    pts[15], pts[16] = [150.0, 120.0], [200.0, 120.0]  # ankles
    return pts


def _pose_frame(f, kp, kp_conf, cxy=(1.0, 5.0)):
    return {"f": f, "players": [{"id": 1, "kp": kp, "kp_conf": kp_conf,
                                  "cxy": list(cxy), "box": [0.0, 0.0, 50.0, 100.0], "team": "near"}]}


def _pose_contact(frame_idx=100, court_x=1.0, court_y=5.0):
    return {"contact_id": "c1", "frame_idx": frame_idx, "player_id": 1,
            "court_x": court_x, "court_y": court_y, "team": "near"}


def test_compute_feature_vector_high_confidence_excludes_nothing():
    # Two frames so split_step_delay_s (needs an ankle-speed diff between
    # consecutive pre-contact frames) has something real to compute too.
    entries = [_pose_frame(90, _kp17(), [0.95] * 17),
               _pose_frame(100, _kp17(), [0.95] * 17)]
    vec = compute_feature_vector(_pose_contact(), entries, fps=30.0)
    assert vec is not None
    assert vec.low_confidence_metrics == []
    assert vec.knee_bend_deg is not None
    assert vec.shoulder_rotation_deg is not None
    assert vec.elbow_extension_deg is not None
    assert vec.split_step_delay_s is not None


def test_compute_feature_vector_low_confidence_excludes_not_neutral():
    conf = [0.95] * 17
    for idx in (11, 12, 13, 14, 15, 16):  # hips/knees/ankles
        conf[idx] = 0.1
    entries = [_pose_frame(100, _kp17(), conf)]
    vec = compute_feature_vector(_pose_contact(), entries, fps=30.0)
    assert vec is not None
    assert vec.knee_bend_deg is None
    assert "knee_bend_deg" in vec.low_confidence_metrics
    assert vec.split_step_delay_s is None
    assert "split_step_delay_s" in vec.low_confidence_metrics
    # the category actually chosen must never be one whose metrics were all excluded
    if vec.category == "footwork":
        assert False, "footwork has no real support here and must not be picked"


def test_compute_feature_vector_missing_kp_conf_key_is_presence_only_no_regression():
    # Overlay rows written before Step 1's kp_conf capture have no "kp_conf"
    # key at all -- must behave exactly as before (presence-only), not be
    # treated as automatically failing a threshold they never recorded.
    entries = [_pose_frame(90, _kp17(), kp_conf=None),
               _pose_frame(100, _kp17(), kp_conf=None)]
    vec = compute_feature_vector(_pose_contact(), entries, fps=30.0)
    assert vec is not None
    assert vec.low_confidence_metrics == []
    assert vec.knee_bend_deg is not None


def test_system_prompt_keeps_null_safety_and_no_raw_numbers_rules():
    # Rework of the tone framing (see coaching.py's TIPS_SCHEMA/SYSTEM) must
    # never loosen these safety rules — only the delivery style around them
    # changes.
    assert "NEVER mention degrees, angles, confidence/certainty scores" in SYSTEM
    assert "Never state a raw seconds value, a velocity, or any" in SYSTEM
    assert "must NEVER state or\nimply a specific claim about a joint/measurement" in SYSTEM
    assert "no known reading for it — never fabricate a claim about it" in SYSTEM


def test_system_prompt_encodes_three_beat_supportive_framing():
    # The reworked tone must require: acknowledge -> undiluted critique -> encouraging close.
    assert "three-beat" in SYSTEM
    assert "Do NOT soften, hedge, vague-up, or water down the" in SYSTEM
    assert "NOT a compliment sandwich that pads or dilutes" in SYSTEM


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

"""Synthetic feature vectors -> classify()/adaptive-threshold selection and
tip-generation wiring. No DB, no anthropic import needed: exercises the pure
functions directly (_classify, select_worst_best, select_tip_candidates,
build_report with an injected fake llm_fn), same pattern as contact_detect
tests."""
from __future__ import annotations

from .. import coaching
from ..coaching import (MAX_PRAISE_TIPS, MAX_TIPS, MIN_SAMPLES_FOR_DISTRIBUTION, SYSTEM,
                         _classify, _extract_raw_metrics, _finalize_vector, _llm_shot_payload,
                         _player_fences_by_metric, _rel_dev, _robust_fence, _session_context,
                         _summarize_stats, build_report, compute_feature_vector, exclude_serves,
                         generate_llm_tips, select_tip_candidates, select_worst_best)
from ..contracts import ShotFeatureVector

CATEGORIES = ["footwork", "paddle_prep", "contact_point", "follow_through"]


def _vec(contact_id, category, deviation_score):
    """Build a ShotFeatureVector whose fields don't matter for the tests
    below — only category/deviation_score are asserted on — so every other
    field is an arbitrary placeholder."""
    return ShotFeatureVector(
        contact_id=contact_id, category=category,
        knee_bend_deg=130.0, shoulder_rotation_deg=40.0, elbow_extension_deg=140.0,
        weight_transfer_score=0.6, contact_height_m=0.0,
        follow_through_completion_score=0.7, split_step_delay_s=0.2,
        deviation_score=deviation_score)


def _contact(contact_id, confidence=0.8, timestamp_s=1.0, frame_idx=0, rally_id="r1", is_serve=False):
    return {"contact_id": contact_id, "frame_idx": frame_idx, "timestamp_s": timestamp_s,
            "rally_id": rally_id, "player_id": 1, "team": "near", "shot_side": "left",
            "is_serve": is_serve, "confidence": confidence}


# ------------------------------------------------------ relative scoring

METRIC_FIELDS = ("knee_bend_deg", "shoulder_rotation_deg", "elbow_extension_deg",
                  "weight_transfer_score", "contact_height_m",
                  "follow_through_completion_score", "split_step_delay_s")


def _all_zero_devs():
    return {f: 0.0 for f in METRIC_FIELDS}


def test_robust_fence_needs_min_samples():
    assert _robust_fence([1.0, 2.0]) is None  # below MIN_SAMPLES_FOR_DISTRIBUTION
    assert _robust_fence([1.0] * MIN_SAMPLES_FOR_DISTRIBUTION) is not None


def test_rel_dev_none_when_value_or_fence_missing():
    fence = _robust_fence([100.0, 120.0, 140.0])
    assert _rel_dev(None, fence) is None   # no confident reading on this shot
    assert _rel_dev(120.0, None) is None   # not enough of the player's own shots to compare


def test_rel_dev_zero_within_players_own_fence_nonzero_outside():
    fence = _robust_fence([120.0, 125.0, 130.0, 135.0, 140.0])
    assert _rel_dev(130.0, fence) == 0.0        # at this player's own median
    dev = _rel_dev(500.0, fence)
    assert dev is not None and dev > 0          # far outside this player's own spread


def test_classify_all_typical_scores_zero():
    cat, dev = _classify(_all_zero_devs())
    assert dev == 0.0
    assert cat in CATEGORIES


def test_classify_flags_the_worst_offending_category():
    # knee way outside this player's own fence (footwork), everything else typical -> footwork wins.
    devs = _all_zero_devs()
    devs["knee_bend_deg"] = 6.0
    cat, dev = _classify(devs)
    assert cat == "footwork"
    assert dev > 0


def test_classify_never_picks_a_category_whose_metrics_are_all_none():
    # footwork's only two metrics (split_step, knee_bend) are both None (no
    # confident pose support, or not enough of this player's own shots to
    # compare against) -- footwork must not even be an eligible category.
    devs = _all_zero_devs()
    devs["split_step_delay_s"] = None
    devs["knee_bend_deg"] = None
    cat, _dev = _classify(devs)
    assert cat != "footwork"


def test_classify_returns_none_when_nothing_real_anywhere():
    cat, dev = _classify({f: None for f in METRIC_FIELDS})
    assert cat is None
    assert dev is None


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
    assert report.player_themes == []  # dict-only (old-style) double -> no themes, no error


def test_build_report_attaches_validated_player_themes():
    # The model's echoed player_label is only a join key: unknown labels are
    # dropped (hallucinated player), duplicates collapse to the first, and
    # player_id always comes from our own contact data.
    def fake_llm(shots, stats=None, ctx=None):
        return ({s["contact_id"]: {"tip": "t", "tone": "improve"} for s in shots},
                [{"player_label": "Player 1", "theme": "keep working the feet"},
                 {"player_label": "Player 99", "theme": "hallucinated player"},
                 {"player_label": "Player 1", "theme": "duplicate, dropped"}])

    vectors_meta = [(_vec("bad", "footwork", -3.0), _contact("bad"))]
    report = build_report("vid1", vectors_meta, llm_fn=fake_llm)
    assert len(report.player_themes) == 1
    theme = report.player_themes[0]
    assert theme.player_label == "Player 1"
    assert theme.player_id == 1
    assert theme.theme == "keep working the feet"


def test_build_report_passes_whole_session_context_to_llm_fn():
    # session_context must cover EVERY scored shot (here: 12), not just the
    # <=MAX_TIPS candidates that get individual tips.
    captured = {}

    def fake_llm(shots, stats=None, ctx=None):
        captured["shots"] = shots
        captured["ctx"] = ctx
        return {}

    vectors_meta = [(_vec(f"c{i}", "footwork", -3.0), _contact(f"c{i}", timestamp_s=i * 10.0))
                     for i in range(12)]
    build_report("vid1", vectors_meta, llm_fn=fake_llm)
    assert captured["ctx"]["Player 1"]["shots_analyzed"] == 12
    assert len(captured["shots"]) <= MAX_TIPS


# ------------------------------------------------- session context aggregates

def test_session_context_counts_whole_session_per_player():
    vectors_meta = [
        (_vec("a", "footwork", -2.0), _contact("a")),
        (_vec("b", "footwork", 0.0), _contact("b")),
        (_vec("c", "contact_point", 1.5), {**_contact("c"), "player_id": 2}),
    ]
    ctx = _session_context(vectors_meta)
    assert ctx["Player 1"]["shots_analyzed"] == 2
    assert ctx["Player 1"]["typical_shots"] == 1
    assert ctx["Player 1"]["flagged_by_category"] == {"footwork": 1}
    assert ctx["Player 2"]["flagged_by_category"] == {"contact_point": 1}
    assert _session_context([]) is None


# --------------------------------------------- LLM-boundary metric renaming

def test_llm_shot_payload_renames_misleading_metric_names_everywhere():
    shot = {
        "contact_id": "c1",
        "low_confidence_metrics": ["knee_bend_deg"],
        "no_comparison_metrics": ["contact_height_m"],
        "feature_vector": {"contact_height_m": -1.2, "knee_bend_deg": None,
                            "low_confidence_metrics": ["knee_bend_deg"],
                            "no_comparison_metrics": ["contact_height_m"]},
    }
    out = _llm_shot_payload(shot)
    assert "contact_height_m" not in out["feature_vector"]
    assert out["feature_vector"]["distance_from_net_m"] == -1.2
    assert out["no_comparison_metrics"] == ["distance_from_net_m"]
    assert out["feature_vector"]["no_comparison_metrics"] == ["distance_from_net_m"]
    assert out["low_confidence_metrics"] == ["knee_bend_deg"]  # honest names pass through
    # LLM-boundary only: the original shot dict (what gets persisted/served)
    # must keep the internal name untouched.
    assert shot["feature_vector"]["contact_height_m"] == -1.2
    assert shot["no_comparison_metrics"] == ["contact_height_m"]


def test_llm_shot_payload_tolerates_bare_dicts():
    assert _llm_shot_payload({"contact_id": "c1"}) == {"contact_id": "c1"}


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


def test_generate_llm_tips_sends_honest_names_session_context_and_coaching_model(monkeypatch):
    captured = {}

    class FakeTextBlock:
        type = "text"
        text = '{"tips": [], "player_themes": []}'

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

    shots = [{"contact_id": "c1", "feature_vector": {"contact_height_m": 0.5}}]
    ctx = {"Player 1": {"shots_analyzed": 10, "typical_shots": 7,
                        "flagged_by_category": {"footwork": 3}}}
    tips, themes = generate_llm_tips(shots, stats_summary=None, session_context=ctx)
    assert tips == {} and themes == []
    content = captured["messages"][0]["content"]
    # The misleading internal name must never reach the model...
    assert "contact_height_m" not in content
    assert "distance_from_net_m" in content
    # ...the whole-session aggregates ride along in the SAME single call...
    assert "session_context" in content
    # ...on the dedicated coaching model, not the discrepancy-report model.
    assert captured["model"] == coaching.ANTHROPIC_COACHING_MODEL

    captured.clear()
    generate_llm_tips(shots, stats_summary=None, session_context=None)
    assert "session_context" not in captured["messages"][0]["content"]


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


def test_extract_raw_metrics_high_confidence_excludes_nothing():
    # Two frames so split_step_delay_s (needs an ankle-speed diff between
    # consecutive pre-contact frames) has something real to compute too.
    entries = [_pose_frame(90, _kp17(), [0.95] * 17),
               _pose_frame(100, _kp17(), [0.95] * 17)]
    raw = _extract_raw_metrics(_pose_contact(), entries, fps=30.0)
    assert raw is not None
    assert raw["low_confidence_metrics"] == []
    assert raw["knee_bend_deg"] is not None
    assert raw["shoulder_rotation_deg"] is not None
    assert raw["elbow_extension_deg"] is not None
    assert raw["split_step_delay_s"] is not None


def test_extract_raw_metrics_low_confidence_excludes_not_neutral():
    conf = [0.95] * 17
    for idx in (11, 12, 13, 14, 15, 16):  # hips/knees/ankles
        conf[idx] = 0.1
    entries = [_pose_frame(100, _kp17(), conf)]
    raw = _extract_raw_metrics(_pose_contact(), entries, fps=30.0)
    assert raw is not None
    assert raw["knee_bend_deg"] is None
    assert "knee_bend_deg" in raw["low_confidence_metrics"]
    assert raw["split_step_delay_s"] is None
    assert "split_step_delay_s" in raw["low_confidence_metrics"]


def test_extract_raw_metrics_missing_kp_conf_key_is_presence_only_no_regression():
    # Overlay rows written before Step 1's kp_conf capture have no "kp_conf"
    # key at all -- must behave exactly as before (presence-only), not be
    # treated as automatically failing a threshold they never recorded.
    entries = [_pose_frame(90, _kp17(), kp_conf=None),
               _pose_frame(100, _kp17(), kp_conf=None)]
    raw = _extract_raw_metrics(_pose_contact(), entries, fps=30.0)
    assert raw is not None
    assert raw["low_confidence_metrics"] == []
    assert raw["knee_bend_deg"] is not None


# ------------------------------------------- compute_feature_vector (fences)

def test_compute_feature_vector_without_fences_has_no_comparison_anywhere():
    # No player_fences supplied (e.g. this shot's player has too few of their
    # own valid shots this video) -- every metric is "no comparison
    # available", never a fabricated 0/typical reading, and with nothing
    # comparable anywhere the shot is unscoreable (matches _classify's
    # "nothing real anywhere" -> None degrade).
    entries = [_pose_frame(90, _kp17(), [0.95] * 17),
               _pose_frame(100, _kp17(), [0.95] * 17)]
    vec = compute_feature_vector(_pose_contact(), entries, fps=30.0)
    assert vec is None


def test_compute_feature_vector_with_fences_flags_no_comparison_metrics():
    # This player has enough of their own shots (5 >= MIN_SAMPLES_FOR_DISTRIBUTION)
    # for knee_bend_deg to get a real fence, but none for shoulder_rotation_deg
    # -- shoulder_rotation_deg must show up in no_comparison_metrics even
    # though its raw reading on this shot was confident.
    entries = [_pose_frame(90, _kp17(), [0.95] * 17),
               _pose_frame(100, _kp17(), [0.95] * 17)]
    raw = _extract_raw_metrics(_pose_contact(), entries, fps=30.0)
    knee_fence = _robust_fence([raw["knee_bend_deg"]] * MIN_SAMPLES_FOR_DISTRIBUTION)
    fences = {"knee_bend_deg": knee_fence}
    vec = compute_feature_vector(_pose_contact(), entries, fps=30.0, player_fences=fences)
    assert vec is not None
    assert "shoulder_rotation_deg" in vec.no_comparison_metrics
    assert "knee_bend_deg" not in vec.no_comparison_metrics
    # never fabricate a comparison for the low-confidence field itself either
    assert vec.low_confidence_metrics == []


def test_player_fences_by_metric_groups_per_player_and_respects_min_samples():
    raw_by_contact = [
        ({"contact_id": "a", "player_id": 1}, {f: 10.0 for f in METRIC_FIELDS} | {"low_confidence_metrics": []}),
        ({"contact_id": "b", "player_id": 1}, {f: 12.0 for f in METRIC_FIELDS} | {"low_confidence_metrics": []}),
        ({"contact_id": "c", "player_id": 1}, {f: 14.0 for f in METRIC_FIELDS} | {"low_confidence_metrics": []}),
        ({"contact_id": "d", "player_id": 2}, {f: 99.0 for f in METRIC_FIELDS} | {"low_confidence_metrics": []}),
    ]
    fences = _player_fences_by_metric(raw_by_contact)
    assert fences[1]["knee_bend_deg"] is not None    # player 1 has 3 shots -> real fence
    assert fences[2]["knee_bend_deg"] is None        # player 2 has only 1 shot -> no fence
    # player 1's fence is built only from player 1's own values, not player 2's
    q1, q3, _iqr = fences[1]["knee_bend_deg"]
    assert q1 >= 10.0 and q3 <= 14.0


def test_system_prompt_keeps_null_safety_and_no_raw_numbers_rules():
    # Rework of the tone framing (see coaching.py's TIPS_SCHEMA/SYSTEM) must
    # never loosen these safety rules — only the delivery style around them
    # changes.
    assert "NEVER mention degrees, angles, confidence/certainty scores" in SYSTEM
    assert "Never state a raw seconds value, a velocity, or any" in SYSTEM
    assert "must NEVER state or\nimply a specific claim about a joint/measurement" in SYSTEM
    assert "no known reading for it — never fabricate a claim about it" in SYSTEM


def test_system_prompt_frames_deviation_as_relative_to_the_player_not_universal():
    # Must never imply a fixed/universal biomechanical standard -- only
    # comparison to this player's own session.
    assert "relative to THIS PLAYER" in SYSTEM
    assert "no validated \"correct\" biomechanical range" in SYSTEM


def test_system_prompt_bans_comparison_claims_without_enough_of_players_own_shots():
    assert "no_comparison_metrics" in SYSTEM
    assert "You must NEVER claim or imply a comparison" in SYSTEM


def test_system_prompt_encodes_three_beat_supportive_framing():
    # The reworked tone must require: acknowledge -> undiluted critique -> encouraging close.
    assert "three-beat" in SYSTEM
    assert "Do NOT soften, hedge, vague-up, or water down the" in SYSTEM
    assert "NOT a compliment sandwich that pads or dilutes" in SYSTEM


def test_system_prompt_describes_distance_from_net_never_height():
    # The LLM-facing name is the honest one, and the prompt must forbid
    # reading any vertical/contact-height meaning into it (the pipeline has
    # no vertical measurement at all).
    assert "distance_from_net_m" in SYSTEM
    assert "NOTHING about how high or low" in SYSTEM
    assert "contact_height_m" not in SYSTEM


def test_system_prompt_covers_session_context_themes_and_severity_bands():
    assert "session_context" in SYSTEM
    assert "player_themes" in SYSTEM
    # Critique intensity must be calibrated to |deviation_score| bands,
    # qualitatively — never recited as a number.
    assert "Calibrate how strongly you word a critique" in SYSTEM


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

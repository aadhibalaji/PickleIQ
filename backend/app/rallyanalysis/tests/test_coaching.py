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


def test_build_report_attaches_validated_general_feedback():
    # The model's echoed player_label is only a join key: unknown labels are
    # dropped (hallucinated player), duplicates collapse to the first,
    # entries missing either piece are dropped, and player_id always comes
    # from our own contact data.
    def fake_llm(shots, stats=None, ctx=None):
        return ({s["contact_id"]: {"tip": "t", "tone": "improve"} for s in shots},
                [{"player_label": "Player 1", "praise": "solid at the line",
                  "improve": "keep working the feet"},
                 {"player_label": "Player 99", "praise": "hallucinated", "improve": "player"},
                 {"player_label": "Player 1", "praise": "duplicate", "improve": "dropped"}])

    vectors_meta = [(_vec("bad", "footwork", -3.0), _contact("bad"))]
    report = build_report("vid1", vectors_meta, llm_fn=fake_llm)
    assert len(report.general) == 1
    fb = report.general[0]
    assert fb.player_label == "Player 1"
    assert fb.player_id == 1
    assert fb.praise == "solid at the line"
    assert fb.improve == "keep working the feet"
    assert report.player_themes == []  # replaced by general_feedback on new reports


def test_build_report_drops_general_entries_missing_either_piece():
    def fake_llm(shots, stats=None, ctx=None):
        return ({}, [{"player_label": "Player 1", "praise": "only praise, no improve"}])

    report = build_report("vid1", [(_vec("bad", "footwork", -3.0), _contact("bad"))],
                           llm_fn=fake_llm)
    assert report.general == []


def test_general_context_aggregates_by_category_and_picks_deterministically():
    vectors_meta = (
        # Player 1: footwork flagged 2/3, contact_point clean 3/3.
        [(_vec(f"fw{i}", "footwork", -2.0 if i < 2 else 0.0), _contact(f"fw{i}"))
         for i in range(3)]
        + [(_vec(f"cp{i}", "contact_point", 0.0), _contact(f"cp{i}")) for i in range(3)]
    )
    ctx = coaching._general_context(vectors_meta)
    p1 = ctx["Player 1"]
    assert p1["by_category"]["footwork"] == {"n": 3, "flagged": 2}
    assert p1["by_category"]["contact_point"] == {"n": 3, "flagged": 0}
    assert p1["strongest_negative_category"] == "footwork"
    assert p1["strongest_positive_category"] == "contact_point"


def test_general_context_never_praises_the_flagged_category_or_thin_samples():
    # Only one category, and it's flagged: negative pick only, no praise pick.
    flagged_only = [(_vec(f"fw{i}", "footwork", -2.0), _contact(f"fw{i}")) for i in range(4)]
    p1 = coaching._general_context(flagged_only)["Player 1"]
    assert p1["strongest_negative_category"] == "footwork"
    assert p1["strongest_positive_category"] is None
    # Clean but under-sampled category: no praise pick either.
    thin = [(_vec("cp0", "contact_point", 0.0), _contact("cp0"))]
    p1 = coaching._general_context(thin)["Player 1"]
    assert p1["strongest_positive_category"] is None
    assert coaching._general_context([]) is None


# --------------------------------- frame-level general context (impact-based)


def _fs(frame, knee, elbow=None):
    values = {"knee_bend_deg": knee, "elbow_extension_deg": elbow}
    devs = coaching.frame_metrics.frame_deviations(values, coaching.frame_metrics.FORM_TARGETS, None)
    return {"frame": frame, "values": values, "devs": devs}


def test_frame_general_context_never_fabricates_a_negative_pick_when_genuinely_clean():
    # Every tracked metric stays inside its target band the whole session --
    # there is nothing to legitimately criticize. The old design would still
    # have picked SOME category as "strongest_negative" (max() over a dict
    # always returns something); this must instead come back None.
    clean_series = [_fs(f, 130.0) for f in range(50)]  # in-band knee every frame
    ctx = coaching._frame_general_context({1: clean_series}, focus="form", match_type=None)
    p1 = ctx["Player 1"]
    assert p1["strongest_negative_category"] is None
    assert p1["by_category"]["footwork"]["impact_score"] == 0.0
    # A genuinely clean category is still eligible to be the POSITIVE pick --
    # only the negative pick is suppressed when there's no real evidence.
    assert p1["strongest_positive_category"] == "footwork"


def test_frame_general_context_still_picks_a_real_negative_when_evidence_exists():
    series = [_fs(f, 180.0) for f in range(50)]  # every frame flags
    ctx = coaching._frame_general_context({1: series}, focus="form", match_type=None)
    p1 = ctx["Player 1"]
    assert p1["strongest_negative_category"] == "footwork"
    assert p1["by_category"]["footwork"]["impact_score"] > 0.0


def test_build_report_frames_fn_runs_on_candidates_only_and_feeds_llm_fn():
    captured = {}

    def fake_frames(candidates):
        captured["frame_candidates"] = candidates
        return {c["contact_id"]: [{"label": "contact", "jpeg_b64": "abc"}] for c in candidates}

    def fake_llm(shots, stats=None, ctx=None, frames=None):
        captured["frames"] = frames
        return {}

    vectors_meta = [(_vec(f"c{i}", "footwork", -3.0), _contact(f"c{i}", timestamp_s=i * 10.0))
                     for i in range(20)]
    build_report("vid1", vectors_meta, llm_fn=fake_llm, frames_fn=fake_frames)
    # frames extracted only for the shortlisted candidates, never all 20
    assert len(captured["frame_candidates"]) <= coaching.MAX_LLM_CANDIDATES
    assert set(captured["frames"]) == {c["contact_id"] for c in captured["frame_candidates"]}


def test_build_report_frames_fn_failure_degrades_to_frameless_call():
    captured = {}

    def fake_llm(shots, stats=None, ctx=None, frames=None):
        captured["frames"] = frames
        return {s["contact_id"]: {"tip": "t", "tone": "improve"} for s in shots}

    def broken_frames(candidates):
        raise RuntimeError("video file missing")

    vectors_meta = [(_vec("bad", "footwork", -3.0), _contact("bad"))]
    report = build_report("vid1", vectors_meta, llm_fn=fake_llm, frames_fn=broken_frames)
    assert captured["frames"] is None    # degraded, not raised
    assert len(report.tips) == 1         # report still produced


def test_build_report_passes_whole_session_context_to_llm_fn():
    # session_context must cover EVERY scored shot (here: 12), not just the
    # candidates that get individual tips.
    captured = {}

    def fake_llm(shots, stats=None, ctx=None):
        captured["shots"] = shots
        captured["ctx"] = ctx
        return {}

    vectors_meta = [(_vec(f"c{i}", "footwork", -3.0), _contact(f"c{i}", timestamp_s=i * 10.0))
                     for i in range(12)]
    build_report("vid1", vectors_meta, llm_fn=fake_llm)
    assert captured["ctx"]["Player 1"]["shots_analyzed"] == 12
    assert len(captured["shots"]) <= coaching.MAX_LLM_CANDIDATES


def test_build_report_gives_llm_a_wider_shortlist_but_caps_tips_at_max():
    # 20 flagged shots -> the LLM sees up to MAX_LLM_CANDIDATES of them (the
    # widened shortlist for its frame-informed final cut), but even a model
    # that ignores its selection budget and tips EVERYTHING gets hard-capped
    # back to MAX_TIPS, keeping the most salient improve tips.
    captured = {}

    def greedy_llm(shots, stats=None, ctx=None):
        captured["shots"] = shots
        return {s["contact_id"]: {"tip": "t", "tone": "improve"} for s in shots}

    vectors_meta = [(_vec(f"c{i}", "footwork", -2.0 - i * 0.1), _contact(f"c{i}", timestamp_s=i * 10.0))
                     for i in range(20)]
    report = build_report("vid1", vectors_meta, llm_fn=greedy_llm)
    assert len(captured["shots"]) == coaching.MAX_LLM_CANDIDATES
    assert len(report.tips) == MAX_TIPS
    # hard cap keeps the top-salience shots (largest |deviation| here)
    kept = {t.contact_id for t in report.tips}
    assert "c19" in kept and "c0" not in kept


# -------------------------------------------------------- salience ranking

def test_salience_rally_ending_outranks_equal_midrally_outlier():
    # Same deviation magnitude; the shot the rally ended on must win a slot
    # over the mid-rally one when only one slot exists.
    midrally = (_vec("midrally", "footwork", -2.0), _contact("midrally", timestamp_s=10.0))
    ending = (_vec("ending", "footwork", -2.0),
              {**_contact("ending", timestamp_s=100.0), "is_rally_ending": True})
    candidates = select_tip_candidates([midrally, ending], limit=1)
    assert [c["contact_id"] for c in candidates] == ["ending"]


def test_salience_multi_metric_breakdown_outranks_single_metric_outlier():
    # A slightly smaller deviation with THREE metrics deviating together beats
    # a slightly larger single-metric outlier (likely one noisy keypoint).
    single = _vec("single", "footwork", -2.2)
    single.deviating_metrics = ["knee_bend_deg"]
    multi = _vec("multi", "footwork", -2.0)
    multi.deviating_metrics = ["knee_bend_deg", "split_step_delay_s", "weight_transfer_score"]
    candidates = select_tip_candidates(
        [(single, _contact("single", timestamp_s=10.0)),
         (multi, _contact("multi", timestamp_s=100.0))], limit=1)
    assert [c["contact_id"] for c in candidates] == ["multi"]


def test_finalize_vector_records_all_deviating_metrics():
    entries = [_pose_frame(90, _kp17(), [0.95] * 17),
               _pose_frame(100, _kp17(), [0.95] * 17)]
    raw = _extract_raw_metrics(_pose_contact(), entries, fps=30.0)
    fences = {
        # knee way outside its fence, shoulder inside its own
        "knee_bend_deg": _robust_fence([raw["knee_bend_deg"] + 500.0] * MIN_SAMPLES_FOR_DISTRIBUTION),
        "shoulder_rotation_deg": _robust_fence([raw["shoulder_rotation_deg"]] * MIN_SAMPLES_FOR_DISTRIBUTION),
    }
    vec = _finalize_vector(_pose_contact(), raw, fences, None)
    assert "knee_bend_deg" in vec.deviating_metrics
    assert "shoulder_rotation_deg" not in vec.deviating_metrics


# --------------------------------------------------- exemplar praise picking

def test_praise_exemplar_targets_players_most_flagged_category():
    # Player 1's flags are 2x contact_point, 1x footwork -> the praise pick
    # must exemplify contact_point, with its category re-pointed and
    # exemplar_of_category set.
    vectors_meta = [
        (_vec("bad1", "contact_point", -2.0), _contact("bad1", timestamp_s=10.0)),
        (_vec("bad2", "contact_point", -2.5), _contact("bad2", timestamp_s=20.0)),
        (_vec("bad3", "footwork", -2.0), _contact("bad3", timestamp_s=30.0)),
        (_vec("clean", "footwork", 0.0), _contact("clean", confidence=0.9, timestamp_s=100.0)),
    ]
    _worst, best = select_worst_best(vectors_meta)
    assert len(best) == 1
    assert best[0]["contact_id"] == "clean"
    assert best[0]["category"] == "contact_point"
    assert best[0]["exemplar_of_category"] == "contact_point"


def test_praise_exemplar_never_picks_shot_with_no_real_reading_in_target_category():
    # The only in-range shot has ALL contact_point metrics null -> it cannot
    # exemplify contact_point; no praise is fabricated.
    blind = _vec("blind", "footwork", 0.0)
    blind.elbow_extension_deg = None
    blind.weight_transfer_score = None
    blind.contact_height_m = None
    vectors_meta = [
        (_vec("bad", "contact_point", -2.0), _contact("bad", timestamp_s=10.0)),
        (blind, _contact("blind", confidence=0.9, timestamp_s=100.0)),
    ]
    _worst, best = select_worst_best(vectors_meta)
    assert best == []


def test_praise_capped_at_max_praise_tips_prioritizing_flagged_players():
    # Three players with clean shots; only players 1 and 2 have flags.
    # MAX_PRAISE_TIPS (2) exemplars total, and the flag-free player 3 loses
    # the slot — their praise teaches nothing.
    def meta(cid, pid, dev, ts):
        return (_vec(cid, "footwork", dev), {**_contact(cid, confidence=0.9, timestamp_s=ts), "player_id": pid})

    vectors_meta = [
        meta("p1bad", 1, -2.0, 10.0), meta("p1clean", 1, 0.0, 50.0),
        meta("p2bad", 2, -2.0, 20.0), meta("p2clean", 2, 0.0, 60.0),
        meta("p3clean", 3, 0.0, 70.0),
    ]
    _worst, best = select_worst_best(vectors_meta)
    assert len(best) == MAX_PRAISE_TIPS
    assert {b["player_id"] for b in best} == {1, 2}


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


# ---------------------------------------------- shot-type stratified fences

def test_finalize_vector_prefers_same_type_fence_over_player_wide():
    # Player-wide fence says 130 is way out (all-shots values near 20 — think
    # "drives"); the same-type stratum says 130 is dead normal (their other
    # soft shots). Stratified scoring must NOT flag the shot.
    entries = [_pose_frame(90, _kp17(), [0.95] * 17),
               _pose_frame(100, _kp17(), [0.95] * 17)]
    raw = _extract_raw_metrics(_pose_contact(), entries, fps=30.0)
    knee = raw["knee_bend_deg"]
    contact = {**_pose_contact(), "shot_type": "kitchen_soft"}
    player_fence = {"knee_bend_deg": _robust_fence([knee + 500.0] * MIN_SAMPLES_FOR_DISTRIBUTION)}
    stratum_fence = {"knee_bend_deg": _robust_fence([knee] * MIN_SAMPLES_FOR_DISTRIBUTION)}

    vec_stratified = _finalize_vector(contact, raw, player_fence, stratum_fence)
    assert vec_stratified.deviation_score == 0.0                    # typical among same-type shots
    assert "knee_bend_deg" not in vec_stratified.mixed_baseline_metrics
    assert vec_stratified.shot_type == "kitchen_soft"

    vec_mixed = _finalize_vector(contact, raw, player_fence, None)  # no stratum -> falls back
    assert abs(vec_mixed.deviation_score) > 0                       # cross-type flag fires
    assert "knee_bend_deg" in vec_mixed.mixed_baseline_metrics      # ...and is labeled as cross-type


def test_finalize_vector_no_fence_anywhere_is_no_comparison_not_mixed():
    entries = [_pose_frame(90, _kp17(), [0.95] * 17),
               _pose_frame(100, _kp17(), [0.95] * 17)]
    raw = _extract_raw_metrics(_pose_contact(), entries, fps=30.0)
    knee_fence = {"knee_bend_deg": _robust_fence([raw["knee_bend_deg"]] * MIN_SAMPLES_FOR_DISTRIBUTION)}
    vec = _finalize_vector(_pose_contact(), raw, knee_fence, None)
    assert "shoulder_rotation_deg" in vec.no_comparison_metrics
    assert "shoulder_rotation_deg" not in vec.mixed_baseline_metrics  # no comparison happened at all


def test_stratum_fences_group_by_player_and_type_and_skip_untyped():
    raw_metrics = {f: 10.0 for f in METRIC_FIELDS} | {"low_confidence_metrics": []}
    raw_by_contact = (
        [({"contact_id": f"a{i}", "player_id": 1, "shot_type": "kitchen_soft"}, dict(raw_metrics))
         for i in range(MIN_SAMPLES_FOR_DISTRIBUTION)]
        + [({"contact_id": "drive", "player_id": 1, "shot_type": "backcourt_fast"}, dict(raw_metrics))]
        + [({"contact_id": "untyped", "player_id": 1, "shot_type": None}, dict(raw_metrics))]
    )
    from ..coaching import _stratum_fences_by_metric
    fences = _stratum_fences_by_metric(raw_by_contact)
    assert fences[(1, "kitchen_soft")]["knee_bend_deg"] is not None
    assert fences[(1, "backcourt_fast")]["knee_bend_deg"] is None   # 1 shot < min samples
    assert all(key[1] is not None for key in fences)                 # untyped contributes nowhere


def test_shot_summary_carries_shot_type_and_rally_context():
    vec = _vec("c1", "footwork", -2.0)
    vec.shot_type = "kitchen_soft"
    contact = {**_contact("c1"), "shot_type": "kitchen_soft", "rally_shot_index": 4,
               "rally_shot_count": 4, "is_rally_ending": True, "seconds_until_rally_end": 0.5}
    worst, _best = select_worst_best([(vec, contact)])
    s = worst[0]
    assert s["shot_type"] == "kitchen_soft"
    assert s["rally_shot_index"] == 4 and s["rally_shot_count"] == 4
    assert s["is_rally_ending"] is True
    assert s["mixed_baseline_metrics"] == []


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


def test_llm_shot_payload_renames_all_four_metric_name_lists():
    shot = {"contact_id": "c1",
            "mixed_baseline_metrics": ["contact_height_m"],
            "deviating_metrics": ["contact_height_m", "knee_bend_deg"]}
    out = _llm_shot_payload(shot)
    assert out["mixed_baseline_metrics"] == ["distance_from_net_m"]
    assert out["deviating_metrics"] == ["distance_from_net_m", "knee_bend_deg"]


def test_system_prompt_gives_llm_the_final_cut_with_a_budget():
    # The model receives a wider shortlist and must (a) know it can skip
    # tracking-noise moments, (b) never exceed the tip budget.
    assert "THE FINAL CUT IS YOURS" in SYSTEM
    assert f"MOST {MAX_TIPS} moments total" in SYSTEM
    assert "SKIP a moment entirely" in SYSTEM


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
        stop_reason = "end_turn"
        content = [FakeTextBlock()]

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return FakeResponse()

    class FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

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
        stop_reason = "end_turn"
        content = [FakeTextBlock()]

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return FakeResponse()

    class FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

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
    # ...on the dedicated coaching model, not the discrepancy-report model...
    assert captured["model"] == coaching.ANTHROPIC_COACHING_MODEL
    # ...with ADAPTIVE thinking — budget_tokens 400s on claude-sonnet-5.
    assert captured["thinking"] == {"type": "adaptive"}

    captured.clear()
    generate_llm_tips(shots, stats_summary=None, session_context=None)
    assert "session_context" not in captured["messages"][0]["content"]


def test_generate_llm_tips_folds_frames_into_the_same_single_call(monkeypatch):
    captured = {}

    class FakeTextBlock:
        type = "text"
        text = '{"tips": [], "player_themes": []}'

    class FakeResponse:
        stop_reason = "end_turn"
        content = [FakeTextBlock()]

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return FakeResponse()

    class FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

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

    shots = [{"contact_id": "c1"}]
    frames = {"c1": [{"label": "prep", "jpeg_b64": "AAAA"},
                     {"label": "contact", "jpeg_b64": "BBBB"}]}
    generate_llm_tips(shots, frames_by_contact=frames)
    content = captured["messages"][0]["content"]
    assert isinstance(content, list)  # block list: JSON text + labeled images
    image_blocks = [b for b in content if b.get("type") == "image"]
    assert len(image_blocks) == 2
    assert image_blocks[0]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}
    labels = [b["text"] for b in content if b.get("type") == "text"]
    assert any("prep" in t and "c1" in t for t in labels)

    # Without frames the message stays a plain JSON string, exactly as before
    # vision context existed.
    captured.clear()
    generate_llm_tips(shots, frames_by_contact=None)
    assert isinstance(captured["messages"][0]["content"], str)


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


def test_system_prompt_frames_deviation_as_an_absolute_standard():
    # 2026-08-05 redesign: detection is continuous frame-by-frame (see
    # frame_metrics.py), so deviation is judged against an ABSOLUTE
    # standard for every metric now, never the player's own session — the
    # old per-player-relative Tukey-fence framing must not leak back in.
    assert "measured against an ABSOLUTE standard" in SYSTEM
    assert "against this player's own habits" in SYSTEM
    assert "relative to THIS PLAYER" not in SYSTEM
    assert "reference_flagged_metrics" not in SYSTEM


def test_system_prompt_describes_every_moment_as_a_span_not_an_instant():
    assert "Every moment you receive is a SPAN, not an instant" in SYSTEM
    assert "duration_s" in SYSTEM
    assert "never assert either" in SYSTEM


def test_system_prompt_encodes_three_beat_supportive_framing():
    # The reworked tone must require: acknowledge -> undiluted critique -> encouraging close.
    assert "three-beat" in SYSTEM
    assert "Do NOT soften, hedge, vague-up, or water down the" in SYSTEM
    assert "NOT a compliment sandwich that pads or dilutes" in SYSTEM


def test_system_prompt_form_describes_knee_and_elbow_metrics():
    sp = coaching.system_prompt(coaching.FOCUS_FORM)
    assert "knee_bend_deg" in sp
    assert "elbow_extension_deg" in sp
    assert "contact_height_m" not in sp
    assert "distance_from_net_m" not in sp


def test_system_prompt_covers_session_context_general_feedback_and_severity_bands():
    assert "session_context" in SYSTEM
    # One praise + one improve per player, grounded in the deterministic
    # general_context aggregates — replaced the old player_themes.
    assert "general_feedback" in SYSTEM
    assert "general_context" in SYSTEM
    assert "player_themes" not in SYSTEM
    # Critique intensity must be calibrated to |deviation_score| bands,
    # qualitatively — never recited as a number.
    assert "Calibrate how strongly you word a critique" in SYSTEM


def test_system_prompt_demands_layman_language():
    assert "never had a coach and doesn't know coaching" in SYSTEM
    assert "plain-English translation" in SYSTEM


def test_system_prompt_bans_fabricating_a_criticism_when_genuinely_clean():
    assert "genuinely clean" in SYSTEM
    assert "Never manufacture a flaw" in SYSTEM


def test_system_prompt_no_longer_references_retired_shot_type_or_rally_context():
    # 2026-08-05 redesign: shot_type stratification, mixed_baseline_metrics,
    # and rally_shot_index/is_rally_ending were all part of the retired
    # per-contact pipeline and never appear on a FrameSpikeVector — the
    # prompt must not describe fields that no longer exist.
    assert "shot_type" not in SYSTEM
    assert "mixed_baseline_metrics" not in SYSTEM
    assert "is_rally_ending" not in SYSTEM


def test_system_prompt_makes_frames_primary_evidence_without_breaking_voice():
    # Frames outrank the noisy 2D pose numbers when both exist...
    assert "PRIMARY" in SYSTEM
    # ...visible evidence may override a null metric (but never fabrication)...
    assert "not describing what is visibly in front of you" in SYSTEM
    # ...and the coach voice never breaks the fourth wall about images/AI.
    assert "never mention frames, images, crops," in SYSTEM


# ------------------------------------------------- focus-composed prompts


def test_module_level_system_is_the_form_composition():
    # The pinned-substring tests above all target SYSTEM; this guarantees
    # they keep covering what the form path actually sends.
    assert SYSTEM == coaching.system_prompt(coaching.FOCUS_FORM)


def test_system_prompt_positioning_describes_its_own_metrics():
    sp = coaching.system_prompt(coaching.FOCUS_POSITIONING)
    assert "kitchen_distance_m" in sp
    assert "transition_zone_occupancy" in sp
    assert "lateral_offset_m" in sp
    assert "partner_depth_stagger_m" in sp
    # Retired per-contact positioning metrics must not reappear.
    assert "kitchen_distance_at_contact_m" not in sp
    assert "return_depth_m" not in sp
    assert "kitchen_approach_after_return_s" not in sp


def test_both_focus_prompts_share_the_same_absolute_deviation_framing():
    # Both foci now use identical absolute-target framing (2026-08-05
    # redesign) — no focus-specific deviation semantics to diverge anymore.
    form = coaching.system_prompt(coaching.FOCUS_FORM)
    positioning = coaching.system_prompt(coaching.FOCUS_POSITIONING)
    assert "measured against an ABSOLUTE standard" in form
    assert "measured against an ABSOLUTE standard" in positioning


def test_both_focus_prompts_keep_null_discipline_no_raw_numbers_and_final_cut():
    for focus in (coaching.FOCUS_FORM, coaching.FOCUS_POSITIONING):
        sp = coaching.system_prompt(focus)
        assert "NEVER mention degrees, angles, confidence/certainty scores" in sp
        assert "must NEVER state or\nimply a specific claim about a joint/measurement" in sp
        assert "THE FINAL CUT IS YOURS" in sp


def test_system_prompt_rejects_unknown_focus():
    try:
        coaching.system_prompt("strategy")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --------------------------------------------------------- fill to limit


def _soft_vec(contact_id, deviation_score, soft_deviation):
    vec = _vec(contact_id, "footwork", deviation_score)
    vec.soft_deviation = soft_deviation
    return vec


def test_fill_to_limit_tops_up_with_highest_soft_deviation_marked_below_threshold():
    # 3 flagged + 10 in-band shots, all timestamps well beyond MIN_TIP_GAP_S
    # apart. fill_to_limit must top the list up to `limit`, ranking filler by
    # |soft_deviation| and marking every filler summary below_threshold.
    flagged = [(_soft_vec(f"bad{i}", -2.0, -2.0), _contact(f"bad{i}", timestamp_s=i * 10.0))
               for i in range(3)]
    inband = [(_soft_vec(f"ok{i}", 0.0, 0.9 - i * 0.05), _contact(f"ok{i}", timestamp_s=100.0 + i * 10.0))
              for i in range(10)]
    candidates = select_tip_candidates(flagged + inband, limit=8, fill_to_limit=True)
    assert len(candidates) == 8
    flagged_out = [c for c in candidates if not c.get("below_threshold")
                   and c["deviation_score"] != 0.0]
    filler = [c for c in candidates if c.get("below_threshold")]
    assert {c["contact_id"] for c in flagged_out} == {"bad0", "bad1", "bad2"}
    assert len(filler) >= 4
    # Filler ranked by |soft_deviation| desc: ok0 (0.9) before ok1 (0.85), ...
    filler_ids = [c["contact_id"] for c in filler]
    assert filler_ids == sorted(filler_ids, key=lambda cid: int(cid[2:]))
    # Ordering contract build_report's hard cap relies on: flagged first,
    # filler after, praise (if any) at the very end.
    first_filler = candidates.index(filler[0])
    assert all(candidates.index(c) < first_filler for c in flagged_out)


def test_fill_to_limit_off_keeps_adaptive_count():
    flagged = [(_soft_vec(f"bad{i}", -2.0, -2.0), _contact(f"bad{i}", timestamp_s=i * 10.0))
               for i in range(3)]
    inband = [(_soft_vec(f"ok{i}", 0.0, 0.5), _contact(f"ok{i}", timestamp_s=100.0 + i * 10.0))
              for i in range(10)]
    candidates = select_tip_candidates(flagged + inband, limit=8)
    # 3 flagged + at most MAX_PRAISE_TIPS praise exemplars — never padded.
    assert len(candidates) <= 3 + MAX_PRAISE_TIPS
    assert not any(c.get("below_threshold") for c in candidates)
    # And the adaptive flagged/not bar itself is untouched by fill plumbing.
    worst, _best = select_worst_best(flagged + inband)
    assert {s["contact_id"] for s in worst} == {"bad0", "bad1", "bad2"}


def test_fill_never_reintroduces_a_too_close_moment():
    flagged = [(_soft_vec("bad0", -2.0, -2.0), _contact("bad0", timestamp_s=10.0))]
    # In-band moment 1s from the flagged one (same player): must never fill.
    close = (_soft_vec("close", 0.0, 0.9), _contact("close", timestamp_s=11.0, confidence=0.1))
    far = (_soft_vec("far", 0.0, 0.5), _contact("far", timestamp_s=50.0, confidence=0.1))
    candidates = select_tip_candidates(flagged + [close, far], limit=8, fill_to_limit=True)
    ids = {c["contact_id"] for c in candidates}
    assert "close" not in ids
    assert "far" in ids


# ------------------------------------------------------- singles pid map


def test_singles_pid_map_matches_merge_fold_aggregate():
    contacts = (
        [{"player_id": 1, "frame_idx": f} for f in (0, 10, 20, 30)]
        + [{"player_id": 2, "frame_idx": f} for f in (5, 15, 25, 35)]
        + [{"player_id": 3, "frame_idx": 6}]
        + [{"player_id": 4, "frame_idx": 31}]
    )
    assert coaching.singles_pid_map(contacts, "singles") == {3: 2, 4: 1}
    assert coaching.singles_pid_map(contacts, "doubles") == {}
    assert coaching.singles_pid_map(contacts, None) == {}
    assert coaching.singles_pid_map(contacts[:8], "singles") == {}  # already 2 ids


# ---------------------------------------------- universal reference bands


def _raw_metrics(**overrides):
    raw = {f: None for f in METRIC_FIELDS}
    raw["low_confidence_metrics"] = [f for f in METRIC_FIELDS if f not in overrides]
    raw.update(overrides)
    return raw


def test_consistent_gross_flaw_flags_via_reference_despite_perfect_consistency():
    # THE Tukey-fence blind spot: a player who stands straight-legged (178°)
    # on EVERY shot is perfectly consistent — their own fence sees dev 0.0 on
    # every shot — but the universal reference check must still flag it.
    fences = {"knee_bend_deg": coaching._robust_fence([178.0] * 6)}
    vec = _finalize_vector({"contact_id": "c1", "player_id": 1},
                            _raw_metrics(knee_bend_deg=178.0), fences)
    assert vec is not None
    assert vec.deviation_score > 1.0                       # flagged
    assert vec.category == "footwork"
    assert vec.reference_flagged_metrics == ["knee_bend_deg"]
    assert "knee_bend_deg" in vec.deviating_metrics


def test_reference_flags_with_no_personal_fence_at_all():
    # A beginner's very first shots (< MIN_SAMPLES_FOR_DISTRIBUTION) used to
    # be unscoreable; the reference check works from shot one.
    vec = _finalize_vector({"contact_id": "c1", "player_id": 1},
                            _raw_metrics(knee_bend_deg=180.0), fences={})
    assert vec is not None
    assert vec.deviation_score == 2.0                      # (180-168)/6
    assert vec.reference_flagged_metrics == ["knee_bend_deg"]
    assert "knee_bend_deg" in vec.no_comparison_metrics    # still honest: no personal baseline


def test_reference_never_certifies_in_range_readings():
    # A knee angle inside the reference band with no personal fence stays
    # unscoreable — the reference detects gross errors, it never fabricates
    # "solid shot" praise from the mere absence of one.
    vec = _finalize_vector({"contact_id": "c1", "player_id": 1},
                            _raw_metrics(knee_bend_deg=130.0), fences={})
    assert vec is None


def test_personal_deviation_wins_when_larger_and_keeps_personal_framing():
    # Value 170 is barely over the reference bar (dev 0.33) but way outside
    # this player's own tight 100-110 spread — the personal comparison is the
    # bigger story and reference framing must NOT be claimed.
    fences = {"knee_bend_deg": coaching._robust_fence([100.0, 102.0, 105.0, 108.0, 110.0])}
    vec = _finalize_vector({"contact_id": "c1", "player_id": 1},
                            _raw_metrics(knee_bend_deg=170.0), fences)
    assert vec.reference_flagged_metrics == []
    assert vec.deviation_score > 1.0


def test_fast_only_reference_bands_gated_on_verified_fast_shot_type():
    # Zero weight transfer is a flaggable flaw on a drive, legitimate on a
    # dink, and unjudgeable when tempo is unknown.
    raw = _raw_metrics(weight_transfer_score=0.0)
    fast = _finalize_vector({"contact_id": "c1", "player_id": 1, "shot_type": "backcourt_fast"},
                             raw, fences={})
    assert fast is not None and fast.deviation_score == -2.0
    assert fast.reference_flagged_metrics == ["weight_transfer_score"]
    for shot_type in ("kitchen_soft", "midcourt", None):
        assert _finalize_vector({"contact_id": "c1", "player_id": 1, "shot_type": shot_type},
                                 raw, fences={}) is None


# ------------------------------------------------- ball-context plumbing


def test_shot_summary_carries_ball_context_only_when_present():
    vec = _vec("c1", "footwork", -2.0)
    plain = coaching._shot_summary(vec, _contact("c1"))
    assert "tempo_verified" not in plain and "rally_end_ball" not in plain
    enriched = coaching._shot_summary(vec, {**_contact("c1"), "tempo_verified": "fast",
                                             "rally_end_ball": "landed_out"})
    assert enriched["tempo_verified"] == "fast"
    assert enriched["rally_end_ball"] == "landed_out"


def test_classify_shot_types_ball_tempo_override_beats_wrist_inference():
    from ..shot_context import classify_shot_types
    contacts = [{"contact_id": f"c{i}", "player_id": 1, "court_y": 0.5,
                 "hand_speed_peak": 100.0 + i} for i in range(4)]
    # Wrist inference calls c3 fast (above own median); the measured ball
    # speed says soft — ball wins. And an untyped-tempo player (too few wrist
    # samples) still gets a tempo from the ball.
    types = classify_shot_types(contacts, tempo_override={"c3": "soft"})
    assert types["c3"] == "backcourt_soft"
    lone = [{"contact_id": "only", "player_id": 2, "court_y": 0.5, "hand_speed_peak": 50.0}]
    assert classify_shot_types(lone) == {"only": "backcourt"}
    assert classify_shot_types(lone, tempo_override={"only": "fast"}) == {"only": "backcourt_fast"}


# ------------------------------------------------------ category cap


def test_category_cap_stops_one_failure_mode_monopolizing_the_shortlist():
    dominant = [(_soft_vec(f"d{i}", -3.0, -3.0), _contact(f"d{i}", timestamp_s=i * 10.0))
                for i in range(12)]
    other = [(_vec(f"o{i}", "paddle_prep", -1.5), _contact(f"o{i}", timestamp_s=200.0 + i * 10.0))
             for i in range(2)]
    candidates = select_tip_candidates(dominant + other, limit=8, category_cap=4)
    cats = [c["category"] for c in candidates if c["deviation_score"] != 0.0]
    assert cats.count("footwork") <= 4
    assert "paddle_prep" in cats
    # Single-category sessions are never truncated by the cap.
    solo = select_tip_candidates(dominant, limit=8, category_cap=4)
    assert len([c for c in solo if c["deviation_score"] != 0.0]) >= 7


def test_capped_out_flags_never_reappear_as_below_threshold_filler():
    dominant = [(_soft_vec(f"d{i}", -3.0, -3.0), _contact(f"d{i}", timestamp_s=i * 10.0))
                for i in range(12)]
    other = [(_vec("o0", "paddle_prep", -1.5), _contact("o0", timestamp_s=200.0))]
    candidates = select_tip_candidates(dominant + other, limit=16,
                                        category_cap=4, fill_to_limit=True)
    for c in candidates:
        if c.get("below_threshold"):
            raise AssertionError(f"flagged shot mislabeled as filler: {c['contact_id']}")


# ------------------------------------------------- feedback-history pins


def test_both_focus_prompts_cover_feedback_history():
    # ball_context.py's tempo_verified/rally_end_ball fields are not wired
    # into the frame-spike pipeline (2026-08-05 redesign) — spikes carry
    # neither, so the prompt must not describe them.
    for focus in (coaching.FOCUS_FORM, coaching.FOCUS_POSITIONING):
        sp = coaching.system_prompt(focus)
        assert "tempo_verified" not in sp
        assert "rally_end_ball" not in sp
        assert "tip_feedback_history" in sp
        assert "NEVER\nsuppress a clearly severe moment" in sp


def test_generate_llm_tips_folds_feedback_history_into_payload(monkeypatch):
    captured = {}

    class FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)

            class R:
                stop_reason = "end_turn"
                content = [type("B", (), {"type": "text",
                                           "text": '{"tips": [], "general_feedback": []}'})()]

            class S:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def get_final_message(self):
                    return R()
            return S()

    class FakeClient:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setattr(coaching, "ANTHROPIC_API_KEY", "k")
    hist = {"footwork": {"helpful": 3, "not_helpful": 1}}
    generate_llm_tips([{"contact_id": "c1"}], feedback_history=hist)
    import json
    payload = json.loads(captured["messages"][0]["content"])
    assert payload["tip_feedback_history"] == hist
    # And absent history adds no key at all.
    generate_llm_tips([{"contact_id": "c1"}])
    payload = json.loads(captured["messages"][0]["content"])
    assert "tip_feedback_history" not in payload


# --------------------------------- positioning vectors through selection


def test_positioning_vectors_flow_through_selection_with_their_categories():
    from .. import positioning
    from ..contracts import PositioningFeatureVector

    def pvec(contact_id, deviation_score, stagger):
        return PositioningFeatureVector(
            contact_id=contact_id, category="team_spacing",
            kitchen_distance_at_contact_m=None, return_depth_m=None,
            transition_dwell_s=None, kitchen_approach_after_return_s=None,
            recovery_offset_m=None, partner_depth_stagger_m=stagger,
            deviation_score=deviation_score,
            deviating_metrics=["partner_depth_stagger_m"] if deviation_score else [],
            soft_deviation=deviation_score or 0.2)

    vectors_meta = [
        (pvec("flag1", 2.0, 4.5), _contact("flag1", timestamp_s=10.0)),
        (pvec("okay1", 0.0, 1.0), _contact("okay1", timestamp_s=50.0)),
    ]
    candidates = select_tip_candidates(vectors_meta, limit=8,
                                        category_metrics=positioning.CATEGORY_METRICS)
    by_id = {c["contact_id"]: c for c in candidates}
    assert by_id["flag1"]["category"] == "team_spacing"
    # Praise exemplar picked via the POSITIONING category map: the in-band
    # moment has a real partner_depth_stagger_m reading, so it qualifies.
    assert by_id["okay1"]["exemplar_of_category"] == "team_spacing"


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

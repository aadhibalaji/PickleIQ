"""Orchestrator behaviour: splits, end snap, output contract (spec 4-6)."""
from __future__ import annotations

from ..classifiers import RulesClassifier
from ..config import RefineConfig
from ..io_loaders import Pass1Window
from ..refine import refine_window
from .synth import FPS, serve_rally_scene, toss_scene, two_rally_scene

CFG = RefineConfig()

RECORD_KEYS = {
    "pass1_window_id", "pass1_start", "pass1_end",
    "pass1_start_frame", "pass1_end_frame",
    "verdict", "rejection_reason",
    "refined_start", "refined_end", "refined_start_frame", "refined_end_frame",
    "confidence", "splits", "diagnostics",
}
CONFIDENCE_KEYS = {"mean_p_live", "serve_gate_score", "net_crossings_per_10s"}


def test_split_window_emits_two_records_same_id():
    td, spans = two_rally_scene()
    w = Pass1Window("w-split", int((spans[0][0] - 1) * FPS),
                    int((spans[1][1] + 1) * FPS))
    dbg = refine_window(w, td, CFG, RulesClassifier(CFG))
    rallies = [r for r in dbg.records if r["verdict"] == "rally"]
    assert len(rallies) == 2
    assert all(r["pass1_window_id"] == "w-split" for r in rallies)
    assert all(len(r["splits"]) == 2 for r in rallies)
    # records don't overlap and are ordered
    assert rallies[0]["refined_end"] <= rallies[1]["refined_start"]


def test_end_snaps_to_last_bounce_not_hysteresis_lapse():
    td, t_contact, t_last_bounce = serve_rally_scene(tail_s=4.0)
    w = Pass1Window("w-end", int((t_contact - 1) * FPS), td.n - 1)
    dbg = refine_window(w, td, CFG, RulesClassifier(CFG))
    rallies = [r for r in dbg.records if r["verdict"] == "rally"]
    assert len(rallies) == 1
    want = t_last_bounce + CFG.post_roll_s
    assert abs(rallies[0]["refined_end"] - want) < 0.6
    # start anchored at serve contact minus pre-roll
    assert abs(rallies[0]["refined_start"] - (t_contact - CFG.pre_roll_s)) < 0.6


def test_output_contract():
    td, t_contact, _ = serve_rally_scene()
    w = Pass1Window("w-contract", int((t_contact - 1) * FPS), td.n - 1)
    dbg = refine_window(w, td, CFG, RulesClassifier(CFG))
    for r in dbg.records:
        assert set(r) == RECORD_KEYS
        assert set(r["confidence"]) == CONFIDENCE_KEYS
        assert r["verdict"] in ("rally", "rejected")
        if r["verdict"] == "rally":
            assert r["rejection_reason"] is None
            assert isinstance(r["refined_start_frame"], int)
            assert r["refined_start"] < r["refined_end"]
        else:
            assert r["rejection_reason"] is not None
            if r["rejection_reason"] == "too_short":
                # carries the refined (trimmed) span it was measured against,
                # so the UI can show raw-vs-trimmed duration for a drop
                assert r["refined_start"] is not None and r["refined_end"] is not None
                assert r["refined_start"] < r["refined_end"]
            else:
                assert r["refined_start"] is None and r["refined_end"] is None


def test_rejected_window_carries_reason():
    td = toss_scene()
    w = Pass1Window("w-toss", 20, td.n - 20)
    dbg = refine_window(w, td, CFG, RulesClassifier(CFG))
    assert dbg.records
    for r in dbg.records:
        assert r["verdict"] == "rejected"
        assert r["rejection_reason"] in (
            "no_serve", "low_live_ratio", "too_short")

"""Stage D on scripted scenes (spec test 3)."""
from __future__ import annotations

from ..classifiers import RulesClassifier
from ..config import RefineConfig
from ..features import compute_features
from ..io_loaders import Pass1Window
from ..refine import refine_window
from ..serve_gate import score_serve
from .synth import drill_scene, serve_rally_scene, toss_scene

CFG = RefineConfig()


def test_gate_accepts_legal_serve_with_deadzone_gap():
    """The scripted serve keeps the ball hidden in the server's deadzone
    until contact — the gate must still find it and anchor there."""
    td, t_contact, _ = serve_rally_scene()
    feats = compute_features(td, CFG)
    g = score_serve(feats, t_contact + 0.8, t_contact + 5.5, CFG,
                    search_from=t_contact - 1.0)
    assert g.score >= CFG.serve_gate_min + 0.1
    assert g.components["contact"] > 0.3
    assert g.contact_t is not None
    assert abs(g.contact_t - t_contact) < 0.35


def test_gate_rejects_toss_to_partner():
    td = toss_scene()
    feats = compute_features(td, CFG)
    g = score_serve(feats, 1.5, 8.0, CFG)
    assert g.score < CFG.serve_gate_min
    assert g.components["contact"] == 0.0


def test_gate_rejects_midcourt_drill():
    """Kitchen dinking has net crossings but no baseline serve signature —
    end-to-end the window must come out rejected with reason no_serve."""
    td = drill_scene()
    w = Pass1Window("drill", 20, td.n - 20)
    dbg = refine_window(w, td, CFG, RulesClassifier(CFG))
    assert dbg.records, "drill window must produce a record"
    assert all(r["verdict"] == "rejected" for r in dbg.records)
    assert any(r["rejection_reason"] == "no_serve" for r in dbg.records)

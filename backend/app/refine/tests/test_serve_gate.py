"""Stage D on scripted scenes (spec test 3)."""
from __future__ import annotations

from ..classifiers import RulesClassifier
from ..config import RefineConfig
from ..features import compute_features
from ..io_loaders import Pass1Window
from ..refine import refine_window
from ..serve_gate import score_serve
from ..io_loaders import COURT_L
from .synth import (_lerp, add_doubles_players, blank, drill_scene,
                    scripted_rally, scripted_serve, serve_rally_scene,
                    set_ball, toss_scene)

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


def test_gate_finds_real_serve_after_false_start_toss():
    """A Stage C segment can legitimately open with a practice toss before the
    real serve strike. The real contact here sits at t=5.0s -- past the OLD
    min(seg_start + serve_search_s, seg_end) = 3.0s bound -- and must still be
    found by searching the whole segment and trying each contact candidate's
    full gate (contact+flight+landing), not just the single best-looking one."""
    td = blank(16.0)
    add_doubles_players(td, jitter=0.2)
    server_xy = (4.5, COURT_L + 0.2)
    toss_up = (server_xy[0], server_xy[1] - 0.3)
    set_ball(td, _lerp(server_xy, toss_up, 0.5, 1.0), 0.5, 1.0)
    set_ball(td, _lerp(toss_up, server_xy, 1.0, 1.5), 1.0, 1.5)
    t_contact = 5.0
    t_land = scripted_serve(td, t_contact, server_xy=server_xy)
    scripted_rally(td, t_land + 0.15, t_contact + 6.0)

    feats = compute_features(td, CFG)
    g = score_serve(feats, 0.0, 12.0, CFG)
    assert g.score >= CFG.serve_gate_min
    assert g.pass_path is not None
    assert g.contact_t is not None
    assert abs(g.contact_t - t_contact) < 0.5


def test_gate_rejects_midcourt_drill():
    """Kitchen dinking has net crossings but no baseline serve signature —
    end-to-end the window must come out rejected with reason no_serve."""
    td = drill_scene()
    w = Pass1Window("drill", 20, td.n - 20)
    dbg = refine_window(w, td, CFG, RulesClassifier(CFG))
    assert dbg.records, "drill window must produce a record"
    assert all(r["verdict"] == "rejected" for r in dbg.records)
    assert any(r["rejection_reason"] == "no_serve" for r in dbg.records)

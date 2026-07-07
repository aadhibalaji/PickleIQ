"""Deterministic parameter proposals derived from a discrepancy report.

This is the reliable half of compound learning (the LLM report is the
explanatory half). It emits two independent, risk-tagged signals:

  * boundary offsets (LOW risk, auto-applyable): learned from *clean* matched
    pairs only — a boundary that came from a real detected serve / confirmed
    terminal event, not the coarse-gate fallback (whose error is a padding
    artefact, not systematic drift). Start and end are DECOUPLED, so a clip
    with a clean start but messy ends still compounds the start offset. This
    is what lets systematic drift converge toward zero even on imperfect clips.

  * Pass 1 gate (MEDIUM risk, human-gated): signed rally-count pressure
    (missed - extra). Unlike the old "missed>=2 AND zero extra" guards, this
    makes progress on the normal clip that has BOTH missed and extra, as long
    as one clearly dominates; it no-ops when they roughly balance (a balanced
    clip is a structural problem for the LLM/human, not a gate nudge).

All proposals are global — they alter the baseline every future clip starts
from. Offsets are half-median damped and per-report capped so repeated reports
converge instead of oscillating.
"""
from __future__ import annotations

import statistics
from typing import Any

# frames: ignore a "clean" pair whose boundary is off by more than ~1.5 s — that
# far off is a mis-detection masquerading as drift, not a systematic offset.
# (The clean_start/clean_end flags already drop coarse-gate fallbacks; this only
# needs to catch a confidently-wrong serve/terminal detection. Keep it generous
# so it trims outliers without biasing the median of the normal error spread.)
_OFFSET_LEARN_CAP = 45
# frames: never move an offset more than this per report (anti-thrash / anti-
# oscillation, on top of the half-median damping).
_OFFSET_STEP_CAP = 5


def _offset_proposal(param: str, errs: list[int], current: dict[str, Any]) -> dict | None:
    if len(errs) < 2:
        return None
    med = statistics.median(errs)
    if abs(med) < 3:                       # sub-100ms @30fps: not worth chasing
        return None
    cur = int(current.get(param, 0))
    step = int(round(-med / 2))            # error = ai - human, so subtract it
    step = max(-_OFFSET_STEP_CAP, min(_OFFSET_STEP_CAP, step))
    if step == 0:
        return None
    which = param.split("_")[0]            # 'start' | 'end'
    return {
        "param": param, "from": cur, "to": cur + step, "risk": "low",
        "evidence_n": len(errs),
        "rationale": (f"{which} boundary off by median {med:+.0f} frames across "
                      f"{len(errs)} clean pair(s) -> {param} {step:+d}"),
    }


def _gate_proposal(param: str, current: dict[str, Any], factor: float,
                   rationale: str) -> dict:
    cur = float(current.get(param))
    return {"param": param, "from": cur, "to": cur * factor, "risk": "medium",
            "evidence_n": None, "rationale": rationale}


def compute_param_updates(
    discrepancies: list[dict], current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return a list of risk-tagged proposals:
    ``[{param, from, to, risk: 'low'|'medium', rationale, evidence_n}, ...]``.
    Low-risk = boundary offsets (safe to auto-apply after harness validation);
    medium-risk = Pass 1 gate changes (route to human approval)."""
    proposals: list[dict[str, Any]] = []
    drifts = [d for d in discrepancies if d["type"] == "boundary_drift"]
    missed = [d for d in discrepancies if d["type"] == "missed"]
    extra = [d for d in discrepancies if d["type"] == "extra"]

    # ---- boundary offsets (low risk) — clean, capped pairs only ----------
    start_errs = [d["start_error"] for d in drifts
                  if d.get("clean_start") and d.get("start_error") is not None
                  and abs(d["start_error"]) <= _OFFSET_LEARN_CAP]
    end_errs = [d["end_error"] for d in drifts
                if d.get("clean_end") and d.get("end_error") is not None
                and abs(d["end_error"]) <= _OFFSET_LEARN_CAP]
    for p in (_offset_proposal("start_offset_frames", start_errs, current),
              _offset_proposal("end_offset_frames", end_errs, current)):
        if p:
            proposals.append(p)

    # ---- Pass 1 gate (medium risk) — signed rally-count pressure ---------
    net = len(missed) - len(extra)
    if net >= 2:
        proposals.append(_gate_proposal(
            "p1_motion_ratio", current, 0.9,
            f"{len(missed)} missed vs {len(extra)} extra (net +{net}) -> "
            "loosen Pass 1 gate (motion_ratio -10%)"))
        proposals.append(_gate_proposal(
            "p1_off_s", current, 1.15,
            "loosen active->dead hysteresis (+15%) so rallies hold together"))
    elif net <= -2:
        proposals.append(_gate_proposal(
            "p1_motion_ratio", current, 1.1,
            f"{len(extra)} extra vs {len(missed)} missed (net {net}) -> "
            "tighten Pass 1 gate (motion_ratio +10%)"))
        proposals.append(_gate_proposal(
            "p1_min_rally_s", current, 1.1,
            "raise min rally length (+10%) to drop short false positives"))

    return proposals


def split_by_risk(proposals: list[dict], risk: str) -> dict[str, Any]:
    """Collapse proposals of a given risk level into a {param: target} dict
    ready for ``learned_params.apply_updates``."""
    return {p["param"]: p["to"] for p in proposals if p["risk"] == risk}

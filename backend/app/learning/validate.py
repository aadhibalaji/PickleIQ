"""Held-out validation harness — the guardrail that makes auto-applying a
learned change safe. A change is committed globally only if it improves (or at
least does not regress) the aggregate boundary accuracy across EVERY labeled
clip, not just the one that triggered it.

Two modes:

  * analytic (offset changes) — boundary offsets are pure post-processing on
    the raw Pass 2 boundaries stored in each rally's meta (raw_start/raw_end/
    ws/we). A candidate offset is re-scored across all labeled clips in a few
    milliseconds, no CV. This gates the autonomous auto-apply of offsets.

  * full re-run (gate / structural changes) — re-runs Pass 1 + Pass 2 with the
    candidate params on each labeled clip into scratch sources, then compares.
    Expensive; used only on the human-approval path.

The metric pools matched pairs across ALL clips (not an average of per-clip
averages, which would over-weight clips with few rallies): mean absolute
start/end error in frames, plus match_rate and false-positive (extra) count.
"""
from __future__ import annotations

import numpy as np

from .. import db
from ..analysis.metrics import Rally, compare

# A change must not worsen a non-targeted metric by more than this to pass.
_EPS_FRAMES = 0.5
_EPS_RATE = 0.01


def labeled_video_ids() -> list[str]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT DISTINCT video_id FROM rally_labels WHERE source='human'").fetchall()
    return [r["video_id"] for r in rows]


def _load(video_id: str, ai_source: str):
    """Return (human Rallies, [(start, end, meta), ...] for ai_source, fps)."""
    with db.connect() as con:
        v = con.execute("SELECT fps FROM videos WHERE id=?", (video_id,)).fetchone()
        rows = con.execute(
            "SELECT source, start_frame, end_frame, meta FROM rally_labels"
            " WHERE video_id=? AND source IN ('human', ?)", (video_id, ai_source)).fetchall()
    fps = (v["fps"] if v else None) or 30.0
    human = [Rally(r["start_frame"], r["end_frame"]) for r in rows if r["source"] == "human"]
    ai = [(r["start_frame"], r["end_frame"], db.loads(r["meta"], {}))
          for r in rows if r["source"] == ai_source]
    return human, ai, fps


def _pool(per_clip: list[dict]) -> dict:
    """Pool raw per-pair errors across clips into one honest metric set."""
    abs_s = [e for c in per_clip for e in c["abs_start"]]
    abs_e = [e for c in per_clip for e in c["abs_end"]]
    n_human = sum(c["n_human"] for c in per_clip)
    n_matched = sum(c["n_matched"] for c in per_clip)
    n_ai = sum(c["n_ai"] for c in per_clip)
    return {
        "mean_abs_start_err": round(float(np.mean(abs_s)), 2) if abs_s else None,
        "mean_abs_end_err": round(float(np.mean(abs_e)), 2) if abs_e else None,
        "match_rate": round(n_matched / n_human, 3) if n_human else None,
        "n_clips": len(per_clip), "n_human": n_human,
        "n_matched": n_matched, "n_extra": n_ai - n_matched, "n_ai": n_ai,
    }


def _score(human: list[Rally], ai: list[Rally], fps: float) -> dict:
    res = compare(human, ai, fps)
    abs_s = [abs(d["start_error"]) for d in res["discrepancies"] if d["type"] == "boundary_drift"]
    abs_e = [abs(d["end_error"]) for d in res["discrepancies"] if d["type"] == "boundary_drift"]
    s = res["summary"]
    return {"abs_start": abs_s, "abs_end": abs_e, "n_human": s["human_rallies"],
            "n_matched": s["matched"], "n_ai": s["ai_rallies"]}


# --------------------------------------------------------------- analytic ---
def _apply_offsets(ai_raw, start_off: int, end_off: int) -> list[Rally]:
    """Recompute each rally's final boundary from its stored raw boundary and a
    candidate offset — mirrors pass2.py: final = clip(raw + off, ws, we)."""
    out = []
    for (cur_s, cur_e, meta) in ai_raw:
        rs, re = meta.get("raw_start"), meta.get("raw_end")
        ws, we = meta.get("ws"), meta.get("we")
        if rs is None or re is None or ws is None or we is None:
            rs, re, ws, we = cur_s, cur_e, min(cur_s, cur_e), max(cur_s, cur_e)
        s = int(np.clip(rs + start_off, ws, we))
        e = int(np.clip(re + end_off, s + 1, we))
        out.append(Rally(s, e, "", meta))
    return out


def _analytic_stale(ai_raw) -> bool:
    """True if any stored boundary lacks the raw-boundary meta the analytic
    re-eval needs. Pass 2 stopped emitting rally boundaries (and the
    ``DETECTION_CODE_VERSION`` stamp) in the detection-only pivot — Pass 1 is
    the sole rally source and refine output is JSON-only — so any label
    without that meta is stale by definition and the clip is skipped."""
    return any(m.get("code_version") is None or m.get("raw_start") is None
               for (_, _, m) in ai_raw)


def analytic_metrics(start_off: int, end_off: int,
                     video_ids: list[str] | None = None, ai_source: str = "pass2") -> dict:
    video_ids = video_ids if video_ids is not None else labeled_video_ids()
    per_clip, stale = [], []
    for vid in video_ids:
        human, ai_raw, fps = _load(vid, ai_source)
        if not human or not ai_raw:
            continue
        if _analytic_stale(ai_raw):
            stale.append(vid)
            continue
        per_clip.append(_score(human, _apply_offsets(ai_raw, start_off, end_off), fps))
    out = _pool(per_clip)
    out["stale"] = stale
    return out


def evaluate_offset_change(current_params: dict, candidate_updates: dict,
                           video_ids: list[str] | None = None) -> dict:
    """before = current offsets, after = current offsets overridden by the
    candidate. Only start_offset_frames/end_offset_frames are analytic."""
    cur_s = int(current_params.get("start_offset_frames", 0))
    cur_e = int(current_params.get("end_offset_frames", 0))
    new_s = int(candidate_updates.get("start_offset_frames", cur_s))
    new_e = int(candidate_updates.get("end_offset_frames", cur_e))
    targets = ([] + (["start"] if new_s != cur_s else []) + (["end"] if new_e != cur_e else []))
    before = analytic_metrics(cur_s, cur_e, video_ids)
    after = analytic_metrics(new_s, new_e, video_ids)
    return {"before": before, "after": after, "targets": targets,
            "accept": _accept_offset(before, after, targets)}


def _accept_offset(before: dict, after: dict, targets: list[str]) -> bool:
    if not after.get("n_matched"):
        return False
    if (before["match_rate"] is not None and after["match_rate"] is not None
            and after["match_rate"] < before["match_rate"] - _EPS_RATE):
        return False
    metric_of = {"start": "mean_abs_start_err", "end": "mean_abs_end_err"}
    # non-targeted error metrics must not degrade (a start change must not hurt end)
    for tkey, metric in metric_of.items():
        if tkey in targets:
            continue
        b, a = before[metric], after[metric]
        if b is not None and a is not None and a > b + _EPS_FRAMES:
            return False
    # every targeted metric must not regress, and at least one must improve
    improved = False
    for tkey in targets:
        b, a = before[metric_of[tkey]], after[metric_of[tkey]]
        if b is None or a is None or a > b + _EPS_FRAMES:
            return False
        if a < b - _EPS_FRAMES:
            improved = True
    return improved


# -------------------------------------------------------------- full rerun ---
class _SilentHandle:
    def update(self, progress=None, message=None):
        pass

    def cancelled(self) -> bool:
        return False


def _cleanup_eval(video_id: str) -> None:
    from ..cv import overlays
    with db.connect() as con:
        con.execute("DELETE FROM rally_labels WHERE video_id=?"
                    " AND source IN ('pass1_eval','pass2_eval')", (video_id,))
    overlays.clear(video_id, "pass1_eval")
    overlays.clear(video_id, "pass2_eval")


def full_rerun_metrics(candidate_params: dict,
                       video_ids: list[str] | None = None) -> dict:
    """Re-run the whole pipeline with candidate_params on each labeled clip into
    scratch sources, score, and clean up. Expensive (CV) — approval path only."""
    from ..pipeline import effective_calibration
    from ..cv.pass1 import run_pass1
    from ..cv.pass2 import run_pass2
    video_ids = video_ids if video_ids is not None else labeled_video_ids()
    per_clip = []
    for vid in video_ids:
        H, cam, _ = effective_calibration(vid)
        if H is None or cam is None:
            continue
        try:
            run_pass1(vid, candidate_params, H, _SilentHandle(), source="pass1_eval")
            run_pass2(vid, candidate_params, H, cam, _SilentHandle(),
                      source="pass2_eval", p1_source="pass1_eval")
            human, ai_raw, fps = _load(vid, "pass2_eval")
            if human:
                per_clip.append(_score(human, [Rally(s, e, "", m) for (s, e, m) in ai_raw], fps))
        finally:
            _cleanup_eval(vid)
    return _pool(per_clip)


def existing_metrics(video_ids: list[str] | None = None, ai_source: str = "pass2") -> dict:
    """Score the currently-committed labels as-is (the 'before' for a gate change,
    since they were produced by the current params)."""
    video_ids = video_ids if video_ids is not None else labeled_video_ids()
    per_clip = []
    for vid in video_ids:
        human, ai_raw, fps = _load(vid, ai_source)
        if human:
            per_clip.append(_score(human, [Rally(s, e, "", m) for (s, e, m) in ai_raw], fps))
    return _pool(per_clip)


def evaluate_gate_change(candidate_params: dict,
                         video_ids: list[str] | None = None) -> dict:
    """before = current committed labels, after = full re-run with candidate."""
    video_ids = video_ids if video_ids is not None else labeled_video_ids()
    before = existing_metrics(video_ids)
    after = full_rerun_metrics(candidate_params, video_ids)
    return {"before": before, "after": after, "accept": _accept_gate(before, after)}


def _accept_gate(before: dict, after: dict) -> bool:
    if not after.get("n_matched"):
        return False
    if after["n_matched"] < before["n_matched"]:        # lost a real rally -> reject
        return False
    if after["n_extra"] < before["n_extra"]:            # fewer false positives, recall kept
        return True
    # no fewer extras: only accept if a boundary error clearly improved
    for m in ("mean_abs_start_err", "mean_abs_end_err"):
        b, a = before[m], after[m]
        if b is not None and a is not None and a < b - _EPS_FRAMES:
            return True
    return False


def evaluate_model(kind: str, model, video_ids: list[str] | None = None) -> dict:
    """Score a (usually still-shadow) model by forcing it into a full pipeline
    re-run, vs the current committed labels (the heuristic incumbent). A
    rally_end model is accepted only if it lowers end-boundary error without
    losing matches.

    NOTE: for a rigorous promotion the caller should pass a HELD-OUT clip set
    (train excluding it) — leave-one-clip-out. In-sample scoring is optimistic;
    the train script only promotes at >=3 clips where a real split is possible."""
    from . import model_registry
    from . import params as learned_params
    _, current = learned_params.get_latest()
    before = existing_metrics(video_ids)
    model_registry.set_override(kind, model)
    try:
        after = full_rerun_metrics(current, video_ids)
    finally:
        model_registry.clear_override()
    return {"before": before, "after": after, "accept": _accept_model(before, after)}


def _accept_model(before: dict, after: dict) -> bool:
    if not after.get("n_matched"):
        return False
    if after["n_matched"] < before["n_matched"]:
        return False
    b, a = before["mean_abs_end_err"], after["mean_abs_end_err"]
    return b is not None and a is not None and a < b - _EPS_FRAMES

"""The autonomous learning step, run in the background whenever the user saves
corrected labels. It closes the loop safely:

  * LOW-RISK boundary offsets are validated against ALL labeled clips with the
    analytic harness and committed globally ONLY if aggregate accuracy improves
    (so one noisy clip can never move the baseline on its own).
  * MEDIUM-RISK Pass 1 gate changes are NEVER auto-applied — they are queued in
    ``pending_changes`` for human approval (which runs the full-rerun harness).

Every run also records a ``param_eval`` snapshot so the scoreboard can show the
start/end error trending over time. No CV runs here — the analytic harness
re-scores stored raw boundaries, so a save-triggered learn job is milliseconds.
"""
from __future__ import annotations

import hashlib

from .. import db
from ..analysis.metrics import Rally, compare
from . import params as learned_params
from . import validate
from .updates import compute_param_updates, split_by_risk


def labels_hash(video_id: str) -> str:
    with db.connect() as con:
        rows = con.execute(
            "SELECT start_frame, end_frame FROM rally_labels"
            " WHERE video_id=? AND source='human' ORDER BY start_frame", (video_id,)).fetchall()
    key = ";".join(f"{r['start_frame']}-{r['end_frame']}" for r in rows)
    return hashlib.sha1(key.encode()).hexdigest()


def _load_human_ai(video_id: str):
    with db.connect() as con:
        v = con.execute("SELECT fps FROM videos WHERE id=?", (video_id,)).fetchone()
        rows = con.execute(
            "SELECT source, start_frame, end_frame, meta FROM rally_labels"
            " WHERE video_id=? AND source IN ('human','pass2')", (video_id,)).fetchall()
    fps = (v["fps"] if v else None) or 30.0
    human = [Rally(r["start_frame"], r["end_frame"]) for r in rows if r["source"] == "human"]
    ai = [Rally(r["start_frame"], r["end_frame"], "", db.loads(r["meta"], {}))
          for r in rows if r["source"] == "pass2"]
    return human, ai, fps


def record_eval(version, scope: str, metrics: dict, applied: bool, detail: dict) -> None:
    with db.connect() as con:
        con.execute(
            "INSERT INTO param_eval (id, version, scope, mean_abs_start_err, mean_abs_end_err,"
            " match_rate, n_clips, n_matched, n_extra, applied, detail, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (db.new_id(), version, scope, metrics.get("mean_abs_start_err"),
             metrics.get("mean_abs_end_err"), metrics.get("match_rate"),
             metrics.get("n_clips"), metrics.get("n_matched"), metrics.get("n_extra"),
             1 if applied else 0, db.dumps(detail), db.now()))


def _queue_pending(video_id: str, proposals: list[dict]) -> list[str]:
    queued = []
    with db.connect() as con:
        for p in proposals:
            # Supersede any still-pending proposal for the same param (a fresh
            # save replaces stale advice instead of stacking duplicates).
            con.execute("UPDATE pending_changes SET status='superseded', decided_at=?"
                        " WHERE param=? AND status='pending'", (db.now(), p["param"]))
            pid = db.new_id()
            con.execute(
                "INSERT INTO pending_changes (id, video_id, param, from_value, to_value,"
                " risk, rationale, evidence, status, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pid, video_id, p["param"], p["from"], p["to"], p["risk"], p["rationale"],
                 db.dumps({"evidence_n": p["evidence_n"]}), "pending", db.now()))
            queued.append(pid)
    return queued


def run_learn(video_id: str) -> dict:
    """Compute and safely apply learning from the current human labels."""
    h = labels_hash(video_id)
    with db.connect() as con:
        prev = con.execute("SELECT last_labels_hash FROM videos WHERE id=?", (video_id,)).fetchone()
    if prev and prev["last_labels_hash"] == h:
        return {"skipped": "labels unchanged"}

    human, ai, fps = _load_human_ai(video_id)
    if not human or not ai:
        return {"skipped": "need both human labels and Pass 2 rallies"}

    version, current = learned_params.get_latest()
    result = compare(human, ai, fps)
    proposals = compute_param_updates(result["discrepancies"], current)
    low = split_by_risk(proposals, "low")
    medium = [p for p in proposals if p["risk"] == "medium"]

    # One applied snapshot per committed version, so the scoreboard has a point
    # for versions created by other means (manual report, accepted LLM change).
    with db.connect() as con:
        seen = con.execute("SELECT 1 FROM param_eval WHERE version=? AND applied=1 LIMIT 1",
                           (version,)).fetchone()
    if not seen:
        base = validate.analytic_metrics(int(current.get("start_offset_frames", 0)),
                                         int(current.get("end_offset_frames", 0)))
        record_eval(version, "baseline", base, applied=True, detail={"trigger": video_id})

    # LOW-RISK offsets: validate globally, commit only if it helps.
    applied_version = None
    verdict = None
    if low:
        verdict = validate.evaluate_offset_change(current, low)
        if verdict["accept"]:
            applied_version = learned_params.apply_updates(low, origin=f"auto:label:{video_id}")
            record_eval(applied_version, "auto:label", verdict["after"], applied=True,
                        detail={"applied": low, "before": verdict["before"], "trigger": video_id})
        else:
            record_eval(None, "auto:label", verdict["after"], applied=False,
                        detail={"rejected": low, "before": verdict["before"], "trigger": video_id})

    queued = _queue_pending(video_id, medium)

    with db.connect() as con:
        con.execute("UPDATE videos SET last_labels_hash=? WHERE id=?", (h, video_id))

    return {"summary": result["summary"], "applied_version": applied_version,
            "offset_verdict": verdict, "queued_pending": queued,
            "proposals": proposals}

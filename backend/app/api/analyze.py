"""Overlays, comparison, report generation, and the compound-learning
endpoints (LearnedParams versions, applying suggestions)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db
from ..analysis.metrics import Rally, compare
from ..analysis.reporter import generate_llm_report
from ..cv import overlays
from ..learning import params as learned_params
from ..learning.updates import compute_param_updates, split_by_risk

router = APIRouter()


@router.get("/videos/{video_id}/overlays")
def get_overlays(video_id: str, source: str, start: int = 0, end: int = 10**9):
    if source not in ("pass1", "pass2"):
        raise HTTPException(400, "source must be pass1 or pass2")
    return {"entries": overlays.read_range(video_id, source, start, end)}


@router.get("/videos/{video_id}/detect_debug")
def detect_debug(video_id: str):
    """Read-only diagnostics for why a clip matches (or doesn't) its human
    labels. First thing to inspect for a clip showing few matches: is it
    calibrated, did Pass 2 error, and is a single merged window swallowing
    more than one human rally (the bridging symptom)?"""
    from ..pipeline import effective_calibration
    with db.connect() as con:
        vrow = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if vrow is None:
            raise HTTPException(404, "video not found")
        rows = db.rows_to_dicts(con.execute(
            "SELECT source, start_frame, end_frame, meta FROM rally_labels WHERE video_id=?",
            (video_id,)).fetchall())
        pass2_job = con.execute(
            "SELECT status, message, updated_at FROM jobs"
            " WHERE video_id=? AND kind='pass2' ORDER BY updated_at DESC LIMIT 1",
            (video_id,)).fetchone()

    H, _cam, cal_id = effective_calibration(video_id)
    fps = vrow["fps"] or 30.0
    human = [Rally(r["start_frame"], r["end_frame"]) for r in rows if r["source"] == "human"]
    pass1 = [r for r in rows if r["source"] == "pass1"]
    pass2 = [r for r in rows if r["source"] == "pass2"]

    # Reconstruct the merged windows Pass 2 actually used, straight from its
    # stored meta (ws/we) — no need to duplicate the window-merge logic.
    merged = []
    for r in pass2:
        m = db.loads(r["meta"], {})
        ws, we = m.get("ws"), m.get("we")
        if ws is None or we is None:
            continue
        covered = [i for i, h in enumerate(human) if min(h.end, we) - max(h.start, ws) > 0]
        merged.append({"ws": ws, "we": we,
                       "start_source": m.get("start_source"),
                       "end_source": m.get("end_source"),
                       "human_covered": len(covered), "human_idx": covered})

    ai = [Rally(r["start_frame"], r["end_frame"]) for r in pass2]
    summary = compare(human, ai, fps)["summary"] if human else None

    hints: list[str] = []
    if H is None:
        hints.append("No calibration — Pass 2 cannot run (it needs a homography). "
                     "Calibrate the court corners, then Reprocess.")
    if pass2_job and pass2_job["status"] == "error":
        hints.append(f"Latest Pass 2 job errored: {pass2_job['message']}")
    bridged = [w for w in merged if w["human_covered"] > 1]
    if bridged:
        hints.append(
            f"{len(bridged)} merged window(s) each cover >1 human rally — adjacent "
            "rallies bridged into one candidate. Lower p1_merge_gap_s / p1_cross_window_s, "
            "or enable multi-rally Pass 2 (A5).")
    if not pass2:
        hints.append("No Pass 2 rallies yet — run or await processing.")

    return {
        "calibration": {"present": H is not None, "calibration_id": cal_id},
        "pass2_job": dict(pass2_job) if pass2_job else None,
        "counts": {"pass1_windows": len(pass1), "pass2_rallies": len(pass2),
                   "human": len(human)},
        "merged_windows": merged,
        "comparison": summary,
        "hints": hints,
    }


class ReportRequest(BaseModel):
    ai_source: str = "pass2"
    apply_deterministic_updates: bool = True


@router.post("/videos/{video_id}/report")
def generate_report(video_id: str, body: ReportRequest):
    with db.connect() as con:
        vrow = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if vrow is None:
            raise HTTPException(404, "video not found")
        rows = db.rows_to_dicts(con.execute(
            "SELECT * FROM rally_labels WHERE video_id=?", (video_id,)).fetchall())
        notes = db.rows_to_dicts(con.execute(
            "SELECT text, source, scope, video_time FROM user_notes WHERE video_id=?",
            (video_id,)).fetchall())
    human = [Rally(r["start_frame"], r["end_frame"], r["id"])
             for r in rows if r["source"] == "human"]
    ai = [Rally(r["start_frame"], r["end_frame"], r["id"], db.loads(r["meta"], {}))
          for r in rows if r["source"] == body.ai_source]
    if not human:
        raise HTTPException(400, "no human labels yet — label rallies first")

    fps = vrow["fps"] or 30.0
    result = compare(human, ai, fps)
    discrepancies = result["discrepancies"]

    # Per-frame feature traces around each disputed boundary (±2 s), pulled
    # from stored pass2 metadata — these are what let the LLM (and later a
    # trained model) see the signals that drove each decision. Attach them ONLY
    # to boundary_drift discrepancies: the per-frame trace explains WHY a found
    # boundary drifted, but missed/extra rallies are Pass 1 gate / min-duration
    # issues (Pass 2 only ran inside windows, so it has no trace for a rally it
    # never emitted) — shipping their traces just burned tokens for no signal.
    pad = int(fps * 2)
    SAMPLES = 20  # per boundary window (was ~60) — plenty to show the drift shape
    for d in discrepancies:
        if d["type"] != "boundary_drift":
            continue
        anchor = d["ai_rally"] or d["human_rally"]
        trace = {
            "start_window": overlays.read_range(
                video_id, "pass2",
                anchor["start_frame"] - pad, anchor["start_frame"] + pad),
            "end_window": overlays.read_range(
                video_id, "pass2",
                anchor["end_frame"] - pad, anchor["end_frame"] + pad),
        }
        # Slim the trace for storage/LLM: keep signal fields only.
        for k in trace:
            trace[k] = [
                {f: e.get(f) for f in ("f", "ball", "world", "pspeed", "evt", "obs")
                 if e.get(f) is not None}
                for e in trace[k][:: max(1, len(trace[k]) // SAMPLES)]
            ]
        d["feature_trace"] = trace

    version, current_params = learned_params.get_latest()

    # --- deterministic global learning ---------------------------------
    proposals = compute_param_updates(discrepancies, current_params)
    low = split_by_risk(proposals, "low")          # boundary offsets
    pending = [p for p in proposals if p["risk"] != "low"]  # gate changes -> human
    rationale = [p["rationale"] for p in proposals]
    applied_version = None
    report_id = db.new_id()
    # Manual "Generate report" applies the low-risk boundary offsets directly
    # (the fully-autonomous on-save path additionally validates them against a
    # held-out set before committing — see learning/autolearn + validate).
    if low and body.apply_deterministic_updates:
        applied_version = learned_params.apply_updates(low, origin=f"report:{report_id}")

    # --- LLM analysis ----------------------------------------------------
    payload = {
        "video": {"fps": fps, "frame_count": vrow["frame_count"]},
        "summary": result["summary"],
        "discrepancies": discrepancies,
        "user_notes": notes,
        "current_params": current_params,
        "deterministic_updates_applied": low if applied_version else {},
        "deterministic_rationale": rationale,
        "pending_gate_proposals": pending,
    }
    llm_report, suggestions = generate_llm_report(payload)

    with db.connect() as con:
        con.execute(
            "INSERT INTO reports (id, video_id, ai_source, summary, llm_report,"
            " llm_suggestions, applied_params_version, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (report_id, video_id, body.ai_source, db.dumps(result["summary"]),
             llm_report, db.dumps(suggestions), applied_version, db.now()),
        )
        for d in discrepancies:
            con.execute(
                "INSERT INTO discrepancies (id, video_id, report_id, human_rally,"
                " ai_rally, type, temporal_iou, start_error, end_error, feature_trace)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (db.new_id(), video_id, report_id,
                 db.dumps(d["human_rally"]), db.dumps(d["ai_rally"]), d["type"],
                 d["temporal_iou"], d["start_error"], d["end_error"],
                 db.dumps(d.pop("feature_trace", None))),
            )
    return {
        "report_id": report_id,
        "summary": result["summary"],
        "discrepancies": discrepancies,
        "proposals": proposals,
        "deterministic_updates": low,
        "deterministic_rationale": rationale,
        "pending_gate_proposals": pending,
        "applied_params_version": applied_version,
        "llm_report": llm_report,
        "llm_suggestions": suggestions,
    }


@router.get("/videos/{video_id}/reports")
def list_reports(video_id: str):
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM reports WHERE video_id=? ORDER BY created_at DESC", (video_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["summary"] = db.loads(d["summary"], {})
        d["llm_suggestions"] = db.loads(d["llm_suggestions"], [])
        out.append(d)
    return out


class ApplySuggestions(BaseModel):
    suggestions: list[dict]  # subset of a report's llm_suggestions the user accepted


@router.post("/reports/{report_id}/apply_suggestions")
def apply_suggestions(report_id: str, body: ApplySuggestions):
    # clip_local suggestions never touch the global baseline.
    updates = {s["param"]: s["value"] for s in body.suggestions
               if s.get("scope", "global") == "global"}
    skipped = [s["param"] for s in body.suggestions
               if s.get("scope") == "clip_local"]
    if not updates:
        return {"applied_version": None, "skipped_clip_local": skipped}
    version = learned_params.apply_updates(updates, origin=f"report:{report_id}:llm")
    return {"applied_version": version, "applied": updates, "skipped_clip_local": skipped}


@router.get("/pending_changes")
def list_pending_changes():
    """Gate changes awaiting human approval (plus recently decided ones)."""
    with db.connect() as con:
        rows = db.rows_to_dicts(con.execute(
            "SELECT * FROM pending_changes ORDER BY created_at DESC LIMIT 50").fetchall())
    for r in rows:
        r["evidence"] = db.loads(r["evidence"], {})
    return rows


@router.post("/pending_changes/{change_id}/approve")
def approve_change(change_id: str):
    """Approve a queued gate change: apply it immediately as a new global param
    version and mark it approved. The human reviewing the rationale IS the gate
    here, and the value is already BOUNDS-clamped — so this is synchronous and
    instant (the old version kicked off a minutes-long background re-run, which
    is why the button appeared to do nothing). The effect shows on the next
    reprocess; the autonomous on-save path still validates before auto-applying."""
    with db.connect() as con:
        row = con.execute("SELECT * FROM pending_changes WHERE id=?", (change_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "pending change not found")
    if row["status"] != "pending":
        raise HTTPException(400, f"change already {row['status']}")
    version = learned_params.apply_updates({row["param"]: row["to_value"]},
                                           origin=f"approved:{change_id}")
    ev = db.loads(row["evidence"], {})
    ev["applied_version"] = version
    with db.connect() as con:
        con.execute("UPDATE pending_changes SET status='approved', decided_at=?, evidence=?"
                    " WHERE id=?", (db.now(), db.dumps(ev), change_id))
    return {"applied_version": version, "param": row["param"], "to": row["to_value"]}


@router.post("/pending_changes/{change_id}/reject")
def reject_change(change_id: str):
    with db.connect() as con:
        con.execute("UPDATE pending_changes SET status='rejected', decided_at=?"
                    " WHERE id=? AND status='pending'", (db.now(), change_id))
    return {"rejected": True}


@router.get("/learning/scoreboard")
def learning_scoreboard():
    """The 'is it getting better?' trend: mean absolute start/end boundary error
    (frames) and match rate across labeled clips, per committed param version.
    Plus the most recent rejected candidates, so a refusal is visible too."""
    with db.connect() as con:
        evals = db.rows_to_dicts(con.execute(
            "SELECT pe.version, pe.scope, pe.mean_abs_start_err, pe.mean_abs_end_err,"
            " pe.match_rate, pe.n_matched, pe.n_extra, pe.n_clips, pe.created_at, lp.origin"
            " FROM param_eval pe LEFT JOIN learned_params lp ON lp.version=pe.version"
            " WHERE pe.applied=1 ORDER BY pe.version ASC, pe.created_at ASC").fetchall())
        rejected = db.rows_to_dicts(con.execute(
            "SELECT scope, mean_abs_start_err, mean_abs_end_err, match_rate, detail, created_at"
            " FROM param_eval WHERE applied=0 ORDER BY created_at DESC LIMIT 10").fetchall())
        cur = con.execute("SELECT MAX(version) v FROM learned_params").fetchone()
    # keep the latest applied eval per version (one point per version on the trend)
    by_version: dict[int, dict] = {}
    for e in evals:
        by_version[e["version"]] = e
    trend = sorted(by_version.values(), key=lambda e: (e["version"] or 0))
    for r in rejected:
        r["detail"] = db.loads(r["detail"], {})
    return {"current_version": cur["v"] if cur else None, "trend": trend, "rejected": rejected}


@router.get("/params")
def get_params():
    version, params = learned_params.get_latest()
    with db.connect() as con:
        history = db.rows_to_dicts(con.execute(
            "SELECT version, origin, created_at FROM learned_params"
            " ORDER BY version DESC LIMIT 20").fetchall())
    return {"version": version, "params": params, "defaults": learned_params.DEFAULTS,
            "history": history}

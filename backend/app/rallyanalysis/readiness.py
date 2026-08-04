"""Pass 2 readiness gating (patch): nothing else in this package checked that
Pass 2 actually finished for a video, or that its overlay data still covers a
rally's CURRENT start/end frame. rally_labels can change after Pass 2 last
ran (segmentation is under active development on another branch), and
run_pass2 clears the pass2 overlay source at the start of its run — so a
rally can silently be analyzed against a stale or half-written overlay if
nothing checks (a) the most recent jobs row for this video's Pass 2, and
(b) that overlay coverage still reaches both ends of the rally's current
frame range.

Read-only against jobs, overlays, rally_labels — same read pattern the rest
of rallyanalysis/ uses. Writes nothing."""
from __future__ import annotations

from .. import db
from ..cv.overlays import read_range

# A rally's overlay coverage must reach within this many frames of both
# start_frame and end_frame to count as "still covers this rally" — Pass 2
# doesn't necessarily emit an entry for every single frame, so requiring
# exact edge frames would false-negative on healthy data.
COVERAGE_TOLERANCE_FRAMES = 15  # ~0.5s at 30fps

REASON_NOT_DONE = "pass2_not_done"
REASON_RUNNING = "pass2_running"
REASON_STALE = "overlay_stale"
REASON_READY = "ready"


def default_overlay_source(video_id: str) -> str:
    """Pass 2 overlays carry stable identity slots; prefer them, fall back to
    Pass 1 spot-check overlays if Pass 2 hasn't run."""
    with db.connect() as con:
        row = con.execute(
            "SELECT 1 FROM overlays WHERE video_id=? AND source='pass2' LIMIT 1",
            (video_id,)).fetchone()
    return "pass2" if row else "pass1"


def _get_rallies(video_id: str) -> list[dict]:
    """Human labels win when present (ground truth), else Pass 2, else
    Pass 1 — same override order the rest of the app treats as canonical
    (see api/annotate.py get_rallies)."""
    with db.connect() as con:
        rows = db.rows_to_dicts(con.execute(
            "SELECT id, source, start_frame, end_frame FROM rally_labels WHERE video_id=?",
            (video_id,)).fetchall())
    by_source: dict[str, list[dict]] = {"human": [], "pass2": [], "pass1": []}
    for r in rows:
        if r["source"] in by_source:
            by_source[r["source"]].append(r)
    for src in ("human", "pass2", "pass1"):
        if by_source[src]:
            return sorted(by_source[src], key=lambda r: r["start_frame"])
    return []


def _latest_pass2_job_status(video_id: str) -> str | None:
    """Status of the MOST RECENT pass2 jobs row for this video, or None if no
    pass2 job has ever been submitted."""
    with db.connect() as con:
        row = con.execute(
            "SELECT status FROM jobs WHERE video_id=? AND kind='pass2'"
            " ORDER BY created_at DESC LIMIT 1",
            (video_id,)).fetchone()
    return row["status"] if row else None


def _pass2_ready_reason(video_id: str) -> str | None:
    """None if Pass 2 is done (ready so far, pending overlay-coverage check);
    otherwise the reason it isn't."""
    status = _latest_pass2_job_status(video_id)
    if status is None:
        return REASON_NOT_DONE
    if status == "running":
        return REASON_RUNNING
    if status != "done":
        return REASON_NOT_DONE
    return None


def covers_rally(entries: list[dict], start_frame: int, end_frame: int,
                  tolerance: int = COVERAGE_TOLERANCE_FRAMES) -> bool:
    """True if entries is non-empty and its frame coverage reaches within
    `tolerance` frames of both start_frame and end_frame."""
    if not entries:
        return False
    frames = [e["f"] for e in entries]
    lo, hi = min(frames), max(frames)
    return lo <= start_frame + tolerance and hi >= end_frame - tolerance


def get_analyzable_rallies(video_id: str, rally_source: str | None = None,
                            overlay_source: str | None = None) -> list[dict]:
    """Rallies (id, start_frame, end_frame, ...) that are safe to analyze
    right now: Pass 2 done (not running/missing/errored) AND overlay
    coverage still reaches both ends of the rally's current frame range.
    Rallies failing either check are excluded entirely, not partially
    included."""
    overlay_source = overlay_source or default_overlay_source(video_id)
    if rally_source is not None:
        with db.connect() as con:
            rows = db.rows_to_dicts(con.execute(
                "SELECT id, start_frame, end_frame FROM rally_labels"
                " WHERE video_id=? AND source=? ORDER BY start_frame",
                (video_id, rally_source)).fetchall())
    else:
        rows = _get_rallies(video_id)

    if _pass2_ready_reason(video_id) is not None:
        return []

    out = []
    for r in rows:
        entries = read_range(video_id, overlay_source, r["start_frame"], r["end_frame"])
        if covers_rally(entries, r["start_frame"], r["end_frame"]):
            out.append(r)
    return out


def readiness_report(video_id: str, overlay_source: str | None = None) -> dict:
    """{rally_id: {"ready": bool, "reason": "pass2_not_done"|"pass2_running"|
    "overlay_stale"|"ready"}} for every rally currently selected by
    _get_rallies (human > pass2 > pass1 override order)."""
    overlay_source = overlay_source or default_overlay_source(video_id)
    rallies = _get_rallies(video_id)
    pass2_reason = _pass2_ready_reason(video_id)

    report: dict[str, dict] = {}
    for r in rallies:
        if pass2_reason is not None:
            report[r["id"]] = {"ready": False, "reason": pass2_reason}
            continue
        entries = read_range(video_id, overlay_source, r["start_frame"], r["end_frame"])
        if covers_rally(entries, r["start_frame"], r["end_frame"]):
            report[r["id"]] = {"ready": True, "reason": REASON_READY}
        else:
            report[r["id"]] = {"ready": False, "reason": REASON_STALE}
    return report


if __name__ == "__main__":
    # ponytail self-check: coverage tolerance math, no DB.
    entries = [{"f": f} for f in range(100, 201)]
    assert covers_rally(entries, 100, 200) is True
    assert covers_rally(entries, 100, 250) is False   # far short of end_frame
    assert covers_rally(entries, 90, 200, tolerance=15) is True  # within tolerance
    assert covers_rally([], 100, 200) is False
    print("readiness self-check OK")

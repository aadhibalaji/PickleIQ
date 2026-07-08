"""Manual labeling, court calibration, ball spot-corrections, user notes.

Scope routing lives here: court-corner corrections default to the reusable
global CameraCalibration (fixed camera => one-time calibration inherited by
all future clips), with a clip_local escape hatch for anomalous clips.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db
from ..cv import court
from ..cv.court import CameraModel
from ..cv.video_io import iter_frames

router = APIRouter()


# ---------------------------------------------------------------- labels ---
class RallyLabelIn(BaseModel):
    start_frame: int
    end_frame: int


@router.get("/videos/{video_id}/rallies")
def get_rallies(video_id: str):
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM rally_labels WHERE video_id=? ORDER BY start_frame", (video_id,)
        ).fetchall()
    out: dict[str, list] = {"human": [], "pass1": [], "pass2": []}
    for r in rows:
        out[r["source"]].append({
            "id": r["id"], "start_frame": r["start_frame"],
            "end_frame": r["end_frame"], "meta": db.loads(r["meta"], {}),
        })
    return out


@router.put("/videos/{video_id}/labels/human")
def put_human_labels(video_id: str, labels: list[RallyLabelIn]):
    for lb in labels:
        if lb.end_frame <= lb.start_frame:
            raise HTTPException(400, f"end must be after start ({lb.start_frame})")
    with db.connect() as con:
        con.execute(
            "DELETE FROM rally_labels WHERE video_id=? AND source='human'", (video_id,)
        )
        for lb in labels:
            con.execute(
                "INSERT INTO rally_labels (id, video_id, source, start_frame, end_frame)"
                " VALUES (?,?,?,?,?)",
                (db.new_id(), video_id, "human", lb.start_frame, lb.end_frame),
            )

    # Close the learning loop: a background job compares these corrections to
    # the AI, validates a low-risk boundary offset against all labeled clips,
    # and applies it globally only if it improves accuracy (gate changes are
    # queued for approval instead). The JobRunner supersedes a prior pending
    # 'learn' for this clip, so rapid re-saves debounce for free.
    from ..jobs import runner

    def learn_job(handle):
        from ..learning import autolearn
        autolearn.run_learn(video_id)

    learn_job_id = runner.submit(video_id, "learn", learn_job)
    return {"saved": len(labels), "learn_job_id": learn_job_id}


# ----------------------------------------------------------- calibration ---
class CalibrationIn(BaseModel):
    corner_points: list[dict]           # [{name, x, y}] in full-res pixels
    scope: str = "global"               # 'global' (default) | 'clip_local'
    video_id: str | None = None         # required for clip_local


@router.get("/calibration/reference_points")
def reference_points():
    return [{"name": k, "court_xy": v} for k, v in court.REFERENCE_POINTS.items()]


@router.get("/calibration")
def get_calibration(video_id: str | None = None):
    from ..pipeline import effective_calibration
    if video_id:
        H, cam, cal_id = effective_calibration(video_id)
        if H is None:
            return {"calibrated": False}
        with db.connect() as con:
            cal = con.execute("SELECT * FROM calibrations WHERE id=?", (cal_id,)).fetchone()
        return {
            "calibrated": True, "id": cal_id, "scope": cal["scope"],
            "corner_points": db.loads(cal["corner_points"]),
            "homography": db.loads(cal["homography"]),
            "focal_px": cam.focal_px,
            "court": {"w": court.COURT_W, "l": court.COURT_L, "net_y": court.NET_Y,
                      "kitchen": court.KITCHEN},
        }
    with db.connect() as con:
        cal = con.execute(
            "SELECT * FROM calibrations WHERE scope='global' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if cal is None:
        return {"calibrated": False}
    return {"calibrated": True, "id": cal["id"], "scope": "global",
            "corner_points": db.loads(cal["corner_points"]),
            "homography": db.loads(cal["homography"]), "focal_px": cal["focal_px"]}


@router.post("/calibration")
def save_calibration(body: CalibrationIn):
    try:
        H = court.fit_homography(body.corner_points)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    focal = None
    if body.video_id:
        with db.connect() as con:
            v = con.execute("SELECT width, height FROM videos WHERE id=?",
                            (body.video_id,)).fetchone()
        if v:
            focal = CameraModel.from_homography(H, v["width"], v["height"]).focal_px

    cal_id = db.new_id()
    with db.connect() as con:
        con.execute(
            "INSERT INTO calibrations (id, scope, video_id, corner_points, homography,"
            " focal_px, created_at) VALUES (?,?,?,?,?,?,?)",
            (cal_id, body.scope, body.video_id if body.scope == "clip_local" else None,
             db.dumps(body.corner_points), db.dumps(H.tolist()), focal, db.now()),
        )
        if body.scope == "clip_local":
            if not body.video_id:
                raise HTTPException(400, "clip_local calibration needs video_id")
            con.execute("UPDATE videos SET calibration_id=? WHERE id=?",
                        (cal_id, body.video_id))
        elif body.video_id:
            # New global calibration also clears any stale clip-local override
            # on the clip it was calibrated from.
            con.execute("UPDATE videos SET calibration_id=NULL WHERE id=?",
                        (body.video_id,))
    return {"id": cal_id, "scope": body.scope, "focal_px": focal,
            "homography": H.tolist()}


@router.get("/videos/{video_id}/suggest_corners")
def suggest_corners(video_id: str, frame: int = 0):
    with db.connect() as con:
        row = con.execute("SELECT path FROM videos WHERE id=?", (video_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "video not found")
    for _, img in iter_frames(row["path"], start_frame=frame, end_frame=frame + 1):
        pts = court.suggest_corners(img)
        return {"points": pts, "found": bool(pts)}
    raise HTTPException(400, "could not read frame")


@router.get("/calibration/{cal_id}/project")
def project_court(cal_id: str):
    """Court template lines projected to image px — the frontend draws these
    as the court overlay."""
    with db.connect() as con:
        cal = con.execute("SELECT * FROM calibrations WHERE id=?", (cal_id,)).fetchone()
    if cal is None:
        raise HTTPException(404, "calibration not found")
    H = np.array(db.loads(cal["homography"]))
    W, L, NY, K = court.COURT_W, court.COURT_L, court.NET_Y, court.KITCHEN
    lines_court = [
        [(0, 0), (W, 0)], [(0, L), (W, L)],                     # baselines
        [(0, 0), (0, L)], [(W, 0), (W, L)],                     # sidelines
        [(0, NY), (W, NY)],                                     # net line
        [(0, NY - K), (W, NY - K)], [(0, NY + K), (W, NY + K)],  # kitchen
        [(W / 2, 0), (W / 2, NY - K)], [(W / 2, NY + K), (W / 2, L)],  # centerlines
    ]
    out = []
    for (a, b) in lines_court:
        px = court.court_to_img(H, np.array([a, b]))
        out.append([[round(float(px[0][0]), 1), round(float(px[0][1]), 1)],
                    [round(float(px[1][0]), 1), round(float(px[1][1]), 1)]])
    return {"lines": out}


# ------------------------------------------------------ ball corrections ---
# ball_corrections doubles as the GROUND-TRUTH mark store for ball events
# (event_type 'bounce' | 'hit' | 'position'; legacy rows with NULL event_type
# are plain position marks). Court projection and the homography's in/out
# suggestion are computed HERE at save time — one source of truth — and the
# human's final call is stored next to the suggestion, with a disagree flag
# when they differ (those flag calibration error near lines).

_EVENT_TYPES = ("bounce", "hit", "position")


def _project_point(video_id: str, x: float, y: float) -> dict | None:
    """Court-plane projection + in/out suggestion for an image point.
    Returns None when the clip has no calibration."""
    from ..pipeline import effective_calibration
    H, _cam, _ = effective_calibration(video_id)
    if H is None:
        return None
    cxy = court.img_to_court(H, np.array([[float(x), float(y)]]))[0]
    cx, cy = float(cxy[0]), float(cxy[1])
    inside = bool(court.in_court(np.array([[cx, cy]]))[0])
    if inside:  # distance to the NEAREST line (how marginal the call is)
        dist = min(cx, court.COURT_W - cx, cy, court.COURT_L - cy)
    else:
        dx = max(0.0, -cx, cx - court.COURT_W)
        dy = max(0.0, -cy, cy - court.COURT_L)
        dist = (dx * dx + dy * dy) ** 0.5
    return {"court_x": round(cx, 3), "court_y": round(cy, 3),
            "suggested": "in" if inside else "out",
            "line_dist_m": round(dist, 3)}


def _gt_fields(video_id: str, body) -> tuple:
    """(court_x, court_y, in_out, in_out_suggested, disagree) for a mark."""
    proj = _project_point(video_id, body.x, body.y)
    suggested = proj["suggested"] if proj else None
    in_out = body.in_out if body.event_type == "bounce" else None
    disagree = int(bool(in_out and suggested and in_out != suggested))
    return (proj["court_x"] if proj else None, proj["court_y"] if proj else None,
            in_out, suggested if body.event_type == "bounce" else None, disagree)


class BallCorrectionIn(BaseModel):
    frame: int
    x: float
    y: float
    event_type: str = "position"   # 'bounce' | 'hit' | 'position'
    in_out: str | None = None      # human call, bounces only ('in' | 'out')
    scope: str = "clip_local"  # spot-corrections are clip-local by design


@router.get("/videos/{video_id}/project_point")
def project_point(video_id: str, x: float, y: float):
    """Live in/out suggestion for the marking UI (before the mark is saved)."""
    proj = _project_point(video_id, x, y)
    if proj is None:
        return {"calibrated": False}
    return {"calibrated": True, **proj}


@router.get("/videos/{video_id}/ball_corrections")
def list_ball_corrections(video_id: str):
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM ball_corrections WHERE video_id=? ORDER BY frame", (video_id,)
        ).fetchall()
    out = db.rows_to_dicts(rows)
    for r in out:  # legacy rows predate event_type — they are position marks
        if not r.get("event_type"):
            r["event_type"] = "position"
    return out


@router.post("/videos/{video_id}/ball_corrections")
def add_ball_correction(video_id: str, body: BallCorrectionIn):
    if body.event_type not in _EVENT_TYPES:
        raise HTTPException(400, f"event_type must be one of {_EVENT_TYPES}")
    if body.event_type == "bounce" and body.in_out not in ("in", "out"):
        raise HTTPException(400, "a bounce mark needs in_out ('in' or 'out')")
    court_x, court_y, in_out, suggested, disagree = _gt_fields(video_id, body)
    cid = db.new_id()
    try:
        with db.connect() as con:
            con.execute(
                "INSERT INTO ball_corrections (id, video_id, frame, x, y, scope,"
                " created_at, event_type, court_x, court_y, in_out,"
                " in_out_suggested, disagree) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, video_id, body.frame, body.x, body.y, body.scope, db.now(),
                 body.event_type, court_x, court_y, in_out, suggested, disagree),
            )
    except Exception as exc:  # pre-migration schema: fail with instructions
        if "no column" in str(exc) or "has no column" in str(exc):
            raise HTTPException(
                400, "ball_corrections is missing the ground-truth columns — "
                     "apply the schema migration (see db.py migration notes) "
                     "before using the marking tool") from exc
        raise
    return {"id": cid, "event_type": body.event_type, "court_x": court_x,
            "court_y": court_y, "in_out": in_out, "in_out_suggested": suggested,
            "disagree": disagree}


class BallCorrectionUpdate(BaseModel):
    x: float | None = None
    y: float | None = None
    frame: int | None = None
    event_type: str | None = None
    in_out: str | None = None


@router.put("/ball_corrections/{correction_id}")
def update_ball_correction(correction_id: str, body: BallCorrectionUpdate):
    """Edit a mark in place; court projection / suggestion / disagree are
    recomputed from the (possibly moved) pixel position."""
    with db.connect() as con:
        row = con.execute("SELECT * FROM ball_corrections WHERE id=?",
                          (correction_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "mark not found")
    merged = BallCorrectionIn(
        frame=body.frame if body.frame is not None else row["frame"],
        x=body.x if body.x is not None else row["x"],
        y=body.y if body.y is not None else row["y"],
        event_type=body.event_type or (row["event_type"] if "event_type" in row.keys()
                                       and row["event_type"] else "position"),
        in_out=body.in_out if body.in_out is not None
        else (row["in_out"] if "in_out" in row.keys() else None),
    )
    if merged.event_type not in _EVENT_TYPES:
        raise HTTPException(400, f"event_type must be one of {_EVENT_TYPES}")
    court_x, court_y, in_out, suggested, disagree = _gt_fields(row["video_id"], merged)
    if merged.event_type == "bounce" and in_out not in ("in", "out"):
        in_out = suggested  # retyped to bounce without a call: default to suggestion
        disagree = 0
    with db.connect() as con:
        con.execute(
            "UPDATE ball_corrections SET frame=?, x=?, y=?, event_type=?,"
            " court_x=?, court_y=?, in_out=?, in_out_suggested=?, disagree=?"
            " WHERE id=?",
            (merged.frame, merged.x, merged.y, merged.event_type, court_x,
             court_y, in_out, suggested, disagree, correction_id),
        )
    return {"id": correction_id, "event_type": merged.event_type,
            "in_out": in_out, "in_out_suggested": suggested, "disagree": disagree}


@router.delete("/ball_corrections/{correction_id}")
def delete_ball_correction(correction_id: str):
    with db.connect() as con:
        con.execute("DELETE FROM ball_corrections WHERE id=?", (correction_id,))
    return {"deleted": True}


# ------------------------------------------ external ball track (integration) ---
# A dense per-frame ball track produced by an EXTERNAL tracker (e.g. an MTI +
# temporal-integration model). Pass 2 prefers it over its own classical detector
# (see cv/pass2.py). Format is full-res image pixels; diameter is optional but
# enables 3D height reconstruction (and therefore bounce/end detection).
class BallTrackIn(BaseModel):
    source: str = "external"
    track: list[dict]  # [{frame:int, x:float, y:float, diameter?:float, confidence?:float}]


@router.post("/videos/{video_id}/ball_track")
def put_ball_track(video_id: str, body: BallTrackIn):
    d: dict[str, list] = {}
    for p in body.track:
        try:
            d[str(int(p["frame"]))] = [
                float(p["x"]), float(p["y"]),
                float(p["diameter"]) if p.get("diameter") is not None else None,
                float(p.get("confidence", 0.9)),
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"bad track point {p}: {exc}") from exc
    with db.connect() as con:
        con.execute("DELETE FROM ball_tracks WHERE video_id=?", (video_id,))
        con.execute(
            "INSERT INTO ball_tracks (video_id, payload, source, n_frames, created_at)"
            " VALUES (?,?,?,?,?)", (video_id, db.dumps(d), body.source, len(d), db.now()))
    return {"stored": len(d), "source": body.source,
            "note": "reprocess Pass 2 to use it (Report tab → Regenerate video)"}


@router.get("/videos/{video_id}/ball_track")
def get_ball_track_info(video_id: str):
    with db.connect() as con:
        row = con.execute("SELECT source, n_frames, created_at FROM ball_tracks"
                          " WHERE video_id=?", (video_id,)).fetchone()
    return dict(row) if row else {"n_frames": 0, "source": None}


@router.delete("/videos/{video_id}/ball_track")
def delete_ball_track(video_id: str):
    with db.connect() as con:
        con.execute("DELETE FROM ball_tracks WHERE video_id=?", (video_id,))
    return {"deleted": True}


# ------------------------------------------------------------------ notes ---
class NoteIn(BaseModel):
    text: str
    source: str = "typed"               # 'typed' | 'voice'
    scope: str = "global"               # method-level feedback defaults global
    video_time: float | None = None


@router.get("/videos/{video_id}/notes")
def list_notes(video_id: str):
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM user_notes WHERE video_id=? ORDER BY created_at", (video_id,)
        ).fetchall()
    return db.rows_to_dicts(rows)


@router.post("/videos/{video_id}/notes")
def add_note(video_id: str, body: NoteIn):
    if body.scope not in ("global", "clip_local"):
        raise HTTPException(400, "scope must be global or clip_local")
    nid = db.new_id()
    with db.connect() as con:
        con.execute(
            "INSERT INTO user_notes (id, video_id, text, source, scope, video_time, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (nid, video_id, body.text, body.source, body.scope, body.video_time, db.now()),
        )
    return {"id": nid}


@router.delete("/notes/{note_id}")
def delete_note(note_id: str):
    with db.connect() as con:
        con.execute("DELETE FROM user_notes WHERE id=?", (note_id,))
    return {"deleted": True}

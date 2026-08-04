"""Loaders adapting the repo's real data formats to refinement inputs.

Formats (verified against the live schema / cv code):

* ``rally_labels`` (source='pass1'): uuid ``id``, ``start_frame``/``end_frame``
  in FRAME INDICES, ``meta`` JSON. fps lives in the ``videos`` table.
* ``overlays`` (source='pass2'): JSON chunks of per-frame entries::

      {"f": int, "players": [{"id", "num", "box"[px], "cxy"[court m]|null,
                              "team": "near"|"far", "kp"}],
       "pspeed": float,                       # mean player speed m/s
       "ball": [x, y, diam|null, conf],       # full-res image PIXELS
       "interp": bool,                        # true = arc-fill, not observed
       "world": [cx, cy, height|null],        # court metres (size->depth; noisy)
       "obs": bool,
       "evt": {"kind": "hit"|"bounce", "in", "side", "cx", "cy"}}

* Calibration: homography H (image px -> court metres), clip-local override
  else newest global, stored as JSON 3x3 in ``calibrations``.

Ball court positions are re-derived here by projecting the ``ball`` pixels
through H (ground-plane assumption) rather than trusting ``world`` x/y — the
size->depth reconstruction produces occasional wild values for the tiny
far-court ball, while the plane projection is exact whenever the ball is low
(bounces, net crossings — exactly where we read it). ``world``'s height
channel is kept as an optional feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import db
from ..cv import overlays
from ..cv.court import COURT_L, COURT_W, KITCHEN, NET_Y  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# Data containers (also constructed directly by the synthetic test generators)
# ---------------------------------------------------------------------------

@dataclass
class Pass1Window:
    """One Pass 1 candidate rally window (frames, inclusive)."""
    id: str
    start_frame: int
    end_frame: int
    meta: dict = field(default_factory=dict)


@dataclass
class TrackingData:
    """Per-frame tracking over one contiguous frame span [f0, f0+N).

    All positional arrays are in COURT METRES; missing = NaN. ``players``
    maps a stable player slot/track id to an [N,2] court-position array.
    """
    fps: float
    f0: int                                  # first frame index of the span
    present: np.ndarray                      # [N] bool: overlay entry exists
    ball_xy: np.ndarray                      # [N,2] court m (plane-projected)
    ball_conf: np.ndarray                    # [N]
    ball_interp: np.ndarray                  # [N] bool: arc-fill, not observed
    ball_height: np.ndarray                  # [N] m or NaN (from `world`)
    players: dict[int, np.ndarray]           # slot -> [N,2] court m
    pspeed: np.ndarray                       # [N] mean player speed m/s
    bounce: np.ndarray                       # [N] bool (evt kind='bounce')
    hit: np.ndarray                          # [N] bool (evt kind='hit')
    evt_xy: np.ndarray                       # [N,2] court m of the evt or NaN
    video_id: str = ""
    # Raw IMAGE-space tracks, kept alongside the court projection. The ball's
    # court position is corrupted whenever it is airborne (the plane projection
    # assumes height=0 and the size->depth estimate is too noisy for the far
    # ball), so proximity checks at the serve — where the ball is high — must be
    # done in pixels, where the track is accurate. None on synthetic inputs that
    # don't set them (callers fall back to court space).
    ball_px: np.ndarray | None = None        # [N,2] full-res image px or NaN
    player_box: dict = field(default_factory=dict)  # slot -> [N,4] px (x0,y0,x1,y1)
    ball_flights: list = field(default_factory=list)  # ballistic.Flight per fitted arc
                                             # (lets the serve gate extrapolate the
                                             # serve arc back to the untracked strike)
    H: np.ndarray | None = None              # homography used to build this span (image
                                             # px -> court m); lets features re-project
                                             # auxiliary image-space points (e.g. pose)
    player_kp: dict = field(default_factory=dict)  # slot -> [N,17,2] full-res image px
                                             # COCO pose keypoints, NaN where undetected
                                             # (see cv/players.py PoseTracker)

    @property
    def n(self) -> int:
        return len(self.present)

    def frame_index(self, frame: int) -> int:
        return int(frame) - self.f0


# ---------------------------------------------------------------------------
# DB loaders
# ---------------------------------------------------------------------------

def load_video(video_id: str) -> dict:
    """fps / frame_count / dimensions / filename for one clip."""
    with db.connect() as con:
        row = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown video {video_id!r}")
    return dict(row)


def load_pass1_windows(video_id: str, source: str = "pass1") -> list[Pass1Window]:
    """Rally windows this video's refine/keep-drop pass should score.

    Excludes rows meta-tagged user_drawn=True (see
    pipeline._build_do_pass2_selections): those are manual coaching-selection
    ranges, not Pass 1's own candidates, so there is no Pass 1 confidence
    signal to grade — keep/drop is a Pass-1-candidate concept and doesn't
    apply to a range the user already hand-picked."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT id, start_frame, end_frame, meta FROM rally_labels"
            " WHERE video_id=? AND source=? ORDER BY start_frame",
            (video_id, source)).fetchall()
    windows = [Pass1Window(r["id"], r["start_frame"], r["end_frame"],
                           db.loads(r["meta"], {})) for r in rows]
    return [w for w in windows if not w.meta.get("user_drawn")]


def load_homography(video_id: str) -> np.ndarray | None:
    """Effective calibration: clip-local override else newest global.
    (Mirrors pipeline.effective_calibration without importing the CV stack.)"""
    with db.connect() as con:
        vrow = con.execute("SELECT calibration_id FROM videos WHERE id=?",
                           (video_id,)).fetchone()
        cal = None
        if vrow and vrow["calibration_id"]:
            cal = con.execute("SELECT homography FROM calibrations WHERE id=?",
                              (vrow["calibration_id"],)).fetchone()
        if cal is None:
            cal = con.execute(
                "SELECT homography FROM calibrations WHERE scope='global'"
                " ORDER BY created_at DESC LIMIT 1").fetchone()
    if cal is None:
        return None
    return np.asarray(db.loads(cal["homography"]), dtype=np.float64)


def project_px(H: np.ndarray, xy_px: np.ndarray) -> np.ndarray:
    """Image pixels -> court metres through the plane homography (pure numpy;
    equivalent to cv2.perspectiveTransform for finite points)."""
    pts = np.atleast_2d(np.asarray(xy_px, dtype=np.float64))
    hom = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def load_tracking(video_id: str, f0: int, f1: int,
                  H: np.ndarray | None = None,
                  fps: float | None = None,
                  reconstruct_3d: bool = False,
                  max_rms_px: float = 12.0) -> TrackingData:
    """Tracking arrays for frames [f0, f1] inclusive, from the pass2 overlay.

    Frames with no overlay entry (outside any processed window) stay NaN with
    ``present=False`` — features treat them as "nothing tracked", never as
    evidence of dead play by itself.

    ``reconstruct_3d`` replaces the ground-plane ball projection with a ballistic
    3D fit (correct court x/y + height for airborne balls); see
    ``refine/ballistic.py``.
    """
    vid_row = None
    if H is None or (reconstruct_3d and fps is None):
        vid_row = load_video(video_id)
    if H is None:
        H = load_homography(video_id)
    if fps is None:
        fps = float((vid_row or load_video(video_id))["fps"])
    n = f1 - f0 + 1
    td = TrackingData(
        fps=fps, f0=f0,
        present=np.zeros(n, bool),
        ball_xy=np.full((n, 2), np.nan),
        ball_conf=np.zeros(n),
        ball_interp=np.zeros(n, bool),
        ball_height=np.full(n, np.nan),
        players={},
        pspeed=np.zeros(n),
        bounce=np.zeros(n, bool),
        hit=np.zeros(n, bool),
        evt_xy=np.full((n, 2), np.nan),
        video_id=video_id,
        ball_px=np.full((n, 2), np.nan),
        player_box={},
        H=H,
    )
    for e in overlays.read_range(video_id, "pass2", f0, f1):
        i = e["f"] - f0
        td.present[i] = True
        td.pspeed[i] = float(e.get("pspeed") or 0.0)
        ball = e.get("ball")
        if ball is not None:
            td.ball_px[i] = ball[:2]
            if H is not None:
                td.ball_xy[i] = project_px(H, [ball[:2]])[0]
                td.ball_conf[i] = float(ball[3] if ball[3] is not None else 0.0)
                td.ball_interp[i] = bool(e.get("interp", False))
        world = e.get("world")
        if world is not None and world[2] is not None:
            td.ball_height[i] = float(world[2])
        for p in e.get("players") or []:
            slot = p.get("id")
            if slot is None:
                continue
            if p.get("box") is not None:
                bx = td.player_box.get(slot)
                if bx is None:
                    bx = td.player_box[slot] = np.full((n, 4), np.nan)
                bx[i] = p["box"]
            kp = p.get("kp")
            if kp:
                kpa = td.player_kp.get(slot)
                if kpa is None:
                    kpa = td.player_kp[slot] = np.full((n, 17, 2), np.nan)
                m = min(len(kp), 17)
                kpa[i, :m] = kp[:m]
            if p.get("cxy") is None:
                continue
            arr = td.players.get(slot)
            if arr is None:
                arr = td.players[slot] = np.full((n, 2), np.nan)
            arr[i] = p["cxy"]
        evt = e.get("evt")
        if evt is not None:
            (td.bounce if evt.get("kind") == "bounce" else td.hit)[i] = True
            td.evt_xy[i] = (evt.get("cx", np.nan), evt.get("cy", np.nan))

    if reconstruct_3d and H is not None:
        vid_row = vid_row or load_video(video_id)
        if vid_row.get("width") and vid_row.get("height"):
            from ..cv.court import CameraModel
            from . import ballistic
            cam = CameraModel.from_homography(H, vid_row["width"], vid_row["height"])
            ballistic.reconstruct(td, cam, max_rms_px=max_rms_px)
    return td


def clips_with_human_labels() -> list[dict]:
    """Videos that have human GT rallies AND pass1 windows — the trainable set."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT DISTINCT v.id, v.filename, v.fps FROM videos v"
            " JOIN rally_labels h ON h.video_id=v.id AND h.source='human'"
            " JOIN rally_labels p ON p.video_id=v.id AND p.source='pass1'"
            " ORDER BY v.uploaded_at").fetchall()
    return [dict(r) for r in rows]

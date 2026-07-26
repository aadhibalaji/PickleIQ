"""Per-rally ball trajectory samples for the 3D trajectory view (replaces the
old 2D shot-map's always-null ``height_m``).

Reuses ``refine.ballistic``'s segment-fitting primitives (``_segments``,
``fit_flight``) directly rather than ``refine.ballistic.reconstruct()``,
because ``reconstruct()`` overwrites ``TrackingData`` in place and discards
the per-arc pixel RMS residual it computes internally -- exactly the signal
this module needs for confidence, and an existing one, not a new metric.
Every acceptance rule below (min segment length, RMS threshold, physical-
plausibility bounds) matches ``reconstruct()``'s so a point accepted there is
accepted here too, same thresholds, same constants (imported, not
reinvented).

Confidence formula
------------------
``confidence = rms_component * depth_component``, both in [0, 1]:

  ``rms_component = max_rms_px / (max_rms_px + rms)``
      ``rms`` is the fitted arc's per-point pixel reprojection RMS -- the
      same value ``refine.ballistic.reconstruct`` compares against
      ``max_rms_px`` to accept/reject a segment (refine/ballistic.py
      ``fit_flight`` return value). rms=0 -> 1.0; rms==max_rms_px -> 0.5;
      higher rms trends toward 0.

  ``depth_component = clamp(y / COURT_L, DEPTH_FLOOR, 1.0)``
      refine/ballistic.py's module docstring: "far-court ball depth is
      noisy" for this axial camera (the far ball is few pixels wide, so a
      1px diameter error is metres of depth -- see cv/bounce.py's
      ``height_sigma`` discussion of the same effect). Court y=0 is the far
      baseline, y=COURT_L the near baseline (cv/court.py convention), so
      points near y=0 get discounted toward the floor instead of zeroed --
      the RMS gate already screens out the worst fits, this only fades the
      remainder.

Points from segments ``refine.ballistic`` would reject (fit failure, RMS
above ``max_rms_px``, or outside its physical-plausibility bounds) are
omitted entirely -- never fabricated with a ground-plane fallback height.
"""
from __future__ import annotations

import numpy as np

from .. import db
from ..cv.court import CameraModel
from ..refine import ballistic
from ..refine.io_loaders import TrackingData, load_homography, load_tracking, load_video
from .readiness import _default_overlay_source, get_analyzable_rallies

DEPTH_FLOOR = 0.3                # confidence floor for the far baseline
ATTRIBUTION_WINDOW_FRAMES = 45   # ~1.5s at 30fps: how far back a contact can
                                  # be and still "own" a flight arc that starts
                                  # right after it


def _rms_confidence(rms: float, max_rms_px: float) -> float:
    return float(max_rms_px / (max_rms_px + max(rms, 0.0)))


def _depth_confidence(y: float) -> float:
    return float(min(1.0, max(DEPTH_FLOOR, y / ballistic.COURT_L)))


def point_confidence(rms: float, y: float, max_rms_px: float = 12.0) -> float:
    """See module docstring for the formula. Multiplicative so either a bad
    fit OR far-court depth can drag confidence down without one masking the
    other."""
    return _rms_confidence(rms, max_rms_px) * _depth_confidence(y)


def _attribute_contact(frame: float, contacts: list[dict],
                       window: int = ATTRIBUTION_WINDOW_FRAMES) -> dict | None:
    """Nearest contact event at or before `frame` (a flight arc starts at a
    hit), within `window` frames. None when nothing is close enough --
    ambiguous attribution is left null, never guessed."""
    best, best_gap = None, None
    for c in contacts:
        gap = frame - c["frame_idx"]
        if 0 <= gap <= window and (best_gap is None or gap < best_gap):
            best, best_gap = c, gap
    return best


def _attribute_player(frame: float, contacts: list[dict],
                      window: int = ATTRIBUTION_WINDOW_FRAMES) -> int | None:
    c = _attribute_contact(frame, contacts, window)
    return c["player_id"] if c else None


def rally_trajectory_points(td: TrackingData, cam: CameraModel,
                            contacts: list[dict] | None = None,
                            max_rms_px: float = 12.0, min_len: int = 5,
                            attribution_window_frames: int = ATTRIBUTION_WINDOW_FRAMES
                            ) -> list[dict]:
    """Ball trajectory samples for one rally's TrackingData: mirrors
    ``refine.ballistic.reconstruct()``'s segment acceptance exactly (same
    helpers, same thresholds) but returns points carrying confidence +
    player attribution instead of mutating td in place.

    Each point: {frame, x, y, height_m, confidence, player_id, is_serve}.
    player_id is None when no contact event attributes the arc (ambiguous --
    left null, never guessed). is_serve mirrors the attributed contact's own
    flag (False, not None, when unattributed -- there's no serve to show)."""
    if getattr(td, "ball_px", None) is None:
        return []
    contacts = contacts or []
    finite = np.isfinite(td.ball_px[:, 0])
    out: list[dict] = []
    for idx in ballistic._segments(finite, td.bounce, min_len):
        frames = (td.f0 + idx).astype(float)
        gxy = td.ball_xy[idx[0]]
        if not np.isfinite(gxy).all():
            gxy = np.array([3.0, 6.7])          # court centre fallback (matches reconstruct())
        res = ballistic.fit_flight(frames, td.ball_px[idx], cam, td.fps, gxy)
        if res is None:
            continue
        xy, h, rms, _P0, _V0 = res
        plausible = (
            np.all(np.isfinite(h)) and np.all(np.isfinite(xy))
            and np.all((h >= ballistic.MIN_HEIGHT_M) & (h <= ballistic.MAX_HEIGHT_M))
            and np.all((xy[:, 0] >= -ballistic.COURT_MARGIN_M)
                       & (xy[:, 0] <= ballistic.COURT_W + ballistic.COURT_MARGIN_M))
            and np.all((xy[:, 1] >= -ballistic.COURT_MARGIN_M)
                       & (xy[:, 1] <= ballistic.COURT_L + ballistic.COURT_MARGIN_M))
        )
        if not plausible or rms > max_rms_px:
            continue
        contact = _attribute_contact(frames[0], contacts, attribution_window_frames)
        player_id = contact["player_id"] if contact else None
        is_serve = bool(contact["is_serve"]) if contact else False
        for f, (x, y), height in zip(frames, xy, h):
            out.append({
                "frame": int(f),
                "x": float(x),
                "y": float(y),
                "height_m": float(height),
                "confidence": round(point_confidence(rms, float(y), max_rms_px), 4),
                "player_id": player_id,
                "is_serve": is_serve,
            })
    return out


# ---------------------------------------------------------------------------
# DB orchestration (reads rally_labels/overlays/contact_events/calibrations
# only, matching the read-only pattern used across rallyanalysis/)
# ---------------------------------------------------------------------------

def get_video_trajectories(video_id: str, rally_source: str | None = None,
                           max_rms_px: float = 12.0) -> dict:
    """Trajectory samples for every rally readiness.get_analyzable_rallies()
    clears on this video. Computed on request (nothing persisted) -- cheap
    enough for a debug view, same "computed fresh" pattern report_stats.py
    uses. Empty rallies list if there's no calibration or no frame size to
    build a CameraModel from."""
    H = load_homography(video_id)
    if H is None:
        return {"video_id": video_id, "rallies": []}
    vrow = load_video(video_id)
    if not vrow.get("width") or not vrow.get("height"):
        return {"video_id": video_id, "rallies": []}
    cam = CameraModel.from_homography(H, vrow["width"], vrow["height"])
    fps = float(vrow["fps"])
    overlay_source = rally_source or _default_overlay_source(video_id)
    rallies = get_analyzable_rallies(video_id, rally_source=rally_source,
                                     overlay_source=overlay_source)

    with db.connect() as con:
        contact_rows = db.rows_to_dicts(con.execute(
            "SELECT rally_id, frame_idx, player_id, is_serve FROM contact_events WHERE video_id=?",
            (video_id,)).fetchall())
    contacts_by_rally: dict[str, list[dict]] = {}
    for c in contact_rows:
        contacts_by_rally.setdefault(c["rally_id"], []).append(c)

    out = []
    for r in rallies:
        td = load_tracking(video_id, r["start_frame"], r["end_frame"], H=H, fps=fps)
        points = rally_trajectory_points(td, cam, contacts_by_rally.get(r["id"], []),
                                         max_rms_px=max_rms_px)
        out.append({"rally_id": r["id"], "points": points})
    return {"video_id": video_id, "rallies": out}

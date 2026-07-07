"""Player detection / tracking / team + number assignment.

YOLO11-pose (ultralytics, weights auto-download on first run) + the built-in
ByteTrack tracker for persistent IDs. Team = side of the net from the
homography. Official player numbers are assigned at serve time, when players
are stationary and well separated: near side left->right = 1,2; far side
left->right = 3,4 (singles: 1 near, 2 far).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from . import court
from .device import ultralytics_device

log = logging.getLogger("cv.players")

_model_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO

            _model = YOLO("yolo11n-pose.pt")  # auto-downloads official weights
        return _model


@dataclass
class PlayerObs:
    track_id: int
    box: tuple[float, float, float, float]  # x1,y1,x2,y2 pixels
    foot_xy: tuple[float, float]            # pixel point on the ground
    court_xy: tuple[float, float] | None    # metres via homography
    keypoints: list[list[float]] = field(default_factory=list)
    team: str | None = None                 # 'near' | 'far'
    number: int | None = None


class PoseTracker:
    """Stateful per-clip tracker. Call update() with frames in order."""

    def __init__(self, H: np.ndarray | None):
        self.H = H
        self._device = ultralytics_device()
        self._device_ok = True
        self.numbers: dict[int, int] = {}  # track_id -> official number

    def update(self, frame: np.ndarray) -> list[PlayerObs]:
        model = _get_model()
        kwargs = dict(persist=True, tracker="bytetrack.yaml", classes=[0],
                      conf=0.3, verbose=False)
        try:
            res = model.track(frame, device=self._device if self._device_ok else "cpu", **kwargs)
        except Exception:
            if self._device_ok and self._device not in ("cpu",):
                log.warning("tracker failed on %s, falling back to CPU", self._device)
                self._device_ok = False
                res = model.track(frame, device="cpu", **kwargs)
            else:
                raise
        out: list[PlayerObs] = []
        r = res[0]
        if r.boxes is None or r.boxes.id is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i].tolist()
            # Ground contact: mean of ankle keypoints when confident, else
            # bottom-centre of the box.
            foot = ((x1 + x2) / 2, y2)
            if kpts is not None and len(kpts[i]) >= 17:
                ankles = kpts[i][15:17]
                if np.all(ankles > 0):
                    foot = (float(ankles[:, 0].mean()), float(ankles[:, 1].mean()))
            court_xy = None
            team = None
            if self.H is not None:
                cxy = court.img_to_court(self.H, np.array([foot]))[0]
                # Players stand around the court, not only inside it.
                if -3 < cxy[0] < court.COURT_W + 3 and -6 < cxy[1] < court.COURT_L + 6:
                    court_xy = (float(cxy[0]), float(cxy[1]))
                    team = court.side_of_net(cxy[1])
            out.append(PlayerObs(
                track_id=int(tid), box=(x1, y1, x2, y2), foot_xy=foot,
                court_xy=court_xy, team=team,
                keypoints=kpts[i].tolist() if kpts is not None else [],
                number=self.numbers.get(int(tid)),
            ))
        return out

    def assign_numbers_at_serve(self, obs: list[PlayerObs]) -> None:
        """Call once per rally at the serve frame. Overwrites the mapping so
        IDs that got swapped mid-rally are re-anchored every serve."""
        placed = [o for o in obs if o.court_xy is not None]
        near = sorted([o for o in placed if o.team == "near"], key=lambda o: o.court_xy[0])
        far = sorted([o for o in placed if o.team == "far"], key=lambda o: o.court_xy[0])
        n = 1
        for o in near + far:
            self.numbers[o.track_id] = n
            o.number = n
            n += 1


def is_serve_formation(obs: list[PlayerObs]) -> bool:
    """Sparse spot-check used by Pass 1 and the rally-start logic.
    Serve-receive: everyone on their own half, at least one player per side
    near their baseline, and (doubles) at most one far-forward player per
    side — 'one up, rest back'."""
    placed = [o for o in obs if o.court_xy is not None]
    near = [o for o in placed if o.team == "near"]
    far = [o for o in placed if o.team == "far"]
    if not near or not far or len(placed) < 2:
        return False

    def side_ok(players, is_near: bool) -> bool:
        depths = []  # distance behind own kitchen line (positive = deeper)
        for o in players:
            y = o.court_xy[1]
            depths.append((y - (court.NET_Y + court.KITCHEN)) if is_near
                          else ((court.NET_Y - court.KITCHEN) - y))
        deep = sum(1 for d in depths if d > 1.0)
        up = sum(1 for d in depths if d <= 1.0)
        # Someone must be back near the baseline; at most one may be up.
        return deep >= 1 and up <= 1 and max(depths) > 3.0

    return side_ok(near, True) and side_ok(far, False)

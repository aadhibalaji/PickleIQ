"""Player detection / tracking / team + number assignment.

YOLO11-pose (ultralytics, weights auto-download on first run) + the built-in
ByteTrack tracker for persistent IDs. Team = side of the net from the
homography. Official player numbers are assigned at serve time, when players
are stationary and well separated: near side left->right = 1,2; far side
left->right = 3,4 (singles: 1 near, 2 far).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from . import court
from ..config import POSE_MODEL
from .device import ultralytics_device

log = logging.getLogger("cv.players")


def _build_model(name: str):
    """Construct a FRESH YOLO-pose model. Each PoseTracker gets its own instance
    (not a shared singleton): ByteTrack state lives on the model's predictor, so
    two trackers sharing one model object would corrupt each other's track state
    — the failure mode that once wedged tracking mid-clip and made detection stop
    returning players. Per-instance models make that class of bug impossible even
    when passes/windows/jobs overlap. Ultralytics caches the weights file on
    disk, so re-loads are cheap after the first (auto-)download."""
    from ultralytics import YOLO

    return YOLO(name)


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
    """Stateful per-sequence tracker. Call update() with frames in order.

    `track_sequence=True` (Pass 2): frames are consecutive, so ByteTrack runs
    and gives stable IDs for player-number assignment. The tracker is RESET on
    the first frame of this instance (persist=False) so the module-global model
    never leaks ByteTrack state across windows/passes — that leak wedged
    tracking mid-clip and made detection stop returning players.

    `track_sequence=False` (Pass 1 spot checks, ~2 s apart): ByteTrack cannot
    associate across such gaps, so we reset every call and only use detections
    + court positions (IDs are irrelevant to the serve-formation check)."""

    def __init__(self, H: np.ndarray | None, track_sequence: bool = True,
                 conf: float = 0.25, imgsz: int = 640,
                 model_name: str | None = None):
        self.H = H
        self._device = ultralytics_device()
        self._device_ok = True
        self.numbers: dict[int, int] = {}  # track_id -> official number
        self._track_sequence = track_sequence
        self._conf = float(conf)
        self._imgsz = int(imgsz)
        self._started = False
        self._syn = -1  # synthetic ids for detections ByteTrack hasn't confirmed
        # Own model instance -> own ByteTrack state (see _build_model).
        self._model = _build_model(model_name or POSE_MODEL)

    def update(self, frame: np.ndarray) -> list[PlayerObs]:
        model = self._model
        # persist=False on the first frame re-inits ByteTrack for THIS sequence;
        # Pass 1 keeps resetting (sparse), Pass 2 then persists for stable IDs.
        persist = self._track_sequence and self._started
        self._started = True
        kwargs = dict(persist=persist, tracker="bytetrack.yaml", classes=[0],
                      conf=self._conf, imgsz=self._imgsz, verbose=False)
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
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            # ByteTrack returns id=None for tentative (unconfirmed) tracks and
            # whenever it loses lock. The old code returned [] here, silently
            # discarding perfectly visible players — that is the root cause of
            # 'far side players never detected' (sparse Pass 1 spot checks never
            # confirm a track) and 'detection stopped at ~frame 2900' (the shared
            # tracker wedged and stopped confirming for the rest of the clip).
            # Keep the detections; give them synthetic negative ids.
            ids = np.array([self._syn - i for i in range(len(boxes))], dtype=int)
            self._syn -= len(boxes)
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


def is_serve_receive_formation(obs: list[PlayerObs], behind_m: float = 0.75) -> bool:
    """Stricter serve-start gate: a genuine serve-receive shape must exist —
    a SERVER standing behind one baseline and the DIAGONAL (cross-court)
    RECEIVER standing behind the opposite baseline. In pickleball the serve is
    always struck from behind the baseline into the diagonally opposite service
    court, so this pair is the defining signature of a point about to start and
    rejects mid-point / between-point stances that the looser 'one up, rest
    back' check lets through.

    `behind_m` = metres inside a baseline still counted as 'behind' it (players
    toe the line, and the far foot point is noisy). Returns True as soon as one
    valid server+diagonal-receiver pair is found."""
    placed = [o for o in obs if o.court_xy is not None]
    near_deep = [o for o in placed if o.team == "near" and o.court_xy[1] > court.COURT_L - behind_m]
    far_deep = [o for o in placed if o.team == "far" and o.court_xy[1] < behind_m]
    if not near_deep or not far_deep:
        return False
    mid = court.COURT_W / 2.0
    # Server and receiver must sit on OPPOSITE halves of the centre line — that
    # is what makes it a cross-court (diagonal) serve rather than straight-ahead.
    for s in near_deep:
        for r in far_deep:
            if (s.court_xy[0] - mid) * (r.court_xy[0] - mid) < 0:
                return True
    return False

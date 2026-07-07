"""Player detection / tracking / team + number assignment.

YOLO11-pose (ultralytics, weights auto-download on first run) + the built-in
ByteTrack tracker for persistent IDs. Team = side of the net from the
homography. Official player numbers are assigned at serve time, when players
are stationary and well separated: near side left->right = 1,2; far side
left->right = 3,4 (singles: 1 near, 2 far).
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import court
from .exclusions import in_excluded
from .identity import PlayerIdentityTracker
from ..config import DATA_DIR, POSE_MODEL
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


# imgsz -> exported OpenVINO model dir (or None if OV/GPU is unavailable). The
# export is size-specific (static input shape) and cached to disk, so it runs at
# most once per size ever, and the in-process dict avoids re-checking each window.
_OV_CACHE: dict[int, str | None] = {}


def _openvino_pose_dir(imgsz: int) -> str | None:
    """Export yolo11n-pose to OpenVINO FP16 at `imgsz` (once, cached under
    data/models/pose) and return the model dir — this is what lets pose run on the
    Intel Arc GPU, which ultralytics' torch path can't drive (device='xpu' throws
    inside .track(), so it silently falls back to CPU at ~6x the latency). Returns
    None when OpenVINO isn't installed or the export fails, so the caller keeps the
    torch model."""
    if imgsz in _OV_CACHE:
        return _OV_CACHE[imgsz]
    result: str | None = None
    try:
        import openvino  # noqa: F401  (presence check)
        from ultralytics import YOLO

        # Leaf dir MUST keep the '_openvino_model' suffix — that's how ultralytics
        # recognises the format; the image size is encoded in the parent instead.
        dst = DATA_DIR / "models" / "pose" / str(imgsz) / f"{Path(POSE_MODEL).stem}_openvino_model"
        if not (dst / "metadata.yaml").exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = Path(YOLO(POSE_MODEL).export(format="openvino", imgsz=imgsz, half=True))
            if src.resolve() != dst.resolve():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
        result = str(dst)
        log.info("pose: using OpenVINO model %s on the Intel Arc GPU", dst.name)
    except Exception as exc:  # noqa: BLE001
        log.warning("OpenVINO pose unavailable (%s) — falling back to torch/CPU", exc)
    _OV_CACHE[imgsz] = result
    return result


@dataclass
class PlayerObs:
    track_id: int
    box: tuple[float, float, float, float]  # x1,y1,x2,y2 pixels
    foot_xy: tuple[float, float]            # pixel point on the ground
    court_xy: tuple[float, float] | None    # metres via homography
    keypoints: list[list[float]] = field(default_factory=list)
    team: str | None = None                 # 'near' | 'far'
    number: int | None = None
    slot: int | None = None                 # persistent match-stable identity id


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
        self._device_ok = True
        # Official numbers now ride on match-stable identity slots (see identity.py),
        # not on the fragile ByteTrack track_id — that is what stopped the numbers
        # switching whenever a track_id churned on a crossing/occlusion/dropout.
        self.identity = PlayerIdentityTracker()
        self._track_sequence = track_sequence
        self._conf = float(conf)
        self._imgsz = int(imgsz)
        self._started = False
        self._frame_no = 0  # monotonic frame counter for the identity tracker
        self._syn = -1  # synthetic ids for detections ByteTrack hasn't confirmed
        # Prefer the OpenVINO GPU model (~6x faster on the Arc iGPU); fall back to
        # the torch weights if OpenVINO/GPU isn't available. Either way this is a
        # FRESH per-tracker instance -> own ByteTrack state (see _build_model).
        # ByteTrack itself runs in python, so .track() works on both backends.
        ov_dir = _openvino_pose_dir(self._imgsz) if model_name is None else None
        if ov_dir:
            from ultralytics import YOLO
            self._model = YOLO(ov_dir)
            self._device = "intel:gpu"
        else:
            self._model = _build_model(model_name or POSE_MODEL)
            self._device = ultralytics_device()

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
        h_px, w_px = frame.shape[:2]
        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i].tolist()
            # Drop detections in the fixed camera's exclusion zones (neighbouring
            # courts / spectators / fence-line people in the background corners).
            # Box centre is the test point — stable, and always well clear of the
            # zones for real on-court players (see cv/exclusions.py).
            if in_excluded((x1 + x2) / 2, (y1 + y2) / 2, w_px, h_px):
                continue
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
            ))
        # Match-stable identity: assign each detection to a persistent per-team
        # slot (appearance + court position) and stamp its slot + official number.
        # Only for consecutive-frame Pass 2 sequences; Pass 1 spot checks are
        # sparse and never use numbers, so we skip the work there.
        if self._track_sequence:
            self.identity.assign(frame, out, self._frame_no)
            self._frame_no += 1
        return out

    def new_sequence(self) -> None:
        """Start a fresh tracking sequence for the next Pass 2 window WITHOUT
        rebuilding the model. The next update() re-inits ByteTrack via
        persist=False (see update), exactly as a brand-new instance would, so
        IDs/numbers don't leak between windows — but the OpenVINO-compiled model
        is kept, avoiding a ~5 s GPU recompile per window. The isolation that the
        per-instance-model rule protects is still intact: each run owns its own
        PoseTracker; nothing is shared across passes/jobs (that sharing was the
        old mid-clip wedging bug), only across one run's sequential windows."""
        self._started = False
        self._syn = -1
        # Identity slots PERSIST across windows so a player keeps the same number
        # for the whole match; we only forget last positions (they walk around
        # between rallies) so the window re-acquires on appearance. See identity.py.
        self.identity.new_window()

    def assign_numbers_at_serve(self, obs: list[PlayerObs]) -> None:
        """Call once per rally at the serve frame. Re-anchors the official
        left->right numbering onto the persistent identity slots, so a number
        that drifted mid-rally is corrected and then rides with the right body."""
        self.identity.assign_numbers_at_serve(obs)


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

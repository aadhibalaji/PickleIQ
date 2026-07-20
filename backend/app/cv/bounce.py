"""Height reconstruction + bounce-vs-hit classification (the axial-view hard
part).

This camera looks down the length of the court, so a bounce has no clean 2D
"V": height and depth both project onto the image's vertical axis. We
therefore reconstruct the ball's height above the court explicitly:

    apparent size -> camera-frame depth (pickleball diameter is 74 mm)
    homography    -> court-plane geometry / camera pose
    combine       -> world (court_x, court_y, height)

A bounce is a local minimum of that height near zero, away from any player;
a trajectory reversal at height or at a player is a hit/volley (still live).

TODO(fine-tune): the physics heuristic below is the default. A learned
temporal model (BounceGRU at the bottom of this file) is preferred for the
axial-view bounce signature; it trains on accumulated FeatureTraces + human
labels via backend/scripts/train_bounce.py once you have labeled clips.
Apparent-size depth is noisy for the far-baseline ball (few pixels wide) —
that is exactly the region fine-tuning must fix. 60 fps footage materially
improves localization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import court
from .court import CameraModel
from .kalman import TrackPoint


@dataclass
class BallWorld:
    frame: int
    court_x: float
    court_y: float
    height: float
    observed: bool


@dataclass
class BallEvent:
    frame: int
    kind: str                      # 'bounce' | 'hit'
    court_x: float
    court_y: float
    height: float
    in_court: bool
    side: str                      # 'near' | 'far'
    near_player: bool
    observed: bool = True          # real detection at this frame (not bridged)
    height_known: bool = True      # height was reconstructed (not NaN)
    detail: dict = field(default_factory=dict)


def _median_filter(v: np.ndarray, k: int) -> np.ndarray:
    if len(v) < 3:
        return v
    k = min(k if k % 2 else k + 1, len(v) - (1 - len(v) % 2))
    pad = k // 2
    padded = np.pad(v, pad, mode="edge")
    return np.array([np.median(padded[i:i + k]) for i in range(len(v))])


def reconstruct_world(track: list[TrackPoint], cam: CameraModel,
                      height_sigma_max: float | None = None) -> list[BallWorld]:
    """Per-frame world position/height. Diameter is the noisy input, so it is
    median-filtered across observed points and interpolated over bridged
    gaps before the size->depth model is applied.

    `height_sigma_max` (metres): when set, any point whose size->depth HEIGHT
    uncertainty for a +-1 px diameter error exceeds it gets height=NaN and its
    court position from the PLANE projection instead of the 3D estimate. On
    the stored calibration the far-baseline ball is <5 px wide, where 1 px of
    diameter error is ~2.9 m of depth (~2.85 m of court-y, ~0.57 m of height)
    — far "heights" there are noise dressed as measurement, and the 3D court
    position is far worse than the homography projection. Marking them NaN
    makes downstream honest: the bounce heuristic skips NaN heights and the
    phantom-out guard already requires height_known."""
    if not track:
        return []
    frames = np.array([p.frame for p in track])
    diams = np.array([p.diameter_px if p.diameter_px else np.nan for p in track])
    obs_mask = ~np.isnan(diams)
    if obs_mask.sum() < 3:
        # Not enough size info for 3D — degrade to plane projection (height
        # unknown, reported as NaN so downstream logic knows).
        out = []
        for p in track:
            cxy = court.img_to_court(cam.H, np.array([[p.x, p.y]]))[0]
            out.append(BallWorld(p.frame, float(cxy[0]), float(cxy[1]), float("nan"), p.observed))
        return out
    d = diams.copy()
    d[obs_mask] = _median_filter(diams[obs_mask], 7)
    d[~obs_mask] = np.interp(frames[~obs_mask], frames[obs_mask], d[obs_mask])

    out: list[BallWorld] = []
    unreliable: list[bool] = []
    for p, di in zip(track, d):
        di = max(float(di), 1.0)
        bad = (height_sigma_max is not None
               and cam.height_sigma(p.x, p.y, di) > height_sigma_max)
        if bad:
            cxy = court.img_to_court(cam.H, np.array([[p.x, p.y]]))[0]
            out.append(BallWorld(p.frame, float(cxy[0]), float(cxy[1]),
                                 float("nan"), p.observed))
        else:
            w = cam.ball_world_from_size(p.x, p.y, di)
            out.append(BallWorld(p.frame, float(w[0]), float(w[1]), float(w[2]),
                                 p.observed))
        unreliable.append(bad)
    # Smooth height — bounce detection reads local minima off this curve.
    # NaN-aware so the sigma gate only blanks the points it flagged; a
    # reliable point next to an unreliable one keeps its own (smoothed) value.
    h = np.array([b.height for b in out])
    if len(h) >= 3:
        pad = 2
        padded = np.pad(h, pad, mode="edge")
        h = np.array([float(np.nanmedian(padded[i:i + 5]))
                      if not np.isnan(h[i]) else float("nan")
                      for i in range(len(h))])
    for b, hv in zip(out, h):
        b.height = float(hv)
    return out


def detect_events(
    world: list[BallWorld],
    players_by_frame: dict[int, list[tuple[float, float]]],  # frame -> court xy list
    params: dict,
    fps: float,
    model=None,
    entries: dict[int, dict] | None = None,
    H: np.ndarray | None = None,
    ball_px: dict[int, tuple[float, float]] | None = None,
) -> list[BallEvent]:
    """Find trajectory events and classify bounce vs hit.

    Candidate events = local minima of height (bounce candidates) plus sharp
    reversals of court-depth velocity (hit candidates). Classification:
      reversal at court-surface level away from any player  -> bounce
      reversal at height, or right at a player              -> hit
    A gravity sanity check rejects height 'minima' that are just size noise:
    around a real bounce the height curve must actually descend then ascend.

    When `H` and `ball_px` (frame -> observed pixel) are provided, BOUNCE
    court positions come from the plane homography instead of the size->depth
    world estimate: a bounce is by definition ON the court plane, where the
    homography is exact, while the 3D estimate carries metres of far-court
    diameter noise. This is what makes far-court in/out calls usable. Hits
    keep the world estimate (the ball is airborne, so neither is exact).

    If a trained rally_end model is supplied (with the per-frame `entries`
    trace), it decides WHICH frames are bounces/hits and their class; the WHERE
    (court position, in/out, height, side) still comes from the reconstructed
    `world`, so detect_end's out-of-bounds/height safety guards keep applying.
    No model -> the hand-coded heuristic below runs unchanged (graceful default).
    """
    if model is not None and entries:
        return _events_from_model(model, entries, world, players_by_frame, params, fps)
    if len(world) < 7:
        return []
    bounce_h = params.get("bounce_max_height_m", 0.20)
    hit_h = params.get("hit_min_height_m", 0.45)
    player_r = params.get("player_radius_m", 1.3)
    out_max = params.get("out_max_dist_m", 1.5)
    min_gap = max(3, int(fps * params.get("min_event_gap_s", 0.12)))

    h = np.array([b.height for b in world])
    ys = np.array([b.court_y for b in world])
    vy = np.gradient(_median_filter(ys, 5))
    events: list[BallEvent] = []

    def near_player(i: int) -> bool:
        pls = players_by_frame.get(world[i].frame, [])
        return any(
            (world[i].court_x - px) ** 2 + (world[i].court_y - py) ** 2 < player_r ** 2
            for px, py in pls
        )

    def court_pos(i: int, kind: str) -> tuple[float, float]:
        """Court position for the event at index i. Bounces are on the plane,
        so the homography projection of the observed pixel is exact and beats
        the size->depth world estimate (metres of far-court noise)."""
        b = world[i]
        if kind == "bounce" and H is not None and ball_px is not None:
            px = ball_px.get(b.frame)
            if px is not None:
                cxy = court.img_to_court(H, np.array([[px[0], px[1]]]))[0]
                return float(cxy[0]), float(cxy[1])
        return b.court_x, b.court_y

    def add(i: int, kind: str, detail: dict) -> None:
        b = world[i]
        if events and b.frame - events[-1].frame < min_gap:
            return
        cx, cy = court_pos(i, kind)
        events.append(BallEvent(
            frame=b.frame, kind=kind, court_x=cx, court_y=cy,
            height=b.height if not np.isnan(b.height) else -1.0,
            in_court=bool(court.in_court(np.array([[cx, cy]]))[0]),
            side=court.side_of_net(cy), near_player=near_player(i),
            observed=b.observed, height_known=not np.isnan(b.height),
            detail=detail,
        ))

    n = len(world)
    win = max(2, int(fps * 0.08))
    for i in range(win, n - win):
        if np.isnan(h[i]):
            continue
        # --- bounce candidate: local height minimum near the court plane ---
        if h[i] == np.nanmin(h[i - win:i + win + 1]) and h[i] < bounce_h:
            desc = h[i - win] - h[i]
            asc = h[i + win] - h[i]
            if desc > 0.02 and asc > 0.02:  # gravity check: real dip, not noise
                if near_player(i):
                    add(i, "hit", {"reason": "surface-level reversal at player"})
                else:
                    b = world[i]
                    cx, cy = court_pos(i, "bounce")
                    ic = bool(court.in_court(np.array([[cx, cy]]))[0])
                    if not ic:
                        # Phantom-out guard (mirrors detect_end): an out-of-court
                        # bounce must be a REAL, height-known detection landing
                        # within out_max of a line. Otherwise it is tracking
                        # noise near a boundary and must not become a "bounce
                        # OUT" (which also stops fake events feeding pass logic).
                        dx = max(0.0, -cx, cx - court.COURT_W)
                        dy = max(0.0, -cy, cy - court.COURT_L)
                        if (not b.observed or np.isnan(b.height)
                                or (dx * dx + dy * dy) ** 0.5 > out_max):
                            continue
                    add(i, "bounce", {"descent_m": round(float(desc), 3),
                                      "ascent_m": round(float(asc), 3)})
                continue
        # --- hit candidate: depth-velocity reversal at height / at player ---
        if i >= 1 and vy[i - 1] * vy[i + 1] < 0 and abs(vy[i - 1] - vy[i + 1]) > 0.02:
            if (not np.isnan(h[i]) and h[i] > hit_h) or near_player(i):
                add(i, "hit", {"reason": "depth reversal at height or player"})
    return events


# ---------------------------------------------------------------------------
# Learned temporal bounce model — preferred over the heuristic once trained.
# Train with backend/scripts/train_bounce.py on accumulated feature traces +
# human rally labels. Training registers a new SHADOW version in the model
# registry; once validated + promoted to 'live' it is loaded by Pass 2 via
# model_registry.load_live('rally_end') and passed into detect_events (which
# calls _events_from_model). Until a version is promoted, the heuristic runs.
# ---------------------------------------------------------------------------
BOUNCE_MODEL_FEATURES = 8  # [x, y, vx, vy, diam, height, dist_to_player, observed]


def trace_to_features(entries: list[dict]) -> np.ndarray:
    """Per-frame feature vector aligned with BOUNCE_MODEL_FEATURES:
    [x, y, vx, vy, diam, height, dist_to_player, observed].

    SHARED by training (dataset assembly) and inference (Pass 2) so there is no
    train/serve skew — the model sees identical featurization in both paths."""
    out = np.zeros((len(entries), BOUNCE_MODEL_FEATURES), dtype=np.float32)
    prev = None
    for i, e in enumerate(entries):
        ball = e.get("ball")
        world = e.get("world")
        if ball:
            out[i, 0], out[i, 1] = ball[0] / 1000.0, ball[1] / 1000.0
            if prev is not None:
                out[i, 2], out[i, 3] = out[i, 0] - prev[0], out[i, 1] - prev[1]
            out[i, 4] = (ball[2] or 0) / 50.0
            prev = (out[i, 0], out[i, 1])
        if world:
            out[i, 5] = (world[2] if world[2] is not None else -1.0)
            dists = []
            for p in e.get("players", []):
                if p.get("cxy") and world[0] is not None:
                    dists.append(np.hypot(world[0] - p["cxy"][0], world[1] - p["cxy"][1]))
            out[i, 6] = float(min(dists)) if dists else 10.0
        out[i, 7] = 1.0 if e.get("obs") else 0.0
    return out


def _events_from_model(model, entries: dict[int, dict], world: list[BallWorld],
                       players_by_frame: dict, params: dict, fps: float) -> list[BallEvent]:
    """Run the trained per-frame bounce/hit classifier and lift its non-zero
    predictions into BallEvents, taking geometry from the reconstructed world."""
    import torch

    ents = [entries[f] for f in sorted(entries)]
    if len(ents) < 3:
        return []
    feats = trace_to_features(ents)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(feats).unsqueeze(0)), dim=-1)[0].numpy()
    cls = probs.argmax(-1)
    wbf = {w.frame: w for w in world}
    player_r = params.get("player_radius_m", 1.3)
    min_gap = max(3, int(fps * params.get("min_event_gap_s", 0.12)))
    events: list[BallEvent] = []
    for idx, e in enumerate(ents):
        c = int(cls[idx])
        if c == 0:
            continue
        w = wbf.get(e["f"])
        if w is None:
            continue
        if events and e["f"] - events[-1].frame < min_gap:
            continue
        pls = players_by_frame.get(w.frame, [])
        near = any((w.court_x - px) ** 2 + (w.court_y - py) ** 2 < player_r ** 2 for px, py in pls)
        events.append(BallEvent(
            frame=w.frame, kind="bounce" if c == 1 else "hit",
            court_x=w.court_x, court_y=w.court_y,
            height=w.height if not np.isnan(w.height) else -1.0,
            in_court=bool(court.in_court(np.array([[w.court_x, w.court_y]]))[0]),
            side=court.side_of_net(w.court_y), near_player=near,
            observed=w.observed, height_known=not np.isnan(w.height),
            detail={"model": True, "p": round(float(probs[idx, c]), 3)}))
    return events


def build_bounce_model():
    import torch.nn as nn

    class BounceGRU(nn.Module):
        """Per-frame 3-way classification over a sliding window:
        0 = nothing, 1 = bounce, 2 = hit."""

        def __init__(self, feat: int = BOUNCE_MODEL_FEATURES, hidden: int = 64):
            super().__init__()
            self.gru = nn.GRU(feat, hidden, num_layers=2, batch_first=True,
                              bidirectional=True)
            self.head = nn.Linear(hidden * 2, 3)

        def forward(self, x):  # x: (B, T, feat) -> (B, T, 3)
            out, _ = self.gru(x)
            return self.head(out)

    return BounceGRU()


def load_bounce_model(path: str | Path = ""):
    """Returns a ready eval-mode model or None if no trained weights exist."""
    import torch

    from ..config import DATA_DIR
    p = Path(path) if path else DATA_DIR / "models" / "bounce_gru.pt"
    if not p.exists():
        return None
    m = build_bounce_model()
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m


def _demo() -> None:
    """Self-check for the height-uncertainty gate + bounce plane-projection:
      1. reconstruct_world(height_sigma_max=...): a near-camera point (big
         apparent diameter, low height uncertainty) keeps its 3D height; a
         far-camera point (tiny diameter, high uncertainty) gets height=NaN
         and a plane-projected court position instead of a noisy 3D one.
      2. detect_events(H=..., ball_px=...): a bounce event's court position
         comes from the homography plane projection of the observed pixel,
         not the (possibly noisy) reconstructed world position.
    """
    import cv2

    # Synthetic axial-camera homography: far baseline narrow near image top,
    # near baseline wide near image bottom (typical down-the-court view).
    img_pts = np.float32([[400, 200], [880, 200], [1180, 700], [100, 700]])
    court_pts = np.float32([[0, 0], [court.COURT_W, 0],
                            [court.COURT_W, court.COURT_L], [0, court.COURT_L]])
    H, _ = cv2.findHomography(img_pts, court_pts)
    cam = CameraModel.from_homography(H, 1280, 720)

    # --- 1. height_sigma_max NaN gate ---------------------------------------
    # Separate near/far tracks (not mixed) so the diameter median filter can't
    # wash out the single differing sample — each track is internally uniform.
    near_px, near_diam = (640, 650), 40.0   # big blob near the camera/near baseline
    far_px, far_diam = (640, 210), 3.0      # tiny blob at the far baseline
    near_track = [TrackPoint(i, near_px[0], near_px[1], 0.0, 0.0, True, near_diam)
                  for i in range(3)]
    far_track = [TrackPoint(i, far_px[0], far_px[1], 0.0, 0.0, True, far_diam)
                 for i in range(3)]
    near_world = reconstruct_world(near_track, cam, height_sigma_max=0.35)
    far_world = reconstruct_world(far_track, cam, height_sigma_max=0.35)
    assert not any(np.isnan(w.height) for w in near_world), \
        "near, low-uncertainty point wrongly NaN'd"
    assert all(np.isnan(w.height) for w in far_world), \
        "far, high-uncertainty point was NOT flagged NaN"
    # NaN'd point's court position must match the plane projection, not garbage.
    plane_xy = court.img_to_court(cam.H, np.array([[far_px[0], far_px[1]]]))[0]
    assert abs(far_world[0].court_x - plane_xy[0]) < 1e-6
    assert abs(far_world[0].court_y - plane_xy[1]) < 1e-6

    # --- 2. detect_events court_pos: bounce uses the plane, not noisy world -
    fps = 30.0
    n = 20
    mid = n // 2
    frames = list(range(n))
    # A height curve that dips to ~0 at `mid` (a bounce candidate) away from
    # any player, with the WORLD court position at that frame deliberately
    # noisy/wrong so the test can tell whether court_pos() used it or not.
    heights = [0.5 - 0.5 * (1 - abs(i - mid) / mid) for i in frames]
    bogus_world = [
        BallWorld(frame=i, court_x=999.0, court_y=999.0, height=heights[i], observed=True)
        for i in frames
    ]
    ball_px = {i: (640.0, 450.0) for i in frames}  # a plausible in-court pixel
    events = detect_events(bogus_world, {}, {}, fps, H=H, ball_px=ball_px)
    bounces = [e for e in events if e.kind == "bounce"]
    assert bounces, "expected at least one bounce candidate from the height dip"
    exp_xy = court.img_to_court(H, np.array([[640.0, 450.0]]))[0]
    b = bounces[0]
    assert abs(b.court_x - exp_xy[0]) < 1e-6 and abs(b.court_y - exp_xy[1]) < 1e-6, \
        "bounce court position did not come from the plane projection"
    assert (b.court_x, b.court_y) != (999.0, 999.0), \
        "bounce used the noisy world position instead of the plane"

    print("bounce height-sigma + plane-projection demo OK")


if __name__ == "__main__":
    _demo()

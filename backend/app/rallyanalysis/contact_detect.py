"""Pose-only shot/contact detection.

NEVER derives from bounce.py or ball_track.py — ball tracking is unreliable
(especially far-baseline) and must not be a dependency here. Detection runs
entirely off player pose keypoints already stored in ``overlays`` (written by
the existing Pass 1/2 pipeline, which is read-only from here).

Algorithm: for each tracked player in a rally window, take the faster of
their two wrist keypoints per frame (whichever hand is swinging — we don't
know handedness, so "hitting-side wrist" = max(L,R) speed) and look for local
peaks that clear an ADAPTIVE per-player threshold (median + k*MAD of that
player's own wrist speed in this rally) — motion energy varies a lot between
players, so a global constant would either miss quiet players or fire on
every gesture from an energetic one. A minimum time gap between accepted
peaks for the same player collapses one swing's accel/decel into a single
contact and filters rapid non-shot motion (grip adjustments, gesturing).

court_x/court_y come straight from the overlay's ``cxy`` field — already
projected through the calibrated homography upstream by cv/players.py, so
there is no need to re-load calibration or re-project here.
"""
from __future__ import annotations

import math

import numpy as np

from .. import db
from ..cv.court import COURT_W
from ..cv.players import PlayerObs, is_serve_formation
from ..cv import overlays
from .contracts import ContactEvent
from .position_metrics import _default_overlay_source
from .readiness import get_analyzable_rallies

L_WRIST, R_WRIST = 9, 10  # COCO-17 keypoint indices

K_MAD = 3.5              # peak must clear median + K_MAD * MAD of the player's own speed
MIN_ABS_FLOOR = 40.0      # px/s; guards a near-static player against noise-floor "peaks"
MIN_GAP_S = 0.35          # min time between two accepted contacts, same player
SERVE_SEARCH_FRAMES = 15  # +/- frames searched for a nearby overlay entry to check serve formation


# ---------------------------------------------------------------------------
# Pure, DB-free core (unit-tested directly on synthetic sequences)
# ---------------------------------------------------------------------------

def _valid_xy(pt) -> bool:
    return pt is not None and pt[0] > 0 and pt[1] > 0


def extract_wrist_track(entries: list[dict], player_slot) -> list[tuple[int, tuple | None, tuple | None, tuple | None, str | None]]:
    """entries: overlay rows sorted by frame, each {'f','players':[{'id','cxy','team','kp',...]}.
    Returns [(frame_idx, left_wrist_xy|None, right_wrist_xy|None, court_xy|None, team|None)]
    for frames where this player slot is present."""
    out = []
    for e in entries:
        for p in e.get("players", []):
            if p.get("id") == player_slot:
                kp = p.get("kp")
                lw = rw = None
                if kp and len(kp) > R_WRIST:
                    l_raw, r_raw = kp[L_WRIST], kp[R_WRIST]
                    if _valid_xy(l_raw):
                        lw = (float(l_raw[0]), float(l_raw[1]))
                    if _valid_xy(r_raw):
                        rw = (float(r_raw[0]), float(r_raw[1]))
                cxy = p.get("cxy")
                out.append((e["f"], lw, rw, tuple(cxy) if cxy else None, p.get("team")))
                break
    return out


def wrist_speed_series(track: list[tuple]) -> list[tuple[int, float, tuple | None, str | None]]:
    """track: output of extract_wrist_track (fps-agnostic, unit = px/frame here;
    caller scales by fps). Returns [(frame_idx, speed_px_per_frame, court_xy, team)],
    one entry per frame that has a valid diff against the previous present frame
    (max of the two wrists' displacement / frame_gap)."""
    out = []
    prev = None  # (frame_idx, lw, rw)
    for f, lw, rw, cxy, team in track:
        if prev is not None:
            pf, plw, prw = prev
            gap = f - pf
            if gap > 0:
                speeds = []
                if lw is not None and plw is not None:
                    speeds.append(math.hypot(lw[0] - plw[0], lw[1] - plw[1]) / gap)
                if rw is not None and prw is not None:
                    speeds.append(math.hypot(rw[0] - prw[0], rw[1] - prw[1]) / gap)
                if speeds:
                    out.append((f, max(speeds), cxy, team))
        prev = (f, lw, rw)
    return out


def find_peaks(frame_idx: list[int], speed_px_per_frame: list[float], fps: float,
               k: float = K_MAD, min_gap_s: float = MIN_GAP_S,
               floor: float = MIN_ABS_FLOOR) -> list[tuple[int, float, float]]:
    """Local maxima in speed (already scaled to px/s by the caller — see
    detect_contacts_for_rally) that clear this player's own adaptive
    threshold, non-max-suppressed to at most one contact per min_gap_s.
    Returns [(frame_idx, speed_px_s, confidence 0-1)] sorted by frame_idx."""
    n = len(speed_px_per_frame)
    if n < 3:
        return []
    speed = np.asarray(speed_px_per_frame, dtype=float)
    med = float(np.median(speed))
    mad = float(np.median(np.abs(speed - med))) * 1.4826  # normal-consistent MAD estimator
    threshold = max(med + k * mad, floor)

    candidates = []
    for i in range(1, n - 1):
        if speed[i] >= speed[i - 1] and speed[i] >= speed[i + 1] and speed[i] > threshold:
            confidence = min(1.0, (speed[i] - threshold) / threshold) if threshold > 0 else 1.0
            candidates.append((frame_idx[i], float(speed[i]), float(confidence)))

    candidates.sort(key=lambda c: -c[1])  # strongest first, for greedy NMS
    min_gap_frames = min_gap_s * fps
    accepted: list[tuple[int, float, float]] = []
    for f, sp, conf in candidates:
        if all(abs(f - af) >= min_gap_frames for af, _, _ in accepted):
            accepted.append((f, sp, conf))
    accepted.sort(key=lambda c: c[0])
    return accepted


def _players_to_obs(players: list[dict]) -> list[PlayerObs]:
    obs = []
    for p in players:
        box = tuple(p.get("box") or (0.0, 0.0, 0.0, 0.0))
        cxy = p.get("cxy")
        obs.append(PlayerObs(
            track_id=int(p.get("id") or 0),
            box=box,
            foot_xy=((box[0] + box[2]) / 2.0, box[3]),
            court_xy=tuple(cxy) if cxy else None,
            keypoints=p.get("kp") or [],
            team=p.get("team"),
        ))
    return obs


def _check_serve_formation(by_frame: dict[int, dict], frame: int,
                            search: int = SERVE_SEARCH_FRAMES) -> bool:
    """Nearest overlay entry to `frame` (ties favour earlier), serve-formation
    check via the existing cv/players.py helper (never redefined here)."""
    e = by_frame.get(frame)
    if e is not None:
        return is_serve_formation(_players_to_obs(e.get("players", [])))
    for df in range(1, search + 1):
        for f in (frame - df, frame + df):
            e = by_frame.get(f)
            if e is not None:
                return is_serve_formation(_players_to_obs(e.get("players", [])))
    return False


def detect_contacts_for_rally(video_id: str, rally_id: str, entries: list[dict],
                               fps: float) -> list[ContactEvent]:
    """entries: overlay rows for exactly one rally window, any order."""
    if not entries or fps <= 0:
        return []
    entries = sorted(entries, key=lambda e: e["f"])
    by_frame = {e["f"]: e for e in entries}
    slots = sorted({p.get("id") for e in entries for p in e.get("players", [])
                    if p.get("id") is not None})

    all_peaks = []  # dicts: frame, player_id, team, cxy, speed, confidence
    for slot in slots:
        track = extract_wrist_track(entries, slot)
        samples = wrist_speed_series(track)
        if len(samples) < 3:
            continue
        frame_idx = [s[0] for s in samples]
        speed_px_s = [s[1] * fps for s in samples]  # px/frame -> px/s
        cxy_by_frame = {s[0]: (s[2], s[3]) for s in samples}
        for f, sp, conf in find_peaks(frame_idx, speed_px_s, fps):
            cxy, team = cxy_by_frame.get(f, (None, None))
            if cxy is None:
                continue  # can't localize on court -> can't emit a ContactEvent
            all_peaks.append({"frame": f, "player_id": slot, "team": team,
                              "cxy": cxy, "speed": sp, "confidence": conf})

    if not all_peaks:
        return []
    all_peaks.sort(key=lambda p: p["frame"])
    first_is_serve = _check_serve_formation(by_frame, all_peaks[0]["frame"])

    events = []
    for i, p in enumerate(all_peaks):
        shot_side = "left" if p["cxy"][0] < COURT_W / 2.0 else "right"
        events.append(ContactEvent(
            contact_id=db.new_id(),
            video_id=video_id,
            rally_id=rally_id,
            frame_idx=p["frame"],
            timestamp_s=p["frame"] / fps,
            player_id=p["player_id"],
            team=p["team"] or "near",
            court_x=p["cxy"][0],
            court_y=p["cxy"][1],
            shot_side=shot_side,
            hand_speed_peak=p["speed"],
            confidence=p["confidence"],
            is_serve=(i == 0 and first_is_serve),
        ))
    return events


# ---------------------------------------------------------------------------
# DB orchestration (reads rally_labels/overlays only; writes contact_events only)
# ---------------------------------------------------------------------------

def detect_contacts_for_video(video_id: str, rally_source: str | None = None,
                               overlay_source: str | None = None) -> list[ContactEvent]:
    """Only rallies readiness.get_analyzable_rallies() clears (Pass 2 done,
    and its overlay data still covers the rally's current frame range) are
    processed — a rally whose Pass 2 job is running/missing/errored, or
    whose overlay coverage has gone stale relative to its current
    start/end frame, is silently skipped rather than analyzed against
    partial data."""
    # NB: rally_source can now legitimately be None (see module docstring on
    # the rally_source default fix) — fall through to the same pass2-if-
    # available-else-pass1 default readiness.py uses, rather than leaving
    # overlay_source unresolved for the read_range calls below.
    overlay_source = overlay_source or rally_source or _default_overlay_source(video_id)
    with db.connect() as con:
        vrow = con.execute("SELECT fps FROM videos WHERE id=?", (video_id,)).fetchone()
    fps = float(vrow["fps"]) if vrow and vrow["fps"] else 30.0
    rallies = get_analyzable_rallies(video_id, rally_source=rally_source, overlay_source=overlay_source)

    events: list[ContactEvent] = []
    for rally in rallies:
        entries = overlays.read_range(video_id, overlay_source, rally["start_frame"], rally["end_frame"])
        events.extend(detect_contacts_for_rally(video_id, rally["id"], entries, fps))
    return events


def save_events(video_id: str, events: list[ContactEvent]) -> None:
    """Replaces this video's contact_events (own table — re-running detection
    is idempotent, not an append)."""
    with db.connect() as con:
        con.execute("DELETE FROM contact_events WHERE video_id=?", (video_id,))
        con.executemany(
            "INSERT INTO contact_events (contact_id, video_id, rally_id, frame_idx, timestamp_s,"
            " player_id, team, court_x, court_y, shot_side, hand_speed_peak, confidence, is_serve)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(e.contact_id, e.video_id, e.rally_id, e.frame_idx, e.timestamp_s, e.player_id,
              e.team, e.court_x, e.court_y, e.shot_side, e.hand_speed_peak, e.confidence,
              int(e.is_serve)) for e in events])

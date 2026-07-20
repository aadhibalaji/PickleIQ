"""Unit tests for pose-only contact detection, on synthetic pose sequences
(clear swing vs. no swing) — no DB, no video, no ball data involved."""
from __future__ import annotations

from ..contact_detect import (
    detect_contacts_for_rally,
    extract_wrist_track,
    find_peaks,
    wrist_speed_series,
)
from ...cv.court import KITCHEN, NET_Y

FPS = 30.0


def _kp17(right_wrist_xy, left_wrist_xy=(50.0, 50.0)):
    kp = [[100.0, 100.0]] * 17
    kp[9] = list(left_wrist_xy)
    kp[10] = list(right_wrist_xy)
    return kp


def _make_track(right_wrist_positions: list[tuple[float, float]], cxy=(2.0, 3.0), team="near"):
    """One player's overlay entries, frame == index. Left wrist stays still
    (baseline noise-free) so only the right wrist drives the speed signal."""
    entries = []
    for f, xy in enumerate(right_wrist_positions):
        entries.append({
            "f": f,
            "players": [{"id": 1, "box": [0, 0, 50, 200], "cxy": list(cxy),
                        "team": team, "kp": _kp17(xy)}],
        })
    return entries


def test_find_peaks_detects_clear_swing():
    # Baseline: near-static wrist (tiny jitter). One frame has a big jump = a
    # swing. Speed is expressed in px/frame here (matches wrist_speed_series
    # output before the fps scale-up find_peaks itself is agnostic to).
    speed = [1.0, 1.2, 0.9, 1.1, 1.0, 60.0, 1.0, 0.8, 1.1, 0.9, 1.0]
    frame_idx = list(range(len(speed)))
    peaks = find_peaks(frame_idx, speed, FPS)
    assert len(peaks) == 1
    assert peaks[0][0] == 5
    assert peaks[0][1] == 60.0
    assert peaks[0][2] > 0  # confidence


def test_find_peaks_no_swing_stays_quiet():
    # Pure baseline jitter, nothing exceeds the adaptive threshold.
    speed = [1.0, 1.3, 0.8, 1.1, 1.0, 1.4, 0.9, 1.2, 1.0, 1.1]
    frame_idx = list(range(len(speed)))
    peaks = find_peaks(frame_idx, speed, FPS)
    assert peaks == []


def test_find_peaks_enforces_min_gap_between_same_player_contacts():
    # Two spikes 3 frames apart (0.1s @ 30fps) — well under MIN_GAP_S (0.35s) —
    # must collapse to the single stronger one, not double-count a swing's
    # accel/decel (or rapid grip-adjust noise) as two contacts.
    speed = [1.0] * 5 + [80.0] + [1.0] * 2 + [50.0] + [1.0] * 5
    frame_idx = list(range(len(speed)))
    peaks = find_peaks(frame_idx, speed, FPS, min_gap_s=0.35)
    assert len(peaks) == 1
    assert peaks[0][1] == 80.0


def test_wrist_speed_series_picks_faster_wrist():
    track = extract_wrist_track(_make_track([(50.0, 50.0), (250.0, 50.0)]), player_slot=1)
    samples = wrist_speed_series(track)
    assert len(samples) == 1
    assert samples[0][1] == 200.0  # right wrist moved 200px in 1 frame, left stayed put


def test_detect_contacts_for_rally_flags_clear_swing():
    positions = [(50.0, 50.0)] * 10 + [(300.0, 50.0)] + [(50.0, 50.0)] * 10
    entries = _make_track(positions, cxy=(1.0, 3.0))  # left of centreline (COURT_W/2 ~3.05)
    events = detect_contacts_for_rally("vid1", "rally1", entries, FPS)
    assert len(events) == 1
    ev = events[0]
    assert ev.frame_idx == 10
    assert ev.player_id == 1
    assert ev.shot_side == "left"
    assert ev.hand_speed_peak > 0
    assert 0 <= ev.confidence <= 1


def test_detect_contacts_for_rally_no_swing_no_events():
    positions = [(50.0 + (i % 2), 50.0) for i in range(20)]  # 1px jitter only
    entries = _make_track(positions)
    events = detect_contacts_for_rally("vid1", "rally1", entries, FPS)
    assert events == []


def _two_player_entry(f, near_cxy, far_cxy, near_kp=None, far_kp=None):
    return {
        "f": f,
        "players": [
            {"id": 1, "box": [0, 0, 50, 200], "cxy": list(near_cxy), "team": "near",
             "kp": near_kp or _kp17((50.0, 50.0))},
            {"id": 2, "box": [400, 0, 450, 200], "cxy": list(far_cxy), "team": "far",
             "kp": far_kp or _kp17((450.0, 50.0))},
        ],
    }


def test_is_serve_true_for_first_contact_in_serve_formation():
    # Near player deep behind their kitchen line, far player deep behind
    # theirs -> is_serve_formation() (cv/players.py, imported not redefined).
    near_deep = (2.0, NET_Y + KITCHEN + 4.0)
    far_deep = (2.0, NET_Y - KITCHEN - 3.5)
    entries = []
    for f in range(15):
        rw = (300.0, 50.0) if f == 7 else (50.0, 50.0)
        entries.append(_two_player_entry(f, near_deep, far_deep, near_kp=_kp17(rw)))
    events = detect_contacts_for_rally("vid1", "rally1", entries, FPS)
    assert len(events) == 1
    assert events[0].is_serve is True


def test_is_serve_false_when_not_in_serve_formation():
    # Both players parked at the net -> fails the "someone deep" check.
    near_at_net = (2.0, NET_Y + 0.5)
    far_at_net = (2.0, NET_Y - 0.5)
    entries = []
    for f in range(15):
        rw = (300.0, 50.0) if f == 7 else (50.0, 50.0)
        entries.append(_two_player_entry(f, near_at_net, far_at_net, near_kp=_kp17(rw)))
    events = detect_contacts_for_rally("vid1", "rally1", entries, FPS)
    assert len(events) == 1
    assert events[0].is_serve is False

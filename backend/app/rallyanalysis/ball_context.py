"""Ball-derived coaching context — OPPORTUNISTIC enrichment, never a
dependency: consumes whatever ball data the pass2 overlays happen to carry
(TrackNet / classical detector / external tracker via ball_tracks /
PASS2_BALL=1) and yields nothing at all when a rally has none, the same
degrade-honestly pattern as keyframes.py.

Two signals, both chosen because they close known structural gaps in the
coaching pipeline rather than merely adding data:

1. tempo_verified ('soft' | 'fast' | None): the swing's tempo measured from
   actual post-contact ball speed in court metres/second (overlay "world"
   samples — homography + ballistic reconstruction), instead of inferred
   from wrist pixel speed relative to the player's own session median.
   Overrides the tempo half of shot_context.classify_shot_types' pose-only
   stratification key when present. Deliberately three-valued: the band
   between TEMPO_SOFT_MPS and TEMPO_FAST_MPS is ambiguous and yields None
   (fall back to the pose guess) rather than a coin-flip label.

2. rally_end_ball ('landed_out' | 'landed_in_unreturned' | None): what the
   tracked ball actually did after the rally's final swing. This is the one
   evidence source that can honestly lift the coaching prompt's
   winner-vs-error ban — a bounce called out after the last contact means
   that swing ended the rally as an error; an in-bounds bounce nobody
   returned means it won the point. Only ever attached to the rally's LAST
   contact, and only when the event lands within END_EVT_MAX_LAG_S of it
   (a bounce seconds later is the next rally's noise, not this outcome).

Speeds use ONLY court-plane "world" coordinates — image-pixel ball speeds
are camera-distance dependent (a far-side drive crawls in pixels) and would
misclassify tempo by side, the same reason shot_context refuses absolute
pixel-speed cutoffs.
"""
from __future__ import annotations

# Post-contact sampling window: skip the first beat (paddle occlusion and
# detection jitter right at contact), stop well before the far-side bounce.
SPEED_WINDOW_S = (0.05, 0.35)
MIN_SPEED_SAMPLES = 3  # fewer world samples than this in the window -> no speed claim

# Court-metre speed cutoffs. JUDGMENT CALLS pending labeled-clip validation
# (same status as positioning.BANDS): dinks/resets cross the kitchen at a
# few m/s; drives leave the paddle well into double digits. The gap between
# the two thresholds is deliberately unclassified.
TEMPO_SOFT_MPS = 5.0
TEMPO_FAST_MPS = 8.0

# A rally-ending bounce must land within this many seconds of the final
# contact to be attributed to it.
END_EVT_MAX_LAG_S = 3.0

LANDED_OUT = "landed_out"
LANDED_IN_UNRETURNED = "landed_in_unreturned"


def _world_positions(entries: list[dict]) -> dict[int, tuple[float, float]]:
    """{frame: (court_x, court_y)} for every overlay frame with a real
    world-space ball sample. Frames without one are absent, never
    interpolated here — kalman/arc interpolation already happened upstream
    and is marked; this layer adds no fabrication of its own."""
    out: dict[int, tuple[float, float]] = {}
    for e in entries:
        w = e.get("world")
        if w and w[0] is not None and w[1] is not None:
            out[e["f"]] = (float(w[0]), float(w[1]))
    return out


def speed_after_contact(positions: dict[int, tuple[float, float]],
                         contact_frame: int, fps: float) -> float | None:
    """Mean court-plane ball speed (m/s) over SPEED_WINDOW_S after the
    contact — path length between consecutive samples / elapsed time, or
    None when the tracker didn't see enough of the flight to say."""
    if fps <= 0:
        return None
    lo = contact_frame + SPEED_WINDOW_S[0] * fps
    hi = contact_frame + SPEED_WINDOW_S[1] * fps
    frames = sorted(f for f in positions if lo <= f <= hi)
    if len(frames) < MIN_SPEED_SAMPLES:
        return None
    path = 0.0
    for a, b in zip(frames, frames[1:]):
        (x0, y0), (x1, y1) = positions[a], positions[b]
        path += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    dt = (frames[-1] - frames[0]) / fps
    if dt <= 0:
        return None
    return path / dt


def tempo_for_speed(speed_mps: float | None) -> str | None:
    if speed_mps is None:
        return None
    if speed_mps >= TEMPO_FAST_MPS:
        return "fast"
    if speed_mps <= TEMPO_SOFT_MPS:
        return "soft"
    return None  # ambiguous middle band — defer to the pose-based guess


def rally_end_outcome(entries: list[dict], last_contact_frame: int,
                       fps: float) -> str | None:
    """What the ball did after the rally's final swing, from overlay "evt"
    markers: the FIRST bounce at/after the last contact (within
    END_EVT_MAX_LAG_S). Out-of-court bounce -> that swing ended the rally as
    an error; in-court bounce with no later contact in this rally (true by
    construction — this IS the last contact) -> unreturned winner. No
    qualifying bounce (tracker lost the ball, rally ended on a net cord it
    never saw, ...) -> None, and the winner/error ban stays in force."""
    if fps <= 0:
        return None
    max_frame = last_contact_frame + END_EVT_MAX_LAG_S * fps
    for e in sorted(entries, key=lambda e: e["f"]):
        if e["f"] < last_contact_frame or e["f"] > max_frame:
            continue
        evt = e.get("evt")
        if not evt or evt.get("kind") != "bounce":
            continue
        return LANDED_OUT if evt.get("in") is False else LANDED_IN_UNRETURNED
    return None


def annotate_contacts(entries: list[dict], rally_contacts: list[dict],
                       fps: float) -> tuple[dict[str, dict], str | None]:
    """One rally's ball context: ({contact_id: {"ball_speed_after_mps",
    "tempo_verified"}}, rally_end_ball | None). Contacts the tracker can't
    speak to are simply absent from the dict — a missing key means "no ball
    evidence", never "slow" or "fine"."""
    positions = _world_positions(entries)
    by_contact: dict[str, dict] = {}
    if positions:
        for c in rally_contacts:
            speed = speed_after_contact(positions, c["frame_idx"], fps)
            if speed is None:
                continue
            by_contact[c["contact_id"]] = {
                "ball_speed_after_mps": round(speed, 2),
                "tempo_verified": tempo_for_speed(speed),
            }
    end_ball = None
    if rally_contacts:
        last = max(rally_contacts, key=lambda c: c["frame_idx"])
        end_ball = rally_end_outcome(entries, last["frame_idx"], fps)
    return by_contact, end_ball


if __name__ == "__main__":
    # ponytail self-check: speed math, tempo bands, outcome attribution,
    # honest absence. No DB.
    fps = 30.0
    # Ball moving 0.3 m per frame = 9 m/s -> fast.
    entries = [{"f": f, "world": [1.0 + 0.3 * f, 5.0, None]} for f in range(0, 15)]
    pos = _world_positions(entries)
    s = speed_after_contact(pos, 0, fps)
    assert s is not None and abs(s - 9.0) < 0.2, s
    assert tempo_for_speed(s) == "fast"
    assert tempo_for_speed(3.0) == "soft"
    assert tempo_for_speed(6.5) is None          # ambiguous middle: no claim
    assert speed_after_contact({}, 0, fps) is None
    assert speed_after_contact({0: (0, 0), 3: (1, 0)}, 0, fps) is None  # too few samples

    contacts = [{"contact_id": "c1", "frame_idx": 0}, {"contact_id": "last", "frame_idx": 100}]
    out_entries = entries + [{"f": 130, "evt": {"kind": "bounce", "in": False, "side": "far"}}]
    by_c, end = annotate_contacts(out_entries, contacts, fps)
    assert by_c["c1"]["tempo_verified"] == "fast"
    assert "last" not in by_c                    # no flight samples after frame 100
    assert end == LANDED_OUT
    in_entries = entries + [{"f": 130, "evt": {"kind": "bounce", "in": True, "side": "far"}}]
    assert annotate_contacts(in_entries, contacts, fps)[1] == LANDED_IN_UNRETURNED
    # Bounce far too late (or a hit evt) is never attributed to the rally end.
    late = entries + [{"f": 100 + int(4 * fps), "evt": {"kind": "bounce", "in": False}}]
    assert annotate_contacts(late, contacts, fps)[1] is None
    hit_only = entries + [{"f": 110, "evt": {"kind": "hit", "in": True}}]
    assert annotate_contacts(hit_only, contacts, fps)[1] is None
    assert annotate_contacts([], contacts, fps) == ({}, None)
    print("ball_context self-check OK")

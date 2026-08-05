"""Ball-derived coaching context: court-metre speed math, tempo bands,
rally-outcome attribution, honest absence. Pure — no DB."""
from __future__ import annotations

from ..ball_context import (LANDED_IN_UNRETURNED, LANDED_OUT, MIN_SPEED_SAMPLES,
                             TEMPO_FAST_MPS, TEMPO_SOFT_MPS, annotate_contacts,
                             rally_end_outcome, speed_after_contact,
                             tempo_for_speed, _world_positions)

FPS = 30.0


def _flight(frames, step_m):
    """Overlay entries with a ball moving step_m per frame along court x."""
    return [{"f": f, "world": [1.0 + step_m * f, 5.0, None]} for f in frames]


def test_speed_is_court_metres_per_second():
    pos = _world_positions(_flight(range(0, 15), 0.3))  # 0.3 m/frame @ 30fps = 9 m/s
    s = speed_after_contact(pos, 0, FPS)
    assert s is not None and abs(s - 9.0) < 0.2


def test_speed_requires_enough_samples_in_the_window():
    assert speed_after_contact({}, 0, FPS) is None
    sparse = {0: (0.0, 0.0), 3: (1.0, 0.0)}
    assert len(sparse) < MIN_SPEED_SAMPLES
    assert speed_after_contact(sparse, 0, FPS) is None
    # Samples exist but all OUTSIDE the post-contact window -> no claim.
    pos = _world_positions(_flight(range(100, 115), 0.3))
    assert speed_after_contact(pos, 0, FPS) is None


def test_tempo_bands_leave_the_middle_unclassified():
    assert tempo_for_speed(TEMPO_FAST_MPS + 1) == "fast"
    assert tempo_for_speed(TEMPO_SOFT_MPS - 1) == "soft"
    assert tempo_for_speed((TEMPO_SOFT_MPS + TEMPO_FAST_MPS) / 2) is None
    assert tempo_for_speed(None) is None


def test_rally_end_outcome_attribution():
    entries = [{"f": 130, "evt": {"kind": "bounce", "in": False, "side": "far"}}]
    assert rally_end_outcome(entries, 100, FPS) == LANDED_OUT
    entries = [{"f": 130, "evt": {"kind": "bounce", "in": True, "side": "far"}}]
    assert rally_end_outcome(entries, 100, FPS) == LANDED_IN_UNRETURNED
    # Bounce before the final contact, a hit event, or a bounce far too late
    # after it are never attributed to the rally end.
    assert rally_end_outcome([{"f": 90, "evt": {"kind": "bounce", "in": False}}], 100, FPS) is None
    assert rally_end_outcome([{"f": 110, "evt": {"kind": "hit", "in": True}}], 100, FPS) is None
    assert rally_end_outcome([{"f": 100 + int(4 * FPS), "evt": {"kind": "bounce", "in": False}}],
                              100, FPS) is None


def test_annotate_contacts_absent_ball_data_yields_nothing():
    contacts = [{"contact_id": "c1", "frame_idx": 0}]
    assert annotate_contacts([], contacts, FPS) == ({}, None)
    no_world = [{"f": f, "players": []} for f in range(30)]
    assert annotate_contacts(no_world, contacts, FPS) == ({}, None)


def test_annotate_contacts_speeds_and_outcome_together():
    contacts = [{"contact_id": "drive", "frame_idx": 0},
                {"contact_id": "last", "frame_idx": 100}]
    entries = _flight(range(0, 15), 0.3) + [
        {"f": 120, "evt": {"kind": "bounce", "in": False, "side": "far"}}]
    by_c, end = annotate_contacts(entries, contacts, FPS)
    assert by_c["drive"]["tempo_verified"] == "fast"
    assert "last" not in by_c  # tracker lost the ball by then: no fabricated speed
    assert end == LANDED_OUT


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    failures = 0
    for name in dir(mod):
        if name.startswith("test_"):
            try:
                getattr(mod, name)()
                print(f"ok  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)

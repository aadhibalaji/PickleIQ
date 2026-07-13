"""Hardware-profile bundle: SPEED knobs tuned per machine, not for
cross-machine reproducibility. Two people running this on different GPUs
each get inference tuned to their own hardware — outputs are NOT expected
to match between machines.

Separate from learning/params.py LearnedParams on purpose: those are
detection-quality tunables that compound globally via the report; these are
hardware/performance knobs. Never write a profile value into learned_params.

Adding a profile = editing hw_profiles.json, no code change.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env here too (idempotent, doesn't override already-set vars) so
# active_profile() gives the right answer regardless of whether config.py
# has been imported yet — relying on import order for this was a real bug
# (a caller importing this module first got HW_PROFILE cached as unset).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROFILES_PATH = Path(__file__).with_name("hw_profiles.json")

# Base shape. None means "fall back to the existing env var / LearnedParams
# value" so an unset HW_PROFILE, or a key a profile omits, is a no-op.
PROFILE_DEFAULTS: dict[str, Any] = {
    "device": None,        # xpu | cuda | cpu -> device.py; None = use CV_DEVICE
    "precision": "fp16",   # fp16 | fp32 -> pose OpenVINO export + TrackNet
    "openvino": True,      # False on non-Intel profiles
    "tracknet": None,      # None -> use TRACKNET_ENABLED env
    "pass2_ball": None,    # None -> use PASS2_BALL env
    # ponytail: pose inference is frame-serial today (players.py PoseTracker
    # calls model.track() one frame at a time) - no batch path to plug into.
    # Kept as schema so a future batched Pass 2 loop has somewhere to read
    # from; add real wiring when someone measures a GPU-bound pose stage.
    "pose_batch": 1,
    "p1_scale": None,        # None -> keep the LearnedParams value
    "p1_frame_step": None,   # None -> keep the LearnedParams value
}


@functools.lru_cache(maxsize=1)
def active_profile() -> dict[str, Any] | None:
    """Resolved profile dict, or None if HW_PROFILE is unset (current .env
    behavior applies unchanged everywhere)."""
    name = os.getenv("HW_PROFILE", "").strip()
    if not name:
        return None
    profiles = json.loads(PROFILES_PATH.read_text())
    if name not in profiles:
        raise SystemExit(
            f"HW_PROFILE={name!r} not found in {PROFILES_PATH.name} "
            f"(known profiles: {sorted(profiles)})")
    return {**PROFILE_DEFAULTS, **profiles[name]}

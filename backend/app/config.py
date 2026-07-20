"""Central configuration. Loads .env from the repo root."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .hw_profile import active_profile
from .paths import repo_root
from .secrets_bootstrap import load_anthropic_key

REPO_ROOT = repo_root()
load_dotenv(REPO_ROOT / ".env")
load_anthropic_key()

DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data"))
VIDEO_DIR = DATA_DIR / "videos"
DB_PATH = DATA_DIR / "app.db"

# local (default, unchanged behavior) | s3 (EC2 deployment: uploads go to S3,
# triggering the s3:ObjectCreated -> instance-start pipeline). See storage_s3.py.
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")
S3_VIDEO_BUCKET = os.getenv("S3_VIDEO_BUCKET", "")

# Whether THIS container is allowed to run Pass 1/2 itself. Default True —
# unchanged behavior for plain local dev and for the EC2 worker (which must
# still process what it downloads). Set RUN_PROCESSING=0 on a UI-only
# container (STORAGE_BACKEND=s3, e.g. the developer's Mac in the hybrid
# setup): it uploads to S3 and displays/syncs results, but never calls
# pipeline.start_processing/reprocess — that's left entirely to the EC2
# worker instance woken by the S3 upload trigger. See api/media.py's
# upload_video/reprocess (the two gated call sites) and pipeline.py's
# seed_cloud_placeholder/mirror_cloud_chain_status (how a RUN_PROCESSING=0
# container still shows meaningful job status despite never running one).
RUN_PROCESSING = os.getenv("RUN_PROCESSING", "1").strip().lower() not in ("0", "false", "no", "")

if not RUN_PROCESSING and STORAGE_BACKEND != "s3":
    # Nothing else in this deployment would ever process a locally-stored
    # video, so this combination silently strands every upload forever.
    logging.getLogger("config").warning(
        "RUN_PROCESSING=0 with STORAGE_BACKEND=%r — uploaded videos will "
        "never be processed by anything (RUN_PROCESSING=0 only makes sense "
        "with STORAGE_BACKEND=s3, relying on a separate cloud worker)",
        STORAGE_BACKEND)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
# The discrepancy report is a bounded "explain + route these discrepancies"
# task — the actual parameter math is already done deterministically in
# learning.updates before the LLM is ever called, so the model only writes the
# explanatory markdown and the occasional suggestion. That's a Haiku task, and
# using it (with a capped thinking budget, see reporter.py) cuts report cost
# several-fold vs Sonnet. Override with ANTHROPIC_REPORT_MODEL if you want the
# heavier model for a particularly gnarly clip.
ANTHROPIC_REPORT_MODEL = os.getenv("ANTHROPIC_REPORT_MODEL", "claude-haiku-4-5-20251001")

CV_DEVICE = os.getenv("CV_DEVICE", "auto")
# YOLO-pose weights for player detection/tracking. Nano is the default: MEASURED
# on clip #1 that yolo11s/m give no extra far-side recall at 2-4x the cost (see
# learning/params.py pose_imgsz note). Override to try a heavier model without a
# code change, e.g. POSE_MODEL=yolo11s-pose.pt.
POSE_MODEL = os.getenv("POSE_MODEL", "yolo11n-pose.pt")
TRACKNET_WEIGHTS = os.getenv("TRACKNET_WEIGHTS", "")
# TrackNet inference is ~5x slower per Pass 2 frame, and the pretrained
# tennis/badminton weights do NOT transfer well to this fixed down-the-line
# pickleball view without fine-tuning (measured: no end-detection gain on real
# footage). So the installed default weights are OFF unless you opt in with
# TRACKNET_ENABLED=1. An explicit TRACKNET_WEIGHTS path always wins (see
# ball_tracknet.weights_path). The real payoff is a FINE-TUNED model later.
# An explicit env var always wins over the hardware profile default below.
_HW_PROFILE = active_profile()
_tracknet_env = os.getenv("TRACKNET_ENABLED")
if _tracknet_env is not None:
    TRACKNET_ENABLED = _tracknet_env.strip().lower() not in ("0", "false", "no", "")
elif _HW_PROFILE and _HW_PROFILE["tracknet"] is not None:
    TRACKNET_ENABLED = bool(_HW_PROFILE["tracknet"])
else:
    TRACKNET_ENABLED = False

# Pass 2 ball detection/tracking. DISABLED by default right now: rally boundaries
# come solely from Pass 1 and Pass 2's ball outputs are display-only (detect_events
# runs model=None), so turning the ball off breaks nothing while player detection
# is the focus — and it drops the per-frame motion/HSV ball detect + trajectory
# select + per-window Kalman/height/event reasoning (~10 ms/frame, ~11% of Pass 2).
# Set PASS2_BALL=1 to bring it back. Same env-wins-over-profile precedence.
_pass2_ball_env = os.getenv("PASS2_BALL")
if _pass2_ball_env is not None:
    PASS2_BALL_ENABLED = _pass2_ball_env.strip().lower() not in ("0", "false", "no", "")
elif _HW_PROFILE and _HW_PROFILE["pass2_ball"] is not None:
    PASS2_BALL_ENABLED = bool(_HW_PROFILE["pass2_ball"])
else:
    PASS2_BALL_ENABLED = False

# fp16 | fp32 for pose (OpenVINO export) + TrackNet inference; whether to try
# the OpenVINO GPU pose path at all. Both hardware-profile-only (no plain env
# var today) since they're per-GPU, not per-run, choices.
HW_PRECISION = _HW_PROFILE["precision"] if _HW_PROFILE else "fp16"
HW_OPENVINO_ENABLED = _HW_PROFILE["openvino"] if _HW_PROFILE else True

for _d in (DATA_DIR, VIDEO_DIR):
    _d.mkdir(parents=True, exist_ok=True)

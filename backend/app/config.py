"""Central configuration. Loads .env from the repo root."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data"))
VIDEO_DIR = DATA_DIR / "videos"
DB_PATH = DATA_DIR / "app.db"

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
TRACKNET_ENABLED = os.getenv("TRACKNET_ENABLED", "0").strip().lower() not in ("0", "false", "no", "")

# Pass 2 ball detection/tracking. DISABLED by default right now: rally boundaries
# come solely from Pass 1 and Pass 2's ball outputs are display-only (detect_events
# runs model=None), so turning the ball off breaks nothing while player detection
# is the focus — and it drops the per-frame motion/HSV ball detect + trajectory
# select + per-window Kalman/height/event reasoning (~10 ms/frame, ~11% of Pass 2).
# Set PASS2_BALL=1 to bring it back.
PASS2_BALL_ENABLED = os.getenv("PASS2_BALL", "0").strip().lower() not in ("0", "false", "no", "")

for _d in (DATA_DIR, VIDEO_DIR):
    _d.mkdir(parents=True, exist_ok=True)

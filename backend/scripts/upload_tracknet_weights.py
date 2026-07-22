"""Uploads a (presumably fine-tuned) TrackNet checkpoint to S3 so the EC2
worker's cloud_ingest poller picks it up on its next tick (~20s) and mirrors
it into its own data/models/tracknet_v2.pt — same handoff as calibration
(see storage_s3.upload_calibration / cloud_ingest._sync_tracknet_weights).

Run this from wherever the fine-tuned weights actually live (e.g. after
training locally), against the same S3_VIDEO_BUCKET the EC2 worker reads.

Usage:
    .venv/bin/python backend/scripts/upload_tracknet_weights.py [path/to/weights.pt]
    (defaults to data/models/tracknet_v2.pt)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import storage_s3  # noqa: E402
from app.config import DATA_DIR  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "models" / "tracknet_v2.pt"
    if not path.exists():
        sys.exit(f"no such file: {path}")
    key = storage_s3.upload_tracknet_weights(path)
    print(f"OK — uploaded {path} ({path.stat().st_size} bytes) to "
          f"s3://{storage_s3.S3_VIDEO_BUCKET}/{key}")
    print("The EC2 worker will pick it up on its next poll tick (~20s) if it's running.")


if __name__ == "__main__":
    main()

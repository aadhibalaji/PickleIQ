"""Stop this EC2 instance once the job queue runs dry, so GPU time isn't
billed while idle between batches.

Mirrors secrets_bootstrap.py's pattern: everything here — the IMDSv2 token
fetch, the instance-id lookup, and the stop_instances() call itself — sits
inside one try/except. A transient metadata-service or EC2 API hiccup must
never propagate into the caller (the job runner's main loop); it just logs
and no-ops, same as running locally or on RunPod where the metadata service
at 169.254.169.254 doesn't exist at all.
"""
from __future__ import annotations

import json
import logging
import os

import requests

from .config import DATA_DIR, SELF_STOP_ENABLED_DEFAULT

log = logging.getLogger("self_stop")

_METADATA_BASE = "http://169.254.169.254/latest"
_TIMEOUT_S = 2.0

# Testing/ops toggle: lets someone disable the automatic stop-on-empty-queue
# behavior during active testing, without touching manual start/stop or
# anything else in this module. Persisted to a local file (this process's own
# state) rather than the DB — it's a single flag read on the hot path right
# before an instance-stop decision, not data that needs querying/joining.
# On an EC2 worker, this local file is kept in sync from S3 by
# cloud_ingest._sync_self_stop on the same poll cycle already used for
# calibration/refine-config sync (see storage_s3.upload_self_stop_state) —
# the toggle is normally flipped from a separate UI container that has no
# other way to reach this instance's local state.
_STATE_PATH = DATA_DIR / "self_stop_state.json"


def get_enabled() -> bool:
    if not _STATE_PATH.exists():
        return SELF_STOP_ENABLED_DEFAULT
    try:
        return bool(json.loads(_STATE_PATH.read_text())["enabled"])
    except Exception:
        log.warning("self_stop_state.json unreadable, defaulting to %s", SELF_STOP_ENABLED_DEFAULT)
        return SELF_STOP_ENABLED_DEFAULT


def set_enabled(enabled: bool) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps({"enabled": bool(enabled)}))


def stop_this_instance() -> None:
    if not get_enabled():
        log.info("self-stop skipped: disabled via toggle (self_stop_enabled=false)")
        return
    try:
        token = requests.put(
            f"{_METADATA_BASE}/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=_TIMEOUT_S,
        ).text

        instance_id = requests.get(
            f"{_METADATA_BASE}/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=_TIMEOUT_S,
        ).text

        import boto3
        client = boto3.client("ec2", region_name=os.environ.get("AWS_REGION", "us-east-2"))
        client.stop_instances(InstanceIds=[instance_id])
        log.info("stop_instances requested for %s (job queue empty)", instance_id)
    except Exception as exc:
        # No metadata service (local dev, RunPod), no network, IAM missing
        # ec2:StopInstances, transient API error — any of these just skips
        # the self-stop; the job runner keeps running either way.
        log.info("self-stop skipped/failed (%s) — not on EC2 or stop_instances errored", exc)

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

import logging
import os

import requests

log = logging.getLogger("self_stop")

_METADATA_BASE = "http://169.254.169.254/latest"
_TIMEOUT_S = 2.0


def stop_this_instance() -> None:
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

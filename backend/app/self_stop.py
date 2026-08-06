"""Stop this EC2 instance once work is done, so GPU time isn't billed while
idle between batches — with a Pass-1→Pass-2 grace so the human-in-the-loop
selection step doesn't force a cold start.

Mirrors secrets_bootstrap.py's pattern: everything in stop_this_instance —
the IMDSv2 token fetch, the instance-id lookup, and the stop_instances()
call itself — sits inside one try/except. A transient metadata-service or
EC2 API hiccup must never propagate into the caller (the job runner's main
loop); it just logs and no-ops, same as running locally or on RunPod where
the metadata service at 169.254.169.254 doesn't exist at all.

Toggle semantics (see should_auto_stop):
  ON  — smart stop: stay up after Pass 1 for SELF_STOP_PASS1_GRACE_S so the
        user can trigger Pass 2, and after Pass 2 for SELF_STOP_COACHING_
        GRACE_S so the user can request a coaching report; stop when the
        queue is empty and nothing is in either grace window.
  OFF — a real kill switch: never stops, full stop. Checked first, before
        any grace-window logic even runs — e.g. for demo days, where the
        ~4 min cold start between runs is worse than just leaving the GPU
        billing while idle.
"""
from __future__ import annotations

import json
import logging
import os
import time

import requests

from .config import (DATA_DIR, SELF_STOP_COACHING_GRACE_S, SELF_STOP_ENABLED_DEFAULT,
                     SELF_STOP_PASS1_GRACE_S)

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


def _pass2_terminal(status: str | None) -> bool:
    return status in ("done", "error", "cancelled")


def _awaiting_pass2_state() -> tuple[bool, bool]:
    """Returns (any_within_grace, any_grace_expired).

    A video is 'awaiting Pass 2' when its latest pass1 job is done and its
    latest pass2 job is missing or not yet terminal. The grace clock starts
    at pass1's updated_at (when it flipped to done)."""
    from . import db  # local import: self_stop is loaded early from jobs.py

    now = time.time()
    within = False
    expired = False
    with db.connect() as con:
        videos = con.execute("SELECT id FROM videos").fetchall()
        for v in videos:
            vid = v["id"]
            p1 = con.execute(
                "SELECT status, updated_at FROM jobs WHERE video_id=? AND kind='pass1'"
                " ORDER BY created_at DESC LIMIT 1", (vid,)).fetchone()
            if p1 is None or p1["status"] != "done":
                continue
            p2 = con.execute(
                "SELECT status FROM jobs WHERE video_id=? AND kind='pass2'"
                " ORDER BY created_at DESC LIMIT 1", (vid,)).fetchone()
            if p2 is not None and _pass2_terminal(p2["status"]):
                continue
            if p2 is not None and p2["status"] in ("queued", "running"):
                # Real pending work — jobs._has_pending_work should already
                # block the stop; treat as within-grace so we never race it.
                within = True
                continue
            age = now - float(p1["updated_at"] or 0)
            if age <= SELF_STOP_PASS1_GRACE_S:
                within = True
            else:
                expired = True
    return within, expired


def _awaiting_coaching_report_state() -> tuple[bool, bool]:
    """Returns (any_within_grace, any_grace_expired) — same shape and same
    reasoning as _awaiting_pass2_state, one stage later: a video is
    'awaiting a coaching report' when its latest pass2 job is done and no
    coaching_reports row exists for it yet. Generating that report is a
    manual, synchronous API call (see api/rally_analysis.py's
    generate_coaching_report) — never a queued job — so nothing in the
    job-queue view jobs.py checks otherwise knows to wait for it. Without
    this, should_auto_stop fires the moment the queue empties right after
    Pass 2, before the user has had any chance to click "Generate report"
    (only relevant with the toggle ON — OFF is a real kill switch, see this
    module's docstring). The grace clock starts at pass2's updated_at (when
    it flipped terminal)."""
    from . import db  # local import: self_stop is loaded early from jobs.py

    now = time.time()
    within = False
    expired = False
    with db.connect() as con:
        videos = con.execute("SELECT id FROM videos").fetchall()
        for v in videos:
            vid = v["id"]
            p2 = con.execute(
                "SELECT status, updated_at FROM jobs WHERE video_id=? AND kind='pass2'"
                " ORDER BY created_at DESC LIMIT 1", (vid,)).fetchone()
            if p2 is None or p2["status"] != "done":
                continue
            has_report = con.execute(
                "SELECT 1 FROM coaching_reports WHERE video_id=?", (vid,)).fetchone()
            if has_report:
                continue
            age = now - float(p2["updated_at"] or 0)
            if age <= SELF_STOP_COACHING_GRACE_S:
                within = True
            else:
                expired = True
    return within, expired


def _pending_pass2_s3_requests() -> bool:
    """True if a UI container has asked for Pass 2 but this worker hasn't
    ingested the request yet — queue still looks empty during that gap."""
    try:
        from . import storage_s3
        from .config import STORAGE_BACKEND
        if STORAGE_BACKEND != "s3":
            return False
        return bool(storage_s3.list_pass2_request_video_ids())
    except Exception:
        log.exception("self-stop: pass2-request S3 list failed (assuming pending to be safe)")
        return True


def should_auto_stop() -> bool:
    """Decide whether an empty job queue means we should stop the instance.

    Caller (jobs.JobRunner._loop) only invokes this after confirming the
    queue/DB show no queued/running work."""
    if not get_enabled():
        # Real kill switch, checked before any grace-window logic even
        # runs — OFF means never, full stop (e.g. demo days: staying up
        # idle is cheaper than the ~4 min cold start between runs).
        log.info("self-stop skipped: disabled via toggle")
        return False

    if _pending_pass2_s3_requests():
        log.info("self-stop deferred: pending pass2 S3 work request")
        return False

    within_grace, _ = _awaiting_pass2_state()
    if within_grace:
        log.info("self-stop deferred: awaiting Pass 2 after Pass 1 (grace ≤%ss)",
                 SELF_STOP_PASS1_GRACE_S)
        return False

    within_coaching_grace, _ = _awaiting_coaching_report_state()
    if within_coaching_grace:
        log.info("self-stop deferred: awaiting coaching report after Pass 2 (grace ≤%ss)",
                 SELF_STOP_COACHING_GRACE_S)
        return False

    return True


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

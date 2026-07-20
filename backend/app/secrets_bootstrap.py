"""Fetch ANTHROPIC_API_KEY from AWS Secrets Manager when nothing has already
set it in the environment.

Covers the EC2 deployment path: the instance role has read access to
picklecv/anthropic-api-key, so boto3 picks up credentials automatically via
IMDS — nothing to pass in. RunPod pods set ANTHROPIC_API_KEY directly as a
pod env var, and local dev sets it via .env; both short-circuit the `if`
below, so this only ever does anything on a bare EC2 boot. Must run before
config.py reads ANTHROPIC_API_KEY, and after load_dotenv() so .env still
wins locally.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("secrets_bootstrap")

SECRET_ID = os.getenv("ANTHROPIC_SECRET_ID", "picklecv/anthropic-api-key")
SECRET_REGION = os.getenv("AWS_REGION", "us-east-2")


def load_anthropic_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        import boto3
        client = boto3.client("secretsmanager", region_name=SECRET_REGION)
        secret = client.get_secret_value(SecretId=SECRET_ID)["SecretString"]
    except Exception as exc:
        # No boto3 (local dev), no instance role (RunPod/local), no network,
        # secret missing/renamed — any of these just leaves the key unset,
        # same as today: everything except the feedback report still works.
        log.info("ANTHROPIC_API_KEY unset and Secrets Manager fetch unavailable (%s); "
                  "feedback-report feature will be disabled until a key is set.", exc)
        return
    os.environ["ANTHROPIC_API_KEY"] = secret

#!/bin/bash
# Paste-and-run ON the EC2 worker instance to restart it against a freshly-
# pulled image. Not for production — just cuts down on manual retyping while
# rebuilding and restarting constantly during hybrid-architecture testing.
#
# Settings this locks in: STORAGE_BACKEND=s3, RUN_PROCESSING left at its
# default (1) — this container IS the compute worker (cloud_ingest.py polls
# S3 for new uploads and runs Pass 1/2 locally on this box). --gpus all for
# real CUDA, PASS2_BALL=1 to enable ball detection/tracking in Pass 2.
#
# DATA_DIR is bind-mounted to a path on this instance's root EBS volume (NOT
# /tmp or any instance-store-backed path) specifically so it survives both
# `docker rm`+recreate AND a full instance stop/start — this instance's own
# jobs/videos DB is what cloud_ingest.py's dedup relies on to avoid
# re-processing an already-completed video after a restart (see
# cloud_ingest.py's results_exist() guard, which is a second, independent
# safety net on top of this, not a substitute for it).
#
# --restart unless-stopped: survives an unexpected instance reboot without
# needing this script re-run by hand — Docker itself brings the container
# back up on boot. Requires dropping --rm (Docker won't let a container have
# both a restart policy and auto-remove-on-exit at the same time); this
# script already does its own `docker rm -f` before each run, so the
# container not self-removing on a normal stop doesn't leave anything behind.
set -euo pipefail

IMAGE="051404302327.dkr.ecr.us-east-2.amazonaws.com/picklecv-pipeline:latest"
CONTAINER_NAME="picklecv-worker"
PORT="8420"
DATA_DIR="${PICKLECV_DATA_DIR:-/opt/picklecv/data}"
BUCKET="${S3_VIDEO_BUCKET:-picklecv-videos-dev}"
REGION="${AWS_REGION:-us-east-2}"

sudo mkdir -p "$DATA_DIR"

echo "==> logging in to ECR (uses this instance's IAM role)"
aws ecr get-login-password --region "$REGION" \
  | sudo docker login --username AWS --password-stdin 051404302327.dkr.ecr.us-east-2.amazonaws.com

echo "==> removing any existing $CONTAINER_NAME"
sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "==> pulling $IMAGE"
sudo docker pull "$IMAGE"

echo "==> starting $CONTAINER_NAME"
sudo docker run -d --restart unless-stopped --gpus all -p "$PORT:8420" \
  -e STORAGE_BACKEND=s3 \
  -e S3_VIDEO_BUCKET="$BUCKET" \
  -e AWS_REGION="$REGION" \
  -e CV_DEVICE=cuda \
  -e PASS2_BALL=1 \
  -v "$DATA_DIR:/app/data" \
  --name "$CONTAINER_NAME" \
  "$IMAGE"

echo "==> done. API at http://localhost:$PORT — logs: sudo docker logs -f $CONTAINER_NAME"

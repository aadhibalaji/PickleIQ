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
# TRACKNET_ENABLED=1: a fine-tuned checkpoint for this camera setup now syncs
# from S3 the same way calibration does — cloud_ingest._sync_tracknet_weights
# downloads it to $DATA_DIR/models/tracknet_v2.pt on the regular poll tick, no
# TRACKNET_WEIGHTS override needed (that IS ball_tracknet.py's default path).
# Upload a new/updated checkpoint with scripts/upload_tracknet_weights.py.
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

# DLAMI points Docker's data-root at the ephemeral NVMe
# (/opt/dlami/nvme/docker). Instance stop/start wipes that volume, so the
# mount comes back empty and `docker run` fails with:
#   mkdir /opt/dlami/nvme/docker/containers/<id>: no such file or directory
# Recreate the data-root and bounce dockerd whenever it's missing — must
# run before any docker login/pull/run below. deploy.sh always stop/starts
# the box when it was stopped, so this hits essentially every cold deploy.
_ensure_docker_data_root() {
  local data_root="/opt/dlami/nvme/docker"
  if [[ -f /etc/docker/daemon.json ]] && command -v python3 >/dev/null 2>&1; then
    local parsed
    parsed="$(python3 -c 'import json; print(json.load(open("/etc/docker/daemon.json")).get("data-root",""))' 2>/dev/null || true)"
    if [[ -n "${parsed}" ]]; then
      data_root="$parsed"
    fi
  fi
  if [[ ! -d "$data_root" ]]; then
    echo "==> recreating Docker data-root at $data_root (wiped by instance stop/start)"
    sudo mkdir -p "$data_root"
    sudo systemctl restart docker
    # dockerd needs a moment before login/pull work again
    sleep 3
  elif ! sudo docker info >/dev/null 2>&1; then
    echo "==> Docker daemon unhealthy — restarting"
    sudo systemctl restart docker
    sleep 3
  fi
}
_ensure_docker_data_root

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
  -e TRACKNET_ENABLED=1 \
  -v "$DATA_DIR:/app/data" \
  --name "$CONTAINER_NAME" \
  "$IMAGE"

echo "==> done. API at http://localhost:$PORT — logs: sudo docker logs -f $CONTAINER_NAME"

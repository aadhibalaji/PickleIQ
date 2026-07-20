#!/bin/bash
# Restart the LOCAL (UI-only) container against a freshly-pulled image.
# Not for production — just cuts down on manual retyping while rebuilding
# and restarting constantly during hybrid-architecture testing.
#
# Settings this locks in: STORAGE_BACKEND=s3 (uploads go to S3, not local
# disk), RUN_PROCESSING=0 (this container NEVER runs Pass 1/2 itself — see
# config.py's docstring on RUN_PROCESSING — it only uploads, displays, and
# syncs cloud results; the EC2 worker does all the compute).
set -euo pipefail

IMAGE="051404302327.dkr.ecr.us-east-2.amazonaws.com/picklecv-pipeline:latest"
CONTAINER_NAME="picklecv-local"
PORT="8420"
DATA_DIR="${PICKLECV_LOCAL_DATA:-$HOME/picklecv-local-data}"
BUCKET="${S3_VIDEO_BUCKET:-picklecv-videos-dev}"
REGION="${AWS_REGION:-us-east-2}"

mkdir -p "$DATA_DIR"

echo "==> logging in to ECR"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin 051404302327.dkr.ecr.us-east-2.amazonaws.com

echo "==> removing any existing $CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "==> pulling $IMAGE"
docker pull --platform linux/amd64 "$IMAGE"

echo "==> starting $CONTAINER_NAME"
docker run -d --rm --platform linux/amd64 -p "$PORT:8420" \
  -e STORAGE_BACKEND=s3 \
  -e S3_VIDEO_BUCKET="$BUCKET" \
  -e AWS_REGION="$REGION" \
  -e RUN_PROCESSING=0 \
  -v "$HOME/.aws:/root/.aws:ro" \
  -v "$DATA_DIR:/app/data" \
  --name "$CONTAINER_NAME" \
  "$IMAGE"

echo "==> done. API at http://localhost:$PORT — logs: docker logs -f $CONTAINER_NAME"

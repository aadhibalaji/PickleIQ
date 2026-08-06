#!/bin/bash
# Run this from the repo root any time you've changed backend/ code and want
# it live on both the local (UI-only) container and the EC2 GPU worker.
#
# What bit us once already, both baked in here so it can't happen again:
#   1. `docker build` with no -t just makes an orphaned, unpushed image —
#      always build straight into the final tag.
#   2. This repo is developed on Apple Silicon (arm64) but EC2 + the CUDA
#      base image are amd64-only — a plain `docker build` defaults to the
#      HOST platform and produces an image that silently never runs anywhere
#      real. Always pass --platform linux/amd64 explicitly.
#   3. A bad merge can delete a function's DEFINITION while leaving its CALL
#      SITE untouched — git's 3-way merge doesn't always flag that as a
#      conflict if the surrounding lines look unmodified on the losing side.
#      That code then imports fine (Python doesn't check names inside a
#      function body until it actually runs) and passes `docker build`
#      clean, so nothing catches it until the exact code path executes in
#      production — which is exactly how cloud_ingest.py's Pass 2 poller
#      thread silently died on every worker boot for two merges running.
#      pyflakes' undefined-name check catches this class of bug in under a
#      second, so it's a hard gate below, before any image gets built.
# Verification at the end is generic (compares the image ID actually running
# in each container against the ID we just pushed) rather than grepping for
# any specific fix, so it stays useful for whatever you changed this time.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> checking for undefined names (catches a merge that drops a function"
echo "    body but leaves its call site — see point 3 above)"
UNDEFINED="$(backend/.venv/bin/python -m pyflakes backend/app 2>&1 | grep -i "undefined name" || true)"
if [ -n "$UNDEFINED" ]; then
  echo "!! pyflakes found undefined names — refusing to deploy:" >&2
  echo "$UNDEFINED" >&2
  exit 1
fi
echo "    OK — no undefined names"

IMAGE="051404302327.dkr.ecr.us-east-2.amazonaws.com/picklecv-pipeline:latest"
REGION="us-east-2"
INSTANCE_ID="i-0daa6a28d07139de9"   # pickleCvEC2
SSH_KEY="$HOME/.ssh/picklecv/picklecv-key.pem"
SSH_USER="ubuntu"

echo "==> logging in to ECR"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${IMAGE%%/*}"

echo "==> building (linux/amd64 — host is likely arm64, EC2/CUDA is not)"
docker build --platform linux/amd64 -t "$IMAGE" .
NEW_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
echo "    built $NEW_ID"

echo "==> pushing"
docker push "$IMAGE"

echo "==> restarting local container"
./scripts/restart-local.sh
LOCAL_ID="$(docker inspect picklecv-local --format '{{.Image}}')"
if [ "$LOCAL_ID" != "$NEW_ID" ]; then
  echo "!! local container is running $LOCAL_ID, expected $NEW_ID — restart-local.sh may have failed" >&2
  exit 1
fi
echo "    OK — picklecv-local is on the image we just pushed"

echo "==> ensuring EC2 instance is running"
STATE="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)"
if [ "$STATE" != "running" ]; then
  echo "    instance is '$STATE' — starting it"
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" > /dev/null
  aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
  # wait a bit more for sshd, not just the EC2 API's notion of "running"
  sleep 20
fi
IP="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "    instance at $IP"

echo "==> restarting EC2 worker container"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
  scripts/restart-ec2.sh "$SSH_USER@$IP:/tmp/restart-ec2.sh"

# Installs restart-ec2.sh to a permanent path (survives the ephemeral NVMe
# wipe a stop/start does to /opt/dlami/nvme — /opt/picklecv lives on the
# persistent root volume instead) plus a systemd unit that runs it on every
# boot. Without this, self-stop followed by a Lambda-triggered start leaves
# picklecv-worker permanently missing until someone manually re-runs this
# script over SSH — see restart-ec2.sh's _ensure_docker_data_root comment
# for the full story. Idempotent and safe to re-run every deploy; also
# self-heals a brand-new/replaced instance with no separate setup step.
ssh -i "$SSH_KEY" "$SSH_USER@$IP" '
  sudo mkdir -p /opt/picklecv
  sudo cp /tmp/restart-ec2.sh /opt/picklecv/restart-ec2.sh
  sudo chmod +x /opt/picklecv/restart-ec2.sh
  if [ ! -f /etc/systemd/system/picklecv-worker.service ]; then
    cat <<EOF | sudo tee /etc/systemd/system/picklecv-worker.service > /dev/null
[Unit]
Description=PickleCV worker container bootstrap (recreates picklecv-worker on every boot)
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/opt/picklecv/restart-ec2.sh
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable picklecv-worker.service
  fi
'
ssh -i "$SSH_KEY" "$SSH_USER@$IP" "bash /tmp/restart-ec2.sh"

WORKER_ID="$(ssh -i "$SSH_KEY" "$SSH_USER@$IP" "docker inspect picklecv-worker --format '{{.Image}}'")"
if [ "$WORKER_ID" != "$NEW_ID" ]; then
  echo "!! picklecv-worker is running $WORKER_ID, expected $NEW_ID — restart-ec2.sh may have failed" >&2
  exit 1
fi
echo "    OK — picklecv-worker is on the image we just pushed"

echo "==> done. Both containers confirmed running $NEW_ID"

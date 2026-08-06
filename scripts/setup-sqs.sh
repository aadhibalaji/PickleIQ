#!/bin/bash
# One-time setup for the SQS job-handoff fast path (see cloud_ingest.py's
# module docstring for why this exists). Run from your Mac with your AWS
# creds. Creates the queue, wires the bucket notification, adds the new IAM
# permissions the worker needs (also fixes the missing s3:DeleteObject that
# was silently breaking Past Clips deletes), and patches restart-ec2.sh with
# the resulting queue URL. Then deploy as usual.
#
# Safe to re-run: create-queue is idempotent by name, the rest are plain
# overwrites.
set -euo pipefail
cd "$(dirname "$0")"

REGION="us-east-2"
BUCKET="picklecv-videos-dev"
ACCOUNT_ID="051404302327"
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/PickleCV-EC2-InstancePolicy"

echo "==> 1. creating queue"
QUEUE_URL="$(aws sqs create-queue --queue-name picklecv-job-events --region "$REGION" \
  --attributes VisibilityTimeout=60 --query QueueUrl --output text)"
QUEUE_ARN="$(aws sqs get-queue-attributes --queue-url "$QUEUE_URL" --attribute-names QueueArn \
  --region "$REGION" --query Attributes.QueueArn --output text)"
echo "    queue: $QUEUE_URL"
echo "    arn:   $QUEUE_ARN"

echo "==> 2. letting the bucket publish to it"
aws sqs set-queue-attributes --queue-url "$QUEUE_URL" --region "$REGION" --attributes '{
  "Policy": "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"s3.amazonaws.com\"},\"Action\":\"SQS:SendMessage\",\"Resource\":\"'"$QUEUE_ARN"'\",\"Condition\":{\"ArnLike\":{\"aws:SourceArn\":\"arn:aws:s3:::'"$BUCKET"'\"}}}]}"
}'

echo "==> 3. wiring the three bucket notification rules"
aws s3api put-bucket-notification-configuration --bucket "$BUCKET" --region "$REGION" \
  --notification-configuration '{
    "QueueConfigurations": [
      {"QueueArn": "'"$QUEUE_ARN"'", "Events": ["s3:ObjectCreated:*"],
       "Filter": {"Key": {"FilterRules": [{"Name": "prefix", "Value": "default/"}, {"Name": "suffix", "Value": ".mp4"}]}}},
      {"QueueArn": "'"$QUEUE_ARN"'", "Events": ["s3:ObjectCreated:*"],
       "Filter": {"Key": {"FilterRules": [{"Name": "prefix", "Value": "jobs/"}, {"Name": "suffix", "Value": "pass2_request.json"}]}}},
      {"QueueArn": "'"$QUEUE_ARN"'", "Events": ["s3:ObjectCreated:*"],
       "Filter": {"Key": {"FilterRules": [{"Name": "prefix", "Value": "jobs/"}, {"Name": "suffix", "Value": "kill_request.json"}]}}}
    ]}'

echo "==> 4. updating the EC2 role: + s3:DeleteObject, + SQS receive/delete/get-attrs on the new queue"
POLICY_JSON="$(mktemp)"
cat > "$POLICY_JSON" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3VideoAccess",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/*"
    },
    {
      "Sid": "S3ListBucket",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${BUCKET}"
    },
    {
      "Sid": "ReadAnthropicSecret",
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:picklecv/anthropic-api-key-D4kDvB"
    },
    {
      "Sid": "SelfStop",
      "Effect": "Allow",
      "Action": "ec2:StopInstances",
      "Resource": "*",
      "Condition": {"StringEquals": {"ec2:ResourceTag/Project": "PickleCV"}}
    },
    {
      "Sid": "ECRPull",
      "Effect": "Allow",
      "Action": ["ecr:GetAuthorizationToken"],
      "Resource": "*"
    },
    {
      "Sid": "ECRPullRepo",
      "Effect": "Allow",
      "Action": ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
      "Resource": "arn:aws:ecr:${REGION}:${ACCOUNT_ID}:repository/picklecv-pipeline"
    },
    {
      "Sid": "JobQueueAccess",
      "Effect": "Allow",
      "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"],
      "Resource": "${QUEUE_ARN}"
    }
  ]
}
EOF
# IAM keeps at most 5 versions per policy — if this fails with
# LimitExceeded, delete the oldest non-default version first:
#   aws iam list-policy-versions --policy-arn "$POLICY_ARN"
#   aws iam delete-policy-version --policy-arn "$POLICY_ARN" --version-id vN
aws iam create-policy-version --policy-arn "$POLICY_ARN" --set-as-default --policy-document "file://$POLICY_JSON"
rm -f "$POLICY_JSON"

echo "==> 5. pointing restart-ec2.sh at the queue"
sed -i '' "s#SQS_JOB_QUEUE_URL=\"\${SQS_JOB_QUEUE_URL:-}\"#SQS_JOB_QUEUE_URL=\"\${SQS_JOB_QUEUE_URL:-$QUEUE_URL}\"#" \
  restart-ec2.sh

echo
echo "==> done. Queue: $QUEUE_URL"
echo "    Next: ./deploy.sh   (run from scripts/, or repo root's scripts/deploy.sh)"

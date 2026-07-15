#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 KEY_FILE EC2_HOST" >&2
  exit 2
fi

KEY_FILE="$1"
EC2_HOST="$2"
EC2_USER="ec2-user"
REMOTE_DIR="/home/ec2-user/rsna"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH=(ssh -i "$KEY_FILE" -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
RSYNC_SSH="ssh -i $KEY_FILE -o ServerAliveInterval=30 -o ServerAliveCountMax=10"

chmod 400 "$KEY_FILE"

echo "[1/2] Uploading the best-model export script..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/export_best_071_report.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "[2/2] Locking checkpoints, predictions, settings, metadata, and code snapshot..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && /opt/pytorch/bin/python -u export_best_071_report.py"

cat <<EOF

The new best validation-selected model bundle is saved on EC2.

View the summary:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_best_071_locked/BEST_071_LOCKED.txt"

View the full reproducibility report:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_best_071_locked/best_model_report.json"
EOF

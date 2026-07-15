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

echo "Uploading best-model export script..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/export_best_070_report.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Exporting locked best-model bundle on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && /opt/pytorch/bin/python export_best_070_report.py"

cat <<EOF

Best-0.70 report exported on EC2.

View it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_best_070_locked/BEST_070_LOCKED.txt"

Full JSON report:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_best_070_locked/best_model_report.json"
EOF

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

echo "[1/2] Uploading the F1=0.715008 export scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/export_best_071_report.py" \
  "$LOCAL_DIR/export_best_0715_report.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "[2/2] Locking checkpoints and reproducibility artifacts..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && /opt/pytorch/bin/python -u export_best_0715_report.py"

cat <<EOF

View the locked-model summary:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_best_0715_locked/BEST_0715_LOCKED.txt"

Full report:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_best_0715_locked/best_model_report.json"
EOF

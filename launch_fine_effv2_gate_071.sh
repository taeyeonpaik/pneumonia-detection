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

echo "[1/2] Uploading fine precision-gate sweep..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/fine_precision_gate_from_features.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "[2/2] Running the validation-only fine sweep from cached features..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && /opt/pytorch/bin/python -u fine_precision_gate_from_features.py \
    --fusion-dir outputs_efficientnetv2_push_071/fusion/swin_effv2s_balanced \
    --output-dir outputs_fine_effv2_gate_071 2>&1 | tee fine_effv2_gate_071.log"

cat <<EOF

Fine gate sweep complete. View the result with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_fine_effv2_gate_071/FINE_PRECISION_GATE_COMPLETE.txt"
EOF

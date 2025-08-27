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

echo "Uploading SwinV2-Base push scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/run_swinv2_base_push_071_remote.sh" \
  "$LOCAL_DIR/train_classifier.py" \
  "$LOCAL_DIR/fusion_sweep.py" \
  "$LOCAL_DIR/testing_FRCNN.py" \
  "$LOCAL_DIR/pipeline_FRCNN.py" \
  "$LOCAL_DIR/dataset.py" \
  "$LOCAL_DIR/dicom_metadata.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting SwinV2-Base push on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && chmod +x run_swinv2_base_push_071_remote.sh && if [[ -f swinv2_base_push_071.pid ]] && kill -0 \$(cat swinv2_base_push_071.pid) 2>/dev/null; then echo already_running_pid=\$(cat swinv2_base_push_071.pid); else nohup ./run_swinv2_base_push_071_remote.sh > swinv2_base_push_071.log 2>&1 < /dev/null & echo \$! > swinv2_base_push_071.pid; echo started_pid=\$(cat swinv2_base_push_071.pid); fi"

cat <<EOF

SwinV2-Base push is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f swinv2_base_push_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_swinv2_base_push_071/SWINV2_BASE_PUSH_071_COMPLETE.txt || ps -p \$(cat swinv2_base_push_071.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_swinv2_base_push_071/swinv2_base_push_071_summary.csv | head -30"
EOF

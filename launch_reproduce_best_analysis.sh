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

if [[ ! -f "$KEY_FILE" ]]; then
  echo "Missing key file: $KEY_FILE" >&2
  exit 1
fi

chmod 400 "$KEY_FILE"

echo "Testing SSH..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "echo 'SSH verified on ' \$(hostname)"

echo "Uploading/resuming project code..."
RSYNC_COMMAND=(
  rsync -avP
  --exclude __pycache__
  --exclude .DS_Store
  --exclude "outputs_*"
  -e "$RSYNC_SSH"
  "$LOCAL_DIR/"
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"
)
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu "${RSYNC_COMMAND[@]}"
else
  "${RSYNC_COMMAND[@]}"
fi

echo "Verifying GPU and uploaded data..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && test -f stage_2_train_labels.csv && test -d train_images_png && echo train_png_count=\$(find train_images_png -maxdepth 1 -type f | wc -l)"

echo "Installing Python dependencies..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && /opt/pytorch/bin/python -m pip install -r requirements.txt"

echo "Starting best-model reproduction + FP/FN analysis..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && \
   if [[ -f reproduce_best_analysis.pid ]] && kill -0 \$(cat reproduce_best_analysis.pid) 2>/dev/null; then \
     echo pipeline_already_running_pid=\$(cat reproduce_best_analysis.pid); \
   else \
     nohup /opt/pytorch/bin/python -u overnight_reproduce_best_and_analyze.py \
       --data-dir '$REMOTE_DIR' \
       --output-root '$REMOTE_DIR/outputs_reproduce_best_analysis' \
       > reproduce_best_analysis.log 2>&1 < /dev/null & \
     echo \$! > reproduce_best_analysis.pid; \
     echo pipeline_started_pid=\$(cat reproduce_best_analysis.pid); \
   fi"

cat <<EOF

The EC2 reproduction + error-analysis pipeline is running independently.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f reproduce_best_analysis.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && test -f outputs_reproduce_best_analysis/PIPELINE_COMPLETE && cat outputs_reproduce_best_analysis/PIPELINE_COMPLETE || ps -p \$(cat reproduce_best_analysis.pid) -o pid,etime,cmd"
EOF

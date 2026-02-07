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

echo "Uploading stacked rescue/veto scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/stacked_rescue_veto_sweep.py" \
  "$LOCAL_DIR/run_stacked_rescue_veto_071_remote.sh" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting stacked rescue/veto sweep on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && chmod +x run_stacked_rescue_veto_071_remote.sh && if [[ -f stacked_rescue_veto_071.pid ]] && kill -0 \$(cat stacked_rescue_veto_071.pid) 2>/dev/null; then echo already_running_pid=\$(cat stacked_rescue_veto_071.pid); else nohup ./run_stacked_rescue_veto_071_remote.sh > stacked_rescue_veto_071.log 2>&1 < /dev/null & echo \$! > stacked_rescue_veto_071.pid; echo started_pid=\$(cat stacked_rescue_veto_071.pid); fi"

cat <<EOF

Stacked rescue/veto sweep is running independently on EC2.

Watch live progress bars with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f stacked_rescue_veto_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_stacked_rescue_veto_071/STACKED_RESCUE_VETO_COMPLETE.txt || ps -p \$(cat stacked_rescue_veto_071.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_stacked_rescue_veto_071/stacked_rescue_veto_summary.csv | head -30"
EOF

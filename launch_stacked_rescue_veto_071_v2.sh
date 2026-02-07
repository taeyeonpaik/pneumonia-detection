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

echo "Uploading fixed stacked rescue/veto v2 scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/stacked_rescue_veto_sweep.py" \
  "$LOCAL_DIR/run_stacked_rescue_veto_071_remote.sh" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting fixed stacked rescue/veto v2 sweep on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && chmod +x run_stacked_rescue_veto_071_remote.sh && if [[ -f stacked_rescue_veto_071_v2.pid ]] && kill -0 \$(cat stacked_rescue_veto_071_v2.pid) 2>/dev/null; then echo already_running_pid=\$(cat stacked_rescue_veto_071_v2.pid); else STACKED_RESCUE_OUTPUT_NAME=outputs_stacked_rescue_veto_071_v2 nohup ./run_stacked_rescue_veto_071_remote.sh > stacked_rescue_veto_071_v2.log 2>&1 < /dev/null & echo \$! > stacked_rescue_veto_071_v2.pid; echo started_pid=\$(cat stacked_rescue_veto_071_v2.pid); fi"

cat <<EOF

Fixed stacked rescue/veto v2 sweep is running independently on EC2.

Watch live progress bars with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f stacked_rescue_veto_071_v2.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_stacked_rescue_veto_071_v2/STACKED_RESCUE_VETO_COMPLETE.txt || ps -p \$(cat stacked_rescue_veto_071_v2.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_stacked_rescue_veto_071_v2/stacked_rescue_veto_summary.csv | head -30"
EOF

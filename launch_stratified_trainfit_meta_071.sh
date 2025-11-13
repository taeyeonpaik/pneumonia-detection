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

echo "Uploading stratified train-fit meta sweep scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/stratified_trainfit_meta_threshold_sweep.py" \
  "$LOCAL_DIR/run_stratified_trainfit_meta_071_remote.sh" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting stratified train-fit meta sweep on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && chmod +x run_stratified_trainfit_meta_071_remote.sh && if [[ -f stratified_trainfit_meta_071.pid ]] && kill -0 \$(cat stratified_trainfit_meta_071.pid) 2>/dev/null; then echo already_running_pid=\$(cat stratified_trainfit_meta_071.pid); else nohup ./run_stratified_trainfit_meta_071_remote.sh > stratified_trainfit_meta_071.log 2>&1 < /dev/null & echo \$! > stratified_trainfit_meta_071.pid; echo started_pid=\$(cat stratified_trainfit_meta_071.pid); fi"

cat <<EOF

Stratified train-fit meta sweep is running independently on EC2.

Watch the live progress bars with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f stratified_trainfit_meta_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_stratified_trainfit_meta_071/STRATIFIED_TRAINFIT_META_071_COMPLETE.txt || ps -p \$(cat stratified_trainfit_meta_071.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_stratified_trainfit_meta_071/stratified_trainfit_meta_071_summary.csv | head -30"
EOF

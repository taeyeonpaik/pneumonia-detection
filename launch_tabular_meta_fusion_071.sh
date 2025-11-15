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

echo "Uploading tabular meta-fusion scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/tabular_meta_fusion_from_features.py" \
  "$LOCAL_DIR/run_tabular_meta_fusion_071_remote.sh" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting tabular meta-fusion push on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && chmod +x run_tabular_meta_fusion_071_remote.sh && if [[ -f tabular_meta_fusion_071.pid ]] && kill -0 \$(cat tabular_meta_fusion_071.pid) 2>/dev/null; then echo already_running_pid=\$(cat tabular_meta_fusion_071.pid); else nohup ./run_tabular_meta_fusion_071_remote.sh > tabular_meta_fusion_071.log 2>&1 < /dev/null & echo \$! > tabular_meta_fusion_071.pid; echo started_pid=\$(cat tabular_meta_fusion_071.pid); fi"

cat <<EOF

Tabular meta-fusion push is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f tabular_meta_fusion_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_tabular_meta_fusion_071/TABULAR_META_FUSION_071_COMPLETE.txt || ps -p \$(cat tabular_meta_fusion_071.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_tabular_meta_fusion_071/tabular_meta_fusion_071_summary.csv | head -30"
EOF

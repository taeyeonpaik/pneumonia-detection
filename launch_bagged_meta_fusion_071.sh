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

echo "Uploading bagged meta-fusion script..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/bagged_meta_fusion_from_features.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting bagged meta-fusion on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_bagged_meta_fusion_071.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_bagged_meta_fusion_071
METADATA=\$ROOT/dicom_metadata.csv
BASELINE_SUMMARY=\$ROOT/outputs_trainfit_fusion_push_070/trainfit_fusion_summary.csv

mkdir -p \"\$OUT\" \"\$OUT/logs\"

if [[ ! -f \"\$METADATA\" ]]; then
  echo \"Missing metadata CSV: \$METADATA\" >&2
  exit 1
fi

declare -a CANDIDATES

add_candidate() {
  local dir=\"\$1\"
  if [[ -f \"\$dir/fusion_train_features.csv\" ]]; then
    CANDIDATES+=(\"\$dir\")
  fi
}

add_candidate \"\$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_convnext384_trainfit\"
add_candidate \"\$ROOT/outputs_exact_trainfit_meta_export_071/swin_convnext384_trainfit_exact_export\"
add_candidate \"\$ROOT/outputs_convnextv2_recovery_070/fusion/swin_densenet_convnextv2_meta_asl_balanced_512\"
add_candidate \"\$ROOT/outputs_convnextv2_recovery_070/fusion/swin_plus_convnextv2_meta_asl_balanced_512\"
add_candidate \"\$ROOT/outputs_final_push_070/fusion/swin_plus_densenet169_meta_asl_balanced\"
add_candidate \"\$ROOT/outputs_swinv2_base_push_071_v2/fusion/swin_swinv2_base_balanced\"

mapfile -t CANDIDATES < <(printf '%s\n' \"\${CANDIDATES[@]}\" | awk '!seen[\$0]++')

if [[ \${#CANDIDATES[@]} -eq 0 ]]; then
  echo \"No candidate feature directories found.\" >&2
  exit 1
fi

echo \"Candidate feature dirs:\"
printf '  %s\n' \"\${CANDIDATES[@]}\"

if [[ -f \"\$OUT/BAGGED_META_FUSION_COMPLETE.txt\" ]]; then
  echo \"Skipping bagged meta-fusion; already complete:\"
  cat \"\$OUT/BAGGED_META_FUSION_COMPLETE.txt\"
  exit 0
fi

/opt/pytorch/bin/python -u bagged_meta_fusion_from_features.py \
  --fusion-dirs \"\${CANDIDATES[@]}\" \
  --metadata-csv \"\$METADATA\" \
  --baseline-summary-csv \"\$BASELINE_SUMMARY\" \
  --output-dir \"\$OUT\" \
  --feature-modes scores,classifier_detector_metadata,all \
  --model-types logistic \
  --pos-weights 1.25,1.5,2.0,2.5 \
  --l2-values 0.001,0.003,0.01 \
  --lrs 0.02 \
  --fit-seeds 42,1337 \
  --num-bags 5,7 \
  --sample-fracs 0.75,1.0 \
  --feature-fracs 0.65,0.85,1.0 \
  --epochs 350 \
  --thresholds 0.30,0.35,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
  2>&1 | tee \"\$OUT/logs/bagged_meta_fusion.log\"
REMOTE
chmod +x run_bagged_meta_fusion_071.sh
if [[ -f bagged_meta_fusion_071.pid ]] && kill -0 \$(cat bagged_meta_fusion_071.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat bagged_meta_fusion_071.pid)
else
  nohup ./run_bagged_meta_fusion_071.sh > bagged_meta_fusion_071.log 2>&1 < /dev/null &
  echo \$! > bagged_meta_fusion_071.pid
  echo started_pid=\$(cat bagged_meta_fusion_071.pid)
fi"

cat <<EOF

Bagged meta-fusion is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f bagged_meta_fusion_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_bagged_meta_fusion_071/BAGGED_META_FUSION_COMPLETE.txt || ps -p \$(cat bagged_meta_fusion_071.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_bagged_meta_fusion_071/bagged_meta_fusion_summary.csv | head -30"
EOF

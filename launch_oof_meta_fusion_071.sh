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

echo "Uploading OOF meta-fusion script..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/oof_meta_fusion_from_features.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting OOF meta-fusion on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_oof_meta_fusion_071.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_oof_meta_fusion_071
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
  if [[ -f \"\$dir/fusion_validation_features.csv\" && -f \"\$dir/fusion_test_features.csv\" ]]; then
    CANDIDATES+=(\"\$dir\")
  fi
}

add_candidate \"\$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_convnext384_trainfit\"
add_candidate \"\$ROOT/outputs_exact_trainfit_meta_export_071/swin_convnext384_trainfit_exact_export\"
add_candidate \"\$ROOT/outputs_swinv2_base_push_071_v2/fusion/swin_swinv2_base_balanced\"
add_candidate \"\$ROOT/outputs_swinv2_base_push_071_v2/fusion/swin_swinv2_small_balanced\"
add_candidate \"\$ROOT/outputs_convnextv2_recovery_070/fusion/swin_plus_convnextv2_meta_asl_balanced_512\"
add_candidate \"\$ROOT/outputs_convnextv2_recovery_070/fusion/swin_densenet_convnextv2_meta_asl_balanced_512\"
add_candidate \"\$ROOT/outputs_final_push_070/fusion/swin_plus_densenet169_meta_asl_balanced\"
add_candidate \"\$ROOT/outputs_bagged_meta_elite_071\"

mapfile -t CANDIDATES < <(printf '%s\n' \"\${CANDIDATES[@]}\" | awk '!seen[\$0]++')

if [[ \${#CANDIDATES[@]} -eq 0 ]]; then
  echo \"No candidate feature directories found.\" >&2
  exit 1
fi

echo \"Candidate feature dirs:\"
printf '  %s\n' \"\${CANDIDATES[@]}\"

if [[ -f \"\$OUT/OOF_META_FUSION_COMPLETE.txt\" ]]; then
  echo \"Skipping OOF meta-fusion; already complete:\"
  cat \"\$OUT/OOF_META_FUSION_COMPLETE.txt\"
  exit 0
fi

/opt/pytorch/bin/python -u oof_meta_fusion_from_features.py \
  --fusion-dirs \"\${CANDIDATES[@]}\" \
  --metadata-csv \"\$METADATA\" \
  --baseline-summary-csv \"\$BASELINE_SUMMARY\" \
  --output-dir \"\$OUT\" \
  --feature-modes scores,classifier_detector_metadata,all \
  --model-types logistic,mlp \
  --pos-weights 0.75,1.0,1.25,1.5,1.75,2.0,2.5 \
  --l2-values 0.0001,0.001,0.003,0.006,0.01 \
  --lrs 0.01,0.02 \
  --fit-seeds 42,1337,2024 \
  --num-folds 5 \
  --epochs 650 \
  --stratify-columns age_bucket,view_group,sex_group \
  --min-stratify-group-size 80 \
  --thresholds 0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72 \
  2>&1 | tee \"\$OUT/logs/oof_meta_fusion.log\"
REMOTE
chmod +x run_oof_meta_fusion_071.sh
if [[ -f oof_meta_fusion_071.pid ]] && kill -0 \$(cat oof_meta_fusion_071.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat oof_meta_fusion_071.pid)
else
  nohup ./run_oof_meta_fusion_071.sh > oof_meta_fusion_071.log 2>&1 < /dev/null &
  echo \$! > oof_meta_fusion_071.pid
  echo started_pid=\$(cat oof_meta_fusion_071.pid)
fi"

cat <<EOF

OOF meta-fusion is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f oof_meta_fusion_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_oof_meta_fusion_071/OOF_META_FUSION_COMPLETE.txt || ps -p \$(cat oof_meta_fusion_071.pid) -o pid,etime,cmd"

Show validation-ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_oof_meta_fusion_071/oof_meta_fusion_validation_grid.csv | head -30"

Show exploratory test-ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_oof_meta_fusion_071/oof_meta_fusion_test_oracle_grid.csv | head -30"
EOF

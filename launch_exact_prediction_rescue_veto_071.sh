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

echo "Uploading exact-prediction rescue/veto script..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/stacked_rescue_veto_sweep.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting exact-prediction rescue/veto on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_exact_prediction_rescue_veto_071.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_exact_prediction_rescue_veto_071
METADATA=\$ROOT/dicom_metadata.csv
BASELINE_DIR=\$ROOT/outputs_exact_trainfit_meta_export_071/swin_convnext384_trainfit_exact_export
BASELINE_SUMMARY=\$ROOT/outputs_trainfit_fusion_push_070/trainfit_fusion_summary.csv
BASELINE_VALIDATION_PREDICTIONS=\$BASELINE_DIR/fusion_sweep_validation_predictions.csv
BASELINE_TEST_PREDICTIONS=\$BASELINE_DIR/fusion_sweep_test_predictions.csv

mkdir -p \"\$OUT\" \"\$OUT/logs\"

if [[ ! -f \"\$METADATA\" ]]; then
  echo \"Missing metadata CSV: \$METADATA\" >&2
  exit 1
fi
if [[ ! -f \"\$BASELINE_DIR/fusion_train_features.csv\" ]]; then
  echo \"Missing exact baseline train features: \$BASELINE_DIR/fusion_train_features.csv\" >&2
  echo \"Run ./launch_exact_trainfit_meta_export_071.sh first.\" >&2
  exit 1
fi
if [[ ! -f \"\$BASELINE_VALIDATION_PREDICTIONS\" ]]; then
  echo \"Missing exact baseline validation predictions: \$BASELINE_VALIDATION_PREDICTIONS\" >&2
  echo \"Run ./launch_exact_trainfit_meta_export_071.sh first.\" >&2
  exit 1
fi
if [[ ! -f \"\$BASELINE_TEST_PREDICTIONS\" ]]; then
  echo \"Missing exact baseline test predictions: \$BASELINE_TEST_PREDICTIONS\" >&2
  echo \"Run ./launch_exact_trainfit_meta_export_071.sh first.\" >&2
  exit 1
fi

declare -a CANDIDATES
CANDIDATES+=(\"\$BASELINE_DIR\")

add_candidate_tree() {
  local tree=\"\$1\"
  if [[ ! -d \"\$tree\" ]]; then
    return 0
  fi
  while IFS= read -r feature_file; do
    CANDIDATES+=(\"\$(dirname \"\$feature_file\")\")
  done < <(find \"\$tree\" -maxdepth 2 -type f -name fusion_train_features.csv | sort)
}

add_candidate_tree \"\$ROOT/outputs_trainfit_fusion_push_070/fusion\"
add_candidate_tree \"\$ROOT/outputs_rich_tta_fusion_071/fusion\"
add_candidate_tree \"\$ROOT/outputs_swinv2_base_push_071_v2/fusion\"
add_candidate_tree \"\$ROOT/outputs_swinv2_base_push_071/fusion\"
add_candidate_tree \"\$ROOT/outputs_convnextv2_recovery_070/fusion\"
add_candidate_tree \"\$ROOT/outputs_final_push_070/fusion\"

mapfile -t CANDIDATES < <(printf '%s\n' \"\${CANDIDATES[@]}\" | awk '!seen[\$0]++')

echo \"Candidate feature dirs:\"
printf '  %s\n' \"\${CANDIDATES[@]}\"
echo \"Using exact validation predictions: \$BASELINE_VALIDATION_PREDICTIONS\"
echo \"Using exact test predictions: \$BASELINE_TEST_PREDICTIONS\"

if [[ -f \"\$OUT/STACKED_RESCUE_VETO_COMPLETE.txt\" ]]; then
  echo \"Skipping exact-prediction rescue/veto; already complete:\"
  cat \"\$OUT/STACKED_RESCUE_VETO_COMPLETE.txt\"
  exit 0
fi

/opt/pytorch/bin/python -u stacked_rescue_veto_sweep.py \
  --baseline-fusion-dir \"\$BASELINE_DIR\" \
  --candidate-dirs \"\${CANDIDATES[@]}\" \
  --metadata-csv \"\$METADATA\" \
  --baseline-summary-csv \"\$BASELINE_SUMMARY\" \
  --baseline-validation-predictions \"\$BASELINE_VALIDATION_PREDICTIONS\" \
  --baseline-test-predictions \"\$BASELINE_TEST_PREDICTIONS\" \
  --output-dir \"\$OUT\" \
  --feature-modes scores,classifier_detector_metadata,all \
  --model-types logistic,mlp \
  --pos-weights 1.25,1.5,1.75,2.0,2.5,3.0 \
  --l2-values 0.0,0.0001,0.001,0.003,0.01 \
  --lrs 0.01,0.02 \
  --fit-seeds 42,1337 \
  --epochs 700 \
  --thresholds 0.30,0.35,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
  2>&1 | tee \"\$OUT/logs/exact_prediction_rescue_veto.log\"
REMOTE
chmod +x run_exact_prediction_rescue_veto_071.sh
if [[ -f exact_prediction_rescue_veto_071.pid ]] && kill -0 \$(cat exact_prediction_rescue_veto_071.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat exact_prediction_rescue_veto_071.pid)
else
  nohup ./run_exact_prediction_rescue_veto_071.sh > exact_prediction_rescue_veto_071.log 2>&1 < /dev/null &
  echo \$! > exact_prediction_rescue_veto_071.pid
  echo started_pid=\$(cat exact_prediction_rescue_veto_071.pid)
fi"

cat <<EOF

Exact-prediction rescue/veto is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f exact_prediction_rescue_veto_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_exact_prediction_rescue_veto_071/STACKED_RESCUE_VETO_COMPLETE.txt || ps -p \$(cat exact_prediction_rescue_veto_071.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_exact_prediction_rescue_veto_071/stacked_rescue_veto_summary.csv | head -30"
EOF

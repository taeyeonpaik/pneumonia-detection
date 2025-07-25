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

chmod 400 "$KEY_FILE"

echo "Uploading latest stratified sweep script..."
rsync -avP \
  -e "ssh -i $KEY_FILE" \
  "$LOCAL_DIR/stratified_precision_gate_sweep.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Running fine age-bucket threshold sweep..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && /opt/pytorch/bin/python stratified_precision_gate_sweep.py \
    --fusion-dir outputs_reproduce_best_analysis/fusion_original_best_reproduction \
    --metadata-csv dicom_metadata.csv \
    --output-dir outputs_reproduce_best_analysis/stratified_precision_gate_age_fine \
    --group-columns age_bucket \
    --gate-classifier-agree-thresholds 0.60,0.62,0.64,0.65,0.66,0.67,0.68,0.70,0.72 \
    --gate-detector-required-thresholds 0.25,0.30,0.35,0.40,0.45,0.50,0.54,0.58,0.62 \
    --gate-classifier-override-thresholds 0.76,0.78,0.80,0.82,0.84,0.86 \
    --gate-detector-override-thresholds 0.48,0.50,0.54,0.58,0.62,0.66,0.70 \
    --gate-detector-classifier-max-thresholds 0.15,0.20,0.25,0.30,0.35,0.40,0.50,1.00"

echo
echo "Fine sweep metrics:"
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && column -s, -t < outputs_reproduce_best_analysis/stratified_precision_gate_age_fine/stratified_strategy_test_metrics.csv"

echo
echo "Selected fine age-bucket rules:"
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" \
  "cd '$REMOTE_DIR' && column -s, -t < outputs_reproduce_best_analysis/stratified_precision_gate_age_fine/stratified_selected_rules.csv"

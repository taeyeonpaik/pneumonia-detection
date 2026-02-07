#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/ec2-user/rsna
OUTPUT_NAME="${STACKED_RESCUE_OUTPUT_NAME:-outputs_stacked_rescue_veto_071}"
OUT="$ROOT/$OUTPUT_NAME"
METADATA="$ROOT/dicom_metadata.csv"
BASELINE_DIR="$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_convnext384_trainfit"
BASELINE_SUMMARY="$ROOT/outputs_trainfit_fusion_push_070/trainfit_fusion_summary.csv"

mkdir -p "$OUT" "$OUT/logs"

if [[ ! -f "$METADATA" ]]; then
  echo "Missing metadata CSV: $METADATA" >&2
  exit 1
fi
if [[ ! -f "$BASELINE_DIR/fusion_train_features.csv" ]]; then
  echo "Missing baseline train features: $BASELINE_DIR/fusion_train_features.csv" >&2
  exit 1
fi

declare -a CANDIDATES
CANDIDATES+=("$BASELINE_DIR")

add_candidate_tree() {
  local tree="$1"
  if [[ ! -d "$tree" ]]; then
    return 0
  fi
  while IFS= read -r feature_file; do
    CANDIDATES+=("$(dirname "$feature_file")")
  done < <(find "$tree" -maxdepth 2 -type f -name fusion_train_features.csv | sort)
}

add_candidate_tree "$ROOT/outputs_trainfit_fusion_push_070/fusion"
add_candidate_tree "$ROOT/outputs_rich_tta_fusion_071/fusion"
add_candidate_tree "$ROOT/outputs_swinv2_base_push_071_v2/fusion"
add_candidate_tree "$ROOT/outputs_swinv2_base_push_071/fusion"
add_candidate_tree "$ROOT/outputs_convnextv2_recovery_070/fusion"
add_candidate_tree "$ROOT/outputs_final_push_070/fusion"

mapfile -t CANDIDATES < <(printf '%s\n' "${CANDIDATES[@]}" | awk '!seen[$0]++')

echo "Candidate feature dirs:"
printf '  %s\n' "${CANDIDATES[@]}"

if [[ -f "$OUT/STACKED_RESCUE_VETO_COMPLETE.txt" ]]; then
  echo "Skipping stacked rescue/veto; already complete:"
  cat "$OUT/STACKED_RESCUE_VETO_COMPLETE.txt"
  exit 0
fi

echo "=== Stacked score + baseline rescue/veto sweep ==="
/opt/pytorch/bin/python -u stacked_rescue_veto_sweep.py \
  --baseline-fusion-dir "$BASELINE_DIR" \
  --candidate-dirs "${CANDIDATES[@]}" \
  --metadata-csv "$METADATA" \
  --baseline-summary-csv "$BASELINE_SUMMARY" \
  --output-dir "$OUT" \
  --feature-modes scores,classifier_detector_metadata,all \
  --model-types logistic,mlp \
  --pos-weights 1.25,1.5,1.75,2.0,2.5,3.0 \
  --l2-values 0.0,0.0001,0.001,0.003,0.01 \
  --lrs 0.01,0.02 \
  --fit-seeds 42,1337 \
  --epochs 700 \
  --thresholds 0.30,0.35,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
  2>&1 | tee "$OUT/logs/stacked_rescue_veto.log"

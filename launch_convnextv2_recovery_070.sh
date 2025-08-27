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

echo "Uploading ConvNeXtV2 recovery scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/train_classifier.py" \
  "$LOCAL_DIR/fusion_sweep.py" \
  "$LOCAL_DIR/stratified_precision_gate_sweep.py" \
  "$LOCAL_DIR/analyze_best_model_errors.py" \
  "$LOCAL_DIR/dicom_metadata.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting ConvNeXtV2 recovery + fusion run on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_convnextv2_recovery_070.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_convnextv2_recovery_070
mkdir -p \"\$OUT\" \"\$OUT/logs\" \"\$OUT/classifiers\" \"\$OUT/fusion\" \"\$OUT/stratified\"

METADATA=\$ROOT/dicom_metadata.csv
if [[ ! -f \"\$METADATA\" ]]; then
  echo \"Missing metadata CSV: \$METADATA\" >&2
  exit 1
fi

echo \"Installing timm if needed...\"
/opt/pytorch/bin/python -m pip install -q timm || echo \"Warning: timm install failed; ConvNeXtV2 training may fail.\"

DETECTOR_CKPT=\"\"
for candidate in \
  \"\$ROOT/outputs_reproduce_best_analysis/detector_hardneg2_adamw_b16_lr0001/frcnn_epoch_7.pt\" \
  \"\$ROOT/outputs_overnight_rebuild/detector_a_final/frcnn_epoch_7.pt\" \
  \"\$ROOT/outputs_hardneg2_adamw_b16_lr0001/frcnn_epoch_7.pt\" \
  \"\$ROOT/outputs_original_iou_metadata/frcnn_box50_hardneg/frcnn_epoch_7.pt\"; do
  if [[ -f \"\$candidate\" ]]; then
    DETECTOR_CKPT=\"\$candidate\"
    break
  fi
done

BEST_SWIN=\"\"
for candidate in \
  \"\$ROOT/outputs_reproduce_best_analysis/swin_t_hardneg_metadata/classifier_best.pt\" \
  \"\$ROOT/outputs_overnight_rebuild/swin_t_hardneg_metadata/classifier_best.pt\" \
  \"\$ROOT/outputs_original_iou_metadata/swin_t_hardneg_metadata/classifier_best.pt\"; do
  if [[ -f \"\$candidate\" ]]; then
    BEST_SWIN=\"\$candidate\"
    break
  fi
done

BEST_DENSENET=\"\"
for candidate in \
  \"\$ROOT/outputs_final_push_070/classifiers/densenet169_meta_asl_balanced/classifier_best.pt\" \
  \"\$ROOT/outputs_final_push_070/classifiers/densenet169_meta_asl_precision/classifier_best.pt\"; do
  if [[ -f \"\$candidate\" ]]; then
    BEST_DENSENET=\"\$candidate\"
    break
  fi
done

HARD_NEGATIVE_FILE=\"\"
for candidate in \
  \"\$ROOT/outputs_shared_fp_hardneg_swin/shared_fp_mining/hard_negative_patient_ids.csv\" \
  \"\$ROOT/outputs_reproduce_best_analysis/swin_hard_negatives/hard_negative_patient_ids.csv\" \
  \"\$ROOT/outputs_overnight_rebuild/swin_hard_negatives/hard_negative_patient_ids.csv\" \
  \"\$ROOT/outputs_best_swin_hardneg_mining/hard_negative_patient_ids.csv\"; do
  if [[ -f \"\$candidate\" ]]; then
    HARD_NEGATIVE_FILE=\"\$candidate\"
    break
  fi
done

if [[ -z \"\$DETECTOR_CKPT\" ]]; then
  echo \"Could not find detector checkpoint.\" >&2
  exit 1
fi
if [[ -z \"\$BEST_SWIN\" ]]; then
  echo \"Could not find metadata-aware Swin checkpoint.\" >&2
  exit 1
fi
if [[ -z \"\$BEST_DENSENET\" ]]; then
  echo \"Could not find final-push DenseNet169 checkpoint. Run launch_final_push_070.sh first.\" >&2
  exit 1
fi

echo \"Using detector: \$DETECTOR_CKPT\"
echo \"Using Swin: \$BEST_SWIN\"
echo \"Using DenseNet169: \$BEST_DENSENET\"
echo \"Using hard negatives if present: \${HARD_NEGATIVE_FILE:-none}\"

THRESHOLDS=0.25,0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.65,0.66,0.68,0.70,0.72,0.74,0.75,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90

train_convnextv2() {
  local name=\"\$1\"
  local image_size=\"\$2\"
  local batch_size=\"\$3\"
  local grad_accum=\"\$4\"
  local lr=\"\$5\"
  local gamma_neg=\"\$6\"
  local classifier_dir=\"\$OUT/classifiers/\$name\"
  mkdir -p \"\$classifier_dir\"

  if [[ -f \"\$classifier_dir/classifier_best.pt\" ]]; then
    echo \"Skipping classifier \$name; found classifier_best.pt\"
    return 0
  fi

  local cmd=(
    /opt/pytorch/bin/python -u train_classifier.py
    --data-dir \"\$ROOT\"
    --output-dir \"\$classifier_dir\"
    --architecture timm_convnextv2_base
    --image-size \"\$image_size\"
    --epochs 10
    --batch-size \"\$batch_size\"
    --grad-accum-steps \"\$grad_accum\"
    --num-workers 4
    --lr \"\$lr\"
    --weight-decay 0.001
    --augmentation cxr
    --loss asl
    --asl-gamma-neg \"\$gamma_neg\"
    --asl-clip 0.05
    --label-smoothing 0.03
    --balanced-batches
    --mixup-alpha 0.05
    --eval-tta-modes none,hflip
    --thresholds \"\$THRESHOLDS\"
    --metadata-csv \"\$METADATA\"
    --metadata-hidden-dim 64
    --no-save-epoch-checkpoints
    --seed 42
  )
  if [[ -n \"\$HARD_NEGATIVE_FILE\" ]]; then
    cmd+=(--hard-negative-file \"\$HARD_NEGATIVE_FILE\" --hard-negative-weight 1.5)
  fi

  echo \"=== Training \$name (ConvNeXtV2 base, size=\$image_size, gamma_neg=\$gamma_neg) ===\"
  if ! \"\${cmd[@]}\" > \"\$OUT/logs/train_\$name.log\" 2>&1; then
    echo \"WARNING: classifier \$name failed. See \$OUT/logs/train_\$name.log\" >&2
    return 0
  fi
}

run_fusion() {
  local name=\"\$1\"
  local checkpoints=\"\$2\"
  local extra_args=\"\$3\"
  local fusion_dir=\"\$OUT/fusion/\$name\"
  local strat_dir=\"\$OUT/stratified/\$name\"
  mkdir -p \"\$fusion_dir\" \"\$strat_dir\"

  if [[ ! -f \"\$fusion_dir/fusion_sweep_test_metrics.csv\" ]]; then
    echo \"=== Fusion \$name ===\"
    # shellcheck disable=SC2206
    local extra=(\$extra_args)
    if ! /opt/pytorch/bin/python -u fusion_sweep.py \
      --data-dir \"\$ROOT\" \
      --output-dir \"\$fusion_dir\" \
      --metadata-csv \"\$METADATA\" \
      --classifier-checkpoints \"\$checkpoints\" \
      --detector-checkpoint \"\$DETECTOR_CKPT\" \
      --detector-min-size 1024 \
      --detector-ensemble-method none \
      --classifier-tta-modes none,hflip \
      --classifier-temperature-mode validation \
      --methods precision_gate,meta \
      --box-score-thresholds 0.40,0.44,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82 \
      --gate-classifier-agree-thresholds 0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.65,0.66,0.67,0.68,0.70,0.72 \
      --gate-detector-required-thresholds 0.25,0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.65 \
      --gate-classifier-override-thresholds 0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.96 \
      --gate-detector-override-thresholds 0.48,0.50,0.54,0.58,0.60,0.62,0.65,0.68,0.70,0.75,0.80 \
      --gate-detector-classifier-max-thresholds 0.15,0.20,0.25,0.30,0.35,0.40,0.50,1.00 \
      --meta-thresholds 0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
      --meta-pos-weights 1.0,1.5,2.0,2.5,3.0 \
      --meta-l2-values 0.0,0.0001,0.001,0.01 \
      --metric-iou-threshold 0.5 \
      --batch-size 16 \
      --classifier-batch-size 2 \
      --num-workers 0 \
      --min-precision 0 \
      --min-recall 0 \
      --seed 42 \
      \"\${extra[@]}\" \
      > \"\$OUT/logs/fusion_\$name.log\" 2>&1; then
      echo \"WARNING: fusion \$name failed. See \$OUT/logs/fusion_\$name.log\" >&2
      return 0
    fi
  else
    echo \"Skipping fusion \$name; found fusion_sweep_test_metrics.csv\"
  fi

  if [[ ! -f \"\$strat_dir/stratified_strategy_test_metrics.csv\" ]]; then
    echo \"=== Stratified gate \$name ===\"
    if ! /opt/pytorch/bin/python -u stratified_precision_gate_sweep.py \
      --fusion-dir \"\$fusion_dir\" \
      --metadata-csv \"\$METADATA\" \
      --output-dir \"\$strat_dir\" \
      --group-columns age_bucket,view_group,sex_group \
      > \"\$OUT/logs/stratified_\$name.log\" 2>&1; then
      echo \"WARNING: stratified sweep \$name failed. See \$OUT/logs/stratified_\$name.log\" >&2
      return 0
    fi
  else
    echo \"Skipping stratified \$name; found stratified_strategy_test_metrics.csv\"
  fi
}

train_convnextv2 convnextv2_meta_asl_balanced_512 512 2 4 0.00002 3.0
train_convnextv2 convnextv2_meta_asl_precision_512 512 2 4 0.00002 4.0
train_convnextv2 convnextv2_meta_asl_balanced_384 384 4 2 0.00003 3.0

declare -A CKPTS=()
CKPTS[swin]=\"\$BEST_SWIN\"
CKPTS[densenet]=\"\$BEST_DENSENET\"
for name in convnextv2_meta_asl_balanced_512 convnextv2_meta_asl_precision_512 convnextv2_meta_asl_balanced_384; do
  path=\"\$OUT/classifiers/\$name/classifier_best.pt\"
  if [[ -f \"\$path\" ]]; then
    CKPTS[\$name]=\"\$path\"
  fi
done

BASE_EXTRA=\"\"
CALIBRATED_EXTRA=\"--detector-score-calibration classifier_min\"
TIGHT_BOX_EXTRA=\"--detector-post-nms-iou 0.30 --detector-min-area-ratio 0.0005\"

for conv in convnextv2_meta_asl_balanced_512 convnextv2_meta_asl_precision_512 convnextv2_meta_asl_balanced_384; do
  if [[ -z \"\${CKPTS[\$conv]:-}\" ]]; then
    continue
  fi
  run_fusion \"single_\$conv\" \"\${CKPTS[\$conv]}\" \"\$BASE_EXTRA\"
  run_fusion \"swin_plus_\$conv\" \"\${CKPTS[swin]},\${CKPTS[\$conv]}\" \"\$BASE_EXTRA\"
  run_fusion \"swin_plus_\${conv}_detcal\" \"\${CKPTS[swin]},\${CKPTS[\$conv]}\" \"\$CALIBRATED_EXTRA\"
  run_fusion \"densenet_plus_\$conv\" \"\${CKPTS[densenet]},\${CKPTS[\$conv]}\" \"\$BASE_EXTRA\"
  run_fusion \"swin_densenet_\$conv\" \"\${CKPTS[swin]},\${CKPTS[densenet]},\${CKPTS[\$conv]}\" \"\$BASE_EXTRA\"
  run_fusion \"swin_densenet_\${conv}_detcal\" \"\${CKPTS[swin]},\${CKPTS[densenet]},\${CKPTS[\$conv]}\" \"\$CALIBRATED_EXTRA\"
  run_fusion \"swin_densenet_\${conv}_tightbox\" \"\${CKPTS[swin]},\${CKPTS[densenet]},\${CKPTS[\$conv]}\" \"\$TIGHT_BOX_EXTRA\"
done

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('/home/ec2-user/rsna/outputs_convnextv2_recovery_070')
rows = []

baseline_path = Path('/home/ec2-user/rsna/outputs_final_push_070/final_push_summary.csv')
if baseline_path.exists():
    baseline = pd.read_csv(baseline_path).sort_values(['image_f1', 'image_precision'], ascending=[False, False]).head(1)
    for row in baseline.to_dict('records'):
        rows.append({'variant': 'baseline_final_push_best', 'strategy': 'baseline_existing_best', **row})

for path in sorted((root / 'fusion').glob('*/fusion_sweep_test_metrics.csv')):
    variant = path.parent.name
    try:
        table = pd.read_csv(path)
    except Exception:
        continue
    for row in table.to_dict('records'):
        rows.append({'variant': variant, 'strategy': 'global_fusion', **row})

for path in sorted((root / 'stratified').glob('*/stratified_strategy_test_metrics.csv')):
    variant = path.parent.name
    try:
        table = pd.read_csv(path)
    except Exception:
        continue
    for row in table.to_dict('records'):
        rows.append({'variant': variant, 'strategy': row.get('strategy', 'stratified_precision_gate'), **row})

summary = pd.DataFrame(rows)
if summary.empty:
    raise SystemExit('No completed ConvNeXtV2 recovery results found.')

summary = summary.sort_values(['image_f1', 'image_precision', 'image_recall'], ascending=[False, False, False])
summary.to_csv(root / 'convnextv2_recovery_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(root / 'convnextv2_recovery_summary.json').write_text(json.dumps({'best': best, 'num_rows': len(summary)}, indent=2), encoding='utf-8')
(root / 'CONVNEXTV2_RECOVERY_COMPLETE').write_text(
    f\"Best variant: {best.get('variant')}\\n\"
    f\"Strategy: {best.get('strategy')}\\n\"
    f\"Method: {best.get('method', '')}\\n\"
    f\"F1: {best.get('image_f1')}\\n\"
    f\"Precision: {best.get('image_precision')}\\n\"
    f\"Recall: {best.get('image_recall')}\\n\"
    f\"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\\n\",
    encoding='utf-8',
)
print(summary.head(30).to_string(index=False))
PY
REMOTE
chmod +x run_convnextv2_recovery_070.sh
if [[ -f convnextv2_recovery_070.pid ]] && kill -0 \$(cat convnextv2_recovery_070.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat convnextv2_recovery_070.pid)
else
  nohup ./run_convnextv2_recovery_070.sh > convnextv2_recovery_070.log 2>&1 < /dev/null &
  echo \$! > convnextv2_recovery_070.pid
  echo started_pid=\$(cat convnextv2_recovery_070.pid)
fi"

cat <<EOF

ConvNeXtV2 recovery run is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f convnextv2_recovery_070.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_convnextv2_recovery_070/CONVNEXTV2_RECOVERY_COMPLETE || ps -p \$(cat convnextv2_recovery_070.pid) -o pid,etime,cmd"

Show the ranked summary when it finishes:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_convnextv2_recovery_070/convnextv2_recovery_summary.csv | head -30"
EOF

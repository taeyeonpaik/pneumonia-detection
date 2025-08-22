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

echo "Uploading final-push scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/train_classifier.py" \
  "$LOCAL_DIR/fusion_sweep.py" \
  "$LOCAL_DIR/stratified_precision_gate_sweep.py" \
  "$LOCAL_DIR/analyze_best_model_errors.py" \
  "$LOCAL_DIR/dicom_metadata.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting final 0.70 push on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_final_push_070.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_final_push_070
mkdir -p \"\$OUT\" \"\$OUT/logs\" \"\$OUT/fusion\" \"\$OUT/stratified\"

METADATA=\$ROOT/dicom_metadata.csv
if [[ ! -f \"\$METADATA\" ]]; then
  echo \"Missing metadata CSV: \$METADATA\" >&2
  exit 1
fi

echo \"Installing optional model packages if needed...\"
/opt/pytorch/bin/python -m pip install -q timm torchxrayvision || \
  echo \"Warning: optional dependency install failed; timm/XRV variants may be skipped.\"

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
  \"\$ROOT/outputs_original_iou_metadata/swin_t_hardneg_metadata/classifier_best.pt\" \
  \"\$ROOT/outputs_overnight_rebuild/swin_t_hardneg_metadata/classifier_best.pt\"; do
  if [[ -f \"\$candidate\" ]]; then
    BEST_SWIN=\"\$candidate\"
    break
  fi
done

BEST_CONVNEXTV2=\"\"
for candidate in \
  \"\$ROOT/outputs_classifier_convnextv2_base_balanced_mixup/classifier_best.pt\" \
  \"\$ROOT/outputs_overnight_rebuild/convnextv2_base/classifier_best.pt\" \
  \"\$ROOT/outputs_seed2024_convnextv2_base_balanced_mixup/classifier_best.pt\"; do
  if [[ -f \"\$candidate\" ]]; then
    BEST_CONVNEXTV2=\"\$candidate\"
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
  echo \"Could not find the original best Faster R-CNN checkpoint.\" >&2
  exit 1
fi
if [[ -z \"\$BEST_SWIN\" ]]; then
  echo \"Could not find metadata-aware Swin checkpoint.\" >&2
  exit 1
fi

echo \"Using detector: \$DETECTOR_CKPT\"
echo \"Using metadata-aware Swin: \$BEST_SWIN\"
echo \"Using ConvNeXtV2 if present: \${BEST_CONVNEXTV2:-none}\"
echo \"Using hard negatives if present: \${HARD_NEGATIVE_FILE:-none}\"

THRESHOLDS=0.25,0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.65,0.66,0.68,0.70,0.72,0.74,0.75,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90

train_variant() {
  local name=\"\$1\"
  local architecture=\"\$2\"
  local image_size=\"\$3\"
  local batch_size=\"\$4\"
  local grad_accum=\"\$5\"
  local lr=\"\$6\"
  local weight_decay=\"\$7\"
  local loss_mode=\"\$8\"
  local classifier_dir=\"\$OUT/classifiers/\$name\"
  mkdir -p \"\$classifier_dir\"

  if [[ -f \"\$classifier_dir/classifier_best.pt\" ]]; then
    echo \"Skipping classifier \$name; found classifier_best.pt\"
    return 0
  fi

  local loss_args=()
  case \"\$loss_mode\" in
    asl_precision)
      loss_args=(--loss asl --asl-gamma-neg 4.0 --asl-clip 0.05)
      ;;
    asl_balanced)
      loss_args=(--loss asl --asl-gamma-neg 3.0 --asl-clip 0.05)
      ;;
    soft_fbeta_precision)
      loss_args=(--loss soft_fbeta --soft-fbeta-beta 0.5 --soft-fbeta-bce-weight 0.20)
      ;;
    *)
      echo \"Unknown loss mode: \$loss_mode\" >&2
      return 1
      ;;
  esac

  local cmd=(
    /opt/pytorch/bin/python -u train_classifier.py
    --data-dir \"\$ROOT\"
    --output-dir \"\$classifier_dir\"
    --architecture \"\$architecture\"
    --image-size \"\$image_size\"
    --epochs 10
    --batch-size \"\$batch_size\"
    --grad-accum-steps \"\$grad_accum\"
    --num-workers 4
    --lr \"\$lr\"
    --weight-decay \"\$weight_decay\"
    --augmentation cxr
    \"\${loss_args[@]}\"
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

  echo \"=== Training \$name (\$architecture, \$loss_mode) ===\"
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
      --classifier-batch-size 4 \
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

train_variant effb3_meta_asl_precision efficientnet_b3 512 8 2 0.00005 0.0005 asl_precision
train_variant effb3_meta_softfbeta efficientnet_b3 512 8 2 0.00003 0.0005 soft_fbeta_precision
train_variant densenet169_meta_asl_precision densenet169 512 8 2 0.00005 0.0005 asl_precision
train_variant densenet169_meta_asl_balanced densenet169 512 8 2 0.00005 0.0005 asl_balanced
train_variant xrv_rsna_meta_asl_balanced xrv_densenet121_rsna 224 32 1 0.00005 0.0005 asl_balanced
train_variant timm_effb3_meta_asl_precision timm_efficientnet_b3 512 8 2 0.00003 0.0005 asl_precision

declare -A CKPTS=()
CKPTS[swin]=\"\$BEST_SWIN\"
if [[ -n \"\$BEST_CONVNEXTV2\" ]]; then
  CKPTS[convnextv2]=\"\$BEST_CONVNEXTV2\"
fi
for name in effb3_meta_asl_precision effb3_meta_softfbeta densenet169_meta_asl_precision densenet169_meta_asl_balanced xrv_rsna_meta_asl_balanced timm_effb3_meta_asl_precision; do
  path=\"\$OUT/classifiers/\$name/classifier_best.pt\"
  if [[ -f \"\$path\" ]]; then
    CKPTS[\$name]=\"\$path\"
  fi
done

BASE_EXTRA=\"\"
CALIBRATED_EXTRA=\"--detector-score-calibration classifier_min\"
TIGHT_BOX_EXTRA=\"--detector-post-nms-iou 0.30 --detector-min-area-ratio 0.0005\"

for key in \"\${!CKPTS[@]}\"; do
  run_fusion \"single_\$key\" \"\${CKPTS[\$key]}\" \"\$BASE_EXTRA\"
done

for key in effb3_meta_asl_precision effb3_meta_softfbeta densenet169_meta_asl_precision densenet169_meta_asl_balanced xrv_rsna_meta_asl_balanced timm_effb3_meta_asl_precision convnextv2; do
  if [[ -n \"\${CKPTS[\$key]:-}\" ]]; then
    run_fusion \"swin_plus_\$key\" \"\${CKPTS[swin]},\${CKPTS[\$key]}\" \"\$BASE_EXTRA\"
    run_fusion \"swin_plus_\${key}_detcal\" \"\${CKPTS[swin]},\${CKPTS[\$key]}\" \"\$CALIBRATED_EXTRA\"
  fi
done

if [[ -n \"\${CKPTS[effb3_meta_asl_precision]:-}\" && -n \"\${CKPTS[densenet169_meta_asl_precision]:-}\" ]]; then
  run_fusion \"swin_effb3_densenet169\" \"\${CKPTS[swin]},\${CKPTS[effb3_meta_asl_precision]},\${CKPTS[densenet169_meta_asl_precision]}\" \"\$BASE_EXTRA\"
  run_fusion \"swin_effb3_densenet169_tightbox\" \"\${CKPTS[swin]},\${CKPTS[effb3_meta_asl_precision]},\${CKPTS[densenet169_meta_asl_precision]}\" \"\$TIGHT_BOX_EXTRA\"
fi

if [[ -n \"\${CKPTS[convnextv2]:-}\" && -n \"\${CKPTS[effb3_meta_asl_precision]:-}\" ]]; then
  run_fusion \"swin_convnextv2_effb3\" \"\${CKPTS[swin]},\${CKPTS[convnextv2]},\${CKPTS[effb3_meta_asl_precision]}\" \"\$BASE_EXTRA\"
fi

if [[ -n \"\${CKPTS[convnextv2]:-}\" && -n \"\${CKPTS[densenet169_meta_asl_precision]:-}\" ]]; then
  run_fusion \"swin_convnextv2_densenet169\" \"\${CKPTS[swin]},\${CKPTS[convnextv2]},\${CKPTS[densenet169_meta_asl_precision]}\" \"\$BASE_EXTRA\"
fi

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('/home/ec2-user/rsna/outputs_final_push_070')
rows = []

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
    raise SystemExit('No completed final-push results found.')

summary = summary.sort_values(['image_f1', 'image_precision', 'image_recall'], ascending=[False, False, False])
summary.to_csv(root / 'final_push_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(root / 'final_push_summary.json').write_text(json.dumps({'best': best, 'num_rows': len(summary)}, indent=2), encoding='utf-8')
(root / 'FINAL_PUSH_COMPLETE').write_text(
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
chmod +x run_final_push_070.sh
if [[ -f final_push_070.pid ]] && kill -0 \$(cat final_push_070.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat final_push_070.pid)
else
  nohup ./run_final_push_070.sh > final_push_070.log 2>&1 < /dev/null &
  echo \$! > final_push_070.pid
  echo started_pid=\$(cat final_push_070.pid)
fi"

cat <<EOF

Final 0.70 push is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f final_push_070.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_final_push_070/FINAL_PUSH_COMPLETE || ps -p \$(cat final_push_070.pid) -o pid,etime,cmd"

Show the ranked summary when it finishes:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_final_push_070/final_push_summary.csv | head -30"
EOF

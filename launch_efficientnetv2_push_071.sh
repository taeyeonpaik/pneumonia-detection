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

echo "Uploading EfficientNetV2 push scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/train_classifier.py" \
  "$LOCAL_DIR/fusion_sweep.py" \
  "$LOCAL_DIR/testing_FRCNN.py" \
  "$LOCAL_DIR/pipeline_FRCNN.py" \
  "$LOCAL_DIR/dataset.py" \
  "$LOCAL_DIR/dicom_metadata.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting EfficientNetV2 push on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_efficientnetv2_push_071.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_efficientnetv2_push_071
METADATA=\$ROOT/dicom_metadata.csv
mkdir -p \"\$OUT\" \"\$OUT/logs\" \"\$OUT/classifiers\" \"\$OUT/fusion\"

if [[ ! -f \"\$METADATA\" ]]; then
  echo \"Missing metadata CSV: \$METADATA\" >&2
  exit 1
fi

echo \"Checking torchvision EfficientNetV2-S availability...\"
if ! /opt/pytorch/bin/python - <<'PY'
import torchvision
model_fn = getattr(torchvision.models, 'efficientnet_v2_s', None)
weights = getattr(torchvision.models, 'EfficientNet_V2_S_Weights', None)
if model_fn is None or weights is None:
    raise SystemExit(
        f'Missing torchvision EfficientNetV2-S support; torchvision={torchvision.__version__}'
    )
print(f'torchvision={torchvision.__version__}, architecture=efficientnet_v2_s, weights={weights.DEFAULT}')
PY
then
  echo \"EfficientNetV2-S is unavailable in this torchvision install.\" >&2
  exit 1
fi

DETECTOR_CKPT=\"\"
for candidate in \
  \"\$ROOT/outputs_overnight_rebuild/detector_a_final/frcnn_epoch_7.pt\" \
  \"\$ROOT/outputs_reproduce_best_analysis/detector_hardneg2_adamw_b16_lr0001/frcnn_epoch_7.pt\" \
  \"\$ROOT/outputs_hardneg2_adamw_b16_lr0001/frcnn_epoch_7.pt\"; do
  if [[ -f \"\$candidate\" ]]; then
    DETECTOR_CKPT=\"\$candidate\"
    break
  fi
done

SWIN_CKPT=\"\"
for candidate in \
  \"\$ROOT/outputs_overnight_rebuild/swin_t_hardneg_metadata/classifier_best.pt\" \
  \"\$ROOT/outputs_reproduce_best_analysis/swin_t_hardneg_metadata/classifier_best.pt\" \
  \"\$ROOT/outputs_original_iou_metadata/swin_t_hardneg_metadata/classifier_best.pt\"; do
  if [[ -f \"\$candidate\" ]]; then
    SWIN_CKPT=\"\$candidate\"
    break
  fi
done

CONVNEXT384_CKPT=\"\$ROOT/outputs_convnextv2_recovery_070/classifiers/convnextv2_meta_asl_balanced_384/classifier_best.pt\"

if [[ -z \"\$DETECTOR_CKPT\" ]]; then
  echo \"Could not find detector checkpoint.\" >&2
  exit 1
fi
if [[ -z \"\$SWIN_CKPT\" ]]; then
  echo \"Could not find metadata-aware Swin checkpoint.\" >&2
  exit 1
fi
if [[ ! -f \"\$CONVNEXT384_CKPT\" ]]; then
  echo \"Missing ConvNeXtV2-384 checkpoint: \$CONVNEXT384_CKPT\" >&2
  exit 1
fi

HARDNEG=\"\"
for candidate in \
  \"\$ROOT/outputs_shared_fp_hardneg_swin/shared_fp_mining/hard_negative_patient_ids.csv\" \
  \"\$ROOT/outputs_best_swin_hardneg_mining/hard_negative_patient_ids.csv\" \
  \"\$ROOT/outputs_best_swin_hardneg_mining/hard_negative_ids.csv\"; do
  if [[ -f \"\$candidate\" ]]; then
    HARDNEG=\"\$candidate\"
    break
  fi
done

echo \"Using detector: \$DETECTOR_CKPT\"
echo \"Using Swin: \$SWIN_CKPT\"
echo \"Using ConvNeXtV2 384: \$CONVNEXT384_CKPT\"
echo \"Using hard negatives: \${HARDNEG:-none}\"

TOTAL_STAGES=8
STAGE_INDEX=0

train_classifier_variant() {
  local name=\"\$1\"
  local loss=\"\$2\"
  local gamma_neg=\"\$3\"
  local extra_loss_args=\"\$4\"
  local model_dir=\"\$OUT/classifiers/\$name\"
  STAGE_INDEX=\$((STAGE_INDEX + 1))
  mkdir -p \"\$model_dir\"
  if [[ -f \"\$model_dir/classifier_best.pt\" ]]; then
    echo \"=== [\$STAGE_INDEX/\$TOTAL_STAGES] Skipping classifier \$name; classifier_best.pt exists ===\"
    return 0
  fi
  echo \"=== [\$STAGE_INDEX/\$TOTAL_STAGES] Training \$name ===\"
  local hn_args=()
  if [[ -n \"\$HARDNEG\" ]]; then
    hn_args=(--hard-negative-file \"\$HARDNEG\" --hard-negative-weight 3.0)
  fi
  # shellcheck disable=SC2206
  local extra=(\$extra_loss_args)
  /opt/pytorch/bin/python -u train_classifier.py \
    --data-dir \"\$ROOT\" \
    --output-dir \"\$model_dir\" \
    --metadata-csv \"\$METADATA\" \
    --architecture efficientnet_v2_s \
    --image-size 512 \
    --epochs 10 \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --num-workers 4 \
    --lr 0.00003 \
    --weight-decay 0.001 \
    --augmentation cxr \
    --loss \"\$loss\" \
    --asl-gamma-neg \"\$gamma_neg\" \
    --asl-clip 0.05 \
    --balanced-batches \
    --mixup-alpha 0.05 \
    --label-smoothing 0.03 \
    --eval-tta-modes none,hflip,clahe \
    --thresholds 0.30,0.34,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.76,0.80 \
    --seed 2718 \
    --no-save-epoch-checkpoints \
    \"\${hn_args[@]}\" \
    \"\${extra[@]}\" \
    2>&1 | tee \"\$OUT/logs/\$name.train.log\"
}

train_classifier_variant effv2s_meta_asl_balanced asl 3.0 \"\"
train_classifier_variant effv2s_meta_asl_precision asl 4.0 \"\"

run_fusion() {
  local name=\"\$1\"
  local checkpoints=\"\$2\"
  local fusion_dir=\"\$OUT/fusion/\$name\"
  STAGE_INDEX=\$((STAGE_INDEX + 1))
  mkdir -p \"\$fusion_dir\"
  if [[ -f \"\$fusion_dir/fusion_sweep_test_metrics.csv\" ]]; then
    echo \"=== [\$STAGE_INDEX/\$TOTAL_STAGES] Skipping fusion \$name; metrics already exist ===\"
    return 0
  fi
  echo \"=== [\$STAGE_INDEX/\$TOTAL_STAGES] Fusion \$name ===\"
  /opt/pytorch/bin/python -u fusion_sweep.py \
    --data-dir \"\$ROOT\" \
    --output-dir \"\$fusion_dir\" \
    --metadata-csv \"\$METADATA\" \
    --classifier-checkpoints \"\$checkpoints\" \
    --detector-checkpoint \"\$DETECTOR_CKPT\" \
    --detector-min-size 1024 \
    --detector-ensemble-method none \
    --detector-score-calibration classifier_min \
    --classifier-tta-modes none,hflip \
    --classifier-temperature-mode validation \
    --methods meta,precision_gate \
    --meta-fit-split train \
    --box-score-thresholds 0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74 \
    --meta-thresholds 0.34,0.38,0.42,0.46,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.66 \
    --meta-pos-weights 1.0,1.25,1.5,2.0,2.5,3.0 \
    --meta-l2-values 0.0,0.0001,0.001,0.003,0.006,0.01 \
    --meta-epochs 1000 \
    --meta-lr 0.02 \
    --gate-classifier-agree-thresholds 0.50,0.54,0.58,0.62,0.66 \
    --gate-detector-required-thresholds 0.35,0.45,0.50,0.54,0.58,0.62 \
    --gate-classifier-override-thresholds 0.78,0.82,0.84,0.86,0.88 \
    --gate-detector-override-thresholds 0.54,0.60,0.65,0.70,0.75 \
    --gate-detector-classifier-max-thresholds 0.20,0.25,0.35,1.0 \
    --metric-iou-threshold 0.5 \
    --batch-size 16 \
    --classifier-batch-size 4 \
    --num-workers 0 \
    --min-precision 0 \
    --min-recall 0 \
    --seed 42 \
    2>&1 | tee \"\$OUT/logs/\$name.fusion.log\"
}

EFF_BAL=\"\$OUT/classifiers/effv2s_meta_asl_balanced/classifier_best.pt\"
EFF_PREC=\"\$OUT/classifiers/effv2s_meta_asl_precision/classifier_best.pt\"

run_fusion single_effv2s_balanced \"\$EFF_BAL\"
run_fusion single_effv2s_precision \"\$EFF_PREC\"
run_fusion swin_effv2s_balanced \"\$SWIN_CKPT,\$EFF_BAL\"
run_fusion swin_convnext384_effv2s_balanced \"\$SWIN_CKPT,\$CONVNEXT384_CKPT,\$EFF_BAL\"
run_fusion swin_convnext384_effv2s_precision \"\$SWIN_CKPT,\$CONVNEXT384_CKPT,\$EFF_PREC\"
run_fusion swin_convnext384_effv2s_both \"\$SWIN_CKPT,\$CONVNEXT384_CKPT,\$EFF_BAL,\$EFF_PREC\"

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('/home/ec2-user/rsna')
out = root / 'outputs_efficientnetv2_push_071'
rows = []

for baseline_path in [
    root / 'outputs_trainfit_fusion_push_070' / 'trainfit_fusion_summary.csv',
    root / 'outputs_oof_meta_fusion_071' / 'oof_meta_fusion_summary.csv',
    root / 'outputs_stratified_bagged_meta_clean_071' / 'bagged_meta_fusion_summary.csv',
]:
    if baseline_path.exists():
        table = pd.read_csv(baseline_path).sort_values(
            ['image_f1', 'image_precision', 'image_recall'],
            ascending=[False, False, False],
        ).head(1)
        for row in table.to_dict('records'):
            rows.append({'source': baseline_path.parent.name, 'variant': 'existing_best', **row})

for path in sorted((out / 'fusion').glob('*/fusion_sweep_test_metrics.csv')):
    variant = path.parent.name
    table = pd.read_csv(path)
    for row in table.to_dict('records'):
        rows.append({'source': 'efficientnetv2_push', 'variant': variant, **row})

if not rows:
    raise SystemExit('No fusion results found.')

summary = pd.DataFrame(rows).sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
)
summary.to_csv(out / 'efficientnetv2_push_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(out / 'efficientnetv2_push_summary.json').write_text(
    json.dumps({'best': best, 'num_rows': len(summary)}, indent=2),
    encoding='utf-8',
)
(out / 'EFFICIENTNETV2_PUSH_COMPLETE.txt').write_text(
    f\"Best source: {best.get('source')}\\n\"
    f\"Best variant: {best.get('variant')}\\n\"
    f\"Method: {best.get('method')}\\n\"
    f\"F1: {best.get('image_f1')}\\n\"
    f\"Precision: {best.get('image_precision')}\\n\"
    f\"Recall: {best.get('image_recall')}\\n\"
    f\"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\\n\"
    f\"Summary: {out / 'efficientnetv2_push_summary.csv'}\\n\",
    encoding='utf-8',
)
print(summary.head(40).to_string(index=False))
PY
REMOTE
chmod +x run_efficientnetv2_push_071.sh
if [[ -f efficientnetv2_push_071.pid ]] && kill -0 \$(cat efficientnetv2_push_071.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat efficientnetv2_push_071.pid)
else
  nohup ./run_efficientnetv2_push_071.sh > efficientnetv2_push_071.log 2>&1 < /dev/null &
  echo \$! > efficientnetv2_push_071.pid
  echo started_pid=\$(cat efficientnetv2_push_071.pid)
fi"

cat <<EOF

EfficientNetV2 push is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f efficientnetv2_push_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_efficientnetv2_push_071/EFFICIENTNETV2_PUSH_COMPLETE.txt || ps -p \$(cat efficientnetv2_push_071.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_efficientnetv2_push_071/efficientnetv2_push_summary.csv | head -40"
EOF

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

echo "Uploading train-fit fusion scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/fusion_sweep.py" \
  "$LOCAL_DIR/train_classifier.py" \
  "$LOCAL_DIR/testing_FRCNN.py" \
  "$LOCAL_DIR/pipeline_FRCNN.py" \
  "$LOCAL_DIR/dataset.py" \
  "$LOCAL_DIR/dicom_metadata.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting train-fit fusion push on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_trainfit_fusion_push_070.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_trainfit_fusion_push_070
mkdir -p \"\$OUT\" \"\$OUT/logs\" \"\$OUT/fusion\"

METADATA=\$ROOT/dicom_metadata.csv
if [[ ! -f \"\$METADATA\" ]]; then
  echo \"Missing metadata CSV: \$METADATA\" >&2
  exit 1
fi

DETECTOR_CKPT=\"\"
for candidate in \
  \"\$ROOT/outputs_reproduce_best_analysis/detector_hardneg2_adamw_b16_lr0001/frcnn_epoch_7.pt\" \
  \"\$ROOT/outputs_overnight_rebuild/detector_a_final/frcnn_epoch_7.pt\" \
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

DENSENET_CKPT=\"\$ROOT/outputs_final_push_070/classifiers/densenet169_meta_asl_balanced/classifier_best.pt\"
CONVNEXT384_CKPT=\"\$ROOT/outputs_convnextv2_recovery_070/classifiers/convnextv2_meta_asl_balanced_384/classifier_best.pt\"
CONVNEXT512P_CKPT=\"\$ROOT/outputs_convnextv2_recovery_070/classifiers/convnextv2_meta_asl_precision_512/classifier_best.pt\"

if [[ -z \"\$DETECTOR_CKPT\" ]]; then
  echo \"Could not find detector checkpoint.\" >&2
  exit 1
fi
if [[ -z \"\$SWIN_CKPT\" ]]; then
  echo \"Could not find Swin checkpoint.\" >&2
  exit 1
fi
if [[ ! -f \"\$DENSENET_CKPT\" ]]; then
  echo \"Missing DenseNet checkpoint: \$DENSENET_CKPT\" >&2
  exit 1
fi

echo \"Using detector: \$DETECTOR_CKPT\"
echo \"Using Swin: \$SWIN_CKPT\"
echo \"Using DenseNet169: \$DENSENET_CKPT\"
echo \"Using ConvNeXtV2 384 if present: \${CONVNEXT384_CKPT}\"
echo \"Using ConvNeXtV2 512 precision if present: \${CONVNEXT512P_CKPT}\"

run_trainfit() {
  local name=\"\$1\"
  local checkpoints=\"\$2\"
  local extra_args=\"\$3\"
  local fusion_dir=\"\$OUT/fusion/\$name\"
  mkdir -p \"\$fusion_dir\"

  if [[ -f \"\$fusion_dir/fusion_sweep_test_metrics.csv\" ]]; then
    echo \"Skipping \$name; found fusion_sweep_test_metrics.csv\"
    return 0
  fi

  echo \"=== Train-fit fusion \$name ===\"
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
    --methods meta,mlp \
    --meta-fit-split train \
    --mlp-fit-split train \
    --box-score-thresholds 0.50,0.54,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80 \
    --meta-thresholds 0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
    --meta-pos-weights 0.75,1.0,1.25,1.5,2.0,2.5,3.0 \
    --meta-l2-values 0.0,0.0001,0.001,0.01 \
    --meta-epochs 1000 \
    --meta-lr 0.02 \
    --mlp-thresholds 0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
    --mlp-pos-weights 0.75,1.0,1.25,1.5,2.0 \
    --mlp-l2-values 0.0001,0.001,0.01 \
    --mlp-hidden-dims 8,16,32 \
    --mlp-dropout-values 0.10,0.20,0.35 \
    --mlp-epochs 1000 \
    --mlp-lr 0.003 \
    --metric-iou-threshold 0.5 \
    --batch-size 16 \
    --classifier-batch-size 4 \
    --num-workers 0 \
    --min-precision 0 \
    --min-recall 0 \
    --seed 42 \
    \"\${extra[@]}\" \
    > \"\$OUT/logs/\$name.log\" 2>&1; then
    echo \"WARNING: train-fit fusion \$name failed. See \$OUT/logs/\$name.log\" >&2
    return 0
  fi
}

run_trainfit swin_densenet_trainfit \
  \"\$SWIN_CKPT,\$DENSENET_CKPT\" \
  \"\"

if [[ -f \"\$CONVNEXT384_CKPT\" ]]; then
  run_trainfit swin_convnext384_trainfit \
    \"\$SWIN_CKPT,\$CONVNEXT384_CKPT\" \
    \"--detector-score-calibration classifier_min\"
  run_trainfit swin_densenet_convnext384_trainfit \
    \"\$SWIN_CKPT,\$DENSENET_CKPT,\$CONVNEXT384_CKPT\" \
    \"--detector-score-calibration classifier_min\"
fi

if [[ -f \"\$CONVNEXT512P_CKPT\" ]]; then
  run_trainfit swin_convnext512p_trainfit \
    \"\$SWIN_CKPT,\$CONVNEXT512P_CKPT\" \
    \"--detector-score-calibration classifier_min\"
  run_trainfit swin_densenet_convnext512p_trainfit \
    \"\$SWIN_CKPT,\$DENSENET_CKPT,\$CONVNEXT512P_CKPT\" \
    \"--detector-score-calibration classifier_min\"
fi

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('/home/ec2-user/rsna/outputs_trainfit_fusion_push_070')
rows = []

for baseline_path in [
    Path('/home/ec2-user/rsna/outputs_flexible_rule_push_070/flexible_rule_push_summary.csv'),
    Path('/home/ec2-user/rsna/outputs_final_push_070/final_push_summary.csv'),
    Path('/home/ec2-user/rsna/outputs_convnextv2_recovery_070/convnextv2_recovery_summary.csv'),
]:
    if baseline_path.exists():
        table = pd.read_csv(baseline_path).sort_values(['image_f1', 'image_precision'], ascending=[False, False]).head(1)
        for row in table.to_dict('records'):
            rows.append({'source': baseline_path.parent.name, 'variant': 'existing_best', **row})

for path in sorted((root / 'fusion').glob('*/fusion_sweep_test_metrics.csv')):
    variant = path.parent.name
    table = pd.read_csv(path)
    for row in table.to_dict('records'):
        rows.append({'source': 'trainfit_fusion', 'variant': variant, **row})

summary = pd.DataFrame(rows)
if summary.empty:
    raise SystemExit('No train-fit fusion results found.')
summary = summary.sort_values(['image_f1', 'image_precision', 'image_recall'], ascending=[False, False, False])
summary.to_csv(root / 'trainfit_fusion_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(root / 'trainfit_fusion_summary.json').write_text(json.dumps({'best': best, 'num_rows': len(summary)}, indent=2), encoding='utf-8')
(root / 'TRAINFIT_FUSION_COMPLETE').write_text(
    f\"Best source: {best.get('source')}\\n\"
    f\"Best variant: {best.get('variant')}\\n\"
    f\"Selected from: {best.get('selected_from')}\\n\"
    f\"Method: {best.get('method')}\\n\"
    f\"F1: {best.get('image_f1')}\\n\"
    f\"Precision: {best.get('image_precision')}\\n\"
    f\"Recall: {best.get('image_recall')}\\n\"
    f\"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\\n\",
    encoding='utf-8',
)
print(summary.head(30).to_string(index=False))
PY
REMOTE
chmod +x run_trainfit_fusion_push_070.sh
if [[ -f trainfit_fusion_push_070.pid ]] && kill -0 \$(cat trainfit_fusion_push_070.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat trainfit_fusion_push_070.pid)
else
  nohup ./run_trainfit_fusion_push_070.sh > trainfit_fusion_push_070.log 2>&1 < /dev/null &
  echo \$! > trainfit_fusion_push_070.pid
  echo started_pid=\$(cat trainfit_fusion_push_070.pid)
fi"

cat <<EOF

Train-fit fusion push is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f trainfit_fusion_push_070.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_trainfit_fusion_push_070/TRAINFIT_FUSION_COMPLETE || ps -p \$(cat trainfit_fusion_push_070.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_trainfit_fusion_push_070/trainfit_fusion_summary.csv | head -30"
EOF

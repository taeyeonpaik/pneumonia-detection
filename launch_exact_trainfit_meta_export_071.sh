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

echo "Uploading patched fusion export scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/fusion_sweep.py" \
  "$LOCAL_DIR/train_classifier.py" \
  "$LOCAL_DIR/testing_FRCNN.py" \
  "$LOCAL_DIR/pipeline_FRCNN.py" \
  "$LOCAL_DIR/dataset.py" \
  "$LOCAL_DIR/dicom_metadata.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting exact train-fit meta export on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_exact_trainfit_meta_export_071.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_exact_trainfit_meta_export_071
FUSION_DIR=\$OUT/swin_convnext384_trainfit_exact_export
mkdir -p \"\$OUT\" \"\$OUT/logs\" \"\$FUSION_DIR\"

METADATA=\$ROOT/dicom_metadata.csv
if [[ ! -f \"\$METADATA\" ]]; then
  echo \"Missing metadata CSV: \$METADATA\" >&2
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

echo \"Using detector: \$DETECTOR_CKPT\"
echo \"Using metadata-aware Swin: \$SWIN_CKPT\"
echo \"Using metadata-aware ConvNeXtV2-384: \$CONVNEXT384_CKPT\"
echo \"Writing fusion export to: \$FUSION_DIR\"

if [[ -f \"\$FUSION_DIR/fusion_sweep_validation_predictions.csv\" ]]; then
  echo \"Skipping exact export; found \$FUSION_DIR/fusion_sweep_validation_predictions.csv\"
else
  /opt/pytorch/bin/python -u fusion_sweep.py \
    --data-dir \"\$ROOT\" \
    --output-dir \"\$FUSION_DIR\" \
    --metadata-csv \"\$METADATA\" \
    --classifier-checkpoints \"\$SWIN_CKPT,\$CONVNEXT384_CKPT\" \
    --detector-checkpoint \"\$DETECTOR_CKPT\" \
    --detector-min-size 1024 \
    --detector-ensemble-method none \
    --detector-score-calibration classifier_min \
    --classifier-tta-modes none,hflip \
    --classifier-temperature-mode validation \
    --methods meta \
    --meta-fit-split train \
    --box-score-thresholds 0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
    --meta-thresholds 0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62 \
    --meta-pos-weights 2.0,2.25,2.5,2.75,3.0 \
    --meta-l2-values 0.003,0.006,0.01,0.02 \
    --meta-epochs 1000 \
    --meta-lr 0.02 \
    --metric-iou-threshold 0.5 \
    --batch-size 16 \
    --classifier-batch-size 4 \
    --num-workers 0 \
    --min-precision 0 \
    --min-recall 0 \
    --seed 42 \
    2>&1 | tee \"\$OUT/logs/swin_convnext384_exact_export.log\"
fi

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('/home/ec2-user/rsna')
out = root / 'outputs_exact_trainfit_meta_export_071'
fusion_dir = out / 'swin_convnext384_trainfit_exact_export'
rows = []

locked_summary = root / 'outputs_trainfit_fusion_push_070' / 'trainfit_fusion_summary.csv'
if locked_summary.exists():
    locked = pd.read_csv(locked_summary).sort_values(
        ['image_f1', 'image_precision', 'image_recall'],
        ascending=[False, False, False],
    ).head(1)
    for row in locked.to_dict('records'):
        rows.append({'source': 'locked_best_0700882', **row})

metrics_path = fusion_dir / 'fusion_sweep_test_metrics.csv'
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    for row in metrics.to_dict('records'):
        rows.append({'source': 'exact_export_rerun', 'variant': fusion_dir.name, **row})

if not rows:
    raise SystemExit('No locked or rerun rows found.')

summary = pd.DataFrame(rows).sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
)
summary.to_csv(out / 'exact_trainfit_meta_export_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(out / 'exact_trainfit_meta_export_summary.json').write_text(
    json.dumps({'best': best, 'num_rows': len(summary)}, indent=2),
    encoding='utf-8',
)

required = [
    fusion_dir / 'fusion_sweep_validation_predictions.csv',
    fusion_dir / 'fusion_sweep_test_predictions.csv',
    fusion_dir / 'fusion_train_features.csv',
    fusion_dir / 'fusion_validation_features.csv',
    fusion_dir / 'fusion_test_features.csv',
]
missing = [str(path) for path in required if not path.exists()]
(out / 'EXACT_TRAINFIT_META_EXPORT_COMPLETE.txt').write_text(
    f\"Best source: {best.get('source')}\\n\"
    f\"Best variant: {best.get('variant')}\\n\"
    f\"Method: {best.get('method')}\\n\"
    f\"F1: {best.get('image_f1')}\\n\"
    f\"Precision: {best.get('image_precision')}\\n\"
    f\"Recall: {best.get('image_recall')}\\n\"
    f\"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\\n\"
    f\"Validation predictions: {fusion_dir / 'fusion_sweep_validation_predictions.csv'}\\n\"
    f\"Test predictions: {fusion_dir / 'fusion_sweep_test_predictions.csv'}\\n\"
    f\"Missing required files: {missing}\\n\",
    encoding='utf-8',
)
print(summary.to_string(index=False))
if missing:
    raise SystemExit(f'Missing expected export files: {missing}')
PY
REMOTE
chmod +x run_exact_trainfit_meta_export_071.sh
if [[ -f exact_trainfit_meta_export_071.pid ]] && kill -0 \$(cat exact_trainfit_meta_export_071.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat exact_trainfit_meta_export_071.pid)
else
  nohup ./run_exact_trainfit_meta_export_071.sh > exact_trainfit_meta_export_071.log 2>&1 < /dev/null &
  echo \$! > exact_trainfit_meta_export_071.pid
  echo started_pid=\$(cat exact_trainfit_meta_export_071.pid)
fi"

cat <<EOF

Exact train-fit meta export is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f exact_trainfit_meta_export_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_exact_trainfit_meta_export_071/EXACT_TRAINFIT_META_EXPORT_COMPLETE.txt || ps -p \$(cat exact_trainfit_meta_export_071.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_exact_trainfit_meta_export_071/exact_trainfit_meta_export_summary.csv"
EOF

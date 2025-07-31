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

echo "Uploading shared-FP hard-negative scripts..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/train_classifier.py" \
  "$LOCAL_DIR/mine_shared_false_positives.py" \
  "$LOCAL_DIR/stratified_precision_gate_sweep.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting shared-FP hard-negative experiment on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_shared_fp_hardneg_experiment.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_shared_fp_hardneg_swin
mkdir -p \"\$OUT\"

METADATA=\$ROOT/dicom_metadata.csv
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

BASE_SWIN=\"\"
for candidate in \
  \"\$ROOT/outputs_reproduce_best_analysis/swin_t_hardneg_metadata/classifier_best.pt\" \
  \"\$ROOT/outputs_overnight_rebuild/swin_t_hardneg_metadata/classifier_best.pt\" \
  \"\$ROOT/outputs_original_iou_metadata/swin_t_hardneg_metadata/classifier_best.pt\"; do
  if [[ -f \"\$candidate\" ]]; then
    BASE_SWIN=\"\$candidate\"
    break
  fi
done

if [[ ! -f \"\$METADATA\" ]]; then
  echo \"Missing metadata CSV: \$METADATA\" >&2
  exit 1
fi
if [[ -z \"\$DETECTOR_CKPT\" ]]; then
  echo \"Could not find detector checkpoint.\" >&2
  exit 1
fi
if [[ -z \"\$BASE_SWIN\" ]]; then
  echo \"Could not find metadata-aware Swin checkpoint.\" >&2
  exit 1
fi

echo \"Using detector: \$DETECTOR_CKPT\"
echo \"Using base metadata Swin: \$BASE_SWIN\"

MINE_DIR=\$OUT/shared_fp_mining
if [[ ! -f \"\$MINE_DIR/hard_negative_patient_ids.csv\" ]]; then
  /opt/pytorch/bin/python -u mine_shared_false_positives.py \
    --data-dir \"\$ROOT\" \
    --output-dir \"\$MINE_DIR\" \
    --classifier-checkpoint \"\$BASE_SWIN\" \
    --detector-checkpoint \"\$DETECTOR_CKPT\" \
    --metadata-csv \"\$METADATA\" \
    --classifier-threshold 0.66 \
    --detector-threshold 0.44 \
    --min-classifier-probability 0.62 \
    --min-detector-score 0.35 \
    --max-hard-negatives 2500 \
    --batch-size 16 \
    --classifier-batch-size 8 \
    --num-workers 4 \
    --detector-min-size 1024 \
    --tta-modes none,hflip \
    --seed 42
else
  echo \"Skipping shared-FP mining; found \$MINE_DIR/hard_negative_patient_ids.csv\"
fi

run_variant() {
  local name=\"\$1\"
  local weight=\"\$2\"
  local loss=\"\$3\"
  local lr=\"\$4\"
  local epochs=\"\$5\"
  local classifier_dir=\"\$OUT/\$name\"
  local fusion_dir=\"\$OUT/fusion_\$name\"
  local strat_dir=\"\$OUT/stratified_\$name\"

  if [[ ! -f \"\$classifier_dir/classifier_best.pt\" ]]; then
    if [[ \"\$loss\" == \"asl\" ]]; then
      loss_args=(--loss asl --asl-gamma-neg 3.0 --asl-clip 0.05)
    elif [[ \"\$loss\" == \"soft_fbeta\" ]]; then
      loss_args=(--loss soft_fbeta --soft-fbeta-beta 0.5 --soft-fbeta-bce-weight 0.20)
    else
      echo \"Unknown loss: \$loss\" >&2
      exit 1
    fi
    /opt/pytorch/bin/python -u train_classifier.py \
      --data-dir \"\$ROOT\" \
      --output-dir \"\$classifier_dir\" \
      --architecture swin_t \
      --image-size 512 \
      --epochs \"\$epochs\" \
      --batch-size 8 \
      --num-workers 4 \
      --lr \"\$lr\" \
      --weight-decay 0.001 \
      --augmentation cxr \
      \"\${loss_args[@]}\" \
      --label-smoothing 0.03 \
      --balanced-batches \
      --mixup-alpha 0.05 \
      --eval-tta-modes none,hflip \
      --thresholds 0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.65,0.66,0.68,0.70,0.72,0.75,0.80,0.82,0.84,0.86,0.88 \
      --metadata-csv \"\$METADATA\" \
      --resume-checkpoint \"\$BASE_SWIN\" \
      --hard-negative-file \"\$MINE_DIR/hard_negative_patient_ids.csv\" \
      --hard-negative-weight \"\$weight\" \
      --no-save-epoch-checkpoints \
      --seed 42
  else
    echo \"Skipping classifier \$name; found \$classifier_dir/classifier_best.pt\"
  fi

  if [[ ! -f \"\$fusion_dir/fusion_sweep_test_metrics.csv\" ]]; then
    /opt/pytorch/bin/python -u fusion_sweep.py \
      --data-dir \"\$ROOT\" \
      --output-dir \"\$fusion_dir\" \
      --metadata-csv \"\$METADATA\" \
      --classifier-checkpoints \"\$classifier_dir/classifier_best.pt\" \
      --detector-checkpoint \"\$DETECTOR_CKPT\" \
      --detector-min-size 1024 \
      --detector-ensemble-method none \
      --classifier-tta-modes none,hflip \
      --classifier-temperature-mode validation \
      --methods precision_gate \
      --box-score-thresholds 0.40,0.44,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80 \
      --gate-classifier-agree-thresholds 0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.65,0.66,0.68,0.70,0.72 \
      --gate-detector-required-thresholds 0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.65 \
      --gate-classifier-override-thresholds 0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.96 \
      --gate-detector-override-thresholds 0.54,0.58,0.60,0.62,0.65,0.68,0.70,0.75,0.80 \
      --gate-detector-classifier-max-thresholds 0.20,0.25,0.30,0.35,0.40,0.50,1.00 \
      --metric-iou-threshold 0.5 \
      --batch-size 16 \
      --classifier-batch-size 4 \
      --num-workers 0 \
      --min-precision 0 \
      --min-recall 0 \
      --seed 42
  else
    echo \"Skipping fusion \$name; found \$fusion_dir/fusion_sweep_test_metrics.csv\"
  fi

  if [[ ! -f \"\$strat_dir/stratified_strategy_test_metrics.csv\" ]]; then
    /opt/pytorch/bin/python -u stratified_precision_gate_sweep.py \
      --fusion-dir \"\$fusion_dir\" \
      --metadata-csv \"\$METADATA\" \
      --output-dir \"\$strat_dir\" \
      --group-columns age_bucket
  else
    echo \"Skipping stratified sweep \$name; found \$strat_dir/stratified_strategy_test_metrics.csv\"
  fi
}

run_variant sharedfp_asl_w3_lr000005 3.0 asl 0.000005 8
run_variant sharedfp_asl_w5_lr000005 5.0 asl 0.000005 8
run_variant sharedfp_softfbeta_w5_lr000005 5.0 soft_fbeta 0.000005 8

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('/home/ec2-user/rsna/outputs_shared_fp_hardneg_swin')
rows = []
for path in sorted(root.glob('fusion_*/fusion_sweep_test_metrics.csv')):
    variant = path.parent.name.replace('fusion_', '')
    row = pd.read_csv(path).iloc[0].to_dict()
    rows.append({'variant': variant, 'strategy': 'global_precision_gate', **row})
for path in sorted(root.glob('stratified_*/stratified_strategy_test_metrics.csv')):
    variant = path.parent.name.replace('stratified_', '')
    row = pd.read_csv(path).iloc[0].to_dict()
    rows.append({'variant': variant, 'strategy': 'age_stratified_precision_gate', **row})
summary = pd.DataFrame(rows).sort_values(['image_f1', 'image_precision'], ascending=[False, False])
summary.to_csv(root / 'shared_fp_hardneg_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(root / 'SHARED_FP_HARDNEG_COMPLETE').write_text(
    f\"Best variant: {best['variant']}\\n\"
    f\"Strategy: {best['strategy']}\\n\"
    f\"F1: {best['image_f1']}\\n\"
    f\"Precision: {best['image_precision']}\\n\"
    f\"Recall: {best['image_recall']}\\n\",
    encoding='utf-8',
)
(root / 'shared_fp_hardneg_summary.json').write_text(json.dumps({'best': best, 'rows': rows}, indent=2), encoding='utf-8')
print(summary.to_string(index=False))
PY
REMOTE
chmod +x run_shared_fp_hardneg_experiment.sh
nohup ./run_shared_fp_hardneg_experiment.sh > shared_fp_hardneg_experiment.log 2>&1 < /dev/null &
echo \$! > shared_fp_hardneg_experiment.pid
echo started_pid=\$(cat shared_fp_hardneg_experiment.pid)"

cat <<EOF

Shared-FP hard-negative experiment is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f shared_fp_hardneg_experiment.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_shared_fp_hardneg_swin/SHARED_FP_HARDNEG_COMPLETE || ps -p \$(cat shared_fp_hardneg_experiment.pid) -o pid,etime,cmd"

View final table with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_shared_fp_hardneg_swin/shared_fp_hardneg_summary.csv"
EOF

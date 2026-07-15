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

echo "[1/3] Uploading fusion code..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/fusion_sweep.py" \
  "$LOCAL_DIR/train_classifier.py" \
  "$LOCAL_DIR/testing_FRCNN.py" \
  "$LOCAL_DIR/pipeline_FRCNN.py" \
  "$LOCAL_DIR/dataset.py" \
  "$LOCAL_DIR/dicom_metadata.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "[2/3] Installing the remote CLAHE-TTA fusion job..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_effv2_clahe_fusion_071.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_effv2_clahe_fusion_071
FUSION_DIR=\$OUT/swin_effv2s_clahe
METADATA=\$ROOT/dicom_metadata.csv
mkdir -p \"\$OUT\" \"\$FUSION_DIR\"

DETECTOR=\$ROOT/outputs_overnight_rebuild/detector_a_final/frcnn_epoch_7.pt
SWIN=\$ROOT/outputs_overnight_rebuild/swin_t_hardneg_metadata/classifier_best.pt
EFFV2=\$ROOT/outputs_efficientnetv2_push_071/classifiers/effv2s_meta_asl_balanced/classifier_best.pt

for required in \"\$DETECTOR\" \"\$SWIN\" \"\$EFFV2\" \"\$METADATA\"; do
  if [[ ! -f \"\$required\" ]]; then
    echo \"Missing required input: \$required\" >&2
    exit 1
  fi
done

echo \"[1/2] Running Swin + EfficientNetV2-S + Faster R-CNN with none/hflip/CLAHE TTA\"
/opt/pytorch/bin/python -u fusion_sweep.py \
  --data-dir \"\$ROOT\" \
  --output-dir \"\$FUSION_DIR\" \
  --metadata-csv \"\$METADATA\" \
  --classifier-checkpoints \"\$SWIN,\$EFFV2\" \
  --detector-checkpoint \"\$DETECTOR\" \
  --detector-min-size 1024 \
  --detector-ensemble-method none \
  --detector-score-calibration classifier_min \
  --classifier-tta-modes none,hflip,clahe \
  --classifier-temperature-mode validation \
  --methods precision_gate \
  --box-score-thresholds 0.66,0.68,0.69,0.70,0.71,0.72,0.74 \
  --gate-classifier-agree-thresholds 0.50,0.54,0.56,0.58,0.60,0.62,0.64 \
  --gate-detector-required-thresholds 0.20,0.25,0.30,0.35,0.40,0.45 \
  --gate-classifier-override-thresholds 0.68,0.72,0.74,0.76,0.78,0.80,0.82 \
  --gate-detector-override-thresholds 0.40,0.45,0.50,0.54,0.58,0.62 \
  --gate-detector-classifier-max-thresholds 0.10,0.15,0.20,0.25,0.30 \
  --metric-iou-threshold 0.5 \
  --batch-size 16 \
  --classifier-batch-size 4 \
  --num-workers 0 \
  --min-precision 0 \
  --min-recall 0 \
  --seed 42

echo \"[2/2] Comparing with the locked F1=0.701365 result\"
/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('/home/ec2-user/rsna')
out = root / 'outputs_effv2_clahe_fusion_071'
metrics = pd.read_csv(out / 'swin_effv2s_clahe' / 'fusion_sweep_test_metrics.csv')
candidate = metrics.sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
).iloc[0].to_dict()
baseline = {
    'variant': 'locked_swin_effv2s_precision_gate',
    'image_f1': 0.7013651877133106,
    'image_precision': 0.6726677577741407,
    'image_recall': 0.732620320855615,
    'image_accuracy': 0.86,
    'tn': 1739,
    'fp': 200,
    'fn': 150,
    'tp': 411,
}
candidate = {'variant': 'swin_effv2s_clahe', **candidate}
summary = pd.DataFrame([baseline, candidate]).sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
)
summary.to_csv(out / 'clahe_fusion_comparison.csv', index=False)
best = summary.iloc[0].to_dict()
(out / 'clahe_fusion_report.json').write_text(
    json.dumps({'best_observed': best, 'candidate': candidate, 'baseline': baseline}, indent=2),
    encoding='utf-8',
)
(out / 'CLAHE_FUSION_COMPLETE.txt').write_text(
    f\"Best variant: {best.get('variant')}\\n\"
    f\"F1: {best.get('image_f1')}\\n\"
    f\"Precision: {best.get('image_precision')}\\n\"
    f\"Recall: {best.get('image_recall')}\\n\"
    f\"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\\n\"
    f\"Candidate F1: {candidate.get('image_f1')}\\n\",
    encoding='utf-8',
)
print((out / 'CLAHE_FUSION_COMPLETE.txt').read_text(encoding='utf-8'))
PY
REMOTE
chmod +x run_effv2_clahe_fusion_071.sh
if [[ -f effv2_clahe_fusion_071.pid ]] && kill -0 \$(cat effv2_clahe_fusion_071.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat effv2_clahe_fusion_071.pid)
else
  nohup ./run_effv2_clahe_fusion_071.sh > effv2_clahe_fusion_071.log 2>&1 < /dev/null &
  echo \$! > effv2_clahe_fusion_071.pid
  echo started_pid=\$(cat effv2_clahe_fusion_071.pid)
fi"

echo "[3/3] Job launched. Progress bars will appear in the remote log."
cat <<EOF

Watch progress:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f effv2_clahe_fusion_071.log"

Check completion:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_effv2_clahe_fusion_071/CLAHE_FUSION_COMPLETE.txt || ps -p \$(cat effv2_clahe_fusion_071.pid) -o pid,etime,cmd"
EOF

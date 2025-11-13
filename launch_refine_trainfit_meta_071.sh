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

echo "Uploading train-fit meta refinement script..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/refine_trainfit_meta_from_features.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting train-fit meta refinement on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_refine_trainfit_meta_071.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_refine_trainfit_meta_071
mkdir -p \"\$OUT\" \"\$OUT/logs\" \"\$OUT/runs\"

run_refine() {
  local label=\"\$1\"
  local fusion_dir=\"\$2\"
  local run_dir=\"\$OUT/runs/\$label\"

  if [[ ! -f \"\$fusion_dir/fusion_train_features.csv\" ]]; then
    echo \"Skipping \$label; missing train features at \$fusion_dir\"
    return 0
  fi
  if [[ -f \"\$run_dir/REFINED_METAFUSION_COMPLETE.txt\" ]]; then
    echo \"Skipping \$label; already complete\"
    return 0
  fi

  mkdir -p \"\$run_dir\"
  echo \"=== Refining train-fit meta fusion: \$label ===\"
  /opt/pytorch/bin/python -u refine_trainfit_meta_from_features.py \
    --fusion-dir \"\$fusion_dir\" \
    --output-dir \"\$run_dir\" \
    --feature-modes all,compact,classifier_detector_only \
    --pos-weights 1.75,2.0,2.25,2.5,2.75,3.0,3.25 \
    --l2-values 0.0001,0.0003,0.001,0.003,0.006,0.01,0.02 \
    --lrs 0.01,0.02 \
    --fit-seeds 42,1337 \
    --epochs 1200 \
    --thresholds 0.35,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
    > \"\$OUT/logs/\$label.log\" 2>&1
}

run_refine swin_convnext384_trainfit \
  \"\$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_convnext384_trainfit\"

run_refine swin_densenet_convnext384_trainfit \
  \"\$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_densenet_convnext384_trainfit\"

run_refine swin_convnext512p_trainfit \
  \"\$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_convnext512p_trainfit\"

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import shutil
import pandas as pd

root = Path('/home/ec2-user/rsna')
out = root / 'outputs_refine_trainfit_meta_071'
rows = []

baseline_path = root / 'outputs_trainfit_fusion_push_070' / 'trainfit_fusion_summary.csv'
if baseline_path.exists():
    baseline = pd.read_csv(baseline_path).sort_values(
        ['image_f1', 'image_precision', 'image_recall'],
        ascending=[False, False, False],
    ).head(1)
    for row in baseline.to_dict('records'):
        rows.append({'source': 'current_locked_baseline', **row})

for metrics_path in sorted((out / 'runs').glob('*/refined_test_metrics.csv')):
    label = metrics_path.parent.name
    table = pd.read_csv(metrics_path)
    for row in table.to_dict('records'):
        rows.append({'source': 'refined_trainfit_meta', 'variant': label, 'method': 'refined_meta_logistic', **row})

if not rows:
    raise SystemExit('No baseline or refined rows found.')

summary = pd.DataFrame(rows).sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
)
summary.to_csv(out / 'refine_trainfit_meta_071_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(out / 'refine_trainfit_meta_071_summary.json').write_text(
    json.dumps({'best': best, 'num_rows': len(summary)}, indent=2),
    encoding='utf-8',
)

best_run_dir = out / 'runs' / str(best.get('variant'))
if best.get('source') == 'refined_trainfit_meta' and best_run_dir.exists():
    locked = out / 'best_refined_model'
    if locked.exists():
        shutil.rmtree(locked)
    shutil.copytree(best_run_dir, locked)

(out / 'REFINE_TRAINFIT_META_071_COMPLETE.txt').write_text(
    f\"Best source: {best.get('source')}\\n\"
    f\"Best variant: {best.get('variant')}\\n\"
    f\"Method: {best.get('method')}\\n\"
    f\"F1: {best.get('image_f1')}\\n\"
    f\"Precision: {best.get('image_precision')}\\n\"
    f\"Recall: {best.get('image_recall')}\\n\"
    f\"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\\n\"
    f\"Summary: {out / 'refine_trainfit_meta_071_summary.csv'}\\n\",
    encoding='utf-8',
)
print(summary.head(30).to_string(index=False))
PY
REMOTE
chmod +x run_refine_trainfit_meta_071.sh
if [[ -f refine_trainfit_meta_071.pid ]] && kill -0 \$(cat refine_trainfit_meta_071.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat refine_trainfit_meta_071.pid)
else
  nohup ./run_refine_trainfit_meta_071.sh > refine_trainfit_meta_071.log 2>&1 < /dev/null &
  echo \$! > refine_trainfit_meta_071.pid
  echo started_pid=\$(cat refine_trainfit_meta_071.pid)
fi"

cat <<EOF

Train-fit meta refinement is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f refine_trainfit_meta_071.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_refine_trainfit_meta_071/REFINE_TRAINFIT_META_071_COMPLETE.txt || ps -p \$(cat refine_trainfit_meta_071.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_refine_trainfit_meta_071/refine_trainfit_meta_071_summary.csv | head -30"
EOF

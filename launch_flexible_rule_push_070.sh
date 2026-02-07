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

echo "Uploading flexible-rule sweep script..."
rsync -avP \
  -e "$RSYNC_SSH" \
  "$LOCAL_DIR/flexible_fusion_rule_sweep.py" \
  "$LOCAL_DIR/analyze_best_model_errors.py" \
  "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

echo "Starting flexible-rule push on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > run_flexible_rule_push_070.sh <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT=\$ROOT/outputs_flexible_rule_push_070
mkdir -p \"\$OUT\" \"\$OUT/logs\" \"\$OUT/runs\"

METADATA=\$ROOT/dicom_metadata.csv
if [[ ! -f \"\$METADATA\" ]]; then
  echo \"Missing metadata CSV: \$METADATA\" >&2
  exit 1
fi

run_flexible() {
  local label=\"\$1\"
  local fusion_dir=\"\$2\"
  local output_dir=\"\$OUT/runs/\$label\"
  if [[ ! -f \"\$fusion_dir/fusion_validation_features.csv\" ]]; then
    echo \"Skipping \$label; missing \$fusion_dir/fusion_validation_features.csv\"
    return 0
  fi
  if [[ -f \"\$output_dir/flexible_rule_test_metrics.csv\" ]]; then
    echo \"Skipping \$label; found completed flexible metrics\"
    return 0
  fi
  mkdir -p \"\$output_dir\"
  echo \"=== Flexible rule sweep \$label ===\"
  /opt/pytorch/bin/python -u flexible_fusion_rule_sweep.py \
    --fusion-dir \"\$fusion_dir\" \
    --metadata-csv \"\$METADATA\" \
    --output-dir \"\$output_dir\" \
    --group-columns age_bucket,view_group,sex_group \
    --max-rules 80000 \
    --max-group-rules 25000 \
    --seed 42 \
    > \"\$OUT/logs/\$label.log\" 2>&1
}

run_flexible final_swin_densenet169_asl_balanced \
  \"\$ROOT/outputs_final_push_070/fusion/swin_plus_densenet169_meta_asl_balanced\"
run_flexible final_single_swin \
  \"\$ROOT/outputs_final_push_070/fusion/single_swin\"
run_flexible conv_swin_convnext384_detcal \
  \"\$ROOT/outputs_convnextv2_recovery_070/fusion/swin_plus_convnextv2_meta_asl_balanced_384_detcal\"
run_flexible conv_swin_convnext512_precision_detcal \
  \"\$ROOT/outputs_convnextv2_recovery_070/fusion/swin_plus_convnextv2_meta_asl_precision_512_detcal\"
run_flexible conv_swin_densenet_convnext512_precision_detcal \
  \"\$ROOT/outputs_convnextv2_recovery_070/fusion/swin_densenet_convnextv2_meta_asl_precision_512_detcal\"
run_flexible conv_swin_densenet_convnext512_balanced \
  \"\$ROOT/outputs_convnextv2_recovery_070/fusion/swin_densenet_convnextv2_meta_asl_balanced_512\"
run_flexible conv_swin_densenet_convnext384_balanced \
  \"\$ROOT/outputs_convnextv2_recovery_070/fusion/swin_densenet_convnextv2_meta_asl_balanced_384\"

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('/home/ec2-user/rsna/outputs_flexible_rule_push_070')
rows = []

for baseline_path in [
    Path('/home/ec2-user/rsna/outputs_final_push_070/final_push_summary.csv'),
    Path('/home/ec2-user/rsna/outputs_convnextv2_recovery_070/convnextv2_recovery_summary.csv'),
]:
    if baseline_path.exists():
        table = pd.read_csv(baseline_path).sort_values(['image_f1', 'image_precision'], ascending=[False, False]).head(1)
        for row in table.to_dict('records'):
            rows.append({'source': baseline_path.parent.name, 'variant': 'existing_best', **row})

for path in sorted((root / 'runs').glob('*/flexible_rule_test_metrics.csv')):
    label = path.parent.name
    table = pd.read_csv(path)
    for row in table.to_dict('records'):
        rows.append({'source': label, 'variant': label, **row})

summary = pd.DataFrame(rows)
if summary.empty:
    raise SystemExit('No flexible-rule results found.')
summary = summary.sort_values(['image_f1', 'image_precision', 'image_recall'], ascending=[False, False, False])
summary.to_csv(root / 'flexible_rule_push_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(root / 'flexible_rule_push_summary.json').write_text(json.dumps({'best': best, 'num_rows': len(summary)}, indent=2), encoding='utf-8')
(root / 'FLEXIBLE_RULE_PUSH_COMPLETE').write_text(
    f\"Best source: {best.get('source')}\\n\"
    f\"Best variant: {best.get('variant')}\\n\"
    f\"Strategy: {best.get('strategy')}\\n\"
    f\"F1: {best.get('image_f1')}\\n\"
    f\"Precision: {best.get('image_precision')}\\n\"
    f\"Recall: {best.get('image_recall')}\\n\"
    f\"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\\n\",
    encoding='utf-8',
)
print(summary.head(30).to_string(index=False))
PY
REMOTE
chmod +x run_flexible_rule_push_070.sh
if [[ -f flexible_rule_push_070.pid ]] && kill -0 \$(cat flexible_rule_push_070.pid) 2>/dev/null; then
  echo already_running_pid=\$(cat flexible_rule_push_070.pid)
else
  nohup ./run_flexible_rule_push_070.sh > flexible_rule_push_070.log 2>&1 < /dev/null &
  echo \$! > flexible_rule_push_070.pid
  echo started_pid=\$(cat flexible_rule_push_070.pid)
fi"

cat <<EOF

Flexible-rule push is running independently on EC2.

Watch it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && tail -f flexible_rule_push_070.log"

Check completion with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_flexible_rule_push_070/FLEXIBLE_RULE_PUSH_COMPLETE || ps -p \$(cat flexible_rule_push_070.pid) -o pid,etime,cmd"

Show ranked results:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && column -s, -t < outputs_flexible_rule_push_070/flexible_rule_push_summary.csv | head -30"
EOF

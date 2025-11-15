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
SSH=(ssh -i "$KEY_FILE" -o ServerAliveInterval=30 -o ServerAliveCountMax=10)

chmod 400 "$KEY_FILE"

echo "Locking best bagged meta-fusion result on EC2..."
"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && /opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import shutil
import pandas as pd

root = Path('/home/ec2-user/rsna')
source = root / 'outputs_bagged_meta_elite_071'
dest = root / 'outputs_locked_best_f1_0701587'

summary_path = source / 'bagged_meta_fusion_summary.csv'
grid_path = source / 'bagged_meta_fusion_validation_grid.csv'
test_predictions_path = source / 'best_bagged_meta_predictions.csv'
validation_predictions_path = source / 'best_bagged_meta_validation_predictions.csv'

required = [summary_path, grid_path, test_predictions_path, validation_predictions_path]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f'Missing required output files: {missing}')

dest.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(summary_path).sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
)
best = summary.iloc[0].to_dict()

copies = {
    summary_path: dest / 'locked_best_summary.csv',
    grid_path: dest / 'bagged_meta_fusion_validation_grid.csv',
    test_predictions_path: dest / 'locked_best_test_predictions.csv',
    validation_predictions_path: dest / 'locked_best_validation_predictions.csv',
    source / 'bagged_meta_fusion_summary.json': dest / 'bagged_meta_fusion_summary.json',
    root / 'bagged_meta_fusion_from_features.py': dest / 'bagged_meta_fusion_from_features.py',
}
for src, dst in copies.items():
    if src.exists():
        shutil.copy2(src, dst)

(dest / 'LOCKED_BEST_MODEL.json').write_text(
    json.dumps(
        {
            'locked_at': '2026-07-16',
            'source_dir': str(source),
            'model_family': 'bagged logistic meta-fusion over saved fusion feature tables',
            'base_components': [
                'metadata-aware Swin-T classifier',
                'metadata-aware ConvNeXtV2-Base 384 classifier',
                'Faster R-CNN detector',
                'SwinV2-base fusion feature set when available',
            ],
            'selection_policy': 'All thresholds and bagged-fusion settings selected on validation only; test set used once for held-out reporting.',
            'best': best,
        },
        indent=2,
    ),
    encoding='utf-8',
)
(dest / 'LOCKED_BEST_MODEL.txt').write_text(
    f\"Locked best RSNA model\\n\"
    f\"Source: {source}\\n\"
    f\"Method: {best.get('method')}\\n\"
    f\"Feature mode: {best.get('feature_mode')}\\n\"
    f\"F1: {best.get('image_f1')}\\n\"
    f\"Precision: {best.get('image_precision')}\\n\"
    f\"Recall: {best.get('image_recall')}\\n\"
    f\"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\\n\"
    f\"Config: pos_weight={best.get('pos_weight')}, l2={best.get('l2')}, lr={best.get('lr')}, \"
    f\"seed={best.get('fit_seed')}, bags={best.get('num_bags')}, sample_frac={best.get('sample_frac')}, \"
    f\"feature_frac={best.get('feature_frac')}, validation_threshold={best.get('validation_threshold')}\\n\"
    f\"Test predictions: {dest / 'locked_best_test_predictions.csv'}\\n\"
    f\"Validation predictions: {dest / 'locked_best_validation_predictions.csv'}\\n\",
    encoding='utf-8',
)
print((dest / 'LOCKED_BEST_MODEL.txt').read_text(encoding='utf-8'))
PY"

cat <<EOF

Locked artifact directory:
  $REMOTE_DIR/outputs_locked_best_f1_0701587

Show it again with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_locked_best_f1_0701587/LOCKED_BEST_MODEL.txt"
EOF

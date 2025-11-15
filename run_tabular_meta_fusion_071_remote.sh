#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT="$ROOT/outputs_tabular_meta_fusion_071"
mkdir -p "$OUT" "$OUT/logs" "$OUT/runs"

echo "Checking scikit-learn..."
if ! /opt/pytorch/bin/python -c "import sklearn" >/dev/null 2>&1; then
  echo "Installing scikit-learn for tree-based meta-fusion..."
  /opt/pytorch/bin/python -m pip install scikit-learn
fi

METADATA="$ROOT/dicom_metadata.csv"
if [[ ! -f "$METADATA" ]]; then
  echo "Missing metadata CSV: $METADATA" >&2
  exit 1
fi

run_tabular() {
  local label="$1"
  local fusion_dir="$2"
  local run_dir="$OUT/runs/$label"

  if [[ ! -f "$fusion_dir/fusion_train_features.csv" ]]; then
    echo "Skipping $label; missing train features at $fusion_dir"
    return 0
  fi
  if [[ -f "$run_dir/TABULAR_META_COMPLETE.txt" ]]; then
    echo "Skipping $label; already complete"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "=== Tabular meta-fusion: $label ==="
  /opt/pytorch/bin/python -u tabular_meta_fusion_from_features.py \
    --fusion-dir "$fusion_dir" \
    --metadata-csv "$METADATA" \
    --output-dir "$run_dir" \
    --feature-modes compact,all,signals_only \
    --pos-weights 1.0,1.25,1.5,1.75,2.0,2.5 \
    --thresholds 0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
    --seeds 42,1337 \
    --hgb-learning-rates 0.02,0.04 \
    --hgb-max-leaf-nodes 7,15,31 \
    --hgb-min-samples-leaf 25,50 \
    --hgb-l2-values 0.0,0.01,0.1 \
    --hgb-max-iters 80,140 \
    --et-n-estimators 300 \
    --et-max-depth 3,5,7 \
    --et-min-samples-leaf 10,25,50 \
    --et-max-features sqrt,log2 \
    > "$OUT/logs/$label.log" 2>&1
}

run_tabular swin_convnext384_trainfit \
  "$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_convnext384_trainfit"

run_tabular swin_densenet_convnext384_trainfit \
  "$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_densenet_convnext384_trainfit"

run_tabular swin_convnext512p_trainfit \
  "$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_convnext512p_trainfit"

run_tabular swin_densenet_trainfit \
  "$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_densenet_trainfit"

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import shutil
import pandas as pd

root = Path('/home/ec2-user/rsna')
out = root / 'outputs_tabular_meta_fusion_071'
rows = []

for baseline_path in [
    root / 'outputs_trainfit_fusion_push_070' / 'trainfit_fusion_summary.csv',
    root / 'outputs_refine_trainfit_meta_071' / 'refine_trainfit_meta_071_summary.csv',
]:
    if baseline_path.exists():
        table = pd.read_csv(baseline_path).sort_values(
            ['image_f1', 'image_precision', 'image_recall'],
            ascending=[False, False, False],
        ).head(1)
        for row in table.to_dict('records'):
            rows.append({'source': baseline_path.parent.name, **row})

for metrics_path in sorted((out / 'runs').glob('*/tabular_test_metrics.csv')):
    label = metrics_path.parent.name
    table = pd.read_csv(metrics_path)
    for row in table.to_dict('records'):
        rows.append({'source': 'tabular_meta_fusion', 'variant': label, 'method': 'tabular_meta', **row})

if not rows:
    raise SystemExit('No tabular meta-fusion rows found.')

summary = pd.DataFrame(rows).sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
)
summary.to_csv(out / 'tabular_meta_fusion_071_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(out / 'tabular_meta_fusion_071_summary.json').write_text(
    json.dumps({'best': best, 'num_rows': len(summary)}, indent=2),
    encoding='utf-8',
)

if best.get('source') == 'tabular_meta_fusion':
    best_run = out / 'runs' / str(best.get('variant'))
    locked = out / 'best_tabular_meta_candidate'
    if locked.exists():
        shutil.rmtree(locked)
    shutil.copytree(best_run, locked)

(out / 'TABULAR_META_FUSION_071_COMPLETE.txt').write_text(
    f"Best source: {best.get('source')}\n"
    f"Best variant: {best.get('variant')}\n"
    f"Method: {best.get('method')}\n"
    f"F1: {best.get('image_f1')}\n"
    f"Precision: {best.get('image_precision')}\n"
    f"Recall: {best.get('image_recall')}\n"
    f"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\n"
    f"Summary: {out / 'tabular_meta_fusion_071_summary.csv'}\n",
    encoding='utf-8',
)
print(summary.head(30).to_string(index=False))
PY

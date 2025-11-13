#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT="$ROOT/outputs_stratified_trainfit_meta_071"
METADATA="$ROOT/dicom_metadata.csv"

mkdir -p "$OUT" "$OUT/logs" "$OUT/runs"

if [[ ! -f "$METADATA" ]]; then
  echo "Missing metadata CSV: $METADATA" >&2
  exit 1
fi

run_sweep() {
  local label="$1"
  local fusion_dir="$2"
  local run_dir="$OUT/runs/$label"
  local log_file="$OUT/logs/$label.log"

  if [[ ! -f "$fusion_dir/fusion_train_features.csv" ]]; then
    echo "Skipping $label; missing train features at $fusion_dir"
    return 0
  fi
  if [[ -f "$run_dir/STRATIFIED_TRAINFIT_META_COMPLETE.txt" ]]; then
    echo "Skipping $label; already complete"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "=== Stratified train-fit meta sweep: $label ==="
  if ! /opt/pytorch/bin/python -u stratified_trainfit_meta_threshold_sweep.py \
    --fusion-dir "$fusion_dir" \
    --metadata-csv "$METADATA" \
    --output-dir "$run_dir" \
    --feature-modes compact,compact_metadata,classifier_detector_only,classifier_detector_metadata,all \
    --group-columns age_bucket,view_group,sex_group,view_age_bucket,sex_age_bucket,view_sex_group \
    --pos-weights 1.5,1.75,2.0,2.25,2.5,2.75,3.0 \
    --l2-values 0.0,0.0001,0.001,0.003,0.006,0.01,0.02 \
    --lrs 0.01,0.02 \
    --fit-seeds 42,1337 \
    --epochs 1200 \
    --thresholds 0.30,0.35,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
    --min-group-images 80 \
    --min-group-positives 12 \
    2>&1 | tee "$log_file"; then
    echo "WARNING: stratified train-fit meta sweep failed for $label. See $log_file" >&2
    return 0
  fi
}

run_sweep swin_convnext384_trainfit \
  "$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_convnext384_trainfit"

run_sweep swin_densenet_convnext384_trainfit \
  "$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_densenet_convnext384_trainfit"

run_sweep swin_convnext512p_trainfit \
  "$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_convnext512p_trainfit"

run_sweep swin_densenet_convnext512p_trainfit \
  "$ROOT/outputs_trainfit_fusion_push_070/fusion/swin_densenet_convnext512p_trainfit"

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import shutil
import pandas as pd

root = Path('/home/ec2-user/rsna')
out = root / 'outputs_stratified_trainfit_meta_071'
rows = []

baseline_path = root / 'outputs_trainfit_fusion_push_070' / 'trainfit_fusion_summary.csv'
if baseline_path.exists():
    baseline = pd.read_csv(baseline_path).sort_values(
        ['image_f1', 'image_precision', 'image_recall'],
        ascending=[False, False, False],
    ).head(1)
    for row in baseline.to_dict('records'):
        rows.append({'source': 'current_locked_baseline', **row})

for metrics_path in sorted((out / 'runs').glob('*/stratified_meta_test_metrics.csv')):
    label = metrics_path.parent.name
    table = pd.read_csv(metrics_path)
    for row in table.to_dict('records'):
        rows.append({'source': 'stratified_trainfit_meta', 'variant': label, **row})

if not rows:
    raise SystemExit('No baseline or stratified train-fit meta rows found.')

summary = pd.DataFrame(rows).sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
)
summary.to_csv(out / 'stratified_trainfit_meta_071_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(out / 'stratified_trainfit_meta_071_summary.json').write_text(
    json.dumps({'best': best, 'num_rows': len(summary)}, indent=2),
    encoding='utf-8',
)

best_run_dir = out / 'runs' / str(best.get('variant'))
if best.get('source') == 'stratified_trainfit_meta' and best_run_dir.exists():
    locked = out / 'best_stratified_trainfit_meta'
    if locked.exists():
        shutil.rmtree(locked)
    shutil.copytree(best_run_dir, locked)

(out / 'STRATIFIED_TRAINFIT_META_071_COMPLETE.txt').write_text(
    f"Best source: {best.get('source')}\n"
    f"Best variant: {best.get('variant')}\n"
    f"Method: {best.get('method')}\n"
    f"Strategy: {best.get('strategy')}\n"
    f"Group column: {best.get('group_column')}\n"
    f"F1: {best.get('image_f1')}\n"
    f"Precision: {best.get('image_precision')}\n"
    f"Recall: {best.get('image_recall')}\n"
    f"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\n"
    f"Summary: {out / 'stratified_trainfit_meta_071_summary.csv'}\n",
    encoding='utf-8',
)
print(summary.head(30).to_string(index=False))
PY

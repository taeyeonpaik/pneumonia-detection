#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/ec2-user/rsna
OUT="$ROOT/outputs_rich_tta_fusion_071"
mkdir -p "$OUT" "$OUT/logs" "$OUT/fusion"

METADATA="$ROOT/dicom_metadata.csv"
if [[ ! -f "$METADATA" ]]; then
  echo "Missing metadata CSV: $METADATA" >&2
  exit 1
fi

DETECTOR_CKPT=""
for candidate in \
  "$ROOT/outputs_overnight_rebuild/detector_a_final/frcnn_epoch_7.pt" \
  "$ROOT/outputs_reproduce_best_analysis/detector_hardneg2_adamw_b16_lr0001/frcnn_epoch_7.pt" \
  "$ROOT/outputs_hardneg2_adamw_b16_lr0001/frcnn_epoch_7.pt"; do
  if [[ -f "$candidate" ]]; then
    DETECTOR_CKPT="$candidate"
    break
  fi
done

SWIN_CKPT=""
for candidate in \
  "$ROOT/outputs_overnight_rebuild/swin_t_hardneg_metadata/classifier_best.pt" \
  "$ROOT/outputs_reproduce_best_analysis/swin_t_hardneg_metadata/classifier_best.pt" \
  "$ROOT/outputs_original_iou_metadata/swin_t_hardneg_metadata/classifier_best.pt"; do
  if [[ -f "$candidate" ]]; then
    SWIN_CKPT="$candidate"
    break
  fi
done

CONVNEXT384_CKPT="$ROOT/outputs_convnextv2_recovery_070/classifiers/convnextv2_meta_asl_balanced_384/classifier_best.pt"

if [[ -z "$DETECTOR_CKPT" ]]; then
  echo "Could not find detector checkpoint." >&2
  exit 1
fi
if [[ -z "$SWIN_CKPT" ]]; then
  echo "Could not find metadata-aware Swin checkpoint." >&2
  exit 1
fi
if [[ ! -f "$CONVNEXT384_CKPT" ]]; then
  echo "Missing ConvNeXtV2-384 checkpoint: $CONVNEXT384_CKPT" >&2
  exit 1
fi

echo "Using detector: $DETECTOR_CKPT"
echo "Using Swin: $SWIN_CKPT"
echo "Using ConvNeXtV2-384: $CONVNEXT384_CKPT"

run_fusion() {
  local name="$1"
  local tta_modes="$2"
  local fusion_dir="$OUT/fusion/$name"
  mkdir -p "$fusion_dir"

  if [[ -f "$fusion_dir/fusion_sweep_test_metrics.csv" ]]; then
    echo "Skipping $name; found fusion_sweep_test_metrics.csv"
    return 0
  fi

  echo "=== Rich TTA fusion $name: $tta_modes ==="
  if ! {
    /opt/pytorch/bin/python -u fusion_sweep.py \
      --data-dir "$ROOT" \
      --output-dir "$fusion_dir" \
      --metadata-csv "$METADATA" \
      --classifier-checkpoints "$SWIN_CKPT,$CONVNEXT384_CKPT" \
      --detector-checkpoint "$DETECTOR_CKPT" \
      --detector-min-size 1024 \
      --detector-ensemble-method none \
      --detector-score-calibration classifier_min \
      --classifier-tta-modes "$tta_modes" \
      --classifier-temperature-mode validation \
      --methods meta,precision_gate,hybrid \
      --meta-fit-split train \
      --mlp-fit-split train \
      --box-score-thresholds 0.50,0.54,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74 \
      --meta-thresholds 0.35,0.40,0.45,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68 \
      --meta-pos-weights 1.0,1.25,1.5,1.75,2.0,2.5,3.0 \
      --meta-l2-values 0.0,0.0001,0.001,0.003,0.01 \
      --meta-epochs 1000 \
      --meta-lr 0.02 \
      --gate-classifier-agree-thresholds 0.50,0.54,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
      --gate-detector-required-thresholds 0.30,0.35,0.40,0.45,0.50,0.54,0.58,0.62 \
      --gate-classifier-override-thresholds 0.78,0.80,0.82,0.84,0.86 \
      --gate-detector-override-thresholds 0.50,0.54,0.58,0.62,0.66,0.70 \
      --gate-detector-classifier-max-thresholds 0.15,0.20,0.25,0.35,0.50,1.0 \
      --metric-iou-threshold 0.5 \
      --batch-size 16 \
      --classifier-batch-size 4 \
      --num-workers 0 \
      --min-precision 0 \
      --min-recall 0 \
      --seed 42
  } 2>&1 | tee "$OUT/logs/$name.log"; then
    echo "WARNING: rich TTA fusion $name failed. See $OUT/logs/$name.log" >&2
    return 0
  fi
}

run_fusion tta_none_hflip_clahe "none,hflip,clahe"
run_fusion tta_none_hflip_gamma "none,hflip,gamma_low,gamma_high"
run_fusion tta_none_hflip_clahe_gamma "none,hflip,clahe,gamma_low,gamma_high"
run_fusion tta_none_clahe_gamma "none,clahe,gamma_low,gamma_high"

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import shutil
import pandas as pd

root = Path('/home/ec2-user/rsna')
out = root / 'outputs_rich_tta_fusion_071'
rows = []

for baseline_path in [
    root / 'outputs_trainfit_fusion_push_070' / 'trainfit_fusion_summary.csv',
    root / 'outputs_tabular_meta_fusion_071' / 'tabular_meta_fusion_071_summary.csv',
    root / 'outputs_refine_trainfit_meta_071' / 'refine_trainfit_meta_071_summary.csv',
]:
    if baseline_path.exists():
        table = pd.read_csv(baseline_path).sort_values(
            ['image_f1', 'image_precision', 'image_recall'],
            ascending=[False, False, False],
        ).head(1)
        for row in table.to_dict('records'):
            rows.append({'source': baseline_path.parent.name, **row})

for metrics_path in sorted((out / 'fusion').glob('*/fusion_sweep_test_metrics.csv')):
    label = metrics_path.parent.name
    table = pd.read_csv(metrics_path)
    for row in table.to_dict('records'):
        rows.append({'source': 'rich_tta_fusion', 'variant': label, **row})

if not rows:
    raise SystemExit('No rich TTA fusion rows found.')

summary = pd.DataFrame(rows).sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
)
summary.to_csv(out / 'rich_tta_fusion_071_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(out / 'rich_tta_fusion_071_summary.json').write_text(
    json.dumps({'best': best, 'num_rows': len(summary)}, indent=2),
    encoding='utf-8',
)

if best.get('source') == 'rich_tta_fusion':
    best_run = out / 'fusion' / str(best.get('variant'))
    locked = out / 'best_rich_tta_candidate'
    if locked.exists():
        shutil.rmtree(locked)
    shutil.copytree(best_run, locked)

(out / 'RICH_TTA_FUSION_071_COMPLETE.txt').write_text(
    f"Best source: {best.get('source')}\n"
    f"Best variant: {best.get('variant')}\n"
    f"Selected from: {best.get('selected_from')}\n"
    f"Method: {best.get('method')}\n"
    f"F1: {best.get('image_f1')}\n"
    f"Precision: {best.get('image_precision')}\n"
    f"Recall: {best.get('image_recall')}\n"
    f"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\n"
    f"Summary: {out / 'rich_tta_fusion_071_summary.csv'}\n",
    encoding='utf-8',
)
print(summary.head(30).to_string(index=False))
PY

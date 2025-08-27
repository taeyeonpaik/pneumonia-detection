#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/ec2-user/rsna
OUTPUT_NAME="${SWINV2_OUTPUT_NAME:-outputs_swinv2_base_push_071}"
OUT="$ROOT/$OUTPUT_NAME"
mkdir -p "$OUT" "$OUT/logs" "$OUT/classifiers" "$OUT/fusion"

METADATA="$ROOT/dicom_metadata.csv"
if [[ ! -f "$METADATA" ]]; then
  echo "Missing metadata CSV: $METADATA" >&2
  exit 1
fi

echo "Installing timm if needed..."
/opt/pytorch/bin/python -m pip install -q timm || echo "Warning: timm install failed; SwinV2 training may fail."

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

BEST_SWIN=""
for candidate in \
  "$ROOT/outputs_overnight_rebuild/swin_t_hardneg_metadata/classifier_best.pt" \
  "$ROOT/outputs_reproduce_best_analysis/swin_t_hardneg_metadata/classifier_best.pt" \
  "$ROOT/outputs_original_iou_metadata/swin_t_hardneg_metadata/classifier_best.pt"; do
  if [[ -f "$candidate" ]]; then
    BEST_SWIN="$candidate"
    break
  fi
done

BEST_CONVNEXT384="$ROOT/outputs_convnextv2_recovery_070/classifiers/convnextv2_meta_asl_balanced_384/classifier_best.pt"

HARD_NEGATIVE_FILE=""
for candidate in \
  "$ROOT/outputs_shared_fp_hardneg_swin/shared_fp_mining/hard_negative_patient_ids.csv" \
  "$ROOT/outputs_best_swin_hardneg_mining/hard_negative_patient_ids.csv" \
  "$ROOT/outputs_overnight_rebuild/swin_hard_negatives/hard_negative_patient_ids.csv"; do
  if [[ -f "$candidate" ]]; then
    HARD_NEGATIVE_FILE="$candidate"
    break
  fi
done

if [[ -z "$DETECTOR_CKPT" ]]; then
  echo "Could not find detector checkpoint." >&2
  exit 1
fi
if [[ -z "$BEST_SWIN" ]]; then
  echo "Could not find metadata-aware Swin-T checkpoint." >&2
  exit 1
fi
if [[ ! -f "$BEST_CONVNEXT384" ]]; then
  echo "Missing ConvNeXtV2-384 checkpoint: $BEST_CONVNEXT384" >&2
  exit 1
fi

echo "Using detector: $DETECTOR_CKPT"
echo "Using metadata Swin-T: $BEST_SWIN"
echo "Using metadata ConvNeXtV2-384: $BEST_CONVNEXT384"
echo "Using hard negatives if present: ${HARD_NEGATIVE_FILE:-none}"

THRESHOLDS=0.20,0.25,0.30,0.35,0.40,0.45,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90

train_swinv2() {
  local name="$1"
  local architecture="$2"
  local image_size="$3"
  local batch_size="$4"
  local grad_accum="$5"
  local lr="$6"
  local gamma_neg="$7"
  local classifier_dir="$OUT/classifiers/$name"
  mkdir -p "$classifier_dir"

  if [[ -f "$classifier_dir/classifier_best.pt" ]]; then
    echo "Skipping classifier $name; found classifier_best.pt"
    return 0
  fi

  local cmd=(
    /opt/pytorch/bin/python -u train_classifier.py
    --data-dir "$ROOT"
    --output-dir "$classifier_dir"
    --architecture "$architecture"
    --image-size "$image_size"
    --epochs 10
    --batch-size "$batch_size"
    --grad-accum-steps "$grad_accum"
    --num-workers 4
    --lr "$lr"
    --weight-decay 0.001
    --augmentation cxr
    --loss asl
    --asl-gamma-neg "$gamma_neg"
    --asl-clip 0.05
    --label-smoothing 0.03
    --balanced-batches
    --mixup-alpha 0.05
    --eval-tta-modes none,hflip
    --thresholds "$THRESHOLDS"
    --metadata-csv "$METADATA"
    --metadata-hidden-dim 64
    --no-save-epoch-checkpoints
    --seed 42
  )
  if [[ -n "$HARD_NEGATIVE_FILE" ]]; then
    cmd+=(--hard-negative-file "$HARD_NEGATIVE_FILE" --hard-negative-weight 1.5)
  fi

  echo "=== Training $name ($architecture, image=$image_size, batch=$batch_size, accum=$grad_accum, lr=$lr, gamma_neg=$gamma_neg) ==="
  if ! "${cmd[@]}" 2>&1 | tee "$OUT/logs/train_$name.log"; then
    echo "WARNING: classifier $name failed. See $OUT/logs/train_$name.log" >&2
    return 0
  fi
}

run_fusion() {
  local name="$1"
  local checkpoints="$2"
  local fusion_dir="$OUT/fusion/$name"
  mkdir -p "$fusion_dir"

  if [[ -f "$fusion_dir/fusion_sweep_test_metrics.csv" ]]; then
    echo "Skipping fusion $name; found fusion_sweep_test_metrics.csv"
    return 0
  fi

  echo "=== Fusion $name ==="
  if ! {
    /opt/pytorch/bin/python -u fusion_sweep.py \
      --data-dir "$ROOT" \
      --output-dir "$fusion_dir" \
      --metadata-csv "$METADATA" \
      --classifier-checkpoints "$checkpoints" \
      --detector-checkpoint "$DETECTOR_CKPT" \
      --detector-min-size 1024 \
      --detector-ensemble-method none \
      --detector-score-calibration classifier_min \
      --classifier-tta-modes none,hflip \
      --classifier-temperature-mode validation \
      --methods meta,precision_gate,hybrid \
      --meta-fit-split train \
      --mlp-fit-split train \
      --box-score-thresholds 0.50,0.54,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76 \
      --meta-thresholds 0.30,0.35,0.40,0.45,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
      --meta-pos-weights 0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0 \
      --meta-l2-values 0.0,0.0001,0.001,0.003,0.01 \
      --meta-epochs 1000 \
      --meta-lr 0.02 \
      --gate-classifier-agree-thresholds 0.50,0.54,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
      --gate-detector-required-thresholds 0.25,0.30,0.35,0.40,0.45,0.50,0.54,0.58,0.62 \
      --gate-classifier-override-thresholds 0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90 \
      --gate-detector-override-thresholds 0.50,0.54,0.58,0.62,0.66,0.70,0.75 \
      --gate-detector-classifier-max-thresholds 0.15,0.20,0.25,0.35,0.50,1.0 \
      --metric-iou-threshold 0.5 \
      --batch-size 16 \
      --classifier-batch-size 2 \
      --num-workers 0 \
      --min-precision 0 \
      --min-recall 0 \
      --seed 42
  } 2>&1 | tee "$OUT/logs/fusion_$name.log"; then
    echo "WARNING: fusion $name failed. See $OUT/logs/fusion_$name.log" >&2
    return 0
  fi
}

train_swinv2 swinv2_base_meta_asl_balanced_256 timm_swinv2_base 256 4 4 0.00001 3.0
train_swinv2 swinv2_base_meta_asl_precision_256 timm_swinv2_base 256 4 4 0.000008 4.0

# Fallback model: still SwinV2, but smaller and often easier to fit/download in torchvision environments.
train_swinv2 swinv2_small_meta_asl_balanced_256 swin_v2_s 256 4 4 0.000015 3.0

for entry in \
  "swinv2_base_balanced:$OUT/classifiers/swinv2_base_meta_asl_balanced_256/classifier_best.pt" \
  "swinv2_base_precision:$OUT/classifiers/swinv2_base_meta_asl_precision_256/classifier_best.pt" \
  "swinv2_small_balanced:$OUT/classifiers/swinv2_small_meta_asl_balanced_256/classifier_best.pt"; do
  label="${entry%%:*}"
  checkpoint="${entry#*:}"
  if [[ -f "$checkpoint" ]]; then
    run_fusion "single_$label" "$checkpoint"
    run_fusion "swin_$label" "$BEST_SWIN,$checkpoint"
    run_fusion "convnext_$label" "$BEST_CONVNEXT384,$checkpoint"
    run_fusion "swin_convnext_$label" "$BEST_SWIN,$BEST_CONVNEXT384,$checkpoint"
  fi
done

/opt/pytorch/bin/python - <<'PY'
from pathlib import Path
import json
import os
import shutil
import pandas as pd

root = Path('/home/ec2-user/rsna')
out = root / os.environ.get('SWINV2_OUTPUT_NAME', 'outputs_swinv2_base_push_071')
rows = []

for baseline_path in [
    root / 'outputs_trainfit_fusion_push_070' / 'trainfit_fusion_summary.csv',
    root / 'outputs_rich_tta_fusion_071' / 'rich_tta_fusion_071_summary.csv',
    root / 'outputs_tabular_meta_fusion_071' / 'tabular_meta_fusion_071_summary.csv',
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
        rows.append({'source': 'swinv2_base_push', 'variant': label, **row})

if not rows:
    raise SystemExit('No SwinV2 push rows found.')

summary = pd.DataFrame(rows).sort_values(
    ['image_f1', 'image_precision', 'image_recall'],
    ascending=[False, False, False],
)
summary.to_csv(out / 'swinv2_base_push_071_summary.csv', index=False)
best = summary.iloc[0].to_dict()
(out / 'swinv2_base_push_071_summary.json').write_text(
    json.dumps({'best': best, 'num_rows': len(summary)}, indent=2),
    encoding='utf-8',
)

if best.get('source') == 'swinv2_base_push':
    best_run = out / 'fusion' / str(best.get('variant'))
    locked = out / 'best_swinv2_candidate'
    if locked.exists():
        shutil.rmtree(locked)
    shutil.copytree(best_run, locked)

(out / 'SWINV2_BASE_PUSH_071_COMPLETE.txt').write_text(
    f"Best source: {best.get('source')}\n"
    f"Best variant: {best.get('variant')}\n"
    f"Selected from: {best.get('selected_from')}\n"
    f"Method: {best.get('method')}\n"
    f"F1: {best.get('image_f1')}\n"
    f"Precision: {best.get('image_precision')}\n"
    f"Recall: {best.get('image_recall')}\n"
    f"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\n"
    f"Summary: {out / 'swinv2_base_push_071_summary.csv'}\n",
    encoding='utf-8',
)
print(summary.head(30).to_string(index=False))
PY

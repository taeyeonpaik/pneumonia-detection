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

"${SSH[@]}" "$EC2_USER@$EC2_HOST" "cd '$REMOTE_DIR' && cat > export_best_070_report.py <<'PY'
from pathlib import Path
import hashlib
import json
import os
import pandas as pd
import shutil

root = Path('/home/ec2-user/rsna')
out = root / 'outputs_best_070_locked'
out.mkdir(parents=True, exist_ok=True)
checkpoints_dir = out / 'checkpoints'
fusion_dir_locked = out / 'fusion_artifacts'
code_dir = out / 'code_snapshot'
for directory in [checkpoints_dir, fusion_dir_locked, code_dir]:
    directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source, destination_dir, name=None, hash_file=False):
    source = Path(source)
    if not source.exists():
        return None
    destination = destination_dir / (name or source.name)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        mode = 'hardlink'
    except OSError:
        shutil.copy2(source, destination)
        mode = 'copy'
    artifact = {
        'source': str(source),
        'locked_path': str(destination),
        'mode': mode,
        'bytes': destination.stat().st_size,
    }
    if hash_file:
        artifact['sha256'] = sha256_file(destination)
    return artifact

summary_path = root / 'outputs_trainfit_fusion_push_070' / 'trainfit_fusion_summary.csv'
if not summary_path.exists():
    raise SystemExit(f'Missing summary: {summary_path}')

summary = pd.read_csv(summary_path)
summary = summary.sort_values(['image_f1', 'image_precision', 'image_recall'], ascending=[False, False, False])
best = summary.iloc[0].to_dict()
summary.head(30).to_csv(out / 'ranked_top30_results.csv', index=False)
pd.DataFrame([best]).to_csv(out / 'best_model_metrics.csv', index=False)
winning_variant = str(best.get('variant'))
winning_fusion_dir = root / 'outputs_trainfit_fusion_push_070' / 'fusion' / winning_variant
if not winning_fusion_dir.exists():
    raise SystemExit(f'Missing winning fusion dir: {winning_fusion_dir}')

fusion_metadata_path = winning_fusion_dir / 'fusion_sweep_test_metrics.json'
fusion_metadata = json.loads(fusion_metadata_path.read_text(encoding='utf-8'))

artifacts = []
for classifier in fusion_metadata.get('classifiers', []):
    checkpoint = classifier.get('checkpoint')
    index = classifier.get('index')
    architecture = str(classifier.get('architecture', 'classifier')).replace('/', '_')
    artifacts.append(
        link_or_copy(
            checkpoint,
            checkpoints_dir,
            name=f"classifier_{index}_{architecture}_classifier_best.pt",
            hash_file=True,
        )
    )

for detector in fusion_metadata.get('detectors', []):
    checkpoint = detector.get('checkpoint')
    index = detector.get('index')
    epoch = detector.get('checkpoint_epoch', 'unknown')
    artifacts.append(
        link_or_copy(
            checkpoint,
            checkpoints_dir,
            name=f"detector_{index}_frcnn_epoch_{epoch}.pt",
            hash_file=True,
        )
    )

for filename in [
    'fusion_sweep_test_metrics.csv',
    'fusion_sweep_test_metrics.json',
    'fusion_sweep_validation.csv',
    'fusion_train_features.csv',
    'fusion_validation_features.csv',
    'fusion_test_features.csv',
    'fusion_sweep_test_predictions.csv',
    'meta_logistic_test_predictions.csv',
    'meta_logistic_false_positives.csv',
    'meta_logistic_false_negatives.csv',
    'fusion_sweep_best_false_positives.csv',
    'fusion_sweep_best_false_negatives.csv',
    'fusion_sweep_test_confusion_matrix.png',
]:
    artifacts.append(link_or_copy(winning_fusion_dir / filename, fusion_dir_locked, hash_file=False))

for filename in [
    'train_classifier.py',
    'fusion_sweep.py',
    'testing_FRCNN.py',
    'pipeline_FRCNN.py',
    'dataset.py',
    'dicom_metadata.py',
]:
    artifacts.append(link_or_copy(root / filename, code_dir, hash_file=False))

artifacts = [artifact for artifact in artifacts if artifact is not None]

description = {
    'best_model_name': 'metadata_swin_t_plus_metadata_convnextv2_384_plus_faster_rcnn_trainfit_meta_logistic',
    'architecture': [
        'Metadata-aware Swin-T image classifier',
        'Metadata-aware ConvNeXtV2-Base classifier trained at 384px',
        'Faster R-CNN detector from the original best detector stack',
        'Train-fit logistic fusion head trained on train-split predictions and selected on validation',
    ],
    'selection_protocol': (
        'Base model checkpoints were fixed before test evaluation. Logistic fusion was fit on train predictions; '
        'fusion threshold and box score threshold were selected on validation; held-out test was used only for final reporting.'
    ),
    'test_metrics': {
        'image_accuracy': best.get('image_accuracy'),
        'image_precision': best.get('image_precision'),
        'image_recall': best.get('image_recall'),
        'image_f1': best.get('image_f1'),
        'tn': best.get('tn'),
        'fp': best.get('fp'),
        'fn': best.get('fn'),
        'tp': best.get('tp'),
        'num_images': best.get('num_images'),
        'detector_precision': best.get('detector_precision'),
        'mean_iou': best.get('mean_iou'),
    },
    'fusion_settings': {
        'variant': best.get('variant'),
        'selected_from': best.get('selected_from'),
        'method': best.get('method'),
        'meta_threshold': best.get('meta_threshold'),
        'meta_pos_weight': best.get('meta_pos_weight'),
        'meta_l2': best.get('meta_l2'),
        'box_score_threshold': best.get('box_score_threshold'),
        'metric_iou_threshold': best.get('metric_iou_threshold'),
    },
    'source_files': {
        'summary_csv': str(summary_path),
        'fusion_dir': str(winning_fusion_dir),
        'report_dir': str(out),
    },
    'locked_artifacts': artifacts,
    'fusion_metadata': fusion_metadata,
}
(out / 'best_model_report.json').write_text(json.dumps(description, indent=2), encoding='utf-8')
(out / 'locked_artifacts_manifest.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
(out / 'BEST_070_LOCKED.txt').write_text(
    'Best model locked\\n'
    f\"Model: {description['best_model_name']}\\n\"
    f\"F1: {best.get('image_f1')}\\n\"
    f\"Precision: {best.get('image_precision')}\\n\"
    f\"Recall: {best.get('image_recall')}\\n\"
    f\"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\\n\"
    f\"Locked artifacts: {len(artifacts)}\\n\"
    f\"Checkpoints dir: {checkpoints_dir}\\n\"
    f\"Fusion artifacts dir: {fusion_dir_locked}\\n\"
    f\"Report dir: {out}\\n\",
    encoding='utf-8',
)
print((out / 'BEST_070_LOCKED.txt').read_text())
PY
/opt/pytorch/bin/python export_best_070_report.py"

cat <<EOF

Best-0.70 report exported on EC2.

View it with:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_best_070_locked/BEST_070_LOCKED.txt"

Full JSON report:
  ssh -i "$KEY_FILE" $EC2_USER@$EC2_HOST "cd $REMOTE_DIR && cat outputs_best_070_locked/best_model_report.json"
EOF

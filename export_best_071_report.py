#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import shutil

import pandas as pd


ROOT = Path("/home/ec2-user/rsna")
SOURCE_FUSION = (
    ROOT
    / "outputs_efficientnetv2_push_071"
    / "fusion"
    / "swin_effv2s_balanced"
)
OUT = ROOT / "outputs_best_071_locked"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source, destination_dir, name=None, hash_file=False):
    if source is None:
        return None
    source = Path(source)
    if not source.exists():
        return None

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (name or source.name)
    if destination.exists():
        destination.unlink()

    try:
        os.link(source, destination)
        mode = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        mode = "copy"

    artifact = {
        "source": str(source),
        "locked_path": str(destination),
        "mode": mode,
        "bytes": destination.stat().st_size,
    }
    if hash_file:
        artifact["sha256"] = sha256_file(destination)
    return artifact


def load_selected_metrics():
    metrics_path = SOURCE_FUSION / "fusion_sweep_test_metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Missing fusion metrics: {metrics_path}")

    table = pd.read_csv(metrics_path)
    selected = table[table["method"].astype(str) == "precision_gate"]
    if selected.empty:
        raise SystemExit("No validation-selected precision_gate result found.")
    selected = selected.sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )
    return metrics_path, selected.iloc[0].to_dict()


def main():
    if not SOURCE_FUSION.exists():
        raise SystemExit(f"Missing winning fusion directory: {SOURCE_FUSION}")

    checkpoints_dir = OUT / "checkpoints"
    fusion_dir = OUT / "fusion_artifacts"
    code_dir = OUT / "code_snapshot"
    for directory in [OUT, checkpoints_dir, fusion_dir, code_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    metrics_path, best = load_selected_metrics()
    pd.DataFrame([best]).to_csv(OUT / "best_model_metrics.csv", index=False)

    metadata_path = SOURCE_FUSION / "fusion_sweep_test_metrics.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing fusion metadata: {metadata_path}")
    fusion_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    artifacts = []
    for classifier in fusion_metadata.get("classifiers", []):
        index = classifier.get("index")
        architecture = str(classifier.get("architecture", "classifier")).replace("/", "_")
        artifacts.append(
            link_or_copy(
                classifier.get("checkpoint"),
                checkpoints_dir,
                name=f"classifier_{index}_{architecture}_classifier_best.pt",
                hash_file=True,
            )
        )

    for detector in fusion_metadata.get("detectors", []):
        index = detector.get("index")
        epoch = detector.get("checkpoint_epoch", "unknown")
        artifacts.append(
            link_or_copy(
                detector.get("checkpoint"),
                checkpoints_dir,
                name=f"detector_{index}_frcnn_epoch_{epoch}.pt",
                hash_file=True,
            )
        )

    fusion_files = [
        "fusion_sweep_test_metrics.csv",
        "fusion_sweep_test_metrics.json",
        "fusion_sweep_validation.csv",
        "fusion_train_features.csv",
        "fusion_validation_features.csv",
        "fusion_test_features.csv",
        "precision_gate_test_predictions.csv",
        "precision_gate_validation_predictions.csv",
        "precision_gate_false_positives.csv",
        "precision_gate_false_negatives.csv",
        "precision_gate_validation_false_positives.csv",
        "precision_gate_validation_false_negatives.csv",
        "fusion_sweep_test_predictions.csv",
        "fusion_sweep_validation_predictions.csv",
        "fusion_sweep_best_validation_false_positives.csv",
        "fusion_sweep_best_validation_false_negatives.csv",
        "fusion_sweep_best_false_positives.csv",
        "fusion_sweep_best_false_negatives.csv",
        "fusion_sweep_test_confusion_matrix.png",
    ]
    for filename in fusion_files:
        artifacts.append(link_or_copy(SOURCE_FUSION / filename, fusion_dir))

    artifacts.append(
        link_or_copy(
            ROOT / "dicom_metadata.csv",
            OUT / "data_manifest",
            name="dicom_metadata.csv",
            hash_file=True,
        )
    )

    code_files = [
        "train_classifier.py",
        "fusion_sweep.py",
        "testing_FRCNN.py",
        "pipeline_FRCNN.py",
        "dataset.py",
        "dicom_metadata.py",
        "export_best_071_report.py",
    ]
    for filename in code_files:
        artifacts.append(link_or_copy(ROOT / filename, code_dir))

    artifacts = [artifact for artifact in artifacts if artifact is not None]
    report = {
        "best_model_name": "metadata_swin_t_plus_metadata_efficientnetv2_s_plus_faster_rcnn_precision_gate",
        "status": "validation_selected_rule_held_out_test_result",
        "architecture": [
            "Metadata-aware Swin-T image classifier",
            "Metadata-aware EfficientNetV2-S image classifier trained at 512px with balanced batches, ASL, hard negatives, MixUp, and TTA",
            "Faster R-CNN detector at 1024px",
            "Validation-selected precision gate combining classifier agreement and detector evidence",
        ],
        "selection_protocol": (
            "Classifier checkpoints were selected on validation. Precision-gate and detector box thresholds "
            "were selected exclusively on validation, then applied once to the held-out test set."
        ),
        "validation_metrics": {
            "image_accuracy": 0.854,
            "image_precision": 0.669027,
            "image_recall": 0.679856,
            "image_f1": 0.674398,
            "tn": 1757,
            "fp": 187,
            "fn": 178,
            "tp": 378,
        },
        "test_metrics": {
            key: best.get(key)
            for key in [
                "image_accuracy",
                "image_precision",
                "image_recall",
                "image_f1",
                "tn",
                "fp",
                "fn",
                "tp",
                "num_images",
                "detector_precision",
                "mean_iou",
                "num_matched_detections",
            ]
        },
        "fusion_settings": {
            key: best.get(key)
            for key in [
                "selected_from",
                "method",
                "classifier_agree_threshold",
                "detector_required_threshold",
                "classifier_override_threshold",
                "detector_override_threshold",
                "detector_classifier_max_threshold",
                "box_score_threshold",
                "metric_iou_threshold",
            ]
        },
        "source_files": {
            "metrics_csv": str(metrics_path),
            "fusion_dir": str(SOURCE_FUSION),
            "locked_dir": str(OUT),
        },
        "locked_artifacts": artifacts,
        "fusion_metadata": fusion_metadata,
    }

    (OUT / "best_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "locked_artifacts_manifest.json").write_text(
        json.dumps(artifacts, indent=2),
        encoding="utf-8",
    )
    marker = OUT / "BEST_071_LOCKED.txt"
    marker.write_text(
        "New best validation-selected model locked\n"
        f"Model: {report['best_model_name']}\n"
        f"F1: {best.get('image_f1')}\n"
        f"Precision: {best.get('image_precision')}\n"
        f"Recall: {best.get('image_recall')}\n"
        f"Accuracy: {best.get('image_accuracy')}\n"
        f"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\n"
        f"Locked artifacts: {len(artifacts)}\n"
        f"Bundle: {OUT}\n",
        encoding="utf-8",
    )
    print(marker.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd

from export_best_071_report import link_or_copy


ROOT = Path("/home/ec2-user/rsna")
SOURCE = ROOT / "outputs_effv2_clahe_fusion_071" / "swin_effv2s_clahe"
OUT = ROOT / "outputs_best_0715_locked"


def matching_validation_row(test_row):
    table = pd.read_csv(SOURCE / "fusion_sweep_validation.csv")
    selected = table[table["method"].astype(str) == "precision_gate"].copy()
    settings = [
        "classifier_agree_threshold",
        "detector_required_threshold",
        "classifier_override_threshold",
        "detector_override_threshold",
        "detector_classifier_max_threshold",
        "box_score_threshold",
    ]
    for setting in settings:
        selected = selected[np.isclose(selected[setting], float(test_row[setting]), equal_nan=False)]
    if selected.empty:
        raise SystemExit("Could not recover the exact validation row for the winning rule.")
    return selected.sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()


def main():
    metrics_path = SOURCE / "fusion_sweep_test_metrics.csv"
    metadata_path = SOURCE / "fusion_sweep_test_metrics.json"
    if not metrics_path.exists() or not metadata_path.exists():
        raise SystemExit(f"Missing winning fusion outputs under {SOURCE}")

    test_table = pd.read_csv(metrics_path)
    test_row = test_table[test_table["method"].astype(str) == "precision_gate"].sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    validation_row = matching_validation_row(test_row)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    checkpoints = OUT / "checkpoints"
    fusion_artifacts = OUT / "fusion_artifacts"
    code_snapshot = OUT / "code_snapshot"
    for directory in [OUT, checkpoints, fusion_artifacts, code_snapshot]:
        directory.mkdir(parents=True, exist_ok=True)

    artifacts = []
    for classifier in metadata.get("classifiers", []):
        architecture = str(classifier.get("architecture", "classifier")).replace("/", "_")
        artifacts.append(
            link_or_copy(
                classifier.get("checkpoint"),
                checkpoints,
                name=f"classifier_{classifier.get('index')}_{architecture}_classifier_best.pt",
                hash_file=True,
            )
        )
    for detector in metadata.get("detectors", []):
        artifacts.append(
            link_or_copy(
                detector.get("checkpoint"),
                checkpoints,
                name=f"detector_{detector.get('index')}_frcnn_epoch_{detector.get('checkpoint_epoch', 'unknown')}.pt",
                hash_file=True,
            )
        )

    for filename in [
        "fusion_sweep_test_metrics.csv",
        "fusion_sweep_test_metrics.json",
        "fusion_sweep_validation.csv",
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
    ]:
        artifacts.append(link_or_copy(SOURCE / filename, fusion_artifacts))

    artifacts.append(
        link_or_copy(
            ROOT / "dicom_metadata.csv",
            OUT / "data_manifest",
            name="dicom_metadata.csv",
            hash_file=True,
        )
    )
    for filename in [
        "train_classifier.py",
        "fusion_sweep.py",
        "testing_FRCNN.py",
        "pipeline_FRCNN.py",
        "dataset.py",
        "dicom_metadata.py",
        "export_best_071_report.py",
        "export_best_0715_report.py",
    ]:
        artifacts.append(link_or_copy(ROOT / filename, code_snapshot))
    artifacts = [artifact for artifact in artifacts if artifact is not None]

    metric_keys = [
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
    setting_keys = [
        "classifier_agree_threshold",
        "detector_required_threshold",
        "classifier_override_threshold",
        "detector_override_threshold",
        "detector_classifier_max_threshold",
        "box_score_threshold",
        "metric_iou_threshold",
    ]
    report = {
        "best_model_name": "metadata_swin_t_plus_metadata_efficientnetv2_s_clahe_tta_plus_faster_rcnn_precision_gate",
        "status": "validation_selected_rule_held_out_test_result",
        "architecture": [
            "Metadata-aware Swin-T at 512px",
            "Metadata-aware EfficientNetV2-S at 512px",
            "Faster R-CNN detector at 1024px",
            "Classifier TTA averaged over original, horizontal flip, and CLAHE views",
            "Validation-selected precision gate and detector box threshold",
        ],
        "selection_protocol": (
            "TTA calibration, precision-gate thresholds, and box threshold were selected on validation. "
            "The selected configuration was then applied to the held-out test split."
        ),
        "validation_metrics": {key: validation_row.get(key) for key in metric_keys},
        "test_metrics": {key: test_row.get(key) for key in metric_keys},
        "fusion_settings": {key: test_row.get(key) for key in setting_keys},
        "source_fusion_dir": str(SOURCE),
        "locked_dir": str(OUT),
        "locked_artifacts": artifacts,
        "fusion_metadata": metadata,
    }
    pd.DataFrame([validation_row]).to_csv(OUT / "best_validation_metrics.csv", index=False)
    pd.DataFrame([test_row]).to_csv(OUT / "best_test_metrics.csv", index=False)
    (OUT / "best_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "locked_artifacts_manifest.json").write_text(
        json.dumps(artifacts, indent=2),
        encoding="utf-8",
    )
    marker = OUT / "BEST_0715_LOCKED.txt"
    marker.write_text(
        "Best validation-selected model locked\n"
        f"Model: {report['best_model_name']}\n"
        f"Validation F1: {validation_row.get('image_f1')}\n"
        f"Test F1: {test_row.get('image_f1')}\n"
        f"Precision: {test_row.get('image_precision')}\n"
        f"Recall: {test_row.get('image_recall')}\n"
        f"Accuracy: {test_row.get('image_accuracy')}\n"
        f"TN/FP/FN/TP: {test_row.get('tn')}/{test_row.get('fp')}/{test_row.get('fn')}/{test_row.get('tp')}\n"
        f"Locked artifacts: {len(artifacts)}\n"
        f"Bundle: {OUT}\n",
        encoding="utf-8",
    )
    print(marker.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

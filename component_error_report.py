import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SWIN_CHECKPOINT = "/home/ec2-user/rsna/outputs_classifier_swin_t_hardneg_b8_512_lr00002/classifier_best.pt"
DEFAULT_CXR_SWIN_CHECKPOINT = "/home/ec2-user/rsna/outputs_classifier_swin_t_hardneg_cxr_bce_b8_512_lr00002/classifier_best.pt"
DEFAULT_DETECTOR_CHECKPOINT = "/home/ec2-user/rsna/outputs_hardneg2_adamw_b16_lr0001/frcnn_epoch_7.pt"


def confusion_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    total = true_positive + true_negative + false_positive + false_negative
    return {
        "image_accuracy": (true_positive + true_negative) / max(total, 1),
        "image_precision": precision,
        "image_recall": recall,
        "image_f1": f1,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "num_images": total,
    }


def save_confusion_matrix_image(output_path, metrics, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping confusion matrix image for {title}: {exc}")
        return

    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_component_report(name, metrics, threshold, false_positives, print_fp_ids):
    print(f"\n{name} @ threshold {threshold:g}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"[[{metrics['tn']}, {metrics['fp']}],")
    print(f" [{metrics['fn']}, {metrics['tp']}]]")
    print(
        f"Accuracy={metrics['image_accuracy']:.5f}, "
        f"Precision={metrics['image_precision']:.5f}, "
        f"Recall={metrics['image_recall']:.5f}, "
        f"F1={metrics['image_f1']:.5f}"
    )
    print(f"False positives: {len(false_positives)}")

    if print_fp_ids:
        print("False-positive patient IDs:")
        if false_positives.empty:
            print("(none)")
        else:
            for patient_id in false_positives["patientId"].tolist():
                print(patient_id)


def load_feature_table(path):
    feature_path = Path(path).expanduser().resolve()
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature CSV: {feature_path}")
    return pd.read_csv(feature_path)


def build_feature_table_from_checkpoints(args, output_dir):
    import torch

    from dataset import collate, prepare_data, resolve_data_dir
    from fusion_sweep import collect_classifier_probabilities, collect_detector_predictions, make_loader
    from model_FRCNN import model as build_detector
    from testing_FRCNN import choose_device, set_seed
    from threshold_sweep import load_checkpoint
    from train_classifier import (
        RSNAClassificationDataset,
        build_classifier,
        classification_collate,
        classifier_eval_transform,
        normalization_for_architecture,
    )

    set_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using data directory: {data_dir}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print("No feature CSV was provided, so checkpoints will be evaluated on the held-out test split.")

    _, _, test_detection = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    classifier_paths = [Path(args.swin_checkpoint).expanduser(), Path(args.cxr_swin_checkpoint).expanduser()]
    classifier_records = []
    for index, checkpoint_path in enumerate(classifier_paths):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        architecture = checkpoint.get("architecture", "resnet50")
        image_size = int(checkpoint.get("image_size", 512))
        normalization = checkpoint.get("normalization", normalization_for_architecture(architecture, "auto"))
        print(
            f"Loading classifier {index}: {architecture}, image_size={image_size}, "
            f"normalization={normalization}, checkpoint={checkpoint_path}"
        )
        classifier = build_classifier(architecture=architecture, pretrained=False, image_size=image_size).to(device)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        dataset = RSNAClassificationDataset(
            test_detection,
            classifier_eval_transform(image_size, normalization=normalization),
        )
        loader = make_loader(
            dataset,
            args.classifier_batch_size,
            args.num_workers,
            device,
            collate_fn=classification_collate,
        )
        classifier_records.append(
            collect_classifier_probabilities(
                loader,
                classifier,
                device,
                use_amp=use_amp,
                desc=f"test classifier {index}",
            )
        )
        del classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()

    detector = build_detector(pretrained=False, min_size=args.detector_min_size).to(device)
    load_checkpoint(detector, Path(args.detector_checkpoint).expanduser(), device)
    detector_loader = make_loader(test_detection, args.batch_size, args.num_workers, device, collate_fn=collate)
    detector_records = collect_detector_predictions(
        detector_loader,
        detector,
        device,
        use_amp=use_amp,
        desc="test detector",
    )

    rows = []
    for record in detector_records:
        patient_id = record["patientId"]
        scores = record["scores"]
        row = {
            "patientId": patient_id,
            "truth": int(len(record["truth_boxes"]) > 0),
            "classifier_0_probability": float(classifier_records[0][patient_id]["classifier_probability"]),
            "classifier_1_probability": float(classifier_records[1][patient_id]["classifier_probability"]),
            "detector_max_score": float(np.max(scores)) if len(scores) else 0.0,
            "detector_num_boxes_at_threshold": int((scores >= args.detector_threshold).sum()) if len(scores) else 0,
        }
        rows.append(row)

    features = pd.DataFrame(rows)
    feature_path = output_dir / "component_test_features.csv"
    features.to_csv(feature_path, index=False)
    print(f"Saved computed feature table: {feature_path}")
    return features


def require_columns(features, columns):
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")


def component_report(features, output_dir, args):
    components = [
        {
            "name": args.swin_name,
            "score_column": args.swin_column,
            "threshold": args.swin_threshold,
        },
        {
            "name": args.cxr_swin_name,
            "score_column": args.cxr_swin_column,
            "threshold": args.cxr_swin_threshold,
        },
        {
            "name": args.detector_name,
            "score_column": args.detector_column,
            "threshold": args.detector_threshold,
        },
    ]
    require_columns(features, ["patientId", "truth"] + [component["score_column"] for component in components])

    prediction_table = features[["patientId", "truth"]].copy()
    metric_rows = []
    false_positive_tables = []

    for component in components:
        name = component["name"]
        score_column = component["score_column"]
        threshold = float(component["threshold"])
        score_values = features[score_column].astype(float)
        predictions = (score_values >= threshold).astype(int)

        prediction_table[f"{name}_score"] = score_values
        prediction_table[f"{name}_prediction"] = predictions

        metrics = confusion_metrics(features["truth"], predictions)
        metric_row = {
            "component": name,
            "score_column": score_column,
            "threshold": threshold,
            **metrics,
        }
        metric_rows.append(metric_row)

        component_predictions = pd.DataFrame(
            {
                "patientId": features["patientId"],
                "truth": features["truth"].astype(int),
                "prediction": predictions,
                "score": score_values,
            }
        )
        component_predictions.to_csv(output_dir / f"{name}_test_predictions.csv", index=False)

        false_positives = component_predictions[
            (component_predictions["truth"] == 0) & (component_predictions["prediction"] == 1)
        ].sort_values("score", ascending=False)
        false_positives.insert(0, "component", name)
        false_positives.to_csv(output_dir / f"{name}_false_positives.csv", index=False)
        (output_dir / f"{name}_false_positive_patient_ids.txt").write_text(
            "\n".join(false_positives["patientId"].astype(str).tolist()) + ("\n" if len(false_positives) else ""),
            encoding="utf-8",
        )
        false_positive_tables.append(false_positives)

        save_confusion_matrix_image(
            output_dir / f"{name}_confusion_matrix.png",
            metrics,
            f"{name} test confusion, threshold={threshold:g}",
        )
        print_component_report(name, metrics, threshold, false_positives, args.print_fp_ids)

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(output_dir / "component_test_metrics.csv", index=False)
    with (output_dir / "component_test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metric_rows, metrics_file, indent=2)

    prediction_table.to_csv(output_dir / "component_test_predictions.csv", index=False)
    pd.concat(false_positive_tables, ignore_index=True).to_csv(
        output_dir / "component_false_positives.csv",
        index=False,
    )

    print(f"\nSaved component metrics: {output_dir / 'component_test_metrics.csv'}")
    print(f"Saved all component predictions: {output_dir / 'component_test_predictions.csv'}")
    print(f"Saved combined false positives: {output_dir / 'component_false_positives.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print confusion matrices and false-positive patient IDs for the best fusion model components."
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--feature-csv",
        default=None,
        help="Optional fusion_test_features.csv. If provided, the report runs without model inference.",
    )
    parser.add_argument("--swin-checkpoint", default=DEFAULT_SWIN_CHECKPOINT)
    parser.add_argument("--cxr-swin-checkpoint", default=DEFAULT_CXR_SWIN_CHECKPOINT)
    parser.add_argument("--detector-checkpoint", default=DEFAULT_DETECTOR_CHECKPOINT)
    parser.add_argument("--swin-column", default="classifier_0_probability")
    parser.add_argument("--cxr-swin-column", default="classifier_1_probability")
    parser.add_argument("--detector-column", default="detector_max_score")
    parser.add_argument("--swin-name", default="swin")
    parser.add_argument("--cxr-swin-name", default="cxr_bce_swin")
    parser.add_argument("--detector-name", default="faster_rcnn")
    parser.add_argument("--swin-threshold", type=float, default=0.50)
    parser.add_argument("--cxr-swin-threshold", type=float, default=0.50)
    parser.add_argument("--detector-threshold", type=float, default=0.54)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--detector-min-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--print-fp-ids", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path("component_error_report").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to: {output_dir}")

    if args.feature_csv:
        print(f"Using saved feature CSV: {args.feature_csv}")
        features = load_feature_table(args.feature_csv)
    else:
        features = build_feature_table_from_checkpoints(args, output_dir)

    component_report(features, output_dir, args)


if __name__ == "__main__":
    main()

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import collate, get_data_loader, prepare_data, resolve_data_dir, seed_worker
from model_FRCNN import model as build_detector
from pipeline_FRCNN import autocast_context as detector_autocast_context
from testing_FRCNN import (
    calculate_image_precision,
    choose_device,
    matched_prediction_ious,
    parse_thresholds,
    save_confusion_matrix,
    set_seed,
)
from threshold_sweep import load_checkpoint, parse_score_thresholds
from train_classifier import (
    RSNAClassificationDataset,
    autocast_context as classifier_autocast_context,
    build_classifier,
    classification_collate,
    classifier_eval_transform,
    normalization_for_architecture,
)


def parse_modes(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_classifier(classifier, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def collect_classifier_probabilities(dataloader, classifier, device, use_amp=False, desc="Classifier"):
    rows = []
    classifier.eval()
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for images, labels, image_ids in tqdm(dataloader, total=len(dataloader), desc=desc, leave=False):
            images = images.to(device, non_blocking=non_blocking)
            with classifier_autocast_context(use_amp):
                logits = classifier(images).flatten()
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            for image_id, label, probability in zip(image_ids, labels_np, probabilities):
                rows.append(
                    {
                        "patientId": image_id,
                        "truth": int(label),
                        "classifier_probability": float(probability),
                    }
                )
    return {row["patientId"]: row for row in rows}


def collect_detector_predictions(dataloader, detector, device, use_amp=False, desc="Detector"):
    records = []
    detector.eval()
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for images, targets, image_ids in tqdm(dataloader, total=len(dataloader), desc=desc, leave=False):
            images = [image.to(device, non_blocking=non_blocking) for image in images]
            with detector_autocast_context(use_amp):
                outputs = detector(images)

            for index, output in enumerate(outputs):
                records.append(
                    {
                        "patientId": image_ids[index],
                        "truth_boxes": targets[index]["boxes"].detach().cpu().numpy(),
                        "boxes": output["boxes"].detach().cpu().numpy(),
                        "scores": output["scores"].detach().cpu().numpy(),
                    }
                )
    return records


def filter_detector_predictions(boxes, scores, score_threshold, max_detections):
    if len(scores) == 0:
        return boxes, scores

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    if len(scores) == 0:
        return boxes, scores

    sorted_indexes = np.argsort(scores)[::-1]
    if max_detections is not None and max_detections > 0:
        sorted_indexes = sorted_indexes[:max_detections]
    return boxes[sorted_indexes], scores[sorted_indexes]


def combine_prediction(mode, classifier_positive, detector_positive):
    if mode == "classifier":
        return classifier_positive
    if mode == "detector":
        return detector_positive
    if mode == "and":
        return classifier_positive and detector_positive
    if mode == "or":
        return classifier_positive or detector_positive
    raise ValueError(f"Unsupported mode: {mode}")


def kept_boxes_for_mode(mode, predicted_label, classifier_positive, detector_boxes):
    if mode in {"classifier", "and"} and not classifier_positive:
        return detector_boxes[:0]
    if not predicted_label:
        return detector_boxes[:0]
    return detector_boxes


def metrics_from_records(
    detector_records,
    classifier_records,
    detector_iou_thresholds,
    classifier_threshold,
    detector_score_threshold,
    mode,
    max_detections,
    metric_iou_threshold,
):
    image_precisions = []
    matched_ious = []
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    prediction_rows = []

    for record in detector_records:
        patient_id = record["patientId"]
        classifier_probability = classifier_records[patient_id]["classifier_probability"]
        classifier_positive = classifier_probability >= classifier_threshold
        detector_boxes, detector_scores = filter_detector_predictions(
            record["boxes"],
            record["scores"],
            score_threshold=detector_score_threshold,
            max_detections=max_detections,
        )
        detector_positive = len(detector_boxes) > 0
        predicted_label = combine_prediction(mode, classifier_positive, detector_positive)
        kept_boxes = kept_boxes_for_mode(mode, predicted_label, classifier_positive, detector_boxes)

        truth_boxes = record["truth_boxes"]
        true_label = int(len(truth_boxes) > 0)
        predicted_label_int = int(predicted_label)

        image_precisions.append(calculate_image_precision(truth_boxes, kept_boxes, detector_iou_thresholds))
        matched_ious.extend(matched_prediction_ious(truth_boxes, kept_boxes, metric_iou_threshold))

        if true_label == 1 and predicted_label_int == 1:
            true_positive += 1
        elif true_label == 0 and predicted_label_int == 1:
            false_positive += 1
        elif true_label == 0 and predicted_label_int == 0:
            true_negative += 1
        else:
            false_negative += 1

        prediction_rows.append(
            {
                "patientId": patient_id,
                "truth": true_label,
                "prediction": predicted_label_int,
                "classifier_probability": classifier_probability,
                "classifier_positive": int(classifier_positive),
                "detector_positive": int(detector_positive),
                "num_predicted_boxes": len(kept_boxes),
                "max_detector_score": float(detector_scores[0]) if len(detector_scores) else 0.0,
            }
        )

    total = true_positive + true_negative + false_positive + false_negative
    image_precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * image_precision * recall / max(image_precision + recall, 1e-12)
    metrics = {
        "detector_precision": float(pd.Series(image_precisions).mean()) if image_precisions else 0.0,
        "mean_iou": float(pd.Series(matched_ious).mean()) if matched_ious else 0.0,
        "image_accuracy": (true_positive + true_negative) / max(total, 1),
        "image_precision": image_precision,
        "image_recall": recall,
        "image_f1": f1,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "num_images": total,
        "num_matched_detections": len(matched_ious),
        "classifier_threshold": classifier_threshold,
        "detector_score_threshold": detector_score_threshold,
        "mode": mode,
        "metric_iou_threshold": metric_iou_threshold,
    }
    return metrics, pd.DataFrame(prediction_rows)


def make_classifier_loaders(args, data_dir, device, image_size, normalization):
    _, valid_detection, test_detection = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    valid_dataset = RSNAClassificationDataset(
        valid_detection,
        classifier_eval_transform(image_size, normalization=normalization),
    )
    test_dataset = RSNAClassificationDataset(
        test_detection,
        classifier_eval_transform(image_size, normalization=normalization),
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=False,
        collate_fn=classification_collate,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=False,
        collate_fn=classification_collate,
        **loader_kwargs,
    )
    return valid_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep classifier + detector image-level decision rules.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--detector-min-size", type=int, default=1024)
    parser.add_argument("--classifier-thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--detector-score-thresholds", default="0.35,0.40,0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.65,0.70")
    parser.add_argument("--modes", default="classifier,and,or,detector")
    parser.add_argument("--detector-iou-thresholds", default="0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75")
    parser.add_argument("--metric-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "combined_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_checkpoint_path = Path(args.classifier_checkpoint).expanduser().resolve()
    detector_checkpoint_path = Path(args.detector_checkpoint).expanduser().resolve()

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    classifier_thresholds = parse_score_thresholds(args.classifier_thresholds)
    detector_score_thresholds = parse_score_thresholds(args.detector_score_thresholds)
    detector_iou_thresholds = parse_thresholds(args.detector_iou_thresholds)
    modes = parse_modes(args.modes)

    classifier_checkpoint = torch.load(classifier_checkpoint_path, map_location=device)
    architecture = classifier_checkpoint.get("architecture", "resnet50")
    image_size = int(classifier_checkpoint.get("image_size", 512))
    normalization = classifier_checkpoint.get("normalization", normalization_for_architecture(architecture, "auto"))

    print(f"Using data directory: {data_dir}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using classifier checkpoint: {classifier_checkpoint_path}")
    print(f"Using detector checkpoint: {detector_checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"Classifier architecture: {architecture}, image size: {image_size}, normalization: {normalization}")

    classifier = build_classifier(
        architecture=architecture,
        pretrained=False,
        image_size=image_size,
    ).to(device)
    load_classifier(classifier, classifier_checkpoint_path, device)
    detector = build_detector(pretrained=False, min_size=args.detector_min_size).to(device)
    detector_checkpoint = load_checkpoint(detector, detector_checkpoint_path, device)

    classifier_valid_loader, classifier_test_loader = make_classifier_loaders(
        args,
        data_dir,
        device,
        image_size,
        normalization,
    )
    _, detector_valid_loader, detector_test_loader = get_data_loader(
        args.batch_size,
        data_dir=data_dir,
        num_workers=args.num_workers,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    classifier_valid = collect_classifier_probabilities(
        classifier_valid_loader,
        classifier,
        device,
        use_amp=use_amp,
        desc="validation classifier",
    )
    detector_valid = collect_detector_predictions(
        detector_valid_loader,
        detector,
        device,
        use_amp=use_amp,
        desc="validation detector",
    )

    rows = []
    best = None
    rules = list(product(modes, classifier_thresholds, detector_score_thresholds))
    print(f"Evaluating {len(rules)} validation decision rules")
    for mode, classifier_threshold, detector_score_threshold in rules:
        metrics, _ = metrics_from_records(
            detector_valid,
            classifier_valid,
            detector_iou_thresholds,
            classifier_threshold=classifier_threshold,
            detector_score_threshold=detector_score_threshold,
            mode=mode,
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
        )
        row = {
            "classifier_checkpoint": str(classifier_checkpoint_path),
            "classifier_checkpoint_epoch": classifier_checkpoint.get("epoch"),
            "detector_checkpoint": str(detector_checkpoint_path),
            "detector_checkpoint_epoch": detector_checkpoint.get("epoch"),
            **metrics,
        }
        rows.append(row)
        if best is None or (
            row["image_f1"],
            row["image_precision"],
            row["detector_precision"],
            row["mean_iou"],
        ) > (
            best["image_f1"],
            best["image_precision"],
            best["detector_precision"],
            best["mean_iou"],
        ):
            best = row
            print(
                "New best validation: "
                f"mode={mode}, clf={classifier_threshold}, det={detector_score_threshold}, "
                f"F1={row['image_f1']:.5f}, precision={row['image_precision']:.5f}, "
                f"recall={row['image_recall']:.5f}, detector_precision={row['detector_precision']:.5f}"
            )

    validation_table = pd.DataFrame(rows).sort_values(
        ["image_f1", "image_precision", "detector_precision", "mean_iou"],
        ascending=[False, False, False, False],
    )
    validation_table.to_csv(output_dir / "classifier_detector_sweep_validation.csv", index=False)

    classifier_test = collect_classifier_probabilities(
        classifier_test_loader,
        classifier,
        device,
        use_amp=use_amp,
        desc="test classifier",
    )
    detector_test = collect_detector_predictions(
        detector_test_loader,
        detector,
        device,
        use_amp=use_amp,
        desc="test detector",
    )
    test_metrics, test_predictions = metrics_from_records(
        detector_test,
        classifier_test,
        detector_iou_thresholds,
        classifier_threshold=float(best["classifier_threshold"]),
        detector_score_threshold=float(best["detector_score_threshold"]),
        mode=best["mode"],
        max_detections=args.max_detections,
        metric_iou_threshold=args.metric_iou_threshold,
    )
    selected = {
        "selected_from": "validation_classifier_detector_sweep",
        "classifier_checkpoint": str(classifier_checkpoint_path),
        "classifier_checkpoint_epoch": classifier_checkpoint.get("epoch"),
        "detector_checkpoint": str(detector_checkpoint_path),
        "detector_checkpoint_epoch": detector_checkpoint.get("epoch"),
        **test_metrics,
    }
    pd.DataFrame([selected]).to_csv(output_dir / "classifier_detector_sweep_test_metrics.csv", index=False)
    with (output_dir / "classifier_detector_sweep_test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(selected, metrics_file, indent=2)
    test_predictions.to_csv(output_dir / "classifier_detector_sweep_test_predictions.csv", index=False)
    save_confusion_matrix(
        output_dir / "classifier_detector_sweep_test_confusion_matrix.png",
        selected,
        (
            f"Classifier + Detector Test, mode={selected['mode']}, "
            f"clf={selected['classifier_threshold']:g}, det={selected['detector_score_threshold']:g}"
        ),
    )

    print("\nBest validation classifier + detector rule:")
    print(pd.Series(best).to_string())
    print("\nHeld-out test metrics at selected rule:")
    print(pd.Series(selected).to_string())


if __name__ == "__main__":
    main()

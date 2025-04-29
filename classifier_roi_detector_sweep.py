import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import find_training_image_directory, prepare_data, resolve_data_dir, seed_worker
from roi_utils import ROICandidateDataset, roi_eval_transform, truth_boxes_for_image
from testing_FRCNN import (
    calculate_image_precision,
    choose_device,
    matched_prediction_ious,
    parse_thresholds,
    save_confusion_matrix,
    set_seed,
)
from threshold_sweep import parse_score_thresholds
from train_classifier import (
    RSNAClassificationDataset,
    autocast_context,
    build_classifier,
    classification_collate,
    classifier_eval_transform,
    normalization_for_architecture,
)


def parse_modes(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def split_candidates(candidates, split_name):
    aliases = {
        "validation": {"validation", "valid", "val"},
        "test": {"test"},
    }
    return candidates[candidates["split"].isin(aliases[split_name])].copy()


def make_loader(dataset, batch_size, num_workers, device, collate_fn=None):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker
    return DataLoader(dataset, **loader_kwargs)


def collect_image_classifier_probabilities(dataloader, classifier, device, use_amp, desc):
    rows = []
    classifier.eval()
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for images, labels, image_ids in tqdm(dataloader, total=len(dataloader), desc=desc, leave=False):
            images = images.to(device, non_blocking=non_blocking)
            with autocast_context(use_amp):
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


def collect_roi_probabilities(dataloader, classifier, device, use_amp, desc):
    rows = []
    classifier.eval()
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for crops, _, candidate_ids, patient_ids in tqdm(dataloader, total=len(dataloader), desc=desc, leave=False):
            crops = crops.to(device, non_blocking=non_blocking)
            with autocast_context(use_amp):
                logits = classifier(crops).flatten()
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            for candidate_id, patient_id, probability in zip(candidate_ids, patient_ids, probabilities):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "patientId": patient_id,
                        "roi_probability": float(probability),
                    }
                )
    return pd.DataFrame(rows)


def add_roi_probabilities(candidates, probabilities):
    merged = candidates.merge(probabilities[["candidate_id", "roi_probability"]], on="candidate_id", how="left")
    if merged["roi_probability"].isna().any():
        missing = int(merged["roi_probability"].isna().sum())
        raise ValueError(f"Missing ROI probabilities for {missing} candidates")
    return merged


def filter_roi_candidates(candidates, detector_score_threshold, roi_threshold, max_detections):
    if candidates is None or len(candidates) == 0:
        return candidates

    kept = candidates[
        (candidates["detector_score"] >= detector_score_threshold)
        & (candidates["roi_probability"] >= roi_threshold)
    ].copy()
    if len(kept) == 0:
        return kept

    kept["ranking_score"] = kept["detector_score"] * kept["roi_probability"]
    kept = kept.sort_values("ranking_score", ascending=False)
    if max_detections is not None and max_detections > 0:
        kept = kept.head(max_detections)
    return kept


def combine_prediction(mode, classifier_positive, roi_detector_positive):
    if mode == "classifier":
        return classifier_positive
    if mode == "roi_detector":
        return roi_detector_positive
    if mode == "and":
        return classifier_positive and roi_detector_positive
    if mode == "or":
        return classifier_positive or roi_detector_positive
    raise ValueError(f"Unsupported mode: {mode}")


def metrics_from_records(
    detection_dataset,
    candidate_table,
    classifier_records,
    detector_iou_thresholds,
    classifier_threshold,
    detector_score_threshold,
    roi_threshold,
    mode,
    max_detections,
    metric_iou_threshold,
):
    candidate_groups = {
        patient_id: rows.reset_index(drop=True)
        for patient_id, rows in candidate_table.groupby("patientId", sort=False)
    }
    image_precisions = []
    matched_ious = []
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    prediction_rows = []

    for image_id in detection_dataset.image_ids:
        truth_boxes = truth_boxes_for_image(detection_dataset, image_id)
        candidates = candidate_groups.get(image_id)
        kept = filter_roi_candidates(
            candidates,
            detector_score_threshold=detector_score_threshold,
            roi_threshold=roi_threshold,
            max_detections=max_detections,
        )
        if kept is None or len(kept) == 0:
            roi_detector_positive = False
            kept_boxes = np.zeros((0, 4), dtype=np.float32)
            max_detector_score = 0.0
            max_roi_probability = 0.0
        else:
            roi_detector_positive = True
            kept_boxes = kept[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32).reshape(-1, 4)
            max_detector_score = float(kept["detector_score"].max())
            max_roi_probability = float(kept["roi_probability"].max())

        classifier_probability = classifier_records[image_id]["classifier_probability"]
        classifier_positive = classifier_probability >= classifier_threshold
        predicted_label = combine_prediction(mode, classifier_positive, roi_detector_positive)
        if not predicted_label:
            kept_boxes = np.zeros((0, 4), dtype=np.float32)
        elif mode == "and" and not classifier_positive:
            kept_boxes = np.zeros((0, 4), dtype=np.float32)

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

        num_candidates = 0 if candidates is None else len(candidates)
        prediction_rows.append(
            {
                "patientId": image_id,
                "truth": true_label,
                "prediction": predicted_label_int,
                "classifier_probability": classifier_probability,
                "classifier_positive": int(classifier_positive),
                "roi_detector_positive": int(roi_detector_positive),
                "num_candidates": num_candidates,
                "num_predicted_boxes": len(kept_boxes),
                "max_detector_score": max_detector_score,
                "max_roi_probability": max_roi_probability,
            }
        )

    total = true_positive + true_negative + false_positive + false_negative
    image_precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * image_precision * recall / max(image_precision + recall, 1e-12)
    metrics = {
        "detector_precision": float(np.mean(image_precisions)) if image_precisions else 0.0,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
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
        "roi_threshold": roi_threshold,
        "mode": mode,
        "metric_iou_threshold": metric_iou_threshold,
    }
    return metrics, pd.DataFrame(prediction_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep whole-image classifier + ROI filter + detector decision rules.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--roi-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--classifier-thresholds", default="0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68")
    parser.add_argument("--detector-score-thresholds", default="0.35,0.40,0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60")
    parser.add_argument("--roi-thresholds", default="0.30,0.35,0.40,0.45,0.48,0.50,0.52,0.55,0.60,0.65,0.70")
    parser.add_argument("--modes", default="and,or,roi_detector")
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
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "classifier_roi_detector_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = Path(args.candidate_csv).expanduser().resolve()
    classifier_checkpoint_path = Path(args.classifier_checkpoint).expanduser().resolve()
    roi_checkpoint_path = Path(args.roi_checkpoint).expanduser().resolve()
    detector_checkpoint_path = Path(args.detector_checkpoint).expanduser().resolve()
    image_directory = find_training_image_directory(data_dir)

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    classifier_thresholds = parse_score_thresholds(args.classifier_thresholds)
    detector_score_thresholds = parse_score_thresholds(args.detector_score_thresholds)
    roi_thresholds = parse_score_thresholds(args.roi_thresholds)
    detector_iou_thresholds = parse_thresholds(args.detector_iou_thresholds)
    modes = parse_modes(args.modes)

    image_checkpoint = torch.load(classifier_checkpoint_path, map_location=device)
    image_architecture = image_checkpoint.get("architecture", "swin_t")
    image_size = int(image_checkpoint.get("image_size", 512))
    image_normalization = image_checkpoint.get(
        "normalization",
        normalization_for_architecture(image_architecture, "auto"),
    )
    roi_checkpoint = torch.load(roi_checkpoint_path, map_location=device)
    roi_architecture = roi_checkpoint.get("architecture", "swin_t")
    roi_image_size = int(roi_checkpoint.get("image_size", 384))
    roi_padding_ratio = float(roi_checkpoint.get("padding_ratio", 0.25))
    detector_checkpoint = torch.load(detector_checkpoint_path, map_location="cpu")
    detector_epoch = detector_checkpoint.get("epoch") if isinstance(detector_checkpoint, dict) else None

    print(f"Using data directory: {data_dir}")
    print(f"Using candidate CSV: {candidate_csv}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using classifier checkpoint: {classifier_checkpoint_path}")
    print(f"Using ROI checkpoint: {roi_checkpoint_path}")
    print(f"Using detector checkpoint metadata: {detector_checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(
        f"Image classifier: {image_architecture}, image size: {image_size}, "
        f"normalization: {image_normalization}"
    )
    print(f"ROI classifier: {roi_architecture}, image size: {roi_image_size}, padding: {roi_padding_ratio:g}")

    image_classifier = build_classifier(
        architecture=image_architecture,
        pretrained=False,
        image_size=image_size,
    ).to(device)
    image_classifier.load_state_dict(image_checkpoint["model_state_dict"])

    roi_classifier = build_classifier(
        architecture=roi_architecture,
        pretrained=False,
        image_size=roi_image_size,
    ).to(device)
    roi_classifier.load_state_dict(roi_checkpoint["model_state_dict"])

    _, valid_dataset, test_dataset = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    valid_image_dataset = RSNAClassificationDataset(
        valid_dataset,
        classifier_eval_transform(image_size, normalization=image_normalization),
    )
    test_image_dataset = RSNAClassificationDataset(
        test_dataset,
        classifier_eval_transform(image_size, normalization=image_normalization),
    )
    valid_image_loader = make_loader(
        valid_image_dataset,
        args.classifier_batch_size,
        args.num_workers,
        device,
        collate_fn=classification_collate,
    )
    test_image_loader = make_loader(
        test_image_dataset,
        args.classifier_batch_size,
        args.num_workers,
        device,
        collate_fn=classification_collate,
    )

    candidates = pd.read_csv(candidate_csv)
    valid_candidates = split_candidates(candidates, "validation")
    test_candidates = split_candidates(candidates, "test")
    if valid_candidates.empty:
        raise ValueError("No validation candidates were found.")
    if test_candidates.empty:
        raise ValueError("No test candidates were found.")

    valid_roi_dataset = ROICandidateDataset(
        valid_candidates,
        image_directory,
        transform=roi_eval_transform(roi_image_size),
        padding_ratio=roi_padding_ratio,
    )
    test_roi_dataset = ROICandidateDataset(
        test_candidates,
        image_directory,
        transform=roi_eval_transform(roi_image_size),
        padding_ratio=roi_padding_ratio,
    )
    valid_roi_loader = make_loader(valid_roi_dataset, args.batch_size, args.num_workers, device)
    test_roi_loader = make_loader(test_roi_dataset, args.batch_size, args.num_workers, device)

    valid_classifier_records = collect_image_classifier_probabilities(
        valid_image_loader,
        image_classifier,
        device,
        use_amp,
        desc="validation image classifier",
    )
    test_classifier_records = collect_image_classifier_probabilities(
        test_image_loader,
        image_classifier,
        device,
        use_amp,
        desc="test image classifier",
    )
    valid_roi_probabilities = collect_roi_probabilities(
        valid_roi_loader,
        roi_classifier,
        device,
        use_amp,
        desc="validation ROI classifier",
    )
    test_roi_probabilities = collect_roi_probabilities(
        test_roi_loader,
        roi_classifier,
        device,
        use_amp,
        desc="test ROI classifier",
    )
    valid_candidates = add_roi_probabilities(valid_candidates, valid_roi_probabilities)
    test_candidates = add_roi_probabilities(test_candidates, test_roi_probabilities)
    valid_candidates.to_csv(output_dir / "classifier_roi_detector_validation_candidate_scores.csv", index=False)
    test_candidates.to_csv(output_dir / "classifier_roi_detector_test_candidate_scores.csv", index=False)

    rows = []
    best = None
    rules = list(product(modes, classifier_thresholds, detector_score_thresholds, roi_thresholds))
    print(f"Evaluating {len(rules)} validation combined rules")
    for mode, classifier_threshold, detector_score_threshold, roi_threshold in rules:
        metrics, _ = metrics_from_records(
            valid_dataset,
            valid_candidates,
            valid_classifier_records,
            detector_iou_thresholds,
            classifier_threshold=classifier_threshold,
            detector_score_threshold=detector_score_threshold,
            roi_threshold=roi_threshold,
            mode=mode,
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
        )
        row = {
            "classifier_checkpoint": str(classifier_checkpoint_path),
            "classifier_checkpoint_epoch": image_checkpoint.get("epoch"),
            "roi_checkpoint": str(roi_checkpoint_path),
            "roi_checkpoint_epoch": roi_checkpoint.get("epoch"),
            "detector_checkpoint": str(detector_checkpoint_path),
            "detector_checkpoint_epoch": detector_epoch,
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
                f"mode={mode}, clf={classifier_threshold}, det={detector_score_threshold}, roi={roi_threshold}, "
                f"F1={row['image_f1']:.5f}, precision={row['image_precision']:.5f}, "
                f"recall={row['image_recall']:.5f}, detector_precision={row['detector_precision']:.5f}"
            )

    validation_table = pd.DataFrame(rows).sort_values(
        ["image_f1", "image_precision", "detector_precision", "mean_iou"],
        ascending=[False, False, False, False],
    )
    validation_table.to_csv(output_dir / "classifier_roi_detector_sweep_validation.csv", index=False)

    test_metrics, test_predictions = metrics_from_records(
        test_dataset,
        test_candidates,
        test_classifier_records,
        detector_iou_thresholds,
        classifier_threshold=float(best["classifier_threshold"]),
        detector_score_threshold=float(best["detector_score_threshold"]),
        roi_threshold=float(best["roi_threshold"]),
        mode=best["mode"],
        max_detections=args.max_detections,
        metric_iou_threshold=args.metric_iou_threshold,
    )
    selected = {
        "selected_from": "validation_classifier_roi_detector_sweep",
        "classifier_checkpoint": str(classifier_checkpoint_path),
        "classifier_checkpoint_epoch": image_checkpoint.get("epoch"),
        "roi_checkpoint": str(roi_checkpoint_path),
        "roi_checkpoint_epoch": roi_checkpoint.get("epoch"),
        "detector_checkpoint": str(detector_checkpoint_path),
        "detector_checkpoint_epoch": detector_epoch,
        **test_metrics,
    }
    pd.DataFrame([selected]).to_csv(output_dir / "classifier_roi_detector_sweep_test_metrics.csv", index=False)
    with (output_dir / "classifier_roi_detector_sweep_test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(selected, metrics_file, indent=2)
    test_predictions.to_csv(output_dir / "classifier_roi_detector_sweep_test_predictions.csv", index=False)
    save_confusion_matrix(
        output_dir / "classifier_roi_detector_sweep_test_confusion_matrix.png",
        selected,
        (
            f"Classifier + ROI + Detector Test, mode={selected['mode']}, "
            f"clf={selected['classifier_threshold']:g}, det={selected['detector_score_threshold']:g}, "
            f"roi={selected['roi_threshold']:g}"
        ),
    )

    print("\nBest validation classifier + ROI + detector rule:")
    print(pd.Series(best).to_string())
    print("\nHeld-out test metrics at selected combined rule:")
    print(pd.Series(selected).to_string())


if __name__ == "__main__":
    main()

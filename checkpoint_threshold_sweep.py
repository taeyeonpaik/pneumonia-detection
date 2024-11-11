import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from dataset import get_data_loader, resolve_data_dir
from model_FRCNN import model as build_model
from pipeline_FRCNN import autocast_context
from testing_FRCNN import (
    calculate_image_precision,
    choose_device,
    evaluate_labeled_split,
    filter_predictions,
    matched_prediction_ious,
    parse_thresholds,
    save_confusion_matrix,
    set_seed,
)
from threshold_sweep import load_checkpoint, parse_score_thresholds


def find_checkpoints(output_dir, pattern):
    checkpoints = sorted(output_dir.glob(pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir} matching {pattern}")
    return checkpoints


def collect_predictions(dataloader, model, device, use_amp=False, desc="Predicting"):
    records = []
    model.eval()
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for images, targets, image_ids in tqdm(dataloader, total=len(dataloader), desc=desc, leave=False):
            images = [image.to(device, non_blocking=non_blocking) for image in images]
            with autocast_context(use_amp):
                outputs = model(images)

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


def metrics_from_records(records, detector_iou_thresholds, score_threshold, max_detections, metric_iou_threshold):
    image_precisions = []
    matched_ious = []
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    for record in records:
        predictions, filtered_scores = filter_predictions(
            record["boxes"],
            record["scores"],
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        truth_boxes = record["truth_boxes"]
        image_precisions.append(calculate_image_precision(truth_boxes, predictions, detector_iou_thresholds))
        matched_ious.extend(matched_prediction_ious(truth_boxes, predictions, metric_iou_threshold))

        true_label = int(len(truth_boxes) > 0)
        predicted_label = int(len(predictions) > 0)

        if true_label == 1 and predicted_label == 1:
            true_positive += 1
        elif true_label == 0 and predicted_label == 1:
            false_positive += 1
        elif true_label == 0 and predicted_label == 0:
            true_negative += 1
        else:
            false_negative += 1

    total = true_positive + true_negative + false_positive + false_negative
    image_precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * image_precision * recall / max(image_precision + recall, 1e-12)

    return {
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
        "score_threshold": score_threshold,
        "metric_iou_threshold": metric_iou_threshold,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep checkpoints and confidence thresholds for best validation F1.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-pattern", default="frcnn_epoch_*.pt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--min-size", type=int, default=1024)
    parser.add_argument(
        "--score-thresholds",
        default="0.05,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.35,0.40,0.50",
    )
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
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    score_thresholds = parse_score_thresholds(args.score_thresholds)
    detector_iou_thresholds = parse_thresholds(args.detector_iou_thresholds)
    checkpoints = find_checkpoints(output_dir, args.checkpoint_pattern)

    print(f"Using data directory: {data_dir}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"Evaluating {len(checkpoints)} checkpoints x {len(score_thresholds)} thresholds")

    _, valid_loader, test_loader = get_data_loader(
        args.batch_size,
        data_dir=data_dir,
        num_workers=args.num_workers,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    rows = []
    best = None
    detector = build_model(pretrained=False, min_size=args.min_size).to(device)

    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(detector, checkpoint_path, device)
        detector.eval()
        checkpoint_epoch = checkpoint.get("epoch")
        print(f"\nCheckpoint: {checkpoint_path.name} (epoch={checkpoint_epoch})")
        validation_records = collect_predictions(
            valid_loader,
            detector,
            device,
            use_amp=use_amp,
            desc=f"validation predictions epoch {checkpoint_epoch}",
        )

        for score_threshold in score_thresholds:
            metrics = metrics_from_records(
                validation_records,
                detector_iou_thresholds,
                score_threshold,
                args.max_detections,
                args.metric_iou_threshold,
            )
            row = {
                "checkpoint": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch,
                **metrics,
            }
            rows.append(row)

            if best is None or (
                row["image_f1"],
                row["detector_precision"],
                row["mean_iou"],
            ) > (
                best["image_f1"],
                best["detector_precision"],
                best["mean_iou"],
            ):
                best = row
                print(
                    "New best validation: "
                    f"epoch={row['checkpoint_epoch']}, threshold={row['score_threshold']}, "
                    f"F1={row['image_f1']:.5f}, detector_precision={row['detector_precision']:.5f}, "
                    f"mean_iou={row['mean_iou']:.5f}"
                )

    validation_table = pd.DataFrame(rows).sort_values(
        ["image_f1", "detector_precision", "mean_iou"],
        ascending=[False, False, False],
    )
    validation_table.to_csv(output_dir / "checkpoint_threshold_sweep_validation.csv", index=False)

    best_checkpoint_path = Path(best["checkpoint_path"])
    load_checkpoint(detector, best_checkpoint_path, device)
    test_metrics, test_predictions = evaluate_labeled_split(
        test_loader,
        detector,
        device,
        detector_iou_thresholds,
        score_threshold=float(best["score_threshold"]),
        max_detections=args.max_detections,
        metric_iou_threshold=args.metric_iou_threshold,
        use_amp=use_amp,
        desc=f"test epoch {best['checkpoint_epoch']} threshold {best['score_threshold']:g}",
    )
    selected = {
        "selected_from": "validation_checkpoint_threshold_sweep",
        "checkpoint": best["checkpoint"],
        "checkpoint_epoch": best["checkpoint_epoch"],
        **test_metrics,
    }

    pd.DataFrame([selected]).to_csv(output_dir / "checkpoint_threshold_sweep_test_metrics.csv", index=False)
    with (output_dir / "checkpoint_threshold_sweep_test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(selected, metrics_file, indent=2)
    test_predictions.to_csv(output_dir / "checkpoint_threshold_sweep_test_predictions.csv", index=False)
    save_confusion_matrix(
        output_dir / "checkpoint_threshold_sweep_test_confusion_matrix.png",
        selected,
        f"Held-Out Test, epoch={selected['checkpoint_epoch']}, threshold={selected['score_threshold']:g}",
    )

    print("\nBest validation setting:")
    print(pd.Series(best).to_string())
    print("\nHeld-out test metrics at selected setting:")
    print(pd.Series(selected).to_string())


if __name__ == "__main__":
    main()

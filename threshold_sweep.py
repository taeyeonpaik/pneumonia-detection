import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from dataset import get_data_loader, resolve_data_dir
from model_FRCNN import model as build_model
from testing_FRCNN import choose_device, evaluate_labeled_split, parse_thresholds, save_confusion_matrix, set_seed


def parse_score_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        return {"epoch": None}
    return {"epoch": None}


def evaluate_thresholds(
    dataloader,
    model,
    device,
    detector_iou_thresholds,
    score_thresholds,
    metric_iou_threshold,
    max_detections,
    use_amp,
    split_name,
):
    rows = []
    prediction_tables = {}

    for score_threshold in score_thresholds:
        metrics, predictions = evaluate_labeled_split(
            dataloader,
            model,
            device,
            detector_iou_thresholds,
            score_threshold=score_threshold,
            max_detections=max_detections,
            metric_iou_threshold=metric_iou_threshold,
            use_amp=use_amp,
            desc=f"{split_name} threshold {score_threshold:g}",
        )
        rows.append({"split": split_name, **metrics})
        prediction_tables[score_threshold] = predictions

    metrics_table = pd.DataFrame(rows)
    metrics_table = metrics_table.sort_values(
        ["image_f1", "detector_precision", "mean_iou"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return metrics_table, prediction_tables


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for a trained Faster R-CNN checkpoint.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--min-size", type=int, default=1024)
    parser.add_argument("--score-thresholds", default="0.01,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
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

    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else output_dir / "frcnn_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    score_thresholds = parse_score_thresholds(args.score_thresholds)
    detector_iou_thresholds = parse_thresholds(args.detector_iou_thresholds)

    print(f"Using data directory: {data_dir}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")

    _, valid_loader, test_loader = get_data_loader(
        args.batch_size,
        data_dir=data_dir,
        num_workers=args.num_workers,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    detector = build_model(pretrained=False, min_size=args.min_size).to(device)
    checkpoint = load_checkpoint(detector, checkpoint_path, device)
    detector.eval()

    valid_table, _ = evaluate_thresholds(
        valid_loader,
        detector,
        device,
        detector_iou_thresholds,
        score_thresholds,
        args.metric_iou_threshold,
        args.max_detections,
        use_amp,
        "validation",
    )
    valid_table.to_csv(output_dir / "threshold_sweep_validation.csv", index=False)
    best_threshold = float(valid_table.iloc[0]["score_threshold"])

    print("\nBest validation threshold:")
    print(valid_table.iloc[0].to_string())

    test_metrics, test_predictions = evaluate_labeled_split(
        test_loader,
        detector,
        device,
        detector_iou_thresholds,
        score_threshold=best_threshold,
        max_detections=args.max_detections,
        metric_iou_threshold=args.metric_iou_threshold,
        use_amp=use_amp,
        desc=f"test selected threshold {best_threshold:g}",
    )
    test_metrics = {
        "selected_from": "validation",
        "checkpoint_epoch": checkpoint.get("epoch"),
        **test_metrics,
    }

    pd.DataFrame([test_metrics]).to_csv(output_dir / "threshold_sweep_test_metrics.csv", index=False)
    with (output_dir / "threshold_sweep_test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(test_metrics, metrics_file, indent=2)
    test_predictions.to_csv(output_dir / "threshold_sweep_test_predictions.csv", index=False)
    save_confusion_matrix(
        output_dir / "threshold_sweep_test_confusion_matrix.png",
        test_metrics,
        f"Held-Out Test Confusion Matrix, threshold={best_threshold:g}",
    )

    print("\nHeld-out test metrics at selected threshold:")
    print(pd.Series(test_metrics).to_string())


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import collate, get_valid_test_transform, prepare_data, resolve_data_dir
from model_FRCNN import model as build_model
from pipeline_FRCNN import autocast_context
from testing_FRCNN import choose_device, filter_predictions, set_seed
from threshold_sweep import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Mine false-positive training images for hard-negative retraining.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--min-size", type=int, default=1024)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--max-hard-negatives", type=int, default=2000)
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
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    train_dataset, _, _ = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    train_dataset.transform = get_valid_test_transform()
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(
        train_dataset,
        **loader_kwargs,
    )

    detector = build_model(pretrained=False, min_size=args.min_size).to(device)
    checkpoint = load_checkpoint(detector, checkpoint_path, device)
    detector.eval()

    print(f"Using data directory: {data_dir}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")

    rows = []
    false_positive_rows = []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for images, targets, image_ids in tqdm(dataloader, total=len(dataloader), desc="Mining hard negatives"):
            images = [image.to(device, non_blocking=non_blocking) for image in images]
            with autocast_context(use_amp):
                outputs = detector(images)

            for index, output in enumerate(outputs):
                truth_boxes = targets[index]["boxes"].detach().cpu().numpy()
                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                predictions, filtered_scores = filter_predictions(
                    boxes,
                    scores,
                    score_threshold=args.score_threshold,
                    max_detections=args.max_detections,
                )

                true_label = int(len(truth_boxes) > 0)
                predicted_label = int(len(predictions) > 0)
                max_score = float(filtered_scores[0]) if len(filtered_scores) else 0.0
                max_area = 0.0
                if len(predictions):
                    widths = np.maximum(0.0, predictions[:, 2] - predictions[:, 0])
                    heights = np.maximum(0.0, predictions[:, 3] - predictions[:, 1])
                    max_area = float(np.max(widths * heights))

                row = {
                    "patientId": image_ids[index],
                    "truth": true_label,
                    "prediction": predicted_label,
                    "max_score": max_score,
                    "num_predicted_boxes": len(predictions),
                    "max_box_area": max_area,
                }
                rows.append(row)
                if true_label == 0 and predicted_label == 1:
                    false_positive_rows.append(row)

    all_predictions = pd.DataFrame(rows)
    hard_negatives = pd.DataFrame(false_positive_rows)
    if not hard_negatives.empty:
        hard_negatives = hard_negatives.sort_values(
            ["max_score", "num_predicted_boxes", "max_box_area"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        hard_negatives = hard_negatives.head(args.max_hard_negatives)

    all_path = output_dir / "hard_negative_mining_train_predictions.csv"
    hard_path = output_dir / "hard_negatives_train.csv"
    ids_path = output_dir / "hard_negative_ids.csv"
    summary_path = output_dir / "hard_negative_mining_summary.json"

    all_predictions.to_csv(all_path, index=False)
    hard_negatives.to_csv(hard_path, index=False)
    hard_negatives[["patientId"]].to_csv(ids_path, index=False)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "score_threshold": args.score_threshold,
        "num_train_images": int(len(all_predictions)),
        "num_false_positive_negatives": int(len(false_positive_rows)),
        "num_saved_hard_negatives": int(len(hard_negatives)),
        "hard_negative_ids": str(ids_path),
        "hard_negative_details": str(hard_path),
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("\nHard-negative mining summary:")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()

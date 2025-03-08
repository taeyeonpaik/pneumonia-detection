import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import prepare_data, resolve_data_dir, seed_worker
from testing_FRCNN import choose_device, set_seed
from train_classifier import (
    RSNAClassificationDataset,
    autocast_context,
    build_classifier,
    classification_collate,
    classifier_eval_transform,
    normalization_for_architecture,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Mine false-positive training images for classifier hard-negative retraining.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--max-hard-negatives", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "classifier_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", "resnet50")
    image_size = int(checkpoint.get("image_size", 512))
    normalization = checkpoint.get("normalization", normalization_for_architecture(architecture, "auto"))
    score_threshold = args.score_threshold
    if score_threshold is None:
        score_threshold = float(checkpoint.get("validation_metrics", {}).get("score_threshold", 0.5))

    train_detection, _, _ = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    train_dataset = RSNAClassificationDataset(
        train_detection,
        classifier_eval_transform(image_size, normalization=normalization),
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": classification_collate,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker
    dataloader = DataLoader(train_dataset, **loader_kwargs)

    classifier = build_classifier(architecture=architecture, pretrained=False, image_size=image_size).to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    print(f"Using data directory: {data_dir}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"Classifier architecture: {architecture}, image size: {image_size}, normalization: {normalization}")
    print(f"Mining threshold: {score_threshold:g}")

    rows = []
    false_positive_rows = []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for images, labels, image_ids in tqdm(dataloader, total=len(dataloader), desc="Mining classifier hard negatives"):
            images = images.to(device, non_blocking=non_blocking)
            with autocast_context(use_amp):
                logits = classifier(images).flatten()
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            for image_id, label, probability in zip(image_ids, labels_np, probabilities):
                true_label = int(label)
                predicted_label = int(float(probability) >= score_threshold)
                row = {
                    "patientId": image_id,
                    "truth": true_label,
                    "prediction": predicted_label,
                    "probability": float(probability),
                }
                rows.append(row)
                if true_label == 0 and predicted_label == 1:
                    false_positive_rows.append(row)

    all_predictions = pd.DataFrame(rows)
    hard_negatives = pd.DataFrame(false_positive_rows)
    if not hard_negatives.empty:
        hard_negatives = hard_negatives.sort_values("probability", ascending=False).reset_index(drop=True)
        hard_negatives = hard_negatives.head(args.max_hard_negatives)

    all_path = output_dir / "classifier_hard_negative_mining_train_predictions.csv"
    hard_path = output_dir / "classifier_hard_negatives_train.csv"
    ids_path = output_dir / "classifier_hard_negative_ids.csv"
    summary_path = output_dir / "classifier_hard_negative_mining_summary.json"

    all_predictions.to_csv(all_path, index=False)
    hard_negatives.to_csv(hard_path, index=False)
    hard_negatives[["patientId"]].to_csv(ids_path, index=False)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "score_threshold": score_threshold,
        "num_train_images": int(len(all_predictions)),
        "num_false_positive_negatives": int(len(false_positive_rows)),
        "num_saved_hard_negatives": int(len(hard_negatives)),
        "hard_negative_ids": str(ids_path),
        "hard_negative_details": str(hard_path),
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("\nClassifier hard-negative mining summary:")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()

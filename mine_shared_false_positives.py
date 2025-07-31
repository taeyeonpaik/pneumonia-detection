#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import collate, prepare_data, resolve_data_dir, seed_worker
from fusion_sweep import collect_classifier_probabilities_tta
from model_FRCNN import model as build_detector
from pipeline_FRCNN import autocast_context
from testing_FRCNN import choose_device, filter_predictions, set_seed
from threshold_sweep import load_checkpoint
from train_classifier import (
    METADATA_FEATURE_COLUMNS,
    RSNAClassificationDataset,
    build_classifier,
    classifier_eval_transform,
    classification_collate,
    normalization_for_architecture,
    parse_tta_modes,
)
from dicom_metadata import load_metadata_table


def make_classifier_loaders(
    train_detection,
    checkpoint,
    metadata_by_id,
    batch_size,
    num_workers,
    device,
    tta_modes,
):
    architecture = checkpoint.get("architecture", "swin_t")
    image_size = int(checkpoint.get("image_size", 512))
    normalization = checkpoint.get("normalization", normalization_for_architecture(architecture, "auto"))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": classification_collate,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker

    loaders = []
    for mode in tta_modes:
        dataset = RSNAClassificationDataset(
            train_detection,
            classifier_eval_transform(image_size, normalization=normalization, tta_mode=mode),
            metadata_by_id=metadata_by_id,
        )
        loaders.append(
            (
                mode,
                DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs),
            )
        )
    return loaders


def collect_detector_train_predictions(
    train_detection,
    detector,
    batch_size,
    num_workers,
    device,
    use_amp,
    score_threshold,
    max_detections,
):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker
    dataloader = DataLoader(train_detection, **loader_kwargs)

    rows = []
    detector.eval()
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for images, targets, image_ids in tqdm(dataloader, total=len(dataloader), desc="train detector predictions"):
            images = [image.to(device, non_blocking=non_blocking) for image in images]
            with autocast_context(use_amp):
                outputs = detector(images)

            for index, output in enumerate(outputs):
                truth_boxes = targets[index]["boxes"].detach().cpu().numpy()
                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                filtered_boxes, filtered_scores = filter_predictions(
                    boxes,
                    scores,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                )
                max_score = float(filtered_scores[0]) if len(filtered_scores) else 0.0
                rows.append(
                    {
                        "patientId": image_ids[index],
                        "truth": int(len(truth_boxes) > 0),
                        "detector_probability": max_score,
                        "detector_prediction": int(len(filtered_boxes) > 0),
                        "detector_num_boxes": int(len(filtered_boxes)),
                    }
                )
    return pd.DataFrame(rows)


def resolve_threshold(value, checkpoint, default):
    if value is not None:
        return float(value)
    metrics = checkpoint.get("validation_metrics", {}) or {}
    return float(metrics.get("score_threshold", default))


def main():
    parser = argparse.ArgumentParser(
        description="Mine training-split normal images where both metadata-aware Swin and Faster R-CNN fire."
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--classifier-threshold", type=float, default=None)
    parser.add_argument("--detector-threshold", type=float, default=0.44)
    parser.add_argument("--min-classifier-probability", type=float, default=None)
    parser.add_argument("--min-detector-score", type=float, default=None)
    parser.add_argument("--max-hard-negatives", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detector-min-size", type=int, default=1024)
    parser.add_argument("--tta-modes", default="none,hflip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    train_detection, _, _ = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    classifier_checkpoint_path = Path(args.classifier_checkpoint).expanduser().resolve()
    detector_checkpoint_path = Path(args.detector_checkpoint).expanduser().resolve()
    classifier_checkpoint = torch.load(classifier_checkpoint_path, map_location=device)
    architecture = classifier_checkpoint.get("architecture", "swin_t")
    image_size = int(classifier_checkpoint.get("image_size", 512))
    metadata_dim = int(classifier_checkpoint.get("metadata_dim", 0))
    metadata_hidden_dim = int(classifier_checkpoint.get("metadata_hidden_dim", 32))
    metadata_by_id = load_metadata_table(args.metadata_csv) if metadata_dim > 0 else None
    classifier_threshold = resolve_threshold(args.classifier_threshold, classifier_checkpoint, 0.5)
    min_classifier_probability = (
        float(args.min_classifier_probability)
        if args.min_classifier_probability is not None
        else classifier_threshold
    )
    min_detector_score = (
        float(args.min_detector_score)
        if args.min_detector_score is not None
        else float(args.detector_threshold)
    )

    print(f"Using data directory: {data_dir}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Classifier checkpoint: {classifier_checkpoint_path}")
    print(f"Detector checkpoint: {detector_checkpoint_path}")
    print(f"Architecture={architecture}, image_size={image_size}, metadata_dim={metadata_dim}")
    print(f"Classifier threshold={classifier_threshold:g}, min={min_classifier_probability:g}")
    print(f"Detector threshold={args.detector_threshold:g}, min={min_detector_score:g}")
    print(f"Using device={device}, AMP={use_amp}")

    classifier = build_classifier(
        architecture=architecture,
        pretrained=False,
        image_size=image_size,
        metadata_dim=metadata_dim,
        metadata_hidden_dim=metadata_hidden_dim,
    ).to(device)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"])

    classifier_loaders = make_classifier_loaders(
        train_detection,
        classifier_checkpoint,
        metadata_by_id,
        args.classifier_batch_size,
        args.num_workers,
        device,
        parse_tta_modes(args.tta_modes),
    )
    classifier_records = collect_classifier_probabilities_tta(
        classifier_loaders,
        classifier,
        device,
        use_amp=use_amp,
        desc="train metadata Swin",
    )
    classifier_rows = pd.DataFrame(classifier_records.values()).rename(
        columns={"classifier_probability": "classifier_probability"}
    )
    classifier_rows["classifier_prediction"] = (
        classifier_rows["classifier_probability"] >= classifier_threshold
    ).astype(int)
    del classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()

    detector = build_detector(pretrained=False, min_size=args.detector_min_size).to(device)
    detector_checkpoint = load_checkpoint(detector, detector_checkpoint_path, device)
    detector_rows = collect_detector_train_predictions(
        train_detection,
        detector,
        args.batch_size,
        args.num_workers,
        device,
        use_amp,
        args.detector_threshold,
        max_detections=100,
    )
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    merged = classifier_rows.merge(detector_rows, on=["patientId", "truth"], how="inner")
    merged["shared_fp"] = (
        (merged["truth"] == 0)
        & (merged["classifier_probability"] >= min_classifier_probability)
        & (merged["detector_probability"] >= min_detector_score)
    ).astype(int)
    merged["shared_fp_score"] = (
        merged["classifier_probability"].astype(float) * merged["detector_probability"].astype(float)
    )
    shared = merged[merged["shared_fp"] == 1].sort_values(
        ["shared_fp_score", "classifier_probability", "detector_probability"],
        ascending=[False, False, False],
    )
    if args.max_hard_negatives > 0:
        shared = shared.head(args.max_hard_negatives)

    merged.to_csv(output_dir / "shared_train_predictions.csv", index=False)
    shared.to_csv(output_dir / "shared_false_positive_hard_negatives.csv", index=False)
    shared[["patientId"]].to_csv(output_dir / "shared_false_positive_ids.csv", index=False)
    shared[["patientId"]].to_csv(output_dir / "hard_negative_patient_ids.csv", index=False)

    summary = {
        "classifier_checkpoint": str(classifier_checkpoint_path),
        "detector_checkpoint": str(detector_checkpoint_path),
        "detector_checkpoint_epoch": detector_checkpoint.get("epoch"),
        "classifier_threshold": classifier_threshold,
        "detector_threshold": args.detector_threshold,
        "min_classifier_probability": min_classifier_probability,
        "min_detector_score": min_detector_score,
        "num_train_images": int(len(merged)),
        "num_shared_false_positives": int(len(shared)),
        "max_hard_negatives": int(args.max_hard_negatives),
    }
    (output_dir / "shared_false_positive_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("\nShared false-positive mining summary:")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()

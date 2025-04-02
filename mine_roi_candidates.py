import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import collate, get_valid_test_transform, prepare_data, resolve_data_dir, seed_worker
from model_FRCNN import model as build_detector
from pipeline_FRCNN import autocast_context
from roi_utils import roi_label_from_iou
from testing_FRCNN import box_iou, choose_device, filter_predictions, load_model_checkpoint, set_seed


def parse_splits(value):
    aliases = {"val": "validation", "valid": "validation"}
    splits = []
    for item in value.split(","):
        split = aliases.get(item.strip(), item.strip())
        if split:
            splits.append(split)
    return splits


def make_loader(dataset, batch_size, num_workers, device):
    dataset.transform = get_valid_test_transform()
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker
    return DataLoader(dataset, **loader_kwargs)


def mine_split(
    split_name,
    dataloader,
    detector,
    device,
    score_threshold,
    max_detections,
    positive_iou_threshold,
    negative_iou_threshold,
    use_amp,
    include_ground_truth_positives=False,
):
    rows = []
    detector.eval()
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for images, targets, image_ids in tqdm(dataloader, total=len(dataloader), desc=f"Mining {split_name}", leave=False):
            images = [image.to(device, non_blocking=non_blocking) for image in images]
            with autocast_context(use_amp):
                outputs = detector(images)

            for batch_index, output in enumerate(outputs):
                image_id = image_ids[batch_index]
                truth_boxes = targets[batch_index]["boxes"].detach().cpu().numpy()
                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                boxes, scores = filter_predictions(
                    boxes,
                    scores,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                )

                for candidate_index, (box, score) in enumerate(zip(boxes, scores)):
                    if len(truth_boxes):
                        ious = box_iou(box, truth_boxes)
                        max_iou = float(np.max(ious)) if len(ious) else 0.0
                    else:
                        max_iou = 0.0
                    roi_label = roi_label_from_iou(
                        max_iou,
                        positive_iou_threshold=positive_iou_threshold,
                        negative_iou_threshold=negative_iou_threshold,
                    )
                    if roi_label == 1:
                        candidate_type = "matched_detection"
                    elif roi_label == 0 and len(truth_boxes) == 0:
                        candidate_type = "false_positive_negative_image"
                    elif roi_label == 0:
                        candidate_type = "unmatched_positive_image"
                    else:
                        candidate_type = "ambiguous_detection"
                    rows.append(
                        {
                            "candidate_id": f"{split_name}_{image_id}_detector_{candidate_index}",
                            "split": split_name,
                            "patientId": image_id,
                            "source": "detector",
                            "candidate_type": candidate_type,
                            "is_detector_false_positive": int(roi_label == 0),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            "detector_score": float(score),
                            "max_iou": max_iou,
                            "roi_label": roi_label,
                            "num_truth_boxes": int(len(truth_boxes)),
                        }
                    )

                if include_ground_truth_positives and len(truth_boxes):
                    for truth_index, truth_box in enumerate(truth_boxes):
                        rows.append(
                            {
                                "candidate_id": f"{split_name}_{image_id}_ground_truth_{truth_index}",
                                "split": split_name,
                                "patientId": image_id,
                                "source": "ground_truth",
                                "candidate_type": "ground_truth",
                                "is_detector_false_positive": 0,
                                "x1": float(truth_box[0]),
                                "y1": float(truth_box[1]),
                                "x2": float(truth_box[2]),
                                "y2": float(truth_box[3]),
                                "detector_score": 1.0,
                                "max_iou": 1.0,
                                "roi_label": 1,
                                "num_truth_boxes": int(len(truth_boxes)),
                            }
                        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Mine Faster R-CNN candidate regions for an ROI false-positive filter.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--detector-min-size", type=int, default=1024)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--positive-iou-threshold", type=float, default=0.5)
    parser.add_argument("--negative-iou-threshold", type=float, default=0.3)
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-ground-truth-positives", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "roi_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_checkpoint_path = Path(args.detector_checkpoint).expanduser().resolve()

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using data directory: {data_dir}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using detector checkpoint: {detector_checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"Candidate detector score threshold: {args.score_threshold:g}")

    detector = build_detector(pretrained=False, min_size=args.detector_min_size).to(device)
    detector_checkpoint = load_model_checkpoint(detector, detector_checkpoint_path, device)
    print(f"Detector checkpoint epoch: {detector_checkpoint.get('epoch')}")

    train_dataset, valid_dataset, test_dataset = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    datasets = {
        "train": train_dataset,
        "validation": valid_dataset,
        "test": test_dataset,
    }

    all_rows = []
    requested_splits = parse_splits(args.splits)
    for split_name in requested_splits:
        if split_name not in datasets:
            raise ValueError(f"Unsupported split: {split_name}")
        loader = make_loader(datasets[split_name], args.batch_size, args.num_workers, device)
        all_rows.extend(
            mine_split(
                split_name,
                loader,
                detector,
                device,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                positive_iou_threshold=args.positive_iou_threshold,
                negative_iou_threshold=args.negative_iou_threshold,
                use_amp=use_amp,
                include_ground_truth_positives=(
                    split_name == "train" and not args.no_ground_truth_positives
                ),
            )
        )

    candidates = pd.DataFrame(all_rows)
    candidates_path = output_dir / "roi_candidates.csv"
    candidates.to_csv(candidates_path, index=False)
    for split_name, split_rows in candidates.groupby("split", sort=False):
        split_rows.to_csv(output_dir / f"roi_candidates_{split_name}.csv", index=False)

    trainable = candidates[candidates["roi_label"].isin([0, 1])]
    summary = {
        "detector_checkpoint": str(detector_checkpoint_path),
        "detector_checkpoint_epoch": detector_checkpoint.get("epoch"),
        "score_threshold": args.score_threshold,
        "max_detections": args.max_detections,
        "positive_iou_threshold": args.positive_iou_threshold,
        "negative_iou_threshold": args.negative_iou_threshold,
        "num_candidates": int(len(candidates)),
        "num_trainable_candidates": int(len(trainable)),
        "candidate_csv": str(candidates_path),
        "counts_by_split": candidates["split"].value_counts().to_dict(),
        "counts_by_source": candidates["source"].value_counts().to_dict(),
        "counts_by_candidate_type": candidates["candidate_type"].value_counts().to_dict(),
        "counts_by_label": candidates["roi_label"].value_counts().sort_index().to_dict(),
    }
    with (output_dir / "roi_candidate_mining_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("\nROI candidate mining summary:")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()

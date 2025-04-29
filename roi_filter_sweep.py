import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import find_training_image_directory, prepare_data, resolve_data_dir, seed_worker
from roi_utils import ROICandidateDataset, metrics_from_roi_predictions, roi_eval_transform
from testing_FRCNN import choose_device, parse_thresholds, save_confusion_matrix, set_seed
from threshold_sweep import parse_score_thresholds
from train_classifier import autocast_context, build_classifier


def split_candidates(candidates, split_name):
    aliases = {
        "validation": {"validation", "valid", "val"},
        "test": {"test"},
    }
    return candidates[candidates["split"].isin(aliases[split_name])].copy()


def make_loader(dataset, batch_size, num_workers, device):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker
    return DataLoader(dataset, **loader_kwargs)


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


def parse_optional_score_thresholds(value):
    if value is None or not value.strip():
        return [None]
    thresholds = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        thresholds.append(None if item in {"none", "off"} else float(item))
    return thresholds or [None]


def parse_args():
    parser = argparse.ArgumentParser(description="Apply an ROI false-positive filter to Faster R-CNN candidate boxes.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--roi-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--detector-score-thresholds", default="0.35,0.40,0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.65,0.70")
    parser.add_argument("--roi-thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument(
        "--strong-detector-thresholds",
        default="",
        help="Optional comma-separated detector scores that bypass the ROI filter for very confident boxes.",
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
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "roi_filter_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = Path(args.candidate_csv).expanduser().resolve()
    roi_checkpoint_path = Path(args.roi_checkpoint).expanduser().resolve()
    detector_checkpoint_path = Path(args.detector_checkpoint).expanduser().resolve()
    image_directory = find_training_image_directory(data_dir)

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    detector_score_thresholds = parse_score_thresholds(args.detector_score_thresholds)
    roi_thresholds = parse_score_thresholds(args.roi_thresholds)
    strong_detector_thresholds = parse_optional_score_thresholds(args.strong_detector_thresholds)
    detector_iou_thresholds = parse_thresholds(args.detector_iou_thresholds)

    roi_checkpoint = torch.load(roi_checkpoint_path, map_location=device)
    architecture = roi_checkpoint.get("architecture", "swin_t")
    image_size = int(roi_checkpoint.get("image_size", 224))
    padding_ratio = float(roi_checkpoint.get("padding_ratio", 0.25))

    print(f"Using data directory: {data_dir}")
    print(f"Using candidate CSV: {candidate_csv}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using ROI checkpoint: {roi_checkpoint_path}")
    print(f"Using detector checkpoint metadata: {detector_checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"ROI architecture: {architecture}, image size: {image_size}, padding: {padding_ratio:g}")

    classifier = build_classifier(
        architecture=architecture,
        pretrained=False,
        image_size=image_size,
    ).to(device)
    classifier.load_state_dict(roi_checkpoint["model_state_dict"])

    _, valid_dataset, test_dataset = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
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
        transform=roi_eval_transform(image_size),
        padding_ratio=padding_ratio,
    )
    test_roi_dataset = ROICandidateDataset(
        test_candidates,
        image_directory,
        transform=roi_eval_transform(image_size),
        padding_ratio=padding_ratio,
    )
    valid_loader = make_loader(valid_roi_dataset, args.batch_size, args.num_workers, device)
    test_loader = make_loader(test_roi_dataset, args.batch_size, args.num_workers, device)

    valid_probs = collect_roi_probabilities(
        valid_loader,
        classifier,
        device,
        use_amp,
        desc="validation ROI classifier",
    )
    test_probs = collect_roi_probabilities(
        test_loader,
        classifier,
        device,
        use_amp,
        desc="test ROI classifier",
    )
    valid_candidates = add_roi_probabilities(valid_candidates, valid_probs)
    test_candidates = add_roi_probabilities(test_candidates, test_probs)
    valid_candidates.to_csv(output_dir / "roi_filter_validation_candidate_scores.csv", index=False)
    test_candidates.to_csv(output_dir / "roi_filter_test_candidate_scores.csv", index=False)

    rows = []
    best = None
    rules = list(product(detector_score_thresholds, roi_thresholds, strong_detector_thresholds))
    print(f"Evaluating {len(rules)} validation ROI filter rules")
    for detector_score_threshold, roi_threshold, strong_detector_threshold in rules:
        metrics, _ = metrics_from_roi_predictions(
            valid_dataset,
            valid_candidates,
            detector_iou_thresholds,
            detector_score_threshold=detector_score_threshold,
            roi_threshold=roi_threshold,
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
            strong_detector_threshold=strong_detector_threshold,
        )
        row = {
            "roi_checkpoint": str(roi_checkpoint_path),
            "roi_checkpoint_epoch": roi_checkpoint.get("epoch"),
            "detector_checkpoint": str(detector_checkpoint_path),
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
                f"det={detector_score_threshold}, roi={roi_threshold}, "
                f"strong_det={strong_detector_threshold}, "
                f"F1={row['image_f1']:.5f}, precision={row['image_precision']:.5f}, "
                f"recall={row['image_recall']:.5f}, detector_precision={row['detector_precision']:.5f}"
            )

    validation_table = pd.DataFrame(rows).sort_values(
        ["image_f1", "image_precision", "detector_precision", "mean_iou"],
        ascending=[False, False, False, False],
    )
    validation_table.to_csv(output_dir / "roi_filter_sweep_validation.csv", index=False)

    test_metrics, test_predictions = metrics_from_roi_predictions(
        test_dataset,
        test_candidates,
        detector_iou_thresholds,
        detector_score_threshold=float(best["detector_score_threshold"]),
        roi_threshold=float(best["roi_threshold"]),
        max_detections=args.max_detections,
        metric_iou_threshold=args.metric_iou_threshold,
        strong_detector_threshold=(
            None
            if pd.isna(best.get("strong_detector_threshold"))
            else float(best["strong_detector_threshold"])
        ),
    )
    selected = {
        "selected_from": "validation_roi_filter_sweep",
        "roi_checkpoint": str(roi_checkpoint_path),
        "roi_checkpoint_epoch": roi_checkpoint.get("epoch"),
        "detector_checkpoint": str(detector_checkpoint_path),
        **test_metrics,
    }
    pd.DataFrame([selected]).to_csv(output_dir / "roi_filter_sweep_test_metrics.csv", index=False)
    with (output_dir / "roi_filter_sweep_test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(selected, metrics_file, indent=2)
    test_predictions.to_csv(output_dir / "roi_filter_sweep_test_predictions.csv", index=False)
    save_confusion_matrix(
        output_dir / "roi_filter_sweep_test_confusion_matrix.png",
        selected,
        (
            "ROI Filter Test, "
            f"det={selected['detector_score_threshold']:g}, roi={selected['roi_threshold']:g}, "
            f"strong_det={selected.get('strong_detector_threshold')}"
        ),
    )

    print("\nBest validation ROI filter rule:")
    print(pd.Series(best).to_string())
    print("\nHeld-out test metrics at selected ROI filter rule:")
    print(pd.Series(selected).to_string())


if __name__ == "__main__":
    main()

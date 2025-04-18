import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import find_training_image_directory, resolve_data_dir, seed_worker
from roi_utils import ROICandidateDataset, roi_eval_transform, roi_train_transform
from testing_FRCNN import choose_device, parse_thresholds, save_confusion_matrix, set_seed
from train_classifier import build_classifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def autocast_context(use_amp):
    if not use_amp:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except AttributeError:
        return torch.cuda.amp.autocast()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probabilities = torch.sigmoid(inputs)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce_loss
        return focal_loss.mean()


def roi_metrics_from_probabilities(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = (probabilities >= threshold).astype(np.int64)

    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (true_positive + true_negative) / max(len(labels), 1)
    return {
        "roi_accuracy": accuracy,
        "roi_precision": precision,
        "roi_recall": recall,
        "roi_f1": f1,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "num_candidates": int(len(labels)),
        "roi_threshold": float(threshold),
    }


def best_threshold_metrics(labels, probabilities, thresholds):
    rows = [roi_metrics_from_probabilities(labels, probabilities, threshold) for threshold in thresholds]
    table = pd.DataFrame(rows).sort_values(
        ["roi_f1", "roi_precision", "roi_recall"],
        ascending=[False, False, False],
    )
    return table.iloc[0].to_dict(), table


def limit_training_rows(candidates, max_positive, max_negative, seed):
    if max_positive <= 0 and max_negative <= 0:
        return candidates

    parts = []
    positives = candidates[candidates["roi_label"] == 1]
    negatives = candidates[candidates["roi_label"] == 0]
    if max_positive > 0 and len(positives) > max_positive:
        positives = positives.sample(n=max_positive, random_state=seed)
    if max_negative > 0 and len(negatives) > max_negative:
        negatives = negatives.sample(n=max_negative, random_state=seed)
    parts.extend([positives, negatives])
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_loader(dataset, batch_size, num_workers, device, shuffle=False, sampler=None):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker
    return DataLoader(dataset, **loader_kwargs)


def make_balanced_sampler(labels, seed, detector_false_positive_flags=None, detector_negative_weight=1.0):
    labels = np.asarray(labels, dtype=np.int64)
    if detector_false_positive_flags is None:
        detector_false_positive_flags = np.zeros_like(labels, dtype=np.int64)
    detector_false_positive_flags = np.asarray(detector_false_positive_flags, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    weights = []
    for label, is_detector_false_positive in zip(labels, detector_false_positive_flags):
        weight = 1.0 / max(class_counts[label], 1)
        if label == 0 and is_detector_false_positive:
            weight *= detector_negative_weight
        weights.append(weight)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=generator)


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    total_examples = 0
    non_blocking = device.type == "cuda"

    for crops, labels, _, _ in tqdm(dataloader, total=len(dataloader), desc="Training ROI", leave=False):
        crops = crops.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = model(crops).flatten()
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu()) * crops.shape[0]
        total_examples += crops.shape[0]

    return total_loss / max(total_examples, 1)


def evaluate(model, dataloader, criterion, device, use_amp, desc="Evaluating ROI"):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels_all = []
    probabilities = []
    rows = []
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for crops, labels, candidate_ids, patient_ids in tqdm(dataloader, total=len(dataloader), desc=desc, leave=False):
            crops = crops.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            with autocast_context(use_amp):
                logits = model(crops).flatten()
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            total_loss += float(loss.detach().cpu()) * crops.shape[0]
            total_examples += crops.shape[0]
            probabilities.extend(probs.tolist())
            labels_all.extend(labels_np.tolist())
            for candidate_id, patient_id, label, probability in zip(candidate_ids, patient_ids, labels_np, probs):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "patientId": patient_id,
                        "truth": int(label),
                        "probability": float(probability),
                    }
                )

    return total_loss / max(total_examples, 1), labels_all, probabilities, pd.DataFrame(rows)


def save_plots(output_dir, train_losses, validation_f1):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.legend()
    plt.savefig(output_dir / "roi_classifier_loss.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(validation_f1, label="Validation ROI F1")
    plt.legend()
    plt.savefig(output_dir / "roi_classifier_f1.png", bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a crop-level ROI false-positive filter.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--architecture",
        choices=[
            "resnet18",
            "resnet50",
            "efficientnet_b0",
            "densenet121",
            "swin_t",
            "vit_b_16",
            "convnext_tiny",
            "convnext_small",
        ],
        default="swin_t",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--padding-ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--augmentation", choices=["basic", "cxr"], default="basic")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--max-train-positives", type=int, default=0)
    parser.add_argument("--max-train-negatives", type=int, default=0)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--detector-negative-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    candidate_csv = Path(args.candidate_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "roi_classifier_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_directory = find_training_image_directory(data_dir)

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    thresholds = parse_thresholds(args.thresholds)

    candidates = pd.read_csv(candidate_csv)
    candidates = candidates[candidates["roi_label"].isin([0, 1])].copy()
    train_candidates = candidates[candidates["split"] == "train"].copy()
    valid_candidates = candidates[candidates["split"].isin(["validation", "valid", "val"])].copy()
    train_candidates = limit_training_rows(
        train_candidates,
        max_positive=args.max_train_positives,
        max_negative=args.max_train_negatives,
        seed=args.seed,
    )

    if train_candidates.empty:
        raise ValueError("No train ROI candidates with labels 0/1 were found.")
    if valid_candidates.empty:
        raise ValueError("No validation ROI candidates with labels 0/1 were found.")

    print(f"Using data directory: {data_dir}")
    print(f"Using candidate CSV: {candidate_csv}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"Architecture: {args.architecture}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Loss: {args.loss}")
    print(f"Detector false-positive negative weight: {args.detector_negative_weight:g}")
    print(f"Train ROI candidates: {len(train_candidates)}")
    print(f"Validation ROI candidates: {len(valid_candidates)}")
    print(f"Train label counts: {train_candidates['roi_label'].value_counts().to_dict()}")

    train_dataset = ROICandidateDataset(
        train_candidates,
        image_directory,
        transform=roi_train_transform(args.image_size, augmentation=args.augmentation),
        padding_ratio=args.padding_ratio,
    )
    valid_dataset = ROICandidateDataset(
        valid_candidates,
        image_directory,
        transform=roi_eval_transform(args.image_size),
        padding_ratio=args.padding_ratio,
    )

    sampler = None
    if args.balanced_sampler or args.detector_negative_weight != 1.0:
        detector_false_positive_flags = train_candidates.get(
            "is_detector_false_positive",
            pd.Series(0, index=train_candidates.index),
        ).to_numpy(dtype=np.int64)
        sampler = make_balanced_sampler(
            train_candidates["roi_label"].to_numpy(dtype=np.int64),
            args.seed,
            detector_false_positive_flags=detector_false_positive_flags,
            detector_negative_weight=args.detector_negative_weight,
        )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        device,
        shuffle=not args.balanced_sampler,
        sampler=sampler,
    )
    valid_loader = make_loader(valid_dataset, args.batch_size, args.num_workers, device)

    num_positive = int((train_candidates["roi_label"] == 1).sum())
    num_negative = int((train_candidates["roi_label"] == 0).sum())
    if args.pos_weight is not None:
        pos_weight_value = args.pos_weight
    elif args.balanced_sampler:
        pos_weight_value = 1.0
    else:
        pos_weight_value = num_negative / max(num_positive, 1)

    classifier = build_classifier(
        args.architecture,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
    ).to(device)
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    train_losses = []
    validation_f1 = []
    best = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(classifier, train_loader, optimizer, criterion, device, scaler, use_amp)
        valid_loss, valid_labels, valid_probs, valid_predictions = evaluate(
            classifier,
            valid_loader,
            criterion,
            device,
            use_amp,
            desc="Validating ROI",
        )
        valid_best, valid_threshold_table = best_threshold_metrics(valid_labels, valid_probs, thresholds)
        valid_predictions["prediction"] = (
            valid_predictions["probability"] >= float(valid_best["roi_threshold"])
        ).astype(int)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            **valid_best,
        }
        history.append(epoch_metrics)
        train_losses.append(train_loss)
        validation_f1.append(valid_best["roi_f1"])
        pd.DataFrame(history).to_csv(output_dir / "roi_classifier_validation_metrics.csv", index=False)
        valid_threshold_table.to_csv(output_dir / f"roi_validation_thresholds_epoch_{epoch}.csv", index=False)
        valid_predictions.to_csv(output_dir / f"roi_validation_predictions_epoch_{epoch}.csv", index=False)

        is_best = best is None or (
            epoch_metrics["roi_f1"],
            epoch_metrics["roi_precision"],
            epoch_metrics["roi_recall"],
        ) > (
            best["roi_f1"],
            best["roi_precision"],
            best["roi_recall"],
        )
        if is_best:
            best = epoch_metrics
            best_state = {
                "epoch": epoch,
                "architecture": args.architecture,
                "image_size": args.image_size,
                "padding_ratio": args.padding_ratio,
                "augmentation": args.augmentation,
                "loss": args.loss,
                "detector_negative_weight": args.detector_negative_weight,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_metrics": best,
            }
            torch.save(best_state, output_dir / "roi_classifier_best.pt")

        torch.save(
            {
                "epoch": epoch,
                "architecture": args.architecture,
                "image_size": args.image_size,
                "padding_ratio": args.padding_ratio,
                "augmentation": args.augmentation,
                "loss": args.loss,
                "detector_negative_weight": args.detector_negative_weight,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_metrics": epoch_metrics,
            },
            output_dir / f"roi_classifier_epoch_{epoch}.pt",
        )
        print(
            f"Epoch #{epoch}, Loss: {train_loss:.5f}, Val Loss: {valid_loss:.5f}, "
            f"ROI F1: {valid_best['roi_f1']:.5f}, Precision: {valid_best['roi_precision']:.5f}, "
            f"Recall: {valid_best['roi_recall']:.5f}, Threshold: {valid_best['roi_threshold']:.2f}"
        )

    save_plots(output_dir, train_losses, validation_f1)
    if best_state is None:
        best_state = torch.load(output_dir / "roi_classifier_best.pt", map_location=device)

    with (output_dir / "roi_classifier_best_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(best, metrics_file, indent=2)
    save_confusion_matrix(
        output_dir / "roi_classifier_validation_confusion_matrix.png",
        best,
        f"ROI Classifier Validation, threshold={best['roi_threshold']:g}",
    )

    print("\nBest validation ROI classifier setting:")
    print(pd.Series(best).to_string())


if __name__ == "__main__":
    main()

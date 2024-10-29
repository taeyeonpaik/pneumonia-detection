import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_data_loader, resolve_data_dir
from model_FRCNN import model as build_model
from pipeline_FRCNN import autocast_context, train


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        return 0 if self.iterations == 0 else self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.array([], dtype=np.float32)

    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def calculate_image_precision(truths, predictions, thresholds):
    truths = np.asarray(truths, dtype=np.float32).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 4)
    image_precision = 0.0

    for threshold in thresholds:
        matched_truths = set()
        true_positive = 0
        false_positive = 0

        for prediction in predictions:
            if len(truths) == 0:
                false_positive += 1
                continue

            ious = box_iou(prediction, truths)
            if matched_truths:
                ious[list(matched_truths)] = -1

            best_match_index = int(np.argmax(ious))
            if ious[best_match_index] >= threshold:
                true_positive += 1
                matched_truths.add(best_match_index)
            else:
                false_positive += 1

        false_negative = len(truths) - len(matched_truths)
        denominator = true_positive + false_positive + false_negative
        precision_at_threshold = 1.0 if denominator == 0 else true_positive / denominator
        image_precision += precision_at_threshold / len(thresholds)

    return image_precision


def matched_prediction_ious(truths, predictions, iou_threshold):
    truths = np.asarray(truths, dtype=np.float32).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 4)
    matched_truths = set()
    matched_ious = []

    for prediction in predictions:
        if len(truths) == 0:
            break
        ious = box_iou(prediction, truths)
        if matched_truths:
            ious[list(matched_truths)] = -1

        best_match_index = int(np.argmax(ious))
        best_iou = float(ious[best_match_index])
        if best_iou >= iou_threshold:
            matched_truths.add(best_match_index)
            matched_ious.append(best_iou)

    return matched_ious


def format_prediction_string(boxes, scores):
    prediction_strings = []
    for score, box in zip(scores, boxes):
        prediction_strings.append(
            "{0:.4f} {1} {2} {3} {4}".format(
                float(score),
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
            )
        )
    return " ".join(prediction_strings)


def filter_predictions(boxes, scores, score_threshold=0.05, max_detections=100):
    if score_threshold is not None:
        keep = scores >= score_threshold
        boxes = boxes[keep]
        scores = scores[keep]

    sorted_indexes = np.argsort(scores)[::-1]
    if max_detections is not None and max_detections > 0:
        sorted_indexes = sorted_indexes[:max_detections]
    return boxes[sorted_indexes], scores[sorted_indexes]


def evaluate_labeled_split(
    dataloader,
    model,
    device,
    thresholds,
    score_threshold=0.05,
    max_detections=100,
    metric_iou_threshold=0.5,
    use_amp=False,
    desc="Evaluating",
):
    valid_image_precision = []
    matched_ious = []
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    image_rows = []
    model.eval()
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for images, targets, image_ids in tqdm(dataloader, total=len(dataloader), desc=desc, leave=False):
            images = [image.to(device, non_blocking=non_blocking) for image in images]
            targets = [
                {key: value.to(device, non_blocking=non_blocking) for key, value in target.items()}
                for target in targets
            ]
            with autocast_context(use_amp):
                outputs = model(images)

            for index, _ in enumerate(images):
                boxes = outputs[index]["boxes"].detach().cpu().numpy()
                scores = outputs[index]["scores"].detach().cpu().numpy()
                truth_boxes = targets[index]["boxes"].detach().cpu().numpy()
                predictions_sorted, filtered_scores = filter_predictions(
                    boxes,
                    scores,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                )
                image_precision = calculate_image_precision(truth_boxes, predictions_sorted, thresholds)
                valid_image_precision.append(image_precision)
                matched_ious.extend(
                    matched_prediction_ious(truth_boxes, predictions_sorted, metric_iou_threshold)
                )

                true_label = int(len(truth_boxes) > 0)
                predicted_label = int(len(predictions_sorted) > 0)
                max_score = float(filtered_scores[0]) if len(filtered_scores) else 0.0

                if true_label == 1 and predicted_label == 1:
                    true_positive += 1
                elif true_label == 0 and predicted_label == 1:
                    false_positive += 1
                elif true_label == 0 and predicted_label == 0:
                    true_negative += 1
                else:
                    false_negative += 1

                image_rows.append(
                    {
                        "patientId": image_ids[index],
                        "truth": true_label,
                        "prediction": predicted_label,
                        "max_score": max_score,
                        "num_truth_boxes": len(truth_boxes),
                        "num_predicted_boxes": len(predictions_sorted),
                    }
                )

    total = true_positive + true_negative + false_positive + false_negative
    classification_precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * classification_precision * recall / max(classification_precision + recall, 1e-12)
    accuracy = (true_positive + true_negative) / max(total, 1)

    metrics = {
        "detector_precision": float(np.mean(valid_image_precision)) if valid_image_precision else 0.0,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "image_accuracy": accuracy,
        "image_precision": classification_precision,
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
    return metrics, pd.DataFrame(image_rows)


def find_test_image_directory(data_dir):
    candidates = [
        data_dir / "test_images_png",
        data_dir / "stage_2_test_images_png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing test image directory. Checked: {', '.join(str(path) for path in candidates)}")


def annotate(model, device, data_dir, output_dir, threshold, max_detections=100, use_amp=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    test_image_directory = find_test_image_directory(data_dir)
    bbox_output_directory = output_dir / "test_images_bbox"
    bbox_output_directory.mkdir(parents=True, exist_ok=True)

    test_images = sorted(path for path in test_image_directory.iterdir() if path.suffix.lower() == ".png")
    model.to(device).eval()
    results = []
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for image_path in tqdm(test_images, total=len(test_images), desc="Annotating"):
            original_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if original_image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")

            image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.as_tensor(image, dtype=torch.float32).to(device, non_blocking=non_blocking)

            with autocast_context(use_amp):
                output = model([image])[0]
            boxes_xyxy = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            boxes_xyxy, scores = filter_predictions(
                boxes_xyxy,
                scores,
                score_threshold=threshold,
                max_detections=max_detections,
            )

            boxes_xywh = boxes_xyxy.copy()
            if len(boxes_xywh) > 0:
                boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
                boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]

            for box in boxes_xyxy:
                cv2.rectangle(
                    original_image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 0, 255),
                    3,
                )

            cv2.imwrite(str(bbox_output_directory / image_path.name), original_image)
            results.append(
                {
                    "patientId": image_path.stem,
                    "PredictionString": format_prediction_string(boxes_xywh, scores),
                }
            )

    submission_dataframe = pd.DataFrame(results, columns=["patientId", "PredictionString"])
    submission_path = output_dir / "submission.csv"
    submission_dataframe.to_csv(submission_path, index=False)
    return submission_path


def save_training_plots(output_dir, train_loss, precision_history, f1_history=None, iou_history=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.legend()
    plt.savefig(output_dir / "loss.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(precision_history, label="Validation Precision")
    plt.legend()
    plt.savefig(output_dir / "precision.png", bbox_inches="tight")
    plt.close()

    if f1_history:
        plt.figure()
        plt.plot(f1_history, label="Validation F1")
        plt.legend()
        plt.savefig(output_dir / "f1.png", bbox_inches="tight")
        plt.close()

    if iou_history:
        plt.figure()
        plt.plot(iou_history, label="Validation Mean IoU")
        plt.legend()
        plt.savefig(output_dir / "iou.png", bbox_inches="tight")
        plt.close()


def save_confusion_matrix(output_path, metrics, title):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_metrics_csv(output_path, metrics_history):
    pd.DataFrame(metrics_history).to_csv(output_path, index=False)


def save_checkpoint(
    output_dir,
    epoch,
    model,
    optimizer,
    scheduler,
    train_loss,
    precision_history,
    metrics_history,
    is_best=False,
    is_best_f1=False,
    save_epoch=True,
    save_latest=True,
    save_best_precision=True,
    save_precision_copy=True,
    weights_only=False,
    checkpoint_metadata=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "precision_history": precision_history,
        "metrics_history": metrics_history,
    }
    if not weights_only:
        checkpoint.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            }
        )
    checkpoint.update(checkpoint_metadata or {})
    epoch_path = output_dir / f"frcnn_epoch_{epoch}.pt"
    latest_path = output_dir / "frcnn_latest.pt"
    if save_epoch:
        torch.save(checkpoint, epoch_path)
    if save_latest:
        torch.save(checkpoint, latest_path)
    if is_best and save_best_precision:
        torch.save(checkpoint, output_dir / "frcnn_best.pt")
        if save_precision_copy:
            torch.save(checkpoint, output_dir / "frcnn_best_detector_precision.pt")
    if is_best_f1:
        torch.save(checkpoint, output_dir / "frcnn_best_f1.pt")


def parse_thresholds(value):
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def choose_device(device_name):
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_patient_ids(path):
    if path is None:
        return []

    id_path = Path(path).expanduser()
    if not id_path.exists():
        raise FileNotFoundError(f"Missing hard-negative file: {id_path}")

    if id_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(id_path)
        if "patientId" not in dataframe.columns:
            raise ValueError(f"Expected a patientId column in {id_path}")
        return dataframe["patientId"].dropna().astype(str).tolist()

    return [line.strip() for line in id_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        return {"epoch": None}
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and run RSNA Faster R-CNN.")
    parser.add_argument("--data-dir", default=None, help="Directory containing labels and image folders.")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints, plots, boxes, and submission.csv.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr-step-size", type=int, default=4)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--min-size", type=int, default=1024)
    parser.add_argument(
        "--box-fg-iou-threshold",
        type=float,
        default=0.5,
        help="Minimum proposal IoU treated as foreground by the Faster R-CNN RoI head.",
    )
    parser.add_argument(
        "--box-bg-iou-threshold",
        type=float,
        default=None,
        help="Maximum proposal IoU treated as background; defaults to the foreground threshold.",
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--augmentation", choices=["basic", "cxr"], default="basic")
    parser.add_argument("--thresholds", default="0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75")
    parser.add_argument("--val-score-threshold", type=float, default=0.05)
    parser.add_argument("--test-score-threshold", type=float, default=0.05)
    parser.add_argument("--metric-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Dataset split seed; defaults to --seed for backward compatibility.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Model, augmentation, and sampling seed; defaults to --seed.",
    )
    parser.add_argument("--balanced-sampler", action="store_true", help="Oversample positive images during training.")
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--hard-negative-file", default=None, help="CSV/text file of hard-negative patient IDs to oversample.")
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load COCO pretrained weights.")
    parser.add_argument("--resume-checkpoint", default=None, help="Load model weights from an existing checkpoint.")
    parser.add_argument("--no-save-epoch-checkpoints", action="store_true", help="Do not save frcnn_epoch_N.pt after each epoch.")
    parser.add_argument("--no-save-latest-checkpoint", action="store_true", help="Do not save frcnn_latest.pt after each epoch.")
    parser.add_argument("--no-save-best-precision", action="store_true", help="Do not save frcnn_best.pt based on detector precision.")
    parser.add_argument(
        "--no-save-precision-copy",
        action="store_true",
        help="Do not save the duplicate frcnn_best_detector_precision.pt checkpoint.",
    )
    parser.add_argument(
        "--weights-only-checkpoints",
        action="store_true",
        help="Save model weights and metrics only, omitting optimizer and scheduler state.",
    )
    parser.add_argument("--annotate-unlabeled", action="store_true", help="Also annotate official unlabeled test images.")
    parser.add_argument("--skip-annotate", action="store_true", help="Deprecated alias; annotation is off by default.")
    return parser.parse_args()


def main():
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed
    set_seed(model_seed)
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    thresholds = parse_thresholds(args.thresholds)
    print(f"Using data directory: {data_dir}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"Training augmentation: {args.augmentation}")
    print(f"Split seed: {split_seed}; model seed: {model_seed}")

    loss_history = Averager()
    box_bg_iou_threshold = (
        args.box_fg_iou_threshold
        if args.box_bg_iou_threshold is None
        else args.box_bg_iou_threshold
    )
    print(
        "RoI box IoU thresholds: "
        f"foreground={args.box_fg_iou_threshold:g}, background={box_bg_iou_threshold:g}"
    )
    detector = build_model(
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        box_fg_iou_thresh=args.box_fg_iou_threshold,
        box_bg_iou_thresh=box_bg_iou_threshold,
    ).to(device)
    if args.resume_checkpoint:
        checkpoint = load_model_checkpoint(detector, args.resume_checkpoint, device)
        print(f"Resumed model weights from {args.resume_checkpoint} (epoch={checkpoint.get('epoch')})")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    params = [parameter for parameter in detector.parameters() if parameter.requires_grad]
    hard_negative_ids = load_patient_ids(args.hard_negative_file)
    if hard_negative_ids:
        print(f"Using {len(hard_negative_ids)} hard-negative images with weight {args.hard_negative_weight:g}")
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(
        args.batch_size,
        data_dir=data_dir,
        num_workers=args.num_workers,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=split_seed,
        sampling_seed=model_seed,
        balanced_sampler=args.balanced_sampler,
        positive_weight=args.positive_weight,
        hard_negative_ids=hard_negative_ids,
        hard_negative_weight=args.hard_negative_weight,
        train_augmentation=args.augmentation,
    )
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    train_loss = []
    precision_history = []
    f1_history = []
    iou_history = []
    metrics_history = []
    best_precision = -1.0
    best_f1 = -1.0
    for epoch in range(args.epochs):
        train_loss_history, end, start = train(
            train_data_loader,
            detector,
            optimizer,
            device,
            loss_history,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_metrics, val_predictions = evaluate_labeled_split(
            valid_data_loader,
            detector,
            device,
            thresholds,
            score_threshold=args.val_score_threshold,
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
            use_amp=use_amp,
            desc="Validating",
        )
        precision = val_metrics["detector_precision"]
        train_loss.append(train_loss_history.value)
        precision_history.append(precision)
        f1_history.append(val_metrics["image_f1"])
        iou_history.append(val_metrics["mean_iou"])

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss_history.value,
            **val_metrics,
        }
        metrics_history.append(epoch_metrics)
        save_metrics_csv(output_dir / "validation_metrics.csv", metrics_history)
        save_confusion_matrix(
            output_dir / f"validation_confusion_epoch_{epoch + 1}.png",
            val_metrics,
            f"Validation Confusion Matrix, Epoch {epoch + 1}",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        val_predictions.to_csv(output_dir / f"validation_predictions_epoch_{epoch + 1}.csv", index=False)

        learning_rate_scheduler.step()
        is_best = precision > best_precision
        if is_best:
            best_precision = precision
        is_best_f1 = val_metrics["image_f1"] > best_f1
        if is_best_f1:
            best_f1 = val_metrics["image_f1"]
        save_checkpoint(
            output_dir,
            epoch + 1,
            detector,
            optimizer,
            learning_rate_scheduler,
            train_loss,
            precision_history,
            metrics_history,
            is_best=is_best,
            is_best_f1=is_best_f1,
            save_epoch=not args.no_save_epoch_checkpoints,
            save_latest=not args.no_save_latest_checkpoint,
            save_best_precision=not args.no_save_best_precision,
            save_precision_copy=not args.no_save_precision_copy,
            weights_only=args.weights_only_checkpoints,
            checkpoint_metadata={
                "min_size": args.min_size,
                "box_fg_iou_threshold": args.box_fg_iou_threshold,
                "box_bg_iou_threshold": box_bg_iou_threshold,
                "split_seed": split_seed,
                "model_seed": model_seed,
            },
        )
        print(
            f"Epoch #{epoch + 1}, Loss: {train_loss_history.value:.5f}, "
            f"Detector Precision: {precision:.5f}, F1: {val_metrics['image_f1']:.5f}, "
            f"Mean IoU: {val_metrics['mean_iou']:.5f}, Time: {(end - start) / 60:.3f} Minutes"
        )

    save_training_plots(output_dir, train_loss, precision_history, f1_history, iou_history)

    best_checkpoint_path = next(
        (
            path
            for path in (
                output_dir / "frcnn_best_f1.pt",
                output_dir / "frcnn_best.pt",
                output_dir / "frcnn_latest.pt",
            )
            if path.exists()
        ),
        None,
    )
    if best_checkpoint_path is not None:
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        detector.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics, test_predictions = evaluate_labeled_split(
        test_data_loader,
        detector,
        device,
        thresholds,
        score_threshold=args.test_score_threshold,
        max_detections=args.max_detections,
        metric_iou_threshold=args.metric_iou_threshold,
        use_amp=use_amp,
        desc="Testing",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(test_metrics, metrics_file, indent=2)
    save_confusion_matrix(output_dir / "test_confusion_matrix.png", test_metrics, "Held-Out Test Confusion Matrix")
    print(
        "Held-out test metrics: "
        f"Detector Precision={test_metrics['detector_precision']:.5f}, "
        f"F1={test_metrics['image_f1']:.5f}, "
        f"Mean IoU={test_metrics['mean_iou']:.5f}, "
        f"TP={test_metrics['tp']}, FP={test_metrics['fp']}, "
        f"TN={test_metrics['tn']}, FN={test_metrics['fn']}"
    )

    if args.annotate_unlabeled and not args.skip_annotate:
        submission_path = annotate(
            detector,
            device,
            data_dir,
            output_dir,
            threshold=args.threshold,
            max_detections=args.max_detections,
            use_amp=use_amp,
        )
        print(f"Saved submission: {submission_path}")


if __name__ == "__main__":
    main()

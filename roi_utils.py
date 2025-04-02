from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from testing_FRCNN import box_iou, calculate_image_precision, matched_prediction_ious


def roi_train_transform(image_size, augmentation="basic"):
    transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.4),
    ]
    if augmentation == "cxr":
        transforms.extend(
            [
                A.CLAHE(clip_limit=4.0, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.GaussNoise(p=0.25),
            ]
        )
    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def roi_eval_transform(image_size):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def crop_with_padding(image, box, padding_ratio=0.25, square=True, min_size=8):
    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = [float(value) for value in box]
    x_min = np.clip(x_min, 0, image_width - 1)
    y_min = np.clip(y_min, 0, image_height - 1)
    x_max = np.clip(x_max, x_min + 1, image_width)
    y_max = np.clip(y_max, y_min + 1, image_height)

    box_width = max(x_max - x_min, 1.0)
    box_height = max(y_max - y_min, 1.0)
    if square:
        side = max(box_width, box_height, float(min_size)) * (1.0 + 2.0 * padding_ratio)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        crop_x_min = center_x - side / 2.0
        crop_y_min = center_y - side / 2.0
        crop_x_max = center_x + side / 2.0
        crop_y_max = center_y + side / 2.0
    else:
        pad_x = max(box_width * padding_ratio, min_size / 2.0)
        pad_y = max(box_height * padding_ratio, min_size / 2.0)
        crop_x_min = x_min - pad_x
        crop_y_min = y_min - pad_y
        crop_x_max = x_max + pad_x
        crop_y_max = y_max + pad_y

    crop_x_min = int(np.floor(np.clip(crop_x_min, 0, image_width - 1)))
    crop_y_min = int(np.floor(np.clip(crop_y_min, 0, image_height - 1)))
    crop_x_max = int(np.ceil(np.clip(crop_x_max, crop_x_min + 1, image_width)))
    crop_y_max = int(np.ceil(np.clip(crop_y_max, crop_y_min + 1, image_height)))

    crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    if crop.size == 0:
        return image
    return crop


class ROICandidateDataset(Dataset):
    def __init__(self, candidates, image_directory, transform=None, padding_ratio=0.25):
        self.candidates = candidates.reset_index(drop=True).copy()
        if "candidate_id" not in self.candidates.columns:
            self.candidates["candidate_id"] = [f"candidate_{index}" for index in range(len(self.candidates))]
        self.image_directory = Path(image_directory)
        self.transform = transform
        self.padding_ratio = padding_ratio

    def __getitem__(self, index):
        row = self.candidates.iloc[index]
        image_path = self.image_directory / f"{row['patientId']}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box = [row["x1"], row["y1"], row["x2"], row["y2"]]
        crop = crop_with_padding(image, box, padding_ratio=self.padding_ratio)
        if self.transform is not None:
            crop = self.transform(image=crop)["image"]

        label = float(row["roi_label"]) if "roi_label" in row and not pd.isna(row["roi_label"]) else 0.0
        return crop, torch.tensor(label, dtype=torch.float32), str(row["candidate_id"]), str(row["patientId"])

    def __len__(self):
        return len(self.candidates)


def truth_boxes_for_image(detection_dataset, image_id):
    records = detection_dataset.records_by_image_id[image_id]
    boxes = []
    for _, row in records.iterrows():
        if int(row["Target"]) != 1:
            continue
        x_min = float(row["x"])
        y_min = float(row["y"])
        width = float(row["width"])
        height = float(row["height"])
        boxes.append([x_min, y_min, x_min + width, y_min + height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def roi_label_from_iou(max_iou, positive_iou_threshold=0.5, negative_iou_threshold=0.3):
    if max_iou >= positive_iou_threshold:
        return 1
    if max_iou < negative_iou_threshold:
        return 0
    return -1


def metrics_from_roi_predictions(
    detection_dataset,
    candidate_table,
    detector_iou_thresholds,
    detector_score_threshold,
    roi_threshold,
    max_detections,
    metric_iou_threshold,
    strong_detector_threshold=None,
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
        if candidates is None:
            kept = candidates
        else:
            keep_mask = (
                (candidates["detector_score"] >= detector_score_threshold)
                & (candidates["roi_probability"] >= roi_threshold)
            )
            if strong_detector_threshold is not None:
                keep_mask = keep_mask | (candidates["detector_score"] >= strong_detector_threshold)
            kept = candidates[keep_mask].copy()
            if len(kept):
                kept["ranking_score"] = kept["detector_score"] * kept["roi_probability"]
                kept = kept.sort_values("ranking_score", ascending=False)
                if max_detections is not None and max_detections > 0:
                    kept = kept.head(max_detections)

        if kept is None or len(kept) == 0:
            kept_boxes = np.zeros((0, 4), dtype=np.float32)
            max_detector_score = 0.0
            max_roi_probability = 0.0
        else:
            kept_boxes = kept[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32).reshape(-1, 4)
            max_detector_score = float(kept["detector_score"].max())
            max_roi_probability = float(kept["roi_probability"].max())

        true_label = int(len(truth_boxes) > 0)
        predicted_label = int(len(kept_boxes) > 0)

        image_precisions.append(calculate_image_precision(truth_boxes, kept_boxes, detector_iou_thresholds))
        matched_ious.extend(matched_prediction_ious(truth_boxes, kept_boxes, metric_iou_threshold))

        if true_label == 1 and predicted_label == 1:
            true_positive += 1
        elif true_label == 0 and predicted_label == 1:
            false_positive += 1
        elif true_label == 0 and predicted_label == 0:
            true_negative += 1
        else:
            false_negative += 1

        num_candidates = 0 if candidates is None else len(candidates)
        prediction_rows.append(
            {
                "patientId": image_id,
                "truth": true_label,
                "prediction": predicted_label,
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
    return {
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
        "detector_score_threshold": detector_score_threshold,
        "roi_threshold": roi_threshold,
        "strong_detector_threshold": strong_detector_threshold,
        "metric_iou_threshold": metric_iou_threshold,
    }, pd.DataFrame(prediction_rows)

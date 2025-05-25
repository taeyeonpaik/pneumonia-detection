import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dicom_metadata import load_metadata_table
from dataset import collate, prepare_data, resolve_data_dir, seed_worker
from model_FRCNN import model as build_detector
from pipeline_FRCNN import autocast_context as detector_autocast_context
from testing_FRCNN import (
    calculate_image_precision,
    choose_device,
    matched_prediction_ious,
    parse_thresholds,
    save_confusion_matrix,
    set_seed,
)
from threshold_sweep import load_checkpoint, parse_score_thresholds
from train_classifier import (
    autocast_context as classifier_autocast_context,
    build_classifier,
    classifier_logits,
    forward_classifier,
    make_eval_loaders,
    normalization_for_architecture,
    parse_tta_modes,
    unpack_classification_batch,
)


def parse_paths(value):
    return [Path(item.strip()).expanduser().resolve() for item in value.split(",") if item.strip()]


def parse_values(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_values(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_methods(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def filter_detector_predictions(
    boxes,
    scores,
    score_threshold,
    max_detections,
    min_area_ratio=0.0,
    max_area_ratio=1.0,
    image_size=1024.0,
):
    if len(scores) == 0:
        return boxes, scores
    keep = scores >= score_threshold
    if min_area_ratio > 0.0 or max_area_ratio < 1.0:
        widths = np.maximum(0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(0, boxes[:, 3] - boxes[:, 1])
        area_ratios = (widths * heights) / max(image_size * image_size, 1.0)
        keep = keep & (area_ratios >= min_area_ratio) & (area_ratios <= max_area_ratio)
    boxes = boxes[keep]
    scores = scores[keep]
    if len(scores) == 0:
        return boxes, scores
    order = np.argsort(scores)[::-1]
    if max_detections is not None and max_detections > 0:
        order = order[:max_detections]
    return boxes[order], scores[order]


def filter_boxes_by_area(boxes, scores, min_area_ratio=0.0, max_area_ratio=1.0, image_size=1024.0):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0 or (min_area_ratio <= 0.0 and max_area_ratio >= 1.0):
        return boxes, scores

    widths = np.maximum(0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0, boxes[:, 3] - boxes[:, 1])
    area_ratios = (widths * heights) / max(image_size * image_size, 1.0)
    keep = (area_ratios >= min_area_ratio) & (area_ratios <= max_area_ratio)
    return boxes[keep], scores[keep]


def nms_boxes(boxes, scores, iou_threshold=0.5, max_boxes=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return boxes, scores

    order = np.argsort(scores)[::-1]
    keep_indices = []
    while len(order) > 0:
        current = order[0]
        keep_indices.append(current)
        if max_boxes is not None and max_boxes > 0 and len(keep_indices) >= max_boxes:
            break
        if len(order) == 1:
            break

        remaining = order[1:]
        ious = box_iou_matrix(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    return boxes[keep_indices], scores[keep_indices]


def box_iou_matrix(box, boxes):
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


def weighted_box_fusion(boxes, scores, iou_threshold=0.55, score_threshold=0.001, max_boxes=300):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return boxes, scores

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    if len(scores) == 0:
        return boxes, scores

    order = np.argsort(scores)[::-1]
    if max_boxes is not None and max_boxes > 0:
        order = order[:max_boxes]
    boxes = boxes[order]
    scores = scores[order]

    clusters = []
    for box, score in zip(boxes, scores):
        best_index = None
        best_iou = 0.0
        for index, cluster in enumerate(clusters):
            iou = float(box_iou_matrix(box, cluster["box"][None, :])[0])
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is None or best_iou < iou_threshold:
            clusters.append(
                {
                    "boxes": [box],
                    "scores": [float(score)],
                    "box": box.astype(np.float32),
                }
            )
            continue

        cluster = clusters[best_index]
        cluster["boxes"].append(box)
        cluster["scores"].append(float(score))
        cluster_boxes = np.asarray(cluster["boxes"], dtype=np.float32)
        cluster_scores = np.asarray(cluster["scores"], dtype=np.float32)
        weights = np.maximum(cluster_scores, 1e-6)
        cluster["box"] = np.average(cluster_boxes, axis=0, weights=weights).astype(np.float32)

    fused_boxes = []
    fused_scores = []
    for cluster in clusters:
        cluster_scores = np.asarray(cluster["scores"], dtype=np.float32)
        # Favor high-confidence agreement without letting duplicate boxes exceed 1.0.
        fused_score = 1.0 - float(np.prod(1.0 - np.clip(cluster_scores, 0.0, 1.0)))
        fused_boxes.append(cluster["box"])
        fused_scores.append(fused_score)

    fused_boxes = np.asarray(fused_boxes, dtype=np.float32).reshape(-1, 4)
    fused_scores = np.asarray(fused_scores, dtype=np.float32).reshape(-1)
    order = np.argsort(fused_scores)[::-1]
    return fused_boxes[order], fused_scores[order]


def concatenate_detector_boxes(boxes, scores, score_threshold=0.001, max_boxes=300):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return boxes, scores

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    if len(scores) == 0:
        return boxes, scores

    order = np.argsort(scores)[::-1]
    if max_boxes is not None and max_boxes > 0:
        order = order[:max_boxes]
    return boxes[order], scores[order]


def merge_detector_record_lists(
    record_lists,
    method="wbf",
    iou_threshold=0.55,
    score_threshold=0.001,
    max_boxes=300,
):
    if not record_lists:
        return []
    if len(record_lists) == 1 and method == "none":
        return record_lists[0]

    patient_ids = [record["patientId"] for record in record_lists[0]]
    records_by_source = [
        {record["patientId"]: record for record in records}
        for records in record_lists
    ]
    merged_records = []
    for patient_id in tqdm(patient_ids, total=len(patient_ids), desc="Merging detector boxes", leave=True, file=sys.stdout):
        first_record = records_by_source[0][patient_id]
        all_boxes = []
        all_scores = []
        for source_records in records_by_source:
            record = source_records[patient_id]
            if len(record["scores"]) == 0:
                continue
            all_boxes.append(record["boxes"])
            all_scores.append(record["scores"])

        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)

        if method == "none":
            merged_boxes, merged_scores = concatenate_detector_boxes(
                boxes,
                scores,
                score_threshold=score_threshold,
                max_boxes=max_boxes,
            )
        elif method == "wbf":
            merged_boxes, merged_scores = weighted_box_fusion(
                boxes,
                scores,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                max_boxes=max_boxes,
            )
        else:
            raise ValueError(f"Unsupported detector ensemble method: {method}")

        merged_records.append(
            {
                "patientId": patient_id,
                "truth_boxes": first_record["truth_boxes"],
                "boxes": merged_boxes,
                "scores": merged_scores,
            }
        )
    return merged_records


def resize_boxes_around_center(boxes, scale=1.0, image_size=1024.0):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if len(boxes) == 0 or abs(scale - 1.0) < 1e-6:
        return boxes

    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    widths = (boxes[:, 2] - boxes[:, 0]).clip(min=0) * scale
    heights = (boxes[:, 3] - boxes[:, 1]).clip(min=0) * scale
    boxes[:, 0] = centers_x - widths / 2.0
    boxes[:, 2] = centers_x + widths / 2.0
    boxes[:, 1] = centers_y - heights / 2.0
    boxes[:, 3] = centers_y + heights / 2.0
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_size)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_size)
    return boxes


def probability_to_logit(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def apply_temperature_to_probability(probability, temperature):
    logits = probability_to_logit(probability) / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_nll(labels, probabilities):
    labels = np.asarray(labels, dtype=np.float32)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)))


def calibrate_classifier_temperature(valid_records, temperature_grid):
    labels = np.asarray([row["truth"] for row in valid_records.values()], dtype=np.float32)
    probabilities = np.asarray(
        [row["classifier_probability"] for row in valid_records.values()],
        dtype=np.float32,
    )
    best_temperature = 1.0
    best_loss = binary_nll(labels, probabilities)
    for temperature in temperature_grid:
        calibrated = apply_temperature_to_probability(probabilities, temperature)
        loss = binary_nll(labels, calibrated)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature, best_loss


def apply_classifier_temperature(records, temperature):
    calibrated_records = {}
    for patient_id, row in records.items():
        calibrated = dict(row)
        calibrated["classifier_probability"] = float(
            apply_temperature_to_probability(row["classifier_probability"], temperature)
        )
        calibrated_records[patient_id] = calibrated
    return calibrated_records


def calibrate_classifier_record_sets(valid_records_list, test_records_list, temperature_grid):
    calibrated_valid = []
    calibrated_test = []
    temperatures = []
    for index, (valid_records, test_records) in enumerate(zip(valid_records_list, test_records_list)):
        temperature, loss = calibrate_classifier_temperature(valid_records, temperature_grid)
        print(
            f"Classifier {index} validation temperature={temperature:g}, calibrated NLL={loss:.5f}",
            flush=True,
        )
        calibrated_valid.append(apply_classifier_temperature(valid_records, temperature))
        calibrated_test.append(apply_classifier_temperature(test_records, temperature))
        temperatures.append(float(temperature))
    return calibrated_valid, calibrated_test, temperatures


def postprocess_detector_records(
    detector_records,
    classifier_records_list,
    score_calibration="none",
    box_scale=1.0,
    min_area_ratio=0.0,
    max_area_ratio=1.0,
    nms_iou=0.0,
):
    if (
        score_calibration == "none"
        and abs(box_scale - 1.0) < 1e-6
        and min_area_ratio <= 0.0
        and max_area_ratio >= 1.0
        and nms_iou <= 0.0
    ):
        return detector_records

    classifier_maps = []
    for classifier_records in classifier_records_list:
        classifier_maps.append(
            {
                patient_id: float(record["classifier_probability"])
                for patient_id, record in classifier_records.items()
            }
        )

    postprocessed = []
    for record in detector_records:
        patient_id = record["patientId"]
        boxes = resize_boxes_around_center(record["boxes"], scale=box_scale)
        scores = np.asarray(record["scores"], dtype=np.float32).copy()
        if score_calibration != "none" and classifier_maps:
            probabilities = np.asarray(
                [classifier_map[patient_id] for classifier_map in classifier_maps],
                dtype=np.float32,
            )
            if score_calibration == "classifier_max":
                multiplier = float(np.max(probabilities))
            elif score_calibration == "classifier_mean":
                multiplier = float(np.mean(probabilities))
            elif score_calibration == "classifier_min":
                multiplier = float(np.min(probabilities))
            else:
                raise ValueError(f"Unsupported detector score calibration: {score_calibration}")
            scores = scores * multiplier

        boxes, scores = filter_boxes_by_area(
            boxes,
            scores,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        if nms_iou > 0.0:
            boxes, scores = nms_boxes(boxes, scores, iou_threshold=nms_iou)

        postprocessed.append(
            {
                "patientId": patient_id,
                "truth_boxes": record["truth_boxes"],
                "boxes": boxes,
                "scores": scores,
            }
        )
    return postprocessed


def make_loader(dataset, batch_size, num_workers, device, collate_fn=None):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker
    return DataLoader(dataset, **loader_kwargs)


def collect_classifier_probabilities(dataloader, classifier, device, use_amp=False, desc="Classifier"):
    rows = []
    classifier.eval()
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader), desc=desc, leave=True, file=sys.stdout)
        for batch in progress:
            images, metadata, labels, image_ids = unpack_classification_batch(batch)
            images = images.to(device, non_blocking=non_blocking)
            if metadata is not None:
                metadata = metadata.to(device, non_blocking=non_blocking)
            with classifier_autocast_context(use_amp):
                logits = classifier_logits(forward_classifier(classifier, images, metadata)).flatten()
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            for image_id, label, probability in zip(image_ids, labels_np, probabilities):
                rows.append(
                    {
                        "patientId": image_id,
                        "truth": int(label),
                        "classifier_probability": float(probability),
                    }
                )
    return {row["patientId"]: row for row in rows}


def collect_classifier_probabilities_tta(eval_loaders, classifier, device, use_amp=False, desc="Classifier"):
    probability_sums = {}
    probability_counts = {}
    labels_by_id = {}

    for mode, dataloader in eval_loaders:
        mode_desc = desc if len(eval_loaders) == 1 else f"{desc} {mode}"
        records = collect_classifier_probabilities(
            dataloader,
            classifier,
            device,
            use_amp=use_amp,
            desc=mode_desc,
        )
        for patient_id, row in records.items():
            probability_sums[patient_id] = probability_sums.get(patient_id, 0.0) + float(row["classifier_probability"])
            probability_counts[patient_id] = probability_counts.get(patient_id, 0) + 1
            labels_by_id[patient_id] = int(row["truth"])

    return {
        patient_id: {
            "patientId": patient_id,
            "truth": labels_by_id[patient_id],
            "classifier_probability": probability_sums[patient_id] / max(probability_counts[patient_id], 1),
        }
        for patient_id in labels_by_id
    }


def collect_detector_predictions(dataloader, detector, device, use_amp=False, desc="Detector"):
    rows = []
    detector.eval()
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader), desc=desc, leave=True, file=sys.stdout)
        for images, targets, image_ids in progress:
            images = [image.to(device, non_blocking=non_blocking) for image in images]
            with detector_autocast_context(use_amp):
                outputs = detector(images)

            for index, output in enumerate(outputs):
                rows.append(
                    {
                        "patientId": image_ids[index],
                        "truth_boxes": targets[index]["boxes"].detach().cpu().numpy(),
                        "boxes": output["boxes"].detach().cpu().numpy(),
                        "scores": output["scores"].detach().cpu().numpy(),
                    }
                )
    return rows


def detector_feature_values(
    record,
    feature_score_threshold,
    max_feature_detections,
    min_area_ratio=0.0,
    max_area_ratio=1.0,
):
    boxes, scores = filter_detector_predictions(
        record["boxes"],
        record["scores"],
        score_threshold=feature_score_threshold,
        max_detections=max_feature_detections,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    if len(scores) == 0:
        return {
            "detector_max_score": 0.0,
            "detector_second_score": 0.0,
            "detector_mean_score": 0.0,
            "detector_sum_score": 0.0,
            "detector_count_norm": 0.0,
            "detector_max_area_ratio": 0.0,
            "detector_total_area_ratio": 0.0,
            "detector_top2_gap": 0.0,
            "detector_top_width_ratio": 0.0,
            "detector_top_height_ratio": 0.0,
            "detector_top_center_x": 0.0,
            "detector_top_center_y": 0.0,
            "detector_top_aspect_ratio": 0.0,
            "detector_top_score_area": 0.0,
            "detector_mean_center_x": 0.0,
            "detector_mean_center_y": 0.0,
        }

    widths = (boxes[:, 2] - boxes[:, 0]).clip(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clip(min=0)
    areas = widths * heights
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    area_denominator = 1024.0 * 1024.0
    max_score = float(scores[0])
    second_score = float(scores[1]) if len(scores) > 1 else 0.0
    top_area_ratio = float(areas[0] / area_denominator) if len(areas) else 0.0
    return {
        "detector_max_score": max_score,
        "detector_second_score": second_score,
        "detector_mean_score": float(np.mean(scores)),
        "detector_sum_score": float(np.sum(scores)),
        "detector_count_norm": float(min(len(scores), 10) / 10.0),
        "detector_max_area_ratio": float(np.max(areas) / area_denominator) if len(areas) else 0.0,
        "detector_total_area_ratio": float(np.sum(areas) / area_denominator) if len(areas) else 0.0,
        "detector_top2_gap": max_score - second_score,
        "detector_top_width_ratio": float(widths[0] / 1024.0),
        "detector_top_height_ratio": float(heights[0] / 1024.0),
        "detector_top_center_x": float(centers_x[0] / 1024.0),
        "detector_top_center_y": float(centers_y[0] / 1024.0),
        "detector_top_aspect_ratio": float(widths[0] / max(heights[0], 1e-6)),
        "detector_top_score_area": max_score * top_area_ratio,
        "detector_mean_center_x": float(np.mean(centers_x) / 1024.0),
        "detector_mean_center_y": float(np.mean(centers_y) / 1024.0),
    }


def build_feature_table(
    detector_records,
    classifier_records_list,
    feature_score_threshold,
    max_feature_detections,
    min_area_ratio=0.0,
    max_area_ratio=1.0,
):
    rows = []
    for record in detector_records:
        patient_id = record["patientId"]
        classifier_probs = [
            float(classifier_records[patient_id]["classifier_probability"])
            for classifier_records in classifier_records_list
        ]
        classifier_array = np.asarray(classifier_probs, dtype=np.float32)
        detector_features = detector_feature_values(
            record,
            feature_score_threshold=feature_score_threshold,
            max_feature_detections=max_feature_detections,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        classifier_max = float(np.max(classifier_array))
        detector_max = detector_features["detector_max_score"]
        row = {
            "patientId": patient_id,
            "truth": int(len(record["truth_boxes"]) > 0),
            "classifier_mean_probability": float(np.mean(classifier_array)),
            "classifier_max_probability": classifier_max,
            "classifier_min_probability": float(np.min(classifier_array)),
            "classifier_std_probability": float(np.std(classifier_array)),
            **detector_features,
            "classifier_detector_product": classifier_max * detector_max,
            "classifier_detector_abs_diff": abs(classifier_max - detector_max),
        }
        for index, probability in enumerate(classifier_probs):
            row[f"classifier_{index}_probability"] = probability
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_from_decisions(
    detector_records,
    decisions,
    scores,
    detector_iou_thresholds,
    box_score_threshold,
    max_detections,
    metric_iou_threshold,
    min_area_ratio=0.0,
    max_area_ratio=1.0,
):
    image_precisions = []
    matched_ious = []
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    prediction_rows = []

    for record in detector_records:
        patient_id = record["patientId"]
        predicted_label = int(decisions[patient_id])
        boxes, detector_scores = filter_detector_predictions(
            record["boxes"],
            record["scores"],
            score_threshold=box_score_threshold,
            max_detections=max_detections,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        if not predicted_label:
            boxes = boxes[:0]
            detector_scores = detector_scores[:0]

        truth_boxes = record["truth_boxes"]
        true_label = int(len(truth_boxes) > 0)
        image_precisions.append(calculate_image_precision(truth_boxes, boxes, detector_iou_thresholds))
        matched_ious.extend(matched_prediction_ious(truth_boxes, boxes, metric_iou_threshold))

        if true_label == 1 and predicted_label == 1:
            true_positive += 1
        elif true_label == 0 and predicted_label == 1:
            false_positive += 1
        elif true_label == 0 and predicted_label == 0:
            true_negative += 1
        else:
            false_negative += 1

        prediction_rows.append(
            {
                "patientId": patient_id,
                "truth": true_label,
                "prediction": predicted_label,
                "fusion_score": float(scores.get(patient_id, predicted_label)),
                "num_predicted_boxes": len(boxes),
                "max_detector_score": float(detector_scores[0]) if len(detector_scores) else 0.0,
            }
        )

    total = true_positive + true_negative + false_positive + false_negative
    image_precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * image_precision * recall / max(image_precision + recall, 1e-12)
    metrics = {
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
        "box_score_threshold": box_score_threshold,
        "min_box_area_ratio": min_area_ratio,
        "max_box_area_ratio": max_area_ratio,
        "metric_iou_threshold": metric_iou_threshold,
    }
    return metrics, pd.DataFrame(prediction_rows)


def image_metrics_from_array(labels, predicted):
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    true_positive = int(((labels == 1) & (predicted == 1)).sum())
    true_negative = int(((labels == 0) & (predicted == 0)).sum())
    false_positive = int(((labels == 0) & (predicted == 1)).sum())
    false_negative = int(((labels == 1) & (predicted == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    total = true_positive + true_negative + false_positive + false_negative
    return {
        "image_accuracy": (true_positive + true_negative) / max(total, 1),
        "image_precision": precision,
        "image_recall": recall,
        "image_f1": f1,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "num_images": total,
    }


def candidate_is_better(candidate, best, min_precision, min_recall):
    candidate_ok = (
        candidate["image_precision"] >= min_precision
        and candidate["image_recall"] >= min_recall
    )
    candidate_key = (
        int(candidate_ok),
        candidate["image_f1"],
        candidate["image_precision"],
        candidate["image_recall"],
        candidate.get("detector_precision", 0.0),
        candidate.get("mean_iou", 0.0),
    )
    if best is None:
        return True
    best_ok = best["image_precision"] >= min_precision and best["image_recall"] >= min_recall
    best_key = (
        int(best_ok),
        best["image_f1"],
        best["image_precision"],
        best["image_recall"],
        best.get("detector_precision", 0.0),
        best.get("mean_iou", 0.0),
    )
    return candidate_key > best_key


def standardize_features(train_features, test_features):
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_features - mean) / std, (test_features - mean) / std, mean, std


def apply_feature_standardization(features, mean, std):
    return (features - mean) / std


class LogisticFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.linear(features).flatten()


def train_logistic_fusion(features, labels, pos_weight, l2, epochs, lr, seed):
    torch.manual_seed(seed)
    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
    model = LogisticFusion(features.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features_tensor)
        loss = criterion(logits, labels_tensor)
        if l2 > 0:
            loss = loss + l2 * model.linear.weight.pow(2).sum()
        loss.backward()
        optimizer.step()
    return model


def predict_logistic(model, features):
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float32))
    return torch.sigmoid(logits).detach().cpu().numpy()


class MLPFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features).flatten()


def train_mlp_fusion(features, labels, pos_weight, l2, hidden_dim, dropout, epochs, lr, seed):
    torch.manual_seed(seed)
    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
    model = MLPFusion(features.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features_tensor)
        loss = criterion(logits, labels_tensor)
        loss.backward()
        optimizer.step()
    return model


def predict_mlp(model, features):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float32))
    return torch.sigmoid(logits).detach().cpu().numpy()


def run_meta_fusion(
    valid_records,
    test_records,
    fit_features,
    valid_features,
    test_features,
    feature_columns,
    detector_iou_thresholds,
    meta_thresholds,
    box_score_thresholds,
    pos_weights,
    l2_values,
    max_detections,
    metric_iou_threshold,
    min_box_area_ratio,
    max_box_area_ratio,
    epochs,
    lr,
    seed,
    min_precision,
    min_recall,
):
    x_fit = fit_features[feature_columns].to_numpy(dtype=np.float32)
    y_fit = fit_features["truth"].to_numpy(dtype=np.float32)
    x_valid = valid_features[feature_columns].to_numpy(dtype=np.float32)
    x_test = test_features[feature_columns].to_numpy(dtype=np.float32)
    x_fit_scaled, x_valid_scaled, mean, std = standardize_features(x_fit, x_valid)
    x_test_scaled = apply_feature_standardization(x_test, mean, std)

    rows = []
    best = None
    best_model = None
    best_test_scores = None
    patient_ids_valid = valid_features["patientId"].tolist()
    patient_ids_test = test_features["patientId"].tolist()
    labels = valid_features["truth"].to_numpy(dtype=np.int64)

    model_grid = list(product(pos_weights, l2_values))
    total_rules = len(model_grid) * len(meta_thresholds)
    print(
        f"Starting meta logistic sweep: {len(model_grid)} models x "
        f"{len(meta_thresholds)} meta thresholds "
        f"= {total_rules} rules",
        flush=True,
    )
    evaluated_rules = 0
    for model_index, (pos_weight, l2) in enumerate(model_grid, start=1):
        print(
            f"Training meta logistic model {model_index}/{len(model_grid)} "
            f"(pos_weight={pos_weight:g}, l2={l2:g})",
            flush=True,
        )
        model = train_logistic_fusion(
            x_fit_scaled,
            y_fit,
            pos_weight=pos_weight,
            l2=l2,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
        valid_probabilities = predict_logistic(model, x_valid_scaled)
        for meta_threshold in meta_thresholds:
            evaluated_rules += 1
            predicted = valid_probabilities >= meta_threshold
            metrics = image_metrics_from_array(labels, predicted)
            row = {
                "method": "meta_logistic",
                "meta_threshold": meta_threshold,
                "meta_pos_weight": pos_weight,
                "meta_l2": l2,
                "constraint_satisfied": int(
                    metrics["image_precision"] >= min_precision
                    and metrics["image_recall"] >= min_recall
                ),
                **metrics,
            }
            rows.append(row)
            if candidate_is_better(row, best, min_precision, min_recall):
                best = row
                best_model = model
                best_test_scores = predict_logistic(model, x_test_scaled)
                print(
                    "New best meta validation: "
                    f"rule={evaluated_rules}/{total_rules}, threshold={meta_threshold:g}, "
                    f"pos_weight={pos_weight:g}, l2={l2:g}, "
                    f"F1={row['image_f1']:.5f}, precision={row['image_precision']:.5f}, "
                    f"recall={row['image_recall']:.5f}, ok={row['constraint_satisfied']}",
                    flush=True,
                )
            elif evaluated_rules % 100 == 0:
                print(
                    f"Meta sweep progress: {evaluated_rules}/{total_rules} rules evaluated",
                    flush=True,
                )

    valid_scores = predict_logistic(best_model, x_valid_scaled)
    valid_score_map = {
        patient_id: float(probability)
        for patient_id, probability in zip(patient_ids_valid, valid_scores)
    }
    valid_decisions = {
        patient_id: bool(probability >= best["meta_threshold"])
        for patient_id, probability in zip(patient_ids_valid, valid_scores)
    }
    validation_box_rows = []
    best_box_metrics = None
    for box_score_threshold in box_score_thresholds:
        box_metrics, _ = metrics_from_decisions(
            valid_records,
            valid_decisions,
            valid_score_map,
            detector_iou_thresholds,
            box_score_threshold=box_score_threshold,
            max_detections=max_detections,
            metric_iou_threshold=metric_iou_threshold,
            min_area_ratio=min_box_area_ratio,
            max_area_ratio=max_box_area_ratio,
        )
        box_row = {**best, **box_metrics}
        validation_box_rows.append(box_row)
        if best_box_metrics is None or (
            box_row["detector_precision"],
            box_row["mean_iou"],
            -box_row["num_matched_detections"],
        ) > (
            best_box_metrics["detector_precision"],
            best_box_metrics["mean_iou"],
            -best_box_metrics["num_matched_detections"],
        ):
            best_box_metrics = box_row
    best = best_box_metrics

    test_scores = {
        patient_id: float(probability)
        for patient_id, probability in zip(patient_ids_test, best_test_scores)
    }
    test_decisions = {
        patient_id: bool(probability >= best["meta_threshold"])
        for patient_id, probability in zip(patient_ids_test, best_test_scores)
    }
    validation_metrics, validation_predictions = metrics_from_decisions(
        valid_records,
        valid_decisions,
        valid_score_map,
        detector_iou_thresholds,
        box_score_threshold=float(best["box_score_threshold"]),
        max_detections=max_detections,
        metric_iou_threshold=metric_iou_threshold,
        min_area_ratio=min_box_area_ratio,
        max_area_ratio=max_box_area_ratio,
    )
    test_metrics, test_predictions = metrics_from_decisions(
        test_records,
        test_decisions,
        test_scores,
        detector_iou_thresholds,
        box_score_threshold=float(best["box_score_threshold"]),
        max_detections=max_detections,
        metric_iou_threshold=metric_iou_threshold,
        min_area_ratio=min_box_area_ratio,
        max_area_ratio=max_box_area_ratio,
    )
    selected = {"selected_from": "validation_meta_logistic", **best, **test_metrics}
    validation_predictions.attrs["metrics"] = validation_metrics
    return rows + validation_box_rows, best, selected, test_predictions, validation_predictions


def run_mlp_fusion(
    valid_records,
    test_records,
    fit_features,
    valid_features,
    test_features,
    feature_columns,
    detector_iou_thresholds,
    mlp_thresholds,
    box_score_thresholds,
    pos_weights,
    l2_values,
    hidden_dims,
    dropout_values,
    max_detections,
    metric_iou_threshold,
    min_box_area_ratio,
    max_box_area_ratio,
    epochs,
    lr,
    seed,
    min_precision,
    min_recall,
):
    x_fit = fit_features[feature_columns].to_numpy(dtype=np.float32)
    y_fit = fit_features["truth"].to_numpy(dtype=np.float32)
    x_valid = valid_features[feature_columns].to_numpy(dtype=np.float32)
    y_valid = valid_features["truth"].to_numpy(dtype=np.float32)
    x_test = test_features[feature_columns].to_numpy(dtype=np.float32)
    x_fit_scaled, x_valid_scaled, mean, std = standardize_features(x_fit, x_valid)
    x_test_scaled = apply_feature_standardization(x_test, mean, std)

    rows = []
    best = None
    best_model = None
    best_test_scores = None
    patient_ids_valid = valid_features["patientId"].tolist()
    patient_ids_test = test_features["patientId"].tolist()
    labels = valid_features["truth"].to_numpy(dtype=np.int64)

    model_grid = list(product(pos_weights, l2_values, hidden_dims, dropout_values))
    total_rules = len(model_grid) * len(mlp_thresholds)
    print(
        f"Starting MLP fusion sweep: {len(model_grid)} models x "
        f"{len(mlp_thresholds)} thresholds = {total_rules} rules",
        flush=True,
    )
    evaluated_rules = 0
    for model_index, (pos_weight, l2, hidden_dim, dropout) in enumerate(model_grid, start=1):
        print(
            f"Training MLP fusion model {model_index}/{len(model_grid)} "
            f"(pos_weight={pos_weight:g}, l2={l2:g}, hidden={int(hidden_dim)}, dropout={dropout:g})",
            flush=True,
        )
        model = train_mlp_fusion(
            x_fit_scaled,
            y_fit,
            pos_weight=pos_weight,
            l2=l2,
            hidden_dim=int(hidden_dim),
            dropout=dropout,
            epochs=epochs,
            lr=lr,
            seed=seed + model_index,
        )
        valid_probabilities = predict_mlp(model, x_valid_scaled)
        for mlp_threshold in mlp_thresholds:
            evaluated_rules += 1
            predicted = valid_probabilities >= mlp_threshold
            metrics = image_metrics_from_array(labels, predicted)
            row = {
                "method": "mlp_fusion",
                "mlp_threshold": mlp_threshold,
                "mlp_pos_weight": pos_weight,
                "mlp_l2": l2,
                "mlp_hidden_dim": int(hidden_dim),
                "mlp_dropout": dropout,
                "constraint_satisfied": int(
                    metrics["image_precision"] >= min_precision
                    and metrics["image_recall"] >= min_recall
                ),
                **metrics,
            }
            rows.append(row)
            if candidate_is_better(row, best, min_precision, min_recall):
                best = row
                best_model = model
                best_test_scores = predict_mlp(model, x_test_scaled)
                print(
                    "New best MLP validation: "
                    f"rule={evaluated_rules}/{total_rules}, threshold={mlp_threshold:g}, "
                    f"pos_weight={pos_weight:g}, l2={l2:g}, hidden={int(hidden_dim)}, "
                    f"dropout={dropout:g}, F1={row['image_f1']:.5f}, "
                    f"precision={row['image_precision']:.5f}, recall={row['image_recall']:.5f}, "
                    f"ok={row['constraint_satisfied']}",
                    flush=True,
                )
            elif evaluated_rules % 100 == 0:
                print(f"MLP sweep progress: {evaluated_rules}/{total_rules} rules evaluated", flush=True)

    valid_scores = predict_mlp(best_model, x_valid_scaled)
    valid_score_map = {
        patient_id: float(probability)
        for patient_id, probability in zip(patient_ids_valid, valid_scores)
    }
    valid_decisions = {
        patient_id: bool(probability >= best["mlp_threshold"])
        for patient_id, probability in zip(patient_ids_valid, valid_scores)
    }
    validation_box_rows = []
    best_box_metrics = None
    for box_score_threshold in box_score_thresholds:
        box_metrics, _ = metrics_from_decisions(
            valid_records,
            valid_decisions,
            valid_score_map,
            detector_iou_thresholds,
            box_score_threshold=box_score_threshold,
            max_detections=max_detections,
            metric_iou_threshold=metric_iou_threshold,
            min_area_ratio=min_box_area_ratio,
            max_area_ratio=max_box_area_ratio,
        )
        box_row = {**best, **box_metrics}
        validation_box_rows.append(box_row)
        if best_box_metrics is None or (
            box_row["detector_precision"],
            box_row["mean_iou"],
            -box_row["num_matched_detections"],
        ) > (
            best_box_metrics["detector_precision"],
            best_box_metrics["mean_iou"],
            -best_box_metrics["num_matched_detections"],
        ):
            best_box_metrics = box_row
    best = best_box_metrics

    test_scores = {
        patient_id: float(probability)
        for patient_id, probability in zip(patient_ids_test, best_test_scores)
    }
    test_decisions = {
        patient_id: bool(probability >= best["mlp_threshold"])
        for patient_id, probability in zip(patient_ids_test, best_test_scores)
    }
    test_metrics, test_predictions = metrics_from_decisions(
        test_records,
        test_decisions,
        test_scores,
        detector_iou_thresholds,
        box_score_threshold=float(best["box_score_threshold"]),
        max_detections=max_detections,
        metric_iou_threshold=metric_iou_threshold,
        min_area_ratio=min_box_area_ratio,
        max_area_ratio=max_box_area_ratio,
    )
    selected = {"selected_from": "validation_mlp_fusion", **best, **test_metrics}
    return rows + validation_box_rows, best, selected, test_predictions


def run_hybrid_gate(
    valid_records,
    test_records,
    valid_features,
    test_features,
    detector_iou_thresholds,
    and_classifier_thresholds,
    and_detector_thresholds,
    or_classifier_thresholds,
    or_detector_thresholds,
    box_score_thresholds,
    max_detections,
    metric_iou_threshold,
    min_box_area_ratio,
    max_box_area_ratio,
    min_precision,
    min_recall,
):
    image_rows = []
    best = None

    valid_ids = valid_features["patientId"].tolist()
    labels = valid_features["truth"].to_numpy(dtype=np.int64)
    classifier_scores = valid_features["classifier_max_probability"].to_numpy(dtype=np.float32)
    detector_scores = valid_features["detector_max_score"].to_numpy(dtype=np.float32)
    rule_grid = list(product(
        and_classifier_thresholds,
        and_detector_thresholds,
        or_classifier_thresholds,
        or_detector_thresholds,
    ))
    print(f"Starting hybrid gate sweep: {len(rule_grid)} rules", flush=True)
    print(
        f"Selection constraints: precision >= {min_precision:g}, recall >= {min_recall:g}",
        flush=True,
    )
    for rule_index, (and_cls, and_det, or_cls, or_det) in enumerate(
        tqdm(rule_grid, total=len(rule_grid), desc="hybrid sweep", leave=True, file=sys.stdout),
        start=1,
    ):
        classifier_positive = classifier_scores >= and_cls
        detector_positive = detector_scores >= and_det
        classifier_override = classifier_scores >= or_cls
        detector_override = detector_scores >= or_det
        predicted = (classifier_positive & detector_positive) | classifier_override | detector_override
        metrics = image_metrics_from_array(labels, predicted)
        row = {
            "method": "hybrid_gate",
            "and_classifier_threshold": and_cls,
            "and_detector_threshold": and_det,
            "or_classifier_threshold": or_cls,
            "or_detector_threshold": or_det,
            "constraint_satisfied": int(
                metrics["image_precision"] >= min_precision
                and metrics["image_recall"] >= min_recall
            ),
            **metrics,
        }
        image_rows.append(row)
        if candidate_is_better(row, best, min_precision, min_recall):
            best = row
            print(
                "New best hybrid validation: "
                f"rule={rule_index}/{len(rule_grid)}, and_cls={and_cls:g}, and_det={and_det:g}, "
                f"or_cls={or_cls:g}, or_det={or_det:g}, "
                f"F1={row['image_f1']:.5f}, precision={row['image_precision']:.5f}, "
                f"recall={row['image_recall']:.5f}, ok={row['constraint_satisfied']}",
                flush=True,
            )
        elif rule_index % 500 == 0:
            print(
                f"Hybrid sweep progress: {rule_index}/{len(rule_grid)} rules evaluated",
                flush=True,
            )

    valid_decisions = {
        patient_id: bool(value)
        for patient_id, value in zip(
            valid_ids,
            (
                ((classifier_scores >= best["and_classifier_threshold"]) & (detector_scores >= best["and_detector_threshold"]))
                | (classifier_scores >= best["or_classifier_threshold"])
                | (detector_scores >= best["or_detector_threshold"])
            ),
        )
    }
    valid_scores = {
        patient_id: float(score)
        for patient_id, score in zip(valid_ids, np.maximum(classifier_scores, detector_scores))
    }
    validation_box_rows = []
    best_box_metrics = None
    for box_score_threshold in box_score_thresholds:
        box_metrics, _ = metrics_from_decisions(
            valid_records,
            valid_decisions,
            valid_scores,
            detector_iou_thresholds,
            box_score_threshold=box_score_threshold,
            max_detections=max_detections,
            metric_iou_threshold=metric_iou_threshold,
            min_area_ratio=min_box_area_ratio,
            max_area_ratio=max_box_area_ratio,
        )
        box_row = {**best, **box_metrics}
        validation_box_rows.append(box_row)
        if best_box_metrics is None or (
            box_row["detector_precision"],
            box_row["mean_iou"],
            -box_row["num_matched_detections"],
        ) > (
            best_box_metrics["detector_precision"],
            best_box_metrics["mean_iou"],
            -best_box_metrics["num_matched_detections"],
        ):
            best_box_metrics = box_row
    best = best_box_metrics

    test_ids = test_features["patientId"].tolist()
    test_classifier_scores = test_features["classifier_max_probability"].to_numpy(dtype=np.float32)
    test_detector_scores = test_features["detector_max_score"].to_numpy(dtype=np.float32)
    classifier_positive = test_classifier_scores >= best["and_classifier_threshold"]
    detector_positive = test_detector_scores >= best["and_detector_threshold"]
    classifier_override = test_classifier_scores >= best["or_classifier_threshold"]
    detector_override = test_detector_scores >= best["or_detector_threshold"]
    predicted = (classifier_positive & detector_positive) | classifier_override | detector_override
    decisions = {patient_id: bool(value) for patient_id, value in zip(test_ids, predicted)}
    scores = {
        patient_id: float(score)
        for patient_id, score in zip(
            test_ids,
            np.maximum(test_classifier_scores, test_detector_scores),
        )
    }
    test_metrics, test_predictions = metrics_from_decisions(
        test_records,
        decisions,
        scores,
        detector_iou_thresholds,
        box_score_threshold=float(best["box_score_threshold"]),
        max_detections=max_detections,
        metric_iou_threshold=metric_iou_threshold,
        min_area_ratio=min_box_area_ratio,
        max_area_ratio=max_box_area_ratio,
    )
    selected = {"selected_from": "validation_hybrid_gate", **best, **test_metrics}
    return image_rows + validation_box_rows, best, selected, test_predictions


def run_precision_gate(
    valid_records,
    test_records,
    valid_features,
    test_features,
    detector_iou_thresholds,
    classifier_agree_thresholds,
    detector_required_thresholds,
    classifier_override_thresholds,
    detector_override_thresholds,
    detector_classifier_max_thresholds,
    box_score_thresholds,
    max_detections,
    metric_iou_threshold,
    min_box_area_ratio,
    max_box_area_ratio,
    min_precision,
    min_recall,
):
    image_rows = []
    best = None

    valid_ids = valid_features["patientId"].tolist()
    labels = valid_features["truth"].to_numpy(dtype=np.int64)
    classifier_min_scores = valid_features["classifier_min_probability"].to_numpy(dtype=np.float32)
    classifier_mean_scores = valid_features["classifier_mean_probability"].to_numpy(dtype=np.float32)
    detector_scores = valid_features["detector_max_score"].to_numpy(dtype=np.float32)

    rule_grid = list(
        product(
            classifier_agree_thresholds,
            detector_required_thresholds,
            classifier_override_thresholds,
            detector_override_thresholds,
            detector_classifier_max_thresholds,
        )
    )
    print(f"Starting precision gate sweep: {len(rule_grid)} rules", flush=True)
    print(
        f"Selection constraints: precision >= {min_precision:g}, recall >= {min_recall:g}",
        flush=True,
    )
    for rule_index, (
        classifier_agree_threshold,
        detector_required_threshold,
        classifier_override_threshold,
        detector_override_threshold,
        detector_classifier_max_threshold,
    ) in enumerate(
        tqdm(rule_grid, total=len(rule_grid), desc="precision gate sweep", leave=True, file=sys.stdout),
        start=1,
    ):
        classifier_detector_agreement = (
            (classifier_min_scores >= classifier_agree_threshold)
            & (detector_scores >= detector_required_threshold)
        )
        classifier_rescue = classifier_mean_scores >= classifier_override_threshold
        detector_rescue = (
            (detector_scores >= detector_override_threshold)
            & (classifier_mean_scores <= detector_classifier_max_threshold)
        )
        predicted = classifier_detector_agreement | classifier_rescue | detector_rescue
        metrics = image_metrics_from_array(labels, predicted)
        row = {
            "method": "precision_gate",
            "classifier_agree_threshold": classifier_agree_threshold,
            "detector_required_threshold": detector_required_threshold,
            "classifier_override_threshold": classifier_override_threshold,
            "detector_override_threshold": detector_override_threshold,
            "detector_classifier_max_threshold": detector_classifier_max_threshold,
            "constraint_satisfied": int(
                metrics["image_precision"] >= min_precision
                and metrics["image_recall"] >= min_recall
            ),
            **metrics,
        }
        image_rows.append(row)
        if candidate_is_better(row, best, min_precision, min_recall):
            best = row
            print(
                "New best precision gate validation: "
                f"rule={rule_index}/{len(rule_grid)}, "
                f"cls_agree={classifier_agree_threshold:g}, det_req={detector_required_threshold:g}, "
                f"cls_override={classifier_override_threshold:g}, det_override={detector_override_threshold:g}, "
                f"det_cls_max={detector_classifier_max_threshold:g}, "
                f"F1={row['image_f1']:.5f}, precision={row['image_precision']:.5f}, "
                f"recall={row['image_recall']:.5f}, ok={row['constraint_satisfied']}",
                flush=True,
            )
        elif rule_index % 500 == 0:
            print(f"Precision gate progress: {rule_index}/{len(rule_grid)} rules evaluated", flush=True)

    valid_decisions = {
        patient_id: bool(value)
        for patient_id, value in zip(
            valid_ids,
            (
                ((classifier_min_scores >= best["classifier_agree_threshold"])
                & (detector_scores >= best["detector_required_threshold"]))
                | (classifier_mean_scores >= best["classifier_override_threshold"])
                | (
                    (detector_scores >= best["detector_override_threshold"])
                    & (classifier_mean_scores <= best["detector_classifier_max_threshold"])
                )
            ),
        )
    }
    valid_scores = {
        patient_id: float(score)
        for patient_id, score in zip(valid_ids, np.maximum(classifier_mean_scores, detector_scores))
    }
    validation_box_rows = []
    best_box_metrics = None
    for box_score_threshold in box_score_thresholds:
        box_metrics, _ = metrics_from_decisions(
            valid_records,
            valid_decisions,
            valid_scores,
            detector_iou_thresholds,
            box_score_threshold=box_score_threshold,
            max_detections=max_detections,
            metric_iou_threshold=metric_iou_threshold,
            min_area_ratio=min_box_area_ratio,
            max_area_ratio=max_box_area_ratio,
        )
        box_row = {**best, **box_metrics}
        validation_box_rows.append(box_row)
        if best_box_metrics is None or (
            box_row["detector_precision"],
            box_row["mean_iou"],
            -box_row["num_matched_detections"],
        ) > (
            best_box_metrics["detector_precision"],
            best_box_metrics["mean_iou"],
            -best_box_metrics["num_matched_detections"],
        ):
            best_box_metrics = box_row
    best = best_box_metrics

    test_ids = test_features["patientId"].tolist()
    test_classifier_min_scores = test_features["classifier_min_probability"].to_numpy(dtype=np.float32)
    test_classifier_mean_scores = test_features["classifier_mean_probability"].to_numpy(dtype=np.float32)
    test_detector_scores = test_features["detector_max_score"].to_numpy(dtype=np.float32)
    predicted = (
        ((test_classifier_min_scores >= best["classifier_agree_threshold"])
        & (test_detector_scores >= best["detector_required_threshold"]))
        | (test_classifier_mean_scores >= best["classifier_override_threshold"])
        | (
            (test_detector_scores >= best["detector_override_threshold"])
            & (test_classifier_mean_scores <= best["detector_classifier_max_threshold"])
        )
    )
    decisions = {patient_id: bool(value) for patient_id, value in zip(test_ids, predicted)}
    scores = {
        patient_id: float(score)
        for patient_id, score in zip(test_ids, np.maximum(test_classifier_mean_scores, test_detector_scores))
    }
    test_metrics, test_predictions = metrics_from_decisions(
        test_records,
        decisions,
        scores,
        detector_iou_thresholds,
        box_score_threshold=float(best["box_score_threshold"]),
        max_detections=max_detections,
        metric_iou_threshold=metric_iou_threshold,
        min_area_ratio=min_box_area_ratio,
        max_area_ratio=max_box_area_ratio,
    )
    selected = {"selected_from": "validation_precision_gate", **best, **test_metrics}
    return image_rows + validation_box_rows, best, selected, test_predictions


def sequential_classifier_probability_columns(features):
    columns = []
    index = 0
    while f"classifier_{index}_probability" in features.columns:
        columns.append(f"classifier_{index}_probability")
        index += 1
    return columns


def run_xrv_rescue_gate(
    valid_records,
    test_records,
    valid_features,
    test_features,
    detector_iou_thresholds,
    base_classifier_agree_thresholds,
    detector_required_thresholds,
    base_classifier_override_thresholds,
    detector_override_thresholds,
    detector_classifier_max_thresholds,
    xrv_rescue_thresholds,
    xrv_detector_thresholds,
    xrv_base_min_thresholds,
    box_score_thresholds,
    max_detections,
    metric_iou_threshold,
    min_box_area_ratio,
    max_box_area_ratio,
    min_precision,
    min_recall,
):
    classifier_columns = sequential_classifier_probability_columns(valid_features)
    if len(classifier_columns) < 2:
        raise ValueError("xrv_rescue_gate requires at least two classifier checkpoints.")

    # Convention: pass base classifiers first and the CXR-pretrained XRV classifier last.
    base_columns = classifier_columns[:-1]
    xrv_column = classifier_columns[-1]

    image_rows = []
    best = None

    valid_ids = valid_features["patientId"].tolist()
    labels = valid_features["truth"].to_numpy(dtype=np.int64)
    base_scores = valid_features[base_columns].to_numpy(dtype=np.float32)
    base_min_scores = base_scores.min(axis=1)
    base_mean_scores = base_scores.mean(axis=1)
    base_max_scores = base_scores.max(axis=1)
    xrv_scores = valid_features[xrv_column].to_numpy(dtype=np.float32)
    detector_scores = valid_features["detector_max_score"].to_numpy(dtype=np.float32)

    rule_grid = list(
        product(
            base_classifier_agree_thresholds,
            detector_required_thresholds,
            base_classifier_override_thresholds,
            detector_override_thresholds,
            detector_classifier_max_thresholds,
            xrv_rescue_thresholds,
            xrv_detector_thresholds,
            xrv_base_min_thresholds,
        )
    )
    print(
        f"Starting XRV rescue gate sweep: {len(rule_grid)} rules "
        f"(base classifiers={base_columns}, xrv={xrv_column})",
        flush=True,
    )
    print(
        f"Selection constraints: precision >= {min_precision:g}, recall >= {min_recall:g}",
        flush=True,
    )
    for rule_index, (
        base_classifier_agree_threshold,
        detector_required_threshold,
        base_classifier_override_threshold,
        detector_override_threshold,
        detector_classifier_max_threshold,
        xrv_rescue_threshold,
        xrv_detector_threshold,
        xrv_base_min_threshold,
    ) in enumerate(
        tqdm(rule_grid, total=len(rule_grid), desc="xrv rescue gate sweep", leave=True, file=sys.stdout),
        start=1,
    ):
        base_agreement = (
            (base_min_scores >= base_classifier_agree_threshold)
            & (detector_scores >= detector_required_threshold)
        )
        base_classifier_rescue = base_mean_scores >= base_classifier_override_threshold
        base_detector_rescue = (
            (detector_scores >= detector_override_threshold)
            & (base_mean_scores <= detector_classifier_max_threshold)
        )
        xrv_rescue = (
            (xrv_scores >= xrv_rescue_threshold)
            & (detector_scores >= xrv_detector_threshold)
            & (base_max_scores >= xrv_base_min_threshold)
        )
        predicted = base_agreement | base_classifier_rescue | base_detector_rescue | xrv_rescue
        metrics = image_metrics_from_array(labels, predicted)
        row = {
            "method": "xrv_rescue_gate",
            "base_classifier_agree_threshold": base_classifier_agree_threshold,
            "detector_required_threshold": detector_required_threshold,
            "base_classifier_override_threshold": base_classifier_override_threshold,
            "detector_override_threshold": detector_override_threshold,
            "detector_classifier_max_threshold": detector_classifier_max_threshold,
            "xrv_rescue_threshold": xrv_rescue_threshold,
            "xrv_detector_threshold": xrv_detector_threshold,
            "xrv_base_min_threshold": xrv_base_min_threshold,
            "constraint_satisfied": int(
                metrics["image_precision"] >= min_precision
                and metrics["image_recall"] >= min_recall
            ),
            **metrics,
        }
        image_rows.append(row)
        if candidate_is_better(row, best, min_precision, min_recall):
            best = row
            print(
                "New best XRV rescue validation: "
                f"rule={rule_index}/{len(rule_grid)}, "
                f"base_agree={base_classifier_agree_threshold:g}, det_req={detector_required_threshold:g}, "
                f"base_override={base_classifier_override_threshold:g}, det_override={detector_override_threshold:g}, "
                f"xrv={xrv_rescue_threshold:g}, xrv_det={xrv_detector_threshold:g}, "
                f"xrv_base_min={xrv_base_min_threshold:g}, "
                f"F1={row['image_f1']:.5f}, precision={row['image_precision']:.5f}, "
                f"recall={row['image_recall']:.5f}, ok={row['constraint_satisfied']}",
                flush=True,
            )
        elif rule_index % 5000 == 0:
            print(f"XRV rescue gate progress: {rule_index}/{len(rule_grid)} rules evaluated", flush=True)

    valid_predicted = (
        (
            (base_min_scores >= best["base_classifier_agree_threshold"])
            & (detector_scores >= best["detector_required_threshold"])
        )
        | (base_mean_scores >= best["base_classifier_override_threshold"])
        | (
            (detector_scores >= best["detector_override_threshold"])
            & (base_mean_scores <= best["detector_classifier_max_threshold"])
        )
        | (
            (xrv_scores >= best["xrv_rescue_threshold"])
            & (detector_scores >= best["xrv_detector_threshold"])
            & (base_max_scores >= best["xrv_base_min_threshold"])
        )
    )
    valid_decisions = {patient_id: bool(value) for patient_id, value in zip(valid_ids, valid_predicted)}
    valid_scores = {
        patient_id: float(score)
        for patient_id, score in zip(valid_ids, np.maximum.reduce([base_mean_scores, xrv_scores, detector_scores]))
    }
    validation_box_rows = []
    best_box_metrics = None
    for box_score_threshold in box_score_thresholds:
        box_metrics, _ = metrics_from_decisions(
            valid_records,
            valid_decisions,
            valid_scores,
            detector_iou_thresholds,
            box_score_threshold=box_score_threshold,
            max_detections=max_detections,
            metric_iou_threshold=metric_iou_threshold,
            min_area_ratio=min_box_area_ratio,
            max_area_ratio=max_box_area_ratio,
        )
        box_row = {**best, **box_metrics}
        validation_box_rows.append(box_row)
        if best_box_metrics is None or (
            box_row["detector_precision"],
            box_row["mean_iou"],
            -box_row["num_matched_detections"],
        ) > (
            best_box_metrics["detector_precision"],
            best_box_metrics["mean_iou"],
            -best_box_metrics["num_matched_detections"],
        ):
            best_box_metrics = box_row
    best = best_box_metrics

    test_ids = test_features["patientId"].tolist()
    test_base_scores = test_features[base_columns].to_numpy(dtype=np.float32)
    test_base_min_scores = test_base_scores.min(axis=1)
    test_base_mean_scores = test_base_scores.mean(axis=1)
    test_base_max_scores = test_base_scores.max(axis=1)
    test_xrv_scores = test_features[xrv_column].to_numpy(dtype=np.float32)
    test_detector_scores = test_features["detector_max_score"].to_numpy(dtype=np.float32)
    predicted = (
        (
            (test_base_min_scores >= best["base_classifier_agree_threshold"])
            & (test_detector_scores >= best["detector_required_threshold"])
        )
        | (test_base_mean_scores >= best["base_classifier_override_threshold"])
        | (
            (test_detector_scores >= best["detector_override_threshold"])
            & (test_base_mean_scores <= best["detector_classifier_max_threshold"])
        )
        | (
            (test_xrv_scores >= best["xrv_rescue_threshold"])
            & (test_detector_scores >= best["xrv_detector_threshold"])
            & (test_base_max_scores >= best["xrv_base_min_threshold"])
        )
    )
    decisions = {patient_id: bool(value) for patient_id, value in zip(test_ids, predicted)}
    scores = {
        patient_id: float(score)
        for patient_id, score in zip(
            test_ids,
            np.maximum.reduce([test_base_mean_scores, test_xrv_scores, test_detector_scores]),
        )
    }
    test_metrics, test_predictions = metrics_from_decisions(
        test_records,
        decisions,
        scores,
        detector_iou_thresholds,
        box_score_threshold=float(best["box_score_threshold"]),
        max_detections=max_detections,
        metric_iou_threshold=metric_iou_threshold,
        min_area_ratio=min_box_area_ratio,
        max_area_ratio=max_box_area_ratio,
    )
    selected = {"selected_from": "validation_xrv_rescue_gate", **best, **test_metrics}
    return image_rows + validation_box_rows, best, selected, test_predictions


def run_xrv_weighted_gate(
    valid_records,
    test_records,
    valid_features,
    test_features,
    detector_iou_thresholds,
    xrv_weights,
    classifier_agree_thresholds,
    detector_required_thresholds,
    xrv_override_thresholds,
    xrv_swin_min_thresholds,
    detector_override_thresholds,
    detector_classifier_min_thresholds,
    box_score_thresholds,
    max_detections,
    metric_iou_threshold,
    min_box_area_ratio,
    max_box_area_ratio,
    min_precision,
    min_recall,
):
    required_columns = {"classifier_0_probability", "classifier_1_probability"}
    missing = required_columns.difference(valid_features.columns)
    if missing:
        raise ValueError(
            "xrv_weighted_gate requires exactly the first two classifier columns "
            f"(missing: {', '.join(sorted(missing))})."
        )

    image_rows = []
    best = None

    valid_ids = valid_features["patientId"].tolist()
    labels = valid_features["truth"].to_numpy(dtype=np.int64)
    swin_scores = valid_features["classifier_0_probability"].to_numpy(dtype=np.float32)
    xrv_scores = valid_features["classifier_1_probability"].to_numpy(dtype=np.float32)
    detector_scores = valid_features["detector_max_score"].to_numpy(dtype=np.float32)

    rule_grid = list(
        product(
            xrv_weights,
            classifier_agree_thresholds,
            detector_required_thresholds,
            xrv_override_thresholds,
            xrv_swin_min_thresholds,
            detector_override_thresholds,
            detector_classifier_min_thresholds,
        )
    )
    print(f"Starting XRV-weighted gate sweep: {len(rule_grid)} rules", flush=True)
    print(
        f"Selection constraints: precision >= {min_precision:g}, recall >= {min_recall:g}",
        flush=True,
    )
    for rule_index, (
        xrv_weight,
        classifier_agree_threshold,
        detector_required_threshold,
        xrv_override_threshold,
        xrv_swin_min_threshold,
        detector_override_threshold,
        detector_classifier_min_threshold,
    ) in enumerate(
        tqdm(rule_grid, total=len(rule_grid), desc="xrv weighted gate sweep", leave=True, file=sys.stdout),
        start=1,
    ):
        swin_weight = 1.0 - xrv_weight
        weighted_classifier = swin_weight * swin_scores + xrv_weight * xrv_scores
        both_classifiers_agree = (
            (swin_scores >= classifier_agree_threshold)
            & (xrv_scores >= classifier_agree_threshold)
        )
        detector_required = detector_scores >= detector_required_threshold
        xrv_rescue = (
            (xrv_scores >= xrv_override_threshold)
            & (swin_scores >= xrv_swin_min_threshold)
            & detector_required
        )
        detector_rescue = (
            (detector_scores >= detector_override_threshold)
            & (weighted_classifier >= detector_classifier_min_threshold)
        )
        predicted = (both_classifiers_agree & detector_required) | xrv_rescue | detector_rescue
        metrics = image_metrics_from_array(labels, predicted)
        row = {
            "method": "xrv_weighted_gate",
            "xrv_weight": xrv_weight,
            "classifier_agree_threshold": classifier_agree_threshold,
            "detector_required_threshold": detector_required_threshold,
            "xrv_override_threshold": xrv_override_threshold,
            "xrv_swin_min_threshold": xrv_swin_min_threshold,
            "detector_override_threshold": detector_override_threshold,
            "detector_classifier_min_threshold": detector_classifier_min_threshold,
            "constraint_satisfied": int(
                metrics["image_precision"] >= min_precision
                and metrics["image_recall"] >= min_recall
            ),
            **metrics,
        }
        image_rows.append(row)
        if candidate_is_better(row, best, min_precision, min_recall):
            best = row
            print(
                "New best XRV-weighted validation: "
                f"rule={rule_index}/{len(rule_grid)}, xrv_w={xrv_weight:g}, "
                f"cls={classifier_agree_threshold:g}, det={detector_required_threshold:g}, "
                f"xrv_hi={xrv_override_threshold:g}, swin_min={xrv_swin_min_threshold:g}, "
                f"det_hi={detector_override_threshold:g}, det_cls_min={detector_classifier_min_threshold:g}, "
                f"F1={row['image_f1']:.5f}, precision={row['image_precision']:.5f}, "
                f"recall={row['image_recall']:.5f}, ok={row['constraint_satisfied']}",
                flush=True,
            )
        elif rule_index % 1000 == 0:
            print(f"XRV-weighted gate progress: {rule_index}/{len(rule_grid)} rules evaluated", flush=True)

    valid_weighted_classifier = (
        (1.0 - best["xrv_weight"]) * swin_scores
        + best["xrv_weight"] * xrv_scores
    )
    valid_predicted = (
        (
            (swin_scores >= best["classifier_agree_threshold"])
            & (xrv_scores >= best["classifier_agree_threshold"])
            & (detector_scores >= best["detector_required_threshold"])
        )
        | (
            (xrv_scores >= best["xrv_override_threshold"])
            & (swin_scores >= best["xrv_swin_min_threshold"])
            & (detector_scores >= best["detector_required_threshold"])
        )
        | (
            (detector_scores >= best["detector_override_threshold"])
            & (valid_weighted_classifier >= best["detector_classifier_min_threshold"])
        )
    )
    valid_decisions = {patient_id: bool(value) for patient_id, value in zip(valid_ids, valid_predicted)}
    valid_scores = {
        patient_id: float(score)
        for patient_id, score in zip(valid_ids, np.maximum(valid_weighted_classifier, detector_scores))
    }
    validation_box_rows = []
    best_box_metrics = None
    for box_score_threshold in box_score_thresholds:
        box_metrics, _ = metrics_from_decisions(
            valid_records,
            valid_decisions,
            valid_scores,
            detector_iou_thresholds,
            box_score_threshold=box_score_threshold,
            max_detections=max_detections,
            metric_iou_threshold=metric_iou_threshold,
            min_area_ratio=min_box_area_ratio,
            max_area_ratio=max_box_area_ratio,
        )
        box_row = {**best, **box_metrics}
        validation_box_rows.append(box_row)
        if best_box_metrics is None or (
            box_row["detector_precision"],
            box_row["mean_iou"],
            -box_row["num_matched_detections"],
        ) > (
            best_box_metrics["detector_precision"],
            best_box_metrics["mean_iou"],
            -best_box_metrics["num_matched_detections"],
        ):
            best_box_metrics = box_row
    best = best_box_metrics

    test_ids = test_features["patientId"].tolist()
    test_swin_scores = test_features["classifier_0_probability"].to_numpy(dtype=np.float32)
    test_xrv_scores = test_features["classifier_1_probability"].to_numpy(dtype=np.float32)
    test_detector_scores = test_features["detector_max_score"].to_numpy(dtype=np.float32)
    test_weighted_classifier = (
        (1.0 - best["xrv_weight"]) * test_swin_scores
        + best["xrv_weight"] * test_xrv_scores
    )
    test_predicted = (
        (
            (test_swin_scores >= best["classifier_agree_threshold"])
            & (test_xrv_scores >= best["classifier_agree_threshold"])
            & (test_detector_scores >= best["detector_required_threshold"])
        )
        | (
            (test_xrv_scores >= best["xrv_override_threshold"])
            & (test_swin_scores >= best["xrv_swin_min_threshold"])
            & (test_detector_scores >= best["detector_required_threshold"])
        )
        | (
            (test_detector_scores >= best["detector_override_threshold"])
            & (test_weighted_classifier >= best["detector_classifier_min_threshold"])
        )
    )
    decisions = {patient_id: bool(value) for patient_id, value in zip(test_ids, test_predicted)}
    scores = {
        patient_id: float(score)
        for patient_id, score in zip(test_ids, np.maximum(test_weighted_classifier, test_detector_scores))
    }
    test_metrics, test_predictions = metrics_from_decisions(
        test_records,
        decisions,
        scores,
        detector_iou_thresholds,
        box_score_threshold=float(best["box_score_threshold"]),
        max_detections=max_detections,
        metric_iou_threshold=metric_iou_threshold,
        min_area_ratio=min_box_area_ratio,
        max_area_ratio=max_box_area_ratio,
    )
    selected = {"selected_from": "validation_xrv_weighted_gate", **best, **test_metrics}
    return image_rows + validation_box_rows, best, selected, test_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Learn or sweep smarter fusion rules for image classifiers + Faster R-CNN.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--classifier-checkpoints", required=True, help="Comma-separated classifier checkpoints.")
    parser.add_argument("--detector-checkpoint", required=True, help="One checkpoint or comma-separated detector checkpoints.")
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="DICOM metadata CSV required by metadata-aware classifier checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--classifier-tta-modes", default="none")
    parser.add_argument("--classifier-temperature-mode", choices=["none", "validation"], default="none")
    parser.add_argument(
        "--classifier-temperature-grid",
        default="0.50,0.60,0.70,0.80,0.90,1.00,1.10,1.20,1.35,1.50,1.75,2.00,2.50,3.00",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--detector-min-size", type=int, default=1024)
    parser.add_argument(
        "--detector-min-sizes",
        default=None,
        help="Optional comma-separated inference min sizes. If multiple sizes are provided, each detector checkpoint is run at each size.",
    )
    parser.add_argument(
        "--detector-min-size-mode",
        choices=["cartesian", "pair"],
        default="cartesian",
        help=(
            "Use every checkpoint at every min size (cartesian), or pair checkpoint N "
            "with min size N (pair)."
        ),
    )
    parser.add_argument("--detector-ensemble-method", choices=["wbf", "none"], default="wbf")
    parser.add_argument("--detector-ensemble-iou", type=float, default=0.55)
    parser.add_argument("--detector-ensemble-score-threshold", type=float, default=0.001)
    parser.add_argument("--detector-ensemble-max-boxes", type=int, default=300)
    parser.add_argument(
        "--detector-score-calibration",
        choices=["none", "classifier_max", "classifier_mean", "classifier_min"],
        default="none",
    )
    parser.add_argument("--detector-box-scale", type=float, default=1.0)
    parser.add_argument("--detector-min-area-ratio", type=float, default=0.0)
    parser.add_argument("--detector-max-area-ratio", type=float, default=1.0)
    parser.add_argument(
        "--detector-post-nms-iou",
        type=float,
        default=0.0,
        help="Optional final NMS IoU threshold after detector merging/calibration. Use 0 to disable.",
    )
    parser.add_argument("--feature-score-threshold", type=float, default=0.05)
    parser.add_argument("--max-feature-detections", type=int, default=10)
    parser.add_argument("--box-score-thresholds", default="0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62")
    parser.add_argument("--detector-iou-thresholds", default="0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75")
    parser.add_argument("--metric-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--methods", default="hybrid,meta")
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--meta-thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--meta-pos-weights", default="1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--meta-l2-values", default="0.0,0.0001,0.001,0.01")
    parser.add_argument("--meta-epochs", type=int, default=1200)
    parser.add_argument("--meta-lr", type=float, default=0.02)
    parser.add_argument(
        "--meta-fit-split",
        choices=["validation", "train"],
        default="validation",
        help="Use validation features for fast logistic fitting, or train features for cleaner learned fusion.",
    )
    parser.add_argument("--mlp-thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--mlp-pos-weights", default="1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--mlp-l2-values", default="0.0,0.0001,0.001,0.01")
    parser.add_argument("--mlp-hidden-dims", default="16,32,64")
    parser.add_argument("--mlp-dropout-values", default="0.0,0.10,0.20")
    parser.add_argument("--mlp-epochs", type=int, default=1600)
    parser.add_argument("--mlp-lr", type=float, default=0.005)
    parser.add_argument(
        "--mlp-fit-split",
        choices=["validation", "train"],
        default="validation",
        help="Use validation features for fast MLP fitting, or train features for cleaner learned fusion.",
    )
    parser.add_argument("--and-classifier-thresholds", default="0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62")
    parser.add_argument("--and-detector-thresholds", default="0.35,0.40,0.45,0.50,0.52,0.54,0.56,0.58,0.60,0.62")
    parser.add_argument("--or-classifier-thresholds", default="0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--or-detector-thresholds", default="0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--gate-classifier-agree-thresholds", default="0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.65")
    parser.add_argument("--gate-detector-required-thresholds", default="0.35,0.40,0.45,0.50,0.52,0.54,0.56,0.58,0.60")
    parser.add_argument("--gate-classifier-override-thresholds", default="0.84,0.86,0.88,0.90,0.92,0.94,0.96")
    parser.add_argument("--gate-detector-override-thresholds", default="0.54,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--gate-detector-classifier-max-thresholds", default="0.25,0.30,0.35,0.40,1.00")
    parser.add_argument("--xrv-gate-weights", default="0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--xrv-gate-classifier-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--xrv-gate-detector-thresholds", default="0.45,0.50,0.52,0.54,0.56,0.58,0.60,0.65")
    parser.add_argument("--xrv-gate-override-thresholds", default="0.80,0.84,0.86,0.88,0.90,0.92,0.94")
    parser.add_argument("--xrv-gate-swin-min-thresholds", default="0.30,0.35,0.40,0.45")
    parser.add_argument("--xrv-gate-detector-override-thresholds", default="0.75,0.80,0.85,0.90")
    parser.add_argument("--xrv-gate-detector-classifier-min-thresholds", default="0.30,0.35,0.40,0.45")
    parser.add_argument("--xrv-rescue-base-classifier-thresholds", default="0.54,0.56,0.58,0.60,0.62")
    parser.add_argument("--xrv-rescue-detector-required-thresholds", default="0.45,0.50,0.52,0.54")
    parser.add_argument("--xrv-rescue-base-override-thresholds", default="0.84,0.86,0.88,0.90")
    parser.add_argument("--xrv-rescue-detector-override-thresholds", default="0.65,0.70,0.75")
    parser.add_argument("--xrv-rescue-detector-classifier-max-thresholds", default="0.30,0.35,0.40,1.00")
    parser.add_argument("--xrv-rescue-thresholds", default="0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--xrv-rescue-detector-thresholds", default="0.55,0.60,0.65,0.70")
    parser.add_argument("--xrv-rescue-base-min-thresholds", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "fusion_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_checkpoint_paths = parse_paths(args.classifier_checkpoints)
    detector_checkpoint_paths = parse_paths(args.detector_checkpoint)
    detector_min_sizes = parse_int_values(args.detector_min_sizes) if args.detector_min_sizes else [args.detector_min_size]
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    methods = parse_methods(args.methods)
    classifier_tta_modes = parse_tta_modes(args.classifier_tta_modes)
    classifier_temperature_grid = parse_values(args.classifier_temperature_grid)
    detector_iou_thresholds = parse_thresholds(args.detector_iou_thresholds)
    box_score_thresholds = parse_score_thresholds(args.box_score_thresholds)

    print(f"Using data directory: {data_dir}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using classifier checkpoints: {[str(path) for path in classifier_checkpoint_paths]}")
    print(f"Using detector checkpoints: {[str(path) for path in detector_checkpoint_paths]}")
    print(f"Using detector min sizes: {detector_min_sizes}")
    print(
        "Detector ensemble: "
        f"method={args.detector_ensemble_method}, iou={args.detector_ensemble_iou:g}, "
        f"score_threshold={args.detector_ensemble_score_threshold:g}"
    )
    print(
        "Detector post-processing: "
        f"score_calibration={args.detector_score_calibration}, box_scale={args.detector_box_scale:g}, "
        f"min_area_ratio={args.detector_min_area_ratio:g}, "
        f"max_area_ratio={args.detector_max_area_ratio:g}, "
        f"post_nms_iou={args.detector_post_nms_iou:g}"
    )
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"Classifier TTA modes: {','.join(classifier_tta_modes)}")
    print(f"Classifier temperature mode: {args.classifier_temperature_mode}")
    sys.stdout.flush()

    train_detection, valid_detection, test_detection = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    collect_train_features = (
        ("mlp" in methods and args.mlp_fit_split == "train")
        or ("meta" in methods and args.meta_fit_split == "train")
    )
    classifier_train_records = []
    classifier_valid_records = []
    classifier_test_records = []
    classifier_metadata = []
    metadata_by_id = load_metadata_table(args.metadata_csv) if args.metadata_csv else None
    for index, checkpoint_path in enumerate(classifier_checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        architecture = checkpoint.get("architecture", "resnet50")
        image_size = int(checkpoint.get("image_size", 512))
        normalization = checkpoint.get("normalization", normalization_for_architecture(architecture, "auto"))
        metadata_dim = int(checkpoint.get("metadata_dim", 0))
        metadata_hidden_dim = int(checkpoint.get("metadata_hidden_dim", 32))
        if metadata_dim > 0 and metadata_by_id is None:
            checkpoint_metadata_csv = checkpoint.get("metadata_csv")
            if checkpoint_metadata_csv and Path(checkpoint_metadata_csv).expanduser().exists():
                metadata_by_id = load_metadata_table(checkpoint_metadata_csv)
            else:
                raise ValueError(
                    f"Classifier {checkpoint_path} requires DICOM metadata. Pass --metadata-csv."
                )
        classifier_metadata.append(
            {
                "index": index,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "architecture": architecture,
                "image_size": image_size,
                "normalization": normalization,
                "metadata_dim": metadata_dim,
            }
        )
        print(
            f"Loading classifier {index}: {architecture}, image size={image_size}, "
            f"normalization={normalization}, metadata_dim={metadata_dim}, checkpoint={checkpoint_path}"
        )
        classifier = build_classifier(
            architecture=architecture,
            pretrained=False,
            image_size=image_size,
            metadata_dim=metadata_dim,
            metadata_hidden_dim=metadata_hidden_dim,
        ).to(device)
        classifier.load_state_dict(checkpoint["model_state_dict"])

        valid_eval_loaders = make_eval_loaders(
            valid_detection,
            image_size,
            normalization,
            args.classifier_batch_size,
            args.num_workers,
            device,
            classifier_tta_modes,
            metadata_by_id=metadata_by_id if metadata_dim > 0 else None,
        )
        test_eval_loaders = make_eval_loaders(
            test_detection,
            image_size,
            normalization,
            args.classifier_batch_size,
            args.num_workers,
            device,
            classifier_tta_modes,
            metadata_by_id=metadata_by_id if metadata_dim > 0 else None,
        )
        if collect_train_features:
            train_eval_loaders = make_eval_loaders(
                train_detection,
                image_size,
                normalization,
                args.classifier_batch_size,
                args.num_workers,
                device,
                classifier_tta_modes,
                metadata_by_id=metadata_by_id if metadata_dim > 0 else None,
            )
            classifier_train_records.append(
                collect_classifier_probabilities_tta(
                    train_eval_loaders,
                    classifier,
                    device,
                    use_amp=use_amp,
                    desc=f"train classifier {index}",
                )
            )
            print(f"Finished train classifier {index}", flush=True)
        classifier_valid_records.append(
            collect_classifier_probabilities_tta(
                valid_eval_loaders,
                classifier,
                device,
                use_amp=use_amp,
                desc=f"validation classifier {index}",
            )
        )
        print(f"Finished validation classifier {index}", flush=True)
        classifier_test_records.append(
            collect_classifier_probabilities_tta(
                test_eval_loaders,
                classifier,
                device,
                use_amp=use_amp,
                desc=f"test classifier {index}",
            )
        )
        print(f"Finished test classifier {index}", flush=True)
        del classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()

    classifier_temperatures = [1.0 for _ in classifier_valid_records]
    if args.classifier_temperature_mode == "validation":
        print("Calibrating classifier probabilities on validation split", flush=True)
        classifier_valid_records, classifier_test_records, classifier_temperatures = calibrate_classifier_record_sets(
            classifier_valid_records,
            classifier_test_records,
            classifier_temperature_grid,
        )
        if classifier_train_records:
            calibrated_train = []
            for train_records, temperature in zip(classifier_train_records, classifier_temperatures):
                calibrated_train.append(apply_classifier_temperature(train_records, temperature))
            classifier_train_records = calibrated_train

    train_detector_record_lists = []
    valid_detector_record_lists = []
    test_detector_record_lists = []
    detector_metadata = []
    if args.detector_min_size_mode == "pair":
        if len(detector_checkpoint_paths) != len(detector_min_sizes):
            raise ValueError(
                "--detector-min-size-mode pair requires one min size per detector checkpoint; "
                f"got {len(detector_checkpoint_paths)} checkpoints and {len(detector_min_sizes)} sizes"
            )
        detector_specs = list(zip(detector_checkpoint_paths, detector_min_sizes))
    else:
        detector_specs = [
            (checkpoint_path, min_size)
            for checkpoint_path in detector_checkpoint_paths
            for min_size in detector_min_sizes
        ]
    for detector_index, (checkpoint_path, min_size) in enumerate(detector_specs):
        print(
            f"Loading detector {detector_index}: checkpoint={checkpoint_path}, min_size={min_size}",
            flush=True,
        )
        detector = build_detector(pretrained=False, min_size=min_size).to(device)
        detector_checkpoint = load_checkpoint(detector, checkpoint_path, device)
        detector_metadata.append(
            {
                "index": detector_index,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": detector_checkpoint.get("epoch"),
                "min_size": min_size,
            }
        )
        valid_detector_loader = make_loader(valid_detection, args.batch_size, args.num_workers, device, collate_fn=collate)
        test_detector_loader = make_loader(test_detection, args.batch_size, args.num_workers, device, collate_fn=collate)
        if collect_train_features:
            train_detector_loader = make_loader(train_detection, args.batch_size, args.num_workers, device, collate_fn=collate)
            train_records = collect_detector_predictions(
                train_detector_loader,
                detector,
                device,
                use_amp=use_amp,
                desc=f"train detector {detector_index}",
            )
            print(f"Finished train detector {detector_index}: {len(train_records)} images", flush=True)
            train_detector_record_lists.append(train_records)
        valid_records = collect_detector_predictions(
            valid_detector_loader,
            detector,
            device,
            use_amp=use_amp,
            desc=f"validation detector {detector_index}",
        )
        print(f"Finished validation detector {detector_index}: {len(valid_records)} images", flush=True)
        test_records = collect_detector_predictions(
            test_detector_loader,
            detector,
            device,
            use_amp=use_amp,
            desc=f"test detector {detector_index}",
        )
        print(f"Finished test detector {detector_index}: {len(test_records)} images", flush=True)
        valid_detector_record_lists.append(valid_records)
        test_detector_record_lists.append(test_records)
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Merging {len(valid_detector_record_lists)} detector prediction sets", flush=True)
    train_detector_records = None
    if collect_train_features:
        train_detector_records = merge_detector_record_lists(
            train_detector_record_lists,
            method=args.detector_ensemble_method,
            iou_threshold=args.detector_ensemble_iou,
            score_threshold=args.detector_ensemble_score_threshold,
            max_boxes=args.detector_ensemble_max_boxes,
        )
    valid_detector_records = merge_detector_record_lists(
        valid_detector_record_lists,
        method=args.detector_ensemble_method,
        iou_threshold=args.detector_ensemble_iou,
        score_threshold=args.detector_ensemble_score_threshold,
        max_boxes=args.detector_ensemble_max_boxes,
    )
    test_detector_records = merge_detector_record_lists(
        test_detector_record_lists,
        method=args.detector_ensemble_method,
        iou_threshold=args.detector_ensemble_iou,
        score_threshold=args.detector_ensemble_score_threshold,
        max_boxes=args.detector_ensemble_max_boxes,
    )
    print(
        f"Merged detector records: validation={len(valid_detector_records)}, test={len(test_detector_records)}",
        flush=True,
    )

    if collect_train_features:
        train_detector_records = postprocess_detector_records(
            train_detector_records,
            classifier_train_records,
            score_calibration=args.detector_score_calibration,
            box_scale=args.detector_box_scale,
            min_area_ratio=args.detector_min_area_ratio,
            max_area_ratio=args.detector_max_area_ratio,
            nms_iou=args.detector_post_nms_iou,
        )
    valid_detector_records = postprocess_detector_records(
        valid_detector_records,
        classifier_valid_records,
        score_calibration=args.detector_score_calibration,
        box_scale=args.detector_box_scale,
        min_area_ratio=args.detector_min_area_ratio,
        max_area_ratio=args.detector_max_area_ratio,
        nms_iou=args.detector_post_nms_iou,
    )
    test_detector_records = postprocess_detector_records(
        test_detector_records,
        classifier_test_records,
        score_calibration=args.detector_score_calibration,
        box_scale=args.detector_box_scale,
        min_area_ratio=args.detector_min_area_ratio,
        max_area_ratio=args.detector_max_area_ratio,
        nms_iou=args.detector_post_nms_iou,
    )

    print("Building fusion feature tables", flush=True)
    train_features = None
    if collect_train_features:
        train_features = build_feature_table(
            train_detector_records,
            classifier_train_records,
            feature_score_threshold=args.feature_score_threshold,
            max_feature_detections=args.max_feature_detections,
            min_area_ratio=args.detector_min_area_ratio,
            max_area_ratio=args.detector_max_area_ratio,
        )
        train_features.to_csv(output_dir / "fusion_train_features.csv", index=False)
    valid_features = build_feature_table(
        valid_detector_records,
        classifier_valid_records,
        feature_score_threshold=args.feature_score_threshold,
        max_feature_detections=args.max_feature_detections,
        min_area_ratio=args.detector_min_area_ratio,
        max_area_ratio=args.detector_max_area_ratio,
    )
    test_features = build_feature_table(
        test_detector_records,
        classifier_test_records,
        feature_score_threshold=args.feature_score_threshold,
        max_feature_detections=args.max_feature_detections,
        min_area_ratio=args.detector_min_area_ratio,
        max_area_ratio=args.detector_max_area_ratio,
    )
    valid_features.to_csv(output_dir / "fusion_validation_features.csv", index=False)
    test_features.to_csv(output_dir / "fusion_test_features.csv", index=False)
    print(
        f"Saved feature tables: validation={len(valid_features)} rows, test={len(test_features)} rows",
        flush=True,
    )
    if train_features is not None:
        print(f"Saved train feature table: train={len(train_features)} rows", flush=True)

    all_rows = []
    selected_rows = []
    test_predictions_by_method = {}
    validation_predictions_by_method = {}
    best_overall = None

    if "hybrid" in methods:
        rows, best, selected, predictions = run_hybrid_gate(
            valid_detector_records,
            test_detector_records,
            valid_features,
            test_features,
            detector_iou_thresholds,
            and_classifier_thresholds=parse_values(args.and_classifier_thresholds),
            and_detector_thresholds=parse_values(args.and_detector_thresholds),
            or_classifier_thresholds=parse_values(args.or_classifier_thresholds),
            or_detector_thresholds=parse_values(args.or_detector_thresholds),
            box_score_thresholds=box_score_thresholds,
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
            min_box_area_ratio=args.detector_min_area_ratio,
            max_box_area_ratio=args.detector_max_area_ratio,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        all_rows.extend(rows)
        selected_rows.append(selected)
        test_predictions_by_method["hybrid_gate"] = predictions
        print("\nBest validation hybrid gate:")
        print(pd.Series(best).to_string())
        print("\nHeld-out test metrics for selected hybrid gate:")
        print(pd.Series(selected).to_string())

    if "precision_gate" in methods:
        rows, best, selected, predictions = run_precision_gate(
            valid_detector_records,
            test_detector_records,
            valid_features,
            test_features,
            detector_iou_thresholds,
            classifier_agree_thresholds=parse_values(args.gate_classifier_agree_thresholds),
            detector_required_thresholds=parse_values(args.gate_detector_required_thresholds),
            classifier_override_thresholds=parse_values(args.gate_classifier_override_thresholds),
            detector_override_thresholds=parse_values(args.gate_detector_override_thresholds),
            detector_classifier_max_thresholds=parse_values(args.gate_detector_classifier_max_thresholds),
            box_score_thresholds=box_score_thresholds,
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
            min_box_area_ratio=args.detector_min_area_ratio,
            max_box_area_ratio=args.detector_max_area_ratio,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        all_rows.extend(rows)
        selected_rows.append(selected)
        test_predictions_by_method["precision_gate"] = predictions
        print("\nBest validation precision gate:")
        print(pd.Series(best).to_string())
        print("\nHeld-out test metrics for selected precision gate:")
        print(pd.Series(selected).to_string())

    if "xrv_weighted_gate" in methods:
        rows, best, selected, predictions = run_xrv_weighted_gate(
            valid_detector_records,
            test_detector_records,
            valid_features,
            test_features,
            detector_iou_thresholds,
            xrv_weights=parse_values(args.xrv_gate_weights),
            classifier_agree_thresholds=parse_values(args.xrv_gate_classifier_thresholds),
            detector_required_thresholds=parse_values(args.xrv_gate_detector_thresholds),
            xrv_override_thresholds=parse_values(args.xrv_gate_override_thresholds),
            xrv_swin_min_thresholds=parse_values(args.xrv_gate_swin_min_thresholds),
            detector_override_thresholds=parse_values(args.xrv_gate_detector_override_thresholds),
            detector_classifier_min_thresholds=parse_values(args.xrv_gate_detector_classifier_min_thresholds),
            box_score_thresholds=box_score_thresholds,
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
            min_box_area_ratio=args.detector_min_area_ratio,
            max_box_area_ratio=args.detector_max_area_ratio,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        all_rows.extend(rows)
        selected_rows.append(selected)
        test_predictions_by_method["xrv_weighted_gate"] = predictions
        print("\nBest validation XRV-weighted gate:")
        print(pd.Series(best).to_string())
        print("\nHeld-out test metrics for selected XRV-weighted gate:")
        print(pd.Series(selected).to_string())

    if "xrv_rescue_gate" in methods:
        rows, best, selected, predictions = run_xrv_rescue_gate(
            valid_detector_records,
            test_detector_records,
            valid_features,
            test_features,
            detector_iou_thresholds,
            base_classifier_agree_thresholds=parse_values(args.xrv_rescue_base_classifier_thresholds),
            detector_required_thresholds=parse_values(args.xrv_rescue_detector_required_thresholds),
            base_classifier_override_thresholds=parse_values(args.xrv_rescue_base_override_thresholds),
            detector_override_thresholds=parse_values(args.xrv_rescue_detector_override_thresholds),
            detector_classifier_max_thresholds=parse_values(args.xrv_rescue_detector_classifier_max_thresholds),
            xrv_rescue_thresholds=parse_values(args.xrv_rescue_thresholds),
            xrv_detector_thresholds=parse_values(args.xrv_rescue_detector_thresholds),
            xrv_base_min_thresholds=parse_values(args.xrv_rescue_base_min_thresholds),
            box_score_thresholds=box_score_thresholds,
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
            min_box_area_ratio=args.detector_min_area_ratio,
            max_box_area_ratio=args.detector_max_area_ratio,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        all_rows.extend(rows)
        selected_rows.append(selected)
        test_predictions_by_method["xrv_rescue_gate"] = predictions
        print("\nBest validation XRV rescue gate:")
        print(pd.Series(best).to_string())
        print("\nHeld-out test metrics for selected XRV rescue gate:")
        print(pd.Series(selected).to_string())

    if "meta" in methods:
        feature_columns = [
            column
            for column in valid_features.columns
            if column not in {"patientId", "truth"}
        ]
        rows, best, selected, predictions, validation_predictions = run_meta_fusion(
            valid_detector_records,
            test_detector_records,
            fit_features=train_features if train_features is not None else valid_features,
            valid_features=valid_features,
            test_features=test_features,
            feature_columns=feature_columns,
            detector_iou_thresholds=detector_iou_thresholds,
            meta_thresholds=parse_values(args.meta_thresholds),
            box_score_thresholds=box_score_thresholds,
            pos_weights=parse_values(args.meta_pos_weights),
            l2_values=parse_values(args.meta_l2_values),
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
            min_box_area_ratio=args.detector_min_area_ratio,
            max_box_area_ratio=args.detector_max_area_ratio,
            epochs=args.meta_epochs,
            lr=args.meta_lr,
            seed=args.seed,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        all_rows.extend(rows)
        selected_rows.append(selected)
        test_predictions_by_method["meta_logistic"] = predictions
        validation_predictions_by_method["meta_logistic"] = validation_predictions
        print("\nBest validation meta logistic fusion:")
        print(pd.Series(best).to_string())
        print("\nHeld-out test metrics for selected meta logistic fusion:")
        print(pd.Series(selected).to_string())

    if "mlp" in methods:
        feature_columns = [
            column
            for column in valid_features.columns
            if column not in {"patientId", "truth"}
        ]
        rows, best, selected, predictions = run_mlp_fusion(
            valid_detector_records,
            test_detector_records,
            fit_features=train_features if train_features is not None else valid_features,
            valid_features=valid_features,
            test_features=test_features,
            feature_columns=feature_columns,
            detector_iou_thresholds=detector_iou_thresholds,
            mlp_thresholds=parse_values(args.mlp_thresholds),
            box_score_thresholds=box_score_thresholds,
            pos_weights=parse_values(args.mlp_pos_weights),
            l2_values=parse_values(args.mlp_l2_values),
            hidden_dims=parse_values(args.mlp_hidden_dims),
            dropout_values=parse_values(args.mlp_dropout_values),
            max_detections=args.max_detections,
            metric_iou_threshold=args.metric_iou_threshold,
            min_box_area_ratio=args.detector_min_area_ratio,
            max_box_area_ratio=args.detector_max_area_ratio,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            seed=args.seed,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        all_rows.extend(rows)
        selected_rows.append(selected)
        test_predictions_by_method["mlp_fusion"] = predictions
        print("\nBest validation MLP fusion:")
        print(pd.Series(best).to_string())
        print("\nHeld-out test metrics for selected MLP fusion:")
        print(pd.Series(selected).to_string())

    validation_table = pd.DataFrame(all_rows).sort_values(
        ["image_f1", "image_precision", "detector_precision", "mean_iou"],
        ascending=[False, False, False, False],
    )
    validation_table.to_csv(output_dir / "fusion_sweep_validation.csv", index=False)
    selected_table = pd.DataFrame(selected_rows).sort_values(
        ["image_f1", "image_precision", "detector_precision", "mean_iou"],
        ascending=[False, False, False, False],
    )
    selected_table.to_csv(output_dir / "fusion_sweep_test_metrics.csv", index=False)
    selected_records = selected_table.to_dict(orient="records")
    with (output_dir / "fusion_sweep_test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "classifiers": classifier_metadata,
                "detectors": detector_metadata,
                "detector_ensemble": {
                    "method": args.detector_ensemble_method,
                    "iou_threshold": args.detector_ensemble_iou,
                    "score_threshold": args.detector_ensemble_score_threshold,
                    "max_boxes": args.detector_ensemble_max_boxes,
                    "score_calibration": args.detector_score_calibration,
                    "box_scale": args.detector_box_scale,
                    "min_area_ratio": args.detector_min_area_ratio,
                    "max_area_ratio": args.detector_max_area_ratio,
                    "post_nms_iou": args.detector_post_nms_iou,
                },
                "classifier_temperature": {
                    "mode": args.classifier_temperature_mode,
                    "temperatures": classifier_temperatures,
                },
                "meta": {
                    "fit_split": args.meta_fit_split,
                },
                "mlp": {
                    "fit_split": args.mlp_fit_split,
                },
                "selected": selected_records,
            },
            metrics_file,
            indent=2,
        )

    for method_name, predictions in test_predictions_by_method.items():
        method_predictions_path = output_dir / f"{method_name}_test_predictions.csv"
        predictions.to_csv(method_predictions_path, index=False)
        false_positives = predictions[(predictions["truth"] == 0) & (predictions["prediction"] == 1)]
        false_negatives = predictions[(predictions["truth"] == 1) & (predictions["prediction"] == 0)]
        false_positives.to_csv(output_dir / f"{method_name}_false_positives.csv", index=False)
        false_negatives.to_csv(output_dir / f"{method_name}_false_negatives.csv", index=False)

    for method_name, predictions in validation_predictions_by_method.items():
        method_predictions_path = output_dir / f"{method_name}_validation_predictions.csv"
        predictions.to_csv(method_predictions_path, index=False)
        false_positives = predictions[(predictions["truth"] == 0) & (predictions["prediction"] == 1)]
        false_negatives = predictions[(predictions["truth"] == 1) & (predictions["prediction"] == 0)]
        false_positives.to_csv(output_dir / f"{method_name}_validation_false_positives.csv", index=False)
        false_negatives.to_csv(output_dir / f"{method_name}_validation_false_negatives.csv", index=False)

    best_method = str(selected_table.iloc[0]["method"])
    best_predictions = test_predictions_by_method[best_method]
    best_predictions.to_csv(output_dir / "fusion_sweep_test_predictions.csv", index=False)
    if best_method in validation_predictions_by_method:
        best_validation_predictions = validation_predictions_by_method[best_method]
        best_validation_predictions.to_csv(output_dir / "fusion_sweep_validation_predictions.csv", index=False)
        best_validation_predictions[
            (best_validation_predictions["truth"] == 0) & (best_validation_predictions["prediction"] == 1)
        ].to_csv(
            output_dir / "fusion_sweep_best_validation_false_positives.csv",
            index=False,
        )
        best_validation_predictions[
            (best_validation_predictions["truth"] == 1) & (best_validation_predictions["prediction"] == 0)
        ].to_csv(
            output_dir / "fusion_sweep_best_validation_false_negatives.csv",
            index=False,
        )
    best_predictions[(best_predictions["truth"] == 0) & (best_predictions["prediction"] == 1)].to_csv(
        output_dir / "fusion_sweep_best_false_positives.csv",
        index=False,
    )
    best_predictions[(best_predictions["truth"] == 1) & (best_predictions["prediction"] == 0)].to_csv(
        output_dir / "fusion_sweep_best_false_negatives.csv",
        index=False,
    )
    save_confusion_matrix(
        output_dir / "fusion_sweep_test_confusion_matrix.png",
        selected_table.iloc[0].to_dict(),
        f"Fusion Test Confusion Matrix, method={best_method}",
    )

    print("\nBest held-out test fusion result:")
    print(selected_table.iloc[0].to_string())


if __name__ == "__main__":
    main()

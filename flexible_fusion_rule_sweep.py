#!/usr/bin/env python3

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_best_model_errors import confusion_metrics, normalize_metadata


def parse_values(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def add_metadata_groups(features, metadata_csv):
    if not metadata_csv:
        return features
    metadata_path = Path(metadata_csv).expanduser()
    if not metadata_path.exists():
        return features
    metadata = normalize_metadata(pd.read_csv(metadata_path, dtype={"patientId": str}))
    keep_columns = [
        column
        for column in ["patientId", "age_bucket", "sex_group", "view_group"]
        if column in metadata.columns
    ]
    if not keep_columns or keep_columns == ["patientId"]:
        return features
    return features.merge(metadata[keep_columns], on="patientId", how="left")


def classifier_columns(features):
    columns = []
    index = 0
    while f"classifier_{index}_probability" in features.columns:
        columns.append(f"classifier_{index}_probability")
        index += 1
    return columns


def feature_arrays(features):
    columns = classifier_columns(features)
    if not columns:
        raise ValueError("Fusion features must contain classifier_0_probability.")
    classifier = features[columns].to_numpy(dtype=np.float32)
    detector = features["detector_max_score"].to_numpy(dtype=np.float32)
    arrays = {
        "classifier_mean": classifier.mean(axis=1),
        "classifier_max": classifier.max(axis=1),
        "classifier_min": classifier.min(axis=1),
        "classifier_std": classifier.std(axis=1),
        "detector": detector,
        "detector_count": features.get("detector_count_norm", pd.Series(0.0, index=features.index)).to_numpy(
            dtype=np.float32
        ),
        "detector_area": features.get("detector_max_area_ratio", pd.Series(0.0, index=features.index)).to_numpy(
            dtype=np.float32
        ),
    }
    return arrays


def apply_rule(arrays, rule):
    mean_score = arrays["classifier_mean"]
    max_score = arrays["classifier_max"]
    min_score = arrays["classifier_min"]
    std_score = arrays["classifier_std"]
    detector = arrays["detector"]
    count = arrays["detector_count"]
    area = arrays["detector_area"]

    classifier_detector_agree = (
        (mean_score >= rule["mean_threshold"])
        & (detector >= rule["detector_threshold"])
    )
    all_classifier_agree = (
        (min_score >= rule["min_threshold"])
        & (detector >= rule["min_detector_threshold"])
    )
    high_classifier_rescue = (
        (max_score >= rule["max_rescue_threshold"])
        & (mean_score >= rule["mean_rescue_floor"])
        & (detector >= rule["rescue_detector_floor"])
    )
    high_detector_rescue = (
        (detector >= rule["detector_rescue_threshold"])
        & (mean_score >= rule["detector_rescue_classifier_floor"])
    )

    predicted = (
        classifier_detector_agree
        | all_classifier_agree
        | high_classifier_rescue
        | high_detector_rescue
    )

    low_evidence_veto = (
        (mean_score < rule["veto_mean_threshold"])
        & (detector < rule["veto_detector_threshold"])
    )
    disagreement_veto = (
        (std_score >= rule["veto_std_threshold"])
        & (min_score < rule["veto_min_threshold"])
        & (detector < rule["veto_std_detector_threshold"])
    )
    tiny_weak_box_veto = (
        (detector < rule["veto_tiny_detector_threshold"])
        & (area < rule["veto_min_area_ratio"])
        & (count <= rule["veto_count_threshold"])
    )
    predicted = predicted & ~(low_evidence_veto | disagreement_veto | tiny_weak_box_veto)
    return predicted.astype(np.int64)


def rule_key(row):
    return (row["image_f1"], row["image_precision"], row["image_recall"], row["image_accuracy"])


def sampled_rules(args, max_rules):
    rng = np.random.default_rng(args.seed)
    grids = {
        "mean_threshold": parse_values(args.mean_thresholds),
        "detector_threshold": parse_values(args.detector_thresholds),
        "min_threshold": parse_values(args.min_thresholds),
        "min_detector_threshold": parse_values(args.min_detector_thresholds),
        "max_rescue_threshold": parse_values(args.max_rescue_thresholds),
        "mean_rescue_floor": parse_values(args.mean_rescue_floors),
        "rescue_detector_floor": parse_values(args.rescue_detector_floors),
        "detector_rescue_threshold": parse_values(args.detector_rescue_thresholds),
        "detector_rescue_classifier_floor": parse_values(args.detector_rescue_classifier_floors),
        "veto_mean_threshold": parse_values(args.veto_mean_thresholds),
        "veto_detector_threshold": parse_values(args.veto_detector_thresholds),
        "veto_std_threshold": parse_values(args.veto_std_thresholds),
        "veto_min_threshold": parse_values(args.veto_min_thresholds),
        "veto_std_detector_threshold": parse_values(args.veto_std_detector_thresholds),
        "veto_tiny_detector_threshold": parse_values(args.veto_tiny_detector_thresholds),
        "veto_min_area_ratio": parse_values(args.veto_min_area_ratios),
        "veto_count_threshold": parse_values(args.veto_count_thresholds),
    }
    anchors = [
        {
            "mean_threshold": 0.62,
            "detector_threshold": 0.35,
            "min_threshold": 0.62,
            "min_detector_threshold": 0.35,
            "max_rescue_threshold": 0.82,
            "mean_rescue_floor": 0.52,
            "rescue_detector_floor": 0.00,
            "detector_rescue_threshold": 0.54,
            "detector_rescue_classifier_floor": 0.20,
            "veto_mean_threshold": 0.45,
            "veto_detector_threshold": 0.30,
            "veto_std_threshold": 0.20,
            "veto_min_threshold": 0.35,
            "veto_std_detector_threshold": 0.50,
            "veto_tiny_detector_threshold": 0.45,
            "veto_min_area_ratio": 0.0005,
            "veto_count_threshold": 0.20,
        },
        {
            "mean_threshold": 0.66,
            "detector_threshold": 0.35,
            "min_threshold": 0.54,
            "min_detector_threshold": 0.25,
            "max_rescue_threshold": 0.76,
            "mean_rescue_floor": 0.48,
            "rescue_detector_floor": 0.20,
            "detector_rescue_threshold": 0.65,
            "detector_rescue_classifier_floor": 0.30,
            "veto_mean_threshold": 0.40,
            "veto_detector_threshold": 0.20,
            "veto_std_threshold": 0.30,
            "veto_min_threshold": 0.25,
            "veto_std_detector_threshold": 0.60,
            "veto_tiny_detector_threshold": 0.35,
            "veto_min_area_ratio": 0.0000,
            "veto_count_threshold": 0.10,
        },
    ]
    seen = set()
    for rule in anchors:
        key = tuple(rule.items())
        seen.add(key)
        yield rule
    for _ in range(max(0, int(max_rules) - len(anchors))):
        rule = {name: float(rng.choice(values)) for name, values in grids.items()}
        key = tuple(rule.items())
        if key in seen:
            continue
        seen.add(key)
        yield rule


def tune_rule(features, args, min_images=25, max_rules=None):
    if len(features) < min_images:
        return None
    arrays = feature_arrays(features)
    labels = features["truth"].to_numpy(dtype=np.int64)
    best = None
    best_rule = None
    rows = []
    limit = args.max_rules if max_rules is None else max_rules
    for rule in sampled_rules(args, limit):
        predictions = apply_rule(arrays, rule)
        metrics = confusion_metrics(labels, predictions)
        row = {**rule, **metrics}
        rows.append(row)
        if best is None or rule_key(row) > rule_key(best):
            best = row
            best_rule = rule
    return best_rule, best, rows


def evaluate_rule(features, rule):
    predictions = apply_rule(feature_arrays(features), rule)
    metrics = confusion_metrics(features["truth"].to_numpy(dtype=np.int64), predictions)
    return metrics, predictions


def evaluate_global(valid, test, args):
    rule, valid_metrics, rows = tune_rule(valid, args, min_images=1, max_rules=args.max_rules)
    test_metrics, test_predictions = evaluate_rule(test, rule)
    selected = {
        "strategy": "global_flexible_rule",
        "selected_from": "validation_flexible_rule",
        **rule,
        **test_metrics,
        "validation_image_f1": valid_metrics["image_f1"],
        "validation_image_precision": valid_metrics["image_precision"],
        "validation_image_recall": valid_metrics["image_recall"],
    }
    return selected, rows, test_predictions


def evaluate_stratified(valid, test, args, group_column):
    fallback_rule, fallback_metrics, _ = tune_rule(valid, args, min_images=1)
    rules = {}
    selected_rules = []
    for group, group_valid in valid.groupby(group_column, dropna=False):
        group_key = str(group)
        result = tune_rule(group_valid, args, min_images=args.min_group_images, max_rules=args.max_group_rules)
        if result is None:
            rule = fallback_rule
            valid_metrics = fallback_metrics
            used_fallback = 1
        else:
            rule, valid_metrics, _ = result
            used_fallback = 0
        rules[group_key] = rule
        selected_rules.append(
            {
                "group_column": group_column,
                "group": group_key,
                "validation_num_images": int(len(group_valid)),
                "used_fallback": used_fallback,
                **rule,
                **valid_metrics,
            }
        )

    predictions = np.zeros(len(test), dtype=np.int64)
    for group, indices in test.groupby(group_column, dropna=False).groups.items():
        group_key = str(group)
        rule = rules.get(group_key, fallback_rule)
        group_predictions = apply_rule(feature_arrays(test.loc[indices]), rule)
        predictions[test.index.get_indexer(indices)] = group_predictions
    metrics = confusion_metrics(test["truth"].to_numpy(dtype=np.int64), predictions)
    selected = {
        "strategy": f"stratified_flexible_rule_by_{group_column}",
        "selected_from": "validation_stratified_flexible_rule",
        "group_column": group_column,
        **metrics,
    }
    return selected, selected_rules, predictions


def main():
    parser = argparse.ArgumentParser(description="Flexible validation-only fusion rule sweep.")
    parser.add_argument("--fusion-dir", required=True)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-columns", default="age_bucket,view_group,sex_group")
    parser.add_argument("--min-group-images", type=int, default=75)
    parser.add_argument("--max-rules", type=int, default=60000)
    parser.add_argument("--max-group-rules", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mean-thresholds", default="0.54,0.58,0.60,0.62,0.64,0.66,0.68")
    parser.add_argument("--detector-thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.54")
    parser.add_argument("--min-thresholds", default="0.50,0.54,0.58,0.62,0.66")
    parser.add_argument("--min-detector-thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--max-rescue-thresholds", default="0.76,0.80,0.84,0.88,0.92")
    parser.add_argument("--mean-rescue-floors", default="0.48,0.52,0.56,0.60")
    parser.add_argument("--rescue-detector-floors", default="0.00,0.20,0.30,0.40")
    parser.add_argument("--detector-rescue-thresholds", default="0.50,0.58,0.65,0.72,0.80")
    parser.add_argument("--detector-rescue-classifier-floors", default="0.20,0.30,0.40,0.50")
    parser.add_argument("--veto-mean-thresholds", default="0.40,0.45,0.50")
    parser.add_argument("--veto-detector-thresholds", default="0.20,0.30,0.40")
    parser.add_argument("--veto-std-thresholds", default="0.10,0.15,0.20,0.30")
    parser.add_argument("--veto-min-thresholds", default="0.25,0.35,0.45")
    parser.add_argument("--veto-std-detector-thresholds", default="0.40,0.50,0.60")
    parser.add_argument("--veto-tiny-detector-thresholds", default="0.35,0.45,0.55")
    parser.add_argument("--veto-min-area-ratios", default="0.0000,0.0005,0.0010")
    parser.add_argument("--veto-count-thresholds", default="0.10,0.20,0.30")
    args = parser.parse_args()

    fusion_dir = Path(args.fusion_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = add_metadata_groups(pd.read_csv(fusion_dir / "fusion_validation_features.csv"), args.metadata_csv)
    test = add_metadata_groups(pd.read_csv(fusion_dir / "fusion_test_features.csv"), args.metadata_csv)

    selected_rows = []
    selected_rule_rows = []
    global_selected, global_rows, global_predictions = evaluate_global(valid, test, args)
    selected_rows.append(global_selected)
    pd.DataFrame(global_rows).sort_values(["image_f1", "image_precision"], ascending=[False, False]).to_csv(
        output_dir / "global_validation_rules.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "patientId": test["patientId"],
            "truth": test["truth"],
            "prediction": global_predictions,
        }
    ).to_csv(output_dir / "global_test_predictions.csv", index=False)

    best = global_selected
    best_predictions = global_predictions
    for group_column in [item.strip() for item in args.group_columns.split(",") if item.strip()]:
        if group_column not in valid.columns or group_column not in test.columns:
            continue
        selected, rule_rows, predictions = evaluate_stratified(valid, test, args, group_column)
        selected_rows.append(selected)
        selected_rule_rows.extend(rule_rows)
        if rule_key(selected) > rule_key(best):
            best = selected
            best_predictions = predictions

    pd.DataFrame(selected_rows).sort_values(["image_f1", "image_precision"], ascending=[False, False]).to_csv(
        output_dir / "flexible_rule_test_metrics.csv",
        index=False,
    )
    pd.DataFrame(selected_rule_rows).to_csv(output_dir / "stratified_selected_rules.csv", index=False)
    prediction_table = pd.DataFrame(
        {
            "patientId": test["patientId"],
            "truth": test["truth"],
            "prediction": best_predictions,
        }
    )
    prediction_table.to_csv(output_dir / "best_test_predictions.csv", index=False)
    prediction_table[(prediction_table["truth"] == 0) & (prediction_table["prediction"] == 1)].to_csv(
        output_dir / "best_false_positives.csv",
        index=False,
    )
    prediction_table[(prediction_table["truth"] == 1) & (prediction_table["prediction"] == 0)].to_csv(
        output_dir / "best_false_negatives.csv",
        index=False,
    )
    (output_dir / "flexible_rule_summary.json").write_text(
        json.dumps({"best": best, "all": selected_rows}, indent=2),
        encoding="utf-8",
    )
    print(pd.Series(best).to_string())


if __name__ == "__main__":
    main()

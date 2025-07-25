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
    metadata = normalize_metadata(pd.read_csv(metadata_csv, dtype={"patientId": str}))
    keep_columns = [
        column
        for column in [
            "patientId",
            "PatientAge",
            "PatientSex",
            "ViewPosition",
            "age_bucket",
            "sex_group",
            "view_group",
        ]
        if column in metadata.columns
    ]
    return features.merge(metadata[keep_columns], on="patientId", how="left")


def decision_array(features, rule):
    classifier_min = features["classifier_min_probability"].to_numpy(dtype=np.float32)
    classifier_mean = features["classifier_mean_probability"].to_numpy(dtype=np.float32)
    detector_score = features["detector_max_score"].to_numpy(dtype=np.float32)
    return (
        ((classifier_min >= rule["classifier_agree_threshold"]) & (detector_score >= rule["detector_required_threshold"]))
        | (classifier_mean >= rule["classifier_override_threshold"])
        | (
            (detector_score >= rule["detector_override_threshold"])
            & (classifier_mean <= rule["detector_classifier_max_threshold"])
        )
    ).astype(np.int64)


def tune_rule(features, grids, min_images=25):
    if len(features) < min_images:
        return None
    labels = features["truth"].to_numpy(dtype=np.int64)
    best = None
    for values in product(
        grids["classifier_agree_thresholds"],
        grids["detector_required_thresholds"],
        grids["classifier_override_thresholds"],
        grids["detector_override_thresholds"],
        grids["detector_classifier_max_thresholds"],
    ):
        rule = {
            "classifier_agree_threshold": values[0],
            "detector_required_threshold": values[1],
            "classifier_override_threshold": values[2],
            "detector_override_threshold": values[3],
            "detector_classifier_max_threshold": values[4],
        }
        predicted = decision_array(features, rule)
        metrics = confusion_metrics(labels, predicted)
        row = {**rule, **metrics}
        if best is None or (
            row["image_f1"],
            row["image_precision"],
            row["image_recall"],
        ) > (
            best["image_f1"],
            best["image_precision"],
            best["image_recall"],
        ):
            best = row
    return best


def apply_group_rules(features, group_column, rules, fallback_rule):
    predictions = np.zeros(len(features), dtype=np.int64)
    applied_groups = []
    for group, indices in features.groupby(group_column, dropna=False).groups.items():
        group_key = str(group)
        rule = rules.get(group_key, fallback_rule)
        group_predictions = decision_array(features.loc[indices], rule)
        predictions[features.index.get_indexer(indices)] = group_predictions
        applied_groups.extend([group_key] * len(indices))
    return predictions, applied_groups


def evaluate_group_strategy(valid_features, test_features, group_column, grids, min_group_images):
    fallback_rule = tune_rule(valid_features, grids, min_images=1)
    if fallback_rule is None:
        raise RuntimeError("Could not tune fallback rule.")

    rows = []
    rules = {}
    for group, group_valid in valid_features.groupby(group_column, dropna=False):
        group_key = str(group)
        selected = tune_rule(group_valid, grids, min_images=min_group_images)
        if selected is None:
            selected = fallback_rule
            selected = {**selected, "used_fallback": 1}
        else:
            selected = {**selected, "used_fallback": 0}
        rules[group_key] = selected
        rows.append(
            {
                "group_column": group_column,
                "group": group_key,
                "validation_num_images": int(len(group_valid)),
                **selected,
            }
        )

    test_pred, applied_groups = apply_group_rules(test_features, group_column, rules, fallback_rule)
    metrics = confusion_metrics(test_features["truth"], test_pred)
    predictions = test_features[["patientId", "truth", group_column]].copy()
    predictions["prediction"] = test_pred
    predictions["applied_group"] = applied_groups
    return rows, metrics, predictions, fallback_rule


def main():
    parser = argparse.ArgumentParser(description="Tune separate precision-gate rules by metadata subgroup.")
    parser.add_argument("--fusion-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-columns", default="view_group,sex_group,age_bucket")
    parser.add_argument("--min-group-images", type=int, default=50)
    parser.add_argument("--gate-classifier-agree-thresholds", default="0.54,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.75")
    parser.add_argument("--gate-detector-required-thresholds", default="0.30,0.35,0.40,0.45,0.50,0.54,0.58,0.62")
    parser.add_argument("--gate-classifier-override-thresholds", default="0.78,0.80,0.82,0.84,0.86,0.88,0.90,0.92")
    parser.add_argument("--gate-detector-override-thresholds", default="0.50,0.54,0.58,0.62,0.66,0.70,0.75,0.80")
    parser.add_argument("--gate-detector-classifier-max-thresholds", default="0.15,0.20,0.25,0.30,0.35,0.40,0.50,1.00")
    args = parser.parse_args()

    fusion_dir = Path(args.fusion_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = add_metadata_groups(pd.read_csv(fusion_dir / "fusion_validation_features.csv"), args.metadata_csv)
    test = add_metadata_groups(pd.read_csv(fusion_dir / "fusion_test_features.csv"), args.metadata_csv)

    grids = {
        "classifier_agree_thresholds": parse_values(args.gate_classifier_agree_thresholds),
        "detector_required_thresholds": parse_values(args.gate_detector_required_thresholds),
        "classifier_override_thresholds": parse_values(args.gate_classifier_override_thresholds),
        "detector_override_thresholds": parse_values(args.gate_detector_override_thresholds),
        "detector_classifier_max_thresholds": parse_values(args.gate_detector_classifier_max_thresholds),
    }

    strategy_rows = []
    selected_rule_rows = []
    best = None
    best_predictions = None
    for group_column in [item.strip() for item in args.group_columns.split(",") if item.strip()]:
        rule_rows, test_metrics, predictions, fallback_rule = evaluate_group_strategy(
            valid,
            test,
            group_column,
            grids,
            args.min_group_images,
        )
        strategy = {
            "strategy": f"stratified_by_{group_column}",
            "group_column": group_column,
            **test_metrics,
        }
        strategy_rows.append(strategy)
        for row in rule_rows:
            selected_rule_rows.append(row)
        predictions.to_csv(output_dir / f"{group_column}_test_predictions.csv", index=False)
        predictions[(predictions["truth"] == 0) & (predictions["prediction"] == 1)].to_csv(
            output_dir / f"{group_column}_false_positives.csv",
            index=False,
        )
        predictions[(predictions["truth"] == 1) & (predictions["prediction"] == 0)].to_csv(
            output_dir / f"{group_column}_false_negatives.csv",
            index=False,
        )
        if best is None or (strategy["image_f1"], strategy["image_precision"]) > (
            best["image_f1"],
            best["image_precision"],
        ):
            best = strategy
            best_predictions = predictions

    pd.DataFrame(strategy_rows).sort_values(["image_f1", "image_precision"], ascending=[False, False]).to_csv(
        output_dir / "stratified_strategy_test_metrics.csv",
        index=False,
    )
    pd.DataFrame(selected_rule_rows).to_csv(output_dir / "stratified_selected_rules.csv", index=False)
    best_predictions.to_csv(output_dir / "stratified_best_test_predictions.csv", index=False)
    best_predictions[(best_predictions["truth"] == 0) & (best_predictions["prediction"] == 1)].to_csv(
        output_dir / "stratified_best_false_positives.csv",
        index=False,
    )
    best_predictions[(best_predictions["truth"] == 1) & (best_predictions["prediction"] == 0)].to_csv(
        output_dir / "stratified_best_false_negatives.csv",
        index=False,
    )

    summary = {"best_strategy": best, "all_strategies": strategy_rows}
    (output_dir / "stratified_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest stratified strategy:")
    print(pd.Series(best).to_string())
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()

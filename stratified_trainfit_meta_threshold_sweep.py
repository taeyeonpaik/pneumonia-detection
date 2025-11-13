#!/usr/bin/env python3

import argparse
import json
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


ID_COLUMNS = {"patientId", "truth"}


def parse_values(text, cast=float):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def image_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "image_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "image_precision": precision,
        "image_recall": recall,
        "image_f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_images": int(tp + tn + fp + fn),
    }


def parse_age(value):
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)([YMWD]?)$", text)
    if match is None:
        return np.nan
    amount = float(match.group(1))
    unit = match.group(2) or "Y"
    return amount * {"Y": 1.0, "M": 1.0 / 12.0, "W": 1.0 / 52.0, "D": 1.0 / 365.25}[unit]


def age_bucket(age):
    if not np.isfinite(age):
        return "unknown"
    if age <= 18:
        return "<=18"
    if age <= 40:
        return "19-40"
    if age <= 60:
        return "41-60"
    if age <= 80:
        return "61-80"
    return "81+"


def normalize_view(value):
    text = str(value or "").strip().upper().replace(" ", "")
    if text.startswith("AP"):
        return "AP"
    if text.startswith("PA"):
        return "PA"
    return "unknown"


def normalize_sex(value):
    text = str(value or "").strip().upper()
    if text == "F":
        return "F"
    if text == "M":
        return "M"
    return "unknown"


def metadata_features(metadata_csv):
    table = pd.read_csv(metadata_csv, dtype={"patientId": str})
    if "patientId" not in table.columns:
        raise ValueError(f"Expected patientId column in {metadata_csv}")
    table = table.drop_duplicates("patientId", keep="last").copy()
    age_source = "PatientAge" if "PatientAge" in table.columns else "age"
    sex_source = "PatientSex" if "PatientSex" in table.columns else "sex"
    view_source = "ViewPosition" if "ViewPosition" in table.columns else "view"

    table["age_years"] = table[age_source].map(parse_age) if age_source in table.columns else np.nan
    table["age_bucket"] = table["age_years"].map(age_bucket)
    table["sex_group"] = table[sex_source].map(normalize_sex) if sex_source in table.columns else "unknown"
    table["view_group"] = table[view_source].map(normalize_view) if view_source in table.columns else "unknown"
    table["view_age_bucket"] = table["view_group"].astype(str) + "_" + table["age_bucket"].astype(str)
    table["sex_age_bucket"] = table["sex_group"].astype(str) + "_" + table["age_bucket"].astype(str)
    table["view_sex_group"] = table["view_group"].astype(str) + "_" + table["sex_group"].astype(str)

    table["age_scaled"] = table["age_years"].clip(lower=0, upper=100).fillna(0.0) / 100.0
    table["age_missing"] = (~np.isfinite(table["age_years"])).astype(float)
    table["sex_male"] = (table["sex_group"] == "M").astype(float)
    table["sex_female"] = (table["sex_group"] == "F").astype(float)
    table["sex_unknown"] = (table["sex_group"] == "unknown").astype(float)
    table["view_ap"] = (table["view_group"] == "AP").astype(float)
    table["view_pa"] = (table["view_group"] == "PA").astype(float)
    table["view_unknown"] = (table["view_group"] == "unknown").astype(float)

    return table[
        [
            "patientId",
            "age_bucket",
            "sex_group",
            "view_group",
            "view_age_bucket",
            "sex_age_bucket",
            "view_sex_group",
            "age_scaled",
            "age_missing",
            "sex_male",
            "sex_female",
            "sex_unknown",
            "view_ap",
            "view_pa",
            "view_unknown",
        ]
    ]


def add_metadata(features, metadata):
    merged = features.merge(metadata, on="patientId", how="left")
    for column in ["age_bucket", "sex_group", "view_group", "view_age_bucket", "sex_age_bucket", "view_sex_group"]:
        merged[column] = merged[column].fillna("unknown")
    for column in ["age_scaled", "age_missing", "sex_male", "sex_female", "sex_unknown", "view_ap", "view_pa", "view_unknown"]:
        merged[column] = merged[column].fillna(0.0)
    return merged


def classifier_probability_columns(features):
    columns = []
    index = 0
    while f"classifier_{index}_probability" in features.columns:
        columns.append(f"classifier_{index}_probability")
        index += 1
    return columns


def feature_columns_for_mode(features, mode):
    classifier_columns = classifier_probability_columns(features)
    compact = [
        "classifier_mean_probability",
        "classifier_max_probability",
        "classifier_min_probability",
        "classifier_std_probability",
        *classifier_columns,
        "detector_max_score",
        "detector_second_score",
        "detector_mean_score",
        "detector_count_norm",
        "detector_max_area_ratio",
        "detector_total_area_ratio",
        "detector_top2_gap",
        "detector_top_score_area",
    ]
    metadata_columns = [
        "age_scaled",
        "age_missing",
        "sex_male",
        "sex_female",
        "sex_unknown",
        "view_ap",
        "view_pa",
        "view_unknown",
    ]
    if mode == "all":
        return [
            column
            for column in features.columns
            if column not in ID_COLUMNS
            and not column.endswith("_group")
            and not column.endswith("_bucket")
            and column not in {"age_years"}
            and pd.api.types.is_numeric_dtype(features[column])
        ]
    if mode == "compact":
        return [column for column in compact if column in features.columns]
    if mode == "compact_metadata":
        return [column for column in [*compact, *metadata_columns] if column in features.columns]
    if mode == "classifier_detector_only":
        keep = [
            "classifier_mean_probability",
            "classifier_max_probability",
            "classifier_min_probability",
            "classifier_std_probability",
            *classifier_columns,
            "detector_max_score",
        ]
        return [column for column in keep if column in features.columns]
    if mode == "classifier_detector_metadata":
        keep = [
            "classifier_mean_probability",
            "classifier_max_probability",
            "classifier_min_probability",
            "classifier_std_probability",
            *classifier_columns,
            "detector_max_score",
            *metadata_columns,
        ]
        return [column for column in keep if column in features.columns]
    raise ValueError(f"Unknown feature mode: {mode}")


def standardize(train, valid, test):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (valid - mean) / std, (test - mean) / std, mean, std


class LogisticFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.linear(features).flatten()


def train_model(features, labels, pos_weight, l2, lr, epochs, seed):
    torch.manual_seed(int(seed))
    model = LogisticFusion(features.shape[1])
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if l2 > 0:
            loss = loss + float(l2) * model.linear.weight.pow(2).sum()
        loss.backward()
        optimizer.step()
    return model


def predict(model, features):
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float32))
    return torch.sigmoid(logits).detach().cpu().numpy()


def threshold_candidates(scores, manual_thresholds):
    quantiles = np.quantile(scores, np.linspace(0.02, 0.98, 193))
    values = [*manual_thresholds, *quantiles.tolist()]
    return sorted({round(float(value), 5) for value in values if np.isfinite(value)})


def choose_threshold(labels, scores, thresholds):
    best = None
    best_prediction = None
    for threshold in thresholds:
        prediction = (scores >= threshold).astype(np.int64)
        metrics = image_metrics(labels, prediction)
        row = {"threshold": float(threshold), **metrics}
        key = (row["image_f1"], row["image_precision"], row["image_recall"])
        if best is None or key > (best["image_f1"], best["image_precision"], best["image_recall"]):
            best = row
            best_prediction = prediction
    return best, best_prediction


def tune_group_thresholds(valid, scores, group_column, manual_thresholds, min_group_images, min_group_positives):
    labels = valid["truth"].to_numpy(dtype=np.int64)
    global_rule, _ = choose_threshold(labels, scores, threshold_candidates(scores, manual_thresholds))
    predictions = np.zeros(len(valid), dtype=np.int64)
    group_rules = []
    for group, indices in valid.groupby(group_column, dropna=False).groups.items():
        group_indices = np.asarray(valid.index.get_indexer(indices), dtype=np.int64)
        group_labels = labels[group_indices]
        group_scores = scores[group_indices]
        use_fallback = len(group_indices) < min_group_images or int(group_labels.sum()) < min_group_positives
        if use_fallback:
            rule = {**global_rule, "group": str(group), "used_fallback": 1}
        else:
            group_thresholds = threshold_candidates(group_scores, manual_thresholds)
            selected, _ = choose_threshold(group_labels, group_scores, group_thresholds)
            rule = {**selected, "group": str(group), "used_fallback": 0}
        predictions[group_indices] = (group_scores >= rule["threshold"]).astype(np.int64)
        group_rules.append(
            {
                "group_column": group_column,
                "group": str(group),
                "validation_num_images": int(len(group_indices)),
                "validation_num_positives": int(group_labels.sum()),
                **rule,
            }
        )
    strategy_metrics = image_metrics(labels, predictions)
    return {"strategy": f"stratified_by_{group_column}", "group_column": group_column, **strategy_metrics}, group_rules


def apply_group_thresholds(test, scores, group_column, group_rules, fallback_threshold):
    rule_by_group = {row["group"]: float(row["threshold"]) for row in group_rules}
    predictions = np.zeros(len(test), dtype=np.int64)
    thresholds = np.zeros(len(test), dtype=np.float32)
    for group, indices in test.groupby(group_column, dropna=False).groups.items():
        group_indices = np.asarray(test.index.get_indexer(indices), dtype=np.int64)
        threshold = rule_by_group.get(str(group), fallback_threshold)
        predictions[group_indices] = (scores[group_indices] >= threshold).astype(np.int64)
        thresholds[group_indices] = threshold
    return predictions, thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Train logistic fusion from saved train features, then tune global and metadata-stratified thresholds."
    )
    parser.add_argument("--fusion-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-modes", default="compact,compact_metadata,classifier_detector_only,classifier_detector_metadata,all")
    parser.add_argument("--group-columns", default="age_bucket,view_group,sex_group,view_age_bucket,sex_age_bucket,view_sex_group")
    parser.add_argument("--pos-weights", default="1.5,1.75,2.0,2.25,2.5,2.75,3.0")
    parser.add_argument("--l2-values", default="0.0,0.0001,0.001,0.003,0.006,0.01,0.02")
    parser.add_argument("--lrs", default="0.01,0.02")
    parser.add_argument("--fit-seeds", default="42,1337")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--thresholds", default="0.30,0.35,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70")
    parser.add_argument("--min-group-images", type=int, default=80)
    parser.add_argument("--min-group-positives", type=int, default=12)
    args = parser.parse_args()

    fusion_dir = Path(args.fusion_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = metadata_features(args.metadata_csv)
    train = add_metadata(pd.read_csv(fusion_dir / "fusion_train_features.csv", dtype={"patientId": str}), metadata)
    valid = add_metadata(pd.read_csv(fusion_dir / "fusion_validation_features.csv", dtype={"patientId": str}), metadata)
    test = add_metadata(pd.read_csv(fusion_dir / "fusion_test_features.csv", dtype={"patientId": str}), metadata)

    feature_modes = [item.strip() for item in args.feature_modes.split(",") if item.strip()]
    group_columns = [item.strip() for item in args.group_columns.split(",") if item.strip()]
    pos_weights = parse_values(args.pos_weights)
    l2_values = parse_values(args.l2_values)
    lrs = parse_values(args.lrs)
    fit_seeds = parse_values(args.fit_seeds, int)
    manual_thresholds = parse_values(args.thresholds)

    configs = list(product(feature_modes, pos_weights, l2_values, lrs, fit_seeds))
    print(f"Sweeping {len(configs)} train-fit meta models x {1 + len(group_columns)} threshold strategies", flush=True)

    y_train = train["truth"].to_numpy(dtype=np.float32)
    y_valid = valid["truth"].to_numpy(dtype=np.int64)
    y_test = test["truth"].to_numpy(dtype=np.int64)

    validation_rows = []
    best_validation_key = None
    best_payload = None
    best_group_rules = []
    best_test_predictions = None

    progress = tqdm(configs, desc="train-fit stratified meta", leave=True, file=sys.stdout)
    for feature_mode, pos_weight, l2, lr, fit_seed in progress:
        columns = feature_columns_for_mode(train, feature_mode)
        if not columns:
            continue
        x_train_raw = train[columns].to_numpy(dtype=np.float32)
        x_valid_raw = valid[columns].to_numpy(dtype=np.float32)
        x_test_raw = test[columns].to_numpy(dtype=np.float32)
        x_train, x_valid, x_test, mean, std = standardize(x_train_raw, x_valid_raw, x_test_raw)
        model = train_model(
            x_train,
            y_train,
            pos_weight=float(pos_weight),
            l2=float(l2),
            lr=float(lr),
            epochs=args.epochs,
            seed=int(fit_seed),
        )
        valid_scores = predict(model, x_valid)
        test_scores = predict(model, x_test)

        global_rule, global_valid_pred = choose_threshold(
            y_valid,
            valid_scores,
            threshold_candidates(valid_scores, manual_thresholds),
        )
        strategies = [
            (
                {
                    "strategy": "global",
                    "group_column": "",
                    **global_rule,
                },
                [],
                global_rule["threshold"],
            )
        ]
        for group_column in group_columns:
            strategy, group_rules = tune_group_thresholds(
                valid,
                valid_scores,
                group_column,
                manual_thresholds,
                min_group_images=args.min_group_images,
                min_group_positives=args.min_group_positives,
            )
            strategies.append((strategy, group_rules, global_rule["threshold"]))

        for strategy, group_rules, fallback_threshold in strategies:
            validation_row = {
                "feature_mode": feature_mode,
                "pos_weight": float(pos_weight),
                "l2": float(l2),
                "lr": float(lr),
                "fit_seed": int(fit_seed),
                "num_features": len(columns),
                **{f"validation_{key}": value for key, value in strategy.items() if key.startswith("image_") or key in {"tn", "fp", "fn", "tp", "num_images"}},
                "strategy": strategy["strategy"],
                "group_column": strategy.get("group_column", ""),
                "global_threshold": float(global_rule["threshold"]),
            }
            validation_rows.append(validation_row)
            validation_key = (
                validation_row["validation_image_f1"],
                validation_row["validation_image_precision"],
                validation_row["validation_image_recall"],
            )
            if best_validation_key is None or validation_key > best_validation_key:
                if strategy["strategy"] == "global":
                    test_pred = (test_scores >= global_rule["threshold"]).astype(np.int64)
                    applied_thresholds = np.full(len(test), float(global_rule["threshold"]), dtype=np.float32)
                else:
                    test_pred, applied_thresholds = apply_group_thresholds(
                        test,
                        test_scores,
                        strategy["group_column"],
                        group_rules,
                        fallback_threshold,
                    )
                test_metrics = image_metrics(y_test, test_pred)
                best_validation_key = validation_key
                best_payload = {
                    "method": "stratified_trainfit_meta",
                    "feature_mode": feature_mode,
                    "strategy": strategy["strategy"],
                    "group_column": strategy.get("group_column", ""),
                    "pos_weight": float(pos_weight),
                    "l2": float(l2),
                    "lr": float(lr),
                    "fit_seed": int(fit_seed),
                    "num_features": len(columns),
                    "global_threshold": float(global_rule["threshold"]),
                    **{f"validation_{key}": value for key, value in strategy.items() if key.startswith("image_") or key in {"tn", "fp", "fn", "tp", "num_images"}},
                    **test_metrics,
                    "feature_columns": columns,
                }
                best_group_rules = group_rules
                best_test_predictions = test[["patientId", "truth"]].copy()
                best_test_predictions["score"] = test_scores
                best_test_predictions["applied_threshold"] = applied_thresholds
                best_test_predictions["prediction"] = test_pred
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "feature_columns": columns,
                        "feature_mean": mean.tolist(),
                        "feature_std": std.tolist(),
                        "payload": best_payload,
                        "group_rules": best_group_rules,
                    },
                    output_dir / "best_stratified_trainfit_meta.pt",
                )

    if best_payload is None:
        raise RuntimeError("No model configurations were evaluated.")

    validation_table = pd.DataFrame(validation_rows).sort_values(
        ["validation_image_f1", "validation_image_precision", "validation_image_recall"],
        ascending=[False, False, False],
    )
    validation_table.to_csv(output_dir / "stratified_meta_validation_grid.csv", index=False)
    pd.DataFrame([best_payload]).drop(columns=["feature_columns"]).to_csv(
        output_dir / "stratified_meta_test_metrics.csv",
        index=False,
    )
    pd.DataFrame(best_group_rules).to_csv(output_dir / "stratified_meta_selected_group_rules.csv", index=False)
    best_test_predictions.to_csv(output_dir / "stratified_meta_best_test_predictions.csv", index=False)
    best_test_predictions[(best_test_predictions["truth"] == 0) & (best_test_predictions["prediction"] == 1)].to_csv(
        output_dir / "stratified_meta_false_positives.csv",
        index=False,
    )
    best_test_predictions[(best_test_predictions["truth"] == 1) & (best_test_predictions["prediction"] == 0)].to_csv(
        output_dir / "stratified_meta_false_negatives.csv",
        index=False,
    )
    (output_dir / "stratified_meta_summary.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    (output_dir / "STRATIFIED_TRAINFIT_META_COMPLETE.txt").write_text(
        f"F1: {best_payload['image_f1']}\n"
        f"Precision: {best_payload['image_precision']}\n"
        f"Recall: {best_payload['image_recall']}\n"
        f"TN/FP/FN/TP: {best_payload['tn']}/{best_payload['fp']}/{best_payload['fn']}/{best_payload['tp']}\n"
        f"Validation F1: {best_payload['validation_image_f1']}\n"
        f"Strategy: {best_payload['strategy']}\n"
        f"Group column: {best_payload['group_column']}\n"
        f"Feature mode: {best_payload['feature_mode']}\n",
        encoding="utf-8",
    )
    printable = pd.Series({k: v for k, v in best_payload.items() if k != "feature_columns"})
    print("\nBest stratified train-fit meta result:")
    print(printable.to_string())


if __name__ == "__main__":
    main()

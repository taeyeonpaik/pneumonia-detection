#!/usr/bin/env python3

import argparse
import json
import pickle
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required for tabular meta-fusion. Install it with "
        "`/opt/pytorch/bin/python -m pip install scikit-learn`."
    ) from exc


def parse_values(text, cast=float):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


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


def parse_patient_age(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().upper()
    if not text:
        return np.nan
    match = re.match(r"^(\d+(?:\.\d+)?)([YMWD]?)$", text)
    if match is None:
        return np.nan
    amount = float(match.group(1))
    unit = match.group(2) or "Y"
    return amount * {"Y": 1.0, "M": 1.0 / 12.0, "W": 1.0 / 52.0, "D": 1.0 / 365.25}[unit]


def normalize_metadata(metadata):
    table = metadata.copy()
    if "PatientAge" in table.columns:
        ages = table["PatientAge"].map(parse_patient_age)
    elif "age_scaled" in table.columns:
        ages = pd.to_numeric(table["age_scaled"], errors="coerce") * 100.0
    else:
        ages = pd.Series(np.nan, index=table.index)

    table["age_years"] = pd.to_numeric(ages, errors="coerce").clip(lower=0, upper=100)
    table["age_scaled_meta"] = (table["age_years"] / 100.0).fillna(0.0)
    table["age_missing_meta"] = table["age_years"].isna().astype(float)
    table["age_bucket_meta"] = pd.cut(
        table["age_years"],
        bins=[-np.inf, 18, 40, 60, 80, np.inf],
        labels=["age_le18", "age_19_40", "age_41_60", "age_61_80", "age_81_plus"],
    ).astype(str)
    table.loc[table["age_years"].isna(), "age_bucket_meta"] = "age_unknown"

    sex = table.get("PatientSex", pd.Series("unknown", index=table.index)).fillna("unknown").astype(str).str.upper()
    table["sex_group_meta"] = sex.replace("", "unknown")

    view = table.get("ViewPosition", pd.Series("unknown", index=table.index)).fillna("unknown").astype(str).str.upper()
    view = view.str.replace(" ", "", regex=False).replace("", "unknown")
    table["view_group_meta"] = np.where(view.str.startswith("AP"), "AP", np.where(view.str.startswith("PA"), "PA", "unknown"))

    for column in [
        "age_scaled",
        "age_missing",
        "sex_male",
        "sex_female",
        "sex_other_unknown",
        "view_ap",
        "view_pa",
        "view_other_unknown",
    ]:
        if column not in table.columns:
            table[column] = np.nan
        table[f"meta_{column}"] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)

    keep = [
        "patientId",
        "age_years",
        "age_scaled_meta",
        "age_missing_meta",
        "age_bucket_meta",
        "sex_group_meta",
        "view_group_meta",
        "meta_age_scaled",
        "meta_age_missing",
        "meta_sex_male",
        "meta_sex_female",
        "meta_sex_other_unknown",
        "meta_view_ap",
        "meta_view_pa",
        "meta_view_other_unknown",
    ]
    return table[keep].drop_duplicates("patientId", keep="last")


def classifier_probability_columns(features):
    columns = []
    index = 0
    while f"classifier_{index}_probability" in features.columns:
        columns.append(f"classifier_{index}_probability")
        index += 1
    return columns


def add_metadata(features, metadata):
    if metadata is None:
        return features
    merged = features.merge(metadata, on="patientId", how="left")
    for column in ["age_bucket_meta", "sex_group_meta", "view_group_meta"]:
        if column in merged.columns:
            merged[column] = merged[column].fillna("unknown")
    numeric_meta = [column for column in merged.columns if column.startswith("meta_") or column in {"age_years", "age_scaled_meta", "age_missing_meta"}]
    for column in numeric_meta:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def add_engineered_features(features):
    table = features.copy()
    classifier_cols = classifier_probability_columns(table)
    detector_cols = [
        "detector_max_score",
        "detector_second_score",
        "detector_mean_score",
        "detector_count_norm",
        "detector_max_area_ratio",
        "detector_total_area_ratio",
        "detector_top2_gap",
        "detector_top_score_area",
    ]
    detector_cols = [column for column in detector_cols if column in table.columns]

    for column in classifier_cols + detector_cols + [
        "classifier_mean_probability",
        "classifier_max_probability",
        "classifier_min_probability",
        "classifier_std_probability",
    ]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)

    if classifier_cols:
        cls = table[classifier_cols].to_numpy(dtype=np.float32)
        table["classifier_range_probability"] = cls.max(axis=1) - cls.min(axis=1)
        table["classifier_vote_050"] = (cls >= 0.50).mean(axis=1)
        table["classifier_vote_065"] = (cls >= 0.65).mean(axis=1)
        for threshold in [0.50, 0.60, 0.70, 0.80]:
            table[f"classifier_any_ge_{str(threshold).replace('.', '_')}"] = (cls >= threshold).any(axis=1).astype(float)

    if "detector_max_score" in table.columns:
        for threshold in [0.35, 0.50, 0.62, 0.72, 0.80]:
            table[f"detector_ge_{str(threshold).replace('.', '_')}"] = (
                table["detector_max_score"] >= threshold
            ).astype(float)

    key_classifier_cols = [
        column
        for column in [
            "classifier_mean_probability",
            "classifier_max_probability",
            "classifier_min_probability",
            *classifier_cols,
        ]
        if column in table.columns
    ]
    key_detector_cols = [
        column
        for column in [
            "detector_max_score",
            "detector_mean_score",
            "detector_count_norm",
            "detector_max_area_ratio",
            "detector_top_score_area",
        ]
        if column in table.columns
    ]
    for classifier_column in key_classifier_cols:
        for detector_column in key_detector_cols:
            table[f"{classifier_column}_x_{detector_column}"] = table[classifier_column] * table[detector_column]
            table[f"{classifier_column}_minus_{detector_column}"] = table[classifier_column] - table[detector_column]

    for meta_column in [
        "meta_view_ap",
        "meta_view_pa",
        "meta_sex_male",
        "meta_sex_female",
        "age_scaled_meta",
    ]:
        if meta_column in table.columns:
            for signal_column in key_classifier_cols[:4] + key_detector_cols[:3]:
                table[f"{signal_column}_x_{meta_column}"] = table[signal_column] * table[meta_column]

    categorical_columns = [column for column in ["age_bucket_meta", "sex_group_meta", "view_group_meta"] if column in table.columns]
    if categorical_columns:
        table = pd.get_dummies(table, columns=categorical_columns, dummy_na=False)
    return table


def feature_columns_for_mode(features, mode):
    drop = {"patientId", "truth"}
    all_numeric = [
        column
        for column in features.columns
        if column not in drop and pd.api.types.is_numeric_dtype(features[column])
    ]
    classifier_cols = classifier_probability_columns(features)
    compact = [
        "classifier_mean_probability",
        "classifier_max_probability",
        "classifier_min_probability",
        "classifier_std_probability",
        "classifier_range_probability",
        "classifier_vote_050",
        "classifier_vote_065",
        *classifier_cols,
        "detector_max_score",
        "detector_second_score",
        "detector_mean_score",
        "detector_count_norm",
        "detector_max_area_ratio",
        "detector_total_area_ratio",
        "detector_top2_gap",
        "detector_top_score_area",
        "meta_age_scaled",
        "meta_age_missing",
        "meta_sex_male",
        "meta_sex_female",
        "meta_view_ap",
        "meta_view_pa",
    ]
    if mode == "all":
        return all_numeric
    if mode == "compact":
        return [column for column in compact if column in features.columns]
    if mode == "signals_only":
        return [
            column
            for column in compact
            if column in features.columns and not column.startswith("meta_")
        ]
    raise ValueError(f"Unknown feature mode: {mode}")


def align_feature_frames(train, valid, test):
    train, valid = train.align(valid, join="outer", axis=1, fill_value=0)
    train, test = train.align(test, join="outer", axis=1, fill_value=0)
    valid, test = valid.align(test, join="outer", axis=1, fill_value=0)
    train, valid = train.align(valid, join="outer", axis=1, fill_value=0)
    return train, valid, test


def threshold_grid(scores, manual_thresholds):
    score_thresholds = np.quantile(scores, np.linspace(0.02, 0.98, 241))
    values = sorted(set(round(float(value), 5) for value in list(manual_thresholds) + score_thresholds.tolist()))
    return values


def build_model_specs(args):
    specs = []
    for learning_rate, max_leaf_nodes, min_samples_leaf, l2, max_iter in product(
        parse_values(args.hgb_learning_rates),
        parse_values(args.hgb_max_leaf_nodes, int),
        parse_values(args.hgb_min_samples_leaf, int),
        parse_values(args.hgb_l2_values),
        parse_values(args.hgb_max_iters, int),
    ):
        specs.append(
            (
                f"hgb_lr{learning_rate}_leaf{max_leaf_nodes}_min{min_samples_leaf}_l2{l2}_it{max_iter}",
                "hgb",
                {
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                    "min_samples_leaf": min_samples_leaf,
                    "l2_regularization": l2,
                    "max_iter": max_iter,
                },
            )
        )
    for n_estimators, max_depth, min_samples_leaf, max_features in product(
        parse_values(args.et_n_estimators, int),
        parse_values(args.et_max_depth, int),
        parse_values(args.et_min_samples_leaf, int),
        args.et_max_features.split(","),
    ):
        specs.append(
            (
                f"et_n{n_estimators}_d{max_depth}_min{min_samples_leaf}_mf{max_features}",
                "extra_trees",
                {
                    "n_estimators": n_estimators,
                    "max_depth": None if max_depth <= 0 else max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features.strip(),
                },
            )
        )
    return specs


def fit_predict(model_type, params, x_train, y_train, x_valid, x_test, pos_weight, seed):
    sample_weight = np.where(y_train == 1, float(pos_weight), 1.0)
    if model_type == "hgb":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=seed,
                        early_stopping=True,
                        validation_fraction=0.15,
                        scoring="loss",
                        **params,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train, model__sample_weight=sample_weight)
    elif model_type == "extra_trees":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    ExtraTreesClassifier(
                        random_state=seed,
                        n_jobs=-1,
                        class_weight={0: 1.0, 1: float(pos_weight)},
                        **params,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
    else:
        raise ValueError(model_type)
    return model, model.predict_proba(x_valid)[:, 1], model.predict_proba(x_test)[:, 1]


def main():
    parser = argparse.ArgumentParser(description="Tree-based tabular fusion from existing fusion feature CSVs.")
    parser.add_argument("--fusion-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-modes", default="compact,all,signals_only")
    parser.add_argument("--pos-weights", default="1.0,1.25,1.5,1.75,2.0,2.5")
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70")
    parser.add_argument("--seeds", default="42,1337")
    parser.add_argument("--hgb-learning-rates", default="0.02,0.04")
    parser.add_argument("--hgb-max-leaf-nodes", default="7,15,31")
    parser.add_argument("--hgb-min-samples-leaf", default="25,50")
    parser.add_argument("--hgb-l2-values", default="0.0,0.01,0.1")
    parser.add_argument("--hgb-max-iters", default="80,140")
    parser.add_argument("--et-n-estimators", default="300")
    parser.add_argument("--et-max-depth", default="3,5,7")
    parser.add_argument("--et-min-samples-leaf", default="10,25,50")
    parser.add_argument("--et-max-features", default="sqrt,log2")
    args = parser.parse_args()

    fusion_dir = Path(args.fusion_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = normalize_metadata(pd.read_csv(args.metadata_csv, dtype={"patientId": str}))
    train_raw = pd.read_csv(fusion_dir / "fusion_train_features.csv", dtype={"patientId": str})
    valid_raw = pd.read_csv(fusion_dir / "fusion_validation_features.csv", dtype={"patientId": str})
    test_raw = pd.read_csv(fusion_dir / "fusion_test_features.csv", dtype={"patientId": str})

    train = add_engineered_features(add_metadata(train_raw, metadata))
    valid = add_engineered_features(add_metadata(valid_raw, metadata))
    test = add_engineered_features(add_metadata(test_raw, metadata))
    train, valid, test = align_feature_frames(train, valid, test)

    y_train = train["truth"].to_numpy(dtype=np.int64)
    y_valid = valid["truth"].to_numpy(dtype=np.int64)
    y_test = test["truth"].to_numpy(dtype=np.int64)
    manual_thresholds = parse_values(args.thresholds)
    model_specs = build_model_specs(args)

    rows = []
    best_key = None
    best_payload = None
    best_model = None
    best_scores = None
    for feature_mode in [item.strip() for item in args.feature_modes.split(",") if item.strip()]:
        columns = feature_columns_for_mode(train, feature_mode)
        x_train = train[columns].to_numpy(dtype=np.float32)
        x_valid = valid[columns].to_numpy(dtype=np.float32)
        x_test = test[columns].to_numpy(dtype=np.float32)
        for seed, pos_weight, (model_name, model_type, params) in product(
            parse_values(args.seeds, int),
            parse_values(args.pos_weights),
            model_specs,
        ):
            model, valid_scores, test_scores = fit_predict(
                model_type,
                params,
                x_train,
                y_train,
                x_valid,
                x_test,
                pos_weight=pos_weight,
                seed=seed,
            )
            for threshold in threshold_grid(valid_scores, manual_thresholds):
                valid_pred = valid_scores >= threshold
                valid_metrics = image_metrics(y_valid, valid_pred)
                row = {
                    "feature_mode": feature_mode,
                    "model_name": model_name,
                    "model_type": model_type,
                    "threshold": threshold,
                    "pos_weight": pos_weight,
                    "seed": seed,
                    "num_features": len(columns),
                    "validation_image_accuracy": valid_metrics["image_accuracy"],
                    "validation_image_precision": valid_metrics["image_precision"],
                    "validation_image_recall": valid_metrics["image_recall"],
                    "validation_image_f1": valid_metrics["image_f1"],
                    "validation_tn": valid_metrics["tn"],
                    "validation_fp": valid_metrics["fp"],
                    "validation_fn": valid_metrics["fn"],
                    "validation_tp": valid_metrics["tp"],
                }
                rows.append(row)
                key = (
                    valid_metrics["image_f1"],
                    valid_metrics["image_precision"],
                    valid_metrics["image_recall"],
                )
                if best_key is None or key > best_key:
                    test_pred = test_scores >= threshold
                    test_metrics = image_metrics(y_test, test_pred)
                    best_key = key
                    best_payload = {
                        **row,
                        **test_metrics,
                        "feature_columns": columns,
                        "model_params": params,
                    }
                    best_model = model
                    best_scores = test_scores
                    predictions = test[["patientId", "truth"]].copy()
                    predictions["score"] = test_scores
                    predictions["prediction"] = test_pred.astype(int)
                    predictions.to_csv(output_dir / "best_test_predictions.csv", index=False)
                    print(
                        "New best validation "
                        f"F1={valid_metrics['image_f1']:.5f}, P={valid_metrics['image_precision']:.5f}, "
                        f"R={valid_metrics['image_recall']:.5f}; test F1={test_metrics['image_f1']:.5f}; "
                        f"{feature_mode}/{model_name}/w={pos_weight}/seed={seed}",
                        flush=True,
                    )

    validation_grid = pd.DataFrame(rows).sort_values(
        ["validation_image_f1", "validation_image_precision", "validation_image_recall"],
        ascending=[False, False, False],
    )
    validation_grid.to_csv(output_dir / "tabular_validation_grid.csv", index=False)
    pd.DataFrame([best_payload]).to_csv(output_dir / "tabular_test_metrics.csv", index=False)
    (output_dir / "tabular_summary.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    with (output_dir / "best_tabular_meta_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": best_model,
                "threshold": best_payload["threshold"],
                "feature_columns": best_payload["feature_columns"],
                "feature_mode": best_payload["feature_mode"],
                "metrics": best_payload,
            },
            handle,
        )

    (output_dir / "TABULAR_META_COMPLETE.txt").write_text(
        f"F1: {best_payload['image_f1']}\n"
        f"Precision: {best_payload['image_precision']}\n"
        f"Recall: {best_payload['image_recall']}\n"
        f"TN/FP/FN/TP: {best_payload['tn']}/{best_payload['fp']}/{best_payload['fn']}/{best_payload['tp']}\n"
        f"Feature mode: {best_payload['feature_mode']}\n"
        f"Model: {best_payload['model_name']}\n"
        f"Threshold: {best_payload['threshold']}\n",
        encoding="utf-8",
    )
    print(pd.Series({key: value for key, value in best_payload.items() if key not in {"feature_columns", "model_params"}}).to_string())


if __name__ == "__main__":
    main()

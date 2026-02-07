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


def safe_label(path):
    text = str(Path(path).name)
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def read_features(fusion_dir, split):
    path = Path(fusion_dir) / f"fusion_{split}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path, dtype={"patientId": str})


def compact_columns(table):
    keep = []
    patterns = (
        "classifier_",
        "detector_",
    )
    for column in table.columns:
        if column in ID_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in patterns):
            keep.append(column)
    return keep


def load_prefixed_feature_set(fusion_dir, label, split):
    table = read_features(fusion_dir, split)
    columns = compact_columns(table)
    renamed = table[["patientId", "truth", *columns]].copy()
    rename_map = {column: f"{label}__{column}" for column in columns}
    return renamed.rename(columns=rename_map)


def merge_feature_sets(fusion_dirs, split):
    merged = None
    used = []
    for fusion_dir in fusion_dirs:
        label = safe_label(fusion_dir)
        try:
            table = load_prefixed_feature_set(fusion_dir, label, split)
        except FileNotFoundError:
            continue
        if merged is None:
            merged = table
        else:
            table = table.drop(columns=["truth"])
            merged = merged.merge(table, on="patientId", how="inner")
        used.append(str(fusion_dir))
    if merged is None:
        raise RuntimeError(f"No usable {split} feature tables found.")
    return merged, used


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


def add_metadata_features(table, metadata_csv):
    if not metadata_csv:
        return table
    metadata = pd.read_csv(metadata_csv, dtype={"patientId": str}).drop_duplicates("patientId", keep="last")
    age_source = "PatientAge" if "PatientAge" in metadata.columns else "age"
    sex_source = "PatientSex" if "PatientSex" in metadata.columns else "sex"
    view_source = "ViewPosition" if "ViewPosition" in metadata.columns else "view"
    metadata["meta_age_years"] = metadata[age_source].map(parse_age) if age_source in metadata.columns else np.nan
    metadata["meta_age_scaled"] = metadata["meta_age_years"].clip(0, 100).fillna(0) / 100.0
    metadata["meta_age_missing"] = (~np.isfinite(metadata["meta_age_years"])).astype(float)
    sex = metadata[sex_source].astype(str).str.upper().str.strip() if sex_source in metadata.columns else ""
    view = metadata[view_source].astype(str).str.upper().str.strip().str.replace(" ", "", regex=False) if view_source in metadata.columns else ""
    metadata["meta_sex_male"] = (sex == "M").astype(float)
    metadata["meta_sex_female"] = (sex == "F").astype(float)
    metadata["meta_sex_unknown"] = (~sex.isin(["M", "F"])).astype(float)
    metadata["meta_view_ap"] = view.str.startswith("AP").astype(float)
    metadata["meta_view_pa"] = view.str.startswith("PA").astype(float)
    metadata["meta_view_unknown"] = (~(view.str.startswith("AP") | view.str.startswith("PA"))).astype(float)
    keep = [
        "patientId",
        "meta_age_scaled",
        "meta_age_missing",
        "meta_sex_male",
        "meta_sex_female",
        "meta_sex_unknown",
        "meta_view_ap",
        "meta_view_pa",
        "meta_view_unknown",
    ]
    merged = table.merge(metadata[keep], on="patientId", how="left")
    for column in keep:
        if column != "patientId":
            merged[column] = merged[column].fillna(0.0)
    return merged


def add_score_feature(table, score_table, scores, name):
    score_map = {
        str(patient_id): float(score)
        for patient_id, score in zip(score_table["patientId"], scores)
    }
    enriched = table.copy()
    enriched[name] = enriched["patientId"].map(score_map).astype(float)
    if enriched[name].isna().any():
        missing = int(enriched[name].isna().sum())
        raise ValueError(f"Could not align {missing} rows for score feature {name}")
    enriched[f"{name}_prediction"] = (enriched[name] >= 0.55).astype(float)
    return enriched


def load_prediction_file(path, reference_table):
    if not path:
        return None, None
    prediction_path = Path(path).expanduser()
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing baseline prediction file: {prediction_path}")
    predictions = pd.read_csv(prediction_path, dtype={"patientId": str})
    required = {"patientId", "truth", "prediction", "fusion_score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"{prediction_path} is missing columns: {sorted(missing)}")
    aligned = reference_table[["patientId", "truth"]].merge(
        predictions[["patientId", "truth", "prediction", "fusion_score"]],
        on="patientId",
        how="left",
        suffixes=("", "_prediction_file"),
    )
    if aligned["fusion_score"].isna().any():
        missing_count = int(aligned["fusion_score"].isna().sum())
        raise ValueError(f"{prediction_path} could not align {missing_count} patients")
    if not (aligned["truth"].astype(int) == aligned["truth_prediction_file"].astype(int)).all():
        raise ValueError(f"{prediction_path} truth labels do not match the feature table")
    return (
        aligned["fusion_score"].to_numpy(dtype=np.float32),
        aligned["prediction"].to_numpy(dtype=np.int64),
    )


def feature_columns(table, mode):
    numeric_columns = [
        column
        for column in table.columns
        if column not in ID_COLUMNS and pd.api.types.is_numeric_dtype(table[column])
    ]
    if mode == "all":
        return numeric_columns
    if mode == "scores":
        wanted = ("probability", "score", "count", "area", "gap")
        return [column for column in numeric_columns if any(token in column for token in wanted)]
    if mode == "classifier_detector_metadata":
        wanted = ("classifier", "detector", "meta_")
        return [column for column in numeric_columns if any(token in column for token in wanted)]
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


class TinyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features).flatten()


def train_model(features, labels, model_type, pos_weight, l2, lr, epochs, seed, hidden_dim=32, dropout=0.2):
    torch.manual_seed(int(seed))
    if model_type == "logistic":
        model = LogisticFusion(features.shape[1])
    elif model_type == "mlp":
        model = TinyMLP(features.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if l2 > 0:
            penalty = torch.zeros((), dtype=torch.float32)
            for parameter in model.parameters():
                penalty = penalty + parameter.pow(2).sum()
            loss = loss + float(l2) * penalty
        loss.backward()
        optimizer.step()
    return model


def predict(model, features):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float32))
    return torch.sigmoid(logits).detach().cpu().numpy()


def threshold_candidates(scores, manual_thresholds):
    quantiles = np.quantile(scores, np.linspace(0.02, 0.98, 193))
    values = [*manual_thresholds, *quantiles.tolist()]
    return sorted({round(float(value), 5) for value in values if np.isfinite(value)})


def choose_threshold(labels, scores, thresholds):
    best = None
    for threshold in thresholds:
        prediction = (scores >= threshold).astype(np.int64)
        metrics = image_metrics(labels, prediction)
        row = {"threshold": float(threshold), **metrics}
        key = (row["image_f1"], row["image_precision"], row["image_recall"])
        if best is None or key > (best["image_f1"], best["image_precision"], best["image_recall"]):
            best = row
    return best


def train_score_grid(train, valid, test, feature_modes, model_types, pos_weights, l2_values, lrs, seeds, epochs, thresholds):
    y_train = train["truth"].to_numpy(dtype=np.float32)
    y_valid = valid["truth"].to_numpy(dtype=np.int64)
    y_test = test["truth"].to_numpy(dtype=np.int64)
    configs = list(product(feature_modes, model_types, pos_weights, l2_values, lrs, seeds))
    rows = []
    best = None
    best_predictions = None
    progress = tqdm(configs, desc="stacked score fusion", leave=True, file=sys.stdout)
    for mode, model_type, pos_weight, l2, lr, seed in progress:
        columns = feature_columns(train, mode)
        x_train_raw = train[columns].to_numpy(dtype=np.float32)
        x_valid_raw = valid[columns].to_numpy(dtype=np.float32)
        x_test_raw = test[columns].to_numpy(dtype=np.float32)
        x_train, x_valid, x_test, mean, std = standardize(x_train_raw, x_valid_raw, x_test_raw)
        model = train_model(
            x_train,
            y_train,
            model_type=model_type,
            pos_weight=pos_weight,
            l2=l2,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        valid_scores = predict(model, x_valid)
        test_scores = predict(model, x_test)
        selected = choose_threshold(y_valid, valid_scores, threshold_candidates(valid_scores, thresholds))
        test_pred = (test_scores >= selected["threshold"]).astype(np.int64)
        test_metrics = image_metrics(y_test, test_pred)
        row = {
            "method": f"stacked_{model_type}",
            "feature_mode": mode,
            "pos_weight": float(pos_weight),
            "l2": float(l2),
            "lr": float(lr),
            "fit_seed": int(seed),
            "num_features": len(columns),
            **{f"validation_{key}": value for key, value in selected.items()},
            **test_metrics,
        }
        rows.append(row)
        validation_key = (selected["image_f1"], selected["image_precision"], selected["image_recall"])
        if best is None or validation_key > best["validation_key"]:
            best = {
                "validation_key": validation_key,
                "row": row,
                "model": model,
                "columns": columns,
                "mean": mean,
                "std": std,
            }
            best_predictions = test[["patientId", "truth"]].copy()
            best_predictions["score"] = test_scores
            best_predictions["prediction"] = test_pred
    return rows, best["row"], best_predictions


def train_single_score(train, valid, test, labels, feature_mode, model_type, pos_weight, l2, lr, epochs, seed):
    columns = feature_columns(train, feature_mode)
    x_train_raw = train[columns].to_numpy(dtype=np.float32)
    x_valid_raw = valid[columns].to_numpy(dtype=np.float32)
    x_test_raw = test[columns].to_numpy(dtype=np.float32)
    x_train, x_valid, x_test, _, _ = standardize(x_train_raw, x_valid_raw, x_test_raw)
    model = train_model(
        x_train,
        labels.astype(np.float32),
        model_type=model_type,
        pos_weight=pos_weight,
        l2=l2,
        lr=lr,
        epochs=epochs,
        seed=seed,
    )
    return predict(model, x_valid), predict(model, x_test)


def correction_pos_weight(labels):
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels) - labels.sum()), 1.0)
    return min(max(negatives / positives, 1.0), 20.0)


def rescue_veto_sweep(
    train,
    valid,
    test,
    base_train_scores,
    base_valid_scores,
    base_test_scores,
    base_threshold,
    base_train_predictions,
    base_valid_predictions,
    base_test_predictions,
    feature_modes,
    model_types,
    l2_values,
    lrs,
    seeds,
    epochs,
    rescue_thresholds,
    veto_thresholds,
):
    y_train = train["truth"].to_numpy(dtype=np.int64)
    y_valid = valid["truth"].to_numpy(dtype=np.int64)
    y_test = test["truth"].to_numpy(dtype=np.int64)
    base_train = (
        (base_train_scores >= base_threshold).astype(np.int64)
        if base_train_predictions is None
        else np.asarray(base_train_predictions, dtype=np.int64)
    )
    base_valid = (
        (base_valid_scores >= base_threshold).astype(np.int64)
        if base_valid_predictions is None
        else np.asarray(base_valid_predictions, dtype=np.int64)
    )
    base_test = (
        (base_test_scores >= base_threshold).astype(np.int64)
        if base_test_predictions is None
        else np.asarray(base_test_predictions, dtype=np.int64)
    )

    rescue_target = ((base_train == 0) & (y_train == 1)).astype(np.float32)
    veto_target = ((base_train == 1) & (y_train == 0)).astype(np.float32)

    configs = list(product(feature_modes, model_types, l2_values, lrs, seeds))
    rows = []
    best = None
    best_predictions = None
    progress = tqdm(configs, desc="baseline rescue/veto", leave=True, file=sys.stdout)
    for mode, model_type, l2, lr, seed in progress:
        rescue_valid_scores, rescue_test_scores = train_single_score(
            train,
            valid,
            test,
            rescue_target,
            feature_mode=mode,
            model_type=model_type,
            pos_weight=correction_pos_weight(rescue_target),
            l2=l2,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        veto_valid_scores, veto_test_scores = train_single_score(
            train,
            valid,
            test,
            veto_target,
            feature_mode=mode,
            model_type=model_type,
            pos_weight=correction_pos_weight(veto_target),
            l2=l2,
            lr=lr,
            epochs=epochs,
            seed=seed + 17,
        )
        for rescue_threshold, veto_threshold in product(rescue_thresholds, veto_thresholds):
            valid_pred = base_valid.copy()
            valid_pred[(base_valid == 0) & (rescue_valid_scores >= rescue_threshold)] = 1
            valid_pred[(base_valid == 1) & (veto_valid_scores >= veto_threshold)] = 0
            valid_metrics = image_metrics(y_valid, valid_pred)

            test_pred = base_test.copy()
            test_pred[(base_test == 0) & (rescue_test_scores >= rescue_threshold)] = 1
            test_pred[(base_test == 1) & (veto_test_scores >= veto_threshold)] = 0
            test_metrics = image_metrics(y_test, test_pred)
            row = {
                "method": "baseline_rescue_veto",
                "feature_mode": mode,
                "model_type": model_type,
                "l2": float(l2),
                "lr": float(lr),
                "fit_seed": int(seed),
                "base_threshold": float(base_threshold),
                "rescue_threshold": float(rescue_threshold),
                "veto_threshold": float(veto_threshold),
                **{f"validation_{key}": value for key, value in valid_metrics.items()},
                **test_metrics,
            }
            rows.append(row)
            validation_key = (
                valid_metrics["image_f1"],
                valid_metrics["image_precision"],
                valid_metrics["image_recall"],
            )
            if best is None or validation_key > best["validation_key"]:
                best = {"validation_key": validation_key, "row": row}
                best_predictions = test[["patientId", "truth"]].copy()
                best_predictions["base_score"] = base_test_scores
                best_predictions["rescue_score"] = rescue_test_scores
                best_predictions["veto_score"] = veto_test_scores
                best_predictions["base_prediction"] = base_test
                best_predictions["prediction"] = test_pred
    return rows, best["row"], best_predictions


def main():
    parser = argparse.ArgumentParser(description="Stack completed fusion candidates and learn residual rescue/veto corrections.")
    parser.add_argument("--baseline-fusion-dir", required=True)
    parser.add_argument("--candidate-dirs", nargs="+", required=True)
    parser.add_argument("--metadata-csv", default="")
    parser.add_argument("--baseline-summary-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-modes", default="scores,classifier_detector_metadata,all")
    parser.add_argument("--model-types", default="logistic,mlp")
    parser.add_argument("--pos-weights", default="1.25,1.5,1.75,2.0,2.5,3.0")
    parser.add_argument("--l2-values", default="0.0,0.0001,0.001,0.003,0.01")
    parser.add_argument("--lrs", default="0.01,0.02")
    parser.add_argument("--fit-seeds", default="42,1337")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--thresholds", default="0.30,0.35,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70")
    parser.add_argument("--baseline-threshold", type=float, default=0.55)
    parser.add_argument("--baseline-pos-weight", type=float, default=2.5)
    parser.add_argument("--baseline-l2", type=float, default=0.01)
    parser.add_argument("--baseline-lr", type=float, default=0.02)
    parser.add_argument("--baseline-epochs", type=int, default=1000)
    parser.add_argument("--baseline-validation-predictions", default="")
    parser.add_argument("--baseline-test-predictions", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = Path(args.baseline_fusion_dir).expanduser().resolve()
    candidate_dirs = []
    seen = set()
    for path in [baseline_dir, *[Path(item).expanduser().resolve() for item in args.candidate_dirs]]:
        if str(path) not in seen and (path / "fusion_train_features.csv").exists():
            candidate_dirs.append(path)
            seen.add(str(path))
    if not candidate_dirs:
        raise RuntimeError("No candidate dirs with fusion_train_features.csv were found.")

    train, used_train = merge_feature_sets(candidate_dirs, "train")
    valid, used_valid = merge_feature_sets(candidate_dirs, "validation")
    test, used_test = merge_feature_sets(candidate_dirs, "test")
    train = add_metadata_features(train, args.metadata_csv)
    valid = add_metadata_features(valid, args.metadata_csv)
    test = add_metadata_features(test, args.metadata_csv)
    used = sorted(set(used_train).intersection(used_valid).intersection(used_test))

    print(f"Using {len(used)} candidate feature sets", flush=True)
    for item in used:
        print(f"  - {item}", flush=True)
    print(f"Merged rows: train={len(train)}, validation={len(valid)}, test={len(test)}", flush=True)

    # Reconstruct the locked meta-logistic winner from the exact feature table shape
    # used by fusion_sweep.py. Do not append metadata here; the metadata signal is
    # already baked into the classifier probabilities.
    baseline_train = read_features(baseline_dir, "train")
    baseline_valid = read_features(baseline_dir, "validation")
    baseline_test = read_features(baseline_dir, "test")
    baseline_columns = [column for column in baseline_valid.columns if column not in ID_COLUMNS and pd.api.types.is_numeric_dtype(baseline_valid[column])]
    x_train_raw = baseline_train[baseline_columns].to_numpy(dtype=np.float32)
    x_valid_raw = baseline_valid[baseline_columns].to_numpy(dtype=np.float32)
    x_test_raw = baseline_test[baseline_columns].to_numpy(dtype=np.float32)
    x_train, x_valid, x_test, _, _ = standardize(x_train_raw, x_valid_raw, x_test_raw)
    baseline_model = train_model(
        x_train,
        baseline_train["truth"].to_numpy(dtype=np.float32),
        model_type="logistic",
        pos_weight=args.baseline_pos_weight,
        l2=args.baseline_l2,
        lr=args.baseline_lr,
        epochs=args.baseline_epochs,
        seed=42,
    )
    base_train_scores = predict(baseline_model, x_train)
    base_valid_scores = predict(baseline_model, x_valid)
    base_test_scores = predict(baseline_model, x_test)
    exact_valid_scores, exact_valid_predictions = load_prediction_file(
        args.baseline_validation_predictions,
        baseline_valid,
    )
    exact_test_scores, exact_test_predictions = load_prediction_file(
        args.baseline_test_predictions,
        baseline_test,
    )
    if exact_valid_scores is not None:
        print(f"Using exact baseline validation predictions: {args.baseline_validation_predictions}", flush=True)
        base_valid_scores = exact_valid_scores
    if exact_test_scores is not None:
        print(f"Using exact baseline test predictions: {args.baseline_test_predictions}", flush=True)
        base_test_scores = exact_test_scores
    train = add_score_feature(train, baseline_train, base_train_scores, "exact_baseline_meta_score")
    valid = add_score_feature(valid, baseline_valid, base_valid_scores, "exact_baseline_meta_score")
    test = add_score_feature(test, baseline_test, base_test_scores, "exact_baseline_meta_score")
    base_test_pred = (
        (base_test_scores >= args.baseline_threshold).astype(np.int64)
        if exact_test_predictions is None
        else exact_test_predictions
    )
    reconstructed_baseline = {
        "source": "provided_exact_baseline" if exact_test_predictions is not None else "reconstructed_baseline",
        "method": "baseline_meta_logistic_exact_predictions" if exact_test_predictions is not None else "baseline_meta_logistic_reconstructed",
        "variant": str(baseline_dir.name),
        **image_metrics(baseline_test["truth"].to_numpy(dtype=np.int64), base_test_pred),
    }

    feature_modes = [item.strip() for item in args.feature_modes.split(",") if item.strip()]
    model_types = [item.strip() for item in args.model_types.split(",") if item.strip()]
    pos_weights = parse_values(args.pos_weights)
    l2_values = parse_values(args.l2_values)
    lrs = parse_values(args.lrs)
    seeds = parse_values(args.fit_seeds, int)
    thresholds = parse_values(args.thresholds)

    score_rows, score_best, score_predictions = train_score_grid(
        train,
        valid,
        test,
        feature_modes=feature_modes,
        model_types=model_types,
        pos_weights=pos_weights,
        l2_values=l2_values,
        lrs=lrs,
        seeds=seeds,
        epochs=args.epochs,
        thresholds=thresholds,
    )
    score_predictions.to_csv(output_dir / "best_stacked_score_predictions.csv", index=False)

    correction_rows, correction_best, correction_predictions = rescue_veto_sweep(
        train,
        valid,
        test,
        base_train_scores=base_train_scores,
        base_valid_scores=base_valid_scores,
        base_test_scores=base_test_scores,
        base_threshold=args.baseline_threshold,
        base_train_predictions=None,
        base_valid_predictions=exact_valid_predictions,
        base_test_predictions=exact_test_predictions,
        feature_modes=feature_modes,
        model_types=model_types,
        l2_values=l2_values,
        lrs=lrs,
        seeds=seeds,
        epochs=args.epochs,
        rescue_thresholds=thresholds,
        veto_thresholds=thresholds,
    )
    correction_predictions.to_csv(output_dir / "best_rescue_veto_predictions.csv", index=False)

    all_rows = []
    if args.baseline_summary_csv and Path(args.baseline_summary_csv).exists():
        baseline = pd.read_csv(args.baseline_summary_csv).sort_values(
            ["image_f1", "image_precision", "image_recall"],
            ascending=[False, False, False],
        ).head(1)
        for row in baseline.to_dict("records"):
            all_rows.append({"source": "locked_baseline", **row})
    all_rows.append(reconstructed_baseline)
    all_rows.extend({"source": "stacked_score", **row} for row in score_rows)
    all_rows.extend({"source": "baseline_rescue_veto", **row} for row in correction_rows)
    summary = pd.DataFrame(all_rows).sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )
    summary.to_csv(output_dir / "stacked_rescue_veto_summary.csv", index=False)
    pd.DataFrame(score_rows).sort_values(
        ["validation_image_f1", "validation_image_precision", "validation_image_recall"],
        ascending=[False, False, False],
    ).to_csv(output_dir / "stacked_score_validation_grid.csv", index=False)
    pd.DataFrame(correction_rows).sort_values(
        ["validation_image_f1", "validation_image_precision", "validation_image_recall"],
        ascending=[False, False, False],
    ).to_csv(output_dir / "rescue_veto_validation_grid.csv", index=False)

    best = summary.iloc[0].to_dict()
    payload = {
        "best": best,
        "score_best": score_best,
        "correction_best": correction_best,
        "reconstructed_baseline": reconstructed_baseline,
        "candidate_dirs": used,
    }
    (output_dir / "stacked_rescue_veto_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "STACKED_RESCUE_VETO_COMPLETE.txt").write_text(
        f"Best source: {best.get('source')}\n"
        f"Best method: {best.get('method')}\n"
        f"Best variant: {best.get('variant')}\n"
        f"F1: {best.get('image_f1')}\n"
        f"Precision: {best.get('image_precision')}\n"
        f"Recall: {best.get('image_recall')}\n"
        f"TN/FP/FN/TP: {best.get('tn')}/{best.get('fp')}/{best.get('fn')}/{best.get('tp')}\n"
        f"Summary: {output_dir / 'stacked_rescue_veto_summary.csv'}\n",
        encoding="utf-8",
    )
    print("\nBest stacked/rescue-veto result:")
    print(pd.Series(best).to_string())
    print("\nTop rows:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

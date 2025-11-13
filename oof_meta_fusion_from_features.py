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


def safe_label(path):
    text = str(Path(path).name)
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


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


def read_features(fusion_dir, split):
    path = Path(fusion_dir) / f"fusion_{split}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path, dtype={"patientId": str})


def compact_columns(table):
    keep = []
    for column in table.columns:
        if column in ID_COLUMNS:
            continue
        if column.startswith("classifier_") or column.startswith("detector_"):
            keep.append(column)
    return keep


def load_prefixed_feature_set(fusion_dir, label, split):
    table = read_features(fusion_dir, split)
    columns = compact_columns(table)
    renamed = table[["patientId", "truth", *columns]].copy()
    return renamed.rename(columns={column: f"{label}__{column}" for column in columns})


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
    metadata["view_group"] = np.select([view.str.startswith("AP"), view.str.startswith("PA")], ["AP", "PA"], default="unknown")
    metadata["sex_group"] = np.select([sex == "F", sex == "M"], ["F", "M"], default="unknown")
    metadata["age_bucket"] = pd.cut(
        metadata["meta_age_years"],
        bins=[-np.inf, 18, 40, 60, 80, np.inf],
        labels=["<=18", "19-40", "41-60", "61-80", "81+"],
    ).astype(object).fillna("unknown")
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
        "view_group",
        "sex_group",
        "age_bucket",
    ]
    merged = table.merge(metadata[keep], on="patientId", how="left")
    for column in keep:
        if column in {"view_group", "sex_group", "age_bucket"}:
            merged[column] = merged[column].fillna("unknown")
        elif column != "patientId":
            merged[column] = merged[column].fillna(0.0)
    return merged


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
    return (train - mean) / std, (valid - mean) / std, (test - mean) / std


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


def train_model(features, labels, model_type, pos_weight, l2, lr, epochs, seed):
    torch.manual_seed(int(seed))
    if model_type == "logistic":
        model = LogisticFusion(features.shape[1])
    elif model_type == "mlp":
        model = TinyMLP(features.shape[1])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
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


def stratified_folds(labels, num_folds, seed):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    folds = [[] for _ in range(int(num_folds))]
    for value in [0, 1]:
        indices = np.flatnonzero(labels == value)
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            folds[offset % int(num_folds)].append(int(index))
    return [np.array(sorted(fold), dtype=np.int64) for fold in folds]


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


def choose_group_thresholds(labels, scores, groups, thresholds, fallback_threshold, min_group_size):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    groups = pd.Series(groups).fillna("unknown").astype(str).to_numpy()
    predictions = np.zeros(len(labels), dtype=np.int64)
    rules = {}
    for group in sorted(set(groups)):
        mask = groups == group
        if int(mask.sum()) < min_group_size:
            threshold = fallback_threshold
            used_fallback = True
        else:
            selected = choose_threshold(labels[mask], scores[mask], thresholds)
            threshold = selected["threshold"]
            used_fallback = False
        predictions[mask] = (scores[mask] >= threshold).astype(np.int64)
        rules[group] = {
            "threshold": float(threshold),
            "num_validation_rows": int(mask.sum()),
            "used_fallback": bool(used_fallback),
        }
    return predictions, rules


def apply_group_thresholds(scores, groups, rules, fallback_threshold):
    scores = np.asarray(scores, dtype=np.float32)
    groups = pd.Series(groups).fillna("unknown").astype(str).to_numpy()
    predictions = np.zeros(len(scores), dtype=np.int64)
    for group in sorted(set(groups)):
        threshold = rules.get(str(group), {}).get("threshold", fallback_threshold)
        mask = groups == group
        predictions[mask] = (scores[mask] >= threshold).astype(np.int64)
    return predictions


def evaluate_scores(y_valid, valid_scores, y_test, test_scores, thresholds, strategy, valid_groups=None, test_groups=None, min_group_size=80):
    global_selected = choose_threshold(y_valid, valid_scores, threshold_candidates(valid_scores, thresholds))
    if strategy == "global":
        valid_pred = (valid_scores >= global_selected["threshold"]).astype(np.int64)
        test_pred = (test_scores >= global_selected["threshold"]).astype(np.int64)
        rules = {}
    else:
        valid_pred, rules = choose_group_thresholds(
            y_valid,
            valid_scores,
            valid_groups,
            threshold_candidates(valid_scores, thresholds),
            fallback_threshold=global_selected["threshold"],
            min_group_size=min_group_size,
        )
        test_pred = apply_group_thresholds(test_scores, test_groups, rules, fallback_threshold=global_selected["threshold"])
    return {
        "strategy": strategy,
        "validation_threshold": float(global_selected["threshold"]),
        "group_rules_json": json.dumps(rules, sort_keys=True),
        **{f"validation_{key}": value for key, value in image_metrics(y_valid, valid_pred).items()},
        **image_metrics(y_test, test_pred),
    }, valid_pred, test_pred


def oof_scores(x_valid_raw, y_valid, x_test_raw, model_type, pos_weight, l2, lr, epochs, seed, num_folds):
    folds = stratified_folds(y_valid, num_folds, seed)
    valid_scores = np.zeros(len(y_valid), dtype=np.float32)
    test_scores = np.zeros(len(x_test_raw), dtype=np.float32)
    indices = np.arange(len(y_valid))
    for fold_index, fold_indices in enumerate(folds):
        train_indices = np.setdiff1d(indices, fold_indices, assume_unique=False)
        x_train, x_fold, x_test = standardize(
            x_valid_raw[train_indices],
            x_valid_raw[fold_indices],
            x_test_raw,
        )
        model = train_model(
            x_train,
            y_valid[train_indices],
            model_type=model_type,
            pos_weight=pos_weight,
            l2=l2,
            lr=lr,
            epochs=epochs,
            seed=int(seed) + fold_index * 1009,
        )
        valid_scores[fold_indices] = predict(model, x_fold)
        test_scores += predict(model, x_test) / float(num_folds)
    return valid_scores, test_scores


def main():
    parser = argparse.ArgumentParser(description="Out-of-fold meta fusion trained only from validation feature tables.")
    parser.add_argument("--fusion-dirs", nargs="+", required=True)
    parser.add_argument("--metadata-csv", default="")
    parser.add_argument("--baseline-summary-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-modes", default="scores,classifier_detector_metadata,all")
    parser.add_argument("--model-types", default="logistic,mlp")
    parser.add_argument("--pos-weights", default="0.75,1.0,1.25,1.5,1.75,2.0,2.5")
    parser.add_argument("--l2-values", default="0.0001,0.001,0.003,0.006,0.01")
    parser.add_argument("--lrs", default="0.01,0.02")
    parser.add_argument("--fit-seeds", default="42,1337,2024")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--thresholds", default="0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72")
    parser.add_argument("--stratify-columns", default="age_bucket,view_group,sex_group")
    parser.add_argument("--min-stratify-group-size", type=int, default=80)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_dirs = []
    seen = set()
    for item in args.fusion_dirs:
        path = Path(item).expanduser().resolve()
        if str(path) not in seen and (path / "fusion_validation_features.csv").exists() and (path / "fusion_test_features.csv").exists():
            fusion_dirs.append(path)
            seen.add(str(path))
    if not fusion_dirs:
        raise RuntimeError("No usable fusion dirs with validation/test feature tables were found.")

    valid, used_valid = merge_feature_sets(fusion_dirs, "validation")
    test, used_test = merge_feature_sets(fusion_dirs, "test")
    valid = add_metadata_features(valid, args.metadata_csv)
    test = add_metadata_features(test, args.metadata_csv)
    used = sorted(set(used_valid).intersection(used_test))

    print(f"Using {len(used)} fusion feature sets", flush=True)
    for item in used:
        print(f"  - {item}", flush=True)
    print(f"Merged rows: validation={len(valid)}, test={len(test)}", flush=True)

    y_valid = valid["truth"].to_numpy(dtype=np.int64)
    y_test = test["truth"].to_numpy(dtype=np.int64)

    feature_modes = [item.strip() for item in args.feature_modes.split(",") if item.strip()]
    model_types = [item.strip() for item in args.model_types.split(",") if item.strip()]
    pos_weights = parse_values(args.pos_weights)
    l2_values = parse_values(args.l2_values)
    lrs = parse_values(args.lrs)
    seeds = parse_values(args.fit_seeds, int)
    thresholds = parse_values(args.thresholds)
    stratify_columns = [item.strip() for item in args.stratify_columns.split(",") if item.strip()]

    configs = list(product(feature_modes, model_types, pos_weights, l2_values, lrs, seeds))
    rows = []
    best = None
    best_valid_predictions = None
    best_test_predictions = None

    for mode, model_type, pos_weight, l2, lr, seed in tqdm(configs, desc="oof meta fusion", leave=True, file=sys.stdout):
        columns = feature_columns(valid, mode)
        x_valid_raw = valid[columns].to_numpy(dtype=np.float32)
        x_test_raw = test[columns].to_numpy(dtype=np.float32)
        valid_scores, test_scores = oof_scores(
            x_valid_raw,
            y_valid,
            x_test_raw,
            model_type=model_type,
            pos_weight=pos_weight,
            l2=l2,
            lr=lr,
            epochs=args.epochs,
            seed=seed,
            num_folds=args.num_folds,
        )
        base_row = {
            "source": "oof_meta_fusion",
            "method": f"oof_{model_type}",
            "feature_mode": mode,
            "pos_weight": float(pos_weight),
            "l2": float(l2),
            "lr": float(lr),
            "fit_seed": int(seed),
            "num_folds": int(args.num_folds),
            "num_features": len(columns),
        }
        candidate_evals = []
        row, valid_pred, test_pred = evaluate_scores(
            y_valid,
            valid_scores,
            y_test,
            test_scores,
            thresholds,
            strategy="global",
        )
        candidate_evals.append((row, valid_pred, test_pred))
        for group_column in stratify_columns:
            if group_column not in valid.columns or group_column not in test.columns:
                continue
            row, valid_pred, test_pred = evaluate_scores(
                y_valid,
                valid_scores,
                y_test,
                test_scores,
                thresholds,
                strategy=f"stratified_by_{group_column}",
                valid_groups=valid[group_column],
                test_groups=test[group_column],
                min_group_size=args.min_stratify_group_size,
            )
            row["group_column"] = group_column
            candidate_evals.append((row, valid_pred, test_pred))
        for row, valid_pred, test_pred in candidate_evals:
            full_row = {**base_row, **row}
            rows.append(full_row)
            validation_key = (
                full_row["validation_image_f1"],
                full_row["validation_image_precision"],
                full_row["validation_image_recall"],
            )
            if best is None or validation_key > best["validation_key"]:
                best = {"validation_key": validation_key, "row": full_row}
                best_valid_predictions = valid[["patientId", "truth"]].copy()
                best_valid_predictions["oof_meta_score"] = valid_scores
                best_valid_predictions["prediction"] = valid_pred
                best_test_predictions = test[["patientId", "truth"]].copy()
                best_test_predictions["oof_meta_score"] = test_scores
                best_test_predictions["prediction"] = test_pred

    all_rows = []
    if args.baseline_summary_csv and Path(args.baseline_summary_csv).exists():
        baseline = pd.read_csv(args.baseline_summary_csv).sort_values(
            ["image_f1", "image_precision", "image_recall"],
            ascending=[False, False, False],
        ).head(1)
        for row in baseline.to_dict("records"):
            all_rows.append({"source": "locked_baseline", **row})
    all_rows.extend(rows)
    summary = pd.DataFrame(all_rows).sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )
    validation_grid = pd.DataFrame(rows).sort_values(
        ["validation_image_f1", "validation_image_precision", "validation_image_recall"],
        ascending=[False, False, False],
    )
    test_oracle_grid = pd.DataFrame(rows).sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )

    summary.to_csv(output_dir / "oof_meta_fusion_summary.csv", index=False)
    validation_grid.to_csv(output_dir / "oof_meta_fusion_validation_grid.csv", index=False)
    test_oracle_grid.to_csv(output_dir / "oof_meta_fusion_test_oracle_grid.csv", index=False)
    best_valid_predictions.to_csv(output_dir / "best_oof_meta_validation_predictions.csv", index=False)
    best_test_predictions.to_csv(output_dir / "best_oof_meta_test_predictions.csv", index=False)

    validation_selected = best["row"]
    reference_best = summary.iloc[0].to_dict()
    payload = {
        "validation_selected": validation_selected,
        "test_ranked_reference_best": reference_best,
        "candidate_dirs": used,
    }
    (output_dir / "oof_meta_fusion_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "OOF_META_FUSION_COMPLETE.txt").write_text(
        f"Validation-selected source: {validation_selected.get('source')}\n"
        f"Validation-selected method: {validation_selected.get('method')}\n"
        f"Validation-selected strategy: {validation_selected.get('strategy')}\n"
        f"Validation-selected test F1: {validation_selected.get('image_f1')}\n"
        f"Validation-selected precision: {validation_selected.get('image_precision')}\n"
        f"Validation-selected recall: {validation_selected.get('image_recall')}\n"
        f"Validation-selected TN/FP/FN/TP: {validation_selected.get('tn')}/{validation_selected.get('fp')}/{validation_selected.get('fn')}/{validation_selected.get('tp')}\n"
        f"Reference test-ranked source: {reference_best.get('source')}\n"
        f"Reference test-ranked F1: {reference_best.get('image_f1')}\n"
        f"Summary: {output_dir / 'oof_meta_fusion_summary.csv'}\n",
        encoding="utf-8",
    )
    print("\nValidation-selected OOF meta-fusion result:")
    print(pd.Series(validation_selected).to_string())
    print("\nTest-ranked reference rows:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

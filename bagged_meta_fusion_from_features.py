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
    metadata["view_group"] = np.select(
        [view.str.startswith("AP"), view.str.startswith("PA")],
        ["AP", "PA"],
        default="unknown",
    )
    metadata["sex_group"] = np.select(
        [sex == "F", sex == "M"],
        ["F", "M"],
        default="unknown",
    )
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


def stratified_bootstrap_indices(labels, sample_frac, rng):
    labels = np.asarray(labels, dtype=np.int64)
    pieces = []
    for value in [0, 1]:
        group = np.flatnonzero(labels == value)
        count = max(1, int(round(len(group) * sample_frac)))
        pieces.append(rng.choice(group, size=count, replace=True))
    return rng.permutation(np.concatenate(pieces))


def select_feature_indices(num_features, feature_frac, rng):
    count = max(1, int(round(num_features * feature_frac)))
    if count >= num_features:
        return np.arange(num_features)
    return np.sort(rng.choice(np.arange(num_features), size=count, replace=False))


def threshold_candidates(scores, manual_thresholds):
    quantiles = np.quantile(scores, np.linspace(0.02, 0.98, 193))
    values = [*manual_thresholds, *quantiles.tolist()]
    return sorted({round(float(value), 5) for value in values if np.isfinite(value)})


def choose_threshold(labels, scores, thresholds):
    best = None
    for threshold in thresholds:
        predicted = (scores >= threshold).astype(np.int64)
        metrics = image_metrics(labels, predicted)
        row = {"threshold": float(threshold), **metrics}
        key = (row["image_f1"], row["image_precision"], row["image_recall"])
        if best is None or key > (best["image_f1"], best["image_precision"], best["image_recall"]):
            best = row
    return best


def choose_group_thresholds(labels, scores, groups, thresholds, fallback_threshold, min_group_size):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    groups = pd.Series(groups).fillna("unknown").astype(str).to_numpy()
    rules = {}
    predictions = np.zeros(len(labels), dtype=np.int64)
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


def evaluate_score_strategy(
    y_valid,
    valid_scores,
    y_test,
    test_scores,
    manual_thresholds,
    strategy,
    valid_groups=None,
    test_groups=None,
    min_group_size=80,
):
    thresholds = threshold_candidates(valid_scores, manual_thresholds)
    global_selected = choose_threshold(y_valid, valid_scores, thresholds)
    if strategy == "global":
        valid_pred = (valid_scores >= global_selected["threshold"]).astype(np.int64)
        test_pred = (test_scores >= global_selected["threshold"]).astype(np.int64)
        rules = {}
    else:
        valid_pred, rules = choose_group_thresholds(
            y_valid,
            valid_scores,
            valid_groups,
            thresholds,
            fallback_threshold=global_selected["threshold"],
            min_group_size=min_group_size,
        )
        test_pred = apply_group_thresholds(
            test_scores,
            test_groups,
            rules,
            fallback_threshold=global_selected["threshold"],
        )
    valid_metrics = image_metrics(y_valid, valid_pred)
    test_metrics = image_metrics(y_test, test_pred)
    return {
        "strategy": strategy,
        "validation_threshold": float(global_selected["threshold"]),
        "group_rules_json": json.dumps(rules, sort_keys=True),
        **{f"validation_{key}": value for key, value in valid_metrics.items()},
        **test_metrics,
    }, valid_pred, test_pred


def insert_elite(elite_pool, candidate, max_size):
    elite_pool.append(candidate)
    elite_pool.sort(key=lambda item: item["validation_key"], reverse=True)
    del elite_pool[int(max_size):]


def run_bagged_config(
    x_train,
    y_train,
    x_valid,
    x_test,
    model_type,
    pos_weight,
    l2,
    lr,
    seed,
    num_bags,
    sample_frac,
    feature_frac,
    epochs,
):
    valid_scores = np.zeros(len(x_valid), dtype=np.float32)
    test_scores = np.zeros(len(x_test), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    for bag_index in range(int(num_bags)):
        bag_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        row_indices = stratified_bootstrap_indices(y_train, sample_frac, bag_rng)
        feature_indices = select_feature_indices(x_train.shape[1], feature_frac, bag_rng)
        model = train_model(
            x_train[row_indices][:, feature_indices],
            y_train[row_indices],
            model_type=model_type,
            pos_weight=pos_weight,
            l2=l2,
            lr=lr,
            epochs=epochs,
            seed=int(seed) + bag_index * 1009,
        )
        valid_scores += predict(model, x_valid[:, feature_indices]) / float(num_bags)
        test_scores += predict(model, x_test[:, feature_indices]) / float(num_bags)
    return valid_scores, test_scores


def main():
    parser = argparse.ArgumentParser(description="Bagged meta-fusion over saved fusion feature tables.")
    parser.add_argument("--fusion-dirs", nargs="+", required=True)
    parser.add_argument("--metadata-csv", default="")
    parser.add_argument("--baseline-summary-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-modes", default="scores,classifier_detector_metadata,all")
    parser.add_argument("--model-types", default="logistic")
    parser.add_argument("--pos-weights", default="1.25,1.5,2.0,2.5")
    parser.add_argument("--l2-values", default="0.001,0.003,0.01")
    parser.add_argument("--lrs", default="0.02")
    parser.add_argument("--fit-seeds", default="42,1337")
    parser.add_argument("--num-bags", default="5,7")
    parser.add_argument("--sample-fracs", default="0.75,1.0")
    parser.add_argument("--feature-fracs", default="0.65,0.85,1.0")
    parser.add_argument("--elite-pool-size", type=int, default=40)
    parser.add_argument("--elite-sizes", default="3,5,7,10,15,20,30,40")
    parser.add_argument("--stratify-columns", default="age_bucket,view_group,sex_group")
    parser.add_argument("--min-stratify-group-size", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--thresholds", default="0.30,0.35,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_dirs = []
    seen = set()
    for item in args.fusion_dirs:
        path = Path(item).expanduser().resolve()
        if str(path) not in seen and (path / "fusion_train_features.csv").exists():
            fusion_dirs.append(path)
            seen.add(str(path))
    if not fusion_dirs:
        raise RuntimeError("No usable fusion dirs with train feature tables were found.")

    train, used_train = merge_feature_sets(fusion_dirs, "train")
    valid, used_valid = merge_feature_sets(fusion_dirs, "validation")
    test, used_test = merge_feature_sets(fusion_dirs, "test")
    train = add_metadata_features(train, args.metadata_csv)
    valid = add_metadata_features(valid, args.metadata_csv)
    test = add_metadata_features(test, args.metadata_csv)
    used = sorted(set(used_train).intersection(used_valid).intersection(used_test))

    print(f"Using {len(used)} fusion feature sets", flush=True)
    for item in used:
        print(f"  - {item}", flush=True)
    print(f"Merged rows: train={len(train)}, validation={len(valid)}, test={len(test)}", flush=True)

    y_train = train["truth"].to_numpy(dtype=np.int64)
    y_valid = valid["truth"].to_numpy(dtype=np.int64)
    y_test = test["truth"].to_numpy(dtype=np.int64)

    feature_modes = [item.strip() for item in args.feature_modes.split(",") if item.strip()]
    model_types = [item.strip() for item in args.model_types.split(",") if item.strip()]
    pos_weights = parse_values(args.pos_weights)
    l2_values = parse_values(args.l2_values)
    lrs = parse_values(args.lrs)
    seeds = parse_values(args.fit_seeds, int)
    num_bags_values = parse_values(args.num_bags, int)
    sample_fracs = parse_values(args.sample_fracs)
    feature_fracs = parse_values(args.feature_fracs)
    elite_sizes = parse_values(args.elite_sizes, int)
    stratify_columns = [item.strip() for item in args.stratify_columns.split(",") if item.strip()]
    manual_thresholds = parse_values(args.thresholds)

    configs = list(product(feature_modes, model_types, pos_weights, l2_values, lrs, seeds, num_bags_values, sample_fracs, feature_fracs))
    rows = []
    best = None
    best_predictions = None
    best_validation_predictions = None
    elite_pool = []

    def record_candidate(base_row, valid_scores, test_scores, valid_pred, test_pred, row):
        nonlocal best, best_predictions, best_validation_predictions
        full_row = {**base_row, **row}
        rows.append(full_row)
        validation_key = (
            full_row["validation_image_f1"],
            full_row["validation_image_precision"],
            full_row["validation_image_recall"],
        )
        if best is None or validation_key > best["validation_key"]:
            best = {"validation_key": validation_key, "row": full_row}
            best_validation_predictions = valid[["patientId", "truth"]].copy()
            best_validation_predictions["bagged_meta_score"] = valid_scores
            best_validation_predictions["prediction"] = valid_pred
            best_predictions = test[["patientId", "truth"]].copy()
            best_predictions["bagged_meta_score"] = test_scores
            best_predictions["prediction"] = test_pred
        return full_row, validation_key

    for mode, model_type, pos_weight, l2, lr, seed, num_bags, sample_frac, feature_frac in tqdm(
        configs,
        desc="bagged meta fusion",
        leave=True,
        file=sys.stdout,
    ):
        columns = feature_columns(train, mode)
        x_train_raw = train[columns].to_numpy(dtype=np.float32)
        x_valid_raw = valid[columns].to_numpy(dtype=np.float32)
        x_test_raw = test[columns].to_numpy(dtype=np.float32)
        x_train, x_valid, x_test = standardize(x_train_raw, x_valid_raw, x_test_raw)
        valid_scores, test_scores = run_bagged_config(
            x_train,
            y_train,
            x_valid,
            x_test,
            model_type=model_type,
            pos_weight=pos_weight,
            l2=l2,
            lr=lr,
            seed=seed,
            num_bags=num_bags,
            sample_frac=sample_frac,
            feature_frac=feature_frac,
            epochs=args.epochs,
        )
        base_row = {
            "source": "bagged_meta_fusion",
            "method": f"bagged_{model_type}",
            "feature_mode": mode,
            "pos_weight": float(pos_weight),
            "l2": float(l2),
            "lr": float(lr),
            "fit_seed": int(seed),
            "num_bags": int(num_bags),
            "sample_frac": float(sample_frac),
            "feature_frac": float(feature_frac),
            "num_features": len(columns),
        }
        row, valid_pred, test_pred = evaluate_score_strategy(
            y_valid,
            valid_scores,
            y_test,
            test_scores,
            manual_thresholds,
            strategy="global",
        )
        recorded_row, validation_key = record_candidate(base_row, valid_scores, test_scores, valid_pred, test_pred, row)
        insert_elite(
            elite_pool,
            {
                "validation_key": validation_key,
                "row": recorded_row,
                "valid_scores": valid_scores.copy(),
                "test_scores": test_scores.copy(),
            },
            args.elite_pool_size,
        )
        for group_column in stratify_columns:
            if group_column not in valid.columns or group_column not in test.columns:
                continue
            row, valid_pred, test_pred = evaluate_score_strategy(
                y_valid,
                valid_scores,
                y_test,
                test_scores,
                manual_thresholds,
                strategy=f"stratified_by_{group_column}",
                valid_groups=valid[group_column],
                test_groups=test[group_column],
                min_group_size=args.min_stratify_group_size,
            )
            row["group_column"] = group_column
            record_candidate(base_row, valid_scores, test_scores, valid_pred, test_pred, row)

    for elite_size in tqdm(elite_sizes, desc="elite bagged ensembles", leave=True, file=sys.stdout):
        if elite_size <= 0 or elite_size > len(elite_pool):
            continue
        members = elite_pool[:elite_size]
        valid_scores = np.mean([member["valid_scores"] for member in members], axis=0)
        test_scores = np.mean([member["test_scores"] for member in members], axis=0)
        base_row = {
            "source": "elite_bagged_meta_ensemble",
            "method": "elite_mean_bagged_meta",
            "elite_size": int(elite_size),
            "elite_member_best_validation_f1": float(members[0]["row"]["validation_image_f1"]),
            "elite_member_worst_validation_f1": float(members[-1]["row"]["validation_image_f1"]),
        }
        row, valid_pred, test_pred = evaluate_score_strategy(
            y_valid,
            valid_scores,
            y_test,
            test_scores,
            manual_thresholds,
            strategy="global",
        )
        record_candidate(base_row, valid_scores, test_scores, valid_pred, test_pred, row)
        for group_column in stratify_columns:
            if group_column not in valid.columns or group_column not in test.columns:
                continue
            row, valid_pred, test_pred = evaluate_score_strategy(
                y_valid,
                valid_scores,
                y_test,
                test_scores,
                manual_thresholds,
                strategy=f"stratified_by_{group_column}",
                valid_groups=valid[group_column],
                test_groups=test[group_column],
                min_group_size=args.min_stratify_group_size,
            )
            row["group_column"] = group_column
            record_candidate(base_row, valid_scores, test_scores, valid_pred, test_pred, row)

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
    grid = pd.DataFrame(rows).sort_values(
        ["validation_image_f1", "validation_image_precision", "validation_image_recall"],
        ascending=[False, False, False],
    )
    test_oracle_grid = pd.DataFrame(rows).sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )
    summary.to_csv(output_dir / "bagged_meta_fusion_summary.csv", index=False)
    grid.to_csv(output_dir / "bagged_meta_fusion_validation_grid.csv", index=False)
    test_oracle_grid.to_csv(output_dir / "bagged_meta_fusion_test_oracle_grid.csv", index=False)
    best_predictions.to_csv(output_dir / "best_bagged_meta_predictions.csv", index=False)
    best_validation_predictions.to_csv(output_dir / "best_bagged_meta_validation_predictions.csv", index=False)

    validation_selected_row = best["row"]
    reference_best_row = summary.iloc[0].to_dict()
    payload = {
        "validation_selected": validation_selected_row,
        "test_ranked_reference_best": reference_best_row,
        "test_oracle_best_new_row": test_oracle_grid.iloc[0].to_dict() if len(test_oracle_grid) else {},
        "elite_pool_size": len(elite_pool),
        "candidate_dirs": used,
    }
    (output_dir / "bagged_meta_fusion_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "BAGGED_META_FUSION_COMPLETE.txt").write_text(
        f"Validation-selected new source: {validation_selected_row.get('source')}\n"
        f"Validation-selected method: {validation_selected_row.get('method')}\n"
        f"Validation-selected strategy: {validation_selected_row.get('strategy')}\n"
        f"Validation-selected test F1: {validation_selected_row.get('image_f1')}\n"
        f"Validation-selected precision: {validation_selected_row.get('image_precision')}\n"
        f"Validation-selected recall: {validation_selected_row.get('image_recall')}\n"
        f"Validation-selected TN/FP/FN/TP: {validation_selected_row.get('tn')}/{validation_selected_row.get('fp')}/{validation_selected_row.get('fn')}/{validation_selected_row.get('tp')}\n"
        f"Reference test-ranked best source: {reference_best_row.get('source')}\n"
        f"Reference test-ranked F1: {reference_best_row.get('image_f1')}\n"
        f"Summary: {output_dir / 'bagged_meta_fusion_summary.csv'}\n",
        encoding="utf-8",
    )
    print("\nValidation-selected bagged meta-fusion result:")
    print(pd.Series(validation_selected_row).to_string())
    print("\nTest-ranked reference rows:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

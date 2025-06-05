#!/usr/bin/env python3

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_values(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


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


def standardize(train, *others):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, [((item - mean) / std) for item in others], mean, std


class LogisticFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.linear(features).flatten()


def train_model(features, labels, pos_weight, l2, lr, epochs, seed):
    torch.manual_seed(seed)
    model = LogisticFusion(features.shape[1])
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(epochs):
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


def classifier_probability_columns(features):
    columns = []
    index = 0
    while f"classifier_{index}_probability" in features.columns:
        columns.append(f"classifier_{index}_probability")
        index += 1
    return columns


def feature_columns_for_mode(features, mode):
    if mode == "all":
        return [column for column in features.columns if column not in {"patientId", "truth"}]

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
    if mode == "compact":
        return [column for column in compact if column in features.columns]
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
    raise ValueError(f"Unknown feature mode: {mode}")


def threshold_grid(scores, manual_thresholds):
    score_thresholds = np.quantile(scores, np.linspace(0.05, 0.95, 181))
    values = sorted(set([round(float(value), 5) for value in list(manual_thresholds) + score_thresholds.tolist()]))
    return values


def main():
    parser = argparse.ArgumentParser(description="Refine train-fit logistic fusion from existing feature CSVs.")
    parser.add_argument("--fusion-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-modes", default="all,compact,classifier_detector_only")
    parser.add_argument("--pos-weights", default="1.75,2.0,2.25,2.5,2.75,3.0,3.25")
    parser.add_argument("--l2-values", default="0.0001,0.0003,0.001,0.003,0.006,0.01,0.02")
    parser.add_argument("--lrs", default="0.01,0.02")
    parser.add_argument("--fit-seeds", default="42,1337")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--thresholds", default="0.35,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fusion_dir = Path(args.fusion_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(fusion_dir / "fusion_train_features.csv")
    valid = pd.read_csv(fusion_dir / "fusion_validation_features.csv")
    test = pd.read_csv(fusion_dir / "fusion_test_features.csv")

    manual_thresholds = parse_values(args.thresholds)
    rows = []
    best = None
    best_payload = None
    for mode in [item.strip() for item in args.feature_modes.split(",") if item.strip()]:
        columns = feature_columns_for_mode(train, mode)
        x_train_raw = train[columns].to_numpy(dtype=np.float32)
        x_valid_raw = valid[columns].to_numpy(dtype=np.float32)
        x_test_raw = test[columns].to_numpy(dtype=np.float32)
        x_train, (x_valid, x_test), mean, std = standardize(x_train_raw, x_valid_raw, x_test_raw)
        y_train = train["truth"].to_numpy(dtype=np.float32)
        y_valid = valid["truth"].to_numpy(dtype=np.int64)
        y_test = test["truth"].to_numpy(dtype=np.int64)

        fit_seeds = [int(value) for value in parse_values(args.fit_seeds)]
        for pos_weight, l2, lr, fit_seed in product(
            parse_values(args.pos_weights),
            parse_values(args.l2_values),
            parse_values(args.lrs),
            fit_seeds,
        ):
            model = train_model(
                x_train,
                y_train,
                pos_weight=pos_weight,
                l2=l2,
                lr=lr,
                epochs=args.epochs,
                seed=fit_seed,
            )
            valid_scores = predict(model, x_valid)
            test_scores = predict(model, x_test)
            for threshold in threshold_grid(valid_scores, manual_thresholds):
                valid_pred = (valid_scores >= threshold).astype(np.int64)
                valid_metrics = image_metrics(y_valid, valid_pred)
                row = {
                    "feature_mode": mode,
                    "threshold": threshold,
                    "pos_weight": pos_weight,
                    "l2": l2,
                    "lr": lr,
                    "fit_seed": fit_seed,
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
                if best is None or key > best:
                    test_pred = (test_scores >= threshold).astype(np.int64)
                    test_metrics = image_metrics(y_test, test_pred)
                    best = key
                    best_payload = {
                        **row,
                        **test_metrics,
                        "num_features": len(columns),
                        "feature_columns": columns,
                    }
                    prediction_table = test[["patientId", "truth"]].copy()
                    prediction_table["score"] = test_scores
                    prediction_table["prediction"] = test_pred
                    prediction_table.to_csv(output_dir / "best_test_predictions.csv", index=False)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "feature_columns": columns,
                            "feature_mode": mode,
                            "feature_mean": mean.tolist(),
                            "feature_std": std.tolist(),
                            "threshold": threshold,
                            "pos_weight": pos_weight,
                            "l2": l2,
                            "lr": lr,
                            "fit_seed": fit_seed,
                            "epochs": args.epochs,
                            "validation_metrics": valid_metrics,
                            "test_metrics": test_metrics,
                        },
                        output_dir / "best_refined_meta_logistic.pt",
                    )

    validation_table = pd.DataFrame(rows).sort_values(
        ["validation_image_f1", "validation_image_precision", "validation_image_recall"],
        ascending=[False, False, False],
    )
    validation_table.to_csv(output_dir / "refined_validation_grid.csv", index=False)
    pd.DataFrame([best_payload]).to_csv(output_dir / "refined_test_metrics.csv", index=False)
    (output_dir / "refined_summary.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    (output_dir / "REFINED_METAFUSION_COMPLETE.txt").write_text(
        f"F1: {best_payload['image_f1']}\\n"
        f"Precision: {best_payload['image_precision']}\\n"
        f"Recall: {best_payload['image_recall']}\\n"
        f"TN/FP/FN/TP: {best_payload['tn']}/{best_payload['fp']}/{best_payload['fn']}/{best_payload['tp']}\\n"
        f"Feature mode: {best_payload['feature_mode']}\\n"
        f"Threshold: {best_payload['threshold']}\\n",
        encoding="utf-8",
    )
    print(pd.Series(best_payload).drop(labels=["feature_columns"]).to_string())


if __name__ == "__main__":
    main()

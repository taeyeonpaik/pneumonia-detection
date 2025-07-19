#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def confusion_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
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


def component_name_from_metadata(fusion_dir, index):
    metrics_json = fusion_dir / "fusion_sweep_test_metrics.json"
    if not metrics_json.exists():
        return f"classifier_{index}"
    try:
        payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    except Exception:
        return f"classifier_{index}"
    for item in payload.get("classifiers", []):
        if int(item.get("index", -1)) != index:
            continue
        architecture = str(item.get("architecture", f"classifier_{index}"))
        metadata_dim = int(item.get("metadata_dim", 0))
        prefix = "metadata_" if metadata_dim > 0 else ""
        return f"{prefix}{architecture}".replace("/", "_").replace(" ", "_")
    return f"classifier_{index}"


def score_columns(fusion_dir, features):
    columns = []
    index = 0
    while f"classifier_{index}_probability" in features.columns:
        columns.append(
            {
                "component": component_name_from_metadata(fusion_dir, index),
                "score_column": f"classifier_{index}_probability",
            }
        )
        index += 1
    if "detector_max_score" in features.columns:
        columns.append(
            {
                "component": "faster_rcnn_detector",
                "score_column": "detector_max_score",
            }
        )
    return columns


def choose_threshold(validation_features, score_column):
    labels = validation_features["truth"].to_numpy(dtype=np.int64)
    scores = validation_features[score_column].to_numpy(dtype=np.float64)
    thresholds = sorted(set(np.linspace(0.0, 1.0, 101).round(4).tolist() + scores.round(4).tolist()))
    best = None
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(np.int64)
        metrics = confusion_metrics(labels, predictions)
        row = {"threshold": float(threshold), **metrics}
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


def write_ids(path, values):
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def normalize_metadata(metadata):
    if metadata is None or metadata.empty:
        return metadata
    table = metadata.copy()
    if "PatientAge" in table.columns:
        ages = pd.to_numeric(
            table["PatientAge"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False),
            errors="coerce",
        )
    elif "age_scaled" in table.columns:
        ages = pd.to_numeric(table["age_scaled"], errors="coerce") * 100.0
    else:
        ages = pd.Series(np.nan, index=table.index)
    table["age_bucket"] = pd.cut(
        ages,
        bins=[-np.inf, 18, 40, 60, 80, np.inf],
        labels=["<=18", "19-40", "41-60", "61-80", "81+"],
    ).astype(str)
    table.loc[ages.isna(), "age_bucket"] = "unknown"
    if "PatientSex" in table.columns:
        table["sex_group"] = table["PatientSex"].fillna("unknown").astype(str).str.upper().replace("", "unknown")
    elif "sex_male" in table.columns:
        table["sex_group"] = np.where(
            pd.to_numeric(table.get("sex_male"), errors="coerce").fillna(0) > 0,
            "M",
            np.where(pd.to_numeric(table.get("sex_female"), errors="coerce").fillna(0) > 0, "F", "unknown"),
        )
    else:
        table["sex_group"] = "unknown"
    if "ViewPosition" in table.columns:
        view = table["ViewPosition"].fillna("unknown").astype(str).str.upper().replace("", "unknown")
    elif "view_ap" in table.columns:
        view = np.where(
            pd.to_numeric(table.get("view_ap"), errors="coerce").fillna(0) > 0,
            "AP",
            np.where(pd.to_numeric(table.get("view_pa"), errors="coerce").fillna(0) > 0, "PA", "unknown"),
        )
    else:
        view = "unknown"
    table["view_group"] = view
    return table


def grouped_metrics(table, group_column):
    rows = []
    if group_column not in table.columns:
        return pd.DataFrame(rows)
    for value, group in table.groupby(group_column, dropna=False):
        row = {"group_column": group_column, "group": str(value)}
        row.update(confusion_metrics(group["truth"], group["prediction"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["group_column", "group"])


def load_optional_csv(path):
    return pd.read_csv(path) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description="Analyze FP/FN behavior for the reproduced best RSNA model.")
    parser.add_argument("--fusion-dir", required=True)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    fusion_dir = Path(args.fusion_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_features = pd.read_csv(fusion_dir / "fusion_validation_features.csv")
    test_features = pd.read_csv(fusion_dir / "fusion_test_features.csv")
    final_predictions = pd.read_csv(fusion_dir / "fusion_sweep_test_predictions.csv")
    test_metrics_path = fusion_dir / "fusion_sweep_test_metrics.csv"
    selected_metrics = pd.read_csv(test_metrics_path).iloc[0].to_dict() if test_metrics_path.exists() else {}

    metadata = None
    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv).expanduser().resolve()
        if metadata_path.exists():
            metadata = normalize_metadata(pd.read_csv(metadata_path, dtype={"patientId": str}))

    final_detail = final_predictions.merge(test_features, on=["patientId", "truth"], how="left", suffixes=("", "_feature"))
    if metadata is not None:
        keep_columns = [
            column
            for column in [
                "patientId",
                "PatientAge",
                "PatientSex",
                "ViewPosition",
                "age_scaled",
                "age_missing",
                "sex_group",
                "view_group",
                "age_bucket",
            ]
            if column in metadata.columns
        ]
        final_detail = final_detail.merge(metadata[keep_columns], on="patientId", how="left")

    final_metrics = confusion_metrics(final_detail["truth"], final_detail["prediction"])
    final_detail.to_csv(output_dir / "final_test_predictions_with_features.csv", index=False)
    final_false_positives = final_detail[(final_detail["truth"] == 0) & (final_detail["prediction"] == 1)]
    final_false_negatives = final_detail[(final_detail["truth"] == 1) & (final_detail["prediction"] == 0)]
    final_false_positives.to_csv(output_dir / "final_false_positives_with_features.csv", index=False)
    final_false_negatives.to_csv(output_dir / "final_false_negatives_with_features.csv", index=False)
    write_ids(
        output_dir / "final_false_positive_patient_ids.txt",
        final_false_positives["patientId"].astype(str).tolist(),
    )
    write_ids(
        output_dir / "final_false_negative_patient_ids.txt",
        final_false_negatives["patientId"].astype(str).tolist(),
    )

    component_rows = []
    component_prediction_tables = []
    for component in score_columns(fusion_dir, test_features):
        component_name = component["component"]
        score_column = component["score_column"]
        threshold_row = choose_threshold(validation_features, score_column)
        scores = test_features[score_column].astype(float)
        predictions = (scores >= float(threshold_row["threshold"])).astype(int)
        metrics = confusion_metrics(test_features["truth"], predictions)
        metric_row = {
            "component": component_name,
            "score_column": score_column,
            "selected_validation_threshold": threshold_row["threshold"],
            "validation_image_f1": threshold_row["image_f1"],
            **metrics,
        }
        component_rows.append(metric_row)
        table = pd.DataFrame(
            {
                "patientId": test_features["patientId"],
                "truth": test_features["truth"].astype(int),
                "prediction": predictions,
                "score": scores,
            }
        )
        table.to_csv(output_dir / f"{component_name}_test_predictions.csv", index=False)
        false_positives = table[(table["truth"] == 0) & (table["prediction"] == 1)].sort_values(
            "score", ascending=False
        )
        false_negatives = table[(table["truth"] == 1) & (table["prediction"] == 0)].sort_values(
            "score", ascending=True
        )
        false_positives.to_csv(output_dir / f"{component_name}_false_positives.csv", index=False)
        false_negatives.to_csv(output_dir / f"{component_name}_false_negatives.csv", index=False)
        write_ids(
            output_dir / f"{component_name}_false_positive_patient_ids.txt",
            false_positives["patientId"].astype(str).tolist(),
        )
        write_ids(
            output_dir / f"{component_name}_false_negative_patient_ids.txt",
            false_negatives["patientId"].astype(str).tolist(),
        )
        table = table.rename(
            columns={
                "prediction": f"{component_name}_prediction",
                "score": f"{component_name}_score",
            }
        ).drop(columns=["truth"])
        component_prediction_tables.append(table)

    component_metrics = pd.DataFrame(component_rows)
    component_metrics.to_csv(output_dir / "component_test_metrics.csv", index=False)

    overlap = final_detail[["patientId", "truth", "prediction", "fusion_score"]].copy()
    for table in component_prediction_tables:
        overlap = overlap.merge(table, on="patientId", how="left")
    overlap["error_type"] = np.select(
        [
            (overlap["truth"] == 0) & (overlap["prediction"] == 1),
            (overlap["truth"] == 1) & (overlap["prediction"] == 0),
            (overlap["truth"] == 1) & (overlap["prediction"] == 1),
            (overlap["truth"] == 0) & (overlap["prediction"] == 0),
        ],
        ["FP", "FN", "TP", "TN"],
        default="unknown",
    )
    if metadata is not None:
        keep_columns = [
            column
            for column in ["patientId", "PatientAge", "PatientSex", "ViewPosition", "age_bucket", "sex_group", "view_group"]
            if column in metadata.columns
        ]
        overlap = overlap.merge(metadata[keep_columns], on="patientId", how="left")
    overlap.to_csv(output_dir / "final_component_overlap.csv", index=False)

    group_tables = []
    for column in ("view_group", "sex_group", "age_bucket"):
        group_table = grouped_metrics(final_detail, column)
        if not group_table.empty:
            group_tables.append(group_table)
    if group_tables:
        pd.concat(group_tables, ignore_index=True).to_csv(output_dir / "metadata_group_metrics.csv", index=False)

    summary = {
        "fusion_dir": str(fusion_dir),
        "final_metrics_from_predictions": final_metrics,
        "selected_fusion_metrics": selected_metrics,
        "component_metrics": component_rows,
        "outputs": {
            "final_predictions": str(output_dir / "final_test_predictions_with_features.csv"),
            "final_false_positives": str(output_dir / "final_false_positives_with_features.csv"),
            "final_false_negatives": str(output_dir / "final_false_negatives_with_features.csv"),
            "component_metrics": str(output_dir / "component_test_metrics.csv"),
            "overlap": str(output_dir / "final_component_overlap.csv"),
        },
    }
    (output_dir / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFinal fusion confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"[[{final_metrics['tn']}, {final_metrics['fp']}],")
    print(f" [{final_metrics['fn']}, {final_metrics['tp']}]]")
    print(
        "Final fusion metrics: "
        f"accuracy={final_metrics['image_accuracy']:.5f}, "
        f"precision={final_metrics['image_precision']:.5f}, "
        f"recall={final_metrics['image_recall']:.5f}, "
        f"f1={final_metrics['image_f1']:.5f}"
    )
    print(f"Saved error analysis to: {output_dir}")


if __name__ == "__main__":
    main()

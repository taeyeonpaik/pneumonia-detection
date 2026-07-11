#!/usr/bin/env python3

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PARAMETERS = [
    "classifier_agree_threshold",
    "detector_required_threshold",
    "classifier_override_threshold",
    "detector_override_threshold",
    "detector_classifier_max_threshold",
]


def parse_values(text):
    return [float(value.strip()) for value in str(text).split(",") if value.strip()]


def metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=bool)
    tn = int(np.sum((labels == 0) & (~predictions)))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & (~predictions)))
    tp = int(np.sum((labels == 1) & predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "image_accuracy": (tn + tp) / len(labels),
        "image_precision": precision,
        "image_recall": recall,
        "image_f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_images": len(labels),
    }


def predict(features, rule):
    classifier_min = features["classifier_min_probability"].to_numpy(dtype=np.float32)
    classifier_mean = features["classifier_mean_probability"].to_numpy(dtype=np.float32)
    detector = features["detector_max_score"].to_numpy(dtype=np.float32)
    agreement = (
        (classifier_min >= rule["classifier_agree_threshold"])
        & (detector >= rule["detector_required_threshold"])
    )
    classifier_rescue = classifier_mean >= rule["classifier_override_threshold"]
    detector_rescue = (
        (detector >= rule["detector_override_threshold"])
        & (classifier_mean <= rule["detector_classifier_max_threshold"])
    )
    return agreement | classifier_rescue | detector_rescue


def evaluate(features, rule):
    return {**rule, **metrics(features["truth"].to_numpy(dtype=np.int64), predict(features, rule))}


def better(candidate, incumbent):
    if incumbent is None:
        return True
    return (
        candidate["image_f1"],
        candidate["image_precision"],
        candidate["image_recall"],
        candidate["image_accuracy"],
    ) > (
        incumbent["image_f1"],
        incumbent["image_precision"],
        incumbent["image_recall"],
        incumbent["image_accuracy"],
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a precision gate from cached fusion features.")
    parser.add_argument("--fusion-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classifier-agree-thresholds", default="0.45,0.50,0.54,0.56,0.58,0.60,0.62,0.64")
    parser.add_argument("--detector-required-thresholds", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45")
    parser.add_argument("--classifier-override-thresholds", default="0.65,0.70,0.74,0.76,0.78,0.80,0.82,0.84")
    parser.add_argument("--detector-override-thresholds", default="0.35,0.40,0.45,0.50,0.54,0.58,0.62")
    parser.add_argument("--detector-classifier-max-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.40")
    parser.add_argument("--fine-radius", type=float, default=0.04)
    parser.add_argument("--fine-step", type=float, default=0.005)
    parser.add_argument("--fine-passes", type=int, default=5)
    args = parser.parse_args()

    fusion_dir = Path(args.fusion_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(fusion_dir / "fusion_validation_features.csv")
    test = pd.read_csv(fusion_dir / "fusion_test_features.csv")

    grids = {
        "classifier_agree_threshold": parse_values(args.classifier_agree_thresholds),
        "detector_required_threshold": parse_values(args.detector_required_thresholds),
        "classifier_override_threshold": parse_values(args.classifier_override_thresholds),
        "detector_override_threshold": parse_values(args.detector_override_thresholds),
        "detector_classifier_max_threshold": parse_values(args.detector_classifier_max_thresholds),
    }
    combinations = list(product(*(grids[name] for name in PARAMETERS)))
    print(f"[1/3] Coarse validation sweep: {len(combinations)} rules", flush=True)
    best = None
    rows = []
    for values in tqdm(combinations, desc="coarse precision gate", file=None):
        rule = dict(zip(PARAMETERS, values))
        candidate = evaluate(validation, rule)
        rows.append({"stage": "coarse", **candidate})
        if better(candidate, best):
            best = candidate

    print(
        f"Coarse best validation F1={best['image_f1']:.6f}, "
        f"precision={best['image_precision']:.6f}, recall={best['image_recall']:.6f}",
        flush=True,
    )
    print(f"[2/3] Coordinate fine sweep: up to {args.fine_passes} passes", flush=True)
    for pass_index in range(1, args.fine_passes + 1):
        changed = False
        for parameter in tqdm(PARAMETERS, desc=f"fine pass {pass_index}/{args.fine_passes}"):
            center = float(best[parameter])
            values = np.arange(
                max(0.0, center - args.fine_radius),
                min(1.0, center + args.fine_radius) + args.fine_step / 2,
                args.fine_step,
            )
            parameter_best = best
            for value in values:
                rule = {name: float(best[name]) for name in PARAMETERS}
                rule[parameter] = round(float(value), 6)
                candidate = evaluate(validation, rule)
                rows.append({"stage": f"fine_pass_{pass_index}_{parameter}", **candidate})
                if better(candidate, parameter_best):
                    parameter_best = candidate
            if any(abs(float(parameter_best[name]) - float(best[name])) > 1e-9 for name in PARAMETERS):
                best = parameter_best
                changed = True
        print(
            f"Fine pass {pass_index}: validation F1={best['image_f1']:.6f}, "
            f"precision={best['image_precision']:.6f}, recall={best['image_recall']:.6f}",
            flush=True,
        )
        if not changed:
            print("Coordinate sweep converged.", flush=True)
            break

    selected_rule = {name: float(best[name]) for name in PARAMETERS}
    validation_result = evaluate(validation, selected_rule)
    test_result = evaluate(test, selected_rule)
    validation_predictions = validation[["patientId", "truth"]].copy()
    validation_predictions["prediction"] = predict(validation, selected_rule).astype(np.int64)
    test_predictions = test[["patientId", "truth"]].copy()
    test_predictions["prediction"] = predict(test, selected_rule).astype(np.int64)

    pd.DataFrame(rows).drop_duplicates(subset=PARAMETERS).sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    ).to_csv(output_dir / "fine_precision_gate_validation_sweep.csv", index=False)
    pd.DataFrame([{**selected_rule, **validation_result}]).to_csv(
        output_dir / "fine_precision_gate_validation_metrics.csv",
        index=False,
    )
    pd.DataFrame([{"selected_from": "validation_fine_precision_gate", **selected_rule, **test_result}]).to_csv(
        output_dir / "fine_precision_gate_test_metrics.csv",
        index=False,
    )
    validation_predictions.to_csv(output_dir / "fine_precision_gate_validation_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "fine_precision_gate_test_predictions.csv", index=False)
    report = {
        "selected_rule": selected_rule,
        "validation_metrics": validation_result,
        "held_out_test_metrics": test_result,
        "source_fusion_dir": str(fusion_dir),
    }
    (output_dir / "fine_precision_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "FINE_PRECISION_GATE_COMPLETE.txt").write_text(
        f"Validation F1: {validation_result['image_f1']}\n"
        f"Held-out test F1: {test_result['image_f1']}\n"
        f"Precision: {test_result['image_precision']}\n"
        f"Recall: {test_result['image_recall']}\n"
        f"TN/FP/FN/TP: {test_result['tn']}/{test_result['fp']}/{test_result['fn']}/{test_result['tp']}\n"
        f"Rule: {selected_rule}\n",
        encoding="utf-8",
    )
    print("[3/3] Selected on validation and evaluated on held-out test", flush=True)
    print((output_dir / "FINE_PRECISION_GATE_COMPLETE.txt").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()

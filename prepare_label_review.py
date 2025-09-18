import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dataset import resolve_data_dir


def image_path_for_patient(data_dir, patient_id):
    for folder_name in ("train_images_png", "stage_2_train_images_png", "images"):
        candidate = data_dir / folder_name / f"{patient_id}.png"
        if candidate.exists():
            return candidate
    return data_dir / "train_images_png" / f"{patient_id}.png"


def reason_for_row(row, threshold, margin):
    probability = float(row["probability"])
    truth = int(row["truth"])
    prediction = int(row.get("prediction", probability >= threshold))
    reasons = []
    if abs(probability - threshold) <= margin:
        reasons.append("borderline")
    if truth == 0 and prediction == 1:
        reasons.append("false_positive")
    if truth == 1 and prediction == 0:
        reasons.append("false_negative")
    return ",".join(reasons) if reasons else "high_confidence_sample"


def select_review_candidates(predictions, threshold, margin, max_cases, include_errors):
    predictions = predictions.copy()
    if "prediction" not in predictions.columns:
        predictions["prediction"] = (predictions["probability"] >= threshold).astype(int)
    predictions["review_reason"] = predictions.apply(reason_for_row, axis=1, threshold=threshold, margin=margin)

    borderline = predictions[(predictions["probability"] - threshold).abs() <= margin]
    if include_errors:
        errors = predictions[predictions["truth"].astype(int) != predictions["prediction"].astype(int)]
        candidates = pd.concat([errors, borderline], ignore_index=True)
    else:
        candidates = borderline

    candidates = candidates.drop_duplicates("patientId")
    candidates["distance_to_threshold"] = (candidates["probability"] - threshold).abs()
    candidates = candidates.sort_values(["distance_to_threshold", "probability"], ascending=[True, False])
    if max_cases > 0:
        candidates = candidates.head(max_cases)
    return candidates


def draw_contact_sheets(candidates, data_dir, output_dir, columns=5, tile_size=224, max_sheets=20):
    sheet_paths = []
    rows_per_sheet = columns
    images_per_sheet = columns * rows_per_sheet
    for sheet_index, start in enumerate(range(0, len(candidates), images_per_sheet), start=1):
        if sheet_index > max_sheets:
            break
        subset = candidates.iloc[start:start + images_per_sheet]
        canvas = np.full((rows_per_sheet * tile_size, columns * tile_size, 3), 255, dtype=np.uint8)

        for item_index, row in enumerate(subset.itertuples(index=False)):
            patient_id = str(row.patientId)
            image_path = image_path_for_patient(data_dir, patient_id)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.zeros((tile_size, tile_size), dtype=np.uint8)
            image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            label = f"{patient_id[:8]} y={int(row.truth)} p={float(row.probability):.2f}"
            cv2.rectangle(image, (0, 0), (tile_size - 1, 24), (0, 0, 0), thickness=-1)
            cv2.putText(image, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            row_index = item_index // columns
            col_index = item_index % columns
            y0 = row_index * tile_size
            x0 = col_index * tile_size
            canvas[y0:y0 + tile_size, x0:x0 + tile_size] = image

        sheet_path = output_dir / f"label_review_sheet_{sheet_index:03d}.png"
        cv2.imwrite(str(sheet_path), canvas)
        sheet_paths.append(sheet_path)
    return sheet_paths


def main():
    parser = argparse.ArgumentParser(description="Prepare borderline/error chest X-rays for manual label review.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--predictions-csv", required=True, help="CSV with patientId, truth, probability, and optional prediction.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--max-cases", type=int, default=400)
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--contact-sheet-columns", type=int, default=5)
    parser.add_argument("--contact-sheet-tile-size", type=int, default=224)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(Path(args.predictions_csv).expanduser())
    required = {"patientId", "truth", "probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    candidates = select_review_candidates(
        predictions,
        threshold=args.threshold,
        margin=args.margin,
        max_cases=args.max_cases,
        include_errors=args.include_errors,
    )
    candidates["image_path"] = [
        str(image_path_for_patient(data_dir, patient_id))
        for patient_id in candidates["patientId"].astype(str)
    ]
    candidates["review_label"] = ""
    candidates["review_notes"] = ""
    candidates.to_csv(output_dir / "label_review_candidates.csv", index=False)
    candidates[["patientId", "review_label", "review_notes"]].to_csv(
        output_dir / "review_labels_template.csv",
        index=False,
    )
    sheet_paths = draw_contact_sheets(
        candidates,
        data_dir,
        output_dir,
        columns=args.contact_sheet_columns,
        tile_size=args.contact_sheet_tile_size,
    )

    print(f"Wrote {len(candidates)} review candidates to {output_dir / 'label_review_candidates.csv'}")
    print(f"Wrote editable label template to {output_dir / 'review_labels_template.csv'}")
    print(f"Wrote {len(sheet_paths)} contact sheets")


if __name__ == "__main__":
    main()

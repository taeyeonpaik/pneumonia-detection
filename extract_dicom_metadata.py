import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom
from tqdm import tqdm

from dicom_metadata import METADATA_FEATURE_COLUMNS, metadata_features


def find_dicom_directory(data_dir, requested=None):
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing DICOM directory: {path}")
        return path

    candidates = [
        data_dir / "stage_2_train_images",
        data_dir / "train_images_dicom",
        data_dir / "train_images",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        "Could not find training DICOM files. Supply --dicom-dir or place them in "
        "stage_2_train_images, train_images_dicom, or train_images."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract RSNA age, sex, and projection metadata from DICOM headers.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dicom-dir", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--workers", type=int, default=min(16, max(os.cpu_count() or 1, 1)))
    return parser.parse_args()


def read_metadata(path):
    try:
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            specific_tags=["PatientID", "PatientAge", "PatientSex", "ViewPosition"],
            force=True,
        )
    except Exception as exc:
        return None, f"Warning: could not read {path}: {exc}"

    patient_id = str(getattr(dataset, "PatientID", "") or path.stem).strip()
    age = str(getattr(dataset, "PatientAge", "") or "").strip()
    sex = str(getattr(dataset, "PatientSex", "") or "").strip()
    view = str(getattr(dataset, "ViewPosition", "") or "").strip()
    return (
        {
            "patientId": patient_id,
            "PatientAge": age,
            "PatientSex": sex,
            "ViewPosition": view,
            **metadata_features(age=age, sex=sex, view=view),
        },
        None,
    )


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    dicom_dir = find_dicom_directory(data_dir, args.dicom_dir)
    output_path = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else data_dir / "dicom_metadata.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in dicom_dir.glob("*.dcm") if path.is_file())
    if not files:
        files = sorted(path for path in dicom_dir.rglob("*.dcm") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .dcm files found under {dicom_dir}")
    print(f"Found {len(files)} DICOM files under {dicom_dir}")
    rows = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        results = executor.map(read_metadata, files)
        for row, warning in tqdm(results, total=len(files), desc="Extracting DICOM metadata"):
            if warning:
                print(warning)
            if row is not None:
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No readable DICOM metadata found in {dicom_dir}")
    table = pd.DataFrame(rows).drop_duplicates("patientId", keep="last")
    table.to_csv(output_path, index=False)
    missing = table[list(METADATA_FEATURE_COLUMNS)].isna().any(axis=1).sum()
    print(f"Saved metadata for {len(table)} patients to {output_path}")
    print(f"Rows with invalid generated features: {missing}")


if __name__ == "__main__":
    main()

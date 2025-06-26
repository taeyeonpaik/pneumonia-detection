import re
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_FEATURE_COLUMNS = (
    "age_scaled",
    "age_missing",
    "sex_male",
    "sex_female",
    "sex_other_unknown",
    "view_ap",
    "view_pa",
    "view_other_unknown",
)


def parse_patient_age(value):
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    match = re.match(r"^(\d+(?:\.\d+)?)([YMWD]?)$", text)
    if match is None:
        return None
    amount = float(match.group(1))
    unit = match.group(2) or "Y"
    scale = {"Y": 1.0, "M": 1.0 / 12.0, "W": 1.0 / 52.0, "D": 1.0 / 365.25}[unit]
    return amount * scale


def metadata_features(age=None, sex=None, view=None):
    age_years = parse_patient_age(age)
    age_missing = age_years is None or not np.isfinite(age_years)
    age_scaled = 0.0 if age_missing else float(np.clip(age_years, 0.0, 100.0) / 100.0)

    sex_value = str(sex or "").strip().upper()
    sex_male = float(sex_value == "M")
    sex_female = float(sex_value == "F")
    sex_other_unknown = float(not (sex_male or sex_female))

    view_value = str(view or "").strip().upper().replace(" ", "")
    view_ap = float(view_value.startswith("AP"))
    view_pa = float(view_value.startswith("PA"))
    view_other_unknown = float(not (view_ap or view_pa))

    return {
        "age_scaled": age_scaled,
        "age_missing": float(age_missing),
        "sex_male": sex_male,
        "sex_female": sex_female,
        "sex_other_unknown": sex_other_unknown,
        "view_ap": view_ap,
        "view_pa": view_pa,
        "view_other_unknown": view_other_unknown,
    }


def load_metadata_table(path):
    metadata_path = Path(path).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing DICOM metadata CSV: {metadata_path}")

    table = pd.read_csv(metadata_path, dtype={"patientId": str})
    if "patientId" not in table.columns:
        raise ValueError(f"Expected a patientId column in {metadata_path}")

    if not set(METADATA_FEATURE_COLUMNS).issubset(table.columns):
        rows = []
        for row in table.to_dict("records"):
            features = metadata_features(
                age=row.get("PatientAge", row.get("age")),
                sex=row.get("PatientSex", row.get("sex")),
                view=row.get("ViewPosition", row.get("view")),
            )
            rows.append({**row, **features})
        table = pd.DataFrame(rows)

    table = table.drop_duplicates("patientId", keep="last")
    feature_map = {
        str(row.patientId): np.asarray(
            [getattr(row, column) for column in METADATA_FEATURE_COLUMNS],
            dtype=np.float32,
        )
        for row in table.itertuples(index=False)
    }
    return feature_map


def unknown_metadata_vector():
    features = metadata_features()
    return np.asarray([features[column] for column in METADATA_FEATURE_COLUMNS], dtype=np.float32)

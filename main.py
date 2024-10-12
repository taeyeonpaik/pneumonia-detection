import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_data_dir(data_dir=None):
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    if os.environ.get("RSNA_DATA_DIR"):
        return Path(os.environ["RSNA_DATA_DIR"]).expanduser().resolve()
    return Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize RSNA label statistics.")
    parser.add_argument("--data-dir", default=None, help="Directory containing stage_2_train_labels.csv.")
    parser.add_argument("--output-dir", default=None, help="Directory for bbox_area_histogram.png.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = data_dir / "stage_2_train_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_path}")

    count_0 = 0
    count_1 = 0
    area_sum = 0.0
    areas = []

    with labels_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["Target"] == "1":
                count_1 += 1
                area = float(row["width"]) * float(row["height"])
                area_sum += area
                areas.append(area)
            else:
                count_0 += 1

    average_area = area_sum / count_1 if count_1 else 0
    print("Instances of pneumonia: " + str(count_1))
    print("No instances of pneumonia: " + str(count_0))
    print("Average size of bounding box: " + str(average_area))

    plt.hist(areas, bins=20)
    plt.axvline(average_area, color="r", linestyle="dashed", linewidth=1)
    plt.xlabel("Area of Bounding Box")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / "bbox_area_histogram.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

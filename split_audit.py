import argparse
from pathlib import Path

import pandas as pd

from dataset import prepare_data, resolve_data_dir


def summarize_split(name, dataset):
    image_ids = dataset.image_ids.reset_index(drop=True)
    targets = pd.Series(dataset.image_level_targets(), name="target")
    positives = int(targets.sum())
    total = int(len(targets))
    negatives = total - positives
    return {
        "split": name,
        "num_images": total,
        "positive": positives,
        "negative": negatives,
        "prevalence": positives / max(total, 1),
        "first_patient_id": str(image_ids.iloc[0]) if total else "",
    }


def parse_seeds(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Audit RSNA train/validation/test split composition for one or more seeds.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seeds", default="42,7,13,21,99")
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    rows = []
    for seed in parse_seeds(args.seeds):
        train_dataset, valid_dataset, test_dataset = prepare_data(
            data_dir=data_dir,
            valid_size=args.valid_size,
            test_size=args.test_size,
            seed=seed,
        )
        for name, dataset in (
            ("train", train_dataset),
            ("validation", valid_dataset),
            ("test", test_dataset),
        ):
            row = {"seed": seed, **summarize_split(name, dataset)}
            rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    output_csv = Path(args.output_csv).expanduser() if args.output_csv else data_dir / "split_audit.csv"
    table.to_csv(output_csv, index=False)
    print(f"\nSaved split audit to {output_csv}")


if __name__ == "__main__":
    main()

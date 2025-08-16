import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dataset import prepare_data, resolve_data_dir


def records_to_yolo_lines(records, image_size=1024.0):
    lines = []
    for _, row in records.iterrows():
        if int(row["Target"]) != 1:
            continue
        x = float(row["x"])
        y = float(row["y"])
        width = float(row["width"])
        height = float(row["height"])
        center_x = (x + width / 2.0) / image_size
        center_y = (y + height / 2.0) / image_size
        norm_width = width / image_size
        norm_height = height / image_size
        values = [0, center_x, center_y, norm_width, norm_height]
        lines.append(" ".join(f"{value:.8f}" if index else str(value) for index, value in enumerate(values)))
    return lines


def link_or_copy_image(source, destination, copy_images=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_images:
        shutil.copy2(source, destination)
    else:
        os.symlink(source, destination)


def export_split(dataset, split_name, output_dir, copy_images=False, image_size=1024.0):
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for patient_id in tqdm(dataset.image_ids, desc=f"export {split_name}"):
        patient_id = str(patient_id)
        source_image = dataset.image_directory / f"{patient_id}.png"
        target_image = images_dir / f"{patient_id}.png"
        link_or_copy_image(source_image, target_image, copy_images=copy_images)

        records = dataset.records_by_image_id[patient_id]
        yolo_lines = records_to_yolo_lines(records, image_size=image_size)
        label_path = labels_dir / f"{patient_id}.txt"
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        rows.append(
            {
                "patientId": patient_id,
                "split": split_name,
                "target": int(records["Target"].max()),
                "num_boxes": len(yolo_lines),
                "image_path": str(target_image),
                "label_path": str(label_path),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Export RSNA split to YOLO detection format.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument("--image-size", type=float, default=1024.0)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, valid_dataset, test_dataset = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    split_tables = [
        export_split(train_dataset, "train", output_dir, copy_images=args.copy_images, image_size=args.image_size),
        export_split(valid_dataset, "val", output_dir, copy_images=args.copy_images, image_size=args.image_size),
        export_split(test_dataset, "test", output_dir, copy_images=args.copy_images, image_size=args.image_size),
    ]
    manifest = pd.concat(split_tables, ignore_index=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    yaml_text = f"""path: {output_dir}
train: images/train
val: images/val
test: images/test
names:
  0: pneumonia_opacity
"""
    (output_dir / "rsna_yolo.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Exported YOLO dataset to {output_dir}")
    print(f"Wrote {output_dir / 'rsna_yolo.yaml'}")
    print(manifest.groupby(["split", "target"]).size().to_string())


if __name__ == "__main__":
    main()

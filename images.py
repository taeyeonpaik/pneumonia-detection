import argparse
import os
from pathlib import Path

import pydicom
from PIL import Image


def resolve_data_dir(data_dir=None):
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    if os.environ.get("RSNA_DATA_DIR"):
        return Path(os.environ["RSNA_DATA_DIR"]).expanduser().resolve()
    return Path(__file__).resolve().parent


def first_existing(data_dir, names):
    for name in names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    return data_dir / names[0]


def convert_directory(input_directory, output_directory):
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    if not input_directory.exists():
        raise FileNotFoundError(f"Missing DICOM input directory: {input_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)
    for filename in sorted(input_directory.iterdir()):
        if filename.suffix.lower() != ".dcm":
            continue
        dataset = pydicom.dcmread(str(filename))
        image = Image.fromarray(dataset.pixel_array)
        png_filename = filename.with_suffix(".png").name
        image.save(output_directory / png_filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RSNA DICOM images to PNG.")
    parser.add_argument("--data-dir", default=None, help="Directory containing RSNA image folders.")
    parser.add_argument("--train-input", default=None)
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--test-input", default=None)
    parser.add_argument("--test-output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    train_input = Path(args.train_input).expanduser().resolve() if args.train_input else first_existing(
        data_dir,
        ["train_images", "stage_2_train_images"],
    )
    train_output = Path(args.train_output).expanduser().resolve() if args.train_output else data_dir / "train_images_png"
    test_input = Path(args.test_input).expanduser().resolve() if args.test_input else first_existing(
        data_dir,
        ["test_images", "stage_2_test_images"],
    )
    test_output = Path(args.test_output).expanduser().resolve() if args.test_output else data_dir / "test_images_png"

    convert_directory(train_input, train_output)
    convert_directory(test_input, test_output)
    print(f"Saved train PNGs to: {train_output}")
    print(f"Saved test PNGs to: {test_output}")


if __name__ == "__main__":
    main()

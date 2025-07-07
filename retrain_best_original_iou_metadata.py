import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(command, dry_run=False):
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    if "--output-dir" in command:
        index = command.index("--output-dir") + 1
        if index < len(command):
            Path(command[index]).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def run_stage(name, command, expected_path=None, force=False, dry_run=False):
    print(f"\n=== {name} ===", flush=True)
    if expected_path and Path(expected_path).exists() and not force:
        print(f"Skipping {name}; found {expected_path}", flush=True)
        return
    run_command(command, dry_run=dry_run)


def find_hard_negative_file(output_root, requested=None):
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing Swin hard-negative file: {path}")
        return path
    candidates = [
        output_root / "outputs_best_swin_hardneg_mining" / "hard_negative_patient_ids.csv",
        output_root / "outputs_swin_hardneg_mining" / "hard_negative_patient_ids.csv",
    ]
    candidates.extend(sorted(output_root.glob("outputs*hardneg*/*hard_negative_patient_ids.csv")))
    return next((path for path in candidates if path.exists()), None)


def find_detector_hard_negative_file(output_root, requested=None):
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing detector hard-negative file: {path}")
        return path
    candidates = [
        output_root / "outputs_hardneg_mining_v2" / "hard_negative_ids.csv",
        output_root / "outputs_hardneg_mining" / "hard_negative_ids.csv",
    ]
    candidates.extend(sorted(output_root.glob("outputs*hardneg*/*hard_negative_ids.csv")))
    return next((path for path in candidates if path.exists()), None)


def select_best_detector(detector_runs):
    import pandas as pd

    rows = []
    for threshold, output_dir in detector_runs:
        metrics_path = output_dir / "validation_metrics.csv"
        checkpoint_path = output_dir / "frcnn_best_f1.pt"
        if not metrics_path.exists() or not checkpoint_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        if metrics.empty:
            continue
        best = metrics.sort_values(
            ["image_f1", "detector_precision", "mean_iou"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
        rows.append(
            {
                "box_fg_iou_threshold": threshold,
                "checkpoint": str(checkpoint_path),
                **best,
            }
        )
    if not rows:
        raise RuntimeError("No completed detector validation metrics were found.")
    table = pd.DataFrame(rows).sort_values(
        ["image_f1", "detector_precision", "mean_iou"],
        ascending=[False, False, False],
    )
    return table, table.iloc[0].to_dict()


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain the original-split best ensemble with lower detector IoU and DICOM metadata."
    )
    parser.add_argument("--data-dir", default="/home/ec2-user/rsna")
    parser.add_argument("--output-root", default="/home/ec2-user/rsna")
    parser.add_argument("--dicom-dir", default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--swin-hard-negative-file", default=None)
    parser.add_argument("--detector-hard-negative-file", default=None)
    parser.add_argument("--box-iou-thresholds", default="0.30,0.40")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--fusion-batch-size", type=int, default=16)
    parser.add_argument("--fusion-classifier-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    metadata_csv = (
        Path(args.metadata_csv).expanduser().resolve()
        if args.metadata_csv
        else data_dir / "dicom_metadata.csv"
    )

    if not metadata_csv.exists():
        extraction_command = [
            sys.executable,
            str(SCRIPT_DIR / "extract_dicom_metadata.py"),
            "--data-dir",
            str(data_dir),
            "--output-csv",
            str(metadata_csv),
        ]
        if args.dicom_dir:
            extraction_command.extend(["--dicom-dir", args.dicom_dir])
        run_stage(
            "Extract DICOM metadata",
            extraction_command,
            expected_path=metadata_csv,
            force=args.force,
            dry_run=args.dry_run,
        )

    experiment_root = output_root / "outputs_original_iou_metadata"
    experiment_root.mkdir(parents=True, exist_ok=True)
    detector_hard_negative_file = find_detector_hard_negative_file(
        output_root,
        args.detector_hard_negative_file,
    )
    detector_runs = []
    for threshold in parse_thresholds(args.box_iou_thresholds):
        threshold_tag = str(threshold).replace(".", "p")
        detector_dir = experiment_root / f"frcnn_boxiou_{threshold_tag}"
        detector_runs.append((threshold, detector_dir))
        detector_command = [
            sys.executable,
            str(SCRIPT_DIR / "testing_FRCNN.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(detector_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.detector_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--optimizer",
            "adamw",
            "--lr",
            "0.0001",
            "--weight-decay",
            "0.0005",
            "--augmentation",
            "cxr",
            "--box-fg-iou-threshold",
            str(threshold),
            "--box-bg-iou-threshold",
            str(threshold),
            "--seed",
            str(args.seed),
            "--no-save-epoch-checkpoints",
            "--no-save-latest-checkpoint",
            "--no-save-best-precision",
            "--no-save-precision-copy",
            "--weights-only-checkpoints",
        ]
        if detector_hard_negative_file is not None:
            detector_command.extend(
                [
                    "--hard-negative-file",
                    str(detector_hard_negative_file),
                    "--hard-negative-weight",
                    "2.0",
                ]
            )
        else:
            print("Warning: no original-split detector hard-negative file found.")
        run_stage(
            f"Train Faster R-CNN with RoI foreground IoU {threshold:g}",
            detector_command,
            expected_path=detector_dir / "frcnn_best_f1.pt",
            force=args.force,
            dry_run=args.dry_run,
        )

    swin_dir = experiment_root / "swin_t_hardneg_metadata"
    convnext_dir = experiment_root / "convnextv2_base_metadata"
    patch_swin_dir = experiment_root / "patch_swinv2_metadata"
    fusion_dir = experiment_root / "precision_gate_fusion"
    hard_negative_file = find_hard_negative_file(output_root, args.swin_hard_negative_file)

    common_classifier = [
        sys.executable,
        str(SCRIPT_DIR / "train_classifier.py"),
        "--data-dir",
        str(data_dir),
        "--metadata-csv",
        str(metadata_csv),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.classifier_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--augmentation",
        "cxr",
        "--loss",
        "asl",
        "--asl-gamma-neg",
        "3.0",
        "--asl-clip",
        "0.05",
        "--label-smoothing",
        "0.03",
        "--balanced-batches",
        "--mixup-alpha",
        "0.05",
        "--eval-tta-modes",
        "none,hflip",
        "--no-save-epoch-checkpoints",
        "--seed",
        str(args.seed),
    ]

    swin_command = common_classifier + [
        "--output-dir",
        str(swin_dir),
        "--architecture",
        "swin_t",
        "--image-size",
        "512",
        "--lr",
        "0.00002",
        "--weight-decay",
        "0.001",
    ]
    if hard_negative_file is not None:
        swin_command.extend(
            [
                "--hard-negative-file",
                str(hard_negative_file),
                "--hard-negative-weight",
                "1.5",
            ]
        )
    else:
        print("Warning: no original-split Swin hard-negative file found; training without extra weighting.")
    run_stage(
        "Train metadata-aware hard-negative Swin-T",
        swin_command,
        expected_path=swin_dir / "classifier_best.pt",
        force=args.force,
        dry_run=args.dry_run,
    )

    run_stage(
        "Train metadata-aware ConvNeXtV2-Base",
        common_classifier
        + [
            "--output-dir",
            str(convnext_dir),
            "--architecture",
            "timm_convnextv2_base",
            "--image-size",
            "512",
            "--lr",
            "0.00005",
            "--weight-decay",
            "0.0005",
        ],
        expected_path=convnext_dir / "classifier_best.pt",
        force=args.force,
        dry_run=args.dry_run,
    )

    run_stage(
        "Train metadata-aware patch SwinV2-T",
        common_classifier
        + [
            "--output-dir",
            str(patch_swin_dir),
            "--architecture",
            "patch_swin_v2_t",
            "--image-size",
            "512",
            "--lr",
            "0.00002",
            "--weight-decay",
            "0.001",
        ],
        expected_path=patch_swin_dir / "classifier_best.pt",
        force=args.force,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Dry run complete; detector selection and fusion were not executed.")
        return

    detector_table, best_detector = select_best_detector(detector_runs)
    import pandas as pd

    detector_table.to_csv(experiment_root / "detector_iou_validation_summary.csv", index=False)
    print("\nBest detector IoU setting by validation metrics:")
    print(pd.Series(best_detector).to_string())

    run_stage(
        "Run metadata-aware precision-gate fusion",
        [
            sys.executable,
            str(SCRIPT_DIR / "fusion_sweep.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(fusion_dir),
            "--metadata-csv",
            str(metadata_csv),
            "--classifier-checkpoints",
            ",".join(
                [
                    str(swin_dir / "classifier_best.pt"),
                    str(convnext_dir / "classifier_best.pt"),
                    str(patch_swin_dir / "classifier_best.pt"),
                ]
            ),
            "--detector-checkpoint",
            str(best_detector["checkpoint"]),
            "--classifier-tta-modes",
            "none,hflip",
            "--classifier-temperature-mode",
            "validation",
            "--methods",
            "precision_gate",
            "--min-precision",
            "0",
            "--min-recall",
            "0",
            "--box-score-thresholds",
            "0.50,0.54,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72",
            "--batch-size",
            str(args.fusion_batch_size),
            "--classifier-batch-size",
            str(args.fusion_classifier_batch_size),
            "--num-workers",
            "0",
            "--seed",
            str(args.seed),
        ],
        expected_path=fusion_dir / "fusion_sweep_test_metrics.csv",
        force=args.force,
        dry_run=args.dry_run,
    )
    print(f"\nFinal metrics: {fusion_dir / 'fusion_sweep_test_metrics.csv'}")


if __name__ == "__main__":
    main()

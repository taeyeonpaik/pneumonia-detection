#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def printable_command(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command, dry_run=False):
    print(f"\n$ {printable_command(command)}", flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in command], check=True)


def run_stage(name, command, expected_path=None, dry_run=False, force=False):
    print(f"\n{'=' * 12} {name} {'=' * 12}", flush=True)
    if expected_path is not None and Path(expected_path).exists() and not force:
        print(f"Skipping completed stage; found {expected_path}", flush=True)
        return
    run_command(command, dry_run=dry_run)
    if expected_path is not None and not dry_run and not Path(expected_path).exists():
        raise RuntimeError(f"Stage completed without expected output: {expected_path}")


def first_image_directory(data_dir):
    for name in ("train_images_png", "stage_2_train_images_png"):
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Missing train_images_png or stage_2_train_images_png")


def verify_dataset(data_dir, expected_images):
    labels_path = data_dir / "stage_2_train_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {labels_path}")
    image_dir = first_image_directory(data_dir)
    image_count = sum(1 for path in image_dir.iterdir() if path.is_file())
    labels = pd.read_csv(labels_path, usecols=["patientId"])
    patient_count = int(labels["patientId"].nunique())
    print(
        f"Dataset verification: images={image_count}, labeled patients={patient_count}, "
        f"image_dir={image_dir}",
        flush=True,
    )
    if image_count != expected_images or patient_count != expected_images:
        raise RuntimeError(
            f"Incomplete dataset: expected {expected_images} images/patients, "
            f"found {image_count}/{patient_count}"
        )


def pytorch_python():
    candidate = Path("/opt/pytorch/bin/python")
    return candidate if candidate.exists() else Path(sys.executable)


def detector_train_command(
    python,
    data_dir,
    output_dir,
    epochs,
    batch_size,
    num_workers,
    min_size,
    split_seed,
    model_seed,
    hard_negative_file=None,
    save_epoch_checkpoints=False,
):
    command = [
        python,
        SCRIPT_DIR / "testing_FRCNN.py",
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--num-workers",
        num_workers,
        "--optimizer",
        "adamw",
        "--lr",
        "0.0001",
        "--weight-decay",
        "0.0005",
        "--augmentation",
        "cxr",
        "--min-size",
        min_size,
        "--box-fg-iou-threshold",
        "0.5",
        "--box-bg-iou-threshold",
        "0.5",
        "--seed",
        split_seed,
        "--split-seed",
        split_seed,
        "--model-seed",
        model_seed,
        "--no-save-latest-checkpoint",
        "--no-save-best-precision",
        "--no-save-precision-copy",
        "--weights-only-checkpoints",
    ]
    if not save_epoch_checkpoints:
        command.append("--no-save-epoch-checkpoints")
    if hard_negative_file is not None:
        command.extend(["--hard-negative-file", hard_negative_file, "--hard-negative-weight", "2.0"])
    return command


def classifier_common(python, data_dir, epochs, batch_size, num_workers, seed):
    return [
        python,
        SCRIPT_DIR / "train_classifier.py",
        "--data-dir",
        data_dir,
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--num-workers",
        num_workers,
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
        "--thresholds",
        "0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.65,0.66,0.68,0.70,0.72,0.75,0.80",
        "--no-save-epoch-checkpoints",
        "--seed",
        seed,
    ]


def fusion_command(
    python,
    data_dir,
    output_dir,
    metadata_csv,
    classifier_checkpoint,
    detector_checkpoint,
    detector_min_size,
    fusion_batch_size,
    classifier_batch_size,
    split_seed,
):
    return [
        python,
        SCRIPT_DIR / "fusion_sweep.py",
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--metadata-csv",
        metadata_csv,
        "--classifier-checkpoints",
        classifier_checkpoint,
        "--detector-checkpoint",
        detector_checkpoint,
        "--detector-min-size",
        detector_min_size,
        "--detector-ensemble-method",
        "none",
        "--classifier-tta-modes",
        "none,hflip",
        "--classifier-temperature-mode",
        "validation",
        "--methods",
        "precision_gate",
        "--box-score-thresholds",
        "0.40,0.44,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80",
        "--gate-classifier-agree-thresholds",
        "0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.65,0.66,0.68,0.70,0.72",
        "--gate-detector-required-thresholds",
        "0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.65",
        "--gate-classifier-override-thresholds",
        "0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.96",
        "--gate-detector-override-thresholds",
        "0.54,0.58,0.60,0.62,0.65,0.68,0.70,0.75,0.80",
        "--gate-detector-classifier-max-thresholds",
        "0.20,0.25,0.30,0.35,0.40,0.50,1.00",
        "--metric-iou-threshold",
        "0.5",
        "--batch-size",
        fusion_batch_size,
        "--classifier-batch-size",
        classifier_batch_size,
        "--num-workers",
        "0",
        "--min-precision",
        "0",
        "--min-recall",
        "0",
        "--seed",
        split_seed,
    ]


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the best original-split model and generate FP/FN error analysis."
    )
    parser.add_argument("--data-dir", default="/home/ec2-user/rsna")
    parser.add_argument("--output-root", default="/home/ec2-user/rsna/outputs_reproduce_best_analysis")
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--expected-images", type=int, default=26684)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--fusion-batch-size", type=int, default=16)
    parser.add_argument("--fusion-classifier-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--detector-min-size", type=int, default=1024)
    parser.add_argument("--detector-checkpoint", default=None)
    parser.add_argument("--swin-checkpoint", default=None)
    parser.add_argument("--force-fusion", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    python = pytorch_python()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        verify_dataset(data_dir, args.expected_images)

    metadata_csv = Path(args.metadata_csv).expanduser().resolve() if args.metadata_csv else data_dir / "dicom_metadata.csv"
    run_stage(
        "Ensure DICOM metadata exists",
        [
            python,
            SCRIPT_DIR / "extract_dicom_metadata.py",
            "--data-dir",
            data_dir,
            "--dicom-dir",
            data_dir / "train_images",
            "--output-csv",
            metadata_csv,
            "--workers",
            "16",
        ],
        expected_path=metadata_csv,
        dry_run=args.dry_run,
    )

    detector_checkpoint = Path(args.detector_checkpoint).expanduser().resolve() if args.detector_checkpoint else None
    if detector_checkpoint is None:
        detector_checkpoint = first_existing(
            [
                data_dir / "outputs_hardneg2_adamw_b16_lr0001" / "frcnn_epoch_7.pt",
                output_root / "detector_hardneg2_adamw_b16_lr0001" / "frcnn_epoch_7.pt",
                data_dir / "outputs_overnight_rebuild" / "detector_a_final" / "frcnn_epoch_7.pt",
            ]
        )

    if detector_checkpoint is None:
        detector_base_dir = output_root / "detector_base_adamw_b16_lr0001"
        detector_hn_dir = output_root / "detector_hard_negatives"
        detector_hn_file = detector_hn_dir / "hard_negative_ids.csv"
        detector_final_dir = output_root / "detector_hardneg2_adamw_b16_lr0001"
        run_stage(
            "Train base Faster R-CNN for detector hard-negative mining",
            detector_train_command(
                python,
                data_dir,
                detector_base_dir,
                args.epochs,
                args.detector_batch_size,
                args.num_workers,
                args.detector_min_size,
                args.split_seed,
                args.model_seed,
            ),
            expected_path=detector_base_dir / "frcnn_best_f1.pt",
            dry_run=args.dry_run,
        )
        run_stage(
            "Mine detector hard negatives on the seed-42 train split",
            [
                python,
                SCRIPT_DIR / "mine_hard_negatives.py",
                "--data-dir",
                data_dir,
                "--output-dir",
                detector_hn_dir,
                "--checkpoint",
                detector_base_dir / "frcnn_best_f1.pt",
                "--batch-size",
                args.detector_batch_size,
                "--num-workers",
                args.num_workers,
                "--min-size",
                args.detector_min_size,
                "--seed",
                args.split_seed,
                "--score-threshold",
                "0.5",
                "--max-hard-negatives",
                "2000",
            ],
            expected_path=detector_hn_file,
            dry_run=args.dry_run,
        )
        run_stage(
            "Train original-style hard-negative Faster R-CNN",
            detector_train_command(
                python,
                data_dir,
                detector_final_dir,
                args.epochs,
                args.detector_batch_size,
                args.num_workers,
                args.detector_min_size,
                args.split_seed,
                args.model_seed,
                hard_negative_file=detector_hn_file,
                save_epoch_checkpoints=True,
            ),
            expected_path=detector_final_dir / "frcnn_epoch_7.pt",
            dry_run=args.dry_run,
        )
        detector_checkpoint = detector_final_dir / "frcnn_epoch_7.pt"

    if detector_checkpoint is None or (not args.dry_run and not detector_checkpoint.exists()):
        raise FileNotFoundError(f"Could not resolve detector checkpoint: {detector_checkpoint}")
    print(f"Using detector checkpoint: {detector_checkpoint}", flush=True)

    swin_checkpoint = Path(args.swin_checkpoint).expanduser().resolve() if args.swin_checkpoint else None
    if swin_checkpoint is None:
        swin_checkpoint = first_existing(
            [
                output_root / "swin_t_hardneg_metadata" / "classifier_best.pt",
                data_dir / "outputs_overnight_rebuild" / "swin_t_hardneg_metadata" / "classifier_best.pt",
                data_dir / "outputs_original_iou_metadata" / "swin_t_hardneg_metadata" / "classifier_best.pt",
            ]
        )

    if swin_checkpoint is None:
        swin_base_dir = output_root / "swin_t_base"
        swin_hn_dir = output_root / "swin_hard_negatives"
        swin_hn_file = swin_hn_dir / "hard_negative_patient_ids.csv"
        swin_metadata_dir = output_root / "swin_t_hardneg_metadata"
        classifier_base = classifier_common(
            python,
            data_dir,
            args.epochs,
            args.classifier_batch_size,
            args.num_workers,
            args.split_seed,
        )
        run_stage(
            "Train base Swin-T for classifier hard-negative mining",
            classifier_base
            + [
                "--output-dir",
                swin_base_dir,
                "--architecture",
                "swin_t",
                "--image-size",
                "512",
                "--lr",
                "0.00002",
                "--weight-decay",
                "0.001",
            ],
            expected_path=swin_base_dir / "classifier_best.pt",
            dry_run=args.dry_run,
        )
        run_stage(
            "Mine Swin-T classifier hard negatives",
            [
                python,
                SCRIPT_DIR / "mine_swin_hard_negatives.py",
                "--data-dir",
                data_dir,
                "--checkpoint",
                swin_base_dir / "classifier_best.pt",
                "--output-dir",
                swin_hn_dir,
                "--batch-size",
                args.classifier_batch_size,
                "--num-workers",
                args.num_workers,
                "--seed",
                args.split_seed,
                "--tta-modes",
                "none,hflip",
                "--max-hard-negatives",
                "1500",
            ],
            expected_path=swin_hn_file,
            dry_run=args.dry_run,
        )
        run_stage(
            "Train metadata-aware hard-negative Swin-T",
            classifier_base
            + [
                "--output-dir",
                swin_metadata_dir,
                "--architecture",
                "swin_t",
                "--image-size",
                "512",
                "--lr",
                "0.00002",
                "--weight-decay",
                "0.001",
                "--metadata-csv",
                metadata_csv,
                "--hard-negative-file",
                swin_hn_file,
                "--hard-negative-weight",
                "1.5",
            ],
            expected_path=swin_metadata_dir / "classifier_best.pt",
            dry_run=args.dry_run,
        )
        swin_checkpoint = swin_metadata_dir / "classifier_best.pt"

    if swin_checkpoint is None or (not args.dry_run and not swin_checkpoint.exists()):
        raise FileNotFoundError(f"Could not resolve Swin checkpoint: {swin_checkpoint}")
    print(f"Using metadata-aware Swin checkpoint: {swin_checkpoint}", flush=True)

    fusion_dir = output_root / "fusion_original_best_reproduction"
    run_stage(
        "Run metadata-aware Swin + Faster R-CNN precision-gate fusion",
        fusion_command(
            python,
            data_dir,
            fusion_dir,
            metadata_csv,
            swin_checkpoint,
            detector_checkpoint,
            args.detector_min_size,
            args.fusion_batch_size,
            args.fusion_classifier_batch_size,
            args.split_seed,
        ),
        expected_path=fusion_dir / "fusion_sweep_test_metrics.csv",
        dry_run=args.dry_run,
        force=args.force_fusion,
    )

    analysis_dir = output_root / "error_analysis"
    run_stage(
        "Analyze FP/FN behavior for reproduced best model",
        [
            python,
            SCRIPT_DIR / "analyze_best_model_errors.py",
            "--fusion-dir",
            fusion_dir,
            "--metadata-csv",
            metadata_csv,
            "--output-dir",
            analysis_dir,
        ],
        expected_path=analysis_dir / "error_analysis_summary.json",
        dry_run=args.dry_run,
        force=args.force_analysis,
    )

    if args.dry_run:
        print("Dry run complete.", flush=True)
        return

    selected_metrics = pd.read_csv(fusion_dir / "fusion_sweep_test_metrics.csv").iloc[0].to_dict()
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reproduce best original-split metadata-aware Swin-T + Faster R-CNN precision-gate model and analyze FP/FN cases.",
        "detector_checkpoint": str(detector_checkpoint),
        "swin_checkpoint": str(swin_checkpoint),
        "fusion_dir": str(fusion_dir),
        "analysis_dir": str(analysis_dir),
        "selected_test_metrics": selected_metrics,
    }
    (output_root / "reproduce_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "PIPELINE_COMPLETE").write_text(
        f"Completed {summary['completed_at']}\n"
        f"Test F1: {selected_metrics.get('image_f1')}\n"
        f"Fusion dir: {fusion_dir}\n"
        f"Analysis dir: {analysis_dir}\n",
        encoding="utf-8",
    )
    print("\nReproduction + error analysis complete.", flush=True)
    print(pd.Series(selected_metrics).to_string(), flush=True)


if __name__ == "__main__":
    main()

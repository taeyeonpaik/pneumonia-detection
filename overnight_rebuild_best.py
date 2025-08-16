#!/usr/bin/env python3

import argparse
import json
import shlex
import shutil
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


def run_stage(name, command, expected_path=None, dry_run=False):
    print(f"\n{'=' * 12} {name} {'=' * 12}", flush=True)
    if expected_path is not None and Path(expected_path).exists():
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
        command.extend(
            [
                "--hard-negative-file",
                hard_negative_file,
                "--hard-negative-weight",
                "2.0",
            ]
        )
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


def run_detector_checkpoint_sweep(
    python,
    data_dir,
    detector_dir,
    batch_size,
    num_workers,
    min_size,
    split_seed,
    dry_run,
):
    validation_path = detector_dir / "checkpoint_threshold_sweep_validation.csv"
    run_stage(
        f"Validation checkpoint/threshold sweep for {detector_dir.name}",
        [
            python,
            SCRIPT_DIR / "checkpoint_threshold_sweep.py",
            "--data-dir",
            data_dir,
            "--output-dir",
            detector_dir,
            "--checkpoint-pattern",
            "frcnn_epoch_*.pt",
            "--batch-size",
            batch_size,
            "--num-workers",
            num_workers,
            "--min-size",
            min_size,
            "--score-thresholds",
            "0.30,0.35,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.55,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.75,0.80,0.85",
            "--metric-iou-threshold",
            "0.5",
            "--seed",
            split_seed,
        ],
        expected_path=validation_path,
        dry_run=dry_run,
    )
    if dry_run:
        return detector_dir / "frcnn_epoch_SELECTED.pt", {}
    table = pd.read_csv(validation_path)
    if table.empty:
        raise RuntimeError(f"Empty validation sweep: {validation_path}")
    best = table.sort_values(
        ["image_f1", "detector_precision", "mean_iou"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    checkpoint = Path(best["checkpoint_path"])
    if not checkpoint.exists():
        checkpoint = detector_dir / str(best["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint}")
    return checkpoint, best


def preserve_selected_checkpoint(source, destination, keep_epochs):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    if not keep_epochs:
        for checkpoint in source.parent.glob("frcnn_epoch_*.pt"):
            if checkpoint.resolve() != destination.resolve():
                checkpoint.unlink()


def backup_artifact(source, backup_dir, name):
    destination = backup_dir / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def sync_backup_to_s3(backup_dir, s3_uri, dry_run=False):
    if not s3_uri:
        return
    if shutil.which("aws") is None:
        print("Warning: --s3-uri supplied but AWS CLI is unavailable; skipping S3 backup", flush=True)
        return
    command = ["aws", "s3", "sync", backup_dir, s3_uri.rstrip("/") + "/overnight_rebuild"]
    print(f"\n$ {printable_command(command)}", flush=True)
    if not dry_run:
        result = subprocess.run([str(part) for part in command], check=False)
        if result.returncode != 0:
            print("Warning: S3 backup failed; the training pipeline will continue", flush=True)


def fusion_command(
    python,
    data_dir,
    output_dir,
    metadata_csv,
    classifier_checkpoints,
    detector_checkpoints,
    detector_min_sizes,
    wbf_iou,
    detector_ensemble_method,
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
        ",".join(str(path) for path in classifier_checkpoints),
        "--detector-checkpoint",
        ",".join(str(path) for path in detector_checkpoints),
        "--detector-min-sizes",
        ",".join(str(size) for size in detector_min_sizes),
        "--detector-min-size-mode",
        "pair",
        "--detector-ensemble-method",
        detector_ensemble_method,
        "--detector-ensemble-iou",
        wbf_iou,
        "--detector-ensemble-score-threshold",
        "0.001",
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


def validation_result(output_dir, candidate_name, settings):
    path = output_dir / "fusion_sweep_validation.csv"
    table = pd.read_csv(path)
    if table.empty:
        raise RuntimeError(f"Empty fusion validation table: {path}")
    best = table.sort_values(
        ["image_f1", "image_precision", "detector_precision", "mean_iou"],
        ascending=[False, False, False, False],
    ).iloc[0].to_dict()
    return {
        "candidate": candidate_name,
        "output_dir": str(output_dir),
        **settings,
        **{f"validation_{key}": value for key, value in best.items()},
    }


def parse_values(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable overnight rebuild of the best RSNA model and WBF ensemble.")
    parser.add_argument("--data-dir", default="/home/ec2-user/rsna")
    parser.add_argument("--output-root", default="/home/ec2-user/rsna/outputs_overnight_rebuild")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--fusion-batch-size", type=int, default=8)
    parser.add_argument("--fusion-classifier-batch-size", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--detector-a-model-seed", type=int, default=42)
    parser.add_argument("--detector-b-model-seed", type=int, default=1337)
    parser.add_argument("--detector-a-min-size", type=int, default=1024)
    parser.add_argument("--detector-b-min-size", type=int, default=896)
    parser.add_argument("--wbf-ious", default="0.40,0.50,0.55,0.60,0.65")
    parser.add_argument("--expected-images", type=int, default=26684)
    parser.add_argument("--s3-uri", default=None)
    parser.add_argument(
        "--cleanup-epoch-checkpoints",
        action="store_true",
        help="Delete non-selected detector epoch checkpoints after validation selection.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pytorch_python = Path("/opt/pytorch/bin/python")
    python = pytorch_python if pytorch_python.exists() else Path(sys.executable)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    backup_dir = output_root / "selected_checkpoints"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        verify_dataset(data_dir, args.expected_images)

    metadata_csv = data_dir / "dicom_metadata.csv"
    run_stage(
        "Extract DICOM metadata",
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

    detector_a_base = output_root / "detector_a_base"
    detector_hn_dir = output_root / "detector_hard_negatives"
    detector_hn_file = detector_hn_dir / "hard_negative_ids.csv"
    detector_a_dir = output_root / "detector_a_final"
    detector_b_dir = output_root / "detector_b_scale896_seed1337"

    run_stage(
        "Train detector A base",
        detector_train_command(
            python,
            data_dir,
            detector_a_base,
            args.epochs,
            args.detector_batch_size,
            args.num_workers,
            args.detector_a_min_size,
            args.split_seed,
            args.detector_a_model_seed,
        ),
        expected_path=detector_a_base / "frcnn_best_f1.pt",
        dry_run=args.dry_run,
    )
    run_stage(
        "Mine detector hard negatives from the seed-42 training split",
        [
            python,
            SCRIPT_DIR / "mine_hard_negatives.py",
            "--data-dir",
            data_dir,
            "--output-dir",
            detector_hn_dir,
            "--checkpoint",
            detector_a_base / "frcnn_best_f1.pt",
            "--batch-size",
            args.detector_batch_size,
            "--num-workers",
            args.num_workers,
            "--min-size",
            args.detector_a_min_size,
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
        "Train detector A final at 1024",
        detector_train_command(
            python,
            data_dir,
            detector_a_dir,
            args.epochs,
            args.detector_batch_size,
            args.num_workers,
            args.detector_a_min_size,
            args.split_seed,
            args.detector_a_model_seed,
            hard_negative_file=detector_hn_file,
            save_epoch_checkpoints=True,
        ),
        expected_path=detector_a_dir / f"frcnn_epoch_{args.epochs}.pt",
        dry_run=args.dry_run,
    )
    detector_a_source, detector_a_validation = run_detector_checkpoint_sweep(
        python,
        data_dir,
        detector_a_dir,
        args.detector_batch_size,
        args.num_workers,
        args.detector_a_min_size,
        args.split_seed,
        args.dry_run,
    )
    detector_a_selected = backup_dir / "detector_a_selected.pt"
    if not args.dry_run:
        preserve_selected_checkpoint(
            detector_a_source,
            detector_a_selected,
            keep_epochs=not args.cleanup_epoch_checkpoints,
        )

    run_stage(
        "Train independent detector B at 896 with model seed 1337",
        detector_train_command(
            python,
            data_dir,
            detector_b_dir,
            args.epochs,
            args.detector_batch_size,
            args.num_workers,
            args.detector_b_min_size,
            args.split_seed,
            args.detector_b_model_seed,
            hard_negative_file=detector_hn_file,
            save_epoch_checkpoints=True,
        ),
        expected_path=detector_b_dir / f"frcnn_epoch_{args.epochs}.pt",
        dry_run=args.dry_run,
    )
    detector_b_source, detector_b_validation = run_detector_checkpoint_sweep(
        python,
        data_dir,
        detector_b_dir,
        args.detector_batch_size,
        args.num_workers,
        args.detector_b_min_size,
        args.split_seed,
        args.dry_run,
    )
    detector_b_selected = backup_dir / "detector_b_selected.pt"
    if not args.dry_run:
        preserve_selected_checkpoint(
            detector_b_source,
            detector_b_selected,
            keep_epochs=not args.cleanup_epoch_checkpoints,
        )

    swin_base_dir = output_root / "swin_t_base"
    swin_hn_dir = output_root / "swin_hard_negatives"
    swin_hn_file = swin_hn_dir / "hard_negative_patient_ids.csv"
    swin_metadata_dir = output_root / "swin_t_hardneg_metadata"
    convnext_dir = output_root / "convnextv2_base_balanced_mixup"
    classifier_base = classifier_common(
        python,
        data_dir,
        args.epochs,
        args.classifier_batch_size,
        args.num_workers,
        args.split_seed,
    )

    run_stage(
        "Train base Swin-T for hard-negative mining",
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
        "Mine Swin-T hard negatives",
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
    run_stage(
        "Train best ConvNeXtV2-Base classifier",
        classifier_base
        + [
            "--output-dir",
            convnext_dir,
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
        dry_run=args.dry_run,
    )

    swin_checkpoint = swin_metadata_dir / "classifier_best.pt"
    convnext_checkpoint = convnext_dir / "classifier_best.pt"
    if not args.dry_run:
        backup_artifact(swin_checkpoint, backup_dir, "metadata_swin_t_best.pt")
        backup_artifact(convnext_checkpoint, backup_dir, "convnextv2_base_best.pt")
        sync_backup_to_s3(backup_dir, args.s3_uri)

    fusion_root = output_root / "fusion"
    baseline_dir = fusion_root / "baseline_detector_a_swin"
    run_stage(
        "Reproduce the best detector-A + metadata-Swin baseline",
        fusion_command(
            python,
            data_dir,
            baseline_dir,
            metadata_csv,
            [swin_checkpoint],
            [detector_a_selected],
            [args.detector_a_min_size],
            0.55,
            "none",
            args.fusion_batch_size,
            args.fusion_classifier_batch_size,
            args.split_seed,
        ),
        expected_path=baseline_dir / "fusion_sweep_validation.csv",
        dry_run=args.dry_run,
    )

    candidates = []
    if not args.dry_run:
        candidates.append(
            validation_result(
                baseline_dir,
                "baseline_detector_a_metadata_swin",
                {"detector_ensemble_method": "none", "wbf_iou": None, "classifiers": "metadata_swin"},
            )
        )

    classifier_sets = [
        ("metadata_swin", [swin_checkpoint]),
        ("metadata_swin_convnextv2", [swin_checkpoint, convnext_checkpoint]),
    ]
    for classifier_name, classifier_checkpoints in classifier_sets:
        for wbf_iou in parse_values(args.wbf_ious):
            iou_tag = str(wbf_iou).replace(".", "p")
            output_dir = fusion_root / f"wbf_{classifier_name}_iou_{iou_tag}"
            run_stage(
                f"WBF validation: {classifier_name}, IoU={wbf_iou:g}",
                fusion_command(
                    python,
                    data_dir,
                    output_dir,
                    metadata_csv,
                    classifier_checkpoints,
                    [detector_a_selected, detector_b_selected],
                    [args.detector_a_min_size, args.detector_b_min_size],
                    wbf_iou,
                    "wbf",
                    args.fusion_batch_size,
                    args.fusion_classifier_batch_size,
                    args.split_seed,
                ),
                expected_path=output_dir / "fusion_sweep_validation.csv",
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                candidates.append(
                    validation_result(
                        output_dir,
                        f"wbf_{classifier_name}_iou_{wbf_iou:g}",
                        {
                            "detector_ensemble_method": "wbf",
                            "wbf_iou": wbf_iou,
                            "classifiers": classifier_name,
                        },
                    )
                )

    if args.dry_run:
        print("\nDry run complete.", flush=True)
        return

    candidate_table = pd.DataFrame(candidates).sort_values(
        ["validation_image_f1", "validation_image_precision", "validation_detector_precision"],
        ascending=[False, False, False],
    )
    candidate_table.to_csv(output_root / "validation_candidate_summary.csv", index=False)
    best_candidate = candidate_table.iloc[0].to_dict()
    best_output_dir = Path(best_candidate["output_dir"])
    test_table = pd.read_csv(best_output_dir / "fusion_sweep_test_metrics.csv")
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "All checkpoint, WBF IoU, box confidence, and fusion choices ranked on validation only.",
        "best_candidate": best_candidate,
        "selected_candidate_test_metrics": test_table.iloc[0].to_dict(),
        "detector_a_validation_selection": detector_a_validation,
        "detector_b_validation_selection": detector_b_validation,
    }
    with (output_root / "overnight_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, default=str)
    (output_root / "PIPELINE_COMPLETE").write_text(
        f"Completed {summary['completed_at']}\nBest candidate: {best_candidate['candidate']}\n",
        encoding="utf-8",
    )
    sync_backup_to_s3(backup_dir, args.s3_uri)
    print("\nOvernight rebuild complete.", flush=True)
    print(pd.Series(best_candidate).to_string(), flush=True)
    print("\nSelected candidate held-out test metrics:", flush=True)
    print(test_table.iloc[0].to_string(), flush=True)


if __name__ == "__main__":
    main()

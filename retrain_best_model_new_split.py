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
        output_index = command.index("--output-dir") + 1
        if output_index < len(command):
            Path(command[output_index]).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def checkpoint_exists(path):
    return Path(path).expanduser().exists()


def run_stage(name, command, expected_path=None, force=False, dry_run=False):
    print(f"\n=== {name} ===", flush=True)
    if expected_path and checkpoint_exists(expected_path) and not force:
        print(f"Skipping {name}; found {expected_path}", flush=True)
        return
    run_command(command, dry_run=dry_run)


def first_existing(paths):
    for path in paths:
        if Path(path).exists():
            return str(path)
    return str(paths[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the current best RSNA fusion model on a new random split. "
            "This runs a fresh detector, hard-negative mining, classifiers, and precision-gate fusion."
        )
    )
    parser.add_argument("--data-dir", default="/home/ec2-user/rsna")
    parser.add_argument("--output-root", default="/home/ec2-user/rsna")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--fusion-batch-size", type=int, default=2)
    parser.add_argument("--fusion-classifier-batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-hard-negative-mining",
        action="store_true",
        help="Train only one detector/Swin pass without mining new hard negatives. Faster but less faithful to the best model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    seed_tag = f"seed{args.seed}"

    detector_base_dir = output_root / f"outputs_{seed_tag}_frcnn_base_adamw_b16_lr0001"
    detector_mining_dir = output_root / f"outputs_{seed_tag}_frcnn_hardneg_mining"
    detector_final_dir = output_root / f"outputs_{seed_tag}_frcnn_hardneg_adamw_b16_lr0001"

    swin_base_dir = output_root / f"outputs_{seed_tag}_swin_t_base_b8_512_lr00002"
    swin_mining_dir = output_root / f"outputs_{seed_tag}_swin_hardneg_mining"
    swin_final_dir = output_root / f"outputs_{seed_tag}_swin_t_hardneg_b8_512_lr00002"
    convnext_dir = output_root / f"outputs_{seed_tag}_convnextv2_base_balanced_mixup"
    patch_swin_dir = output_root / f"outputs_{seed_tag}_patch_swinv2_balanced"
    fusion_dir = output_root / f"outputs_{seed_tag}_best_precision_gate_fusion"

    common_detector = [
        sys.executable,
        str(SCRIPT_DIR / "testing_FRCNN.py"),
        "--data-dir",
        str(data_dir),
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
        "--seed",
        str(args.seed),
        "--no-save-epoch-checkpoints",
        "--no-save-latest-checkpoint",
        "--no-save-best-precision",
        "--no-save-precision-copy",
        "--weights-only-checkpoints",
    ]
    run_stage(
        "Train base Faster R-CNN on new split",
        common_detector + ["--output-dir", str(detector_base_dir)],
        expected_path=detector_base_dir / "frcnn_best_f1.pt",
        force=args.force,
        dry_run=args.dry_run,
    )

    detector_checkpoint_for_mining = first_existing(
        [
            detector_base_dir / "frcnn_best_f1.pt",
            detector_base_dir / "frcnn_best.pt",
            detector_base_dir / "frcnn_latest.pt",
        ]
    )
    detector_hard_negative_file = detector_mining_dir / "hard_negative_ids.csv"
    if not args.skip_hard_negative_mining:
        run_stage(
            "Mine detector hard negatives on new train split",
            [
                sys.executable,
                str(SCRIPT_DIR / "mine_hard_negatives.py"),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(detector_mining_dir),
                "--checkpoint",
                detector_checkpoint_for_mining,
                "--batch-size",
                str(args.detector_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--score-threshold",
                "0.5",
                "--max-hard-negatives",
                "2000",
            ],
            expected_path=detector_hard_negative_file,
            force=args.force,
            dry_run=args.dry_run,
        )
        run_stage(
            "Retrain Faster R-CNN with new hard negatives",
            common_detector
            + [
                "--output-dir",
                str(detector_final_dir),
                "--hard-negative-file",
                str(detector_hard_negative_file),
                "--hard-negative-weight",
                "2.0",
            ],
            expected_path=detector_final_dir / "frcnn_best_f1.pt",
            force=args.force,
            dry_run=args.dry_run,
        )
    else:
        detector_final_dir = detector_base_dir

    detector_checkpoint = first_existing(
        [
            detector_final_dir / "frcnn_best_f1.pt",
            detector_final_dir / "frcnn_best.pt",
            detector_final_dir / "frcnn_latest.pt",
        ]
    )

    base_classifier_args = [
        sys.executable,
        str(SCRIPT_DIR / "train_classifier.py"),
        "--data-dir",
        str(data_dir),
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

    run_stage(
        "Train base Swin classifier for hard-negative mining",
        base_classifier_args
        + [
            "--output-dir",
            str(swin_base_dir),
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
        force=args.force,
        dry_run=args.dry_run,
    )

    swin_hard_negative_file = swin_mining_dir / "hard_negative_patient_ids.csv"
    if not args.skip_hard_negative_mining:
        run_stage(
            "Mine Swin classifier hard negatives on new train split",
            [
                sys.executable,
                str(SCRIPT_DIR / "mine_swin_hard_negatives.py"),
                "--data-dir",
                str(data_dir),
                "--checkpoint",
                str(swin_base_dir / "classifier_best.pt"),
                "--output-dir",
                str(swin_mining_dir),
                "--batch-size",
                str(args.classifier_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--tta-modes",
                "none,hflip",
                "--max-hard-negatives",
                "1500",
            ],
            expected_path=swin_hard_negative_file,
            force=args.force,
            dry_run=args.dry_run,
        )

    if args.skip_hard_negative_mining:
        swin_final_dir = swin_base_dir
    else:
        swin_final_command = base_classifier_args + [
            "--output-dir",
            str(swin_final_dir),
            "--architecture",
            "swin_t",
            "--image-size",
            "512",
            "--lr",
            "0.00002",
            "--weight-decay",
            "0.001",
        ]
        swin_final_command.extend(
            [
                "--hard-negative-file",
                str(swin_hard_negative_file),
                "--hard-negative-weight",
                "1.5",
            ]
        )
        run_stage(
            "Train final Swin classifier on new split",
            swin_final_command,
            expected_path=swin_final_dir / "classifier_best.pt",
            force=args.force,
            dry_run=args.dry_run,
        )

    run_stage(
        "Train ConvNeXtV2 classifier on new split",
        base_classifier_args
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
        "Train patch SwinV2 classifier on new split",
        base_classifier_args
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

    run_stage(
        "Run precision-gate fusion on new split",
        [
            sys.executable,
            str(SCRIPT_DIR / "fusion_sweep.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(fusion_dir),
            "--classifier-checkpoints",
            ",".join(
                [
                    str(swin_final_dir / "classifier_best.pt"),
                    str(convnext_dir / "classifier_best.pt"),
                    str(patch_swin_dir / "classifier_best.pt"),
                ]
            ),
            "--detector-checkpoint",
            detector_checkpoint,
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

    print("\nDone. Final metrics should be in:")
    print(fusion_dir / "fusion_sweep_test_metrics.csv")


if __name__ == "__main__":
    main()

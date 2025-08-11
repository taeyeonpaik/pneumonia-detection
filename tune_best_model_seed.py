import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


DETECTOR_CONFIGS = [
    {"name": "lr00005_hn2", "lr": 0.00005, "hard_negative_weight": 2.0},
    {"name": "lr0001_hn1", "lr": 0.0001, "hard_negative_weight": 1.0},
    {"name": "lr0001_hn2", "lr": 0.0001, "hard_negative_weight": 2.0},
    {"name": "lr0002_hn2", "lr": 0.0002, "hard_negative_weight": 2.0},
]


CLASSIFIER_CONFIGS = {
    "swin": {
        "architecture": "swin_t",
        "image_size": 512,
        "configs": [
            {"name": "lr00001_g3", "lr": 0.00001, "asl_gamma_neg": 3.0, "weight_decay": 0.001},
            {"name": "lr00002_g2", "lr": 0.00002, "asl_gamma_neg": 2.0, "weight_decay": 0.001},
            {"name": "lr00002_g3", "lr": 0.00002, "asl_gamma_neg": 3.0, "weight_decay": 0.001},
            {"name": "lr00002_g4", "lr": 0.00002, "asl_gamma_neg": 4.0, "weight_decay": 0.001},
            {"name": "lr00005_g3", "lr": 0.00005, "asl_gamma_neg": 3.0, "weight_decay": 0.001},
        ],
    },
    "convnextv2": {
        "architecture": "timm_convnextv2_base",
        "image_size": 512,
        "configs": [
            {"name": "lr00002_g3", "lr": 0.00002, "asl_gamma_neg": 3.0, "weight_decay": 0.0005},
            {"name": "lr00005_g2", "lr": 0.00005, "asl_gamma_neg": 2.0, "weight_decay": 0.0005},
            {"name": "lr00005_g3", "lr": 0.00005, "asl_gamma_neg": 3.0, "weight_decay": 0.0005},
            {"name": "lr00005_g4", "lr": 0.00005, "asl_gamma_neg": 4.0, "weight_decay": 0.0005},
            {"name": "lr0001_g3", "lr": 0.0001, "asl_gamma_neg": 3.0, "weight_decay": 0.0005},
        ],
    },
    "patch_swinv2": {
        "architecture": "patch_swin_v2_t",
        "image_size": 512,
        "configs": [
            {"name": "lr00001_g3", "lr": 0.00001, "asl_gamma_neg": 3.0, "weight_decay": 0.001},
            {"name": "lr00002_g2", "lr": 0.00002, "asl_gamma_neg": 2.0, "weight_decay": 0.001},
            {"name": "lr00002_g3", "lr": 0.00002, "asl_gamma_neg": 3.0, "weight_decay": 0.001},
            {"name": "lr00002_g4", "lr": 0.00002, "asl_gamma_neg": 4.0, "weight_decay": 0.001},
            {"name": "lr00005_g3", "lr": 0.00005, "asl_gamma_neg": 3.0, "weight_decay": 0.001},
        ],
    },
}


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


def run_stage(name, command, expected_path=None, force=False, dry_run=False):
    print(f"\n=== {name} ===", flush=True)
    if expected_path and Path(expected_path).exists() and not force:
        print(f"Skipping {name}; found {expected_path}", flush=True)
        return
    run_command(command, dry_run=dry_run)


def read_best_classifier_row(output_dir, model_name, config_name):
    import pandas as pd

    metrics_path = output_dir / "classifier_validation_metrics.csv"
    checkpoint_path = output_dir / "classifier_best.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None
    row = metrics.sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    row.update(
        {
            "model": model_name,
            "config": config_name,
            "output_dir": str(output_dir),
            "checkpoint": str(checkpoint_path),
        }
    )
    test_path = output_dir / "classifier_test_metrics.csv"
    if test_path.exists():
        test_row = pd.read_csv(test_path).iloc[0].to_dict()
        for key, value in test_row.items():
            row[f"test_{key}"] = value
    return row


def read_best_detector_row(output_dir, config_name):
    import pandas as pd

    metrics_path = output_dir / "validation_metrics.csv"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None
    row = metrics.sort_values(
        ["image_f1", "detector_precision", "mean_iou"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    checkpoint_candidates = [
        output_dir / "frcnn_best_f1.pt",
        output_dir / "frcnn_best.pt",
        output_dir / "frcnn_latest.pt",
    ]
    checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), checkpoint_candidates[0])
    row.update(
        {
            "model": "frcnn",
            "config": config_name,
            "output_dir": str(output_dir),
            "checkpoint": str(checkpoint_path),
        }
    )
    test_path = output_dir / "test_metrics.csv"
    if test_path.exists():
        test_row = pd.read_csv(test_path).iloc[0].to_dict()
        for key, value in test_row.items():
            row[f"test_{key}"] = value
    return row


def top_rows(rows, k):
    if not rows:
        return []
    table = pd.DataFrame(rows).sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )
    return table.head(k).to_dict(orient="records")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compact hyperparameter tuning sweep for the current best RSNA fusion model."
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
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-fusion-runs", type=int, default=12)
    parser.add_argument("--skip-detector-tuning", action="store_true")
    parser.add_argument("--skip-classifier-tuning", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    seed_tag = f"seed{args.seed}"
    tuning_root = output_root / f"outputs_{seed_tag}_tuning"
    if not args.dry_run:
        tuning_root.mkdir(parents=True, exist_ok=True)

    detector_hard_negative_file = output_root / f"outputs_{seed_tag}_frcnn_hardneg_mining" / "hard_negative_ids.csv"
    swin_hard_negative_file = output_root / f"outputs_{seed_tag}_swin_hardneg_mining" / "hard_negative_patient_ids.csv"

    detector_rows = []
    if not args.skip_detector_tuning:
        for config in DETECTOR_CONFIGS:
            output_dir = tuning_root / f"frcnn_{config['name']}"
            command = [
                sys.executable,
                str(SCRIPT_DIR / "testing_FRCNN.py"),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.detector_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--optimizer",
                "adamw",
                "--lr",
                str(config["lr"]),
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
            if detector_hard_negative_file.exists():
                command.extend(
                    [
                        "--hard-negative-file",
                        str(detector_hard_negative_file),
                        "--hard-negative-weight",
                        str(config["hard_negative_weight"]),
                    ]
                )
            else:
                print(f"Warning: missing detector hard-negative file: {detector_hard_negative_file}", flush=True)
            run_stage(
                f"Detector tuning {config['name']}",
                command,
                expected_path=output_dir / "frcnn_best_f1.pt",
                force=args.force,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                row = read_best_detector_row(output_dir, config["name"])
                if row is not None:
                    detector_rows.append(row)

    classifier_rows_by_model = {model_name: [] for model_name in CLASSIFIER_CONFIGS}
    if not args.skip_classifier_tuning:
        for model_name, model_spec in CLASSIFIER_CONFIGS.items():
            for config in model_spec["configs"]:
                output_dir = tuning_root / f"{model_name}_{config['name']}"
                command = [
                    sys.executable,
                    str(SCRIPT_DIR / "train_classifier.py"),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                    "--architecture",
                    model_spec["architecture"],
                    "--image-size",
                    str(model_spec["image_size"]),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.classifier_batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--lr",
                    str(config["lr"]),
                    "--weight-decay",
                    str(config["weight_decay"]),
                    "--augmentation",
                    "cxr",
                    "--loss",
                    "asl",
                    "--asl-gamma-neg",
                    str(config["asl_gamma_neg"]),
                    "--asl-clip",
                    "0.05",
                    "--label-smoothing",
                    "0.03",
                    "--balanced-batches",
                    "--mixup-alpha",
                    "0.05",
                    "--eval-tta-modes",
                    "none,hflip",
                    "--seed",
                    str(args.seed),
                    "--no-save-epoch-checkpoints",
                ]
                if model_name == "swin" and swin_hard_negative_file.exists():
                    command.extend(
                        [
                            "--hard-negative-file",
                            str(swin_hard_negative_file),
                            "--hard-negative-weight",
                            "1.5",
                        ]
                    )
                run_stage(
                    f"Classifier tuning {model_name} {config['name']}",
                    command,
                    expected_path=output_dir / "classifier_best.pt",
                    force=args.force,
                    dry_run=args.dry_run,
                )
                if not args.dry_run:
                    row = read_best_classifier_row(output_dir, model_name, config["name"])
                    if row is not None:
                        classifier_rows_by_model[model_name].append(row)

    if args.dry_run:
        return

    import pandas as pd

    if detector_rows:
        detector_table = pd.DataFrame(detector_rows).sort_values(
            ["image_f1", "detector_precision", "mean_iou"],
            ascending=[False, False, False],
        )
        detector_table.to_csv(tuning_root / "detector_tuning_summary.csv", index=False)
    else:
        fallback_detector = output_root / f"outputs_{seed_tag}_frcnn_hardneg_adamw_b16_lr0001" / "frcnn_best_f1.pt"
        detector_table = pd.DataFrame(
            [
                {
                    "model": "frcnn",
                    "config": "fallback_existing",
                    "image_f1": -1,
                    "detector_precision": -1,
                    "mean_iou": -1,
                    "checkpoint": str(fallback_detector),
                }
            ]
        )

    classifier_tables = {}
    for model_name, rows in classifier_rows_by_model.items():
        if rows:
            table = pd.DataFrame(rows).sort_values(
                ["image_f1", "image_precision", "image_recall"],
                ascending=[False, False, False],
            )
            table.to_csv(tuning_root / f"{model_name}_tuning_summary.csv", index=False)
            classifier_tables[model_name] = table
        else:
            fallback = output_root / {
                "swin": f"outputs_{seed_tag}_swin_t_hardneg_b8_512_lr00002",
                "convnextv2": f"outputs_{seed_tag}_convnextv2_base_balanced_mixup",
                "patch_swinv2": f"outputs_{seed_tag}_patch_swinv2_balanced",
            }[model_name] / "classifier_best.pt"
            classifier_tables[model_name] = pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "config": "fallback_existing",
                        "image_f1": -1,
                        "image_precision": -1,
                        "image_recall": -1,
                        "checkpoint": str(fallback),
                    }
                ]
            )

    if args.skip_fusion:
        print(f"\nTuning summaries saved in {tuning_root}")
        return

    top_detectors = detector_table.head(max(args.top_k, 1)).to_dict(orient="records")
    top_classifiers = [
        classifier_tables["swin"].head(max(args.top_k, 1)).to_dict(orient="records"),
        classifier_tables["convnextv2"].head(max(args.top_k, 1)).to_dict(orient="records"),
        classifier_tables["patch_swinv2"].head(max(args.top_k, 1)).to_dict(orient="records"),
    ]
    combos = list(itertools.product(top_detectors, *top_classifiers))
    combos = combos[: args.max_fusion_runs]

    fusion_rows = []
    for index, combo in enumerate(combos, start=1):
        detector, swin, convnext, patch = combo
        output_dir = tuning_root / f"fusion_combo_{index:02d}"
        classifier_checkpoints = [swin["checkpoint"], convnext["checkpoint"], patch["checkpoint"]]
        command = [
            sys.executable,
            str(SCRIPT_DIR / "fusion_sweep.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--classifier-checkpoints",
            ",".join(classifier_checkpoints),
            "--detector-checkpoint",
            detector["checkpoint"],
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
        ]
        run_stage(
            f"Fusion combo {index:02d}",
            command,
            expected_path=output_dir / "fusion_sweep_test_metrics.csv",
            force=args.force,
            dry_run=args.dry_run,
        )
        metrics_path = output_dir / "fusion_sweep_test_metrics.csv"
        if metrics_path.exists():
            row = pd.read_csv(metrics_path).iloc[0].to_dict()
            row.update(
                {
                    "combo": index,
                    "detector_config": detector.get("config"),
                    "swin_config": swin.get("config"),
                    "convnext_config": convnext.get("config"),
                    "patch_swin_config": patch.get("config"),
                    "detector_checkpoint": detector["checkpoint"],
                    "swin_checkpoint": swin["checkpoint"],
                    "convnext_checkpoint": convnext["checkpoint"],
                    "patch_swin_checkpoint": patch["checkpoint"],
                    "output_dir": str(output_dir),
                }
            )
            fusion_rows.append(row)

    if fusion_rows:
        fusion_table = pd.DataFrame(fusion_rows).sort_values(
            ["image_f1", "image_precision", "detector_precision", "mean_iou"],
            ascending=[False, False, False, False],
        )
        fusion_table.to_csv(tuning_root / "fusion_tuning_summary.csv", index=False)
        with (tuning_root / "fusion_tuning_best.json").open("w", encoding="utf-8") as handle:
            json.dump(fusion_table.iloc[0].to_dict(), handle, indent=2)
        print("\nBest tuned fusion result:")
        print(fusion_table.iloc[0].to_string())
        print(f"\nSaved fusion summary to {tuning_root / 'fusion_tuning_summary.csv'}")


if __name__ == "__main__":
    main()

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_ARCHITECTURES = [
    "xrv_densenet121_chex",
    "xrv_densenet121_mimic",
    "xrv_densenet121_mimic_ch",
    "xrv_densenet121_rsna",
]


def parse_list(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def format_lr(value):
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def run_command(command, dry_run=False):
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def read_best_metrics(output_dir):
    metrics_path = output_dir / "classifier_validation_metrics.csv"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None
    metrics = metrics.sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )
    row = metrics.iloc[0].to_dict()
    row["output_dir"] = str(output_dir)
    row["checkpoint"] = str(output_dir / "classifier_best.pt")

    test_path = output_dir / "classifier_test_metrics.csv"
    if test_path.exists():
        test_row = pd.read_csv(test_path).iloc[0].to_dict()
        for key, value in test_row.items():
            row[f"test_{key}"] = value
    return row


def train_domain_backbone(args, architecture):
    output_dir = Path(args.output_root) / (
        f"outputs_classifier_{architecture}_"
        f"{args.image_size}_lr{format_lr(args.lr)}_{args.loss}"
    )
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_classifier.py"),
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(output_dir),
        "--architecture",
        architecture,
        "--image-size",
        str(args.image_size),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--augmentation",
        args.augmentation,
        "--loss",
        args.loss,
        "--label-smoothing",
        str(args.label_smoothing),
        "--eval-tta-modes",
        args.eval_tta_modes,
        "--thresholds",
        args.thresholds,
    ]
    if args.loss == "focal":
        command.extend(["--focal-alpha", str(args.focal_alpha), "--focal-gamma", str(args.focal_gamma)])
    if args.loss == "asl":
        command.extend(
            [
                "--asl-gamma-pos",
                str(args.asl_gamma_pos),
                "--asl-gamma-neg",
                str(args.asl_gamma_neg),
                "--asl-clip",
                str(args.asl_clip),
            ]
        )
        if args.asl_use_pos_weight:
            command.append("--asl-use-pos-weight")
    if args.balanced_batches:
        command.append("--balanced-batches")
    if args.mixup_alpha > 0:
        command.extend(["--mixup-alpha", str(args.mixup_alpha)])
    if args.hard_negative_file:
        command.extend(
            [
                "--hard-negative-file",
                args.hard_negative_file,
                "--hard-negative-weight",
                str(args.hard_negative_weight),
            ]
        )
    if args.no_amp:
        command.append("--no-amp")

    if output_dir.exists() and (output_dir / "classifier_best.pt").exists() and not args.force:
        print(f"\nSkipping {architecture}; checkpoint already exists at {output_dir / 'classifier_best.pt'}", flush=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_command(command, dry_run=args.dry_run)
    return output_dir


def run_fusion(args, best_checkpoint):
    if not args.swin_checkpoint or not args.detector_checkpoint:
        print("\nSkipping fusion; provide --swin-checkpoint and --detector-checkpoint to run it.", flush=True)
        return

    output_dir = Path(args.output_root) / "outputs_fusion_domain_pretrained_xrv"
    classifier_checkpoints = [args.swin_checkpoint, best_checkpoint]
    if args.extra_classifier_checkpoints:
        classifier_checkpoints.extend(parse_list(args.extra_classifier_checkpoints))

    methods = args.fusion_methods
    if len(classifier_checkpoints) != 2:
        methods = ",".join(method for method in parse_list(methods) if method != "xrv_weighted_gate")
        if not methods:
            methods = "precision_gate,meta"

    command = [
        sys.executable,
        str(SCRIPT_DIR / "fusion_sweep.py"),
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(output_dir),
        "--classifier-checkpoints",
        ",".join(classifier_checkpoints),
        "--detector-checkpoint",
        args.detector_checkpoint,
        "--classifier-tta-modes",
        args.fusion_tta_modes,
        "--classifier-temperature-mode",
        args.classifier_temperature_mode,
        "--methods",
        methods,
        "--min-precision",
        str(args.min_precision),
        "--min-recall",
        str(args.min_recall),
        "--box-score-thresholds",
        args.box_score_thresholds,
        "--batch-size",
        str(args.fusion_batch_size),
        "--classifier-batch-size",
        str(args.fusion_classifier_batch_size),
        "--num-workers",
        str(args.fusion_num_workers),
    ]
    run_command(command, dry_run=args.dry_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train domain-pretrained CXR classifier backbones and optionally fuse the best "
            "candidate with the current Swin/Faster R-CNN system."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", default="/home/ec2-user/rsna")
    parser.add_argument("--architectures", default=",".join(DEFAULT_ARCHITECTURES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--augmentation", choices=["basic", "cxr"], default="cxr")
    parser.add_argument("--loss", choices=["bce", "focal", "asl"], default="asl")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=3.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--asl-use-pos-weight", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--balanced-batches", action="store_true", default=True)
    parser.add_argument("--no-balanced-batches", dest="balanced_batches", action="store_false")
    parser.add_argument("--mixup-alpha", type=float, default=0.05)
    parser.add_argument("--hard-negative-file", default=None)
    parser.add_argument("--hard-negative-weight", type=float, default=1.5)
    parser.add_argument("--eval-tta-modes", default="none,hflip,clahe,gamma_low,gamma_high")
    parser.add_argument(
        "--thresholds",
        default="0.30,0.35,0.40,0.45,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.85,0.90",
    )
    parser.add_argument("--swin-checkpoint", default=None)
    parser.add_argument("--detector-checkpoint", default=None)
    parser.add_argument("--extra-classifier-checkpoints", default=None)
    parser.add_argument("--fusion-methods", default="xrv_weighted_gate,precision_gate,meta")
    parser.add_argument("--fusion-tta-modes", default="none,hflip,clahe,gamma_low,gamma_high")
    parser.add_argument("--classifier-temperature-mode", choices=["none", "validation"], default="validation")
    parser.add_argument("--box-score-thresholds", default="0.50,0.54,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72")
    parser.add_argument("--fusion-batch-size", type=int, default=4)
    parser.add_argument("--fusion-classifier-batch-size", type=int, default=8)
    parser.add_argument(
        "--fusion-num-workers",
        type=int,
        default=0,
        help="DataLoader workers for fusion inference. Use 0 or 1 on memory-constrained EC2 instances.",
    )
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    architectures = parse_list(args.architectures)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for architecture in architectures:
        output_dir = train_domain_backbone(args, architecture)
        if args.dry_run:
            continue
        row = read_best_metrics(output_dir)
        if row is None:
            print(f"Warning: no metrics found for {architecture} in {output_dir}", flush=True)
            continue
        row["architecture"] = architecture
        rows.append(row)

    if args.dry_run:
        return

    if not rows:
        raise RuntimeError("No completed domain-pretrained classifier metrics were found.")

    summary = pd.DataFrame(rows).sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )
    summary_path = output_root / "domain_pretrained_backbone_summary.csv"
    summary.to_csv(summary_path, index=False)
    best = summary.iloc[0].to_dict()
    with (output_root / "domain_pretrained_backbone_best.json").open("w", encoding="utf-8") as best_file:
        json.dump(best, best_file, indent=2)

    print("\nBest domain-pretrained classifier by validation F1:")
    print(pd.Series(best).to_string())
    print(f"\nSaved summary to {summary_path}")

    run_fusion(args, best["checkpoint"])


if __name__ == "__main__":
    main()

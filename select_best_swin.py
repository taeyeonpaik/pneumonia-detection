import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch


def checkpoint_score(path):
    checkpoint = torch.load(path, map_location="cpu")
    metrics = checkpoint.get("validation_metrics", {}) or {}
    return {
        "checkpoint": str(path),
        "architecture": checkpoint.get("architecture"),
        "image_size": checkpoint.get("image_size"),
        "epoch": checkpoint.get("epoch"),
        "image_f1": float(metrics.get("image_f1", 0.0)),
        "image_precision": float(metrics.get("image_precision", 0.0)),
        "image_recall": float(metrics.get("image_recall", 0.0)),
        "score_threshold": float(metrics.get("score_threshold", 0.5)),
    }


def main():
    parser = argparse.ArgumentParser(description="Select the best Swin checkpoint by validation metrics.")
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint paths.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-name", default="best_swin_classifier.pt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = [Path(item.strip()).expanduser().resolve() for item in args.checkpoints.split(",") if item.strip()]
    if not checkpoint_paths:
        raise ValueError("No checkpoints were provided.")

    rows = [checkpoint_score(path) for path in checkpoint_paths]
    table = pd.DataFrame(rows).sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )
    table.to_csv(output_dir / "swin_checkpoint_comparison.csv", index=False)

    best = table.iloc[0].to_dict()
    destination = output_dir / args.copy_name
    shutil.copy2(best["checkpoint"], destination)
    best["selected_checkpoint_copy"] = str(destination)
    with (output_dir / "best_swin_checkpoint.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)

    print("Best Swin checkpoint:")
    print(pd.Series(best).to_string())


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import prepare_data, resolve_data_dir, seed_worker
from testing_FRCNN import choose_device, set_seed
from train_classifier import (
    RSNAClassificationDataset,
    autocast_context,
    build_classifier,
    classifier_eval_transform,
    classifier_logits,
    classification_collate,
    load_patient_ids,
    normalization_for_architecture,
    parse_tta_modes,
)


def make_train_loaders(train_detection, image_size, normalization, batch_size, num_workers, device, tta_modes):
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": classification_collate,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker

    loaders = []
    for mode in tta_modes:
        dataset = RSNAClassificationDataset(
            train_detection,
            classifier_eval_transform(image_size, normalization=normalization, tta_mode=mode),
        )
        loaders.append(
            (
                mode,
                DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs),
            )
        )
    return loaders


def collect_probabilities(loaders, model, device, use_amp):
    probability_sums = {}
    probability_counts = {}
    labels_by_id = {}
    non_blocking = device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for mode, loader in loaders:
            for images, labels, image_ids in tqdm(loader, total=len(loader), desc=f"train predictions {mode}"):
                images = images.to(device, non_blocking=non_blocking)
                with autocast_context(use_amp):
                    logits = classifier_logits(model(images)).flatten()
                probabilities = torch.sigmoid(logits).detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                for image_id, label, probability in zip(image_ids, labels_np, probabilities):
                    probability_sums[image_id] = probability_sums.get(image_id, 0.0) + float(probability)
                    probability_counts[image_id] = probability_counts.get(image_id, 0) + 1
                    labels_by_id[image_id] = int(round(float(label)))

    rows = []
    for patient_id in labels_by_id:
        rows.append(
            {
                "patientId": patient_id,
                "truth": labels_by_id[patient_id],
                "probability": probability_sums[patient_id] / max(probability_counts[patient_id], 1),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Mine train-split hard negatives for a selected Swin classifier.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-probability", type=float, default=None)
    parser.add_argument("--max-hard-negatives", type=int, default=2000)
    parser.add_argument("--extra-fp-file", default=None, help="Optional FP CSV/text file; only IDs in the training split are kept.")
    parser.add_argument("--tta-modes", default="none,hflip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", "swin_t")
    image_size = int(checkpoint.get("image_size", 512))
    normalization = checkpoint.get("normalization", normalization_for_architecture(architecture, "auto"))
    threshold = args.threshold
    if threshold is None:
        threshold = float((checkpoint.get("validation_metrics", {}) or {}).get("score_threshold", 0.5))
    min_probability = args.min_probability if args.min_probability is not None else threshold
    tta_modes = parse_tta_modes(args.tta_modes)

    print(f"Using data directory: {data_dir}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Architecture: {architecture}, image_size={image_size}, normalization={normalization}")
    print(f"Threshold={threshold:g}, min_probability={min_probability:g}, TTA={','.join(tta_modes)}")
    print(f"Using device: {device}, AMP={use_amp}")

    train_detection, _, _ = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    model = build_classifier(architecture=architecture, pretrained=False, image_size=image_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loaders = make_train_loaders(
        train_detection,
        image_size=image_size,
        normalization=normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        tta_modes=tta_modes,
    )
    predictions = collect_probabilities(loaders, model, device, use_amp)
    predictions["prediction"] = (predictions["probability"] >= threshold).astype(int)
    predictions.to_csv(output_dir / "swin_train_predictions.csv", index=False)

    hard_negatives = predictions[
        (predictions["truth"] == 0)
        & (predictions["probability"] >= min_probability)
    ].copy()

    extra_ids = set(load_patient_ids(args.extra_fp_file))
    if extra_ids:
        train_ids = set(predictions["patientId"].astype(str).tolist())
        matched_extra = sorted(extra_ids.intersection(train_ids))
        if matched_extra:
            extra_rows = predictions[predictions["patientId"].astype(str).isin(matched_extra)].copy()
            extra_rows = extra_rows[extra_rows["truth"] == 0]
            hard_negatives = pd.concat([hard_negatives, extra_rows], ignore_index=True)
        print(f"Matched {len(matched_extra)} extra FP IDs in the training split from {len(extra_ids)} provided IDs")

    hard_negatives = hard_negatives.sort_values("probability", ascending=False).drop_duplicates("patientId")
    if args.max_hard_negatives > 0:
        hard_negatives = hard_negatives.head(args.max_hard_negatives)
    hard_negatives.to_csv(output_dir / "hard_negative_ids.csv", index=False)
    hard_negatives[["patientId"]].to_csv(output_dir / "hard_negative_patient_ids.csv", index=False)

    summary = {
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "min_probability": min_probability,
        "num_train_predictions": int(len(predictions)),
        "num_hard_negatives": int(len(hard_negatives)),
    }
    with (output_dir / "hard_negative_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()

# Development experiment log

This working log records successful and unsuccessful experiments during development. The release documentation later consolidates it.

## chore: clean generated artifacts from legacy detector

Removed cached bytecode, local datasets, Finder metadata, generated images, checkpoints, and logs from version control while preserving the original detector source.

## refactor: make DICOM conversion portable

Replaced machine-specific conversion paths with a command-line data directory and explicit input discovery.

## refactor: add configurable data paths and deterministic splits

Unified image discovery, grouped annotations by patient, and introduced reproducible train, validation, and held-out test splits.

## refactor: modernize Faster R-CNN model construction

Updated the legacy detector to a configurable torchvision Faster R-CNN with a ResNet-50 FPN backbone and replaceable prediction head.

## feat: train detector with mixed precision and progress reporting

Added CUDA automatic mixed precision, scaler support, per-batch progress, and stable loss aggregation.

## feat: evaluate image-level F1 and matched-box IoU

Added thresholded image-level confusion metrics, detector precision, matched true-positive IoU, prediction exports, plots, and held-out evaluation.

## fix: create output directories before writing metrics

Fixed failures when resumed or partially deleted runs attempted to write CSV, confusion-matrix, and checkpoint files into missing directories.

## feat: select detector checkpoints and confidence thresholds

Added validation-only checkpoint and score-threshold selection before one-time held-out test evaluation.

## feat: mine false-positive detector hard negatives

Collected normal radiographs repeatedly assigned high-confidence boxes so they could be oversampled during detector retraining.

## feat: add AdamW balanced-sampling detector experiments

Compared optimizer, learning-rate, positive-sampling, and hard-negative weighting configurations against the original SGD detector.

## fix: reduce redundant checkpoint writes

Added weights-only and selective-save controls after full optimizer checkpoints exhausted the EC2 volume during repeated tuning runs.

## refactor: retire the original standalone CNN

Removed the superseded prototype classifier after the Faster R-CNN data and evaluation path became the stable baseline.

## docs: document portable detector training workflow

Documented the cleaned data layout, DICOM conversion, detector training, threshold selection, and generated artifacts.

## feat: add configurable transfer-learning classifiers

Introduced torchvision and timm classifier construction, image-level datasets, threshold sweeps, checkpoint selection, and held-out evaluation.

## experiment: fine-tune Swin-T at 384 pixels

Established the first transformer classifier baseline and compared its image-level errors with Faster R-CNN.

## feat: sweep classifier confidence thresholds

Selected each classifier epoch and operating threshold using validation F1 rather than a fixed probability cutoff.

## fix: clone non-resizable tensors during collation

Resolved DataLoader storage resize errors by cloning transformed tensors before stacking custom batches.

## experiment: fine-tune Swin-T at 512 pixels

Raised classifier resolution to preserve subtle opacities; the higher-resolution Swin improved the later detector fusion despite mixed standalone results.

## feat: combine classifier and detector decisions

Added AND, OR, override, and confidence-gated sweeps across Swin image probabilities and Faster R-CNN evidence.


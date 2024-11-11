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


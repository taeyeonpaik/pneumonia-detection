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


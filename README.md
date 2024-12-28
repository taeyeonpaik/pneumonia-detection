# RSNA Pneumonia Detection

Train and evaluate a Faster R-CNN detector on the RSNA pneumonia dataset.

The pipeline converts DICOM studies, creates a deterministic patient split, trains on CUDA with AMP, and selects checkpoints and confidence thresholds using validation metrics.

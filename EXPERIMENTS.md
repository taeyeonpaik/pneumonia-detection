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

## experiment: fine-tune Swin on classifier hard negatives

Mined high-confidence normal false positives and retrained Swin with weighted hard-negative sampling; the attempt improved some operating points but did not eliminate the shared error ceiling.

## feat: add CXR augmentation and imbalance-aware losses

Added CLAHE, gamma and contrast perturbations, balanced batches, focal and asymmetric loss, label smoothing, MixUp, and soft F-beta experiments.

## feat: mine Faster R-CNN region candidates

Exported detector proposals and padded crop geometry for a second-stage regional classifier.

## experiment: train Swin classifiers on detector regions

Trained a Swin model on proposed opacity regions to veto spurious detections; standalone and fused ROI results remained below the best whole-image gate.

## experiment: sweep ROI filtering rules

Swept detector and region-classifier thresholds and measured localization quality; the best held-out ROI-filter F1 remained around 0.67.

## experiment: evaluate dual-Swin hybrid gating

Compared the original and CXR-loss Swin checkpoints with detector rescue and veto rules; correlated errors limited the gain.

## experiment: add ConvNeXt-Tiny classifier

Introduced a convolutional classifier to diversify the transformer-heavy ensemble and measured its independent precision-recall behavior.

## experiment: scale ConvNeXt classifier capacity

Tested ConvNeXt-Small and larger ConvNeXtV2 variants after the tiny model failed to improve the best fusion rule.

## feat: add logistic meta-fusion

Added calibrated feature extraction and learned logistic fusion over classifier probabilities, detector confidence, box count, and box geometry.

## experiment: add nonlinear MLP fusion

Tested a regularized MLP fusion head. A 0.997 validation F1 followed by 0.567 held-out F1 exposed severe validation-fit leakage rather than a real improvement.

## fix: fit learned fusion on training features

Added a clean train-fit fusion path and retained validation solely for hyperparameter and threshold selection.

## experiment: add RSNA-pretrained XRV DenseNet

Evaluated TorchXRayVision DenseNet-121 pretrained on RSNA chest radiographs. High recall came with excessive false positives, and replacing its output head exposed normalization incompatibilities.

## fix: normalize mixed grayscale and RGB batches

Separated ImageNet and XRV normalization paths after one-channel and three-channel tensors collided in the same batch.

## fix: disable XRV operating-point normalization for custom heads

Cleared TorchXRayVision operating-point thresholds after replacing the 18-output classifier with a binary head, fixing the output-mask shape mismatch.

## experiment: combine transformer and DenseNet features

Evaluated hybrid ViT, Swin, DenseNet-121, DenseNet-169, and DenseNet-201 feature extractors; the larger hybrids trained successfully but did not surpass the detector-Swin gate.

## feat: extract age sex and projection metadata

Extracted robust age, sex, and AP/PA view features from all available DICOM headers with missing-value indicators.

## feat: condition classifiers on DICOM metadata

Fused normalized metadata with image embeddings in custom classifier heads and used the same features during validation and held-out inference.

## experiment: retrain detector IoU variants

Retrained proposal heads at alternate foreground and background IoU thresholds; stricter objectives improved box precision but reduced image-level recall.

## experiment: fuse metadata Swin with original detector

Returned to the strongest IoU-0.50 Faster R-CNN and combined it with metadata-aware Swin and the best ConvNeXt candidate, reaching the first sustained improvement over the earlier 0.688 plateau.

## feat: export component confusion matrices and error lists

Generated per-model confusion matrices, patient-level false-positive and false-negative lists, overlap tables, and metadata subgroup metrics.

## experiment: tune metadata-stratified precision gates

Selected validation rules by age, sex, and projection subgroup. Age buckets produced a small gain, while finer sweeps regressed on held-out data.

## experiment: mine shared ensemble false positives

Focused retraining on false positives shared by classifier and detector. The resulting Swin variants shifted precision and recall but did not improve the overall best F1.

## experiment: audit alternate dataset splits

Measured class prevalence and patient membership across several deterministic seeds before selecting a balanced alternate split for robustness testing.

## experiment: retune classifiers and detector on seed 2024

Retrained the strongest components on the alternate split. Its 0.688 held-out result confirmed that the original split was not the sole bottleneck.

## fix: resume tuning after missing output directories

Made every tuning stage recreate its output directory, skip valid artifacts, and recover safely from deleted partial runs and full-disk checkpoint failures.

## experiment: ensemble diverse detectors with WBF

Trained an independent scaled Faster R-CNN and merged boxes with weighted box fusion. Validation selected WBF settings, but held-out F1 remained below the single-detector metadata ensemble.

## experiment: expand backbone and loss search

Automated EfficientNet-B3, DenseNet-169, XRV, asymmetric-loss, and soft-F-beta experiments with balanced sampling and shared hard negatives.

## experiment: evaluate ConvNeXtV2 and SwinV2

Compared ConvNeXtV2 resolution and loss variants plus SwinV2-Base; added diversity but no configuration reliably cleared the target.

## fix: bound worker memory and expose pipeline progress

Reduced memory pressure after DataLoader workers were killed, added unbuffered stage output, and ensured long EC2 jobs reported per-stage and per-batch progress.

## experiment: prepare borderline-label review workflow

Generated a review template for uncertain and misclassified cases. The workflow was retained for auditability, but dataset labels were ultimately left unchanged.

## experiment: add train-fit and out-of-fold fusion

Added train-fit, stratified, and out-of-fold fusion workflows to evaluate learned combinations without reusing validation labels for fitting.

## experiment: evaluate tabular and bagged meta-fusion

Tested nonlinear tabular models, bootstrap bagging, elite-model selection, and stratified variants. Added complexity did not generalize better than the precision gate.

## experiment: add stacked rescue veto and flexible gates

Explored stacked rescue, veto, calibrated detector support, and flexible Boolean rules over cached predictions; the searches improved selected operating points but remained validation-sensitive.

## experiment: add metadata-aware EfficientNetV2-S

Added a complementary EfficientNetV2-S classifier with the same eight metadata features, hard negatives, imbalance-aware training, and validation-selected thresholds.

## fix: fall back when requested timm weights are unavailable

Recovered from an unavailable timm EfficientNetV2 alias by using the supported torchvision EfficientNetV2-S implementation and preserving checkpoint metadata.

## feat: calibrate classifier confidence and refine precision gate

Applied validation temperature scaling and a coarse-to-fine coordinate search over classifier agreement, detector support, and override thresholds.

## feat: add horizontal-flip and CLAHE test-time augmentation

Averaged original, horizontal-flip, and locally contrast-enhanced classifier predictions. CLAHE recovered additional true positives and raised held-out image F1 to 0.715.

## fix: validate and lock exported model artifacts

Fixed malformed artifact naming, copied the selected detector and classifier checkpoints, recorded thresholds and hashes, and exported final confusion and error reports.


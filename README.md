# Pneumonia Detection

An end-to-end PyTorch pipeline for pneumonia classification and opacity localization on the RSNA Pneumonia Detection Challenge dataset.

The final ensemble combines a Faster R-CNN detector with Swin Transformer and EfficientNetV2 classifiers. The classifiers also use patient age, sex, and X-ray view metadata.

## Results

- Image-level F1: **0.715**
- Accuracy: **0.865**
- Mean IoU: **0.700**

## Setup

```bash
pip install -r requirements.txt
```

The RSNA dataset and trained model checkpoints are not included in this repository.

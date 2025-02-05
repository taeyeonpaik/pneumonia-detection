import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from dicom_metadata import METADATA_FEATURE_COLUMNS, load_metadata_table, unknown_metadata_vector
from dataset import prepare_data, resolve_data_dir, seed_worker
from testing_FRCNN import choose_device, parse_thresholds, save_confusion_matrix, set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt


XRV_ARCHITECTURES = {
    "xrv_densenet121_all": "densenet121-res224-all",
    "xrv_densenet121_rsna": "densenet121-res224-rsna",
    "xrv_densenet121_nih": "densenet121-res224-nih",
    "xrv_densenet121_pc": "densenet121-res224-pc",
    "xrv_densenet121_chex": "densenet121-res224-chex",
    "xrv_densenet121_mimic": "densenet121-res224-mimic_nb",
    "xrv_densenet121_mimic_ch": "densenet121-res224-mimic_ch",
}

HYBRID_XRV_ARCHITECTURES = {
    "hybrid_swin_xrv_densenet121_rsna": ("swin_t", "densenet121-res224-rsna"),
    "hybrid_vit_xrv_densenet121_rsna": ("vit_b_16", "densenet121-res224-rsna"),
}

HYBRID_IMAGENET_ARCHITECTURES = {
    "hybrid_swin_densenet121": ("swin_t", "densenet121"),
    "hybrid_swin_densenet169": ("swin_t", "densenet169"),
    "hybrid_vit_densenet121": ("vit_b_16", "densenet121"),
    "hybrid_vit_densenet169": ("vit_b_16", "densenet169"),
    "hybrid_vit_densenet201": ("vit_b_16", "densenet201"),
    "hybrid_swin_densenet201": ("swin_t", "densenet201"),
    "hybrid_swin_convnext_small": ("swin_t", "convnext_small"),
    "hybrid_swin_v2_t_densenet169": ("swin_v2_t", "densenet169"),
    "hybrid_swin_v2_t_densenet201": ("swin_v2_t", "densenet201"),
    "hybrid_swin_v2_t_convnext_base": ("swin_v2_t", "convnext_base"),
}

TIMM_ARCHITECTURES = {
    "timm_convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "timm_swinv2_base": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "timm_efficientnet_b3": "tf_efficientnet_b3.ns_jft_in1k",
    "timm_efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
}

PATCH_ARCHITECTURES = {
    "patch_swin_t": "swin_t",
    "patch_swin_v2_t": "swin_v2_t",
    "patch_convnext_tiny": "convnext_tiny",
    "patch_convnext_small": "convnext_small",
}


def xrv_normalize_image(image, **kwargs):
    image = image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    if image.max() > 1.0:
        image = image / 255.0
    image = image * 2048.0 - 1024.0
    return image[:, :, None].astype(np.float32)


def hybrid_imagenet_xrv_image(image, **kwargs):
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    imagenet = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    gray = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    xrv = gray.astype(np.float32) / 255.0 * 2048.0 - 1024.0
    return np.concatenate([imagenet.astype(np.float32), xrv[:, :, None]], axis=2)


def normalization_for_architecture(architecture, requested="auto"):
    if requested != "auto":
        return requested
    if architecture in HYBRID_XRV_ARCHITECTURES:
        return "hybrid_xrv_imagenet"
    return "xrv" if architecture in XRV_ARCHITECTURES else "imagenet"


class RSNAClassificationDataset(Dataset):
    def __init__(
        self,
        detection_dataset,
        transform=None,
        label_overrides=None,
        label_smoothing=0.0,
        metadata_by_id=None,
    ):
        self.image_ids = detection_dataset.image_ids.reset_index(drop=True)
        self.records_by_image_id = detection_dataset.records_by_image_id
        self.image_directory = detection_dataset.image_directory
        self.transform = transform
        self.label_overrides = {str(key): float(value) for key, value in (label_overrides or {}).items()}
        self.label_smoothing = float(label_smoothing)
        self.metadata_by_id = metadata_by_id
        self.unknown_metadata = unknown_metadata_vector()

    def __getitem__(self, index):
        image_id = self.image_ids.iloc[index]
        records = self.records_by_image_id[image_id]
        image_path = self.image_directory / f"{image_id}.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.hard_label(image_id)
        if self.label_smoothing > 0.0:
            label = label * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label_tensor = torch.tensor(label, dtype=torch.float32)
        if self.metadata_by_id is None:
            return image, label_tensor, image_id
        metadata = self.metadata_by_id.get(str(image_id), self.unknown_metadata)
        return image, torch.as_tensor(metadata, dtype=torch.float32), label_tensor, image_id

    def __len__(self):
        return len(self.image_ids)

    def hard_label(self, image_id):
        image_id = str(image_id)
        if image_id in self.label_overrides:
            return float(self.label_overrides[image_id])
        return float(self.records_by_image_id[image_id]["Target"].max())

    def labels(self):
        return [int(round(self.hard_label(image_id))) for image_id in self.image_ids]


def classification_collate(batch):
    if len(batch[0]) == 4:
        images, metadata, labels, image_ids = zip(*batch)
        images = torch.stack([image.clone() for image in images], dim=0)
        metadata = torch.stack([values.clone() for values in metadata], dim=0)
        labels = torch.stack(list(labels), dim=0)
        return images, metadata, labels, list(image_ids)
    images, labels, image_ids = zip(*batch)
    images = torch.stack([image.clone() for image in images], dim=0)
    labels = torch.stack(list(labels), dim=0)
    return images, labels, list(image_ids)


def unpack_classification_batch(batch):
    if len(batch) == 4:
        images, metadata, labels, image_ids = batch
        return images, metadata, labels, image_ids
    images, labels, image_ids = batch
    return images, None, labels, image_ids


class BalancedBinaryBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size, seed=42, hard_negative_ids=None, image_ids=None, hard_negative_weight=1.0):
        if batch_size < 2:
            raise ValueError("Balanced batches require batch_size >= 2")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.positive_indices = np.flatnonzero(self.labels == 1)
        self.negative_indices = np.flatnonzero(self.labels == 0)
        if len(self.positive_indices) == 0 or len(self.negative_indices) == 0:
            raise ValueError("Balanced batches require both positive and negative examples")
        self.num_batches = int(np.ceil(len(self.labels) / self.batch_size))

        negative_weights = np.ones(len(self.negative_indices), dtype=np.float64)
        hard_negative_ids = set(hard_negative_ids or [])
        if hard_negative_ids and image_ids is not None:
            image_ids = [str(image_id) for image_id in image_ids]
            for position, index in enumerate(self.negative_indices):
                if image_ids[index] in hard_negative_ids:
                    negative_weights[position] *= float(hard_negative_weight)
        self.negative_weights = negative_weights / negative_weights.sum()

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        positives_per_batch = self.batch_size // 2
        negatives_per_batch = self.batch_size - positives_per_batch
        for _ in range(self.num_batches):
            positive_batch = rng.choice(self.positive_indices, size=positives_per_batch, replace=True)
            negative_batch = rng.choice(
                self.negative_indices,
                size=negatives_per_batch,
                replace=True,
                p=self.negative_weights,
            )
            batch = np.concatenate([positive_batch, negative_batch])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches


def classifier_train_transform(image_size, augmentation="basic", normalization="imagenet"):
    transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.4),
    ]
    if augmentation == "cxr":
        transforms.extend(
            [
                A.CLAHE(clip_limit=4.0, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.GaussNoise(p=0.25),
            ]
        )
    if normalization == "hybrid_xrv_imagenet":
        transforms.extend(
            [
                A.Lambda(image=hybrid_imagenet_xrv_image),
                ToTensorV2(),
            ]
        )
    elif normalization == "xrv":
        transforms.extend(
            [
                A.Lambda(image=xrv_normalize_image),
                ToTensorV2(),
            ]
        )
    else:
        transforms.extend(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose(transforms)


def parse_tta_modes(value):
    if value is None:
        return ["none"]
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not modes:
        return ["none"]

    valid_modes = {"none", "hflip", "clahe", "gamma_low", "gamma_high"}
    unknown_modes = sorted(set(modes).difference(valid_modes))
    if unknown_modes:
        raise ValueError(f"Unsupported TTA modes: {unknown_modes}. Valid modes: {sorted(valid_modes)}")
    return modes


def classifier_eval_transform(image_size, normalization="imagenet", tta_mode="none"):
    transforms = [A.Resize(image_size, image_size)]
    if tta_mode == "hflip":
        transforms.append(A.HorizontalFlip(p=1.0))
    elif tta_mode == "clahe":
        transforms.append(A.CLAHE(clip_limit=4.0, p=1.0))
    elif tta_mode == "gamma_low":
        transforms.append(A.RandomGamma(gamma_limit=(80, 80), p=1.0))
    elif tta_mode == "gamma_high":
        transforms.append(A.RandomGamma(gamma_limit=(120, 120), p=1.0))

    if normalization == "hybrid_xrv_imagenet":
        transforms.extend(
            [
                A.Lambda(image=hybrid_imagenet_xrv_image),
                ToTensorV2(),
            ]
        )
    elif normalization == "xrv":
        transforms.extend(
            [
                A.Lambda(image=xrv_normalize_image),
                ToTensorV2(),
            ]
        )
    else:
        transforms.extend(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose(transforms)


def build_xrv_classifier(weights_name):
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise ImportError(
            "TorchXRayVision is required for xrv_* classifiers. "
            "Install it on EC2 with: /opt/pytorch/bin/python -m pip install torchxrayvision"
        ) from exc

    model = xrv.models.DenseNet(weights=weights_name)
    if not hasattr(model, "classifier"):
        raise AttributeError("Expected TorchXRayVision DenseNet to expose a classifier layer.")
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    if hasattr(model, "op_threshs"):
        model.op_threshs = None
    return model


def build_xrv_feature_extractor(weights_name):
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise ImportError(
            "TorchXRayVision is required for hybrid_xrv_* classifiers. "
            "Install it on EC2 with: /opt/pytorch/bin/python -m pip install torchxrayvision"
        ) from exc

    model = xrv.models.DenseNet(weights=weights_name)
    if not hasattr(model, "features"):
        raise AttributeError("Expected TorchXRayVision DenseNet to expose a features module.")
    feature_dim = int(model.classifier.in_features)
    if hasattr(model, "op_threshs"):
        model.op_threshs = None
    return model, feature_dim


def load_vit_with_interpolated_embeddings(image_size, pretrained=True):
    vit_kwargs = {}
    if image_size is not None:
        vit_kwargs["image_size"] = image_size

    if not pretrained:
        try:
            return torchvision.models.vit_b_16(weights=None, **vit_kwargs)
        except TypeError:
            return torchvision.models.vit_b_16(pretrained=False, **vit_kwargs)

    try:
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=None, **vit_kwargs)
        if image_size is not None and image_size != 224:
            try:
                state_dict = weights.get_state_dict(progress=True, check_hash=True)
            except TypeError:
                state_dict = weights.get_state_dict(progress=True)
            try:
                from torchvision.models.vision_transformer import interpolate_embeddings

                state_dict = interpolate_embeddings(
                    image_size=image_size,
                    patch_size=16,
                    model_state=state_dict,
                    reset_heads=True,
                )
                model.load_state_dict(state_dict, strict=False)
            except (ImportError, TypeError):
                print(
                    "Warning: this torchvision version cannot interpolate ViT positional embeddings; "
                    "using randomly initialized ViT weights for the requested image size."
                )
        else:
            model = torchvision.models.vit_b_16(weights=weights, **vit_kwargs)
        return model
    except (AttributeError, TypeError):
        return torchvision.models.vit_b_16(pretrained=pretrained, **vit_kwargs)


def build_torchvision_swin(name, pretrained=True):
    model_fn = getattr(torchvision.models, name, None)
    if model_fn is None:
        raise ValueError(
            f"{name} is not available in this torchvision install. "
            "Use a timm_* architecture or upgrade torchvision."
        )

    weights = None
    if pretrained:
        weights_name = {
            "swin_t": "Swin_T_Weights",
            "swin_v2_t": "Swin_V2_T_Weights",
            "swin_v2_s": "Swin_V2_S_Weights",
            "swin_v2_b": "Swin_V2_B_Weights",
        }.get(name)
        weights_enum = getattr(torchvision.models, weights_name, None)
        if weights_enum is not None:
            weights = weights_enum.DEFAULT

    try:
        return model_fn(weights=weights)
    except TypeError:
        return model_fn(pretrained=pretrained)


def build_torchvision_convnext(name, pretrained=True):
    model_fn = getattr(torchvision.models, name, None)
    if model_fn is None:
        raise ValueError(
            f"{name} is not available in this torchvision install. "
            "Use a timm_* architecture or upgrade torchvision."
        )

    weights = None
    if pretrained:
        weights_name = {
            "convnext_tiny": "ConvNeXt_Tiny_Weights",
            "convnext_small": "ConvNeXt_Small_Weights",
            "convnext_base": "ConvNeXt_Base_Weights",
        }.get(name)
        weights_enum = getattr(torchvision.models, weights_name, None)
        if weights_enum is not None:
            weights = weights_enum.DEFAULT

    try:
        return model_fn(weights=weights)
    except TypeError:
        return model_fn(pretrained=pretrained)


def build_timm_classifier(model_name, pretrained=True):
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for timm_* classifiers. Install it on EC2 with: "
            "/opt/pytorch/bin/python -m pip install timm"
        ) from exc
    return timm.create_model(model_name, pretrained=pretrained, num_classes=1, in_chans=3)


def build_timm_feature_extractor(model_name, pretrained=True):
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for timm_* feature extractors. Install it on EC2 with: "
            "/opt/pytorch/bin/python -m pip install timm"
        ) from exc
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg", in_chans=3)
    feature_dim = int(model.num_features)
    return model, feature_dim


def build_torchvision_feature_extractor(architecture, pretrained=True, image_size=None):
    if architecture == "resnet18":
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            model = torchvision.models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            model = torchvision.models.resnet18(pretrained=pretrained)
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim

    if architecture == "resnet50":
        try:
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            model = torchvision.models.resnet50(weights=weights)
        except (AttributeError, TypeError):
            model = torchvision.models.resnet50(pretrained=pretrained)
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim

    if architecture in {"swin_t", "swin_v2_t", "swin_v2_s", "swin_v2_b"}:
        model = build_torchvision_swin(architecture, pretrained=pretrained)
        feature_dim = int(model.head.in_features)
        model.head = nn.Identity()
        return model, feature_dim

    if architecture in TIMM_ARCHITECTURES:
        return build_timm_feature_extractor(TIMM_ARCHITECTURES[architecture], pretrained=pretrained)

    if architecture in {"efficientnet_b0", "efficientnet_b3", "efficientnet_v2_s"}:
        if architecture == "efficientnet_b0":
            model_fn = torchvision.models.efficientnet_b0
            weights_enum = getattr(torchvision.models, "EfficientNet_B0_Weights", None)
        elif architecture == "efficientnet_b3":
            model_fn = torchvision.models.efficientnet_b3
            weights_enum = getattr(torchvision.models, "EfficientNet_B3_Weights", None)
        else:
            model_fn = getattr(torchvision.models, "efficientnet_v2_s", None)
            weights_enum = getattr(torchvision.models, "EfficientNet_V2_S_Weights", None)
            if model_fn is None:
                raise ValueError(
                    "efficientnet_v2_s is not available in this torchvision install. "
                    "Upgrade torchvision or use timm_efficientnetv2_s."
                )
        try:
            weights = weights_enum.DEFAULT if pretrained and weights_enum is not None else None
            model = model_fn(weights=weights)
        except (AttributeError, TypeError):
            model = model_fn(pretrained=pretrained)
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        return model, feature_dim

    if architecture == "vit_b_16":
        model = load_vit_with_interpolated_embeddings(image_size=image_size, pretrained=pretrained)
        feature_dim = int(model.heads.head.in_features)
        model.heads.head = nn.Identity()
        return model, feature_dim

    if architecture == "densenet121":
        try:
            weights = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = torchvision.models.densenet121(weights=weights)
        except (AttributeError, TypeError):
            model = torchvision.models.densenet121(pretrained=pretrained)
        feature_dim = int(model.classifier.in_features)
        model.classifier = nn.Identity()
        return model, feature_dim

    if architecture == "densenet169":
        try:
            weights = torchvision.models.DenseNet169_Weights.DEFAULT if pretrained else None
            model = torchvision.models.densenet169(weights=weights)
        except (AttributeError, TypeError):
            model = torchvision.models.densenet169(pretrained=pretrained)
        feature_dim = int(model.classifier.in_features)
        model.classifier = nn.Identity()
        return model, feature_dim

    if architecture == "densenet201":
        try:
            weights = torchvision.models.DenseNet201_Weights.DEFAULT if pretrained else None
            model = torchvision.models.densenet201(weights=weights)
        except (AttributeError, TypeError):
            model = torchvision.models.densenet201(pretrained=pretrained)
        feature_dim = int(model.classifier.in_features)
        model.classifier = nn.Identity()
        return model, feature_dim

    if architecture in {"convnext_tiny", "convnext_small", "convnext_base"}:
        model = build_torchvision_convnext(architecture, pretrained=pretrained)
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Unsupported feature extractor architecture: {architecture}")


class HybridImageNetClassifier(nn.Module):
    def __init__(self, transformer_name, cnn_name, pretrained=True, image_size=None, hidden_dim=512, dropout=0.35):
        super().__init__()
        self.transformer, transformer_dim = build_torchvision_feature_extractor(
            transformer_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.cnn, cnn_dim = build_torchvision_feature_extractor(
            cnn_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim + cnn_dim),
            nn.Linear(transformer_dim + cnn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images):
        transformer_features = self.transformer(images)
        cnn_features = self.cnn(images)
        features = torch.cat([transformer_features, cnn_features], dim=1)
        return self.classifier(features).flatten()


class HybridXRVClassifier(nn.Module):
    def __init__(self, transformer_name, xrv_weights_name, pretrained=True, image_size=None, hidden_dim=512, dropout=0.35):
        super().__init__()
        self.transformer, transformer_dim = build_torchvision_feature_extractor(
            transformer_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.xrv_cnn, xrv_dim = build_xrv_feature_extractor(xrv_weights_name if pretrained else None)
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim + xrv_dim),
            nn.Linear(transformer_dim + xrv_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images):
        transformer_images = images[:, :3, :, :]
        xrv_images = images[:, 3:4, :, :]
        transformer_features = self.transformer(transformer_images)
        xrv_features = self.xrv_cnn.features(xrv_images)
        xrv_features = F.relu(xrv_features, inplace=False)
        xrv_features = F.adaptive_avg_pool2d(xrv_features, (1, 1)).flatten(1)
        features = torch.cat([transformer_features, xrv_features], dim=1)
        return self.classifier(features).flatten()


class PatchMILClassifier(nn.Module):
    def __init__(
        self,
        backbone_name,
        pretrained=True,
        image_size=None,
        grid_size=2,
        hidden_dim=512,
        dropout=0.35,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        if self.grid_size < 1:
            raise ValueError("grid_size must be >= 1")
        self.backbone, feature_dim = build_torchvision_feature_extractor(
            backbone_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.feature_dim = feature_dim * 3

    def forward_features(self, images):
        batch_size, channels, height, width = images.shape
        patch_height = height // self.grid_size
        patch_width = width // self.grid_size
        patches = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y0 = row * patch_height
                x0 = col * patch_width
                y1 = height if row == self.grid_size - 1 else (row + 1) * patch_height
                x1 = width if col == self.grid_size - 1 else (col + 1) * patch_width
                patch = images[:, :, y0:y1, x0:x1]
                if patch.shape[-2:] != (height, width):
                    patch = F.interpolate(patch, size=(height, width), mode="bilinear", align_corners=False)
                patches.append(patch)

        patch_batch = torch.cat(patches, dim=0)
        patch_features = self.backbone(patch_batch)
        patch_features = patch_features.reshape(len(patches), batch_size, -1).permute(1, 0, 2)
        attention_logits = self.attention(patch_features).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        attention_features = torch.sum(patch_features * attention_weights, dim=1)
        max_features = torch.max(patch_features, dim=1).values
        mean_features = torch.mean(patch_features, dim=1)
        return torch.cat([attention_features, max_features, mean_features], dim=1)

    def forward(self, images):
        return self.classifier(self.forward_features(images)).flatten()


class MetadataAwareClassifier(nn.Module):
    def __init__(
        self,
        architecture,
        metadata_dim,
        pretrained=True,
        image_size=None,
        metadata_hidden_dim=32,
        dropout=0.2,
    ):
        super().__init__()
        self.architecture = architecture
        self.metadata_dim = int(metadata_dim)
        self.xrv_encoder = False
        if architecture in PATCH_ARCHITECTURES:
            self.image_encoder = PatchMILClassifier(
                PATCH_ARCHITECTURES[architecture],
                pretrained=pretrained,
                image_size=image_size,
            )
            image_dim = self.image_encoder.feature_dim
            self.patch_encoder = True
        elif architecture in XRV_ARCHITECTURES:
            self.image_encoder, image_dim = build_xrv_feature_extractor(
                XRV_ARCHITECTURES[architecture] if pretrained else None
            )
            self.patch_encoder = False
            self.xrv_encoder = True
        else:
            self.image_encoder, image_dim = build_torchvision_feature_extractor(
                architecture,
                pretrained=pretrained,
                image_size=image_size,
            )
            self.patch_encoder = False

        metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(self.metadata_dim),
            nn.Linear(self.metadata_dim, metadata_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(image_dim + metadata_hidden_dim),
            nn.Linear(image_dim + metadata_hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, images, metadata):
        if metadata is None:
            metadata = images.new_zeros((images.shape[0], self.metadata_dim))
        if self.patch_encoder:
            image_features = self.image_encoder.forward_features(images)
        elif self.xrv_encoder:
            image_features = self.image_encoder.features(images)
            image_features = F.relu(image_features, inplace=False)
            image_features = F.adaptive_avg_pool2d(image_features, (1, 1)).flatten(1)
        else:
            image_features = self.image_encoder(images)
        metadata_features = self.metadata_encoder(metadata)
        return self.classifier(torch.cat([image_features, metadata_features], dim=1)).flatten()


def build_classifier(
    architecture="resnet50",
    pretrained=True,
    image_size=None,
    metadata_dim=0,
    metadata_hidden_dim=32,
):
    if int(metadata_dim) > 0:
        supported = {
            "resnet18",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b3",
            "efficientnet_v2_s",
            "densenet121",
            "densenet169",
            "densenet201",
            "swin_t",
            "swin_v2_t",
            "swin_v2_s",
            "swin_v2_b",
            "vit_b_16",
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            *XRV_ARCHITECTURES.keys(),
            *TIMM_ARCHITECTURES.keys(),
            *PATCH_ARCHITECTURES.keys(),
        }
        if architecture not in supported:
            raise ValueError(
                f"Metadata fusion is currently supported for {sorted(supported)}, got {architecture!r}."
            )
        return MetadataAwareClassifier(
            architecture,
            metadata_dim=metadata_dim,
            pretrained=pretrained,
            image_size=image_size,
            metadata_hidden_dim=metadata_hidden_dim,
        )
    if architecture in XRV_ARCHITECTURES:
        return build_xrv_classifier(XRV_ARCHITECTURES[architecture] if pretrained else None)

    if architecture in PATCH_ARCHITECTURES:
        return PatchMILClassifier(
            PATCH_ARCHITECTURES[architecture],
            pretrained=pretrained,
            image_size=image_size,
        )

    if architecture in TIMM_ARCHITECTURES:
        return build_timm_classifier(TIMM_ARCHITECTURES[architecture], pretrained=pretrained)

    if architecture in HYBRID_XRV_ARCHITECTURES:
        transformer_name, xrv_weights_name = HYBRID_XRV_ARCHITECTURES[architecture]
        return HybridXRVClassifier(
            transformer_name,
            xrv_weights_name,
            pretrained=pretrained,
            image_size=image_size,
        )

    if architecture in HYBRID_IMAGENET_ARCHITECTURES:
        transformer_name, cnn_name = HYBRID_IMAGENET_ARCHITECTURES[architecture]
        return HybridImageNetClassifier(
            transformer_name,
            cnn_name,
            pretrained=pretrained,
            image_size=image_size,
        )

    if architecture == "resnet18":
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            model = torchvision.models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if architecture in {"efficientnet_b0", "efficientnet_b3", "efficientnet_v2_s"}:
        if architecture == "efficientnet_b0":
            model_fn = torchvision.models.efficientnet_b0
            weights_enum = getattr(torchvision.models, "EfficientNet_B0_Weights", None)
        elif architecture == "efficientnet_b3":
            model_fn = torchvision.models.efficientnet_b3
            weights_enum = getattr(torchvision.models, "EfficientNet_B3_Weights", None)
        else:
            model_fn = getattr(torchvision.models, "efficientnet_v2_s", None)
            weights_enum = getattr(torchvision.models, "EfficientNet_V2_S_Weights", None)
            if model_fn is None:
                raise ValueError(
                    "efficientnet_v2_s is not available in this torchvision install. "
                    "Upgrade torchvision or use timm_efficientnetv2_s."
                )
        try:
            weights = weights_enum.DEFAULT if pretrained and weights_enum is not None else None
            model = model_fn(weights=weights)
        except (AttributeError, TypeError):
            model = model_fn(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model

    if architecture == "densenet121":
        try:
            weights = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = torchvision.models.densenet121(weights=weights)
        except (AttributeError, TypeError):
            model = torchvision.models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model

    if architecture == "densenet169":
        try:
            weights = torchvision.models.DenseNet169_Weights.DEFAULT if pretrained else None
            model = torchvision.models.densenet169(weights=weights)
        except (AttributeError, TypeError):
            model = torchvision.models.densenet169(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model

    if architecture == "densenet201":
        try:
            weights = torchvision.models.DenseNet201_Weights.DEFAULT if pretrained else None
            model = torchvision.models.densenet201(weights=weights)
        except (AttributeError, TypeError):
            model = torchvision.models.densenet201(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model

    if architecture in {"swin_t", "swin_v2_t", "swin_v2_s", "swin_v2_b"}:
        model = build_torchvision_swin(architecture, pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, 1)
        return model

    if architecture in {"convnext_tiny", "convnext_small", "convnext_base"}:
        model = build_torchvision_convnext(architecture, pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model

    if architecture == "inception_v3":
        try:
            weights = torchvision.models.Inception_V3_Weights.DEFAULT if pretrained else None
            model = torchvision.models.inception_v3(weights=weights, aux_logits=True)
        except (AttributeError, TypeError):
            model = torchvision.models.inception_v3(pretrained=pretrained, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
        if getattr(model, "AuxLogits", None) is not None:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)
        return model

    if architecture == "vit_b_16":
        model = load_vit_with_interpolated_embeddings(image_size=image_size, pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, 1)
        return model

    try:
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
    except (AttributeError, TypeError):
        model = torchvision.models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def autocast_context(use_amp):
    if not use_amp:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except AttributeError:
        return torch.cuda.amp.autocast()


def classifier_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def forward_classifier(model, images, metadata=None):
    if getattr(model, "metadata_dim", 0) > 0:
        return model(images, metadata)
    return model(images)


def load_patient_ids(path):
    if path is None:
        return []

    id_path = Path(path).expanduser()
    if not id_path.exists():
        raise FileNotFoundError(f"Missing hard-negative file: {id_path}")

    if id_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(id_path)
        if "patientId" not in dataframe.columns:
            raise ValueError(f"Expected a patientId column in {id_path}")
        return dataframe["patientId"].dropna().astype(str).tolist()

    return [line.strip() for line in id_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_label_overrides(path):
    if path is None:
        return {}

    label_path = Path(path).expanduser()
    if not label_path.exists():
        raise FileNotFoundError(f"Missing reviewed-label file: {label_path}")

    dataframe = pd.read_csv(label_path)
    if "patientId" not in dataframe.columns:
        raise ValueError(f"Expected a patientId column in {label_path}")

    label_column = None
    for candidate in ("review_label", "label", "Target", "target"):
        if candidate in dataframe.columns:
            label_column = candidate
            break
    if label_column is None:
        raise ValueError(
            f"Expected one of review_label, label, Target, or target columns in {label_path}"
        )

    overrides = {}
    for row in dataframe[["patientId", label_column]].dropna().itertuples(index=False):
        patient_id = str(row[0])
        raw_label = row[1]
        if isinstance(raw_label, str) and not raw_label.strip():
            continue
        label = float(raw_label)
        if label not in (0.0, 1.0):
            raise ValueError(f"Reviewed label for {patient_id} must be 0 or 1, got {raw_label!r}")
        overrides[patient_id] = label
    return overrides


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probabilities = torch.sigmoid(inputs)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce_loss
        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, pos_weight=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        probabilities = torch.sigmoid(inputs)
        positive_probabilities = probabilities.clamp(min=self.eps, max=1.0 - self.eps)
        negative_probabilities = (1.0 - probabilities).clamp(min=self.eps, max=1.0 - self.eps)
        if self.clip > 0:
            negative_probabilities = (negative_probabilities + self.clip).clamp(max=1.0)

        positive_loss = targets * torch.log(positive_probabilities)
        negative_loss = (1.0 - targets) * torch.log(negative_probabilities)
        loss = positive_loss + negative_loss

        positive_focus = (1.0 - positive_probabilities).pow(self.gamma_pos)
        negative_focus = probabilities.pow(self.gamma_neg)
        focus = targets * positive_focus + (1.0 - targets) * negative_focus
        loss = loss * focus

        if self.pos_weight is not None:
            weight = targets * (self.pos_weight - 1.0) + 1.0
            loss = loss * weight

        return -loss.mean()


class SoftFbetaLoss(nn.Module):
    def __init__(self, beta=1.0, eps=1e-7, bce_weight=0.2, pos_weight=None):
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)
        self.bce_weight = float(bce_weight)
        self.register_buffer(
            "pos_weight_buffer",
            pos_weight.detach().clone() if pos_weight is not None else torch.tensor([]),
            persistent=False,
        )

    def forward(self, inputs, targets):
        probabilities = torch.sigmoid(inputs)
        targets = targets.float()
        true_positive = torch.sum(probabilities * targets)
        false_positive = torch.sum(probabilities * (1.0 - targets))
        false_negative = torch.sum((1.0 - probabilities) * targets)
        beta_squared = self.beta * self.beta
        soft_fbeta = ((1.0 + beta_squared) * true_positive + self.eps) / (
            (1.0 + beta_squared) * true_positive
            + beta_squared * false_negative
            + false_positive
            + self.eps
        )
        loss = 1.0 - soft_fbeta
        if self.bce_weight > 0.0:
            pos_weight = self.pos_weight_buffer if self.pos_weight_buffer.numel() else None
            bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
            loss = loss + self.bce_weight * bce
        return loss


def make_eval_loaders(
    detection_dataset,
    image_size,
    normalization,
    batch_size,
    num_workers,
    device,
    tta_modes=None,
    metadata_by_id=None,
):
    loaders = []
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker

    for mode in tta_modes or ["none"]:
        dataset = RSNAClassificationDataset(
            detection_dataset,
            classifier_eval_transform(image_size, normalization=normalization, tta_mode=mode),
            metadata_by_id=metadata_by_id,
        )
        loaders.append(
            (
                mode,
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=classification_collate,
                    **loader_kwargs,
                ),
            )
        )
    return loaders


def make_loaders(args, data_dir, device):
    train_detection, valid_detection, test_detection = prepare_data(
        data_dir=data_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    label_overrides = load_label_overrides(args.review_label_file)
    metadata_by_id = load_metadata_table(args.metadata_csv) if args.metadata_csv else None
    if metadata_by_id is not None:
        all_ids = set(train_detection.image_ids.astype(str))
        all_ids.update(valid_detection.image_ids.astype(str))
        all_ids.update(test_detection.image_ids.astype(str))
        matched = len(all_ids.intersection(metadata_by_id))
        print(f"Using DICOM metadata for {matched}/{len(all_ids)} split patients")
    if label_overrides:
        train_ids = set(train_detection.image_ids.astype(str).tolist())
        matched_overrides = sorted(set(label_overrides).intersection(train_ids))
        ignored_overrides = sorted(set(label_overrides).difference(train_ids))
        print(
            f"Using {len(matched_overrides)} reviewed training labels "
            f"({len(ignored_overrides)} reviewed IDs are outside the train split and ignored)"
        )

    train_dataset = RSNAClassificationDataset(
        train_detection,
        classifier_train_transform(
            args.image_size,
            augmentation=args.augmentation,
            normalization=args.effective_normalization,
        ),
        label_overrides=label_overrides,
        label_smoothing=args.label_smoothing,
        metadata_by_id=metadata_by_id,
    )
    valid_dataset = RSNAClassificationDataset(
        valid_detection,
        classifier_eval_transform(args.image_size, normalization=args.effective_normalization),
        metadata_by_id=metadata_by_id,
    )
    test_dataset = RSNAClassificationDataset(
        test_detection,
        classifier_eval_transform(args.image_size, normalization=args.effective_normalization),
        metadata_by_id=metadata_by_id,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker

    hard_negative_ids = set(load_patient_ids(args.hard_negative_file))
    sampler = None
    batch_sampler = None
    shuffle = True
    train_labels = train_dataset.labels()
    train_image_ids = train_dataset.image_ids.astype(str).tolist()
    train_id_set = set(train_image_ids)
    matched_hard_negative_ids = hard_negative_ids.intersection(train_id_set)

    if args.balanced_batches:
        batch_sampler = BalancedBinaryBatchSampler(
            train_labels,
            batch_size=args.batch_size,
            seed=args.seed,
            hard_negative_ids=hard_negative_ids,
            image_ids=train_image_ids,
            hard_negative_weight=args.hard_negative_weight,
        )
        shuffle = False
        print(
            f"Using balanced 50/50 classifier batches "
            f"({len(matched_hard_negative_ids)} matched hard negatives with weight {args.hard_negative_weight:g})"
        )
    elif hard_negative_ids:
        matched_hard_negative_ids = hard_negative_ids.intersection(train_id_set)
        weights = [
            args.hard_negative_weight if str(image_id) in hard_negative_ids else 1.0
            for image_id in train_dataset.image_ids
        ]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
        print(
            f"Using {len(matched_hard_negative_ids)} matched classifier hard-negative training images "
            f"from {len(hard_negative_ids)} provided IDs with weight {args.hard_negative_weight:g}"
        )

    train_loader_kwargs = {
        "shuffle": shuffle,
        "sampler": sampler,
        "generator": None if sampler is not None else generator,
        "collate_fn": classification_collate,
        **loader_kwargs,
    }
    if batch_sampler is not None:
        train_loader_kwargs = {
            "batch_sampler": batch_sampler,
            "collate_fn": classification_collate,
            **loader_kwargs,
        }
    else:
        train_loader_kwargs["batch_size"] = args.batch_size
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=classification_collate,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=classification_collate,
        **loader_kwargs,
    )
    return (
        train_loader,
        valid_loader,
        test_loader,
        train_dataset,
        valid_detection,
        test_detection,
        metadata_by_id,
    )


def metrics_from_probabilities(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = (probabilities >= threshold).astype(np.int64)

    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (true_positive + true_negative) / max(len(labels), 1)
    return {
        "image_accuracy": accuracy,
        "image_precision": precision,
        "image_recall": recall,
        "image_f1": f1,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "num_images": int(len(labels)),
        "score_threshold": float(threshold),
    }


def best_threshold_metrics(labels, probabilities, thresholds):
    rows = [metrics_from_probabilities(labels, probabilities, threshold) for threshold in thresholds]
    table = pd.DataFrame(rows).sort_values(
        ["image_f1", "image_precision", "image_recall"],
        ascending=[False, False, False],
    )
    return table.iloc[0].to_dict(), table


def apply_binary_mixup(images, metadata, labels, alpha):
    if alpha <= 0:
        return images, metadata, labels
    hard_labels = labels >= 0.5
    positive_indices = torch.nonzero(hard_labels, as_tuple=False).flatten()
    negative_indices = torch.nonzero(~hard_labels, as_tuple=False).flatten()
    pair_count = min(len(positive_indices), len(negative_indices))
    if pair_count == 0:
        return images, metadata, labels

    positive_indices = positive_indices[torch.randperm(len(positive_indices), device=images.device)[:pair_count]]
    negative_indices = negative_indices[torch.randperm(len(negative_indices), device=images.device)[:pair_count]]
    lam = np.random.beta(alpha, alpha)
    lam = float(max(lam, 1.0 - lam))

    mixed_images = images.clone()
    mixed_metadata = metadata.clone() if metadata is not None else None
    mixed_labels = labels.clone()
    mixed_images[negative_indices] = lam * images[negative_indices] + (1.0 - lam) * images[positive_indices]
    if mixed_metadata is not None:
        mixed_metadata[negative_indices] = (
            lam * metadata[negative_indices] + (1.0 - lam) * metadata[positive_indices]
        )
    mixed_labels[negative_indices] = lam * labels[negative_indices] + (1.0 - lam) * labels[positive_indices]
    return mixed_images, mixed_metadata, mixed_labels


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp, grad_accum_steps=1, mixup_alpha=0.0):
    model.train()
    total_loss = 0.0
    total_examples = 0
    non_blocking = device.type == "cuda"
    grad_accum_steps = max(int(grad_accum_steps), 1)

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(dataloader, total=len(dataloader), desc="Training", leave=True, file=sys.stdout)
    for step, batch in enumerate(progress, start=1):
        images, metadata, labels, _ = unpack_classification_batch(batch)
        images = images.to(device, non_blocking=non_blocking)
        if metadata is not None:
            metadata = metadata.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        images, metadata, labels = apply_binary_mixup(images, metadata, labels, mixup_alpha)

        with autocast_context(use_amp):
            logits = classifier_logits(forward_classifier(model, images, metadata)).flatten()
            loss = criterion(logits, labels)

        scaler.scale(loss / grad_accum_steps).backward()
        if step % grad_accum_steps == 0 or step == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu()) * images.shape[0]
        total_examples += images.shape[0]

    return total_loss / max(total_examples, 1)


def evaluate(model, dataloader, criterion, device, use_amp, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    rows = []
    probabilities = []
    labels_all = []
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader), desc=desc, leave=True, file=sys.stdout)
        for batch in progress:
            images, metadata, labels, image_ids = unpack_classification_batch(batch)
            images = images.to(device, non_blocking=non_blocking)
            if metadata is not None:
                metadata = metadata.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            with autocast_context(use_amp):
                logits = classifier_logits(forward_classifier(model, images, metadata)).flatten()
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            total_loss += float(loss.detach().cpu()) * images.shape[0]
            total_examples += images.shape[0]
            probabilities.extend(probs.tolist())
            labels_all.extend(labels_np.tolist())
            for image_id, label, probability in zip(image_ids, labels_np, probs):
                rows.append(
                    {
                        "patientId": image_id,
                        "truth": int(label),
                        "probability": float(probability),
                    }
                )

    return total_loss / max(total_examples, 1), labels_all, probabilities, pd.DataFrame(rows)


def evaluate_tta(model, eval_loaders, criterion, device, use_amp, desc="Evaluating"):
    probability_sums = {}
    probability_counts = {}
    labels_by_id = {}
    total_loss = 0.0

    for mode, dataloader in eval_loaders:
        mode_desc = desc if len(eval_loaders) == 1 else f"{desc} {mode}"
        mode_loss, _, _, predictions = evaluate(
            model,
            dataloader,
            criterion,
            device,
            use_amp,
            desc=mode_desc,
        )
        total_loss += mode_loss
        for row in predictions.to_dict("records"):
            patient_id = row["patientId"]
            probability_sums[patient_id] = probability_sums.get(patient_id, 0.0) + float(row["probability"])
            probability_counts[patient_id] = probability_counts.get(patient_id, 0) + 1
            labels_by_id[patient_id] = int(row["truth"])

    rows = []
    for patient_id in labels_by_id:
        rows.append(
            {
                "patientId": patient_id,
                "truth": labels_by_id[patient_id],
                "probability": probability_sums[patient_id] / max(probability_counts[patient_id], 1),
            }
        )
    predictions = pd.DataFrame(rows)
    labels_all = predictions["truth"].tolist()
    probabilities = predictions["probability"].tolist()
    return total_loss / max(len(eval_loaders), 1), labels_all, probabilities, predictions


def save_plots(output_dir, train_losses, validation_f1):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.legend()
    plt.savefig(output_dir / "classifier_loss.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(validation_f1, label="Validation F1")
    plt.legend()
    plt.savefig(output_dir / "classifier_f1.png", bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image-level RSNA pneumonia classifier.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--architecture",
        choices=[
            "resnet18",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b3",
            "efficientnet_v2_s",
            "densenet121",
            "densenet169",
            "densenet201",
            "inception_v3",
            "swin_t",
            "swin_v2_t",
            "swin_v2_s",
            "swin_v2_b",
            "vit_b_16",
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            *XRV_ARCHITECTURES.keys(),
            *HYBRID_XRV_ARCHITECTURES.keys(),
            *HYBRID_IMAGENET_ARCHITECTURES.keys(),
            *TIMM_ARCHITECTURES.keys(),
            *PATCH_ARCHITECTURES.keys(),
        ],
        default="resnet50",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--augmentation", choices=["basic", "cxr"], default="basic")
    parser.add_argument("--normalization", choices=["auto", "imagenet", "xrv", "hybrid_xrv_imagenet"], default="auto")
    parser.add_argument("--loss", choices=["bce", "focal", "asl", "soft_fbeta"], default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--asl-use-pos-weight", action="store_true")
    parser.add_argument("--soft-fbeta-beta", type=float, default=1.0)
    parser.add_argument("--soft-fbeta-bce-weight", type=float, default=0.2)
    parser.add_argument("--soft-fbeta-use-pos-weight", action="store_true")
    parser.add_argument("--eval-tta-modes", default="none")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negative-file", default=None, help="CSV/text file of classifier hard-negative patient IDs.")
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--balanced-batches", action="store_true", help="Use 50/50 positive/negative batches for classifier training.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Positive/negative targeted mixup strength. Use 0 to disable.")
    parser.add_argument("--review-label-file", default=None, help="CSV with patientId and review_label/label for manually reviewed training labels.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Binary label smoothing applied to training labels only.")
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="CSV produced by extract_dicom_metadata.py; enables age/sex/view fusion.",
    )
    parser.add_argument("--metadata-hidden-dim", type=int, default=32)
    parser.add_argument("--resume-checkpoint", default=None, help="Load classifier weights from an existing checkpoint before fine-tuning.")
    parser.add_argument(
        "--no-save-epoch-checkpoints",
        action="store_true",
        help="Only save classifier_best.pt instead of saving a full checkpoint after every epoch.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    args.effective_normalization = normalization_for_architecture(args.architecture, args.normalization)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "classifier_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    thresholds = parse_thresholds(args.thresholds)
    eval_tta_modes = parse_tta_modes(args.eval_tta_modes)

    print(f"Using data directory: {data_dir}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"Architecture: {args.architecture}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Normalization: {args.effective_normalization}")
    print(f"Loss: {args.loss}")
    if args.loss == "soft_fbeta":
        print(
            "Soft F-beta settings: "
            f"beta={args.soft_fbeta_beta:g}, "
            f"bce_weight={args.soft_fbeta_bce_weight:g}, "
            f"use_pos_weight={args.soft_fbeta_use_pos_weight}"
        )
    print(f"Label smoothing: {args.label_smoothing:g}")
    print(f"Balanced batches: {args.balanced_batches}")
    print(f"Mixup alpha: {args.mixup_alpha:g}")
    print(f"Eval TTA modes: {','.join(eval_tta_modes)}")
    print(f"DICOM metadata: {args.metadata_csv or 'disabled'}")

    (
        train_loader,
        valid_loader,
        test_loader,
        train_dataset,
        valid_detection,
        test_detection,
        metadata_by_id,
    ) = make_loaders(args, data_dir, device)
    train_labels = train_dataset.labels()
    num_positive = sum(train_labels)
    num_negative = len(train_labels) - num_positive
    pos_weight = torch.tensor([num_negative / max(num_positive, 1)], dtype=torch.float32, device=device)

    classifier = build_classifier(
        args.architecture,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
        metadata_dim=len(METADATA_FEATURE_COLUMNS) if metadata_by_id is not None else 0,
        metadata_hidden_dim=args.metadata_hidden_dim,
    ).to(device)
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        checkpoint_architecture = checkpoint.get("architecture")
        checkpoint_image_size = checkpoint.get("image_size")
        checkpoint_metadata_dim = int(checkpoint.get("metadata_dim", 0))
        if checkpoint_architecture and checkpoint_architecture != args.architecture:
            raise ValueError(
                f"Resume checkpoint architecture is {checkpoint_architecture!r}, "
                f"but --architecture is {args.architecture!r}."
            )
        if checkpoint_image_size and int(checkpoint_image_size) != int(args.image_size):
            raise ValueError(
                f"Resume checkpoint image_size is {checkpoint_image_size}, "
                f"but --image-size is {args.image_size}."
            )
        expected_metadata_dim = len(METADATA_FEATURE_COLUMNS) if metadata_by_id is not None else 0
        if checkpoint_metadata_dim != expected_metadata_dim:
            raise ValueError(
                f"Resume checkpoint metadata_dim is {checkpoint_metadata_dim}, "
                f"but this run expects {expected_metadata_dim}."
            )
        classifier.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed classifier weights from {resume_path} (epoch={checkpoint.get('epoch')})")

    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss == "asl":
        criterion = AsymmetricLoss(
            gamma_pos=args.asl_gamma_pos,
            gamma_neg=args.asl_gamma_neg,
            clip=args.asl_clip,
            pos_weight=pos_weight if args.asl_use_pos_weight else None,
        )
    elif args.loss == "soft_fbeta":
        criterion = SoftFbetaLoss(
            beta=args.soft_fbeta_beta,
            bce_weight=args.soft_fbeta_bce_weight,
            pos_weight=pos_weight if args.soft_fbeta_use_pos_weight else None,
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_losses = []
    validation_f1 = []
    history = []
    best = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            classifier,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            grad_accum_steps=args.grad_accum_steps,
            mixup_alpha=args.mixup_alpha,
        )
        if len(eval_tta_modes) == 1 and eval_tta_modes[0] == "none":
            valid_loss, valid_labels, valid_probs, valid_predictions = evaluate(
                classifier,
                valid_loader,
                criterion,
                device,
                use_amp,
                desc="Validating",
            )
        else:
            valid_loss, valid_labels, valid_probs, valid_predictions = evaluate_tta(
                classifier,
                make_eval_loaders(
                    valid_detection,
                    args.image_size,
                    args.effective_normalization,
                    args.batch_size,
                    args.num_workers,
                    device,
                    eval_tta_modes,
                    metadata_by_id=metadata_by_id,
                ),
                criterion,
                device,
                use_amp,
                desc="Validating TTA",
            )
        valid_best, valid_threshold_table = best_threshold_metrics(valid_labels, valid_probs, thresholds)
        valid_predictions["prediction"] = (
            valid_predictions["probability"] >= float(valid_best["score_threshold"])
        ).astype(int)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            **valid_best,
        }
        history.append(epoch_metrics)
        train_losses.append(train_loss)
        validation_f1.append(valid_best["image_f1"])
        pd.DataFrame(history).to_csv(output_dir / "classifier_validation_metrics.csv", index=False)
        valid_threshold_table.to_csv(output_dir / f"classifier_validation_thresholds_epoch_{epoch}.csv", index=False)
        valid_predictions.to_csv(output_dir / f"classifier_validation_predictions_epoch_{epoch}.csv", index=False)

        is_best = best is None or (
            epoch_metrics["image_f1"],
            epoch_metrics["image_precision"],
            epoch_metrics["image_recall"],
        ) > (
            best["image_f1"],
            best["image_precision"],
            best["image_recall"],
        )
        if is_best:
            best = epoch_metrics
            best_state = {
                "epoch": epoch,
                "architecture": args.architecture,
                "image_size": args.image_size,
                "augmentation": args.augmentation,
                "normalization": args.effective_normalization,
                "grad_accum_steps": args.grad_accum_steps,
                "loss": args.loss,
                "soft_fbeta_beta": args.soft_fbeta_beta,
                "soft_fbeta_bce_weight": args.soft_fbeta_bce_weight,
                "soft_fbeta_use_pos_weight": args.soft_fbeta_use_pos_weight,
                "label_smoothing": args.label_smoothing,
                "balanced_batches": args.balanced_batches,
                "mixup_alpha": args.mixup_alpha,
                "review_label_file": args.review_label_file,
                "resume_checkpoint": args.resume_checkpoint,
                "eval_tta_modes": eval_tta_modes,
                "metadata_csv": args.metadata_csv,
                "metadata_dim": len(METADATA_FEATURE_COLUMNS) if metadata_by_id is not None else 0,
                "metadata_columns": list(METADATA_FEATURE_COLUMNS),
                "metadata_hidden_dim": args.metadata_hidden_dim,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_metrics": best,
            }
            torch.save(best_state, output_dir / "classifier_best.pt")

        if not args.no_save_epoch_checkpoints:
            torch.save(
                {
                    "epoch": epoch,
                    "architecture": args.architecture,
                    "image_size": args.image_size,
                    "augmentation": args.augmentation,
                    "normalization": args.effective_normalization,
                    "grad_accum_steps": args.grad_accum_steps,
                    "loss": args.loss,
                    "soft_fbeta_beta": args.soft_fbeta_beta,
                    "soft_fbeta_bce_weight": args.soft_fbeta_bce_weight,
                    "soft_fbeta_use_pos_weight": args.soft_fbeta_use_pos_weight,
                    "label_smoothing": args.label_smoothing,
                    "balanced_batches": args.balanced_batches,
                    "mixup_alpha": args.mixup_alpha,
                    "review_label_file": args.review_label_file,
                    "resume_checkpoint": args.resume_checkpoint,
                    "eval_tta_modes": eval_tta_modes,
                    "metadata_csv": args.metadata_csv,
                    "metadata_dim": len(METADATA_FEATURE_COLUMNS) if metadata_by_id is not None else 0,
                    "metadata_columns": list(METADATA_FEATURE_COLUMNS),
                    "metadata_hidden_dim": args.metadata_hidden_dim,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": epoch_metrics,
                },
                output_dir / f"classifier_epoch_{epoch}.pt",
            )
        print(
            f"Epoch #{epoch}, Loss: {train_loss:.5f}, Val Loss: {valid_loss:.5f}, "
            f"F1: {valid_best['image_f1']:.5f}, Precision: {valid_best['image_precision']:.5f}, "
            f"Recall: {valid_best['image_recall']:.5f}, Threshold: {valid_best['score_threshold']:.2f}"
        )

    save_plots(output_dir, train_losses, validation_f1)

    if best_state is None:
        best_state = torch.load(output_dir / "classifier_best.pt", map_location=device)
    classifier.load_state_dict(best_state["model_state_dict"])
    if len(eval_tta_modes) == 1 and eval_tta_modes[0] == "none":
        test_loss, test_labels, test_probs, test_predictions = evaluate(
            classifier,
            test_loader,
            criterion,
            device,
            use_amp,
            desc="Testing",
        )
    else:
        test_loss, test_labels, test_probs, test_predictions = evaluate_tta(
            classifier,
            make_eval_loaders(
                test_detection,
                args.image_size,
                args.effective_normalization,
                args.batch_size,
                args.num_workers,
                device,
                eval_tta_modes,
                metadata_by_id=metadata_by_id,
            ),
            criterion,
            device,
            use_amp,
            desc="Testing TTA",
        )
    test_metrics = metrics_from_probabilities(test_labels, test_probs, float(best["score_threshold"]))
    test_metrics = {
        "selected_from": "validation_classifier_threshold",
        "checkpoint_epoch": best["epoch"],
        "test_loss": test_loss,
        **test_metrics,
    }
    test_predictions["prediction"] = (
        test_predictions["probability"] >= float(test_metrics["score_threshold"])
    ).astype(int)
    pd.DataFrame([test_metrics]).to_csv(output_dir / "classifier_test_metrics.csv", index=False)
    with (output_dir / "classifier_test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(test_metrics, metrics_file, indent=2)
    test_predictions.to_csv(output_dir / "classifier_test_predictions.csv", index=False)
    save_confusion_matrix(
        output_dir / "classifier_test_confusion_matrix.png",
        test_metrics,
        f"Classifier Held-Out Test, threshold={test_metrics['score_threshold']:g}",
    )

    print("\nBest validation classifier setting:")
    print(pd.Series(best).to_string())
    print("\nHeld-out test metrics:")
    print(pd.Series(test_metrics).to_string())


if __name__ == "__main__":
    main()

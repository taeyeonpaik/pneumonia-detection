import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def apply_float_clahe(image, **kwargs):
    image = image.astype(np.float32)
    image_uint8 = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(image_uint8)
    equalized = cv2.merge([clahe.apply(channel) for channel in channels])
    return equalized.astype(np.float32) / 255.0


class RSNADataset(Dataset):
    def __init__(self, dataframe, image_directory, transform=None):
        super().__init__()
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_ids = self.dataframe["patientId"].drop_duplicates().reset_index(drop=True)
        self.records_by_image_id = {
            image_id: records for image_id, records in self.dataframe.groupby("patientId", sort=False)
        }
        self.image_directory = Path(image_directory)
        self.transform = transform

    def __getitem__(self, index):
        image_id = self.image_ids.iloc[index]
        records = self.records_by_image_id[image_id]
        image_path = self.image_directory / f"{image_id}.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        boxes = []
        labels = []
        for _, row in records.iterrows():
            if int(row["Target"]) != 1:
                continue
            x = float(row["x"])
            y = float(row["y"])
            width = float(row["width"])
            height = float(row["height"])
            boxes.append([x, y, x + width, y + height])
            labels.append(1)

        if self.transform is not None:
            sample = self.transform(image=image, bboxes=boxes, labels=labels)
            image = sample["image"]
            boxes = sample["bboxes"]
            labels = sample["labels"]

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": iscrowd,
        }
        return image, target, image_id

    def __len__(self):
        return len(self.image_ids)

    def image_level_targets(self):
        targets = []
        for image_id in self.image_ids:
            records = self.records_by_image_id[image_id]
            targets.append(int(records["Target"].max()))
        return targets


def get_train_transform(augmentation="basic"):
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
    ]
    if augmentation == "cxr":
        transforms.extend(
            [
                A.Lambda(image=apply_float_clahe, p=0.35),
                A.RandomGamma(gamma_limit=(85, 115), p=0.35),
                A.RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.08, p=0.35),
                A.GaussNoise(p=0.20),
            ]
        )
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0,
        ),
    )


def get_valid_test_transform():
    return A.Compose(
        [ToTensorV2(p=1.0)],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0,
        ),
    )


def collate(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_data_dir(data_dir=None):
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    if os.environ.get("RSNA_DATA_DIR"):
        return Path(os.environ["RSNA_DATA_DIR"]).expanduser().resolve()
    return Path(__file__).resolve().parent


def find_training_image_directory(data_dir):
    candidates = [
        data_dir / "train_images_png",
        data_dir / "stage_2_train_images_png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing training image directory. Checked: {', '.join(str(path) for path in candidates)}"
    )


def prepare_data(data_dir=None, valid_size=2500, test_size=2500, seed=42, train_augmentation="basic"):
    data_dir = resolve_data_dir(data_dir)
    image_directory = find_training_image_directory(data_dir)
    labels_path = data_dir / "stage_2_train_labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_path}")

    dataframe = pd.read_csv(labels_path)
    image_ids = dataframe["patientId"].drop_duplicates().reset_index(drop=True)

    if len(image_ids) <= valid_size + test_size:
        raise ValueError(
            f"Need more than {valid_size + test_size} unique images for the requested split; "
            f"found {len(image_ids)}."
        )

    image_ids = image_ids.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_ids = image_ids[: -(valid_size + test_size)]
    valid_ids = image_ids[-(valid_size + test_size) : -test_size]
    test_ids = image_ids[-test_size:]

    train_dataframe = dataframe[dataframe["patientId"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["patientId"].isin(valid_ids)].reset_index(drop=True)
    test_dataframe = dataframe[dataframe["patientId"].isin(test_ids)].reset_index(drop=True)

    train_dataset = RSNADataset(train_dataframe, image_directory, get_train_transform(train_augmentation))
    valid_dataset = RSNADataset(valid_dataframe, image_directory, get_valid_test_transform())
    test_dataset = RSNADataset(test_dataframe, image_directory, get_valid_test_transform())
    return train_dataset, valid_dataset, test_dataset


def get_data_loader(
    batch_size,
    data_dir=None,
    num_workers=2,
    valid_size=2500,
    test_size=2500,
    pin_memory=None,
    seed=42,
    sampling_seed=None,
    balanced_sampler=False,
    positive_weight=3.0,
    hard_negative_ids=None,
    hard_negative_weight=1.0,
    train_augmentation="basic",
):
    sampling_seed = seed if sampling_seed is None else int(sampling_seed)
    train_dataset, valid_dataset, test_dataset = prepare_data(
        data_dir=data_dir,
        valid_size=valid_size,
        test_size=test_size,
        seed=seed,
        train_augmentation=train_augmentation,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(sampling_seed)
    worker_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = True
        worker_kwargs["prefetch_factor"] = 2
        worker_kwargs["worker_init_fn"] = seed_worker

    sampler = None
    shuffle = True
    hard_negative_ids = set(hard_negative_ids or [])
    if balanced_sampler or hard_negative_ids:
        image_targets = train_dataset.image_level_targets()
        weights = []
        for image_id, target in zip(train_dataset.image_ids, image_targets):
            weight = positive_weight if balanced_sampler and target == 1 else 1.0
            if image_id in hard_negative_ids:
                weight *= hard_negative_weight
            weights.append(weight)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate,
        generator=None if sampler is not None else generator,
        **worker_kwargs,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        **worker_kwargs,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        **worker_kwargs,
    )
    return train_dataloader, valid_dataloader, test_dataloader

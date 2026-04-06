from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    data_dir: str
    output_dir: str = "./outputs"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    unfreeze_epoch: int = 5
    patience: int = 7
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    num_workers: int = 4
    seed: int = 42
    dropout_1: float = 0.4
    dropout_2: float = 0.2
    export_onnx: bool = True
    freeze_backbone: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Logging
# ============================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


logger = logging.getLogger(__name__)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset / Splitting
# ============================================================

def validate_dataset_dir(data_dir: str) -> None:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    if not path.is_dir():
        raise NotADirectoryError(f"Provided data_dir is not a directory: {data_dir}")

    class_dirs = [p for p in path.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in dataset directory: {data_dir}")


def get_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label = self.subset.dataset[self.subset.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


def stratified_split_indices(
    targets: List[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {}

    for idx, label in enumerate(targets):
        class_to_indices.setdefault(label, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for label, indices in class_to_indices.items():
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Guarantee at least one sample in val/test if possible
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
        elif n == 2:
            n_train = 1
            n_val = 0
        elif n == 1:
            n_train = 1
            n_val = 0

        train_split = indices[:n_train]
        val_split = indices[n_train:n_train + n_val]
        test_split = indices[n_train + n_val:]

        train_indices.extend(train_split)
        val_indices.extend(val_split)
        test_indices.extend(test_split)

        logger.info(
            "Class %s -> total=%d, train=%d, val=%d, test=%d",
            label, n, len(train_split), len(val_split), len(test_split)
        )

    return train_indices, val_indices, test_indices


def build_dataloaders(config: Config):
    train_transform, eval_transform = get_transforms(config.image_size)

    base_dataset = datasets.ImageFolder(config.data_dir)
    class_names = base_dataset.classes
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError("Dataset must contain at least 2 classes.")

    targets = base_dataset.targets
    train_idx, val_idx, test_idx = stratified_split_indices(
        targets=targets,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_subset = TransformSubset(Subset(base_dataset, train_idx), transform=train_transform)
    val_subset = TransformSubset(Subset(base_dataset, val_idx), transform=eval_transform)
    test_subset = TransformSubset(Subset(base_dataset, test_idx), transform=eval_transform)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info("Classes: %s", class_names)
    logger.info(
        "Dataset sizes -> train=%d, val=%d, test=%d",
        len(train_subset), len(val_subset), len(test_subset)
    )

    return train_loader, val_loader, test_loader, class_names, num_classes, targets


def compute_class_weights(targets: List[int], num_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount(targets, minlength=num_classes)
    if np.any(counts == 0):
        logger.warning("Some classes have zero samples in the full dataset.")

    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    logger.info("Class counts: %s", counts.tolist())
    logger.info("Class weights: %s", weights.tolist())

    return weights_tensor
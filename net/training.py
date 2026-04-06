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





# ============================================================
# Model
# ============================================================

def build_model(num_classes: int, config: Config) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if config.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(config.dropout_1),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(config.dropout_2),
        nn.Linear(256, num_classes),
    )

    return model.to(config.device)


def create_optimizer(model: nn.Module, config: Config, backbone_unfrozen: bool = False):
    if not backbone_unfrozen:
        return optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return optim.Adam(
        [
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": head_params, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds.eq(labels)).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds.eq(labels)).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: Config,
):
    optimizer = create_optimizer(model, config, backbone_unfrozen=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -math.inf
    best_state_dict = None
    patience_counter = 0
    backbone_unfrozen = False

    logger.info("=" * 90)
    logger.info(
        "%6s | %10s | %9s | %10s | %9s | %10s",
        "Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "LR"
    )
    logger.info("=" * 90)

    for epoch in range(1, config.epochs + 1):
        if config.freeze_backbone and (not backbone_unfrozen) and epoch == config.unfreeze_epoch + 1:
            logger.info("Unfreezing backbone at epoch %d", epoch)
            for param in model.parameters():
                param.requires_grad = True

            optimizer = create_optimizer(model, config, backbone_unfrozen=True)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.epochs - epoch + 1, 1),
                eta_min=1e-6,
            )
            backbone_unfrozen = True

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, config.device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "%6d | %10.4f | %8.2f%% | %10.4f | %8.2f%% | %10.6f",
            epoch,
            train_loss,
            train_acc * 100,
            val_loss,
            val_acc * 100,
            current_lr,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
            logger.info("New best model found at epoch %d (val_acc=%.4f)", epoch, val_acc)
        else:
            patience_counter += 1
            logger.info(
                "No improvement. Early stopping counter: %d/%d",
                patience_counter, config.patience
            )

        if patience_counter >= config.patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without producing a best model state.")

    model.load_state_dict(best_state_dict)
    return model, history, best_val_acc


# ============================================================
# Saving / Export
# ============================================================

def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    model: nn.Module,
    class_names: List[str],
    config: Config,
    test_acc: float,
    output_dir: Path,
) -> Path:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": config.image_size,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "test_accuracy": test_acc,
        "num_classes": len(class_names),
        "config": asdict(config),
    }

    checkpoint_path = output_dir / "trashnet_resnet18.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)
    return checkpoint_path


def save_metadata(
    class_names: List[str],
    config: Config,
    test_acc: float,
    confusion: List[List[int]],
    output_dir: Path,
) -> Path:
    metadata = {
        "class_names": class_names,
        "image_size": config.image_size,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "test_accuracy": test_acc,
        "confusion_matrix": confusion,
        "config": asdict(config),
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Saved metadata to %s", metadata_path)
    return metadata_path


def export_onnx_model(
    model: nn.Module,
    config: Config,
    output_dir: Path,
) -> Path:
    model.eval()
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size, device=config.device)
    onnx_path = output_dir / "trashnet_resnet18.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    logger.info("Exported ONNX model to %s", onnx_path)
    return onnx_path


# ============================================================
# Orchestrator
# ============================================================

def run_training(config: Config) -> None:
    """Run the whole training + evaluation + export pipeline."""
    setup_logging()

    logger.info("Using device: %s", config.device)
    logger.info("Config: %s", asdict(config))

    set_seed(config.seed)
    validate_dataset_dir(config.data_dir)
    output_dir = ensure_output_dir(config.output_dir)

    train_loader, val_loader, test_loader, class_names, num_classes, targets = build_dataloaders(config)
    class_weights = compute_class_weights(targets, num_classes, config.device)

    model = build_model(num_classes=num_classes, config=config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %d", total_params)
    logger.info("Trainable parameters: %d", trainable_params)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model, history, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
    )

    logger.info("Best validation accuracy: %.2f%%", best_val_acc * 100)

    preds, labels, _probs = predict(model, test_loader, config.device)
    test_acc = float((preds == labels).astype(np.float32).mean())

    logger.info("Test accuracy: %.2f%%", test_acc * 100)
    print("\nClassification Report:\n")
    print(classification_report(labels, preds, target_names=class_names, digits=3))

    cm = confusion_matrix(labels, preds)
    logger.info("Confusion matrix:\n%s", cm)

    save_checkpoint(model, class_names, config, test_acc, output_dir)
    save_metadata(class_names, config, test_acc, cm.tolist(), output_dir)

    if config.export_onnx:
        export_onnx_model(model, config, output_dir)

    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history to %s", history_path)

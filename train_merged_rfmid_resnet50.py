"""
Train ResNet50 on the merged RFMiD dataset with overfitting controls.

Dataset layout expected:
    data/merged_RFMID/<CLASS_NAME>/*.jpg

This script:
- Excludes UNLABELED by default
- Builds stratified train/val/test splits from folder labels
- Uses class-weighted CrossEntropy with label smoothing
- Uses AdamW + ReduceLROnPlateau + early stopping on val macro F1
- Saves best checkpoint with class mapping for explainability consumers
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# Paths and training configuration
DATA_ROOT = Path("data/merged_RFMID")
MODEL_STEM = "resnet50_merged_rfmid"
CURVES_STEM = "training_curves_merged_rfmid_resnet50"

EXCLUDED_CLASSES = {"UNLABELED"}
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_SIZE = 224
BATCH_SIZE_GPU = 32
BATCH_SIZE_CPU = 16
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.4
EARLY_STOPPING_PATIENCE = 6


@dataclass
class Sample:
    path: Path
    class_name: str


class ImageClassificationDataset(Dataset):
    def __init__(self, samples: list[Sample], class_to_idx: dict[str, int], transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        image = Image.open(item.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[item.class_name]
        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_samples() -> tuple[list[Sample], list[str]]:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATA_ROOT}")

    all_samples: list[Sample] = []
    classes: list[str] = []

    for class_dir in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()]):
        class_name = class_dir.name
        if class_name in EXCLUDED_CLASSES:
            continue

        images = sorted([p for p in class_dir.iterdir() if p.is_file() and is_image(p)])
        if not images:
            continue

        classes.append(class_name)
        for image_path in images:
            all_samples.append(Sample(path=image_path, class_name=class_name))

    if not all_samples:
        raise ValueError("No training images found in merged dataset.")

    return all_samples, classes


def stratified_split(samples: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(RANDOM_SEED)
    by_class: dict[str, list[Sample]] = {}
    for sample in samples:
        by_class.setdefault(sample.class_name, []).append(sample)

    train, val, test = [], [], []

    for class_name, class_samples in sorted(by_class.items()):
        rng.shuffle(class_samples)
        n = len(class_samples)

        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        n_test = n - n_train - n_val

        if n_test < 1:
            n_test = 1
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

        split_train = class_samples[:n_train]
        split_val = class_samples[n_train:n_train + n_val]
        split_test = class_samples[n_train + n_val:n_train + n_val + n_test]

        train.extend(split_train)
        val.extend(split_val)
        test.extend(split_test)

        print(
            f"{class_name:8} total={n:4d} train={len(split_train):4d} "
            f"val={len(split_val):4d} test={len(split_test):4d}"
        )

    return train, val, test


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf


def create_model(num_classes: int, use_pretrained: bool) -> nn.Module:
    weights = None
    if use_pretrained:
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        except Exception:
            weights = None

    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(in_features, num_classes),
    )
    return model


def make_class_weights(train_samples: list[Sample], class_to_idx: dict[str, int]) -> torch.Tensor:
    counts = Counter([s.class_name for s in train_samples])
    total = len(train_samples)
    num_classes = len(class_to_idx)

    weights = np.zeros(num_classes, dtype=np.float32)
    for class_name, idx in class_to_idx.items():
        c = counts.get(class_name, 1)
        weights[idx] = total / (num_classes * c)

    weights = np.clip(weights, 0.5, 8.0)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    preds_all: list[int] = []
    labels_all: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = float((np.array(preds_all) == np.array(labels_all)).mean())
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, labels_all, preds_all


def plot_curves(history: dict[str, list[float]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(history["train_f1"], label="Train")
    axes[2].plot(history["val_f1"], label="Val")
    axes[2].set_title("Macro F1")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str,
    normalize: bool,
) -> None:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        matrix = cm.astype(np.float32) / np.clip(row_sums, a_min=1, a_max=None)
    else:
        matrix = cm.astype(np.float32)

    fig_size = max(10, min(24, int(0.35 * len(class_names)) + 6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    preds_all: list[int] = []
    labels_all: list[int] = []
    probs_all: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            preds_all.extend(preds.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            probs_all.append(probs.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    y_true = np.array(labels_all, dtype=np.int64)
    y_pred = np.array(preds_all, dtype=np.int64)
    y_prob = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, 0), dtype=np.float32)

    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0
    return avg_loss, acc, macro_f1, y_true, y_pred, y_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet50 on merged RFMiD dataset")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size (0 keeps device default)")
    parser.add_argument("--workers", type=int, default=-1, help="DataLoader workers (-1 keeps device default)")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights if available")
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional suffix for outputs. If omitted, a timestamped suffix is generated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    run_name = args.run_name.strip().replace(" ", "_")
    if not run_name:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    model_path = Path(f"{MODEL_STEM}_{run_name}_model.pth")
    metrics_json = Path(f"{MODEL_STEM}_{run_name}_metrics.json")
    curves_path = Path(f"{CURVES_STEM}_{run_name}.png")
    confusion_path = Path(f"confusion_matrix_merged_rfmid_resnet50_{run_name}.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU
    if args.batch_size > 0:
        batch_size = args.batch_size

    print("=" * 90)
    print("ResNet50 training on merged RFMiD dataset")
    print(f"Device: {device}")
    print(f"Run name: {run_name}")
    print(f"Output model: {model_path}")
    print(f"Output metrics: {metrics_json}")
    print(f"Output curves: {curves_path}")
    print(f"Output confusion matrix: {confusion_path}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 90)

    samples, classes = load_samples()
    class_to_idx = {name: idx for idx, name in enumerate(sorted(classes))}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    print(f"Total classes used: {len(class_to_idx)}")
    print(f"Total images used: {len(samples)}")

    train_samples, val_samples, test_samples = stratified_split(samples)
    print(
        f"Split sizes: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )

    train_tf, eval_tf = build_transforms()

    train_ds = ImageClassificationDataset(train_samples, class_to_idx, transform=train_tf)
    val_ds = ImageClassificationDataset(val_samples, class_to_idx, transform=eval_tf)
    test_ds = ImageClassificationDataset(test_samples, class_to_idx, transform=eval_tf)

    num_workers = 0 if device.type == "cpu" else 4
    if args.workers >= 0:
        num_workers = args.workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(num_classes=len(class_to_idx), use_pretrained=args.pretrained).to(device)

    class_weights = make_class_weights(train_samples, class_to_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train_mode=True
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, train_mode=False
        )

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "image_size": IMAGE_SIZE,
                    "best_val_macro_f1": best_val_f1,
                    "best_epoch": best_epoch,
                    "run_name": run_name,
                    "pretrained": bool(args.pretrained),
                    "excluded_classes": sorted(EXCLUDED_CLASSES),
                },
                model_path,
            )
            print(f"  Saved best model to {model_path} (val_macro_f1={best_val_f1:.4f})")
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    plot_curves(history, curves_path)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    test_loss, test_acc, test_f1, y_true, y_pred, y_prob = evaluate_model(
        model, test_loader, criterion, device
    )
    class_names = [checkpoint["idx_to_class"][i] for i in range(len(checkpoint["idx_to_class"]))]
    labels_idx = list(range(len(class_names)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(np.float32) / np.clip(cm_row_sums, a_min=1, a_max=None)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    top3_acc = None
    if y_prob.size and y_prob.shape[1] >= 2:
        k = min(3, y_prob.shape[1])
        top3_acc = float(top_k_accuracy_score(y_true, y_prob, k=k, labels=labels_idx))

    roc_auc_ovr_macro = None
    if y_prob.size and y_prob.shape[1] >= 2:
        try:
            roc_auc_ovr_macro = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels_idx)
            )
        except ValueError:
            roc_auc_ovr_macro = None

    plot_confusion_matrix(
        cm,
        class_names,
        confusion_path,
        title="Confusion Matrix (Counts)",
        normalize=False,
    )

    summary = {
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_val_macro_f1": float(checkpoint.get("best_val_macro_f1", -1.0)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_macro_precision": float(macro_precision),
        "test_macro_recall": float(macro_recall),
        "test_weighted_precision": float(weighted_precision),
        "test_weighted_recall": float(weighted_recall),
        "test_weighted_f1": float(weighted_f1),
        "test_balanced_accuracy": float(balanced_acc),
        "test_top3_accuracy": float(top3_acc) if top3_acc is not None else None,
        "test_roc_auc_ovr_macro": roc_auc_ovr_macro,
        "num_classes": len(class_names),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "run_name": run_name,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "classification_report": report,
        "artifacts": {
            "curves": str(curves_path),
            "confusion_matrix": str(confusion_path),
        },
    }

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("Training complete")
    print(f"Best epoch: {summary['best_epoch']} | best val macro F1: {summary['best_val_macro_f1']:.4f}")
    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"Test macro F1: {summary['test_macro_f1']:.4f}")
    print(f"Test balanced accuracy: {summary['test_balanced_accuracy']:.4f}")
    if summary["test_top3_accuracy"] is not None:
        print(f"Test top-3 accuracy: {summary['test_top3_accuracy']:.4f}")
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_json}")
    print(f"Curves: {curves_path}")
    print(f"Confusion matrix: {confusion_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()

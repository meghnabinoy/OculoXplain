"""
RFMiD_2 Validation Pipeline for OculoXplain

Computes robust validation metrics for the trained MobileNetV2 rare-disease model,
and exports faculty-ready artifacts:
- overall_metrics.json
- per_class_metrics.csv
- predictions.csv
- confusion_matrix_primary.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


DISEASE_CLASSES = [
    'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'CME', 'CNV', 'CRAO',
    'CRS', 'CRVO', 'CSC', 'CSR', 'CWS', 'DN', 'DR', 'EDN', 'ERM', 'GRT',
    'HPED', 'HR', 'HTN', 'IIH', 'LS', 'MCA', 'ME', 'MH', 'MHL', 'MS',
    'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'OPDM', 'PRH', 'RD', 'RHL', 'RP',
    'RPEC', 'RS', 'RT', 'RTR', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 'VS', 'WNL'
]

COMMON_CLASSES = {'ARMD', 'DR', 'HTN', 'CME', 'ME', 'MH', 'MHL', 'ERM', 'CNV', 'WNL'}
RARE_CLASSES = [cls for cls in DISEASE_CLASSES if cls not in COMMON_CLASSES]


class RFMiD2Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        raw_path = Path(row['preprocessed_path'])
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(raw_path)
            candidates.append(self.image_root / raw_path)
            if str(raw_path).startswith("data/") or str(raw_path).startswith("data\\"):
                stripped = Path(str(raw_path).replace("data/", "", 1).replace("data\\", "", 1))
                candidates.append(self.image_root / stripped)
            candidates.append(Path.cwd() / raw_path)

        img_path = None
        for cand in candidates:
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            img_path = candidates[0]

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)

        labels = torch.tensor([row[c] for c in DISEASE_CLASSES], dtype=torch.float32)
        return img, labels, str(img_path)


class MobileNetRare(nn.Module):
    def __init__(self, num_classes: int = 51):
        super().__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)


@dataclass
class EvalOutputs:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    primary_true: np.ndarray
    primary_pred: np.ndarray
    image_paths: list[str]


def safe_metric(fn, default=np.nan, **kwargs):
    try:
        return fn(**kwargs)
    except Exception:
        return default


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_boot: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> EvalOutputs:
    model.eval()
    y_true, y_pred, y_score = [], [], []
    primary_true, primary_pred = [], []
    image_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_score.append(probs.cpu().numpy())

            primary_true.append(labels.argmax(dim=1).cpu().numpy())
            primary_pred.append(probs.argmax(dim=1).cpu().numpy())
            image_paths.extend(list(paths))

    return EvalOutputs(
        y_true=np.vstack(y_true),
        y_pred=np.vstack(y_pred),
        y_score=np.vstack(y_score),
        primary_true=np.concatenate(primary_true),
        primary_pred=np.concatenate(primary_pred),
        image_paths=image_paths,
    )


def top_k_accuracy(y_score: np.ndarray, primary_true: np.ndarray, k: int) -> float:
    topk = np.argpartition(-y_score, kth=min(k - 1, y_score.shape[1] - 1), axis=1)[:, :k]
    hit = np.array([primary_true[i] in topk[i] for i in range(len(primary_true))], dtype=np.float32)
    return float(hit.mean())


def build_metrics(outputs: EvalOutputs) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y_true = outputs.y_true.astype(int)
    y_pred = outputs.y_pred.astype(int)
    y_score = outputs.y_score

    overall = {
        "sample_count": int(y_true.shape[0]),
        "class_count": int(y_true.shape[1]),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_samples": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_micro": float(safe_metric(roc_auc_score, y_true=y_true, y_score=y_score, average="micro")),
        "roc_auc_macro": float(safe_metric(roc_auc_score, y_true=y_true, y_score=y_score, average="macro")),
        "pr_auc_micro": float(safe_metric(average_precision_score, y_true=y_true, y_score=y_score, average="micro")),
        "pr_auc_macro": float(safe_metric(average_precision_score, y_true=y_true, y_score=y_score, average="macro")),
    }

    # Label-wise balanced accuracy
    bal_scores = []
    for idx in range(y_true.shape[1]):
        if len(np.unique(y_true[:, idx])) < 2:
            continue
        bal_scores.append(balanced_accuracy_score(y_true[:, idx], y_pred[:, idx]))
    overall["balanced_accuracy_macro"] = float(np.mean(bal_scores)) if bal_scores else float("nan")

    # Primary-class metrics (single-label projection for confusion/top-k)
    primary_true = outputs.primary_true
    primary_pred = outputs.primary_pred
    overall["top1_primary_accuracy"] = float((primary_true == primary_pred).mean())
    overall["top3_primary_accuracy"] = top_k_accuracy(y_score, primary_true, k=3)
    overall["top5_primary_accuracy"] = top_k_accuracy(y_score, primary_true, k=5)
    overall["mcc_primary"] = float(matthews_corrcoef(primary_true, primary_pred))
    overall["kappa_primary"] = float(cohen_kappa_score(primary_true, primary_pred))

    ci_low, ci_high = bootstrap_ci(primary_true, primary_pred, lambda a, b: float((a == b).mean()))
    overall["top1_primary_accuracy_ci95"] = [ci_low, ci_high]

    # Rare vs common subgroup F1
    rare_idx = [DISEASE_CLASSES.index(c) for c in RARE_CLASSES]
    common_idx = [DISEASE_CLASSES.index(c) for c in COMMON_CLASSES]
    overall["f1_macro_rare_only"] = float(f1_score(y_true[:, rare_idx], y_pred[:, rare_idx], average="macro", zero_division=0))
    overall["f1_macro_common_only"] = float(f1_score(y_true[:, common_idx], y_pred[:, common_idx], average="macro", zero_division=0))

    # Per-class table
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    ap = []
    auc = []
    prevalence = y_true.mean(axis=0)
    for i in range(y_true.shape[1]):
        ap.append(safe_metric(average_precision_score, y_true=y_true[:, i], y_score=y_score[:, i]))
        if len(np.unique(y_true[:, i])) < 2:
            auc.append(np.nan)
        else:
            auc.append(safe_metric(roc_auc_score, y_true=y_true[:, i], y_score=y_score[:, i]))

    per_class = pd.DataFrame({
        "class_code": DISEASE_CLASSES,
        "is_rare": [c in RARE_CLASSES for c in DISEASE_CLASSES],
        "support_positive": sup.astype(int),
        "prevalence": prevalence,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "average_precision": ap,
        "roc_auc": auc,
    }).sort_values(["is_rare", "f1"], ascending=[False, False]).reset_index(drop=True)

    predictions = pd.DataFrame({
        "image_path": outputs.image_paths,
        "true_primary": [DISEASE_CLASSES[i] for i in outputs.primary_true],
        "pred_primary": [DISEASE_CLASSES[i] for i in outputs.primary_pred],
        "primary_correct": outputs.primary_true == outputs.primary_pred,
    })

    return overall, per_class, predictions


def save_confusion_primary(outputs: EvalOutputs, out_path: Path):
    cm = confusion_matrix(outputs.primary_true, outputs.primary_pred, labels=np.arange(len(DISEASE_CLASSES)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.title("Primary-Class Confusion Matrix (Normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(DISEASE_CLASSES))
    plt.xticks(ticks, DISEASE_CLASSES, rotation=90, fontsize=7)
    plt.yticks(ticks, DISEASE_CLASSES, fontsize=7)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate RFMiD_2 MobileNetV2 model with comprehensive metrics.")
    parser.add_argument("--metadata", type=str, default="data/RFMiD_2/rfmid2_preprocessed_metadata.csv")
    parser.add_argument("--model", type=str, default="mobilenet_rfmid2_quick_model.pth")
    parser.add_argument("--image-root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--out-dir", type=str, default="outputs/validation_rfmid2")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    model_path = Path(args.model)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = pd.read_csv(metadata_path)
    split = args.split
    if split not in set(df.get("split", [])):
        if "val" in set(df.get("split", [])):
            split = "val"
        else:
            raise ValueError("Requested split not found and no 'val' split exists in metadata.")

    eval_df = df[df["split"] == split].reset_index(drop=True)
    if args.max_samples and args.max_samples > 0:
        eval_df = eval_df.sample(min(args.max_samples, len(eval_df)), random_state=42).reset_index(drop=True)

    def _resolve_candidate(path_str: str) -> Path | None:
        raw_path = Path(path_str)
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(raw_path)
            candidates.append(image_root / raw_path)
            if str(raw_path).startswith("data/") or str(raw_path).startswith("data\\"):
                stripped = Path(str(raw_path).replace("data/", "", 1).replace("data\\", "", 1))
                candidates.append(image_root / stripped)
            candidates.append(Path.cwd() / raw_path)
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    exists_mask = eval_df["preprocessed_path"].apply(lambda p: _resolve_candidate(str(p)) is not None)
    missing_count = int((~exists_mask).sum())
    eval_df = eval_df[exists_mask].reset_index(drop=True)
    if missing_count:
        print(f"Warning: skipped {missing_count} samples with missing image files")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = RFMiD2Dataset(eval_df, image_root=image_root, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetRare(num_classes=len(DISEASE_CLASSES))
    state = torch.load(str(model_path), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)

    outputs = evaluate(model, loader, device=device, threshold=args.threshold)
    overall, per_class, predictions = build_metrics(outputs)

    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    save_confusion_primary(outputs, out_dir / "confusion_matrix_primary.png")

    print("=" * 80)
    print("Validation complete")
    print(f"Split used: {split}")
    print(f"Samples: {overall['sample_count']}")
    print(f"Top-1 Primary Accuracy: {overall['top1_primary_accuracy']:.4f}")
    print(f"Top-3 Primary Accuracy: {overall['top3_primary_accuracy']:.4f}")
    print(f"Top-5 Primary Accuracy: {overall['top5_primary_accuracy']:.4f}")
    print(f"F1 Macro: {overall['f1_macro']:.4f}")
    print(f"F1 Micro: {overall['f1_micro']:.4f}")
    print(f"PR-AUC Macro: {overall['pr_auc_macro']:.4f}")
    print(f"Outputs saved in: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

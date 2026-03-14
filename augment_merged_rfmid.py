"""
Augment merged RFMiD dataset to reduce class imbalance with conservative transforms.

Policy goals:
- Improve minority class representation
- Avoid aggressive transformations that can cause overfitting artifacts
- Exclude UNLABELED class by default

Usage:
    python augment_merged_rfmid.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path("data/merged_RFMID")
METADATA_PATH = ROOT / "merged_dataset_metadata.csv"
REPORT_PATH = ROOT / "augmentation_report.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXCLUDED_CLASSES = {"UNLABELED"}

# Conservative balancing policy
TARGET_PER_CLASS = 72
DEFAULT_MAX_AUG_PER_ORIGINAL = 3
RANDOM_SEED = 42


@dataclass
class ClassPlan:
    class_name: str
    original_count: int
    target_count: int
    to_add: int
    added: int = 0


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _list_class_images(class_dir: Path) -> list[Path]:
    return sorted([p for p in class_dir.iterdir() if _is_image(p)])


def _augment_once(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    angle = rng.uniform(-12.0, 12.0)
    scale = rng.uniform(0.97, 1.03)
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    out = cv2.warpAffine(
        out,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    tx = int(rng.uniform(-0.03, 0.03) * width)
    ty = int(rng.uniform(-0.03, 0.03) * height)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    out = cv2.warpAffine(
        out,
        translation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    alpha = rng.uniform(0.9, 1.1)
    beta = rng.uniform(-12, 12)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.15:
        sigma = rng.uniform(0.1, 0.6)
        out = cv2.GaussianBlur(out, ksize=(3, 3), sigmaX=sigma)

    return out


def _build_class_plans(class_counts: dict[str, int]) -> list[ClassPlan]:
    plans: list[ClassPlan] = []
    for class_name, count in sorted(class_counts.items()):
        if class_name in EXCLUDED_CLASSES:
            plans.append(ClassPlan(class_name, count, count, 0))
            continue

        target = max(count, TARGET_PER_CLASS)
        to_add = target - count
        plans.append(ClassPlan(class_name, count, target, to_add))

    return plans


def _max_aug_per_original_for_class(original_count: int) -> int:
    """
    Adaptive cap to let tiny classes catch up while keeping larger classes conservative.
    """
    if original_count <= 12:
        return 8
    if original_count <= 24:
        return 6
    if original_count <= 36:
        return 4
    return DEFAULT_MAX_AUG_PER_ORIGINAL


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    if not ROOT.exists():
        raise FileNotFoundError(f"Merged dataset directory not found: {ROOT}")

    metadata = pd.read_csv(METADATA_PATH) if METADATA_PATH.exists() else pd.DataFrame()

    class_dirs = [d for d in ROOT.iterdir() if d.is_dir()]
    class_counts: dict[str, int] = {}
    class_images: dict[str, list[Path]] = {}

    for class_dir in class_dirs:
        images = _list_class_images(class_dir)
        if not images:
            continue
        class_counts[class_dir.name] = len(images)
        class_images[class_dir.name] = images

    plans = _build_class_plans(class_counts)

    # Existing augmented counters per original image stem
    existing_counts: dict[str, int] = {}
    for class_name, images in class_images.items():
        for img_path in images:
            stem = img_path.stem
            if "_augm" in stem:
                base = stem.split("_augm")[0]
                existing_counts[f"{class_name}/{base}"] = existing_counts.get(f"{class_name}/{base}", 0) + 1

    new_rows = []

    for plan in plans:
        if plan.to_add <= 0:
            continue

        candidates = [p for p in class_images[plan.class_name] if "_augm" not in p.stem]
        if not candidates:
            continue

        parent_lookup = {}
        if not metadata.empty and "image_file" in metadata.columns and "disease" in metadata.columns:
            subset = metadata[metadata["disease"].astype(str) == plan.class_name].copy()
            if not subset.empty:
                subset["_key"] = subset["image_file"].astype(str)
                parent_lookup = subset.set_index("_key").to_dict("index")

        attempts = 0
        max_attempts = plan.to_add * 20

        while plan.added < plan.to_add and attempts < max_attempts:
            attempts += 1
            source_path = rng.choice(candidates)
            base_stem = source_path.stem
            base_key = f"{plan.class_name}/{base_stem}"
            used = existing_counts.get(base_key, 0)
            per_class_cap = _max_aug_per_original_for_class(plan.original_count)
            if used >= per_class_cap:
                continue

            image = cv2.imread(str(source_path))
            if image is None:
                continue

            aug_img = _augment_once(image, rng)

            next_idx = used + 1
            new_name = f"{base_stem}_augm{next_idx}.jpg"
            out_path = ROOT / plan.class_name / new_name

            while out_path.exists():
                next_idx += 1
                new_name = f"{base_stem}_augm{next_idx}.jpg"
                out_path = ROOT / plan.class_name / new_name

            ok = cv2.imwrite(str(out_path), aug_img)
            if not ok:
                continue

            existing_counts[base_key] = next_idx
            plan.added += 1

            if parent_lookup:
                parent_row = parent_lookup.get(source_path.name)
            else:
                parent_row = None

            row = {
                "disease": plan.class_name,
                "image_file": new_name,
                "source": "augmented",
                "original_file": source_path.name,
                "original_stem": source_path.stem,
                "split": parent_row.get("split") if parent_row else np.nan,
            }
            new_rows.append(row)

    if new_rows:
        add_df = pd.DataFrame(new_rows)
        if metadata.empty:
            metadata = add_df
        else:
            for col in add_df.columns:
                if col not in metadata.columns:
                    metadata[col] = np.nan
            for col in metadata.columns:
                if col not in add_df.columns:
                    add_df[col] = np.nan
            add_df = add_df[metadata.columns]
            metadata = pd.concat([metadata, add_df], ignore_index=True)

        metadata.to_csv(METADATA_PATH, index=False)

    report_rows = [
        {
            "class_name": p.class_name,
            "original_count": p.original_count,
            "target_count": p.target_count,
            "planned_to_add": p.to_add,
            "added": p.added,
            "final_count": p.original_count + p.added,
        }
        for p in plans
    ]
    report_df = pd.DataFrame(report_rows).sort_values("class_name").reset_index(drop=True)
    report_df.to_csv(REPORT_PATH, index=False)

    total_added = int(report_df["added"].sum())
    touched_classes = int((report_df["added"] > 0).sum())

    print("=" * 80)
    print("Merged RFMiD augmentation complete")
    print(f"Target per class: {TARGET_PER_CLASS}")
    print(
        "Adaptive max augmented copies per original: "
        f"<=12->8, <=24->6, <=36->4, otherwise {DEFAULT_MAX_AUG_PER_ORIGINAL}"
    )
    print(f"Classes augmented: {touched_classes}")
    print(f"Total images added: {total_added}")
    print(f"Updated metadata: {METADATA_PATH}")
    print(f"Report: {REPORT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()

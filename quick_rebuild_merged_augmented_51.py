from pathlib import Path
import shutil
import random
import cv2
import numpy as np

INPUT_DIR = Path("data/merged_RFMID")
OUTPUT_DIR = Path("data/merged_RFMID_augmented")
TARGET_PER_CLASS = 120
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED = 42

rng = random.Random(SEED)
np.random.seed(SEED)


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]


def augment_image(img_bgr: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.2:
        out = cv2.flip(out, 0)
    angle = rng.choice([0, 90, 180, 270])
    if angle == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif angle == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rng.random() < 0.7:
        out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.9, 1.15), beta=rng.uniform(-15, 15))
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.3:
        noise = np.random.normal(0, rng.uniform(4, 10), out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_DIR}")

    class_dirs = sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()])
    if len(class_dirs) < 2:
        raise RuntimeError("Need at least 2 class directories")

    build_dir = OUTPUT_DIR.parent / "merged_RFMID_augmented_build_fast51"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for class_dir in class_dirs:
        class_name = class_dir.name
        src_images = list_images(class_dir)
        dst_dir = build_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src in src_images:
            shutil.copy2(src, dst_dir / src.name)

        original_count = len(src_images)
        needed = max(0, TARGET_PER_CLASS - original_count)
        added = 0

        if original_count > 0 and needed > 0:
            for i in range(needed):
                src = src_images[i % original_count]
                img = cv2.imread(str(src))
                if img is None:
                    continue
                aug = augment_image(img)
                out_path = dst_dir / f"aug_fast_{i}_{src.stem}.jpg"
                if cv2.imwrite(str(out_path), aug):
                    added += 1

        class_total = original_count + added
        total_files += class_total
        print(f"{class_name}: original={original_count}, added={added}, final={class_total}")

    backup_dir = OUTPUT_DIR.parent / "merged_RFMID_augmented_prev_fast51"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if OUTPUT_DIR.exists():
        OUTPUT_DIR.rename(backup_dir)

    build_dir.rename(OUTPUT_DIR)

    final_classes = len([p for p in OUTPUT_DIR.iterdir() if p.is_dir()])
    print("=" * 70)
    print(f"DONE: classes={final_classes}, total_files={total_files}, output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()

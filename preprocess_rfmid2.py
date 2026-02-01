import os
from pathlib import Path
import shutil
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data") / "RFMiD_2"
OUT_DIR = DATA_DIR / "preprocessed"
CSV_PATH = DATA_DIR / "rfmid2_preprocessed_metadata.csv"


def is_image_file(fname):
    return fname.suffix.lower() in {'.jpg', '.jpeg', '.png'}


def crop_black_border(img, tol=10):
    # Convert to grayscale and find bbox of non-black pixels
    gray = img.convert('L')
    arr = np.array(gray)
    mask = arr > tol
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img.crop((x0, y0, x1, y1))
    return img


def process_image(src_path, dst_path, size=(224, 224)):
    try:
        img = Image.open(src_path).convert('RGB')
        img = crop_black_border(img)
        img = ImageOps.fit(img, size, Image.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, format='JPEG', quality=90)
        return True
    except Exception as e:
        print(f"Failed to process {src_path}: {e}")
        return False


def gather_paths(data_dir=DATA_DIR):
    rows = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        class_name = child.name
        # skip preprocessed folder
        if class_name.lower() == 'preprocessed':
            continue
        for img_path in child.iterdir():
            if img_path.is_file() and is_image_file(img_path):
                label = 0 if class_name.upper() == 'WNL' else 1
                rows.append({'original_path': str(img_path), 'class_name': class_name, 'label': label})
    df = pd.DataFrame(rows)
    return df


def main():
    if not DATA_DIR.exists():
        print(f"RFMiD_2 folder not found at {DATA_DIR}. Aborting.")
        return

    df = gather_paths()
    if df.empty:
        print("No images found under RFMiD_2. Nothing to do.")
        return

    print(f"Found {len(df)} images across {df['class_name'].nunique()} classes")

    # Stratified split by label (WNL vs others)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    records = []

    def save_rows(sub_df, split_name):
        for i, row in sub_df.iterrows():
            orig = Path(row['original_path'])
            cls = row['class_name']
            label = int(row['label'])
            bucket = 'healthy' if label == 0 else 'disease'
            dst = OUT_DIR / split_name / bucket / f"{cls}__{orig.name}"
            ok = process_image(orig, dst)
            if ok:
                records.append({
                    'preprocessed_path': str(dst.as_posix()),
                    'original_path': str(orig.as_posix()),
                    'class_name': cls,
                    'label': label,
                    'split': split_name
                })

    print("Processing training images...")
    save_rows(train_df, 'train')
    print("Processing validation images...")
    save_rows(val_df, 'val')

    out_df = pd.DataFrame(records)
    out_df.to_csv(CSV_PATH, index=False)
    print(f"Wrote metadata CSV to {CSV_PATH} with {len(out_df)} rows")
    print("Preprocessing complete.")


if __name__ == '__main__':
    main()

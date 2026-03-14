import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np
import argparse
from typing import Optional


# --- CONFIG ---
BASE_DIR = r"D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\ODIR-5K"
DATA_FILE = os.path.join(BASE_DIR, "data.xlsx")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "Training Images")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
TARGET_SIZE = (224, 224)

# --- FUNCTIONS ---
def get_label(row):
    """0 = healthy (N=1), 1 = disease (N=0)."""
    return 0 if row["N"] == 1 else 1

def find_image_file(images_dir: str, img_filename: Optional[str]) -> Optional[str]:
    """Try to resolve an image filename to an existing file in images_dir.
    Accepts exact filenames or basename without extension; tries common extensions.
    Returns full path or None if not found.
    """
    if img_filename is None or (isinstance(img_filename, float) and np.isnan(img_filename)):
        return None
    name = str(img_filename).strip()
    if not name:
        return None

    # If it's already an absolute or relative path, check directly
    candidate = os.path.join(images_dir, name)
    if os.path.isfile(candidate):
        return candidate

    # Try as-is in images_dir
    if os.path.isfile(os.path.join(images_dir, os.path.basename(name))):
        return os.path.join(images_dir, os.path.basename(name))

    # Try common image extensions
    base, ext = os.path.splitext(name)
    extensions = [ext] if ext else []
    extensions += [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    for e in extensions:
        candidate = os.path.join(images_dir, base + e)
        if os.path.isfile(candidate):
            return candidate

    # Fallback: search for files that start with the base name (handles suffixes)
    try:
        for f in os.listdir(images_dir):
            if f.lower().startswith(base.lower()):
                full = os.path.join(images_dir, f)
                if os.path.isfile(full):
                    return full
    except Exception:
        pass

    return None

def preprocess_image(src_path, dst_path):
    """Resize, remove borders, normalize, and save."""
    if not os.path.exists(src_path):
        print(f"Warning: Image not found: {src_path}")
        return False

    try:
        # Open image
        img = Image.open(src_path).convert("RGB")

        # Convert to numpy for black border detection
        img_np = np.array(img)
        gray = np.mean(img_np, axis=2)
        mask = gray > 10  # Threshold for non-black pixels

        # Crop black borders
        if mask.any():
            coords = np.argwhere(mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            img_np = img_np[y_min:y_max, x_min:x_max]

        # Convert back to PIL
        img = Image.fromarray(img_np)

        # Resize to 224x224
        img = ImageOps.fit(img, TARGET_SIZE, method=Image.Resampling.LANCZOS)

        # Normalize pixel values (0–1) - but save as uint8 for storage
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img = Image.fromarray((img_np * 255).astype(np.uint8))  # Save as uint8 for consistency

        # Save
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False
    
def count_files_in_dir(dir_path):
    if not os.path.exists(dir_path):
        return 0
    return sum(1 for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)))


def main():
    print("Starting ODIR-5K preprocessing...")
    print(f"Data file: {DATA_FILE}")
    print(f"Images directory: {TRAIN_IMG_DIR}")
    print(f"Output directory: {PROCESSED_DIR}")
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        return
    
    # Check if images directory exists
    if not os.path.exists(TRAIN_IMG_DIR):
        print(f"ERROR: Images directory not found: {TRAIN_IMG_DIR}")
        return
    
    # --- LOAD DATA ---
    print("Loading data from Excel file...")
    df = pd.read_excel(DATA_FILE)
    print(f"Loaded {len(df)} records from Excel file")
    print(f"Columns: {list(df.columns)}")

    # --- CREATE BINARY LABELS ---
    df["label"] = df.apply(get_label, axis=1)
    print(f"Label distribution: Healthy: {(df['label'] == 0).sum()}, Disease: {(df['label'] == 1).sum()}")

    # Collect left & right images as separate samples
    samples = []
    missing_files = []
    
    for idx, row in df.iterrows():
        for side in ["Left-Fundus", "Right-Fundus"]:
            if side in df.columns:
                img_filename = row[side]
                img_path = os.path.join(TRAIN_IMG_DIR, img_filename)
                
                if os.path.exists(img_path):
                    samples.append({
                        "image": img_path, 
                        "label": row["label"],
                        "filename": img_filename,
                        "side": side
                    })
                else:
                    missing_files.append(img_filename)

    print(f"Found {len(samples)} valid image samples")
    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found")

    if len(samples) == 0:
        print("ERROR: No valid samples found! Check your data structure.")
        return

    df_samples = pd.DataFrame(samples)

    # --- SPLIT TRAIN/VAL ---
    print("Splitting into train/validation sets...")
    train_df, val_df = train_test_split(
        df_samples, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_samples["label"]
    )

    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")
    print(f"Train - Healthy: {(train_df['label'] == 0).sum()}, Disease: {(train_df['label'] == 1).sum()}")
    print(f"Val - Healthy: {(val_df['label'] == 0).sum()}, Disease: {(val_df['label'] == 1).sum()}")

    # --- SAVE TO PROCESSED FOLDER WITH PREPROCESSING ---
    print("Starting image preprocessing and saving...")
    processed_count = 0
    
    for subset_name, subset_df in [("train", train_df), ("val", val_df)]:
        print(f"Processing {subset_name} set...")
        for idx, row in subset_df.iterrows():
            label_folder = "disease" if row["label"] == 1 else "healthy"
            dst_path = os.path.join(PROCESSED_DIR, subset_name, label_folder, row["filename"])
            
            if preprocess_image(row["image"], dst_path):
                processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")

    print(f"Preprocessing complete! Processed {processed_count} images total")
    print(f"Images saved to: {PROCESSED_DIR}")
    
    # Print final directory structure
    print("\nFinal directory structure:")
    for subset in ["train", "val"]:
        for label in ["healthy", "disease"]:
            dir_path = os.path.join(PROCESSED_DIR, subset, label)
            if os.path.exists(dir_path):
                count = len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
                print(f"  {subset}/{label}: {count} images")

if __name__ == "__main__":
    main()

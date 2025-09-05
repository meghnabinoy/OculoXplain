import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np

# --- CONFIG ---
BASE_DIR = r"C:\Users\verth\Documents\projects\OculoXplain\data\ODIR-5K\ODIR-5K\ODIR-5K"
DATA_FILE = os.path.join(BASE_DIR, "data.xlsx")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "Training Images")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
TARGET_SIZE = (224, 224)

# --- FUNCTIONS ---
def get_label(row):
    """0 = healthy (N=1), 1 = disease (N=0)."""
    return 0 if row["N"] == 1 else 1

def preprocess_image(src_path, dst_path):
    """Resize, remove borders, normalize, and save."""
    if not os.path.exists(src_path):
        return False

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

    # Normalize pixel values (0–1)
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    img = Image.fromarray((img_np * 255).astype(np.uint8))  # Save as uint8 for consistency

    # Save
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img.save(dst_path)
    return True

# --- LOAD DATA ---
df = pd.read_excel(DATA_FILE)

# --- CREATE BINARY LABELS ---
df["label"] = df.apply(get_label, axis=1)

# Collect left & right images as separate samples
samples = []
for _, row in df.iterrows():
    for side in ["Left-Fundus", "Right-Fundus"]:
        img_path = os.path.join(TRAIN_IMG_DIR, row[side])
        if os.path.exists(img_path):
            samples.append({"image": img_path, "label": row["label"]})

df_samples = pd.DataFrame(samples)

# --- SPLIT TRAIN/VAL ---
train_df, val_df = train_test_split(df_samples, test_size=0.2, random_state=42, stratify=df_samples["label"])

# --- SAVE TO PROCESSED FOLDER WITH PREPROCESSING ---
for subset_name, subset_df in [("train", train_df), ("val", val_df)]:
    for _, row in subset_df.iterrows():
        label_folder = "disease" if row["label"] == 1 else "healthy"
        dst_path = os.path.join(PROCESSED_DIR, subset_name, label_folder, os.path.basename(row["image"]))
        preprocess_image(row["image"], dst_path)

print("Preprocessing complete with resizing, border removal, and normalization!")
print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

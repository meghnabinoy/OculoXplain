# augment_rare_classes_odir5k.py
"""
Augment rare classes in ODIR-5K based on train-val-test split.
- Only augments rare classes (not common ones like diabetic retinopathy)
- Applies random rotation, flip, and brightness changes
- Saves augmented images in 'data/ODIR-5K/augmented/'
- Updates CSV with new augmented samples
"""
import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm


from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, VerticalFlip, Transpose, RandomBrightnessContrast
)

# Paths
CSV_PATH = os.path.join('data', 'ODIR-5K', 'full_df_with_split.csv')
IMG_DIR = os.path.join('data', 'ODIR-5K', 'ODIR-5K', 'ODIR-5K', 'Training Images')
AUG_DIR = os.path.join('data', 'ODIR-5K', 'augmented')
os.makedirs(AUG_DIR, exist_ok=True)

# Load data
labels = pd.read_csv(CSV_PATH)

# Use the 'labels' column (e.g., ['N'], ['D'], etc.)
labels['main_label'] = labels['labels'].str.extract(r"\['([A-Z])'\]")


# Count per class
class_counts = labels['main_label'].value_counts()


# Define common diseases to exclude from augmentation (e.g., 'N' for Normal, 'D' for Diabetic Retinopathy)
common_classes = ['N', 'D']


# Set a target number: at least 1500 for all rare classes (except 'N' and 'D')
target = 1500

# List of rare classes (not in common_classes)
rare_classes = [k for k in class_counts.index if k not in common_classes]

# Augmentation pipeline
aug = Compose([
    RandomRotate90(),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Transpose(p=0.5),
    RandomBrightnessContrast(p=0.5),
])

augmented_rows = []

for cls in rare_classes:
    cls_df = labels[(labels['main_label'] == cls) & (labels['split'] == 'train')]
    n_to_add = target - len(cls_df)
    if n_to_add <= 0:
        continue
    for i in tqdm(range(n_to_add), desc=f'Augmenting {cls}'):
        row = cls_df.sample(1).iloc[0]
        img_path = os.path.join(IMG_DIR, row['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = aug(image=img)['image']
        # Save new image
        new_filename = f"aug_{cls}_{i}_{row['filename']}"
        save_path = os.path.join(AUG_DIR, new_filename)
        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        # Add new row to CSV
        new_row = row.copy()
        new_row['filename'] = new_filename
        new_row['augmented'] = True
        augmented_rows.append(new_row)

# Save updated CSV
if augmented_rows:
    aug_df = pd.DataFrame(augmented_rows)
    out_csv = os.path.join('data', 'ODIR-5K', 'full_df_with_split_augmented.csv')
    pd.concat([labels, aug_df], ignore_index=True).to_csv(out_csv, index=False)
    print(f"Augmented data saved to {out_csv}")
else:
    print("No augmentation performed (all classes above target)")

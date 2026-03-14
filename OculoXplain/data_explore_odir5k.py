# data_explore_odir5k.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# Set the correct paths based on your folder structure
LABELS_CSV = r'd:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\full_df.csv'
IMAGES_DIR = r'd:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\ODIR-5K\Training Images'

# Load labels
labels = pd.read_csv(LABELS_CSV)

# Show 10 sample images with labels
plt.figure(figsize=(20, 8))
sample = labels.sample(10, random_state=42)
for idx, row in enumerate(sample.itertuples()):
    img_name = getattr(row, 'filename', None)
    if img_name is None:
        plt.subplot(2, 5, idx+1)
        plt.text(0.5, 0.5, 'No filename column', ha='center', va='center')
        plt.axis('off')
        continue
    img_path = os.path.join(IMAGES_DIR, str(img_name))
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 5, idx+1)
        # Show diagnostic keywords and label
        label_str = f"Keywords: {getattr(row, 'Right-Diagnostic Keywords', '')}\nLabel: {getattr(row, 'labels', '')}"
        plt.imshow(img)
        plt.title(label_str, fontsize=10)
        plt.axis('off')
    else:
        plt.subplot(2, 5, idx+1)
        plt.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        plt.axis('off')
plt.tight_layout()
plt.show()

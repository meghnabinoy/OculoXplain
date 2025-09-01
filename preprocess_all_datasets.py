
import os
import cv2
import numpy as np

BASE_DIR = r'D:\OCULOXPLAIN\OculoXplain\data'
DATASETS = [
    'Ocular_Disease_Dataset',
    'ODIR-5K',
    'Retinal Disease Classification',
    'RFMiD_2'
]

def is_image_file(filename):
    ext = filename.lower().split('.')[-1]
    return ext in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']

def remove_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    return img

for dataset in DATASETS:
    dataset_dir = os.path.join(BASE_DIR, dataset)
    processed_dir = os.path.join(dataset_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    print(f"\nProcessing dataset: {dataset}")
    total_images = 0
    processed_images = 0
    for root, dirs, files in os.walk(dataset_dir):
        # Skip the processed folder itself
        if processed_dir in root:
            continue
        print(f"  Scanning folder: {root}")
        for fname in files:
            if not is_image_file(fname):
                continue
            total_images += 1
            fpath = os.path.join(root, fname)
            img = cv2.imread(fpath)
            if img is None:
                print(f"    Could not read image: {fpath}")
                continue
            img = remove_black_border(img)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            save_path = os.path.join(processed_dir, fname)
            cv2.imwrite(save_path, (img * 255).astype(np.uint8))
            processed_images += 1
            print(f"    Processed: {fname}")
    print(f"Summary for {dataset}: Found {total_images} images, processed {processed_images} images.\n")
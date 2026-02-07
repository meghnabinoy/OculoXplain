import os
import cv2
import pandas as pd

CSV_PATH = r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\full_df_with_split_augmented.csv'
IMG_DIR = r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\ODIR-5K\Training Images'
AUG_DIR = r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\augmented'
CLEANED_CSV_PATH = r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\full_df_with_split_augmented_cleaned.csv'


def image_exists_and_valid(img_path):
    if not os.path.exists(img_path):
        return False
    img = cv2.imread(img_path)
    return img is not None

def check_row(row, img_dir, aug_dir):
    filename = row['filename']
    is_aug = row.get('augmented', False)
    if is_aug:
        img_path = os.path.join(aug_dir, filename)
    else:
        img_path = os.path.join(img_dir, filename)
    if image_exists_and_valid(img_path):
        return (row, None, None)
    else:
        if not os.path.exists(img_path):
            return (None, img_path, None)
        else:
            return (None, None, img_path)


from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    df = pd.read_csv(CSV_PATH)
    valid_rows = []
    missing_files = []
    corrupted_files = []

    # Use ThreadPoolExecutor for parallel checking
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(check_row, row, IMG_DIR, AUG_DIR) for idx, row in df.iterrows()]
        for future in as_completed(futures):
            row, missing, corrupted = future.result()
            if row is not None:
                valid_rows.append(row)
            if missing is not None:
                missing_files.append(missing)
            if corrupted is not None:
                corrupted_files.append(corrupted)

    cleaned_df = pd.DataFrame(valid_rows)
    cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"Cleaned CSV saved to {CLEANED_CSV_PATH}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    if missing_files:
        print("Missing files list:")
        for f in missing_files:
            print(f)
    if corrupted_files:
        print("Corrupted files list:")
        for f in corrupted_files:
            print(f)

if __name__ == "__main__":
    main()

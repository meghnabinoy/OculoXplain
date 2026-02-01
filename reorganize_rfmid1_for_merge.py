"""
Reorganize RFMID_1 dataset to match RFMiD_2 structure for easier merging.

This script:
1. Creates a new folder 'mergeable_rfmid_1' with disease-based subfolders
2. Parses label CSVs from Training_set, Validation_set, Test_set
3. Distributes multi-label images to all applicable disease folders
4. Logs the reorganization process and any missing images
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict
import csv

# Paths
RFMID1_ROOT = Path("data/RFMID_1")
MERGEABLE_ROOT = RFMID1_ROOT / "mergeable_rfmid_1"
SPLITS = ["Training_set", "Validation_set", "Test_set"]

# Create mergeable root folder
MERGEABLE_ROOT.mkdir(exist_ok=True)
print(f"✓ Created root folder: {MERGEABLE_ROOT}")

# Track statistics
stats = {
    "total_labels": 0,
    "images_copied": 0,
    "images_missing": 0,
    "unique_images": set(),
    "disease_counts": defaultdict(int),
    "split_info": defaultdict(lambda: defaultdict(int)),
    "multi_label_images": 0,
}

# Track split assignment for each image
image_split_map = {}  # image_id -> split
image_disease_map = {}  # image_id -> list of diseases

# Process all three label files
print("\n" + "="*60)
print("PARSING LABEL FILES")
print("="*60)

for split in SPLITS:
    split_path = RFMID1_ROOT / split
    csv_file = None
    
    # Find the CSV file in this split folder
    for file in split_path.glob("*.csv"):
        csv_file = file
        break
    
    if not csv_file:
        print(f"✗ No CSV found in {split}")
        continue
    
    print(f"\nProcessing: {csv_file.name}")
    
    # Read the CSV with different encoding options
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='iso-8859-1')
    
    # Get disease columns, excluding 'ID' and any unnamed columns
    disease_columns = [col for col in df.columns[1:] if not col.startswith('Unnamed')]
    
    print(f"  Diseases found: {len(disease_columns)}")
    print(f"  Images in CSV: {len(df)}")
    
    # Process each image in this split
    for idx, row in df.iterrows():
        image_id = int(row['ID'])
        
        # Find which diseases this image has (value = 1)
        diseases = [col for col in disease_columns if row[col] == 1]
        
        if diseases:
            image_split_map[image_id] = split.replace("_set", "").lower()
            image_disease_map[image_id] = diseases
            stats["total_labels"] += 1
            stats["split_info"][split][len(diseases)] += 1
            
            if len(diseases) > 1:
                stats["multi_label_images"] += 1
            
            for disease in diseases:
                stats["disease_counts"][disease] += 1

print(f"\nTotal image-disease mappings: {stats['total_labels']}")
print(f"Multi-label images (multiple diseases): {stats['multi_label_images']}")

# Create disease folders and copy images
print("\n" + "="*60)
print("CREATING DISEASE FOLDERS AND COPYING IMAGES")
print("="*60)

all_diseases = sorted(set([disease for diseases in image_disease_map.values() for disease in diseases]))
print(f"\nTotal unique disease classes: {len(all_diseases)}")

# Create all disease folders
for disease in all_diseases:
    disease_folder = MERGEABLE_ROOT / disease
    disease_folder.mkdir(exist_ok=True)

print(f"✓ Created {len(all_diseases)} disease folders")

# Copy images to appropriate disease folders
print("\nCopying images to disease folders...")
missing_images = []

for image_id, diseases in image_disease_map.items():
    # Find the image file in the appropriate split folder
    image_found = False
    
    for split in SPLITS:
        split_folder = RFMID1_ROOT / split
        
        # Try both .jpg and .JPG extensions
        for ext in ['.jpg', '.JPG']:
            image_path = split_folder / f"{image_id}{ext}"
            
            if image_path.exists():
                image_found = True
                stats["unique_images"].add(image_id)
                
                # Copy to each disease folder
                for disease in diseases:
                    disease_folder = MERGEABLE_ROOT / disease
                    dest_path = disease_folder / f"{image_id}{ext}"
                    
                    # Copy file
                    shutil.copy2(image_path, dest_path)
                    stats["images_copied"] += 1
                
                break
        
        if image_found:
            break
    
    if not image_found:
        missing_images.append(image_id)
        stats["images_missing"] += 1

print(f"✓ Copied {stats['images_copied']} image-disease pairs")
print(f"✓ Unique images processed: {len(stats['unique_images'])}")
print(f"✗ Missing images: {stats['images_missing']}")

if missing_images:
    print(f"\nMissing image IDs: {sorted(missing_images)[:20]}{'...' if len(missing_images) > 20 else ''}")

# Create metadata CSV tracking image assignments
print("\n" + "="*60)
print("CREATING METADATA CSV")
print("="*60)

metadata_path = MERGEABLE_ROOT / "rfmid1_image_metadata.csv"
with open(metadata_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'original_split', 'disease_count', 'diseases'])
    
    for image_id in sorted(image_split_map.keys()):
        split = image_split_map[image_id]
        diseases = image_disease_map[image_id]
        writer.writerow([image_id, split, len(diseases), '|'.join(diseases)])

print(f"✓ Created metadata CSV: {metadata_path}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nImages by original split:")
for split in SPLITS:
    count = len([img for img, s in image_split_map.items() if s == split.replace("_set", "").lower()])
    print(f"  {split}: {count}")

print(f"\nDisease distribution (top 10):")
sorted_diseases = sorted(stats["disease_counts"].items(), key=lambda x: x[1], reverse=True)
for disease, count in sorted_diseases[:10]:
    print(f"  {disease}: {count}")

print(f"\nDisease distribution (bottom 10):")
for disease, count in sorted(sorted_diseases[-10:], key=lambda x: x[1]):
    print(f"  {disease}: {count}")

print(f"\nFolder sizes:")
for disease in sorted(all_diseases)[:10]:
    disease_folder = MERGEABLE_ROOT / disease
    file_count = len(list(disease_folder.glob("*.*")))
    print(f"  {disease}/: {file_count} files")

print("\n" + "="*60)
print("REORGANIZATION COMPLETE!")
print("="*60)
print(f"\nNew folder created: {MERGEABLE_ROOT}")
print(f"Ready for merging with RFMiD_2!")

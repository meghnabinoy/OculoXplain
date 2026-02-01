"""
Merge mergeable_rfmid_1 and RFMiD_2 datasets into a single merged_RFMID folder.

This script:
1. Creates merged_RFMID folder with 51 disease subfolders
2. Copies all images from mergeable_rfmid_1
3. Copies all images from RFMiD_2 (excluding preprocessed folder)
4. Creates metadata CSV tracking source dataset for each image
5. Handles duplicates and logs statistics
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import csv

# Paths
DATA_ROOT = Path("data")
RFMID1_MERGEABLE = DATA_ROOT / "RFMID_1" / "mergeable_rfmid_1"
RFMID2_ROOT = DATA_ROOT / "RFMiD_2"
MERGED_ROOT = DATA_ROOT / "merged_RFMID"

# Create merged root folder
MERGED_ROOT.mkdir(exist_ok=True)
print(f"✓ Created root folder: {MERGED_ROOT}")

# Get all disease folders from both datasets
disease_folders_1 = set([f.name for f in RFMID1_MERGEABLE.glob("*") if f.is_dir()])
disease_folders_2 = set([f.name for f in RFMID2_ROOT.glob("*") if f.is_dir() and f.name not in ["preprocessed", "processed"]])
all_diseases = sorted(disease_folders_1 | disease_folders_2)

print(f"\nDisease classes from mergeable_rfmid_1: {len(disease_folders_1)}")
print(f"Disease classes from RFMiD_2: {len(disease_folders_2)}")
print(f"Total unique disease classes: {len(all_diseases)}")

# Create all disease folders in merged dataset
print("\n" + "="*60)
print("CREATING DISEASE FOLDERS")
print("="*60)

for disease in all_diseases:
    disease_folder = MERGED_ROOT / disease
    disease_folder.mkdir(exist_ok=True)

print(f"✓ Created {len(all_diseases)} disease folders")

# Track statistics
stats = {
    "rfmid1_images": 0,
    "rfmid2_images": 0,
    "duplicate_images": 0,
    "disease_counts": defaultdict(int),
    "source_counts": defaultdict(int),
    "file_list": [],
}

# Copy images from mergeable_rfmid_1
print("\n" + "="*60)
print("COPYING IMAGES FROM mergeable_rfmid_1")
print("="*60)

for disease in all_diseases:
    source_folder = RFMID1_MERGEABLE / disease
    
    if source_folder.exists():
        dest_folder = MERGED_ROOT / disease
        
        for img_file in source_folder.glob("*.*"):
            if img_file.is_file():
                dest_path = dest_folder / img_file.name
                
                # Check for duplicates
                if dest_path.exists():
                    print(f"⚠ Duplicate found: {disease}/{img_file.name} (skipping RFMiD_1 version)")
                    stats["duplicate_images"] += 1
                else:
                    shutil.copy2(img_file, dest_path)
                    stats["rfmid1_images"] += 1
                    stats["disease_counts"][disease] += 1
                    stats["source_counts"]["rfmid1"] += 1
                    stats["file_list"].append({
                        "disease": disease,
                        "image_file": img_file.name,
                        "source": "rfmid1",
                        "image_id": img_file.stem
                    })

print(f"✓ Copied {stats['rfmid1_images']} images from mergeable_rfmid_1")

# Copy images from RFMiD_2 (excluding preprocessed and processed folders)
print("\n" + "="*60)
print("COPYING IMAGES FROM RFMiD_2")
print("="*60)

skip_folders = {"preprocessed", "processed", "rfmid2_preprocessed_metadata.csv"}

for disease in all_diseases:
    source_folder = RFMID2_ROOT / disease
    
    if source_folder.exists() and source_folder.is_dir():
        dest_folder = MERGED_ROOT / disease
        
        for img_file in source_folder.glob("*.*"):
            if img_file.is_file():
                dest_path = dest_folder / img_file.name
                
                # Check for duplicates
                if dest_path.exists():
                    print(f"⚠ Duplicate found: {disease}/{img_file.name} (skipping RFMiD_2 version)")
                    stats["duplicate_images"] += 1
                else:
                    shutil.copy2(img_file, dest_path)
                    stats["rfmid2_images"] += 1
                    stats["disease_counts"][disease] += 1
                    stats["source_counts"]["rfmid2"] += 1
                    stats["file_list"].append({
                        "disease": disease,
                        "image_file": img_file.name,
                        "source": "rfmid2",
                        "image_id": img_file.stem
                    })

print(f"✓ Copied {stats['rfmid2_images']} images from RFMiD_2")

# Create metadata CSV
print("\n" + "="*60)
print("CREATING METADATA CSV")
print("="*60)

metadata_path = MERGED_ROOT / "merged_dataset_metadata.csv"
with open(metadata_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["disease", "image_file", "source", "image_id"])
    writer.writeheader()
    writer.writerows(stats["file_list"])

print(f"✓ Created metadata CSV: {metadata_path}")

# Print summary statistics
print("\n" + "="*60)
print("MERGE SUMMARY")
print("="*60)

print(f"\nImages copied:")
print(f"  From mergeable_rfmid_1: {stats['rfmid1_images']}")
print(f"  From RFMiD_2: {stats['rfmid2_images']}")
print(f"  TOTAL: {stats['rfmid1_images'] + stats['rfmid2_images']}")
print(f"  Duplicates skipped: {stats['duplicate_images']}")

print(f"\nTotal unique disease classes: {len(all_diseases)}")

print(f"\nDisease distribution in merged dataset (top 15):")
sorted_diseases = sorted(stats["disease_counts"].items(), key=lambda x: x[1], reverse=True)
for disease, count in sorted_diseases[:15]:
    print(f"  {disease}: {count}")

print(f"\nDisease distribution in merged dataset (bottom 10):")
for disease, count in sorted(sorted_diseases[-10:], key=lambda x: x[1]):
    print(f"  {disease}: {count}")

# Get folder sizes
print(f"\nFolder sizes in merged_RFMID (top 15):")
disease_file_counts = defaultdict(int)
for disease_folder in MERGED_ROOT.glob("*"):
    if disease_folder.is_dir():
        count = len(list(disease_folder.glob("*.*")))
        disease_file_counts[disease_folder.name] = count

for disease, count in sorted(disease_file_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {disease}/: {count} files")

print("\n" + "="*60)
print("MERGE COMPLETE!")
print("="*60)
print(f"\nMerged dataset created: {MERGED_ROOT}")
print(f"Ready for unified preprocessing and training!")

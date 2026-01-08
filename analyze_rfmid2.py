"""
RFMiD_2 Dataset Analyzer
Creates metadata CSV with image paths and multi-label annotations for 51 disease classes
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
DATA_DIR = Path("data/RFMiD_2")
OUTPUT_CSV = DATA_DIR / "rfmid2_metadata.csv"

# Disease class mapping with full names
DISEASE_CLASSES = {
    'AH': 'Arteriolar narrowing/Hypertension',
    'AION': 'Anterior Ischemic Optic Neuropathy',
    'ARMD': 'Age-Related Macular Degeneration',
    'BRVO': 'Branch Retinal Vein Occlusion',
    'CB': "Coats Disease",
    'CF': 'Chorioretinal Folds',
    'CL': 'Central Retinal Artery Ischemia',
    'CME': 'Cystoid Macular Edema',
    'CNV': 'Choroidal Neovascularization',
    'CRAO': 'Central Retinal Artery Occlusion',
    'CRS': 'Central Serous (Chronic)',
    'CRVO': 'Central Retinal Vein Occlusion',
    'CSC': 'Central Serous Chorioretinopathy',
    'CSR': 'Central Serous Retinopathy',
    'CWS': 'Cotton Wool Spots',
    'DN': 'Drusen',
    'DR': 'Diabetic Retinopathy',
    'EDN': 'Epiretinal Membrane with Drusen',
    'ERM': 'Epiretinal Membrane',
    'GRT': 'Giant Retinal Tear',
    'HPED': 'Hemorrhagic Pigment Epithelial Detachment',
    'HR': 'Retinal Hemorrhage',
    'HTN': 'Hypertensive Retinopathy',
    'IIH': 'Idiopathic Intracranial Hypertension',
    'LS': 'Laser Scars',
    'MCA': 'Macular Atrophy',
    'ME': 'Macular Edema',
    'MH': 'Macular Hole',
    'MHL': 'Macular Hole (Large/Lamellar)',
    'MS': 'Myelinated Nerve Fibers',
    'MYA': 'Myopia-related Changes',
    'ODC': 'Optic Disc Cupping',
    'ODE': 'Optic Disc Edema',
    'ODP': 'Optic Disc Pit',
    'ON': 'Optic Neuritis',
    'OPDM': 'Optic Disc Pallor/Dysmyelination',
    'PRH': 'Preretinal Hemorrhage',
    'RD': 'Retinal Detachment',
    'RHL': 'Retinal Hemorrhage (Layered)',
    'RP': 'Retinitis Pigmentosa',
    'RPEC': 'Retinal Pigment Epithelium Changes',
    'RS': 'Retinal Scar',
    'RT': 'Retinal Tear',
    'RTR': 'Recurrent Retinal Tear',
    'SOFE': 'Subretinal Fluid/Exudate',
    'ST': 'Staphyloma',
    'TD': 'Tilted Disc',
    'TSLN': 'Tessellated Fundus',
    'TV': 'Temporal Pallor',
    'VS': 'Vitreous Syneresis/Strands',
    'WNL': 'Within Normal Limits (Normal)'
}

# Rare vs common classification
COMMON_CLASSES = {'ARMD', 'DR', 'HTN', 'CME', 'ME', 'MH', 'MHL', 'ERM', 'CNV', 'WNL'}
RARE_CLASSES = set(DISEASE_CLASSES.keys()) - COMMON_CLASSES

def create_metadata():
    """Create metadata CSV from folder structure"""
    
    print("=" * 80)
    print("RFMiD_2 Dataset Analyzer - Rare Retinal Disease Classification")
    print("=" * 80)
    
    records = []
    class_counts = {cls: 0 for cls in DISEASE_CLASSES.keys()}
    
    # Iterate through all disease folders
    for disease_code, disease_name in DISEASE_CLASSES.items():
        folder_path = DATA_DIR / disease_code
        
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {disease_code}")
            continue
        
        # Get all images in this folder
        images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        
        for img_path in images:
            # Create multi-label vector (all zeros except this disease)
            labels = {cls: 0 for cls in DISEASE_CLASSES.keys()}
            labels[disease_code] = 1
            
            # Full absolute path
            full_path = str(img_path.resolve())
            
            record = {
                'image_path': full_path,
                'image_name': img_path.name,
                'primary_disease': disease_code,
                'disease_name': disease_name,
                'is_rare': 1 if disease_code in RARE_CLASSES else 0,
                'is_normal': 1 if disease_code == 'WNL' else 0,
                **labels  # Add all 51 binary labels
            }
            
            records.append(record)
            class_counts[disease_code] += 1
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"Total images: {len(df)}")
    print(f"Total classes: {len(DISEASE_CLASSES)}")
    print(f"Common diseases: {len(COMMON_CLASSES)}")
    print(f"Rare diseases: {len(RARE_CLASSES)}")
    
    print(f"\n🔢 Class Distribution:")
    print("=" * 80)
    
    # Sort by count
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    rare_total = 0
    common_total = 0
    
    for disease_code, count in sorted_counts:
        rarity = "COMMON" if disease_code in COMMON_CLASSES else "RARE"
        print(f"{disease_code:6} | {DISEASE_CLASSES[disease_code]:50} | {count:4} | {rarity}")
        
        if disease_code in RARE_CLASSES:
            rare_total += count
        else:
            common_total += count
    
    print("=" * 80)
    print(f"Total rare disease images: {rare_total}")
    print(f"Total common disease images: {common_total}")
    print(f"Rare disease ratio: {rare_total / len(df) * 100:.2f}%")
    
    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Metadata saved to: {OUTPUT_CSV}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Rows: {len(df)}")
    
    # Show sample
    print(f"\n📋 Sample records:")
    print(df[['image_name', 'primary_disease', 'disease_name', 'is_rare']].head(10))
    
    return df

if __name__ == "__main__":
    df = create_metadata()
    
    print("\n" + "=" * 80)
    print("✅ Dataset analysis complete!")
    print("=" * 80)

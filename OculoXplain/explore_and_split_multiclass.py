import pandas as pd
from sklearn.model_selection import train_test_split

# Load the ODIR-5K full dataframe
labels_df = pd.read_csv(r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\full_df.csv')

# --- 1. Explore class distribution ---
# Disease columns: N,D,G,C,A,H,M,O (from the CSV header)
disease_cols = ['N','D','G','C','A','H','M','O']
print('Class distribution (number of images per disease):')
for col in disease_cols:
    print(f"{col}: {labels_df[col].sum()}")

# --- 2. Split data into train/val/test without mixing patients ---
# Use 'ID' as patient identifier
unique_patients = labels_df['ID'].unique()
train_ids, temp_ids = train_test_split(unique_patients, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

def assign_split(pid):
    if pid in train_ids:
        return 'train'
    elif pid in val_ids:
        return 'val'
    else:
        return 'test'

labels_df['split'] = labels_df['ID'].apply(assign_split)

# Save the new dataframe with split info
labels_df.to_csv(r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\full_df_with_split.csv', index=False)
print('\nSplit counts:')
print(labels_df['split'].value_counts())

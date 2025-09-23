# check_class_balance_augmented.py
"""
Script to print the number of samples per class in the augmented ODIR-5K dataset.
"""
import pandas as pd

csv_path = 'data/ODIR-5K/full_df_with_split_augmented.csv'
df = pd.read_csv(csv_path)
df['main_label'] = df['labels'].str.extract(r"\['([A-Z])'\]")
print('Class distribution after augmentation:')
print(df['main_label'].value_counts())

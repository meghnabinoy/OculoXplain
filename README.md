# OculoXplain Project Setup

This project is focused on rare retinal disease detection and explainability using deep learning. Below are the steps and scripts created so far to help you get started and organized.

## 1. Requirements Installation
Install the required Python packages:
```
pip install -r requirements.txt
```

## 2. Dataset Preparation
- Download datasets manually from Kaggle (recommended: ODIR-5K, RFMiD, Retinal Disease Classification).
- Unzip each dataset into the `data/` folder, preserving their original folder structure.

## 3. Data Exploration Scripts

### ODIR-5K
- Script: `data_explore_odir5k.py`
- Purpose: Loads `full_df.csv` and displays 10 random sample images with their diagnostic keywords and labels.
- Usage:
  ```
  python data_explore_odir5k.py
  ```
- Make sure your images are in:
  `data/ODIR-5K/ODIR-5K/ODIR-5K/Training Images`
  and your CSV is at:
  `data/ODIR-5K/full_df.csv`

## 4. Notes
- If you add more datasets, follow the same pattern: place images and CSVs in the `data/` folder and adjust scripts as needed.
- For ODIR-5K, the script uses the `filename` column in `full_df.csv` to find images.
- If you encounter issues with image display, check that the filenames in the CSV match the actual files in the image folder.

## 5. Next Steps
- Proceed to preprocessing, modeling, and explainability as outlined in your project plan.
- Update this README as you add new scripts or steps.

---

For any issues or questions, check the script comments or ask for help!

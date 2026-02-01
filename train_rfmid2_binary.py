import os
import glob
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data") / "RFMiD_2"
MODEL_SAVE = Path("resnet50_rfmid2_binary_model.pth")
METADATA_CSV = DATA_DIR / "rfmid2_preprocessed_metadata.csv"


def gather_images(data_dir=DATA_DIR):
    # Prefer metadata CSV produced by preprocessing if available
    if METADATA_CSV.exists():
        print(f"Loading metadata CSV: {METADATA_CSV}")
        df = pd.read_csv(METADATA_CSV)
        # Use preprocessed_path when available, otherwise original_path
        if 'preprocessed_path' in df.columns:
            df = df.rename(columns={'preprocessed_path': 'path'})
        elif 'original_path' in df.columns:
            df = df.rename(columns={'original_path': 'path'})
        df = df[['path', 'label', 'class_name']] if 'class_name' in df.columns else df[['path', 'label']]
        return df

    exts = (".jpg", ".jpeg", ".png")
    rows = []
    for root, dirs, files in os.walk(data_dir):
        base = os.path.basename(root)
        # skip if at top-level
        if root == str(data_dir):
            continue
        label = 0 if base.upper() == 'WNL' else 1
        for f in files:
            if f.lower().endswith(exts):
                rows.append({
                    'path': os.path.join(root, f),
                    'label': label,
                    'class_name': base
                })
    df = pd.DataFrame(rows)
    return df


class RFMiDBinaryDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['label'])
        return img, label


def create_loaders(df, batch_size=32, val_size=0.2):
    train_df, val_df = train_test_split(df, test_size=val_size, stratify=df['label'], random_state=42)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = RFMiDBinaryDataset(train_df, transform=train_transform)
    val_ds = RFMiDBinaryDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def create_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        scheduler.step()

    if best_state is not None:
        torch.save(best_state, MODEL_SAVE)
        print(f"Saved best model with val acc {best_val_acc:.4f} to {MODEL_SAVE}")

    # Plot training curves
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(train_losses, label='Train Loss', marker='o')
        ax1.plot(val_losses, label='Val Loss', marker='s')
        ax1.legend(); ax1.set_title('Loss')
        ax2.plot(train_accs, label='Train Acc', marker='o')
        ax2.plot(val_accs, label='Val Acc', marker='s')
        ax2.legend(); ax2.set_title('Accuracy')
        plt.tight_layout(); plt.savefig('training_curves_rfmid2_binary.png')
    except Exception:
        pass


def main():
    df = gather_images()
    if df.empty:
        print(f"No images found under {DATA_DIR}. Please place RFMiD_2 images as data/RFMiD_2/<CLASS>/*.jpg")
        return

    print(f"Found {len(df)} images. Healthy (WNL): {int((df['label']==0).sum())}, Unhealthy: {int((df['label']==1).sum())}")

    train_loader, val_loader = create_loaders(df, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()

    trained = train(model, train_loader, val_loader, device, epochs=5, lr=1e-4)


if __name__ == '__main__':
    main()

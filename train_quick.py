"""
Quick Training for RFMiD_2 - Faster version with smaller model
Uses MobileNetV2 for rapid training on CPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Fast configuration
DATA_DIR = Path("data/RFMiD_2")
METADATA_CSV = DATA_DIR / "rfmid2_preprocessed_metadata.csv"
MODEL_SAVE_PATH = "mobilenet_rfmid2_quick_model.pth"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 128  # Larger batch
NUM_EPOCHS = 5    # Just 5 epochs for quick results
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASE_CLASSES = [
    'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'CME', 'CNV', 'CRAO',
    'CRS', 'CRVO', 'CSC', 'CSR', 'CWS', 'DN', 'DR', 'EDN', 'ERM', 'GRT',
    'HPED', 'HR', 'HTN', 'IIH', 'LS', 'MCA', 'ME', 'MH', 'MHL', 'MS',
    'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'OPDM', 'PRH', 'RD', 'RHL', 'RP',
    'RPEC', 'RS', 'RT', 'RTR', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 'VS', 'WNL'
]

class RFMiD2Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.disease_classes = DISEASE_CLASSES
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['preprocessed_path']
        if not Path(img_path).is_absolute():
            img_path = Path("data") / img_path
        
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        
        labels = torch.tensor([row[cls] for cls in self.disease_classes], dtype=torch.float32)
        return img, labels

def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class QuickModel(nn.Module):
    """Lightweight MobileNetV2 for fast training"""
    def __init__(self, num_classes=51):
        super(QuickModel, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        
    def forward(self, x):
        return self.model(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.numel()
    
    accuracy = correct / total
    return total_loss / len(loader), accuracy

def main():
    print("="*60)
    print("QUICK TRAINING - RFMiD_2 Rare Disease Classification")
    print("Using MobileNetV2 for fast CPU training")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load data
    df = pd.read_csv(METADATA_CSV)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    
    # Use only subset for faster training
    print(f"\\nUsing 50% of training data for speed...")
    train_df = train_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    
    # Datasets
    train_dataset = RFMiD2Dataset(train_df, transform=get_transforms())
    val_dataset = RFMiD2Dataset(val_df, transform=get_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = QuickModel(num_classes=len(DISEASE_CLASSES)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\\nTraining for {NUM_EPOCHS} epochs (FAST MODE)...")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Model saved!")
    
    print("\\n" + "="*60)
    print(f"✅ Quick training complete!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Time estimate: ~20-30 minutes on CPU")
    print("="*60)

if __name__ == "__main__":
    main()

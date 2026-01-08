"""
RFMiD_2 Multi-Label Rare Disease Classification Training
Uses ResNet50 with multi-label classification for 51 rare retinal diseases
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
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss, f1_score, jaccard_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("data/RFMiD_2")
METADATA_CSV = DATA_DIR / "rfmid2_preprocessed_metadata.csv"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

MODEL_SAVE_PATH = "resnet50_rfmid2_rare_disease_model.pth"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64  # Increased for CPU efficiency
NUM_EPOCHS = 15  # Reduced from 25
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 4  # Stop if no improvement

# Disease classes  
DISEASE_CLASSES = [
    'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'CME', 'CNV', 'CRAO',
    'CRS', 'CRVO', 'CSC', 'CSR', 'CWS', 'DN', 'DR', 'EDN', 'ERM', 'GRT',
    'HPED', 'HR', 'HTN', 'IIH', 'LS', 'MCA', 'ME', 'MH', 'MHL', 'MS',
    'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'OPDM', 'PRH', 'RD', 'RHL', 'RP',
    'RPEC', 'RS', 'RT', 'RTR', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 'VS', 'WNL'
]

class RFMiD2Dataset(Dataset):
    """PyTorch Dataset for RFMiD_2 multi-label classification"""
    
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.disease_classes = DISEASE_CLASSES
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image from preprocessed path (make absolute if needed)
        img_path = row['preprocessed_path']
        if not Path(img_path).is_absolute():
            img_path = Path("data") / img_path  # Prepend data/ if relative
        
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get multi-label targets (51 classes)
        labels = torch.tensor([row[cls] for cls in self.disease_classes], dtype=torch.float32)
        
        return img, labels

def get_transforms(train=True):
    """Get image transforms"""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

class MultiLabelResNet50(nn.Module):
    """ResNet50 adapted for multi-label classification"""
    
    def __init__(self, num_classes=51):
        super(MultiLabelResNet50, self).__init__()
        
        # Load pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace final layer for multi-label classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def calculate_class_weights(df, disease_classes):
    """Calculate class weights for imbalanced dataset"""
    class_counts = []
    total_samples = len(df)
    
    for cls in disease_classes:
        count = df[cls].sum()
        class_counts.append(count if count > 0 else 1)  # Avoid division by zero
    
    # Inverse frequency weighting
    class_weights = [total_samples / (len(disease_classes) * count) for count in class_counts]
    
    return torch.tensor(class_weights, dtype=torch.float32)

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Predictions (threshold at 0.5)
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    avg_loss = running_loss / len(loader)
    hamming = hamming_loss(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    jaccard = jaccard_score(all_labels, all_preds, average='samples', zero_division=0)
    
    return avg_loss, hamming, f1_micro, f1_macro, jaccard

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    avg_loss = running_loss / len(loader)
    hamming = hamming_loss(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    jaccard = jaccard_score(all_labels, all_preds, average='samples', zero_division=0)
    
    return avg_loss, hamming, f1_micro, f1_macro, jaccard

def plot_training_curves(history, save_path="training_curves_rfmid2.png"):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("RFMiD_2 Rare Disease Classification Training", fontsize=16, fontweight='bold')
    
    metrics = ['loss', 'hamming', 'f1_micro', 'f1_macro', 'jaccard']
    titles = ['Loss', 'Hamming Loss', 'F1 Score (Micro)', 'F1 Score (Macro)', 'Jaccard Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        ax.plot(history[f'train_{metric}'], label='Train', marker='o')
        ax.plot(history[f'val_{metric}'], label='Val', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training curves saved to {save_path}")
    plt.close()

def main():
    print("=" * 80)
    print("RFMiD_2 Rare Retinal Disease Classification Training")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load metadata
    print(f"\n📊 Loading metadata from {METADATA_CSV}...")
    df = pd.read_csv(METADATA_CSV)
    
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples: {len(val_df)}")
    print(f"   Classes: {len(DISEASE_CLASSES)}")
    
    # Calculate class weights
    print("\n⚖️  Calculating class weights for imbalanced dataset...")
    class_weights = calculate_class_weights(train_df, DISEASE_CLASSES).to(DEVICE)
    print(f"   Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = RFMiD2Dataset(train_df, transform=get_transforms(train=True))
    val_dataset = RFMiD2Dataset(val_df, transform=get_transforms(train=False))
    
    # Reduce workers for CPU training
    num_workers = 0 if DEVICE.type == 'cpu' else 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🧠 Building model...")
    model = MultiLabelResNet50(num_classes=len(DISEASE_CLASSES))
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_hamming': [], 'val_hamming': [],
        'train_f1_micro': [], 'val_f1_micro': [],
        'train_f1_macro': [], 'val_f1_macro': [],
        'train_jaccard': [], 'val_jaccard': []
    }
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    print("\n🚀 Starting training...")
    if DEVICE.type == 'cpu':
        print("⚠️  WARNING: Training on CPU - will be slow. Install CUDA PyTorch for GPU acceleration.")
    print("=" * 80)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_hamming, train_f1_micro, train_f1_macro, train_jaccard = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_hamming, val_f1_micro, val_f1_macro, val_jaccard = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_hamming'].append(train_hamming)
        history['val_hamming'].append(val_hamming)
        history['train_f1_micro'].append(train_f1_micro)
        history['val_f1_micro'].append(val_f1_micro)
        history['train_f1_macro'].append(train_f1_macro)
        history['val_f1_macro'].append(val_f1_macro)
        history['train_jaccard'].append(train_jaccard)
        history['val_jaccard'].append(val_jaccard)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train F1 (Micro): {train_f1_micro:.4f} | Val F1 (Micro): {val_f1_micro:.4f}")
        print(f"Train F1 (Macro): {train_f1_macro:.4f} | Val F1 (Macro): {val_f1_macro:.4f}")
        print(f"Train Jaccard: {train_jaccard:.4f} | Val Jaccard: {val_jaccard:.4f}")
        print(f"Hamming Loss: {train_hamming:.4f} (Train) | {val_hamming:.4f} (Val)")
        
        # Save best model
        if val_f1_micro > best_val_f1:
            best_val_f1 = val_f1_micro
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Best model saved! (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print(f"Best validation F1 (Micro): {best_val_f1:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    # Plot training curves
    plot_training_curves(history)
    
    print("=" * 80)

if __name__ == "__main__":
    main()

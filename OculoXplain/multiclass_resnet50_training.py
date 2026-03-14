"""
Phase 2 - Person 3: Multi-Class Classification
- Modify ResNet50 for multi-class rare disease classification
- Train & evaluate using accuracy and F1-score
- Handle class imbalance with weighted loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import cv2
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Configuration
CSV_PATH = r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\full_df_with_split_augmented.csv'
IMG_DIR = r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\ODIR-5K\Training Images'
AUG_DIR = r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\augmented'
MODEL_SAVE_PATH = r'resnet50_multiclass_retinal_model.pth'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom Dataset
class RetinalDataset(Dataset):
    def __init__(self, dataframe, img_dir, aug_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.aug_dir = aug_dir
        self.transform = transform
        
        # Create label mapping
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.df['main_label'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_map.items()}
        print(f"Label mapping: {self.label_map}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        
        # Check if it's an augmented image
        if row.get('augmented', False):
            img_path = os.path.join(self.aug_dir, filename)
        else:
            img_path = os.path.join(self.img_dir, filename)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a black image if file not found
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.label_map[row['main_label']]
        
        return image, label

# Data transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_model(num_classes):
    """Create ResNet50 model for multi-class classification"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify the final layer for multi-class classification
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def calculate_class_weights(labels):
    """Calculate class weights for handling imbalance"""
    from collections import Counter
    label_counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(len(label_counts)):
        count = label_counts.get(i, 1)
        weight = total / (len(label_counts) * count)
        weights.append(weight)
    return torch.FloatTensor(weights)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(train_accs, label='Train Acc')
    axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    # F1-Score
    axes[2].plot(train_f1s, label='Train F1')
    axes[2].plot(val_f1s, label='Val F1')
    axes[2].set_title('Training and Validation F1-Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Score')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('multiclass_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

# Quick version with reduced parameters for faster training

# ...same imports and dataset class...

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    df['main_label'] = df['labels'].str.extract(r"\['([A-Z])'\]")
    
    # Use smaller subset for quick testing
    train_df = df[df['split'] == 'train'].sample(1000)  # Only 1000 samples
    val_df = df[df['split'] == 'val'].sample(200)       # Only 200 samples
    
    # Create datasets
    train_dataset = RetinalDataset(train_df, IMG_DIR, AUG_DIR, train_transform)
    val_dataset = RetinalDataset(val_df, IMG_DIR, AUG_DIR, val_transform)
    
    # Smaller batch size, no workers
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Create model
    num_classes = len(train_dataset.label_map)
    model = create_model(num_classes).to(device)
    
    # Simple loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Only 3 epochs for quick test
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'quick_model.pth')
    print("Quick training complete!")

if __name__ == "__main__":
    main()
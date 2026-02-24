"""
ResNet50 Training Script for Merged RFMiD Dataset
Multi-class classification for 51 retinal disease classes
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("Data/merged_RFMID/merged_RFMID")
MODEL_SAVE_PATH = "resnet50_merged_dataset_model.pth"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced batch size
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001  # Reduced learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 5

# Disease classes (51 classes from merged dataset)
DISEASE_CLASSES = [
    'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'CME', 'CNV', 'CRAO',
    'CRS', 'CRVO', 'CSC', 'CSR', 'CWS', 'DN', 'DR', 'EDN', 'ERM', 'GRT',
    'HPED', 'HR', 'HTN', 'IIH', 'LS', 'MCA', 'ME', 'MH', 'MHL', 'MS',
    'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'OPDM', 'PRH', 'RD', 'RHL', 'RP',
    'RPEC', 'RS', 'RT', 'RTR', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 'VS', 'WNL'
]

class MergedDataset(Dataset):
    """PyTorch Dataset for merged RFMiD dataset"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.disease_classes = DISEASE_CLASSES
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.disease_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.image_paths = []
        self.labels = []
        
        print("Loading dataset...")
        for disease_class in self.disease_classes:
            class_dir = self.data_dir / disease_class
            if class_dir.exists():
                class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                for img_path in class_images:
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[disease_class])
                print(f"  {disease_class}: {len(class_images)} images")
            else:
                print(f"  Warning: {disease_class} directory not found")
        
        print(f"Total images loaded: {len(self.image_paths)}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            # Handle corrupted images
            print(f"Warning: Could not load image {img_path}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

def get_transforms(train=True):
    """Get image transforms"""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

class ResNet50Model(nn.Module):
    """ResNet50 model for multi-class classification"""
    
    def __init__(self, num_classes=51):
        super(ResNet50Model, self).__init__()
        
        # Load pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze early layers to prevent overfitting
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer with dropout for regularization
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    class_counts = torch.zeros(len(DISEASE_CLASSES))
    
    for _, label in dataset:
        class_counts[label] += 1
    
    # Only calculate weights for classes that have samples
    non_zero_mask = class_counts > 0
    
    if non_zero_mask.sum() == 0:
        return torch.ones(len(DISEASE_CLASSES))
    
    # Calculate weights only for non-zero classes
    weights = torch.ones(len(DISEASE_CLASSES))
    total_samples = class_counts[non_zero_mask].sum()
    
    for i in range(len(DISEASE_CLASSES)):
        if class_counts[i] > 0:
            weights[i] = total_samples / (non_zero_mask.sum() * class_counts[i])
    
    # Clamp weights to reasonable range
    weights = torch.clamp(weights, 0.1, 10.0)
    
    print(f"Class weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    return weights

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Check for NaN in inputs
        if torch.isnan(images).any() or torch.isnan(labels.float()).any():
            print(f"NaN detected in batch {batch_idx}")
            continue
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            print(f"NaN in model outputs at batch {batch_idx}")
            continue
            
        loss = criterion(outputs, labels)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"NaN loss at batch {batch_idx}")
            continue
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # Predictions for metrics
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Early detection of issues
        if batch_idx > 0 and batch_idx % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            if avg_loss > 100 or np.isnan(avg_loss):
                print(f"Warning: High/NaN loss detected: {avg_loss}")
    
    # Calculate metrics
    epoch_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, accuracy, f1_micro, f1_macro

def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Skip if NaN inputs
            if torch.isnan(images).any() or torch.isnan(labels.float()).any():
                continue
            
            outputs = model(images)
            
            # Skip if NaN outputs
            if torch.isnan(outputs).any():
                continue
                
            loss = criterion(outputs, labels)
            
            # Skip if NaN loss
            if torch.isnan(loss):
                continue
            
            running_loss += loss.item()
            
            # Predictions for metrics
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    if len(all_preds) == 0:
        return float('nan'), 0.0, 0.0, 0.0, [], []
        
    epoch_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, accuracy, f1_micro, f1_macro, all_preds, all_labels

def plot_training_curves(history, save_path="training_curves_merged_dataset.png"):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("ResNet50 Merged Dataset Training", fontsize=16, fontweight='bold')
    
    metrics = ['loss', 'accuracy', 'f1_micro', 'f1_macro']
    titles = ['Loss', 'Accuracy', 'F1 Score (Micro)', 'F1 Score (Macro)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(history[f'train_{metric}'], label='Train', marker='o')
        ax.plot(history[f'val_{metric}'], label='Validation', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training curves saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix_merged_dataset.png"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ResNet50 Merged Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.close()

def main():
    print("=" * 80)
    print("ResNet50 Training on Merged RFMiD Dataset")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Check if dataset exists
    if not DATA_DIR.exists():
        print(f"❌ Error: Dataset directory not found: {DATA_DIR}")
        return
    
    # Create dataset
    print(f"\n📊 Loading dataset from {DATA_DIR}...")
    full_dataset = MergedDataset(DATA_DIR, transform=get_transforms(train=False))
    
    if len(full_dataset) == 0:
        print("❌ Error: No images found in dataset!")
        return
    
    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Classes: {len(DISEASE_CLASSES)}")
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms to split datasets
    train_dataset.dataset.transform = get_transforms(train=True)
    val_dataset.dataset.transform = get_transforms(train=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Calculate class weights
    print("\n⚖️  Calculating class weights for imbalanced dataset...")
    class_weights = calculate_class_weights(full_dataset).to(DEVICE)
    
    # Create model
    print(f"\n🏗️  Creating ResNet50 model...")
    model = ResNet50Model(num_classes=len(DISEASE_CLASSES)).to(DEVICE)
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer with safer settings
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1_micro': [], 'val_f1_micro': [],
        'train_f1_macro': [], 'val_f1_macro': []
    }
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc, train_f1_micro, train_f1_macro = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Check for training issues
        if np.isnan(train_loss):
            print("❌ NaN loss detected! Stopping training.")
            break
        
        # Validate
        val_loss, val_acc, val_f1_micro, val_f1_macro, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, DEVICE
        )
        
        # Update scheduler
        if not np.isnan(val_loss):
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['train_f1_micro'].append(train_f1_micro)
        history['val_f1_micro'].append(val_f1_micro)
        history['train_f1_macro'].append(train_f1_macro)
        history['val_f1_macro'].append(val_f1_macro)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train F1 (Micro): {train_f1_micro:.4f} | Val F1 (Micro): {val_f1_micro:.4f}")
        print(f"Train F1 (Macro): {train_f1_macro:.4f} | Val F1 (Macro): {val_f1_macro:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_f1_micro > best_val_f1 and not np.isnan(val_f1_micro):
            best_val_f1 = val_f1_micro
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'class_to_idx': full_dataset.class_to_idx,
                'disease_classes': DISEASE_CLASSES
            }, MODEL_SAVE_PATH)
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
    
    # Plot confusion matrix for final validation
    if len(val_preds) > 0:
        plot_confusion_matrix(val_labels, val_preds, DISEASE_CLASSES)
    
        # Print classification report with proper label handling
        print("\n📊 Final Validation Classification Report:")
        try:
            # Get unique classes present in validation
            unique_labels = sorted(list(set(val_labels)))
            class_names_present = [DISEASE_CLASSES[i] for i in unique_labels]
            
            print(classification_report(
                val_labels, val_preds, 
                labels=unique_labels,
                target_names=class_names_present, 
                digits=4
            ))
        except Exception as e:
            print(f"Classification report error (non-critical): {e}")
            print(f"Final validation accuracy: {accuracy_score(val_labels, val_preds):.4f}")
            print(f"Classes present in validation: {len(set(val_labels))}/{len(DISEASE_CLASSES)}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
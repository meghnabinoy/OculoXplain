import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =======================
# PATH CONFIGURATION
# =======================
DATA_DIR = r"C:\FinalProject\OculoXplain-main\Data\Ocular_Disease_Dataset\ODIR-5K\ODIR-5K\processed"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

# =======================
# DATA TRANSFORMS
# =======================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =======================
# DATA LOADING
# =======================
def load_data():
    print(f"Loading data from {DATA_DIR}")
    
    # Check if processed data exists
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("ERROR: Processed data not found. Please run preprocessing first.")
        return None, None, None, None
    
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    # Print class mapping
    print("Class mapping:", train_dataset.class_to_idx)
    
    # Expected mapping: {'disease': 0, 'healthy': 1} or {'healthy': 0, 'disease': 1}
    # We want healthy=0, disease=1 for binary classification
    if 'healthy' in train_dataset.class_to_idx and train_dataset.class_to_idx['healthy'] != 0:
        print("⚠ Adjusting labels: healthy=0, disease=1")
        # Swap the labels if needed
        train_dataset.targets = [1 - t for t in train_dataset.targets]
        val_dataset.targets = [1 - t for t in val_dataset.targets]
        
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Use MAXIMUM batch size for your 8GB GPU (from our testing)
    batch_size = 128 if torch.cuda.is_available() else 32
    print(f"Using optimal batch size: {batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=0, pin_memory=torch.cuda.is_available())
    
    return train_loader, val_loader, train_dataset, val_dataset

# =======================
# MODEL CREATION
# =======================
def create_model():
    # Load pretrained ResNet50
    model = models.resnet50(weights='DEFAULT')  # Use new API
    
    # Keep ALL layers trainable for maximum GPU utilization and better performance
    for param in model.parameters():
        param.requires_grad = True
    
    # Replace the classifier for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: healthy, disease
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters (trainable): {trainable_params:,}")
    return model

# =======================
# TRAINING FUNCTION
# =======================
def train_model(model, train_loader, val_loader, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Clear GPU cache for optimal performance
        torch.cuda.empty_cache()
        print("🚀 GPU training optimized for maximum performance!")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay
    
    # Add learning rate scheduler to improve from 69% accuracy
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Reduce LR every 3 epochs
    print("📈 Learning rate scheduler added - LR will reduce every 3 epochs for better convergence")
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 20 == 0:  # More frequent updates with detailed GPU info
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    cached = torch.cuda.memory_reserved() / 1024**3
                    accuracy = 100. * correct / total
                    print(f'Batch {batch_idx+1}/{len(train_loader)}: '
                          f'Loss: {loss.item():.4f} | '
                          f'Acc: {accuracy:.1f}% | '
                          f'GPU: {allocated:.2f}GB/{cached:.2f}GB cached')
                else:
                    print(f'Batch {batch_idx+1}/{len(train_loader)}: Loss: {loss.item():.4f} | Acc: {100.*correct/total:.1f}%')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Learning Rate: {current_lr:.6f}')
    
    return model, train_losses, train_accuracies, val_losses, val_accuracies

# =======================
# PLOTTING FUNCTION
# =======================
def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Acc', marker='o')
    ax2.plot(val_accuracies, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

# =======================
# MAIN FUNCTION
# =======================
def main():
    print("Starting baseline ResNet50 training for retinal disease classification")
    
    # Load data
    train_loader, val_loader, train_dataset, val_dataset = load_data()
    if train_loader is None:
        return
    
    # Create model
    model = create_model()
    print(f"Model created: {model.__class__.__name__}")
    
    # Train model with optimized GPU settings
    print(f"🔥 Starting HIGH-PERFORMANCE GPU training...")
    print("📊 Using 10 epochs - optimal for:")
    print("   • Full dataset (5,600 training images)")
    print("   • Medical image classification (needs more epochs)")
    print("   • Starting from 69% accuracy, targeting 85-90%+")
    trained_model, train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=10  # Increased for full dataset
    )
    
    # Plot results
    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # Save model
    model_save_path = "resnet50_retinal_disease_model.pth"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")
    
    return trained_model, train_dataset, val_dataset

if __name__ == "__main__":
    model, train_dataset, val_dataset = main()

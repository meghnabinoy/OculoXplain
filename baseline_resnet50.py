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
DATA_DIR = r"C:\Users\verth\Documents\projects\OculoXplain\data\ODIR-5K\ODIR-5K\ODIR-5K\processed"
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
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

# Verify label mapping (should be: {'healthy': 0, 'disease': 1})
print("Class mapping:", train_dataset.class_to_idx)
if train_dataset.class_to_idx.get("disease", None) == 0:
    print("⚠ Detected inverted mapping: Swapping labels to match preprocessing!")
    # Swaps disease=1, healthy=0 to match original intended meaning
    train_dataset.targets = [1 - t for t in train_dataset.targets]
    val_dataset.targets = [1 - t for t in val_dataset.targets]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =======================
# MODEL SETUP (BINARY)
# =======================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Prevent overfitting
    nn.Linear(model.fc.in_features, 1)  # Binary classification
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =======================
# TRAINING LOOP
# =======================
num_epochs = 10  # Keep it simple for Phase 1
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(num_epochs):
    # ---- TRAINING ----
    model.train()
    total_train_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # shape (batch, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # ---- VALIDATION ----
    model.eval()
    total_val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = 100 * correct / total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Val Acc: {val_acc:.2f}%")

# =======================
# SAVE MODEL
# =======================
torch.save(model.state_dict(), "resnet50_baseline_odir5k.pth")
print("Model saved as resnet50_baseline_odir5k.pth")

# =======================
# PLOT ACCURACY & LOSS
# =======================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

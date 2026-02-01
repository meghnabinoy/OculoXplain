import os
import torch
from torchvision import models
import torch.nn.functional as F

MODEL_PATH = "resnet50_rfmid2_binary_model.pth"

def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model()
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Loaded weights from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load weights from {MODEL_PATH}: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}; proceeding with randomly initialized head")

    model.to(device).eval()

    # Create a dummy input and run a forward pass
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

    print(f"Dummy forward pass successful. Predicted class: {pred}, probs: [{probs[0]:.4f}, {probs[1]:.4f}]")

if __name__ == '__main__':
    main()

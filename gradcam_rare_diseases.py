"""
Grad-CAM Explainer for RFMiD_2 Rare Disease Model
Generates visual explanations for rare retinal disease predictions
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import pandas as pd

# Configuration
MODEL_PATH = "resnet50_merged_rfmid_model.pth"
DATA_DIR = Path("data/RFMiD_2")
METADATA_CSV = DATA_DIR / "rfmid2_preprocessed_metadata.csv"
OUTPUT_DIR = Path("gradcam_outputs_rfmid2")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disease mapping
DISEASE_CLASSES = [
    'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'CME', 'CNV', 'CRAO',
    'CRS', 'CRVO', 'CSC', 'CSR', 'CWS', 'DN', 'DR', 'EDN', 'ERM', 'GRT',
    'HPED', 'HR', 'HTN', 'IIH', 'LS', 'MCA', 'ME', 'MH', 'MHL', 'MS',
    'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'OPDM', 'PRH', 'RD', 'RHL', 'RP',
    'RPEC', 'RS', 'RT', 'RTR', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 'VS', 'WNL'
]

DISEASE_NAMES = {
    'AH': 'Arteriolar Narrowing',
    'AION': 'Anterior Ischemic Optic Neuropathy',
    'ARMD': 'Age-Related Macular Degeneration',
    'BRVO': 'Branch Retinal Vein Occlusion',
    'CB': "Coats Disease",
    'CF': 'Chorioretinal Folds',
    'CL': 'Central Artery Ischemia',
    'CME': 'Cystoid Macular Edema',
    'CNV': 'Choroidal Neovascularization',
    'CRAO': 'Central Retinal Artery Occlusion',
    'CRS': 'Central Serous (Chronic)',
    'CRVO': 'Central Retinal Vein Occlusion',
    'CSC': 'Central Serous Chorioretinopathy',
    'CSR': 'Central Serous Retinopathy',
    'CWS': 'Cotton Wool Spots',
    'DN': 'Drusen',
    'DR': 'Diabetic Retinopathy',
    'EDN': 'Epiretinal Membrane with Drusen',
    'ERM': 'Epiretinal Membrane',
    'GRT': 'Giant Retinal Tear',
    'HPED': 'Hemorrhagic PED',
    'HR': 'Retinal Hemorrhage',
    'HTN': 'Hypertensive Retinopathy',
    'IIH': 'Intracranial Hypertension',
    'LS': 'Laser Scars',
    'MCA': 'Macular Atrophy',
    'ME': 'Macular Edema',
    'MH': 'Macular Hole',
    'MHL': 'Macular Hole (Large)',
    'MS': 'Myelinated Nerve Fibers',
    'MYA': 'Myopia Changes',
    'ODC': 'Optic Disc Cupping',
    'ODE': 'Optic Disc Edema',
    'ODP': 'Optic Disc Pit',
    'ON': 'Optic Neuritis',
    'OPDM': 'Optic Disc Pallor',
    'PRH': 'Preretinal Hemorrhage',
    'RD': 'Retinal Detachment',
    'RHL': 'Retinal Hemorrhage (Layered)',
    'RP': 'Retinitis Pigmentosa',
    'RPEC': 'RPE Changes',
    'RS': 'Retinal Scar',
    'RT': 'Retinal Tear',
    'RTR': 'Recurrent Retinal Tear',
    'SOFE': 'Subretinal Fluid',
    'ST': 'Staphyloma',
    'TD': 'Tilted Disc',
    'TSLN': 'Tessellated Fundus',
    'TV': 'Temporal Pallor',
    'VS': 'Vitreous Syneresis',
    'WNL': 'Normal (WNL)'
}

RARE_CLASSES = set(DISEASE_CLASSES) - {'ARMD', 'DR', 'HTN', 'CME', 'ME', 'MH', 'MHL', 'ERM', 'CNV', 'WNL'}

class RareDiseaseResNet50(torch.nn.Module):
    def __init__(self, num_classes=51):
        super(RareDiseaseResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(self.model.fc.in_features, num_classes),
        )
        
    def forward(self, x):
        return self.model(x)

def load_model():
    """Load trained model"""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    model = RareDiseaseResNet50(num_classes=len(DISEASE_CLASSES))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """Load and preprocess image"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
    return img_resized, img_normalized, tensor

def predict_diseases(model, tensor, top_k=5):
    """Get top-K disease predictions"""
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    
    # Get top-K predictions
    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = []
    
    for idx in top_indices:
        code = DISEASE_CLASSES[idx]
        predictions.append({
            'code': code,
            'name': DISEASE_NAMES[code],
            'probability': probs[idx],
            'is_rare': code in RARE_CLASSES,
            'index': idx
        })
    
    return predictions, probs

def generate_gradcam(model, tensor, target_class_idx):
    """Generate Grad-CAM for specific class"""
    # Use last convolutional layer
    target_layers = [model.model.layer4[-1]]
    grad_cam = GradCAM(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(target_class_idx)]
    cam = grad_cam(input_tensor=tensor, targets=targets)
    
    return cam[0, :]

def create_visualization(img_array, img_normalized, predictions, cams, save_path):
    """Create comprehensive Grad-CAM visualization"""
    num_preds = len(predictions)
    fig, axes = plt.subplots(2, num_preds + 1, figsize=(4 * (num_preds + 1), 8))
    
    fig.suptitle("Rare Retinal Disease Grad-CAM Analysis", fontsize=16, fontweight='bold')
    
    # Original image (left column)
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original Fundus", fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].axis('off')
    info_text = "🔬 Top Predictions:\\n\\n"
    for i, pred in enumerate(predictions, 1):
        rare_marker = "⚠️ RARE" if pred['is_rare'] else "✓"
        info_text += f"{i}. {pred['name']}\\n"
        info_text += f"   {pred['probability']:.1%} {rare_marker}\\n\\n"
    
    axes[1, 0].text(0.05, 0.95, info_text, transform=axes[1, 0].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Grad-CAM for each prediction
    for i, (pred, cam) in enumerate(zip(predictions, cams), 1):
        # Overlay
        cam_viz = show_cam_on_image(img_normalized, cam, use_rgb=True)
        axes[0, i].imshow(cam_viz)
        title = f"{pred['code']}: {pred['name'][:20]}"
        if pred['is_rare']:
            title = f"⚠️ {title}"
        axes[0, i].set_title(title, fontsize=10, fontweight='bold')
        axes[0, i].axis('off')
        
        # Heatmap
        im = axes[1, i].imshow(cam, cmap='jet')
        axes[1, i].set_title(f"{pred['probability']:.1%}", fontsize=10)
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def analyze_rare_diseases():
    """Analyze sample rare disease images"""
    print("="*70)
    print("Grad-CAM Analysis for Rare Retinal Diseases")
    print("="*70)
    
    # Load model
    print("\\n🧠 Loading model...")
    model = load_model()
    
    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # Select rare disease samples
    rare_df = test_df[test_df['is_rare'] == 1]
    
    print(f"\\n📊 Analyzing {min(10, len(rare_df))} rare disease samples...")
    
    samples = rare_df.sample(n=min(10, len(rare_df)), random_state=42)
    
    for idx, row in samples.iterrows():
        img_path = Path("data") / row['preprocessed_path'] if not Path(row['preprocessed_path']).is_absolute() else row['preprocessed_path']
        
        print(f"\\n  Analyzing: {row['primary_disease']} - {DISEASE_NAMES.get(row['primary_disease'], 'Unknown')}")
        
        # Preprocess
        img_array, img_normalized, tensor = preprocess_image(img_path)
        
        # Predict
        predictions, all_probs = predict_diseases(model, tensor, top_k=5)
        
        # Generate Grad-CAMs
        cams = []
        for pred in predictions:
            cam = generate_gradcam(model, tensor, pred['index'])
            cams.append(cam)
        
        # Save visualization
        save_path = OUTPUT_DIR / f"{row['primary_disease']}_{row['image_name']}.png"
        create_visualization(img_array, img_normalized, predictions, cams, save_path)
    
    print("\\n" + "="*70)
    print(f"✅ Analysis complete! Outputs saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    analyze_rare_diseases()

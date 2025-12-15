"""
OculoXplain - Rare Disease Detection Page
51-class rare retinal disease classification with explainability
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from datetime import datetime
import json

# Disease mapping
DISEASE_CLASSES = [
    'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'CME', 'CNV', 'CRAO',
    'CRS', 'CRVO', 'CSC', 'CSR', 'CWS', 'DN', 'DR', 'EDN', 'ERM', 'GRT',
    'HPED', 'HR', 'HTN', 'IIH', 'LS', 'MCA', 'ME', 'MH', 'MHL', 'MS',
    'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'OPDM', 'PRH', 'RD', 'RHL', 'RP',
    'RPEC', 'RS', 'RT', 'RTR', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 'VS', 'WNL'
]

DISEASE_NAMES = {
    'AH': 'Arteriolar Narrowing', 'AION': 'Anterior Ischemic Optic Neuropathy',
    'ARMD': 'Age-Related Macular Degeneration', 'BRVO': 'Branch Retinal Vein Occlusion',
    'CB': "Coats Disease", 'CF': 'Chorioretinal Folds', 'CL': 'Central Artery Ischemia',
    'CME': 'Cystoid Macular Edema', 'CNV': 'Choroidal Neovascularization',
    'CRAO': 'Central Retinal Artery Occlusion', 'CRS': 'Central Serous (Chronic)',
    'CRVO': 'Central Retinal Vein Occlusion', 'CSC': 'Central Serous Chorioretinopathy',
    'CSR': 'Central Serous Retinopathy', 'CWS': 'Cotton Wool Spots', 'DN': 'Drusen',
    'DR': 'Diabetic Retinopathy', 'EDN': 'Epiretinal Membrane with Drusen',
    'ERM': 'Epiretinal Membrane', 'GRT': 'Giant Retinal Tear', 'HPED': 'Hemorrhagic PED',
    'HR': 'Retinal Hemorrhage', 'HTN': 'Hypertensive Retinopathy',
    'IIH': 'Intracranial Hypertension', 'LS': 'Laser Scars', 'MCA': 'Macular Atrophy',
    'ME': 'Macular Edema', 'MH': 'Macular Hole', 'MHL': 'Macular Hole (Large)',
    'MS': 'Myelinated Nerve Fibers', 'MYA': 'Myopia Changes', 'ODC': 'Optic Disc Cupping',
    'ODE': 'Optic Disc Edema', 'ODP': 'Optic Disc Pit', 'ON': 'Optic Neuritis',
    'OPDM': 'Optic Disc Pallor', 'PRH': 'Preretinal Hemorrhage', 'RD': 'Retinal Detachment',
    'RHL': 'Retinal Hemorrhage (Layered)', 'RP': 'Retinitis Pigmentosa', 'RPEC': 'RPE Changes',
    'RS': 'Retinal Scar', 'RT': 'Retinal Tear', 'RTR': 'Recurrent Retinal Tear',
    'SOFE': 'Subretinal Fluid', 'ST': 'Staphyloma', 'TD': 'Tilted Disc',
    'TSLN': 'Tessellated Fundus', 'TV': 'Temporal Pallor', 'VS': 'Vitreous Syneresis',
    'WNL': 'Normal (WNL)'
}

RARE_CLASSES = set(DISEASE_CLASSES) - {'ARMD', 'DR', 'HTN', 'CME', 'ME', 'MH', 'MHL', 'ERM', 'CNV', 'WNL'}

class RareDiseasesModel(nn.Module):
    def __init__(self, num_classes=51):
        super(RareDiseasesModel, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_rare_disease_model(model_path="../mobilenet_rfmid2_quick_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = RareDiseasesModel(num_classes=51)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        st.success(f"✅ Rare disease model loaded (51 classes)")
        return model, device
    except Exception as e:
        st.error(f"Error loading rare disease model: {e}")
        return None, device

def preprocess_image_rare(image_pil, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image_pil).unsqueeze(0).to(device)
    return tensor

def predict_rare_diseases(model, device, image_tensor, top_k=10):
    if model is None:
        return None, None
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs)[0].cpu().numpy()
    
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

def generate_gradcam_rare(model, image_tensor, device, target_class):
    if model is None:
        return None
    
    target_layers = [model.model.features[-1]]
    grad_cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    cam = grad_cam(input_tensor=image_tensor, targets=targets)
    return cam[0, :]

def page_rare_disease_analysis():
    st.markdown('<h1 class="main-header">🔬 Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown("Advanced 51-class retinal disease classification system")
    st.markdown("---")
    
    model, device = load_rare_disease_model()
    if model is None:
        st.error("❌ Failed to load rare disease model")
        return
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<h3 class="sub-header">📤 Upload Fundus Image</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select retinal fundus image", type=['jpg', 'jpeg', 'png'], key="rare_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            st.markdown('<h3 class="sub-header">🔧 Analysis Options</h3>', unsafe_allow_html=True)
            top_k = st.slider("Top predictions to show", 5, 15, 10)
            show_gradcam = st.checkbox("📊 Generate Grad-CAM", value=True, key="rare_gradcam")
            show_all = st.checkbox("📈 Show all 51 classes", value=False, key="rare_all")
            
            if st.button("🚀 Analyze for Rare Diseases", key="btn_analyze_rare", use_container_width=True):
                st.session_state.analyze_rare = True
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 Rare Disease Analysis</h3>', unsafe_allow_html=True)
        
        if uploaded_file is not None and st.session_state.get("analyze_rare", False):
            with st.spinner("🔄 Analyzing for rare diseases..."):
                try:
                    img_array = np.array(image.resize((224, 224)))
                    img_normalized = img_array.astype(np.float32) / 255.0
                    img_tensor = preprocess_image_rare(image, device)
                    
                    predictions, all_probs = predict_rare_diseases(model, device, img_tensor, top_k=top_k)
                    
                    if predictions is None:
                        st.error("Prediction failed")
                        return
                    
                    # Rare disease alert
                    rare_count = sum(1 for p in predictions[:5] if p['is_rare'])
                    if rare_count >= 3:
                        st.markdown(
                            f"""<div class="warning-box">
                            <h4>⚠️ Rare Disease Alert</h4>
                            <p><strong>{rare_count} of top 5 predictions are RARE diseases</strong></p>
                            <p>Specialist consultation recommended</p>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    
                    # Top predictions
                    st.markdown(f"#### 🎯 Top {top_k} Predictions")
                    
                    for i, pred in enumerate(predictions, 1):
                        col_rank, col_info = st.columns([0.15, 0.85])
                        
                        with col_rank:
                            rank_color = '#dc3545' if pred['is_rare'] else '#28a745'
                            st.markdown(f"<h3 style='color: {rank_color}; text-align: center;'>#{i}</h3>", unsafe_allow_html=True)
                        
                        with col_info:
                            st.progress(float(pred['probability']))
                            if pred['is_rare']:
                                st.markdown(f"**⚠️ {pred['name']}** ({pred['code']}) - RARE")
                            else:
                                st.markdown(f"**{pred['name']}** ({pred['code']})")
                            st.caption(f"Probability: {pred['probability']:.2%}")
                    
                    # All classes distribution
                    if show_all:
                        st.markdown("#### 📊 All 51 Disease Classes")
                        
                        fig, ax = plt.subplots(figsize=(12, 14))
                        names = [DISEASE_NAMES[c] for c in DISEASE_CLASSES]
                        colors = ['#dc3545' if c in RARE_CLASSES else '#28a745' for c in DISEASE_CLASSES]
                        
                        bars = ax.barh(names, all_probs, color=colors, alpha=0.7)
                        ax.set_xlabel("Probability", fontsize=10)
                        ax.set_xlim([0, 1])
                        ax.tick_params(axis='y', labelsize=7)
                        ax.grid(axis='x', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Grad-CAM
                    if show_gradcam and len(predictions) >= 3:
                        st.markdown("#### 🔬 Grad-CAM Explanations")
                        with st.spinner("Generating visual explanations..."):
                            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                            fig.suptitle("Rare Disease Grad-CAM Analysis", fontsize=14, fontweight='bold')
                            
                            axes[0, 0].imshow(img_array)
                            axes[0, 0].set_title("Original", fontweight='bold')
                            axes[0, 0].axis('off')
                            
                            for i, pred in enumerate(predictions[:2], 1):
                                cam = generate_gradcam_rare(model, img_tensor, device, pred['index'])
                                if cam is not None:
                                    cam_viz = show_cam_on_image(img_normalized, cam, use_rgb=True)
                                    axes[0, i].imshow(cam_viz)
                                    title = f"{pred['code']}: {pred['name'][:20]}"
                                    if pred['is_rare']:
                                        title = f"⚠️ {title}"
                                    axes[0, i].set_title(title, fontsize=10, fontweight='bold')
                                    axes[0, i].axis('off')
                            
                            axes[1, 0].axis('off')
                            info_text = "🔍 TOP PREDICTIONS:\n\n"
                            for i, pred in enumerate(predictions[:5], 1):
                                rare_marker = "⚠️" if pred['is_rare'] else "✓"
                                info_text += f"{i}. {pred['name']}\n   {pred['probability']:.1%} {rare_marker}\n\n"
                            
                            axes[1, 0].text(0.05, 0.95, info_text, transform=axes[1, 0].transAxes,
                                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                            
                            for i in range(1, 3):
                                pred = predictions[i-1]
                                cam = generate_gradcam_rare(model, img_tensor, device, pred['index'])
                                if cam is not None:
                                    axes[1, i].imshow(cam, cmap='jet')
                                    axes[1, i].set_title(f"{pred['probability']:.1%}", fontsize=10)
                                    axes[1, i].axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Download report
                    st.markdown("---")
                    report_data = {
                        'timestamp': datetime.now().isoformat(),
                        'image': uploaded_file.name,
                        'model': '51-class rare disease detector',
                        'rare_disease_alert': rare_count >= 3,
                        'top_predictions': [
                            {
                                'rank': i+1,
                                'code': p['code'],
                                'name': p['name'],
                                'probability': float(p['probability']),
                                'is_rare': p['is_rare']
                            }
                            for i, p in enumerate(predictions)
                        ]
                    }
                    
                    st.download_button(
                        label="📥 Download Full Report (JSON)",
                        data=json.dumps(report_data, indent=2),
                        file_name=f"rare_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
        else:
            st.markdown("""
            <div class="info-box">
            <h4>📋 Rare Disease Detection System</h4>
            <p><strong>51 retinal disease classes</strong> including:</p>
            <ul>
                <li>41 RARE diseases (68.93% coverage)</li>
                <li>10 common diseases for comparison</li>
            </ul>
            <p><strong>Featured rare conditions:</strong></p>
            <ul>
                <li>🔬 Retinitis Pigmentosa (RP)</li>
                <li>🔬 Giant Retinal Tear (GRT)</li>
                <li>🔬 Coats Disease (CB)</li>
                <li>🔬 Central Retinal Artery Occlusion (CRAO)</li>
                <li>🔬 And 37 more rare diseases!</li>
            </ul>
            <p><strong>How to use:</strong></p>
            <ol>
                <li>Upload a fundus image</li>
                <li>Select number of top predictions</li>
                <li>Click "Analyze for Rare Diseases"</li>
                <li>Review predictions and Grad-CAM explanations</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

# Export function
__all__ = ['page_rare_disease_analysis']

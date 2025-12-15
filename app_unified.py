# app_unified.py
"""
OculoXplain - Unified Web Interface
Integrated system for retinal disease classification and explainability
Author: OculoXplain Team
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import io
from datetime import datetime
import json

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="OculoXplain - Retinal Disease AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 1.5rem;
        font-weight: bold;
    }
    .info-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #b3dfe8 100%);
        border-left: 4px solid #1f77b4;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #98c68f 100%);
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
        border-left: 4px solid #ffc107;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5a9b8 100%);
        border-left: 4px solid #dc3545;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 3px solid #1f77b4;
    }
    .button-group {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CACHE DECORATORS ====================
@st.cache_resource
def load_binary_model(model_path="resnet50_retinal_disease_model.pth"):
    """Load the binary classification model (Healthy vs Disease)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success(f"✅ Binary model loaded successfully from {model_path}")
        else:
            st.warning(f"⚠️ Model file not found at {model_path}. Using pretrained base model.")
        
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading binary model: {e}")
        return None, device

@st.cache_resource
def load_multiclass_model(model_path="quick_model.pth"):
    """Load the multi-class classification model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, 8)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success(f"✅ Multi-class model loaded successfully from {model_path}")
        else:
            st.warning(f"⚠️ Model file not found at {model_path}. Using pretrained base model.")
        
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading multi-class model: {e}")
        return None, device

# ==================== HELPER FUNCTIONS ====================

def preprocess_image(image_source, size=(224, 224)):
    """Load and preprocess image from file path or PIL Image"""
    try:
        if isinstance(image_source, str):
            img = cv2.imread(str(image_source))
            if img is None:
                raise ValueError(f"Could not load image: {image_source}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(image_source)
        
        img_resized = cv2.resize(img, size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_resized, img_normalized
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

def get_image_tensor(image_pil, device):
    """Convert PIL image to tensor"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image_pil).unsqueeze(0)
    return tensor.to(device)

def predict_binary(model, device, image_tensor):
    """Binary classification prediction"""
    if model is None:
        return None, None
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0].cpu().numpy()
    
    return prediction, confidence

def predict_multiclass(model, device, image_tensor):
    """Multi-class prediction with top-3 results"""
    if model is None:
        return None, None
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence = probs[0].cpu().numpy()
    
    top_3_idx = np.argsort(confidence)[::-1][:3]
    
    label_map = {0: 'A', 1: 'C', 2: 'D', 3: 'G', 4: 'H', 5: 'M', 6: 'N', 7: 'O'}
    disease_names = {
        'N': 'Normal', 'D': 'Diabetic Retinopathy', 'G': 'Glaucoma',
        'C': 'Cataract', 'A': 'AMD', 'H': 'Hypertensive Retinopathy',
        'M': 'Myopia', 'O': 'Other'
    }
    
    results = []
    for idx in top_3_idx:
        code = label_map[idx]
        results.append({
            'code': code,
            'name': disease_names.get(code, code),
            'confidence': confidence[idx],
            'index': idx
        })
    
    return results, confidence

def generate_gradcam(model, image_tensor, device, target_class=None):
    """Generate Grad-CAM heatmap"""
    if model is None:
        return None
    
    target_layers = [model.layer4[-1]]
    grad_cam = GradCAM(model=model, target_layers=target_layers)
    
    if target_class is None:
        with torch.no_grad():
            outputs = model(image_tensor)
            target_class = torch.argmax(outputs, dim=1).item()
    
    targets = [ClassifierOutputTarget(target_class)]
    cam = grad_cam(input_tensor=image_tensor, targets=targets)
    
    return cam[0, :]

def create_download_report(prediction, confidence, image_name, model_type):
    """Create downloadable JSON report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'image_name': image_name,
        'model_type': model_type,
        'prediction': prediction,
        'confidence': float(confidence),
        'device': 'GPU' if torch.cuda.is_available() else 'CPU'
    }
    return json.dumps(report, indent=2)

# ==================== PAGES ====================

def page_home():
    """Home/Dashboard page"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">👁️ OculoXplain</h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 1.2rem; color: #666; margin-top: -1rem;">Explainable AI for Retinal Disease Detection</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("")
        st.markdown("")
        if torch.cuda.is_available():
            st.success(f"🚀 GPU Ready: {torch.cuda.get_device_name(0)}")
        else:
            st.info("💻 CPU Mode")
    
    # Quick start section
    st.markdown('<h3 class="sub-header">🚀 Quick Start</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<h3 class="sub-header"> Quick Start</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("###  Binary Classification")
            st.markdown("Detect Healthy vs Disease")
            if st.button("Start Analysis", key="quick_binary", use_container_width=True):
                st.session_state.page = "binary"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("###  Multi-Class Analysis")
            st.markdown("Identify 8 disease types")
            if st.button("Start Analysis", key="quick_multi", use_container_width=True):
                st.session_state.page = "multiclass"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #999; font-size: 0.9rem;">
    OculoXplain v1.0 | Explainable AI for Retinal Disease Detection<br>
    ⚠️ For research purposes only | Not for clinical diagnosis
    </p>
    """, unsafe_allow_html=True)

def page_binary_classification():
    """Binary classification page"""
    st.markdown('<h1 class="main-header">🔍 Binary Classification</h1>', unsafe_allow_html=True)
    st.markdown("Analyze fundus images to detect disease presence")
    
    st.markdown("---")
    
    # Load model
    model, device = load_binary_model()
    
    if model is None:
        st.error("❌ Failed to load binary model")
        return
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<h3 class="sub-header">📤 Upload Image</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a fundus image", type=['jpg', 'jpeg', 'png'], key="binary_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Fundus Image", use_container_width=True)
            
            # Save temporarily
            temp_path = f"temp_binary_{uploaded_file.name}"
            image.save(temp_path)
            
            st.markdown('<h3 class="sub-header">🔧 Analysis Options</h3>', unsafe_allow_html=True)
            
            show_gradcam = st.checkbox("📊 Generate Grad-CAM Explanation", value=True)
            show_confidence = st.checkbox("📈 Show Confidence Breakdown", value=True)
            
            if st.button("🚀 Analyze Image", key="analyze_binary", use_container_width=True):
                st.session_state.analyze_binary = True
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 Analysis Results</h3>', unsafe_allow_html=True)
        
        if uploaded_file is not None and st.session_state.get("analyze_binary", False):
            with st.spinner("🔄 Analyzing image..."):
                try:
                    # Preprocess
                    img_array, img_norm = preprocess_image(Image.open(temp_path))
                    if img_array is None:
                        st.error("Failed to preprocess image")
                        return
                    
                    img_tensor = get_image_tensor(Image.fromarray(img_array), device)
                    
                    # Predict
                    pred, conf = predict_binary(model, device, img_tensor)
                    
                    if pred is None:
                        st.error("Prediction failed")
                        return
                    
                    class_names = ['Healthy', 'Disease']
                    prediction_class = class_names[pred]
                    prediction_conf = conf[pred]
                    
                    # Display prediction
                    st.markdown(f"<div class='success-box' style='text-align: center;'>" 
                               f"<h2>Prediction: <strong>{prediction_class}</strong></h2>"
                               f"<h3>Confidence: <strong>{prediction_conf:.2%}</strong></h3>"
                               f"</div>", unsafe_allow_html=True)
                    
                    # Metrics
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Primary Class", prediction_class, delta=f"{prediction_conf:.2%}")
                    with col_m2:
                        other_class = class_names[1-pred]
                        st.metric("Other Class", other_class, delta=f"{conf[1-pred]:.2%}")
                    
                    # Confidence breakdown
                    if show_confidence:
                        st.markdown("#### 📈 Confidence Breakdown")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = ['#28a745' if i == pred else '#dc3545' for i in range(2)]
                        bars = ax.bar(class_names, conf, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                        ax.set_ylabel("Confidence Score", fontsize=12, fontweight='bold')
                        ax.set_ylim([0, 1])
                        
                        for i, (bar, v) in enumerate(zip(bars, conf)):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{v:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
                        
                        ax.grid(axis='y', alpha=0.3)
                        st.pyplot(fig)
                    
                    # Grad-CAM
                    if show_gradcam:
                        st.markdown("#### 🔬 Grad-CAM Explanation")
                        with st.spinner("Generating Grad-CAM heatmap..."):
                            cam = generate_gradcam(model, img_tensor, device)
                            
                            if cam is not None:
                                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                                fig.suptitle("Grad-CAM Analysis - Which regions influenced the prediction?", 
                                           fontsize=14, fontweight='bold')
                                
                                # Original
                                axes[0].imshow(img_array)
                                axes[0].set_title("Original Fundus Image", fontsize=12, fontweight='bold')
                                axes[0].axis('off')
                                
                                # Grad-CAM overlay
                                cam_viz = show_cam_on_image(img_norm, cam, use_rgb=True)
                                axes[1].imshow(cam_viz)
                                axes[1].set_title(f"Grad-CAM for {prediction_class}", fontsize=12, fontweight='bold')
                                axes[1].axis('off')
                                
                                # Heatmap
                                im = axes[2].imshow(cam, cmap='jet')
                                axes[2].set_title("Heatmap Intensity", fontsize=12, fontweight='bold')
                                axes[2].axis('off')
                                plt.colorbar(im, ax=axes[2])
                                
                                st.pyplot(fig)
                                
                                st.info("""
                                🔴 **Red/Warm regions**: Areas that support the predicted class
                                🔵 **Blue/Cool regions**: Areas that oppose the predicted class
                                """)
                    
                    # Download report
                    st.markdown("---")
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        report = create_download_report(prediction_class, prediction_conf, uploaded_file.name, "Binary")
                        st.download_button(
                            label="📥 Download Report (JSON)",
                            data=report,
                            file_name=f"binary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col_d2:
                        # Create image with results
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(img_array)
                        ax.set_title(f"Prediction: {prediction_class} ({prediction_conf:.2%})", 
                                   fontsize=16, fontweight='bold', pad=20)
                        ax.axis('off')
                        
                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format='png', bbox_inches='tight')
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label="🖼️ Download Image with Results",
                            data=img_buffer,
                            file_name=f"binary_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                        plt.close(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            st.markdown('<div class="info-box">'
                       '<h4>📋 Instructions</h4>'
                       '<ol>'
                       '<li>Upload a fundus image (JPG or PNG)</li>'
                       '<li>Select analysis options</li>'
                       '<li>Click "Analyze Image" button</li>'
                       '<li>View results and explanations</li>'
                       '<li>Download report if needed</li>'
                       '</ol>'
                       '</div>', unsafe_allow_html=True)

def page_multiclass_classification():
    """Multi-class classification page"""
    st.markdown('<h1 class="main-header">🏥 Multi-Class Disease Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Identify specific retinal diseases with detailed analysis")
    
    st.markdown("---")
    
    # Load model
    model, device = load_multiclass_model()
    
    if model is None:
        st.error("❌ Failed to load multi-class model")
        return
    
    disease_info = {
        'N': {'name': 'Normal', 'desc': 'Healthy retina', 'color': '#28a745'},
        'D': {'name': 'Diabetic Retinopathy', 'desc': 'Diabetes-related vessel damage', 'color': '#dc3545'},
        'G': {'name': 'Glaucoma', 'desc': 'Optic nerve damage', 'color': '#ffc107'},
        'C': {'name': 'Cataract', 'desc': 'Lens opacity', 'color': '#17a2b8'},
        'A': {'name': 'AMD', 'desc': 'Age-related macular degeneration', 'color': '#e83e8c'},
        'H': {'name': 'Hypertensive Retinopathy', 'desc': 'High blood pressure effects', 'color': '#6c757d'},
        'M': {'name': 'Myopia', 'desc': 'Nearsightedness complications', 'color': '#007bff'},
        'O': {'name': 'Other Diseases', 'desc': 'Various other conditions', 'color': '#fd7e14'}
    }
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<h3 class="sub-header">📤 Upload Image</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a fundus image", type=['jpg', 'jpeg', 'png'], key="multi_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Fundus Image", use_container_width=True)
            
            temp_path = f"temp_multi_{uploaded_file.name}"
            image.save(temp_path)
            
            st.markdown('<h3 class="sub-header">🔧 Analysis Options</h3>', unsafe_allow_html=True)
            
            show_gradcam = st.checkbox("📊 Show Grad-CAM for Top Predictions", value=True, key="multi_cam")
            show_all_classes = st.checkbox("📈 Show All Disease Classes", value=True, key="multi_all")
            
            if st.button("🚀 Analyze Diseases", key="analyze_multi", use_container_width=True):
                st.session_state.analyze_multi = True
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 Analysis Results</h3>', unsafe_allow_html=True)
        
        if uploaded_file is not None and st.session_state.get("analyze_multi", False):
            with st.spinner("🔄 Analyzing image for all disease types..."):
                try:
                    # Preprocess
                    img_array, img_norm = preprocess_image(Image.open(temp_path))
                    if img_array is None:
                        st.error("Failed to preprocess image")
                        return
                    
                    img_tensor = get_image_tensor(Image.fromarray(img_array), device)
                    
                    # Predict
                    predictions, conf = predict_multiclass(model, device, img_tensor)
                    
                    if predictions is None:
                        st.error("Prediction failed")
                        return

                    # Set up label map and rare/unknown detection
                    label_map = {0: 'A', 1: 'C', 2: 'D', 3: 'G', 4: 'H', 5: 'M', 6: 'N', 7: 'O'}
                    top_conf = float(predictions[0]['confidence']) if predictions else 0.0
                    second_conf = float(predictions[1]['confidence']) if len(predictions) > 1 else 0.0
                    margin = top_conf - second_conf
                    probs = np.clip(np.array(conf, dtype=np.float32), 1e-9, 1.0)
                    entropy = float(-(probs * np.log(probs)).sum())
                    rare_flag = (top_conf < 0.45) or (margin < 0.15) or (entropy > 1.2)
                    
                    if rare_flag:
                        st.markdown(
                            """
                            <div class="warning-box" style="border: 1px solid #f0ad4e; padding: 14px; border-radius: 10px; background: #fff3cd; color: #6b3b00; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                            <h4 style="margin: 0 0 6px 0;">⚠️ Potential Rare/Unknown Presentation</h4>
                            <p style="margin: 0 0 6px 0; font-weight: 600;">Confidence is low or predictions are ambiguous. Treat as <strong>rare/unknown</strong> and seek specialist review.</p>
                            <ul style="margin: 0 0 0 1rem; padding: 0;">
                                <li>Top confidence: {top:.1%}</li>
                                <li>Margin (top1 - top2): {gap:.1%}</li>
                                <li>Entropy: {entropy_val:.2f}</li>
                            </ul>
                            </div>
                            """.format(top=top_conf, gap=margin, entropy_val=entropy),
                            unsafe_allow_html=True
                        )
                    
                    # Top predictions
                    st.markdown("#### 🎯 Top 3 Predictions")
                    
                    for i, pred in enumerate(predictions, 1):
                        code = pred['code']
                        info = disease_info.get(code, {})
                        
                        col_num, col_pred = st.columns([0.8, 3], gap="small")
                        
                        with col_num:
                            st.markdown(f"<h3 style='color: {info.get('color', '#999')}; text-align: center; margin-top: 0.5rem;'>#{i}</h3>", 
                                       unsafe_allow_html=True)
                        
                        with col_pred:
                            st.progress(float(pred['confidence']))
                            st.markdown(f"**{info.get('name', code)}** ({code})")
                            st.caption(info.get('desc', ''))
                            st.markdown(f"**Confidence:** {pred['confidence']:.1%}")
                    
                    # All classes
                    if show_all_classes:
                        st.markdown("#### 📊 All Disease Classes")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        codes = [label_map[i] for i in range(8)]
                        names = [disease_info[c]['name'] for c in codes]
                        colors_list = [disease_info[c]['color'] for c in codes]
                        values = [conf[i] for i in range(8)]
                        
                        bars = ax.barh(names, values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
                        ax.set_xlabel("Confidence Score", fontsize=12, fontweight='bold')
                        ax.set_xlim([0, 1])
                        
                        for bar, val in zip(bars, values):
                            width = bar.get_width()
                            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                                   f'{val:.1%}', ha='left', va='center', fontsize=10, fontweight='bold')
                        
                        ax.grid(axis='x', alpha=0.3)
                        st.pyplot(fig)
                    
                    # Grad-CAM
                    if show_gradcam and len(predictions) >= 2:
                        st.markdown("#### 🔬 Grad-CAM Explanations")
                        with st.spinner("Generating Grad-CAM visualizations..."):
                            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                            fig.suptitle("Multi-Class Grad-CAM Analysis", fontsize=14, fontweight='bold')
                            
                            # Original
                            axes[0, 0].imshow(img_array)
                            axes[0, 0].set_title("Original Fundus Image", fontsize=12, fontweight='bold')
                            axes[0, 0].axis('off')
                            
                            # Top 2 Grad-CAMs
                            for i, pred in enumerate(predictions[:2]):
                                col = i + 1
                                cam = generate_gradcam(model, img_tensor, device, pred['index'])
                                
                                if cam is not None:
                                    cam_viz = show_cam_on_image(img_norm, cam, use_rgb=True)
                                    axes[0, col].imshow(cam_viz)
                                    
                                    code = pred['code']
                                    name = disease_info.get(code, {}).get('name', code)
                                    axes[0, col].set_title(f"{name}\n{pred['confidence']:.1%}", 
                                                         fontsize=11, fontweight='bold')
                                    axes[0, col].axis('off')
                            
                            # Summary
                            axes[1, 0].axis('off')
                            summary_text = "🔍 SUMMARY\n\n"
                            for i, pred in enumerate(predictions, 1):
                                code = pred['code']
                                name = disease_info[code]['name']
                                summary_text += f"{i}. {name} ({code})\n    {pred['confidence']:.1%}\n\n"
                            
                            axes[1, 0].text(0.05, 0.95, summary_text, transform=axes[1, 0].transAxes,
                                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                            
                            # Confidence chart
                            axes[1, 1].axis('off')
                            chart_text = "📈 CONFIDENCE\n\n"
                            for pred in predictions:
                                bar = "█" * int(pred['confidence'] * 20) + "░" * (20 - int(pred['confidence'] * 20))
                                chart_text += f"{pred['code']}: {bar} {pred['confidence']:.1%}\n"
                            
                            axes[1, 1].text(0.05, 0.95, chart_text, transform=axes[1, 1].transAxes,
                                          fontsize=9, verticalalignment='top', fontfamily='monospace',
                                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                            
                            # Interpretation
                            axes[1, 2].axis('off')
                            interp_text = "🎯 HEATMAP GUIDE\n\n"
                            interp_text += "🔴 RED (Warm):\n"
                            interp_text += "• Supporting prediction\n"
                            interp_text += "• Key features\n\n"
                            interp_text += "🔵 BLUE (Cool):\n"
                            interp_text += "• Opposing prediction\n"
                            interp_text += "• Less relevant\n\n"
                            interp_text += "📍 Focus Areas:\n"
                            interp_text += "• Optic disc\n"
                            interp_text += "• Macula\n"
                            interp_text += "• Vessels"
                            
                            axes[1, 2].text(0.05, 0.95, interp_text, transform=axes[1, 2].transAxes,
                                          fontsize=9, verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Download
                    st.markdown("---")
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        report_data = {
                            'timestamp': datetime.now().isoformat(),
                            'image': uploaded_file.name,
                            'rare_unknown_flag': bool(rare_flag),
                            'rare_metrics': {
                                'top_confidence': top_conf,
                                'margin_top2': margin,
                                'entropy': entropy
                            },
                            'top_3_predictions': [
                                {'rank': i+1, 'disease': p['name'], 'code': p['code'], 'confidence': float(p['confidence'])}
                                for i, p in enumerate(predictions)
                            ],
                            'all_confidences': {
                                disease_info[label_map[i]]['name']: float(conf[i])
                                for i in range(8)
                            }
                        }
                        report_json = json.dumps(report_data, indent=2)
                        st.download_button(
                            label="📥 Download Report (JSON)",
                            data=report_json,
                            file_name=f"multiclass_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col_d2:
                        # Placeholder download button; buffer stays empty until image saving is added
                        st.download_button(
                            label="🖼️ Download Analysis Image",
                            data=(img_buffer := io.BytesIO()),
                            file_name=f"multiclass_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            disabled=True
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            st.markdown('<div class="info-box">'
                       '<h4>📋 Instructions</h4>'
                       '<ol>'
                       '<li>Upload a fundus image (JPG or PNG)</li>'
                       '<li>Choose display options</li>'
                       '<li>Click "Analyze Diseases"</li>'
                       '<li>Review top 3 predictions</li>'
                       '<li>View Grad-CAM explanations</li>'
                       '<li>Download detailed report</li>'
                       '</ol>'
                       '</div>', unsafe_allow_html=True)

def page_about():
    """About/Documentation page"""
    st.markdown('<h1 class="main-header">📚 About OculoXplain</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "How to Use", "Technology", "Limitations"])
    
    with tab1:
        st.markdown("""
        ## 🔬 Explainable AI for Retinal Disease Detection
        
        OculoXplain is an intelligent system designed to assist healthcare professionals in detecting and 
        understanding retinal diseases through AI-powered analysis of fundus images.
        
        ### Key Features
        
        - **Binary Classification**: Quickly detect if a retina shows disease presence
        - **Multi-Class Analysis**: Identify specific disease types among 8 categories
        - **Explainability**: Grad-CAM visualizations show which retinal regions influenced predictions
        - **High Accuracy**: Built on ResNet50 with transfer learning from ImageNet
        - **Fast Processing**: GPU-accelerated analysis in seconds
        
        ### Supported Diseases
        
        1. **N - Normal**: Healthy retina
        2. **D - Diabetic Retinopathy**: Diabetes-related blood vessel damage
        3. **G - Glaucoma**: Optic nerve damage
        4. **C - Cataract**: Lens opacity
        5. **A - AMD**: Age-related macular degeneration
        6. **H - Hypertensive Retinopathy**: High blood pressure effects
        7. **M - Myopia**: Nearsightedness-related complications
        8. **O - Other**: Various other retinal conditions
        """)
    
    with tab2:
        st.markdown("""
        ## 🚀 How to Use OculoXplain
        
        ### Binary Classification Workflow
        
        1. **Upload Image**: Click on the upload button and select your fundus image (JPG/PNG)
        2. **Configure Options**: Choose whether to display Grad-CAM and confidence breakdowns
        3. **Analyze**: Click the "Analyze Image" button
        4. **Review Results**: 
           - Primary prediction (Healthy or Disease)
           - Confidence percentage
           - Confidence breakdown chart
           - Grad-CAM heatmap (optional)
        5. **Download**: Save the report or annotated image
        
        ### Multi-Class Analysis Workflow
        
        1. **Upload Image**: Select your fundus image
        2. **Select Options**: Enable all disease classes and Grad-CAM if desired
        3. **Analyze**: Click "Analyze Diseases"
        4. **Review Results**:
           - Top 3 disease predictions
           - All 8 disease class scores
           - Individual Grad-CAM for top predictions
           - Detailed confidence breakdown
        5. **Download**: Export detailed JSON report
        
        ### Image Requirements
        
        - **Format**: JPG or PNG
        - **Size**: Recommended 400×400 pixels or larger
        - **Quality**: Clear fundus photograph
        - **View**: Optic disc or macula should be visible
        """)
    
    with tab3:
        st.markdown("""
        ## 🔧 Technology Stack
        
        ### Model Architecture
        - **Base Model**: ResNet50 (pretrained on ImageNet)
        - **Transfer Learning**: Fine-tuned on ODIR-5K dataset
        - **Framework**: PyTorch
        - **Input Size**: 224×224 pixels
        
        ### Explainability Method
        - **Technique**: Grad-CAM (Gradient-weighted Class Activation Mapping)
        - **Purpose**: Localize important regions that influence predictions
        - **Visualization**: Heatmap overlay on original image
        - **Interpretation**: Warm colors = supporting prediction, Cool colors = opposing
        
        ### Dataset
        - **Primary Dataset**: ODIR-5K (7,000 fundus images)
        - **Classes**: 8 disease types + Normal
        - **Train/Val Split**: 80/20
        - **Augmentation**: Random rotation, flip, brightness adjustments
        
        ### Performance
        - **Binary Model Accuracy**: ~85-90%
        - **Multi-Class Accuracy**: ~75-80%
        - **Processing Time**: <2 seconds per image (GPU)
        - **GPU Support**: NVIDIA CUDA compatible
        
        ### Web Interface
        - **Framework**: Streamlit
        - **Deployment**: Can run locally or on cloud
        - **Responsiveness**: Mobile-friendly design
        """)
    
    with tab4:
        st.markdown("""
        ## ⚠️ Important Limitations
        
        ### Clinical Limitations
        
        1. **Not FDA Approved**: OculoXplain is for research purposes only
        2. **Not Diagnostic**: Cannot replace clinical diagnosis by qualified ophthalmologists
        3. **Supplementary Tool**: Should be used to support, not replace, expert judgment
        4. **Limited Scope**: Can only analyze fundus photographs, not comprehensive eye exams
        5. **Population Specific**: Trained primarily on certain demographic groups
        
        ### Technical Limitations
        
        1. **Image Quality Dependent**: Requires good-quality fundus images
        2. **Black Borders**: May fail on images with extensive black margins
        3. **Artifact Sensitivity**: Can be affected by media opacities, cataracts
        4. **Limited Context**: Analyzes single images, not patient history
        5. **Class Imbalance**: Some rare diseases may have lower accuracy
        
        ### Grad-CAM Limitations
        
        1. **Correlation, Not Causation**: Heatmaps show correlation, not proof of disease
        2. **Model-Specific**: Explanations are based on model internals, not clinical facts
        3. **Post-Hoc Explanation**: Generated after prediction, may not reflect actual reasoning
        4. **Adversarial Sensitivity**: Can be fooled by adversarial perturbations
        
        ### Recommendations
        
        ✅ **DO**:
        - Use as a screening tool for further investigation
        - Combine with clinical expertise
        - Document all results and analyses
        - Keep image metadata and reports
        
        ❌ **DON'T**:
        - Use for standalone clinical diagnosis
        - Replace comprehensive eye exams
        - Trust predictions on poor-quality images
        - Rely solely on AI without expert review
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #999; font-size: 0.9rem;">
    <strong>OculoXplain v1.0</strong><br>
    Explainable AI for Retinal Disease Detection<br>
    <em>Research Project - Not for Clinical Use</em><br>
    © 2024 | All Rights Reserved
    </p>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    if 'analyze_binary' not in st.session_state:
        st.session_state.analyze_binary = False
    
    if 'analyze_multi' not in st.session_state:
        st.session_state.analyze_multi = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🧭 Navigation")
        
        page_options = ["Home", "Binary Classification", "Multi-Class Analysis", "About"]
        page_map = {
            "Home": "home",
            "Binary Classification": "binary",
            "Multi-Class Analysis": "multiclass",
            "About": "about"
        }
        
        selected_page = st.radio("Select Analysis Type", page_options, 
                                index=["home", "binary", "multiclass", "about"].index(st.session_state.page))
        
        st.session_state.page = page_map[selected_page]
        
        # Quick links
        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        
        if st.button("🏠 Go to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        
        if st.button("📊 Binary Analysis", use_container_width=True):
            st.session_state.page = "binary"
            st.rerun()
        
        if st.button("🏥 Multi-Class Analysis", use_container_width=True):
            st.session_state.page = "multiclass"
            st.rerun()
        
        if st.button("📚 Documentation", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()
    
    # Route to selected page
    if st.session_state.page == "home":
        page_home()
    elif st.session_state.page == "binary":
        page_binary_classification()
    elif st.session_state.page == "multiclass":
        page_multiclass_classification()
    elif st.session_state.page == "about":
        page_about()

if __name__ == "__main__":
    main()
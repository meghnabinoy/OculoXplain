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
import re
from html import escape

# Import rare disease page
from rare_disease_page import page_rare_disease_analysis

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
        background: linear-gradient(135deg, #1f6fa3 0%, #0b4f6c 100%);
        color: #ffffff;
        border-left: 4px solid #1f77b4;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #b7dfc1 0%, #6fa96b 100%);
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #e6b800 0%, #cc9900 100%);
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
def load_binary_model(model_path="./resnet50_rfmid2_binary_model.pth"):
    """Load the binary classification model (Healthy vs Disease)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading binary model: {e}")
        return None, device

@st.cache_resource
def load_multiclass_model(model_path="../resnet50_multiclass_retinal_model.pth"):
    """Load the multi-class classification model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, 8)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        
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


def draw_localization_bbox(grayscale_cam, visualization_img, threshold=0.5,
                           color=(255, 255, 0), label="Focus Region"):
    """
    Draw a corner-bracket bounding box around the most activated GradCAM region.

    Args:
        grayscale_cam:     GradCAM heatmap (H x W, float32, 0-1)
        visualization_img: uint8 RGB image to annotate (H x W x 3)
        threshold:         Fraction of peak activation used to define the focus area
        color:             RGB colour tuple for the box and label
        label:             Text shown above the bounding box

    Returns:
        Annotated uint8 RGB image
    """
    cam_norm = grayscale_cam / (grayscale_cam.max() + 1e-8)
    binary_mask = (cam_norm >= threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return visualization_img.copy()

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    result = visualization_img.copy()

    # Thin full rectangle as backdrop
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 1)

    # Bold corner brackets (targeting-reticle style)
    arm = max(6, min(w, h) // 5)
    t = 2
    for (px, py, dx, dy) in [
        (x,     y,      1,  1),
        (x + w, y,     -1,  1),
        (x,     y + h,  1, -1),
        (x + w, y + h, -1, -1),
    ]:
        cv2.line(result, (px, py), (px + dx * arm, py), color, t + 1)
        cv2.line(result, (px, py), (px, py + dy * arm), color, t + 1)

    cv2.putText(result, label, (x, max(y - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return result


def create_download_report(prediction, confidence, image_name, model_type):
        """Create a downloadable, human-readable HTML report."""
        timestamp = datetime.now()
        confidence_pct = float(confidence) * 100.0

        if confidence_pct >= 90:
                confidence_band = "Very high"
        elif confidence_pct >= 75:
                confidence_band = "High"
        elif confidence_pct >= 60:
                confidence_band = "Moderate"
        else:
                confidence_band = "Low"

        report_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>OculoXplain Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
            line-height: 1.55;
            color: #1f2937;
            margin: 24px;
            background: #f8fafc;
        }}
        .card {{
            max-width: 820px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }}
        h1 {{
            margin: 0 0 6px 0;
            color: #0b4f6c;
            font-size: 1.7rem;
        }}
        .subtitle {{
            margin: 0 0 20px 0;
            color: #4b5563;
            font-size: 0.95rem;
        }}
        .result-box {{
            border-left: 5px solid #0b4f6c;
            background: #eef6fa;
            border-radius: 8px;
            padding: 14px 16px;
            margin: 14px 0 18px 0;
        }}
        .result-box strong {{
            color: #0b4f6c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 14px 0 20px 0;
        }}
        td {{
            border: 1px solid #e5e7eb;
            padding: 10px;
            vertical-align: top;
        }}
        td:first-child {{
            width: 35%;
            font-weight: 600;
            background: #f9fafb;
        }}
        .small {{
            color: #6b7280;
            font-size: 0.85rem;
        }}
        .warn {{
            margin-top: 14px;
            padding: 12px;
            border-radius: 8px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #7c2d12;
        }}
    </style>
</head>
<body>
    <div class=\"card\">
        <h1>OculoXplain Analysis Report</h1>
        <p class=\"subtitle\">Generated on {escape(timestamp.strftime('%Y-%m-%d %H:%M:%S'))}</p>

        <div class=\"result-box\">
            <div><strong>Prediction:</strong> {escape(prediction)}</div>
            <div><strong>Confidence:</strong> {confidence_pct:.2f}% ({confidence_band})</div>
        </div>

        <h3>Case Summary</h3>
        <table>
            <tr><td>Image name</td><td>{escape(image_name)}</td></tr>
            <tr><td>Analysis type</td><td>{escape(model_type)} classification</td></tr>
            <tr><td>Inference hardware</td><td>{'GPU' if torch.cuda.is_available() else 'CPU'}</td></tr>
            <tr><td>Report ID</td><td>OX-{escape(timestamp.strftime('%Y%m%d-%H%M%S'))}</td></tr>
        </table>

        <h3>Interpretation Notes</h3>
        <p>
            This report presents the model output for the uploaded fundus image and the model confidence for the predicted class.
            Confidence reflects model certainty, not confirmed diagnosis.
        </p>

        <div class=\"warn\">
            <strong>Important:</strong> This tool is for research and decision support only.
            It should not be used as a standalone clinical diagnosis.
        </div>

        <p class=\"small\">Tip: This HTML report can be opened in any browser and printed/saved as PDF from the browser print dialog.</p>
    </div>
</body>
</html>
"""

        return report_html


def _safe_read_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _parse_training_acc_from_log(log_path):
    if not os.path.exists(log_path):
        return None

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        matches = re.findall(r"Epoch\s+\d+/\d+\s+\|\s+train_loss=.*?train_acc=([0-9.]+)", text)
        if matches:
            return float(matches[-1]) * 100.0
    except Exception:
        return None
    return None


def _extract_report_metrics(metrics_json):
    """Extract macro/weighted precision-recall-f1 metrics from classification report."""
    out = {
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "weighted_precision": None,
        "weighted_recall": None,
        "weighted_f1": None,
    }
    if not metrics_json:
        return out

    report = metrics_json.get("classification_report", {})
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    try:
        if "precision" in macro:
            out["macro_precision"] = float(macro["precision"]) * 100.0
        if "recall" in macro:
            out["macro_recall"] = float(macro["recall"]) * 100.0
        if "f1-score" in macro:
            out["macro_f1"] = float(macro["f1-score"]) * 100.0
        if "precision" in weighted:
            out["weighted_precision"] = float(weighted["precision"]) * 100.0
        if "recall" in weighted:
            out["weighted_recall"] = float(weighted["recall"]) * 100.0
        if "f1-score" in weighted:
            out["weighted_f1"] = float(weighted["f1-score"]) * 100.0
    except Exception:
        pass

    return out


def get_current_validation_metrics():
    """Load current validation metrics for all expected variants."""
    variants = [
        {
            "label": "ResNet50 (Pretrained)",
            "metrics_file": "resnet50_merged_rfmid_metrics.json",
            "model_file": "resnet50_merged_rfmid_model.pth",
            "size_mb": 94.0,
            "fallback": {
                "train_accuracy": 86.82,
                "val_accuracy": 81.57,
                "test_accuracy": 82.96,
                "best_epoch": 16,
                "best_val_macro_f1": 80.88,
                "test_macro_f1": 81.92,
                "test_loss": 1.3570,
            },
        },
        {
            "label": "MobileNetV2 (Pretrained)",
            "metrics_file": "mobilenetv2_merged_rfmid_pretrained_metrics.json",
            "model_file": "mobilenetv2_merged_rfmid_pretrained_model.pth",
            "size_mb": 14.0,
            "fallback": {
                "train_accuracy": 86.79,
                "val_accuracy": 83.33,
                "test_accuracy": 82.54,
                "best_epoch": 12,
                "best_val_macro_f1": 82.07,
                "test_macro_f1": 81.31,
                "test_loss": 1.4289,
            },
        },
    ]

    out_variants = []
    dataset = {
        "classes": 51,
        "train": 3058,
        "val": 624,
        "test": 716,
    }

    for item in variants:
        js = _safe_read_json(item["metrics_file"])
        fallback = item["fallback"] or {}

        row = {
            "label": item["label"],
            "metrics_file": item["metrics_file"],
            "model_file": item["model_file"],
            "size_mb": item["size_mb"],
            "available": js is not None,
            "train_accuracy": fallback.get("train_accuracy"),
            "val_accuracy": fallback.get("val_accuracy"),
            "test_accuracy": fallback.get("test_accuracy"),
            "best_epoch": fallback.get("best_epoch"),
            "best_val_macro_f1": fallback.get("best_val_macro_f1"),
            "test_macro_f1": fallback.get("test_macro_f1"),
            "test_loss": fallback.get("test_loss"),
            "classification_report": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_precision": None,
            "weighted_recall": None,
            "weighted_f1": None,
        }

        if js:
            row["best_epoch"] = int(js.get("best_epoch", row["best_epoch"] or 0))
            row["best_val_macro_f1"] = float(js.get("best_val_macro_f1", 0.0)) * 100.0
            row["test_macro_f1"] = float(js.get("test_macro_f1", 0.0)) * 100.0
            row["test_accuracy"] = float(js.get("test_accuracy", 0.0)) * 100.0
            row["test_loss"] = float(js.get("test_loss", row["test_loss"] or 0.0))
            row["classification_report"] = js.get("classification_report")
            row.update(_extract_report_metrics(js))

            dataset["classes"] = int(js.get("num_classes", dataset["classes"]))
            dataset["train"] = int(js.get("num_train", dataset["train"]))
            dataset["val"] = int(js.get("num_val", dataset["val"]))
            dataset["test"] = int(js.get("num_test", dataset["test"]))

        out_variants.append(row)

    resnet_train_from_log = _parse_training_acc_from_log("outputs/train_full.log")
    for v in out_variants:
        if v["label"].startswith("ResNet50") and resnet_train_from_log is not None:
            v["train_accuracy"] = resnet_train_from_log

    dataset["total_images"] = dataset["train"] + dataset["val"] + dataset["test"]

    available_variants = [v for v in out_variants if v["available"] and v["test_macro_f1"] is not None]
    selected_label = None
    if available_variants:
        best = max(available_variants, key=lambda x: (x["test_macro_f1"], x["test_accuracy"] or 0.0))
        selected_label = best["label"]

    return {
        "variants": out_variants,
        "dataset": dataset,
        "selected_label": selected_label,
    }

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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Binary Classification")
        st.markdown("Detect Healthy vs Disease")
        if st.button("Start Analysis", key="quick_binary", use_container_width=True):
            st.session_state.page = "binary"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Disease Detection")
        st.markdown("Analyze 51 retinal diseases")
        if st.button("Start Analysis", key="quick_disease", use_container_width=True):
            st.session_state.page = "disease"
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
            
            if st.button("🚀 Analyze Image", key="btn_analyze_binary", use_container_width=True):
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
                                # Draw localization bounding box on the most important region
                                cam_viz = draw_localization_bbox(
                                    cam, cam_viz,
                                    label=f"Focus: {prediction_class}"
                                )
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
                            label="📥 Download Readable Report (HTML)",
                            data=report,
                            file_name=f"binary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
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
        5. **Download**: Export a readable report (HTML, printable as PDF)
        
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


def page_system_validation():
    """System validation page with evidence-focused tabs."""
    st.markdown('<h1 class="main-header">📊 System Validation & Model Comparison</h1>', unsafe_allow_html=True)

    data = get_current_validation_metrics()
    variants = data["variants"]
    dataset = data["dataset"]
    selected_label = data["selected_label"]

    model_map = {v["label"]: v for v in variants}
    selected = model_map.get(selected_label) if selected_label else None

    tab1, tab2, tab3 = st.tabs([
        "🏆 Model Comparison",
        "🔍 Why This Model",
        "✅ System Validation",
    ])

    with tab1:
        st.markdown("## Comparison of Trained Model Variants")
        st.markdown(
            f"All models were evaluated on the merged RFMiD dataset ({dataset['classes']} classes, "
            f"{dataset['train']} train / {dataset['val']} val / {dataset['test']} test)."
        )

        rows = []
        for v in variants:
            rows.append({
                "Model": v["label"],
                "Available": "Yes" if v["available"] else "No",
                "Train Accuracy (%)": round(v["train_accuracy"], 2) if v["train_accuracy"] is not None else np.nan,
                "Test Accuracy (%)": round(v["test_accuracy"], 2) if v["test_accuracy"] is not None else np.nan,
                "Test Macro F1": round((v["test_macro_f1"] or 0.0) / 100.0, 4) if v["test_macro_f1"] is not None else np.nan,
                "Best Val Macro F1": round((v["best_val_macro_f1"] or 0.0) / 100.0, 4) if v["best_val_macro_f1"] is not None else np.nan,
                "Best Epoch": v["best_epoch"] if v["best_epoch"] is not None else "—",
                "Test Loss": round(v["test_loss"], 4) if v["test_loss"] is not None else np.nan,
            })

        df = pd.DataFrame(rows)

        def _highlight_selected(row):
            if selected_label and row["Model"] == selected_label:
                return ["background-color: #0b4f6c; color: #ffffff; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(df.style.apply(_highlight_selected, axis=1), use_container_width=True, hide_index=True)

        plot_rows = [r for r in rows if pd.notna(r["Test Accuracy (%)"]) and pd.notna(r["Test Macro F1"])]
        if plot_rows:
            model_names = [r["Model"] for r in plot_rows]
            accuracies = [r["Test Accuracy (%)"] for r in plot_rows]
            macro_f1s = [r["Test Macro F1"] for r in plot_rows]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

            bars1 = axes[0].bar(range(len(model_names)), accuracies, color=colors[:len(model_names)])
            axes[0].set_title("Test Accuracy (%)", fontweight="bold")
            axes[0].set_ylabel("Accuracy (%)")
            axes[0].set_xticks(range(len(model_names)))
            axes[0].set_xticklabels([n.replace(" (", "\n(") for n in model_names], fontsize=8)
            axes[0].set_ylim(0, 100)
            for b, v in zip(bars1, accuracies):
                axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.6, f"{v:.2f}", ha="center", fontsize=8)

            bars2 = axes[1].bar(range(len(model_names)), macro_f1s, color=colors[:len(model_names)])
            axes[1].set_title("Test Macro F1", fontweight="bold")
            axes[1].set_ylabel("Macro F1")
            axes[1].set_xticks(range(len(model_names)))
            axes[1].set_xticklabels([n.replace(" (", "\n(") for n in model_names], fontsize=8)
            axes[1].set_ylim(0, 1.0)
            for b, v in zip(bars2, macro_f1s):
                axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{v:.3f}", ha="center", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        conv_points = [v for v in variants if v["best_epoch"] is not None and v["best_val_macro_f1"] is not None]
        if conv_points:
            st.markdown("### Training Convergence")
            fig2, ax2 = plt.subplots(figsize=(8, 3.8))
            ax2.set_title("Best Validation Macro F1 vs Convergence Epoch", fontweight="bold")
            ax2.set_xlabel("Best Epoch")
            ax2.set_ylabel("Best Validation Macro F1")
            colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
            for i, v in enumerate(conv_points):
                y_val = (v["best_val_macro_f1"] or 0.0) / 100.0
                ax2.scatter(v["best_epoch"], y_val, s=100, color=colors[i % len(colors)], label=v["label"])
                ax2.annotate(f"Ep{v['best_epoch']} {y_val:.3f}", (v["best_epoch"], y_val), fontsize=8)
            ax2.legend(fontsize=8, loc="lower right")
            st.pyplot(fig2)
            plt.close(fig2)

    with tab2:
        st.markdown("## Why This Model")
        if selected is None:
            st.write("No complete metrics file was found to select a best model.")
        else:
            alt_models = [v for v in variants if v["label"] != selected["label"]]
            c1, c2 = st.columns(2)
            for idx, col in enumerate([c1, c2]):
                if idx < len(alt_models):
                    alt = alt_models[idx]
                    if alt["test_accuracy"] is not None and alt["test_macro_f1"] is not None:
                        acc_delta = selected["test_accuracy"] - alt["test_accuracy"]
                        f1_delta = ((selected["test_macro_f1"] or 0.0) - (alt["test_macro_f1"] or 0.0)) / 100.0
                        col.metric(
                            f"Gain vs {alt['label']}",
                            f"{acc_delta:+.2f}% acc",
                            f"{f1_delta:+.3f} macro F1",
                        )
                    else:
                        col.metric(f"Gain vs {alt['label']}", "N/A", "metrics file missing")

            rationale_rows = []
            criteria = [
                ("Test Accuracy", "test_accuracy", True),
                ("Test Macro F1", "test_macro_f1", False),
                ("Best Val Macro F1", "best_val_macro_f1", False),
                ("Convergence Epoch", "best_epoch", False),
                ("Test Loss", "test_loss", False),
                ("Model Size (MB)", "size_mb", False),
            ]

            for criterion, key, is_pct in criteria:
                row = {"Criterion": criterion}
                for v in variants:
                    val = v.get(key)
                    if val is None:
                        row[v["label"]] = "N/A"
                    else:
                        if key in ["test_macro_f1", "best_val_macro_f1"]:
                            row[v["label"]] = f"{val/100.0:.3f}"
                        elif is_pct:
                            row[v["label"]] = f"{val:.2f}%"
                        elif key == "size_mb":
                            row[v["label"]] = f"~{val:.0f} MB"
                        elif key == "test_loss":
                            row[v["label"]] = f"{val:.4f}"
                        else:
                            row[v["label"]] = str(val)
                rationale_rows.append(row)

            st.dataframe(pd.DataFrame(rationale_rows), use_container_width=True, hide_index=True)

            selected_name = selected["label"]
            st.markdown(
                f"""
                1. **Performance leader on current available artifacts**: selected model is **{selected_name}** based on highest test macro F1 and test accuracy among available runs.
                2. **Convergence behavior considered**: best epoch and validation macro F1 were used to check stability and generalization.
                3. **Transfer learning benefit**: pretrained variants generally achieve stronger metrics than non-pretrained variants when available.
                4. **Model size advantage**: MobileNetV2 variants are ~14 MB, about **6x smaller** than ResNet50 (~94 MB), useful for lightweight deployment.
                5. **Macro F1 fairness**: macro F1 is emphasized because rare classes should be weighted equally, not dominated by frequent classes.
                6. **Deployment decision traceability**: selection is tied to explicit metrics (accuracy, macro F1, val F1, loss, convergence), not a single score.
                """
            )

    with tab3:
        st.markdown("## Overall System Validation (Evidence & Plots)")
        selected_for_sys = selected if selected is not None else next((v for v in variants if v["available"]), None)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", dataset["total_images"])
        c2.metric("Training Set", dataset["train"])
        c3.metric("Validation Set", dataset["val"])
        c4.metric("Test Set", dataset["test"])

        if selected_for_sys is None:
            st.write("No available metrics file to compute system validation cards.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{(selected_for_sys['test_accuracy'] or 0.0):.2f}%")
            c2.metric("Macro F1", f"{((selected_for_sys['test_macro_f1'] or 0.0)/100.0):.4f}")
            c3.metric("Macro Precision", f"{((selected_for_sys['macro_precision'] or 0.0)/100.0):.4f}")
            c4.metric("Macro Recall", f"{((selected_for_sys['macro_recall'] or 0.0)/100.0):.4f}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Weighted Precision", f"{((selected_for_sys['weighted_precision'] or 0.0)/100.0):.4f}")
            c2.metric("Weighted Recall", f"{((selected_for_sys['weighted_recall'] or 0.0)/100.0):.4f}")
            c3.metric("Weighted F1", f"{((selected_for_sys['weighted_f1'] or 0.0)/100.0):.4f}")
            c4.metric("Test Loss", f"{(selected_for_sys['test_loss'] or 0.0):.4f}")

            st.markdown("### Validation Evidence Summary")
            p = (selected_for_sys["test_accuracy"] or 0.0) / 100.0
            n_test = max(1, dataset["test"])
            se = float(np.sqrt(max(p * (1.0 - p), 0.0) / n_test))
            ci95 = 1.96 * se
            ci_low = max(0.0, p - ci95) * 100.0
            ci_high = min(1.0, p + ci95) * 100.0

            train_acc = selected_for_sys["train_accuracy"] or 0.0
            test_acc = selected_for_sys["test_accuracy"] or 0.0
            generalization_gap = train_acc - test_acc

            macro_f1 = (selected_for_sys["macro_f1"] or selected_for_sys["test_macro_f1"] or 0.0) / 100.0
            weighted_f1 = (selected_for_sys["weighted_f1"] or 0.0) / 100.0
            fairness_gap = weighted_f1 - macro_f1

            ev_rows = [
                {"Evidence": "Held-out test accuracy (95% CI)", "Value": f"{test_acc:.2f}% [{ci_low:.2f}, {ci_high:.2f}]", "Interpretation": "Expected test performance range on unseen samples."},
                {"Evidence": "Generalization gap (train - test)", "Value": f"{generalization_gap:+.2f}%", "Interpretation": "Smaller positive gap indicates lower overfitting risk."},
                {"Evidence": "Macro vs Weighted F1 gap", "Value": f"{fairness_gap:+.4f}", "Interpretation": "Near-zero gap indicates more balanced class performance."},
                {"Evidence": "Validation-to-test consistency", "Value": f"Val macro F1={((selected_for_sys['best_val_macro_f1'] or 0.0)/100.0):.4f}, Test macro F1={((selected_for_sys['test_macro_f1'] or 0.0)/100.0):.4f}", "Interpretation": "Closer values suggest stable generalization from validation to test."},
            ]
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

            st.markdown("### Validation Plots")

            # Plot 1: Generalization plot (train vs test accuracy)
            gen_df = pd.DataFrame([
                {
                    "Model": v["label"],
                    "Train Accuracy": (v["train_accuracy"] or 0.0),
                    "Test Accuracy": (v["test_accuracy"] or 0.0),
                }
                for v in variants
            ]).set_index("Model")
            st.bar_chart(gen_df)

            # Plot 2: Macro vs weighted metrics for selected model
            pw_df = pd.DataFrame({
                "Metric": ["Precision", "Recall", "F1"],
                "Macro": [
                    (selected_for_sys["macro_precision"] or 0.0) / 100.0,
                    (selected_for_sys["macro_recall"] or 0.0) / 100.0,
                    (selected_for_sys["macro_f1"] or selected_for_sys["test_macro_f1"] or 0.0) / 100.0,
                ],
                "Weighted": [
                    (selected_for_sys["weighted_precision"] or 0.0) / 100.0,
                    (selected_for_sys["weighted_recall"] or 0.0) / 100.0,
                    (selected_for_sys["weighted_f1"] or 0.0) / 100.0,
                ],
            }).set_index("Metric")
            st.bar_chart(pw_df)

            # Optional confusion matrix if stored by training script.
            cm = selected_for_sys.get("confusion_matrix") if isinstance(selected_for_sys, dict) else None
            labels = selected_for_sys.get("class_names") if isinstance(selected_for_sys, dict) else None
            if cm is not None:
                cm_np = np.array(cm)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                im = ax_cm.imshow(cm_np, cmap="Blues")
                ax_cm.set_title("Confusion Matrix")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("True")
                if labels and len(labels) == cm_np.shape[0]:
                    ax_cm.set_xticks(range(len(labels)))
                    ax_cm.set_yticks(range(len(labels)))
                    ax_cm.set_xticklabels(labels, rotation=90, fontsize=6)
                    ax_cm.set_yticklabels(labels, fontsize=6)
                plt.colorbar(im, ax=ax_cm)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            # Integrated per-class evidence (moved from removed tab)
            cr = selected_for_sys.get("classification_report")
            if isinstance(cr, dict):
                class_f1 = {
                    k: float(v.get("f1-score", 0.0))
                    for k, v in cr.items()
                    if isinstance(v, dict) and "f1-score" in v and k not in ["macro avg", "weighted avg"]
                }
                if class_f1:
                    sorted_items = sorted(class_f1.items(), key=lambda x: x[1], reverse=True)
                    names = [k for k, _ in sorted_items]
                    values = [v for _, v in sorted_items]
                    macro_ref = (selected_for_sys["test_macro_f1"] or 0.0) / 100.0

                    fig_pc, ax_pc = plt.subplots(figsize=(12, max(5, len(names) * 0.22)))
                    colors = ["#4CAF50" if v >= 0.9 else "#FF9800" if v >= 0.7 else "#f44336" for v in values]
                    ax_pc.barh(range(len(names)), values, color=colors)
                    ax_pc.set_yticks(range(len(names)))
                    ax_pc.set_yticklabels(names, fontsize=7)
                    ax_pc.set_xlim(0, 1.05)
                    ax_pc.set_xlabel("F1 Score")
                    ax_pc.set_title("Class-wise F1 Distribution")
                    ax_pc.axvline(macro_ref, color="#1f77b4", linestyle="--", linewidth=1.5, label=f"Macro F1={macro_ref:.3f}")
                    ax_pc.legend(fontsize=8)
                    st.pyplot(fig_pc)
                    plt.close(fig_pc)

# ==================== MAIN APP ====================

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    valid_pages = ["home", "binary", "disease", "system_validation", "about"]
    if st.session_state.page not in valid_pages:
        st.session_state.page = "home"
    
    if 'analyze_binary' not in st.session_state:
        st.session_state.analyze_binary = False
    
    if 'analyze_disease' not in st.session_state:
        st.session_state.analyze_disease = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🧭 Navigation")
        
        page_options = ["Home", "Binary Classification", "Disease Detection", "System Validation", "About"]
        page_map = {
            "Home": "home",
            "Binary Classification": "binary",
            "Disease Detection": "disease",
            "System Validation": "system_validation",
            "About": "about"
        }
        
        selected_page = st.radio("Select Analysis Type", page_options, 
                index=valid_pages.index(st.session_state.page))
        
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
        
        if st.button("🔬 Disease Detection", use_container_width=True):
            st.session_state.page = "disease"
            st.rerun()

        if st.button("✅ System Validation", use_container_width=True):
            st.session_state.page = "system_validation"
            st.rerun()
        
        if st.button("�📚 Documentation", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()
    
    # Route to selected page
    if st.session_state.page == "home":
        page_home()
    elif st.session_state.page == "binary":
        page_binary_classification()
    elif st.session_state.page == "disease":
        page_rare_disease_analysis()
    elif st.session_state.page == "system_validation":
        page_system_validation()
    elif st.session_state.page == "about":
        page_about()

if __name__ == "__main__":
    main()
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
import html

# NLP explainer (BioBERT + scispaCy integration)
_NLP_EXPLAIN_FN = None
_NLP_IMPORT_ERROR = None


def get_nlp_explainer():
    """Lazily import NLP explainer and expose import failures to UI."""
    global _NLP_EXPLAIN_FN, _NLP_IMPORT_ERROR
    if _NLP_EXPLAIN_FN is not None:
        return _NLP_EXPLAIN_FN, _NLP_IMPORT_ERROR

    try:
        from nlp_explainer import explain as nlp_explain_fn
        _NLP_EXPLAIN_FN = nlp_explain_fn
        _NLP_IMPORT_ERROR = None
    except Exception as e:
        _NLP_EXPLAIN_FN = None
        _NLP_IMPORT_ERROR = str(e)

    return _NLP_EXPLAIN_FN, _NLP_IMPORT_ERROR


def build_basic_nlp_fallback(top_pred):
    """Fallback text when advanced NLP stack is unavailable."""
    disease_name = top_pred.get('name', 'retinal condition')
    confidence = float(top_pred.get('probability', 0.0))

    if confidence >= 0.85:
        conf = f"high confidence ({confidence:.1%})"
    elif confidence >= 0.70:
        conf = f"moderate confidence ({confidence:.1%})"
    else:
        conf = f"lower confidence ({confidence:.1%})"

    return {
        "clinical_explanation": (
            f"The model's top prediction is {disease_name} with {conf}. "
            "This is an AI-generated assistive interpretation and should be validated by an ophthalmologist."
        ),
        "patient_explanation": (
            f"The system suggests signs related to {disease_name}. "
            "Please consult an eye specialist for proper diagnosis and treatment planning."
        ),
        "supporting_entities": [],
        "used_external_text": False,
        "is_fallback": True,
    }

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


def _normalize_activation_map(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values

    values = np.maximum(values, 0)
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max > value_min:
        return (values - value_min) / (value_max - value_min)
    return np.zeros_like(values)


def _circular_mask(shape, center_x, center_y, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2


def _estimate_retinal_regions(image_rgb):
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        return {}

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    threshold = max(10, int(np.percentile(gray_blur, 25)))
    fundus_mask = gray_blur > threshold
    fundus_mask = cv2.morphologyEx(
        fundus_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((15, 15), np.uint8),
    ).astype(bool)
    if fundus_mask.mean() < 0.35:
        fundus_mask = np.ones((height, width), dtype=bool)

    brightness = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 0)
    brightness[~fundus_mask] = -1.0
    disc_y, disc_x = np.unravel_index(int(np.argmax(brightness)), brightness.shape)

    center_x = width // 2
    center_y = height // 2
    disc_radius = max(12, int(min(height, width) * 0.10))
    macula_radius = max(14, int(min(height, width) * 0.12))

    if disc_x < center_x:
        macula_x = min(width - 1, disc_x + int(width * 0.28))
    else:
        macula_x = max(0, disc_x - int(width * 0.28))
    macula_y = int(np.clip((disc_y + center_y) / 2.0, 0, height - 1))

    central_radius = max(24, int(min(height, width) * 0.22))
    central_mask = _circular_mask((height, width), center_x, center_y, central_radius)
    disc_mask = _circular_mask((height, width), disc_x, disc_y, disc_radius)
    macula_mask = _circular_mask((height, width), macula_x, macula_y, macula_radius)
    peripheral_mask = fundus_mask & ~central_mask

    return {
        "fundus_mask": fundus_mask,
        "estimated optic disc region": disc_mask & fundus_mask,
        "estimated macular region": macula_mask & fundus_mask,
        "central retina": central_mask & fundus_mask,
        "peripheral retina": peripheral_mask,
        "superior retina": fundus_mask & (np.arange(height)[:, None] < center_y),
        "inferior retina": fundus_mask & (np.arange(height)[:, None] >= center_y),
        "left retinal field": fundus_mask & (np.arange(width)[None, :] < center_x),
        "right retinal field": fundus_mask & (np.arange(width)[None, :] >= center_x),
    }


def _summarize_gradcam(cam, image_rgb=None):
    if cam is None:
        return None

    cam = np.asarray(cam, dtype=np.float32)
    if cam.size == 0:
        return None

    cam = _normalize_activation_map(cam)

    height, width = cam.shape
    y_thirds = np.linspace(0, height, 4, dtype=int)
    x_thirds = np.linspace(0, width, 4, dtype=int)

    center_y0, center_y1 = y_thirds[1], y_thirds[2]
    center_x0, center_x1 = x_thirds[1], x_thirds[2]

    region_arrays = {
        "center of the image": cam[center_y0:center_y1, center_x0:center_x1],
        "upper portion of the image": cam[:center_y0, :],
        "lower portion of the image": cam[center_y1:, :],
        "left side of the image": cam[:, :center_x0],
        "right side of the image": cam[:, center_x1:],
        "upper-left area of the image": cam[:center_y0, :center_x0],
        "upper-right area of the image": cam[:center_y0, center_x1:],
        "lower-left area of the image": cam[center_y1:, :center_x0],
        "lower-right area of the image": cam[center_y1:, center_x1:],
    }

    hotspot_mask = cam >= 0.6
    hotspot_total = max(1, int(hotspot_mask.sum()))

    region_scores = {
        label: float(region.mean()) if region.size else 0.0
        for label, region in region_arrays.items()
    }

    anatomy_regions = _estimate_retinal_regions(image_rgb) if image_rgb is not None else {}
    anatomy_scores = {}
    for label, mask in anatomy_regions.items():
        if label == "fundus_mask":
            continue
        if not np.any(mask):
            continue
        mean_score = float(cam[mask].mean())
        overlap_score = float((hotspot_mask & mask).sum()) / hotspot_total
        anatomy_scores[label] = 0.65 * mean_score + 0.35 * overlap_score

    combined_scores = {**region_scores, **anatomy_scores}
    if not combined_scores:
        return None

    ordered_regions = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    primary_region, primary_score = ordered_regions[0]
    secondary_region = None
    if len(ordered_regions) > 1 and ordered_regions[1][1] > 0.55 * primary_score:
        secondary_region = ordered_regions[1][0]

    hotspot_fraction = float(hotspot_mask.mean())
    center_focus = float(region_scores["center of the image"])
    peripheral_mask = np.ones_like(cam, dtype=bool)
    peripheral_mask[center_y0:center_y1, center_x0:center_x1] = False
    peripheral_focus = float(cam[peripheral_mask].mean()) if peripheral_mask.any() else 0.0

    if hotspot_fraction < 0.10:
        focus_pattern = "focal"
    elif hotspot_fraction < 0.25:
        focus_pattern = "regional"
    else:
        focus_pattern = "diffuse"

    if center_focus >= peripheral_focus * 1.15:
        distribution = "central"
    elif peripheral_focus >= center_focus * 1.15:
        distribution = "peripheral"
    else:
        distribution = "mixed"

    if primary_score >= 0.65:
        attention_strength = "strong"
    elif primary_score >= 0.4:
        attention_strength = "moderate"
    else:
        attention_strength = "subtle"

    return {
        "heatmap_available": True,
        "primary_region": primary_region,
        "secondary_region": secondary_region,
        "primary_anatomy_region": primary_region if primary_region in anatomy_scores else None,
        "secondary_anatomy_region": secondary_region if secondary_region in anatomy_scores else None,
        "focus_pattern": focus_pattern,
        "distribution": distribution,
        "hotspot_fraction": hotspot_fraction,
        "attention_strength": attention_strength,
        "peak_activation": float(cam.max()),
        "heatmap_kind": "Grad-CAM",
        "anatomy_supported": bool(anatomy_scores),
    }

class RareDiseasesModel(nn.Module):
    def __init__(self, num_classes=51):
        super(RareDiseasesModel, self).__init__()
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_rare_disease_model(model_path="./resnet50_merged_rfmid_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=device)

        class_to_idx = None
        idx_to_class = None

        # Newer training pipeline saves a checkpoint dict with metadata.
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            class_to_idx = checkpoint.get("class_to_idx")
            idx_to_class = checkpoint.get("idx_to_class")
            num_classes = len(class_to_idx) if class_to_idx else 51
        else:
            # Backward compatibility: plain state_dict checkpoint.
            state_dict = checkpoint
            num_classes = 51

        model = RareDiseasesModel(num_classes=num_classes)

        # Normalize key prefixes from different save formats.
        normalized_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module.model."):
                new_key = new_key[len("module.model."):]
            elif new_key.startswith("module."):
                new_key = new_key[len("module."):]
            elif new_key.startswith("model."):
                new_key = new_key[len("model."):]
            normalized_state_dict[new_key] = value

        # Load into the wrapped torchvision ResNet backbone.
        model.model.load_state_dict(normalized_state_dict, strict=True)
        model.to(device)
        model.eval()
        st.success(f"✅ Rare disease ResNet50 model loaded ({num_classes} classes)")
        return model, device, class_to_idx, idx_to_class
    except Exception as e:
        st.error(f"Error loading rare disease model: {e}")
        return None, device, None, None

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
            'name': DISEASE_NAMES.get(code, code),
            'probability': probs[idx],
            'is_rare': code in RARE_CLASSES,
            'index': idx
        })
    
    return predictions, probs

def generate_gradcam_rare(model, image_tensor, device, target_class):
    if model is None:
        return None
    
    target_layers = [model.model.layer4[-1]]
    grad_cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    cam = grad_cam(input_tensor=image_tensor, targets=targets)
    return cam[0, :]


def build_disease_metadata(class_to_idx=None, idx_to_class=None):
    """Build class list and names, preferring mappings from checkpoint metadata."""
    if idx_to_class:
        ordered_classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    elif class_to_idx:
        ordered_classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    else:
        ordered_classes = DISEASE_CLASSES

    disease_names = {code: DISEASE_NAMES.get(code, code) for code in ordered_classes}
    rare_classes = set(ordered_classes) - {'ARMD', 'DR', 'HTN', 'CME', 'ME', 'MH', 'MHL', 'ERM', 'CNV', 'WNL'}
    return ordered_classes, disease_names, rare_classes

def page_rare_disease_analysis():
    st.markdown('<h1 class="main-header">🔬 Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown("Advanced 51-class retinal disease classification system")
    st.markdown("---")
    
    model, device, class_to_idx, idx_to_class = load_rare_disease_model()
    if model is None:
        st.error("❌ Failed to load rare disease model")
        return

    disease_classes, disease_names, rare_classes = build_disease_metadata(class_to_idx, idx_to_class)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<h3 class="sub-header">📤 Upload Fundus Image</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select retinal fundus image", type=['jpg', 'jpeg', 'png'], key="rare_upload")
        # Optional clinical text input for NLP explainability (keeps UI minimal)
        clinical_text = st.text_area("Clinical notes (optional)", value="", key="rare_clinical_text", help="Paste short clinical history or symptoms here")
        
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

                    # Align class metadata with checkpoint output size if needed.
                    if len(all_probs) != len(disease_classes):
                        disease_classes = DISEASE_CLASSES[:len(all_probs)]
                        disease_names = {code: DISEASE_NAMES.get(code, code) for code in disease_classes}
                        rare_classes = set(disease_classes) - {'ARMD', 'DR', 'HTN', 'CME', 'ME', 'MH', 'MHL', 'ERM', 'CNV', 'WNL'}

                    # Refresh prediction labels from active class metadata.
                    for pred in predictions:
                        code = disease_classes[pred['index']]
                        pred['code'] = code
                        pred['name'] = disease_names.get(code, code)
                        pred['is_rare'] = code in rare_classes
                    
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
                        names = [disease_names[c] for c in disease_classes]
                        colors = ['#dc3545' if c in rare_classes else '#28a745' for c in disease_classes]
                        
                        bars = ax.barh(names, all_probs, color=colors, alpha=0.7)
                        ax.set_xlabel("Probability", fontsize=10)
                        ax.set_xlim([0, 1])
                        ax.tick_params(axis='y', labelsize=7)
                        ax.grid(axis='x', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Grad-CAM + structured visual evidence for NLP
                    top_cam = None
                    visual_evidence = None
                    if len(predictions) > 0:
                        top_cam = generate_gradcam_rare(model, img_tensor, device, predictions[0]['index'])
                        visual_evidence = _summarize_gradcam(top_cam, img_array)

                    if show_gradcam and len(predictions) >= 3:
                        st.markdown("#### 🔬 Grad-CAM Explanations")
                        with st.spinner("Generating visual explanations..."):
                            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                            fig.suptitle("Rare Disease Grad-CAM Analysis", fontsize=14, fontweight='bold')
                            
                            axes[0, 0].imshow(img_array)
                            axes[0, 0].set_title("Original", fontweight='bold')
                            axes[0, 0].axis('off')
                            
                            for i, pred in enumerate(predictions[:2], 1):
                                cam = top_cam if i == 1 else generate_gradcam_rare(model, img_tensor, device, pred['index'])
                                if cam is not None:
                                    cam_viz = show_cam_on_image(img_normalized, cam, use_rgb=True)
                                    axes[0, i].imshow(cam_viz)
                                    title = f"{pred['code']}: {pred['name'][:20]}"
                                    if pred['is_rare']:
                                        title = f"⚠️ {title}"
                                    axes[0, i].set_title(title, fontsize=10, fontweight='bold')
                                    axes[0, i].axis('off')
                            
                            axes[1, 0].axis('off')
                            info_text = "TOP PREDICTIONS:\n\n"
                            for i, pred in enumerate(predictions[:5], 1):
                                rare_marker = "!" if pred['is_rare'] else "OK"
                                info_text += f"{i}. {pred['name']}\n   {pred['probability']:.1%} {rare_marker}\n\n"
                            
                            axes[1, 0].text(0.05, 0.95, info_text, transform=axes[1, 0].transAxes,
                                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                            
                            for i in range(1, 3):
                                pred = predictions[i-1]
                                cam = top_cam if i == 1 else generate_gradcam_rare(model, img_tensor, device, pred['index'])
                                if cam is not None:
                                    axes[1, i].imshow(cam, cmap='jet')
                                    axes[1, i].set_title(f"{pred['probability']:.1%}", fontsize=10)
                                    axes[1, i].axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Download report
                    st.markdown("---")
                    # NLP explainability (uses predicted top-1)
                    st.markdown('#### 🧾 NLP Explanation')
                    expl = None
                    nlp_explain_fn, nlp_import_error = get_nlp_explainer()
                    top_pred = predictions[0] if len(predictions) > 0 else None

                    if top_pred is not None and nlp_explain_fn is not None:
                        try:
                            clinical_input = st.session_state.get('rare_clinical_text', '')
                            expl = nlp_explain_fn(
                                top_pred['name'],
                                float(top_pred['probability']),
                                clinical_input if clinical_input.strip() else None,
                                visual_evidence=visual_evidence,
                                disease_code=top_pred['code'],
                            )
                        except Exception as e:
                            st.warning(f"Advanced NLP explanation failed, using fallback explanation. Details: {e}")

                    if expl is None and top_pred is not None:
                        if nlp_import_error:
                            st.warning(
                                "Advanced NLP modules are unavailable, using fallback explanation. "
                                f"Import error: {nlp_import_error}"
                            )
                        expl = build_basic_nlp_fallback(top_pred)
                        try:
                            from nlp_explainer import validate_explanation
                            expl['validation'] = validate_explanation(
                                top_pred['code'],
                                float(top_pred['probability']),
                                visual_evidence,
                            )
                        except Exception:
                            expl['validation'] = None

                    if expl is not None:
                        clinical_exp = expl.get('clinical_explanation', '')
                        if clinical_exp:
                            st.markdown(f"**Clinical Explanation:** {clinical_exp}")

                        patient_exp = expl.get('patient_explanation', '')
                        if patient_exp:
                            st.markdown(f"**Patient-Friendly Explanation:** {patient_exp}")

                        heatmap_summary = expl.get('heatmap_summary', '')
                        if heatmap_summary:
                            st.markdown(f"**Heatmap Meaning:** {heatmap_summary}")

                        entities = expl.get('supporting_entities', [])
                        if entities:
                            st.markdown(f"**Key Concepts:** {', '.join(entities)}")

                        if expl.get("used_external_text"):
                            st.caption("ℹ️ Explanation enhanced using external biomedical literature.")
                        elif expl.get("is_fallback"):
                            st.caption("ℹ️ Fallback NLP explanation shown (advanced NLP stack not active).")
                        else:
                            st.caption("ℹ️ Explanation generated using semantic reasoning only.")

                        st.caption("This explanation is generated by an AI system and is not a medical diagnosis.")

                        validation = expl.get('validation')
                        with st.expander("🔍 Explanation Validation — How trustworthy is this explanation?", expanded=False):
                            if validation:
                                rel_score = validation.get('reliability_score', 0.0)
                                rel_label = validation.get('reliability_label', 'Unknown')
                                rel_color = (
                                    '#28a745' if rel_label == 'High' else
                                    '#fd7e14' if rel_label == 'Moderate' else '#dc3545'
                                )
                                st.markdown(
                                    f"**Explanation Reliability: <span style='color:{rel_color};font-weight:bold'>{rel_label} ({rel_score:.0%})</span>**",
                                    unsafe_allow_html=True,
                                )
                                st.progress(rel_score)

                                cs = validation.get('consistency_status', 'unknown')
                                badge = {
                                    'match': ('✅ Consistent', '#28a745'),
                                    'partial': ('⚠️ Partially consistent', '#fd7e14'),
                                    'mismatch': ('❌ Inconsistent', '#dc3545'),
                                    'unknown': ('ℹ️ Could not verify', '#6c757d'),
                                }.get(cs, ('ℹ️ Unknown', '#6c757d'))
                                st.markdown(
                                    f"**Heatmap–Disease Consistency: <span style='color:{badge[1]}'>{badge[0]}</span>**",
                                    unsafe_allow_html=True,
                                )
                                st.caption(validation.get('consistency_message', ''))

                                expected_zones = validation.get('expected_anatomy_zones', [])
                                detected_zone = validation.get('detected_anatomy_zone')
                                if expected_zones:
                                    st.markdown(f"- **Expected anatomy for this disease:** {', '.join(expected_zones)}")
                                if detected_zone:
                                    st.markdown(f"- **Heatmap primary region detected:** {detected_zone}")

                                st.markdown("**Contributing factors:**")
                                factors = validation.get('factors', {})
                                factor_rows = [
                                    ("Model confidence", factors.get('model_confidence', '—')),
                                    ("Heatmap sharpness", factors.get('heatmap_sharpness', '—')),
                                    ("Heatmap strength", factors.get('heatmap_strength', '—')),
                                    ("Literature backing", 'Yes' if factors.get('literature_available') else 'No'),
                                ]
                                for fname, fval in factor_rows:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;• **{fname}:** {fval}", unsafe_allow_html=True)

                                src = validation.get('source_literature')
                                if src:
                                    st.markdown("**Source literature used:**")
                                    st.info(src[:600] + ("..." if len(src) > 600 else ""))
                                    st.caption("Source: PubMed abstract retrieved via NCBI E-utilities.")
                                else:
                                    st.caption("No external literature was retrieved; explanation is based on semantic reasoning only.")
                            else:
                                st.info("Validation data was not available for this explanation.")

                            st.markdown("---")
                            st.markdown(
                                "**How to read this panel:** Reliability combines model confidence, heatmap sharpness, and whether "
                                "the highlighted area matches the disease's expected anatomy. Consistency checks whether the "
                                "heatmap focus is anatomically typical for the predicted disease. A mismatch does not prove the "
                                "prediction is wrong, but it means you should interpret with more caution."
                            )
                    report_data = {
                        'timestamp': datetime.now().isoformat(),
                        'image': uploaded_file.name,
                        'model': 'ResNet50 rare disease detector',
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

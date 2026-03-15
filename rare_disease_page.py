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
from matplotlib.backends.backend_pdf import PdfPages
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from datetime import datetime
import html
import io
import textwrap
import glob
import json
import os

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


def _wrap_report_text(value, width=105):
    text = str(value or "").strip()
    if not text:
        return "N/A"
    return "\n".join(textwrap.wrap(text, width=width))


def create_rare_disease_pdf_report(
    image_name,
    predictions,
    rare_disease_alert,
    img_array,
    img_normalized,
    top_cam,
    expl,
    visual_evidence,
):
    """Build a multi-page PDF report with summary, Grad-CAM, and NLP sections."""
    timestamp = datetime.now()
    report_id = f"OX-RD-{timestamp.strftime('%Y%m%d-%H%M%S')}"

    top_rows = []
    for i, p in enumerate(predictions[:10], start=1):
        top_rows.append([
            str(i),
            str(p.get("code", "N/A")),
            str(p.get("name", "N/A")),
            f"{float(p.get('probability', 0.0)) * 100.0:.2f}%",
            "Yes" if p.get("is_rare") else "No",
        ])

    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Page 1: Summary and top predictions
        fig1 = plt.figure(figsize=(8.27, 11.69))
        fig1.patch.set_facecolor("white")
        fig1.text(0.06, 0.965, "OculoXplain Rare Disease Analysis Report", fontsize=16, fontweight="bold", color="#0b4f6c")
        fig1.text(0.06, 0.943, f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
        fig1.text(0.06, 0.927, f"Report ID: {report_id}", fontsize=10)

        summary_data = [
            ["Image name", str(image_name)],
            ["Model", "ResNet50 rare disease detector (51 classes)"],
            ["Rare disease alert", "Triggered" if rare_disease_alert else "Not triggered"],
            ["Top prediction", str(predictions[0].get("name", "N/A")) if predictions else "N/A"],
            ["Top prediction confidence", f"{float(predictions[0].get('probability', 0.0)) * 100.0:.2f}%" if predictions else "N/A"],
        ]

        ax_summary = fig1.add_axes([0.06, 0.73, 0.88, 0.17])
        ax_summary.axis("off")
        t_summary = ax_summary.table(cellText=summary_data, colLabels=["Field", "Value"], cellLoc="left", loc="upper left")
        t_summary.auto_set_font_size(False)
        t_summary.set_fontsize(9)
        t_summary.scale(1, 1.3)

        fig1.text(0.06, 0.695, "Top Predictions", fontsize=12, fontweight="bold")
        ax_preds = fig1.add_axes([0.06, 0.37, 0.88, 0.30])
        ax_preds.axis("off")
        t_preds = ax_preds.table(
            cellText=top_rows if top_rows else [["-", "-", "No predictions", "-", "-"]],
            colLabels=["Rank", "Code", "Disease", "Probability", "Rare"],
            cellLoc="left",
            loc="upper left",
        )
        t_preds.auto_set_font_size(False)
        t_preds.set_fontsize(9)
        t_preds.scale(1, 1.28)

        caution = (
            "Important: This report is for research and clinical decision support only. "
            "It is not a standalone diagnosis and must be interpreted by a qualified eye specialist."
        )
        fig1.text(0.06, 0.31, _wrap_report_text(caution, width=98), fontsize=9, color="#7c2d12")

        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # Page 2: Grad-CAM evidence
        fig2, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
        fig2.patch.set_facecolor("white")
        fig2.suptitle("Visual Evidence (Grad-CAM)", fontsize=14, fontweight="bold", y=0.985)

        for ax in axes.flatten():
            ax.axis("off")

        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title("Original Image", fontsize=10, fontweight="bold")

        if top_cam is not None:
            cam_viz = show_cam_on_image(img_normalized, top_cam, use_rgb=True)
            top_code = predictions[0].get("code", "Top-1") if predictions else "Top-1"
            cam_viz = draw_localization_bbox(top_cam, cam_viz, color=(255, 255, 0), label=f"Focus: {top_code}")
            axes[0, 1].imshow(cam_viz)
            axes[0, 1].set_title("Grad-CAM Overlay (Top-1)", fontsize=10, fontweight="bold")

            axes[1, 0].imshow(top_cam, cmap="jet")
            axes[1, 0].set_title("Grad-CAM Heatmap (Top-1)", fontsize=10, fontweight="bold")
        else:
            axes[0, 1].text(0.03, 0.5, "Grad-CAM not available for this run.", fontsize=10)

        visual_lines = []
        if isinstance(visual_evidence, dict):
            for key in ["predominant_zone", "activation_strength", "focus_spread", "quadrant"]:
                if key in visual_evidence and visual_evidence.get(key) is not None:
                    visual_lines.append(f"{key.replace('_', ' ').title()}: {visual_evidence.get(key)}")
        if not visual_lines:
            visual_lines = ["No additional visual evidence summary available."]

        axes[1, 1].text(
            0.01,
            0.98,
            _wrap_report_text("\n".join(visual_lines), width=46),
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#f3f7fb", alpha=0.9),
        )
        axes[1, 1].set_title("Visual Summary", fontsize=10, fontweight="bold")

        plt.tight_layout(rect=[0, 0.02, 1, 0.965])
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # Page 3: NLP explanation and validation
        fig3 = plt.figure(figsize=(8.27, 11.69))
        fig3.patch.set_facecolor("white")
        fig3.text(0.06, 0.965, "NLP Explanation", fontsize=14, fontweight="bold", color="#0b4f6c")

        y = 0.935

        def add_block(title, content):
            nonlocal y
            if y < 0.10:
                return
            fig3.text(0.06, y, title, fontsize=11, fontweight="bold")
            y -= 0.018
            wrapped = _wrap_report_text(content, width=106)
            fig3.text(0.065, y, wrapped, fontsize=9, va="top")
            y -= 0.012 * (wrapped.count("\n") + 2)

        if expl:
            add_block("Clinical Explanation", expl.get("clinical_explanation", "N/A"))
            add_block("Patient-Friendly Explanation", expl.get("patient_explanation", "N/A"))
            add_block("Heatmap Meaning", expl.get("heatmap_summary", "N/A"))

            entities = expl.get("supporting_entities") or []
            add_block("Key Concepts", ", ".join(entities) if entities else "N/A")

            validation = expl.get("validation") or {}
            if validation:
                add_block(
                    "Explanation Reliability",
                    f"{validation.get('reliability_label', 'Unknown')} ({float(validation.get('reliability_score', 0.0)) * 100.0:.1f}%)",
                )
                add_block("Consistency Status", validation.get("consistency_status", "unknown"))
                add_block("Consistency Message", validation.get("consistency_message", "N/A"))

                expected = validation.get("expected_anatomy_zones") or []
                add_block("Expected Anatomy Zones", ", ".join(expected) if expected else "N/A")
                add_block("Detected Anatomy Zone", validation.get("detected_anatomy_zone", "N/A"))

                factors = validation.get("factors") or {}
                if factors:
                    factor_text = "\n".join([f"- {k}: {v}" for k, v in factors.items()])
                    add_block("Contributing Factors", factor_text)
        else:
            add_block("NLP Explanation", "No NLP explanation was generated for this analysis run.")

        fig3.text(
            0.06,
            0.05,
            _wrap_report_text(
                "Disclaimer: This AI-generated report supports interpretation but does not replace comprehensive clinical evaluation.",
                width=110,
            ),
            fontsize=9,
            color="#7c2d12",
        )

        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

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
def load_rare_disease_model(model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if model_path is None:
            model_path = _resolve_best_resnet50_model_path()

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError("No ResNet50 rare-disease checkpoint was found.")

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
        return model, device, class_to_idx, idx_to_class
    except Exception as e:
        st.error(f"Error loading rare disease model: {e}")
        return None, device, None, None


def _resolve_best_resnet50_model_path():
    """Pick best available ResNet50 checkpoint from metrics files, fallback to latest model file."""
    metrics_candidates = sorted(glob.glob("resnet50_merged_rfmid*_metrics.json"))
    best_model_path = None
    best_score = None

    for metrics_path in metrics_candidates:
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                js = json.load(f)

            macro_f1 = float(js.get("test_macro_f1", -1.0))
            test_acc = float(js.get("test_accuracy", -1.0))
            metric_mtime = os.path.getmtime(metrics_path)

            explicit_model_path = js.get("artifacts", {}).get("model") if isinstance(js.get("artifacts"), dict) else None
            if explicit_model_path and os.path.exists(explicit_model_path):
                model_path = explicit_model_path
            else:
                model_path = metrics_path.replace("_metrics.json", "_model.pth")

            if not os.path.exists(model_path):
                continue

            score = (macro_f1, test_acc, metric_mtime)
            if best_score is None or score > best_score:
                best_score = score
                best_model_path = model_path
        except Exception:
            continue

    if best_model_path:
        return best_model_path

    model_candidates = sorted(glob.glob("resnet50_merged_rfmid*_model.pth"), key=os.path.getmtime, reverse=True)
    return model_candidates[0] if model_candidates else None

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
        # Training uses CrossEntropyLoss (single-label multi-class), so Softmax is correct here.
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    
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
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            st.markdown('<h3 class="sub-header">🔧 Analysis Options</h3>', unsafe_allow_html=True)
            top_k = st.slider("Top predictions to show", 5, 15, 10)
            show_gradcam = st.checkbox("📊 Generate Grad-CAM", value=True, key="rare_gradcam")
            show_all = st.checkbox("📈 Show all 51 classes", value=False, key="rare_all")
            
            if st.button("Analyze for Rare Diseases", key="btn_analyze_rare", use_container_width=True):
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
                                    cam_viz = draw_localization_bbox(
                                        cam, cam_viz,
                                        color=(255, 255, 0),
                                        label=f"Focus: {pred['code']}"
                                    )
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
                            expl = nlp_explain_fn(
                                top_pred['name'],
                                float(top_pred['probability']),
                                None,
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
                                st.info("Validation data was not available for this explanation.")

                            st.markdown("---")
                            st.markdown(
                                "**How to read this panel:** Reliability combines model confidence, heatmap sharpness, and whether "
                                "the highlighted area matches the disease's expected anatomy. Consistency checks whether the "
                                "heatmap focus is anatomically typical for the predicted disease. A mismatch does not prove the "
                                "prediction is wrong, but it means you should interpret with more caution."
                            )
                    report_pdf = create_rare_disease_pdf_report(
                        image_name=uploaded_file.name,
                        predictions=predictions,
                        rare_disease_alert=(rare_count >= 3),
                        img_array=img_array,
                        img_normalized=img_normalized,
                        top_cam=top_cam,
                        expl=expl,
                        visual_evidence=visual_evidence,
                    )

                    st.download_button(
                        label="📥 Download Full Report (PDF)",
                        data=report_pdf,
                        file_name=f"rare_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
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

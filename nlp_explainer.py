"""
NLP Explainability Module - BioBERT Semantic + spaCy Clinical

Generates medical explanations using:
1. BioBERT semantic embeddings (concept-level reasoning)
2. spaCy Clinical entity extraction (evidence)
3. NO hardcoded disease knowledge

Works with or without clinical notes.
"""

import torch
import numpy as np
from typing import Dict, List
import warnings
from literature_retriever import fetch_biomedical_text
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModel
import spacy

# =========================
# Global model cache
# =========================
_BIOBERT_TOKENIZER = None
_BIOBERT_MODEL = None
_SPACY_MODEL = None


# =========================
# Model loaders
# =========================
def _load_biobert():
    global _BIOBERT_TOKENIZER, _BIOBERT_MODEL
    if _BIOBERT_MODEL is not None:
        return _BIOBERT_TOKENIZER, _BIOBERT_MODEL

    model_name = "dmis-lab/biobert-base-cased-v1.1"
    _BIOBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _BIOBERT_MODEL = AutoModel.from_pretrained(model_name)
    _BIOBERT_MODEL.eval()
    return _BIOBERT_TOKENIZER, _BIOBERT_MODEL


def _load_spacy():
    global _SPACY_MODEL
    if _SPACY_MODEL is not None:
        return _SPACY_MODEL

    try:
        _SPACY_MODEL = spacy.load("en_core_sci_sm")
    except:
        _SPACY_MODEL = spacy.load("en_core_web_sm")
    return _SPACY_MODEL


# =========================
# BioBERT embedding
# =========================
def _get_embedding(text: str) -> np.ndarray:
    tokenizer, model = _load_biobert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        output = model(**inputs)

    return output.last_hidden_state[:, 0, :].cpu().numpy()[0]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =========================
# spaCy entity extraction
# =========================
def _extract_entities(text: str, top_k: int) -> List[str]:
    if not text or not text.strip():
        return []

    nlp = _load_spacy()
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        if (
            len(ent.text) > 2
            and ent.label_ not in {"CARDINAL", "ORDINAL", "QUANTITY"}
        ):
            entities.append(ent.text.lower())


    return list(dict.fromkeys(entities))[:top_k]


# =========================
# Semantic inference (NO KB)
# =========================
def _infer_semantic_profile(disease_embedding: np.ndarray) -> Dict[str, List[str]]:
    """
    Infer disease characteristics via weighted similarity to generic biomedical concepts.
    NO disease-specific rules.
    """

    concepts = {
        "vascular abnormality": "changes involving blood vessels in the retina",
        "fluid accumulation": "buildup of fluid within or under retinal layers",
        "tissue degeneration": "gradual loss or dysfunction of retinal cells",
        "structural damage": "physical disruption or tearing of retinal tissue",
        "inflammatory process": "inflammation affecting retinal tissues",
        "neural involvement": "involvement of nerve-related retinal structures"
    }

    scored = []
    for concept, desc in concepts.items():
        score = _cosine(disease_embedding, _get_embedding(concept))
        scored.append((concept, desc, score))

    # Sort by similarity
    scored.sort(key=lambda x: x[2], reverse=True)

    # Take top 2 meaningful concepts
    top = [s for s in scored if s[2] > 0.35][:2]

    # Fallback to best concept if none pass threshold
    if not top:
        top = scored[:1]

    return {
        "concepts": [t[0] for t in top],
        "descriptions": [t[1] for t in top]
    }

def _infer_semantic_profile_with_text(
    disease_embedding: np.ndarray,
    biomedical_text: str
) -> Dict[str, List[str]]:
    """
    Infer semantic profile using retrieved biomedical text.
    Still NO disease-specific rules.
    """

    text_embedding = _get_embedding(biomedical_text)

    combined_embedding = (disease_embedding + text_embedding) / 2.0

    return _infer_semantic_profile(combined_embedding)
    

# =========================
# Explanation generation
# =========================
def _clinical_explanation(confidence, semantic_profile, entities):

    if confidence >= 0.85:
        conf = f"high confidence ({confidence:.1%})"
    elif confidence >= 0.7:
        conf = f"moderate confidence ({confidence:.1%})"
    else:
        conf = f"lower confidence ({confidence:.1%})"

    desc = " and ".join(semantic_profile["descriptions"])

    text = (
        f"The model predicts a retinal condition with {conf}. "
        f"Semantic analysis suggests {desc}."
    )

    if entities:
        text += f" Supporting clinical indicators include {', '.join(entities[:2])}."

    text += " Specialist ophthalmological evaluation is recommended."

    return text


def _patient_explanation(semantic_profile):
    desc = ", ".join(set(semantic_profile["descriptions"]))

    return (
        f"This result suggests an eye condition involving {desc}. "
        f"This may affect how well vision works over time. "
        f"An eye specialist can confirm this and advise on next steps."
    )



# =========================
# Main API
# =========================
def explain(
    predicted_label: str,
    confidence: float,
    clinical_text: str = None,
    top_k: int = 5
) -> Dict:

    contextual_label = f"{predicted_label} retinal disease ophthalmology"
    disease_embedding = _get_embedding(contextual_label)

    # 🔹 OPTIONAL literature retrieval
    biomedical_text = fetch_biomedical_text(predicted_label)

    if biomedical_text:
        semantic_profile = _infer_semantic_profile_with_text(
            disease_embedding,
            biomedical_text
        )
    else:
        semantic_profile = _infer_semantic_profile(disease_embedding)

    entities = _extract_entities(clinical_text, top_k) if clinical_text else []

    return {
        "clinical_explanation": _clinical_explanation(
            confidence, semantic_profile, entities
        ),
        "patient_explanation": _patient_explanation(semantic_profile),
        "supporting_entities": entities,
        "highlighted_text_spans": entities,
        "used_external_text": bool(biomedical_text and biomedical_text.strip())

    }


__all__ = ["explain"]

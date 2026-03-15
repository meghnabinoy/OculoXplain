"""NLP explainability utilities for OculoXplain.

Provides:
1) semantic explanation generation (BioBERT + optional biomedical text)
2) optional clinical entity extraction (spaCy)
3) heatmap-aware narration and trust/consistency validation
"""

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import spacy
except Exception:
    spacy = None

try:
    from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None
    BertModel = None
    BertTokenizer = None

from literature_retriever import fetch_biomedical_text

warnings.filterwarnings("ignore")

if os.getenv("NLP_ALLOW_ONLINE", "0") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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

    if AutoTokenizer is None or AutoModel is None:
        raise ImportError("transformers is not installed")

    model_name = "dmis-lab/biobert-base-cased-v1.1"
    allow_online = os.getenv("NLP_ALLOW_ONLINE", "0") == "1"
    local_only = not allow_online
    try:
        _BIOBERT_TOKENIZER = BertTokenizer.from_pretrained(
            model_name, local_files_only=local_only
        )
    except Exception:
        _BIOBERT_TOKENIZER = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, local_files_only=local_only
        )
    try:
        _BIOBERT_MODEL = BertModel.from_pretrained(
            model_name, local_files_only=local_only
        )
    except Exception:
        _BIOBERT_MODEL = AutoModel.from_pretrained(
            model_name, local_files_only=local_only
        )
    _BIOBERT_MODEL.eval()
    return _BIOBERT_TOKENIZER, _BIOBERT_MODEL


def _load_spacy():
    global _SPACY_MODEL
    if _SPACY_MODEL is not None:
        return _SPACY_MODEL

    if spacy is None:
        raise ImportError("spacy is not installed")

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

# Generic non-clinical words to exclude from entity results
_CLINICAL_SKIP = {
    "patient", "patients", "history", "reports", "report",
    "case", "cases", "age", "year", "years", "month", "months",
    "day", "days", "complaint", "complaints", "note", "notes",
}

def _extract_entities(text: str, top_k: int) -> List[str]:
    if not text or not text.strip():
        return []

    try:
        nlp = _load_spacy()
        doc = nlp(text)
    except Exception:
        return []

    entities = []

    # 1. Try named entities first — works well with biomedical spaCy models
    for ent in doc.ents:
        if (
            len(ent.text) > 2
            and ent.label_ not in {"CARDINAL", "ORDINAL", "QUANTITY", "DATE", "TIME"}
            and ent.text.lower() not in _CLINICAL_SKIP
        ):
            entities.append(ent.text.lower())

    # 2. If NER returned nothing (general English model has no medical NER),
    #    walk every NOUN token and build a phrase that includes any immediately
    #    preceding adjective or verb modifier (e.g. "blurred vision", "floaters").
    #    This is robust regardless of how spaCy labels medical text.
    if not entities:
        seen: set = set()
        for token in doc:
            if token.pos_ != "NOUN":
                continue
            if token.text.lower() in _CLINICAL_SKIP:
                continue
            # Check the token directly to the left for an ADJ or VERB modifier
            left_mod = ""
            if token.i > 0:
                prev = doc[token.i - 1]
                if prev.pos_ in ("ADJ", "VERB") and prev.dep_ in (
                    "amod", "compound", "ROOT", "advmod"
                ):
                    left_mod = prev.text
            phrase = (f"{left_mod} {token.text}".strip() if left_mod else token.text)
            phrase = phrase.lower()
            if len(phrase) > 2 and phrase not in seen:
                seen.add(phrase)
                entities.append(phrase)

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
    biomedical_text: str,
) -> Dict[str, List[str]]:
    """
    Infer semantic profile using retrieved biomedical text.
    Still NO disease-specific rules.
    """

    text_embedding = _get_embedding(biomedical_text)

    combined_embedding = (disease_embedding + text_embedding) / 2.0

    return _infer_semantic_profile(combined_embedding)


def _fallback_semantic_profile() -> Dict[str, List[str]]:
    return {
        "concepts": ["retinal abnormality"],
        "descriptions": ["patterns in the retina that may indicate disease activity"],
    }


# Disease-specific semantic profiles keyed by disease name (lowercase) and disease code.
# Used when BioBERT is unavailable (offline mode without cached weights).
_DISEASE_SEMANTIC_KB: Dict[str, Dict[str, List[str]]] = {
    # --- Optic disc / nerve ---
    "odc": {
        "concepts": ["optic disc cupping", "increased cup-to-disc ratio"],
        "descriptions": [
            "increased excavation of the optic disc, suggesting possible raised intraocular pressure or glaucomatous nerve damage",
            "elevated cup-to-disc ratio which may indicate progressive optic nerve head changes",
        ],
    },
    "ode": {
        "concepts": ["optic disc oedema", "disc swelling"],
        "descriptions": [
            "swelling of the optic disc that may arise from raised intracranial pressure, optic neuritis, or vascular causes",
        ],
    },
    "odp": {
        "concepts": ["optic disc pit", "structural disc anomaly"],
        "descriptions": [
            "a congenital pit-like depression in the optic disc that can be associated with serous macular detachment",
        ],
    },
    "on": {
        "concepts": ["optic neuritis", "optic nerve inflammation"],
        "descriptions": [
            "inflammation of the optic nerve, often associated with demyelinating conditions such as multiple sclerosis",
        ],
    },
    "opdm": {
        "concepts": ["optic disc pallor", "optic atrophy"],
        "descriptions": [
            "pallor of the optic disc indicating possible optic atrophy from prior ischaemia, compression, or neurodegenerative processes",
        ],
    },
    "aion": {
        "concepts": ["anterior ischaemic optic neuropathy", "optic nerve ischaemia"],
        "descriptions": [
            "sudden loss of blood supply to the optic nerve head, commonly presenting with acute painless visual field loss",
        ],
    },
    "iih": {
        "concepts": ["raised intracranial pressure", "papilloedema"],
        "descriptions": [
            "raised intracranial pressure causing bilateral optic disc oedema, associated with idiopathic intracranial hypertension",
        ],
    },
    "td": {
        "concepts": ["tilted disc syndrome", "optic disc tilting"],
        "descriptions": [
            "an oblique entry of the optic nerve causing disc tilting, commonly associated with myopia and bitemporal visual field defects",
        ],
    },
    "tv": {
        "concepts": ["temporal disc pallor", "optic nerve damage"],
        "descriptions": [
            "pallor confined to the temporal portion of the optic disc, which may indicate axonal loss affecting the papillomacular bundle",
        ],
    },
    "ms": {
        "concepts": ["myelinated nerve fibres", "retinal nerve fibre layer anomaly"],
        "descriptions": [
            "abnormal myelination of retinal nerve fibres appearing as white feathery patches adjacent to the optic disc",
        ],
    },
    # --- Macula ---
    "armd": {
        "concepts": ["age-related macular degeneration", "macular degeneration"],
        "descriptions": [
            "degenerative changes in the macula, characterised by drusen deposits, geographic atrophy, or choroidal neovascularisation, causing central vision loss",
        ],
    },
    "cnv": {
        "concepts": ["choroidal neovascularisation", "subretinal neovascular membrane"],
        "descriptions": [
            "abnormal growth of new blood vessels from the choroid beneath the retina, frequently causing fluid leakage and central vision distortion",
        ],
    },
    "csc": {
        "concepts": ["central serous chorioretinopathy", "subretinal fluid accumulation"],
        "descriptions": [
            "accumulation of subretinal fluid beneath the neurosensory retina at the macula, often causing blurred or distorted central vision",
        ],
    },
    "csr": {
        "concepts": ["central serous retinopathy", "serous macular detachment"],
        "descriptions": [
            "serous detachment of the neurosensory retina, typically in the macular region, related to choroidal hyperpermeability",
        ],
    },
    "crs": {
        "concepts": ["chronic central serous", "persistent subretinal fluid"],
        "descriptions": [
            "chronic form of serous retinopathy with persistent fluid under the macula and secondary RPE changes",
        ],
    },
    "cme": {
        "concepts": ["cystoid macular oedema", "intraretinal fluid"],
        "descriptions": [
            "cystic accumulation of fluid within the macular retinal layers, causing decreased central visual acuity",
        ],
    },
    "me": {
        "concepts": ["macular oedema", "retinal fluid accumulation"],
        "descriptions": [
            "swelling of the macula due to fluid accumulation, often secondary to diabetic retinopathy or retinal vein occlusion",
        ],
    },
    "mh": {
        "concepts": ["macular hole", "full-thickness macular defect"],
        "descriptions": [
            "a full-thickness loss of retinal tissue at the fovea causing central scotoma, often associated with vitreous traction",
        ],
    },
    "mhl": {
        "concepts": ["large macular hole", "significant foveal defect"],
        "descriptions": [
            "a large full-thickness macular hole with significant central retinal tissue loss and corresponding central vision impairment",
        ],
    },
    "mca": {
        "concepts": ["macular atrophy", "geographic atrophy at macula"],
        "descriptions": [
            "focal loss of the retinal pigment epithelium and photoreceptors in the macular region, leading to irreversible central vision reduction",
        ],
    },
    "erm": {
        "concepts": ["epiretinal membrane", "macular pucker"],
        "descriptions": [
            "a fibrocellular membrane on the inner retinal surface at the macula causing distortion and reduced visual acuity",
        ],
    },
    "edn": {
        "concepts": ["epiretinal membrane with drusen", "combined macular pathology"],
        "descriptions": [
            "coexistence of an epiretinal membrane with underlying drusen deposits, combining tractional and degenerative mechanisms",
        ],
    },
    "hped": {
        "concepts": ["haemorrhagic pigment epithelial detachment", "choroidal bleed"],
        "descriptions": [
            "detachment of the pigment epithelium with haemorrhagic contents, often indicating choroidal neovascularisation beneath",
        ],
    },
    "sofe": {
        "concepts": ["subretinal fluid", "serous detachment"],
        "descriptions": [
            "fluid beneath the neurosensory retina indicating disruption of the outer blood-retinal barrier or active choroidal pathology",
        ],
    },
    "rs": {
        "concepts": ["retinal scar", "chorioretinal scar"],
        "descriptions": [
            "a chorioretinal scar resulting from prior inflammation, laser treatment, or trauma, with associated loss of photoreceptors",
        ],
    },
    "rpec": {
        "concepts": ["RPE changes", "retinal pigment epithelium disruption"],
        "descriptions": [
            "changes in the retinal pigment epithelium such as atrophy, hyperpigmentation or hypopigmentation, often secondary to chronic disease",
        ],
    },
    # --- Vascular ---
    "dr": {
        "concepts": ["diabetic retinopathy", "microvascular retinal disease"],
        "descriptions": [
            "diabetic microvascular damage affecting retinal capillaries, presenting as microaneurysms, haemorrhages, exudates, and in advanced stages, neovascularisation",
        ],
    },
    "htn": {
        "concepts": ["hypertensive retinopathy", "hypertension-related retinal changes"],
        "descriptions": [
            "retinal vascular changes caused by chronic hypertension, including arteriolar narrowing, arteriovenous nipping, and flame haemorrhages",
        ],
    },
    "brvo": {
        "concepts": ["branch retinal vein occlusion", "sectoral retinal venous obstruction"],
        "descriptions": [
            "occlusion of a branch retinal vein causing sectoral haemorrhages, oedema, and ischaemia in the affected quadrant",
        ],
    },
    "crvo": {
        "concepts": ["central retinal vein occlusion", "total retinal venous obstruction"],
        "descriptions": [
            "occlusion of the central retinal vein leading to widespread haemorrhage and oedema across all four quadrants",
        ],
    },
    "crao": {
        "concepts": ["central retinal artery occlusion", "acute retinal ischaemia"],
        "descriptions": [
            "acute occlusion of the central retinal artery causing sudden profound visual loss with diffuse retinal whitening",
        ],
    },
    "cl": {
        "concepts": ["central artery ischaemia", "retinal arterial insufficiency"],
        "descriptions": [
            "ischaemic changes in the central retina related to arterial insufficiency, presenting with pallor and reduced perfusion",
        ],
    },
    "ah": {
        "concepts": ["arteriolar narrowing", "hypertensive arteriolar change"],
        "descriptions": [
            "narrowing of retinal arterioles, a hallmark of chronic hypertensive retinopathy indicating increased arterial wall resistance",
        ],
    },
    "hr": {
        "concepts": ["retinal haemorrhage", "intraretinal bleeding"],
        "descriptions": [
            "bleeding within the retinal layers, arising from vascular diseases such as diabetic or hypertensive retinopathy",
        ],
    },
    "rhl": {
        "concepts": ["layered retinal haemorrhage", "subhyaloid haemorrhage"],
        "descriptions": [
            "layered haemorrhage within or just beneath the retina, often associated with proliferative retinopathy or trauma",
        ],
    },
    "prh": {
        "concepts": ["preretinal haemorrhage", "subhyaloid bleed"],
        "descriptions": [
            "haemorrhage between the inner retinal surface and the posterior vitreous face, commonly boat-shaped in appearance",
        ],
    },
    "cws": {
        "concepts": ["cotton wool spots", "retinal nerve fibre infarcts"],
        "descriptions": [
            "whitish focal lesions in the nerve fibre layer representing microinfarcts, associated with hypertensive retinopathy or diabetic changes",
        ],
    },
    "dn": {
        "concepts": ["drusen", "RPE-Bruch membrane deposits"],
        "descriptions": [
            "extracellular deposits beneath the retinal pigment epithelium in Bruch's membrane, an early feature of age-related macular degeneration",
        ],
    },
    # --- Structural / peripheral ---
    "rd": {
        "concepts": ["retinal detachment", "retinal separation"],
        "descriptions": [
            "separation of the neurosensory retina from the underlying retinal pigment epithelium, requiring urgent treatment to preserve vision",
        ],
    },
    "grt": {
        "concepts": ["giant retinal tear", "circumferential retinal break"],
        "descriptions": [
            "a circumferential full-thickness retinal tear spanning at least 90 degrees, often leading to retinal detachment",
        ],
    },
    "rt": {
        "concepts": ["retinal tear", "retinal break"],
        "descriptions": [
            "a full-thickness break in the retina, often caused by vitreoretinal traction, which may lead to retinal detachment if untreated",
        ],
    },
    "rtr": {
        "concepts": ["recurrent retinal tear", "repeat retinal break"],
        "descriptions": [
            "recurrence of a retinal tear after prior treatment, indicating ongoing vitreoretinal traction or incomplete adhesion",
        ],
    },
    "rp": {
        "concepts": ["retinitis pigmentosa", "inherited retinal degeneration"],
        "descriptions": [
            "a hereditary progressive retinal dystrophy causing peripheral photoreceptor degeneration, night blindness, and constriction of the visual field",
        ],
    },
    "cb": {
        "concepts": ["Coats disease", "retinal telangiectasia"],
        "descriptions": [
            "abnormal dilated and tortuous retinal vessels with subretinal exudate deposition, predominantly affecting young males",
        ],
    },
    "st": {
        "concepts": ["posterior staphyloma", "scleral ectasia"],
        "descriptions": [
            "outward bulging of the posterior sclera and uvea, common in pathological myopia, leading to progressive retinal thinning",
        ],
    },
    "mya": {
        "concepts": ["myopic degeneration", "pathological myopia changes"],
        "descriptions": [
            "degenerative changes associated with high myopia including posterior staphyloma, lacquer cracks, and chorioretinal atrophy",
        ],
    },
    "tsln": {
        "concepts": ["tessellated fundus", "tigroid fundus"],
        "descriptions": [
            "visibility of the underlying choroidal vessels through a thin RPE, giving a mosaic appearance typically seen in myopic eyes",
        ],
    },
    "cf": {
        "concepts": ["chorioretinal folds", "subretinal folds"],
        "descriptions": [
            "folds of the choroid and retina resulting from hypotony, orbital mass effect, or raised intracranial pressure",
        ],
    },
    "vs": {
        "concepts": ["vitreous syneresis", "vitreous liquefaction"],
        "descriptions": [
            "age-related breakdown of the vitreous gel structure causing liquid pockets, commonly associated with posterior vitreous detachment and floaters",
        ],
    },
    "ls": {
        "concepts": ["laser scars", "photocoagulation marks"],
        "descriptions": [
            "chorioretinal scars from prior laser photocoagulation treatment for conditions such as diabetic retinopathy or retinal tears",
        ],
    },
    # --- Normal ---
    "wnl": {
        "concepts": ["normal retinal findings", "healthy fundus"],
        "descriptions": [
            "no significant retinal pathology detected; the fundus appearance is within normal limits",
        ],
    },
}

def _get_disease_specific_profile(predicted_label: str, disease_code: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    """Return disease-specific semantic profile by code or name, or None if not found."""
    if disease_code:
        profile = _DISEASE_SEMANTIC_KB.get(disease_code.lower())
        if profile:
            return profile
    # Try matching by disease name (lowercase, partial)
    label_lower = predicted_label.lower()
    for key, profile in _DISEASE_SEMANTIC_KB.items():
        if key in label_lower or label_lower in key:
            return profile
    return None


_DISEASE_ANATOMY_PROFILE: Dict[str, List[str]] = {
    "AION": ["estimated optic disc region"],
    "ODC": ["estimated optic disc region"],
    "ODE": ["estimated optic disc region"],
    "ODP": ["estimated optic disc region"],
    "ON": ["estimated optic disc region"],
    "OPDM": ["estimated optic disc region"],
    "IIH": ["estimated optic disc region"],
    "TD": ["estimated optic disc region"],
    "TV": ["estimated optic disc region", "central retina"],
    "MS": ["estimated optic disc region"],
    "ARMD": ["estimated macular region", "central retina"],
    "CNV": ["estimated macular region", "central retina"],
    "CSC": ["estimated macular region", "central retina"],
    "CSR": ["estimated macular region", "central retina"],
    "CME": ["estimated macular region", "central retina"],
    "ME": ["estimated macular region", "central retina"],
    "MH": ["estimated macular region", "central retina"],
    "MHL": ["estimated macular region", "central retina"],
    "MCA": ["estimated macular region", "central retina"],
    "ERM": ["estimated macular region", "central retina"],
    "HPED": ["estimated macular region", "central retina"],
    "SOFE": ["estimated macular region", "central retina"],
    "CRS": ["estimated macular region", "central retina"],
    "RS": ["estimated macular region", "central retina"],
    "RPEC": ["estimated macular region", "central retina"],
    "EDN": ["estimated macular region", "central retina"],
    "DN": ["central retina", "estimated macular region"],
    "DR": ["central retina", "peripheral retina"],
    "BRVO": ["central retina", "peripheral retina"],
    "CRVO": ["central retina", "peripheral retina"],
    "CRAO": ["central retina"],
    "CL": ["central retina"],
    "HTN": ["central retina", "peripheral retina"],
    "HR": ["central retina", "peripheral retina"],
    "PRH": ["peripheral retina", "central retina"],
    "RHL": ["central retina", "peripheral retina"],
    "AH": ["central retina", "peripheral retina"],
    "CWS": ["central retina", "peripheral retina"],
    "LS": ["central retina", "peripheral retina"],
    "CF": ["central retina", "peripheral retina"],
    "VS": ["central retina", "peripheral retina"],
    "TSLN": ["central retina", "peripheral retina"],
    "RP": ["peripheral retina"],
    "CB": ["peripheral retina", "central retina"],
    "RD": ["peripheral retina"],
    "GRT": ["peripheral retina"],
    "RT": ["peripheral retina"],
    "RTR": ["peripheral retina"],
    "ST": ["peripheral retina"],
    "MYA": ["central retina", "peripheral retina"],
    "WNL": [],
}


def _patient_region_phrase(region_label: str) -> str:
    region_map = {
        "estimated optic disc region": "the optic disc area, where the eye connects to the nerve",
        "estimated macular region": "the macular area near the center of detailed vision",
        "central retina": "the central retina",
        "peripheral retina": "the outer retina",
        "superior retina": "the upper retina",
        "inferior retina": "the lower retina",
        "left retinal field": "the left side of the retinal image",
        "right retinal field": "the right side of the retinal image",
    }
    return region_map.get(region_label, region_label)


def _clinical_region_phrase(region_label: str) -> str:
    if region_label == "estimated optic disc region":
        return "the estimated optic disc region"
    if region_label == "estimated macular region":
        return "the estimated macular region"
    return region_label


def _visual_evidence_summary(visual_evidence: Optional[Dict]) -> Dict[str, str]:
    if not visual_evidence or not visual_evidence.get("heatmap_available"):
        return {
            "clinical": "No Grad-CAM heatmap summary was available for this prediction.",
            "patient": "No heatmap summary was available, so this explanation is based on the disease prediction only.",
            "summary": "Grad-CAM evidence was not available for this case.",
        }

    primary_region = visual_evidence.get("primary_region", "part of the image")
    secondary_region = visual_evidence.get("secondary_region")
    focus_pattern = visual_evidence.get("focus_pattern", "regional")
    distribution = visual_evidence.get("distribution", "mixed")
    attention_strength = visual_evidence.get("attention_strength", "moderate")
    hotspot_fraction = float(visual_evidence.get("hotspot_fraction", 0.0))

    coverage_pct = hotspot_fraction * 100.0
    region_phrase = _clinical_region_phrase(primary_region)
    patient_region_phrase = _patient_region_phrase(primary_region)
    if secondary_region:
        region_phrase = (
            f"{_clinical_region_phrase(primary_region)} and secondarily in the "
            f"{_clinical_region_phrase(secondary_region)}"
        )
        patient_region_phrase = (
            f"{_patient_region_phrase(primary_region)} and secondarily the "
            f"{_patient_region_phrase(secondary_region)}"
        )

    clinical = (
        f"The Grad-CAM heatmap shows {attention_strength} model attention focused mainly in {region_phrase}. "
        f"The attention pattern is {focus_pattern} with a {distribution} distribution across the retinal image. "
        "This heatmap reflects where the model relied most on visual evidence, rather than confirming pathology on its own."
    )

    patient = (
        "The colored heatmap is showing which parts of the eye photo the AI looked at most. "
        f"Here, it mainly focuses on {patient_region_phrase}. "
        f"The highlighted area is {focus_pattern}, meaning the model used "
        f"{'a small concentrated area' if focus_pattern == 'focal' else 'a broader region' if focus_pattern == 'regional' else 'a wider spread of the image'} "
        "when making its decision."
    )

    summary = (
        f"Model attention is strongest in {region_phrase} with a {focus_pattern} pattern, "
        f"covering about {coverage_pct:.0f}% of the highest-activation heatmap area."
    )

    return {
        "clinical": clinical,
        "patient": patient,
        "summary": summary,
    }


def _check_anatomy_consistency(disease_code: str, visual_evidence: Optional[Dict]) -> Dict:
    expected = _DISEASE_ANATOMY_PROFILE.get(disease_code)

    if expected is None:
        return {
            "status": "unknown",
            "message": f"No anatomy profile available for disease code '{disease_code}'.",
            "expected_zones": [],
            "detected_zone": None,
        }

    if not expected:
        return {
            "status": "unknown",
            "message": "This is a normal finding, so no specific retinal region is expected.",
            "expected_zones": [],
            "detected_zone": None,
        }

    if not visual_evidence or not visual_evidence.get("heatmap_available"):
        return {
            "status": "unknown",
            "message": "Heatmap was not available to compare against expected anatomy.",
            "expected_zones": expected,
            "detected_zone": None,
        }

    primary_anatomy = visual_evidence.get("primary_anatomy_region")
    secondary_anatomy = visual_evidence.get("secondary_anatomy_region")

    if not primary_anatomy:
        return {
            "status": "unknown",
            "message": "The heatmap did not resolve to a specific retinal anatomy region.",
            "expected_zones": expected,
            "detected_zone": None,
        }

    detected_set = {r for r in [primary_anatomy, secondary_anatomy] if r}
    expected_set = set(expected)

    if primary_anatomy in expected_set:
        return {
            "status": "match",
            "message": (
                f"The heatmap's primary focus ({primary_anatomy}) matches the expected anatomy "
                "for this disease."
            ),
            "expected_zones": expected,
            "detected_zone": primary_anatomy,
        }

    if detected_set & expected_set:
        matched = list(detected_set & expected_set)[0]
        return {
            "status": "partial",
            "message": (
                f"A secondary heatmap region ({matched}) aligns with expected anatomy, but the "
                f"primary focus ({primary_anatomy}) is less typical."
            ),
            "expected_zones": expected,
            "detected_zone": primary_anatomy,
        }

    return {
        "status": "mismatch",
        "message": (
            f"The heatmap focuses on {primary_anatomy}, which is not typical for this disease "
            f"(usually: {', '.join(expected)})."
        ),
        "expected_zones": expected,
        "detected_zone": primary_anatomy,
    }


def _score_reliability(
    confidence: float,
    visual_evidence: Optional[Dict],
    biomedical_text: str,
    consistency_status: str,
) -> Dict[str, object]:
    score = 0.40

    if confidence >= 0.85:
        score += 0.20
    elif confidence >= 0.70:
        score += 0.12
    elif confidence >= 0.50:
        score += 0.06

    if visual_evidence and visual_evidence.get("heatmap_available"):
        focus = visual_evidence.get("focus_pattern", "diffuse")
        strength = visual_evidence.get("attention_strength", "subtle")
        if focus == "focal":
            score += 0.20
        elif focus == "regional":
            score += 0.12
        else:
            score += 0.04
        if strength == "strong":
            score += 0.05

    if biomedical_text and biomedical_text.strip():
        score += 0.10

    if consistency_status == "match":
        score += 0.10
    elif consistency_status == "partial":
        score += 0.04
    elif consistency_status == "mismatch":
        score -= 0.10

    score = float(max(0.0, min(1.0, score)))
    if score >= 0.75:
        label = "High"
    elif score >= 0.50:
        label = "Moderate"
    else:
        label = "Low"
    return {"score": round(score, 2), "label": label}


def validate_explanation(
    disease_code: str,
    confidence: float,
    visual_evidence: Optional[Dict],
    biomedical_text: str = "",
) -> Dict:
    consistency = _check_anatomy_consistency(disease_code, visual_evidence)
    reliability = _score_reliability(
        confidence, visual_evidence, biomedical_text, consistency["status"]
    )

    focus_pattern = (
        visual_evidence.get("focus_pattern", "unknown")
        if visual_evidence and visual_evidence.get("heatmap_available")
        else "unavailable"
    )
    attention_strength = (
        visual_evidence.get("attention_strength", "unknown")
        if visual_evidence and visual_evidence.get("heatmap_available")
        else "unavailable"
    )

    return {
        "consistency_status": consistency["status"],
        "consistency_message": consistency["message"],
        "expected_anatomy_zones": consistency["expected_zones"],
        "detected_anatomy_zone": consistency["detected_zone"],
        "reliability_score": reliability["score"],
        "reliability_label": reliability["label"],
        "factors": {
            "model_confidence": "high" if confidence >= 0.85 else "moderate" if confidence >= 0.70 else "low",
            "heatmap_sharpness": focus_pattern,
            "heatmap_strength": attention_strength,
            "literature_available": bool(biomedical_text and biomedical_text.strip()),
        },
        "source_literature": biomedical_text.strip() if biomedical_text else None,
    }

# =========================
# Explanation generation
# =========================
def _clinical_explanation(predicted_label, confidence, semantic_profile, entities, visual_summary):

    if confidence >= 0.85:
        conf = f"high confidence ({confidence:.1%})"
    elif confidence >= 0.7:
        conf = f"moderate confidence ({confidence:.1%})"
    else:
        conf = f"lower confidence ({confidence:.1%})"

    desc = " and ".join(semantic_profile["descriptions"])

    text = (
        f"The model predicts {predicted_label} with {conf}. "
        f"Semantic analysis suggests {desc}."
    )

    if visual_summary:
        text += f" {visual_summary}"

    if entities:
        text += f" Supporting clinical indicators include {', '.join(entities[:2])}."

    text += " Specialist ophthalmological evaluation is recommended."

    return text


def _patient_explanation(predicted_label, semantic_profile, visual_summary):
    desc = ", ".join(set(semantic_profile["descriptions"]))

    text = (
        f"The AI thinks this image may show {predicted_label}, involving {desc}. "
        "This may affect how well vision works over time. "
    )

    if visual_summary:
        text += f"{visual_summary} "

    text += "An eye specialist can confirm this and advise on next steps."
    return text



# =========================
# Main API
# =========================
def explain(
    predicted_label: str,
    confidence: float,
    clinical_text: str = None,
    top_k: int = 5,
    visual_evidence: Optional[Dict] = None,
    disease_code: str = None,
) -> Dict:
    biomedical_text = ""
    try:
        contextual_label = f"{predicted_label} retinal disease ophthalmology"
        disease_embedding = _get_embedding(contextual_label)
        biomedical_text = fetch_biomedical_text(predicted_label)
        if biomedical_text:
            semantic_profile = _infer_semantic_profile_with_text(
                disease_embedding,
                biomedical_text,
            )
        else:
            semantic_profile = _infer_semantic_profile(disease_embedding)
    except Exception:
        specific = _get_disease_specific_profile(predicted_label, disease_code)
        semantic_profile = specific if specific is not None else _fallback_semantic_profile()

    try:
        entities = _extract_entities(clinical_text, top_k) if clinical_text else []
    except Exception:
        entities = []

    visual_summary = _visual_evidence_summary(visual_evidence)

    validation = None
    if disease_code:
        try:
            validation = validate_explanation(
                disease_code, confidence, visual_evidence, biomedical_text
            )
        except Exception:
            validation = None

    return {
        "clinical_explanation": _clinical_explanation(
            predicted_label, confidence, semantic_profile, entities, visual_summary["clinical"]
        ),
        "patient_explanation": _patient_explanation(
            predicted_label, semantic_profile, visual_summary["patient"]
        ),
        "heatmap_summary": visual_summary["summary"],
        "supporting_entities": entities,
        "highlighted_text_spans": entities,
        "used_external_text": bool(biomedical_text and biomedical_text.strip()),
        "visual_evidence": visual_evidence or {},
        "validation": validation,
    }


__all__ = ["explain", "validate_explanation"]

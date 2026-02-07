"""Pre-download BioBERT and spaCy model to cache."""
from transformers import AutoTokenizer, AutoModel
import spacy

print('Downloading BioBERT tokenizer...')
AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
print('Downloading BioBERT model...')
AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
print('Loading spaCy model...')
try:
    spacy.load('en_core_sci_sm')
    print('scispaCy model loaded')
except Exception:
    try:
        spacy.load('en_core_web_sm')
        print('spaCy en_core_web_sm loaded')
    except Exception as e:
        print('Failed to load spaCy model:', e)
print('Pre-download complete')

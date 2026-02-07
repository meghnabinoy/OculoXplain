"""
Biomedical Literature Retriever (Lightweight)

Fetches short, neutral biomedical text for a disease name.
Used ONLY as optional context for BioBERT.
No hardcoded disease knowledge.
"""

import requests
import re
from functools import lru_cache

HEADERS = {
    "User-Agent": "OculoXplain/1.0 (research-use)"
}


@lru_cache(maxsize=128)
def fetch_biomedical_text(disease_name: str) -> str | None:
    """
    Fetch 1–3 sentences from PubMed abstracts using NCBI E-utilities.
    Returns cleaned text or None if retrieval fails.
    """

    query = f"{disease_name} retinal disease ophthalmology review"

    # Step 1: search PubMed
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 2
    }

    try:
        search_resp = requests.get(
            search_url,
            params=search_params,
            headers=HEADERS,
            timeout=5
        )   

        pmid_list = search_resp.json()["esearchresult"]["idlist"]

        if not pmid_list:
            return None

        pmid = pmid_list[0]

        # Step 2: fetch abstract
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "text",
            "rettype": "abstract"
        }

        fetch_resp = requests.get(
            fetch_url,
            params=fetch_params,
            headers=HEADERS,
            timeout=5
        )

        text = fetch_resp.text

        # Clean and keep first 2–3 sentences
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)

        result = " ".join(sentences[:3]).strip()
        return result if result else None


    except Exception:
        return None


__all__ = ["fetch_biomedical_text"]

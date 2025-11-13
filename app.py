# app.py
"""
AI Research Paper Assistant (Full-featured)
- Upload PDFs, extract text
- Multi-level summarization via Hugging Face Inference API
- Citation & reference extraction (regex heuristics)
- Semantic search using sentence-transformers (all-MiniLM-L6-v2)
- Topic clustering with KMeans
- Paper-to-paper recommendations using embedding similarity
- Export summaries as .docx, .pdf, .csv
"""

import os
import time
import json
import re
from io import BytesIO
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import streamlit as st
import PyPDF2
import requests
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from docx import Document
from fpdf2 import FPDF
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# -------------------------
#  User-provided HF key (from your message)
#  You can remove this and rely on .env or environment variables instead.
USER_PROVIDED_HF_KEY = "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"
# -------------------------

# Load .env if present
load_dotenv()
# Prefer environment variable; if not present, use the provided key as fallback
if not os.getenv("HUGGINGFACE_API_KEY") and USER_PROVIDED_HF_KEY:
    os.environ["HUGGINGFACE_API_KEY"] = USER_PROVIDED_HF_KEY

# ------------------ Configuration / Defaults ------------------
HF_API_URL_BASE = "https://api-inference.huggingface.co/models"
HF_SUMMARY_MODEL_DEFAULT = "facebook/bart-large-cnn"
EMBEDDING_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------ Data Model ------------------
@dataclass
class ResearchPaper:
    id: str
    filename: str
    status: str  # "processing" or "completed"
    title: str = ""
    authors: List[str] = None
    abstract: str = ""
    executive_summary: str = ""
    key_findings: List[str] = None
    methodology: str = ""
    sections: List[Dict[str, str]] = None
    raw_text: str = ""
    embedding: Optional[np.ndarray] = None
    topics: List[str] = None

    def to_dict(self):
        d = asdict(self)
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d

# ------------------ Utilities ------------------
def make_id(filename: str) -> str:
    return f"{int(time.time()*1000)}_{filename}"

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        try:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
        except Exception:
            continue
    return "\n\n".join(texts).strip()

# ------------------ Hugging Face Inference API ------------------
def hf_call_inference(model: str, inputs: str, params: Dict[str, Any] = None, timeout: int = 120) -> str:
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing HUGGINGFACE_API_KEY environment variable.")
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{HF_API_URL_BASE}/{model}"
    payload = {"inputs": inputs}
    if params:
        payload["parameters"] = params
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Hugging Face inference API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Common formats: list with 'generated_text' or dict with 'summary_text'
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "summary_text" in data:
        return data["summary_text"]
    # fallback to str
        # Check if API returned an error message
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF API error: {data['error']}")
    # Check if model is loading
    if isinstance(data, dict) and "estimated_time" in data:
        raise RuntimeError("Model is loading, please try again in a moment")
    return json.dumps(data)

def hf_summarize(text: str, model: str, length: str = "medium") -> str:
    if not text.strip():
        return ""
    # mapping lengths to model params (for summarization models that accept min/max length)
    if length == "short":
        params = {"min_length": 20, "max_length": 80}
    elif length == "long":
        params = {"min_length": 150, "max_length": 600}
    else:
        params = {"min_length": 80, "max_length": 200}
    prompt = text if len(text) < 20000 else text[:20000]
    return hf_call_inference(model, prompt, params=params)

# ------------------ Citation & Reference Extraction ------------------
CITATION_PATTERNS = [
    r"\[\d+\]",                         # [1], [23]
    r"\([A-Z][A-Za-z\-]+ et al\., \d{4}\)",  # (Smith et al., 2020)
    r"\([A-Z][A-Za-z\-]+, \d{4}\)",     # (Smith, 2020)
    r"\b[A-Z][A-Za-z\-]+ et al\.\b",    # Smith et al.
    r"\bdoi:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+",  # doi:...
]
def extract_citations(text: str) -> List[str]:
    found = set()
    for pat in CITATION_PATTERNS:
        for m in re.findall(pat, text):
            found.add(m)
    refs = []
    lower = text.lower()
    if "references" in lower:
        try:
            idx = lower.index("references")
            refs_text = text[idx:]
            for line in refs_text.splitlines():
                line = line.strip()
                if not line: 
                    continue
                if re.match(r"^\[?\d+\]?", line) or "doi" in line.lower() or re.search(r"\d{4}", line):
                    refs.append(line)
            if refs:
                return list(dict.fromkeys(refs))[:50]
        except Exception:
            pass
    return sorted(found)

# ------------------ Embeddings, Search & Clustering ------------------
@st.cache_resource
def load_embedding_model(name: str):
    return SentenceTransformer(name)

def compute_embeddings_for_papers(model, papers: List[ResearchPaper]):
    texts = []
    ids = []
    for p in papers:
        txt = p.executive_summary or p.abstract or p.raw_text
        texts.append(txt if txt else p.filename)
        ids.append(p.id)
    if not texts:
        return
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    for i, p in enumerate(papers):
        p.embedding = embeddings[i]

def semantic_search(model, papers: List[ResearchPaper], query: str, top_k: int = 5):
    q_emb = model.encode(query, convert_to_numpy=True)
    candidates = []
    for p in papers:
        if p.embedding is None:
            continue
        score = util.cos_sim(q_emb, p.embedding).item()
        candidates.append((score, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:top_k]

def cluster_papers(papers: List[ResearchPaper], n_clusters: int = 4):
    X = []
    ids = []
    for p in papers:
        if p.embedding is not None:
            X.append(p.embedding)
            ids.append(p.id)
    if not X:
        return {}
    X = np.vstack(X)
    kmeans = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42).fit(X)
    clusters = {}
    for idx, label in enumerate(kmeans.labels_):
        pid = ids[idx]
        clusters.setdefault(int(label), []).append(pid)
    topics = {}
    for label, pids in clusters.items():
        words = []
        for pid in pids:
            p = next(p for p in papers if p.id == pid)
            txt = (p.executive_summary or p.abstract or "").lower()
            words += re.findall(r"\b[a-z]{4,}\b", txt)
        top = [w for w, c in pd.Series(words).value_counts().head(5).items()]
        topics[label] = top
    return {"clusters": clusters, "topics": topics}

# ------------------ Exports ------------------
def make_docx_summary(paper: ResearchPaper) -> bytes:
    doc = Document()
    doc.add_heading(paper.title or paper.filename, level=1)
    if paper.authors:
        doc.add_paragraph("Authors: " + ", ".join(paper.authors))
    doc.add_heading("Executive Summary", level=2)
    doc.add_paragraph(paper.executive_summary or paper.abstract or "")
    doc.add_heading("Key Findings", level=2)
    if paper.key_findings:
        for k in paper.key_findings:
            doc.add_paragraph(k, style="List Bullet")
    doc.add_heading("Methodology", level=2)
    doc.add_paragraph(paper.methodology or "")
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()

def make_pdf_summary(paper: ResearchPaper) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def sanitize_text_for_pdf(text: str) -> str:
    """Remove or replace characters that cause PDF encoding issues."""
    if not text:
        return ""
    # Replace problematic Unicode characters with ASCII equivalents
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove any remaining non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

    pdf.set_font("Arial", 'B', 14)
        pdf.multi_cell(0, 8, sanitize_text_for_pdf(paper.title or paper.filename))
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, sanitize_text_for_pdf("Executive Summary:\n" + (paper.executive_summary or paper.abstract or "")))
    pdf.ln(4)
    if paper.key_findings:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 6, "Key Findings:")
        pdf.ln(6)
        pdf.set_font("Arial", size=11)
        for k in paper.key_findings:
            pdf.multi_cell(0, 6, sanitize_text_for_pdf("- " + k))
    return pdf.output()

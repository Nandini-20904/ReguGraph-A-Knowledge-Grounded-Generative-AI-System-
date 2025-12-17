#!/usr/bin/env python3
"""
RBI AUTO PIPELINE + CHATBOT (ONE-COMMAND SYSTEM)

Add a new RBI document → pipeline runs → chatbot answers questions.

Usage:
  python rbi_auto_pipeline_and_chatbot.py --add_pdf data/new_doc.pdf
  python rbi_auto_pipeline_and_chatbot.py --ask "Which entities are covered under Outsourcing Directions?"
"""

import os
import re
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict

# ---------- CONFIG ----------
BASE_DIR = Path("data")
PDF_DIR = BASE_DIR / "rbi_docs"
TEXT_DIR = BASE_DIR / "rbi_extracted"
CHUNK_DIR = BASE_DIR / "semantic_chunks"
KG_DIR = BASE_DIR / "kg_auto"
VECTOR_DIR = BASE_DIR / "vector_db"

TEXT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
KG_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# ---------- EMBEDDINGS ----------
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
chroma = chromadb.Client(
    chromadb.config.Settings(
        persist_directory=str(VECTOR_DIR),
        anonymized_telemetry=False
    )
)
collection = chroma.get_or_create_collection("rbi_documents")

# ---------- LLM ----------
from groq import Groq
groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ============================================================
# 1. PDF → TEXT
# ============================================================

import pdfplumber

def pdf_to_text(pdf_path: Path) -> str:
    text = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)

# ============================================================
# 2. SEMANTIC CHUNKING
# ============================================================

def semantic_chunks(text: str, doc_id: str) -> List[Dict]:
    chunks = []
    current = []
    section = None

    for line in text.splitlines():
        if re.match(r"^\d+\.?\s+[A-Z]", line):
            if current:
                chunks.append({
                    "id": f"{doc_id}_{len(chunks)}",
                    "section": section,
                    "text": "\n".join(current)
                })
                current = []
            section = line.strip()
        current.append(line)

    if current:
        chunks.append({
            "id": f"{doc_id}_{len(chunks)}",
            "section": section,
            "text": "\n".join(current)
        })

    return chunks

# ============================================================
# 3. ONTOLOGY + KG EXTRACTION
# ============================================================

ACTORS = ["commercial bank", "bank", "regulated entity", "nbfc", "lsp"]
APPLICABILITY_PATTERNS = [
    "shall apply to",
    "applicable to",
    "these directions apply"
]

def extract_kg(chunks: List[Dict], doc_id: str):
    nodes = set()
    edges = []

    regulation_node = f"Regulation::{doc_id}"
    nodes.add(regulation_node)

    for c in chunks:
        cid = f"Clause::{c['id']}"
        nodes.add(cid)
        edges.append((cid, "partOf", regulation_node))

        text_lower = c["text"].lower()

        # Applicability
        if any(p in text_lower for p in APPLICABILITY_PATTERNS):
            for a in ACTORS:
                if a in text_lower:
                    actor_node = f"Actor::{a.replace(' ', '_').title()}"
                    nodes.add(actor_node)
                    edges.append((cid, "appliesTo", actor_node))

    return nodes, edges

# ============================================================
# 4. VECTOR STORE
# ============================================================

def store_embeddings(chunks: List[Dict]):
    for c in chunks:
        emb = embedder.encode(c["text"]).tolist()
        collection.add(
            ids=[c["id"]],
            documents=[c["text"]],
            embeddings=[emb],
            metadatas=[{"section": c["section"]}]
        )
    chroma.persist()

# ============================================================
# 5. ADD DOCUMENT PIPELINE
# ============================================================

def add_document(pdf_path: Path):
    doc_id = pdf_path.stem.lower().replace(" ", "_")

    print(f"[PIPELINE] Processing {pdf_path.name}")

    text = pdf_to_text(pdf_path)
    text_file = TEXT_DIR / f"{doc_id}.txt"
    text_file.write_text(text, encoding="utf-8")

    chunks = semantic_chunks(text, doc_id)
    json.dump(chunks, open(CHUNK_DIR / f"{doc_id}_chunks.json", "w"), indent=2)

    nodes, edges = extract_kg(chunks, doc_id)
    json.dump({
        "nodes": list(nodes),
        "edges": edges
    }, open(KG_DIR / f"{doc_id}_kg.json", "w"), indent=2)

    store_embeddings(chunks)

    print(f"[PIPELINE] Document added and indexed successfully")

# ============================================================
# 6. CHATBOT QUERY
# ============================================================

def retrieve_context(query: str):
    q_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=5)
    return results["documents"][0]

def answer_question(query: str):
    docs = retrieve_context(query)
    context = "\n\n".join(docs)

    prompt = f"""
You are an RBI regulatory assistant.
Answer ONLY using the context below.
If information is missing, say so.

CONTEXT:
{context}

QUESTION:
{query}
"""
    resp = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# ============================================================
# 7. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_pdf", help="Add RBI PDF")
    parser.add_argument("--ask", help="Ask chatbot")
    args = parser.parse_args()

    if args.add_pdf:
        add_document(Path(args.add_pdf))

    if args.ask:
        print("\n🤖", answer_question(args.ask))

if __name__ == "__main__":
    main()

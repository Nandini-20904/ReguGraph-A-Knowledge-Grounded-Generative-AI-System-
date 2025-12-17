#!/usr/bin/env python3

"""
store_in_vector_db.py — FINAL WORKING VERSION
--------------------------------------------
✔ Uses new Chroma PersistentClient API
✔ No deprecated Settings()
✔ No migration errors
✔ Works with newest chroma versions
✔ Just run: python store_in_vector_db.py
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import logging

from sentence_transformers import SentenceTransformer

# NEW correct import for persistent client
from chromadb import PersistentClient

# ---------------------------
# CONFIG — EDIT IF NEEDED
# ---------------------------

CHUNKS_DIR = "./data/semantic_chunks"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rbi_directives"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 128

# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("store_in_vector_db")


# ---------------------------
# LOAD CHUNKS
# ---------------------------

def load_chunks():
    p = Path(CHUNKS_DIR)
    if not p.exists():
        raise FileNotFoundError(f"Chunks directory not found: {CHUNKS_DIR}")

    chunk_files = sorted(p.glob("*_chunks.json"))
    all_chunks = []

    for file in chunk_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for c in data:
            if "chunk_id" not in c:
                c["chunk_id"] = f"chunk_{len(all_chunks)}"
            c["_source_file"] = file.name
            all_chunks.append(c)

    log.info(f"Loaded {len(all_chunks)} chunks from {len(chunk_files)} files")
    return all_chunks


# ---------------------------
# METADATA HELPER
# ---------------------------

def prepare_metadata(chunk):
    meta = chunk.get("meta", {}) or {}
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk.get("doc_id"),
        "type": chunk.get("type"),
        "source_file": chunk.get("_source_file"),
        "section_number": meta.get("section_no"),
        "has_penalty": meta.get("has_penalty", False),
        "entities": ",".join([e["text"] for e in meta.get("entities", [])]) if meta.get("entities") else None,
        "keywords": ",".join(meta.get("keywords", [])) if meta.get("keywords") else None,
        "crossrefs": ",".join(meta.get("crossrefs", [])) if meta.get("crossrefs") else None,
        "preview": (chunk.get("text") or "")[:200],
    }


# ---------------------------
# MAIN PIPELINE
# ---------------------------

def main():

    # 1. Load chunks
    chunks = load_chunks()
    if not chunks:
        log.error("No chunks found. Exiting.")
        return

    # 2. Load embedding model
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # 3. Create Chroma persistent client
    log.info(f"Using Persistent Chroma DB at: {PERSIST_DIR}")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = PersistentClient(path=PERSIST_DIR)

    # 4. Create or get collection
    try:
        collection = client.get_collection(COLLECTION_NAME)
        log.info(f"Using existing collection: {COLLECTION_NAME}")
    except:
        collection = client.create_collection(COLLECTION_NAME)
        log.info(f"Created new collection: {COLLECTION_NAME}")

    # 5. Insert in batches
    log.info("Embedding & inserting into Chroma...")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i + BATCH_SIZE]

        ids = [c["chunk_id"] for c in batch]
        docs = [c["text"] for c in batch]
        metas = [prepare_metadata(c) for c in batch]
        embs = model.encode(docs, convert_to_numpy=True).tolist()

        # upsert
        collection.upsert(
            ids=ids,
            embeddings=embs,
            documents=docs,
            metadatas=metas
        )

    log.info("DONE! All chunks stored in persistent Chroma DB.")


# ---------------------------
# ENTRY POINT
# ---------------------------

if __name__ == "__main__":
    main()

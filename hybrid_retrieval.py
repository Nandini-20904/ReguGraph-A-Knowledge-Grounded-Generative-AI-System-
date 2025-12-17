# hybrid_retrieval_v3.py
# Updated for Ontology V3, domain-filtered topics, and new chunk layout

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from kg_retrieval import get_topic_related_nodes, get_kg_facts

# --------------------------------------------------------
# LOAD ALL CHUNKS (NEW FORMAT FOR V3)
# --------------------------------------------------------

def load_all_chunks(chunk_dir="data/semantic_chunks"):
    id2text = {}
    all_ids = []

    for f in Path(chunk_dir).glob("*_chunks.json"):
        data = json.load(open(f, "r", encoding="utf-8"))

        if isinstance(data, dict) and "chunks" in data:
            chunks = data["chunks"]
        else:
            chunks = data

        for ch in chunks:
            cid = ch.get("chunk_id")
            if not cid:
                continue
            id2text[cid] = ch.get("text", "")
            all_ids.append(cid)

    return id2text, all_ids


id2text, all_ids = load_all_chunks()
model = SentenceTransformer("all-mpnet-base-v2")
all_embeddings = model.encode([id2text[i] for i in all_ids], convert_to_tensor=True)


# --------------------------------------------------------
# RAG VECTOR SEARCH
# --------------------------------------------------------
def rag_search(query, top_k=5):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_emb, all_embeddings)[0]
    top = scores.topk(top_k)

    results = [(all_ids[idx], id2text[all_ids[idx]]) for idx in top.indices]
    return results


# --------------------------------------------------------
# HYBRID RETRIEVAL PIPELINE (KG-FIRST + RAG)
# --------------------------------------------------------
def hybrid_retrieve(query, topic_key):
    """
    Topic key = "DLG_Cap" etc.
    """

    # 1. KG nodes for topic
    kg_chunk_ids = get_topic_related_nodes(topic_key)     # returns Chunk::xxxx or Clause::xxxx

    # Normalize → only chunk IDs
    normalized_kg_chunks = []
    for nid in kg_chunk_ids:
        if nid.startswith("Chunk::"):
            normalized_kg_chunks.append(nid.replace("Chunk::", ""))
        elif nid.startswith("Clause::"):
            normalized_kg_chunks.append(nid.replace("Clause::", ""))

    # 2. RAG retrieval
    rag_results = rag_search(query)
    rag_ids = [cid for cid, _ in rag_results]

    # 3. Merge
    merged_ids = list(set(normalized_kg_chunks + rag_ids))

    # 4. Get KG facts
    kg_facts = get_kg_facts(["Chunk::" + cid for cid in merged_ids])

    # 5. Prepare final chunks
    final_chunks = [(cid, id2text.get(cid, "")) for cid in merged_ids]

    return final_chunks, kg_facts

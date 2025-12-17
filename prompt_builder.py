# prompt_builder_v3.py
# Updated for Ontology V3 and KG context grouping

def build_prompt(question, chunks, kg_facts):

    # -----------------------------
    # GROUP KG FACTS BY RELATION
    # -----------------------------
    grouped = {}
    for f in kg_facts:
        rel = f["relation"]
        grouped.setdefault(rel, []).append(f["label"])

    # Remove duplicates
    for rel in grouped:
        grouped[rel] = list(set(grouped[rel]))

    # Format KG info
    kg_section = []
    for rel, items in grouped.items():
        line = f"{rel}: " + ", ".join(items)
        kg_section.append(line)

    kg_text = "\n".join(kg_section)


    # -----------------------------
    # FORMAT DOCUMENT CHUNKS
    # -----------------------------
    chunk_text = "\n\n".join(
        f"[{cid}]: {txt[:1200]}"
        for cid, txt in chunks
    )


    # -----------------------------
    # FINAL PROMPT
    # -----------------------------
    prompt = f"""
You are an RBI regulatory assistant.

User question:
{question}

================ KG INFO ================
{kg_text}

================ DOCUMENT EXCERPTS ================
{chunk_text}

================ INSTRUCTIONS ================
Answer STRICTLY using the above KG + document context.
Do NOT hallucinate.
If information is missing, respond:
"I cannot find this information in the RBI documents."
"""

    return prompt

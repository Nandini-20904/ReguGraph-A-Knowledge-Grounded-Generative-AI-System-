#!/usr/bin/env python3
"""
chatbot_backend.py — V4.1 (STABLE + FAST)

Fixes:
- JSONDecodeError crash fixed (safe LLM parsing)
- Dynamic intent (chit-chat vs RBI)
- Lazy SBERT loading (fast startup)
- KG + RAG preserved
- Follow-ups preserved
- No Redis, no crashes
"""

# ----------------- HARD FIXES (MUST BE AT TOP) -----------------
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ---------------------------------------------------------------
import re
import json
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from groq import Groq

from hybrid_retrieval import hybrid_retrieve
from prompt_builder import build_prompt

# ---------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbi-chatbot-v4.1")

# ----------------- CONFIG -----------------
GROQ_API_KEY = "groq-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

groq_client = Groq(api_key=GROQ_API_KEY)

# ----------------- LAZY SBERT -----------------
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        logger.info("Lazy-loading SBERT...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model

# ----------------- MEMORY -----------------
conversation_memory: Dict[str, str] = {}

def get_last_reply(cid: str) -> str:
    return conversation_memory.get(cid, "")

def set_last_reply(cid: str, reply: str):
    conversation_memory[cid] = reply

def clear_conversation(cid: str):
    conversation_memory.pop(cid, None)

# ----------------- FOLLOW-UP DETECTION -----------------
FOLLOWUP_PATTERNS = [
    "explain again", "repeat", "again", "clarify",
    "more clearly", "elaborate", "explain that"
]

def is_followup(prev: str, q: str) -> bool:
    if not prev:
        return False
    ql = q.lower().strip()
    if any(p in ql for p in FOLLOWUP_PATTERNS):
        return True
    if len(q.split()) <= 4:
        model = get_embed_model()
        sim = util.cos_sim(
            model.encode(prev, convert_to_tensor=True),
            model.encode(q, convert_to_tensor=True)
        ).item()
        return sim > 0.55
    return False

# ----------------- FOLLOW-UP REWRITE -----------------
def rewrite_followup(prev: str, q: str) -> str:
    prompt = f"""
Rewrite this follow-up into a complete RBI regulatory question.

PREVIOUS ANSWER:
{prev}

FOLLOW-UP:
{q}

Return ONLY the rewritten question.
"""
    r = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=80
    )
    return r.choices[0].message.content.strip()

# ----------------- TOPIC FALLBACK -----------------
TOPIC_KEYWORDS = {
    "DLG_Cap": ["dlg", "fldg", "first loss"],
    "Gold_Loan_LTV": ["gold", "ltv"],
    "ECL_Overview": ["ecl", "expected credit"],
    "KYC_Process": ["kyc"],
    "AML_Compliance": ["aml"],
    "Model_Governance_Framework": ["model governance"]
}

def detect_topic_fallback(q: str) -> str:
    ql = q.lower()
    for t, kws in TOPIC_KEYWORDS.items():
        if any(k in ql for k in kws):
            return t
    return "DLG_Cap"

# ----------------- SAFE INTENT CLASSIFIER (FIXED) -----------------
def llm_intent(q: str) -> Tuple[str, Optional[str]]:
    ql = q.lower().strip()

    # 1️⃣ Greeting shortcut (NO LLM)
    if ql in {"hi", "hello", "hey", "hii"} or ql.startswith(("hi ", "hello ", "hey ")):
        return "chit_chat", None

    # 2️⃣ Keyword shortcut (NO LLM)
    if any(k in ql for k in ["rbi", "loan", "dlg", "fldg", "cap", "ltv", "kyc", "ecl"]):
        return "rbi_query", detect_topic_fallback(q)

    # 3️⃣ LLM classifier (SAFE)
    prompt = f"""
Return ONLY valid JSON:
{{"intent":"chit_chat","topic":null}} OR {{"intent":"rbi_query","topic":"DLG_Cap"}}

User message:
{q}
"""
    try:
        r = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=30
        )

        raw = (r.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("intent", "rbi_query"), data.get("topic")

    except Exception as e:
        logger.warning("Intent fallback used: %s", e)

    return "rbi_query", detect_topic_fallback(q)

# ----------------- RETRIEVAL -----------------
def retrieve(query: str, topic: str):
    chunks, kg_facts = hybrid_retrieve(query, topic)
    chunks_for_prompt = []
    chunks_used = []
    for cid, text in chunks:
        cid = cid.replace("Chunk::", "")
        chunks_for_prompt.append({"id": cid, "text": text})
        chunks_used.append({"id": cid, "preview": text[:400]})
    return chunks_for_prompt, kg_facts, chunks_used

# ----------------- PROMPT -----------------
def build_llm_prompt(q, chunks, kg_facts, prev):
    body = build_prompt(q, [(c["id"], c["text"]) for c in chunks], kg_facts)
    return f"""
SYSTEM:
You are an RBI regulatory assistant.
Use ONLY the provided context.
If info is missing say so.

PREVIOUS ANSWER:
{prev or "(none)"}

{body}

TASK:
Answer precisely.
"""

# ----------------- LLM CALL -----------------
def call_llm(prompt: str, temp=0):
    r = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=600
    )
    return r.choices[0].message.content.strip()

# ----------------- FASTAPI -----------------
app = FastAPI(title="RBI Chatbot v4.1")

class Ask(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    clear: Optional[bool] = False

@app.post("/ask")
def ask(req: Ask):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    cid = req.conversation_id or str(uuid.uuid4())
    if req.clear:
        clear_conversation(cid)

    q = req.question.strip()
    prev = get_last_reply(cid)

    intent, topic = llm_intent(q)
    logger.info("Intent=%s Topic=%s", intent, topic)

    # --------- CHIT CHAT ---------
    if intent == "chit_chat":
        reply = call_llm(
            f"You are friendly. Reply briefly to: {q}",
            temp=0.7
        )
        set_last_reply(cid, reply)
        return {"conversation_id": cid, "answer": reply, "chunks_used": [], "kg_facts": []}

    # --------- FOLLOW-UP ---------
    if is_followup(prev, q):
        q = rewrite_followup(prev, q)

    chunks, kg_facts, used = retrieve(q, topic)

    if not chunks and not kg_facts:
        reply = "I cannot find this information in the provided RBI documents."
        set_last_reply(cid, reply)
        return {"conversation_id": cid, "answer": reply, "chunks_used": [], "kg_facts": []}

    prompt = build_llm_prompt(q, chunks, kg_facts, prev)
    answer = call_llm(prompt)

    set_last_reply(cid, answer)
    return {
        "conversation_id": cid,
        "answer": answer,
        "chunks_used": used,
        "kg_facts": kg_facts
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL}

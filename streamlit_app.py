import streamlit as st
import requests
import uuid
import json

API_URL = "http://localhost:8000/ask"   # FastAPI backend

st.set_page_config(
    page_title="RBI Chatbot (KG + RAG v3)",
    page_icon="📘",
    layout="centered"
)

st.title("📘 RBI Regulatory Chatbot — V3 (KG + RAG Enhanced)")

# ---------------------------------------------------------
# Session Variables
# ---------------------------------------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


# ---------------------------------------------------------
# Suggested Questions
# ---------------------------------------------------------
st.subheader("💡 Suggested Questions")

suggestions = [
    "Explain the cap on FLDG",
    "What are the RBI gold loan LTV rules?",
    "What are the model validation requirements under Model Governance?",
    "What is the Expected Credit Loss framework?",
    "Explain the DLG eligibility criteria.",
    "Explain KFS rules for digital lending."
]

cols = st.columns(2)
for i, q in enumerate(suggestions):
    if cols[i % 2].button(q):
        st.session_state.input_text = q

st.markdown("---")


# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------
user_question = st.text_input(
    "Ask your RBI regulatory question:",
    value=st.session_state.input_text,
    placeholder="Type your question..."
)

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# ASK BUTTON
# ---------------------------------------------------------
with col1:
    if st.button("Ask", use_container_width=True):
        if not user_question.strip():
            st.error("Please enter a question.")
        else:

            payload = {
                "question": user_question,
                "conversation_id": st.session_state.conversation_id,
                "clear": False
            }

            try:
                response = requests.post(API_URL, json=payload).json()

                # Backend returns:
                # - answer
                # - rewritten_query (optional)
                # - chunks_used (list)
                # - kg_facts (list)
                answer = response.get("answer", "")
                rewritten = response.get("rewritten_query")
                chunks_used = response.get("chunks_used", [])
                kg_facts = response.get("kg_facts", [])

                # Construct displayed answer
                if rewritten:
                    final_answer = f"🔁 **Rewritten Query:** {rewritten}\n\n{answer}"
                else:
                    final_answer = answer

                # Save chat history
                st.session_state.chat_history.append(
                    {"role": "user", "text": user_question}
                )
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "text": final_answer,
                        "chunks": chunks_used,
                        "facts": kg_facts
                    }
                )

                st.session_state.input_text = ""  # Clear the input box

            except Exception as e:
                st.error(f"Error contacting backend: {e}")


# ---------------------------------------------------------
# CLEAR CHAT
# ---------------------------------------------------------
with col2:
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.input_text = ""
        st.success("Conversation cleared!")


st.markdown("---")


# ---------------------------------------------------------
# Render Chat History
# ---------------------------------------------------------
st.subheader("🗨 Chat History")

for msg in st.session_state.chat_history:

    # USER MESSAGE
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='background-color:#e3f2fd;padding:12px;border-radius:10px;margin-bottom:6px'>
            <strong>🧑 You:</strong> {msg['text']}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ASSISTANT MESSAGE
    else:
        st.markdown(
            f"""
            <div style='background-color:#f1f3f4;padding:12px;border-radius:10px;margin-bottom:6px'>
            <strong>🤖 Assistant:</strong><br>{msg['text']}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Debug info panel (KG facts + Chunk sources)
        with st.expander("📄 View Chunk Sources & KG Facts (Debug Mode)"):
            
            # ---- Chunk Sources ----
            st.write("### 📌 Document Chunks Used:")
            for c in msg.get("chunks", []):
                st.markdown(f"**{c['id']}** — `{c['preview']}`")

            # ---- KG Facts ----
            st.write("### 🧠 Knowledge Graph Facts:")
            if msg.get("facts"):
                for f in msg["facts"]:
                    st.markdown(
                        f"- **{f['source']}** → *{f['relation']}* → **{f['label']}**"
                    )
            else:
                st.write("No KG facts used for this answer.")


st.markdown("---")

st.caption(f"Conversation ID: {st.session_state.conversation_id}")
st.caption("Frontend Version 3.0 • KG + RAG Enhanced Chatbot")

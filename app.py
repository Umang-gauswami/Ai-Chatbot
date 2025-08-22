import os, time
from typing import List, Dict
import streamlit as st
from dotenv import load_dotenv
from rag import RAGSearcher

# Optional: OpenAI for LLM fallback
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Load API key and defaults from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Streamlit page setup
st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="centered")
st.title("AI Chatbot")
st.caption("Ask me anything about orders, refunds, shipping, and policies.")

# Sidebar (only reset + settings, no API key exposure)
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Knowledge Base confidence", 0.0, 1.0, 0.60, 0.01)

    if st.button("🔄 Reset chat"):
        st.session_state.messages = []

# Initialize RAG
@st.cache_resource(show_spinner=True)
def load_searcher():
    return RAGSearcher()

searcher = load_searcher()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict] = [
        {"role": "assistant", "content": "Hi! 👋 Ask me anything about orders, refunds, shipping, and policies."}
    ]

# Render chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG lookup
    results = searcher.search(prompt, top_k=3)
    best = results[0] if results else None
    response_text = ""

    if best and best[0] >= threshold:
        # Answer from Knowledge Base
        response_text = best[1]["answer"]
        source_q = best[1]["question"]
        with st.chat_message("assistant"):
            st.markdown(response_text)
            st.caption(f"📚 From Knowledge Base (match: {best[0]:.2f}) — source: “{source_q}”")

    else:
        # Fallback to LLM (only if API key is available)
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            client = OpenAI(api_key=OPENAI_API_KEY)

            context_chunks = "\n\n".join([f"Q: {r[1]['question']}\nA: {r[1]['answer']}" for r in results])
            system = (
                "You are a helpful support assistant. "
                "Answer concisely and ask for the order ID if the user asks about their order/refund. "
                "Use the provided knowledge base context if possible; otherwise say you are unsure."
            )
            user = f"User question: {prompt}\n\nKnowledge base context (may be empty):\n{context_chunks}"

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full = ""
                try:
                    stream = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        stream=True,
                        temperature=0.3,
                    )
                    for event in stream:
                        delta = event.choices[0].delta.content or ""
                        full += delta
                        placeholder.markdown(full)
                        time.sleep(0.01)
                    response_text = full.strip()
                except Exception:
                    response_text = "⚠️ Sorry, I'm having trouble right now. Please try again later."
                    placeholder.markdown(response_text)
        else:
            # No LLM available
            response_text = (
                "I couldn’t find a confident answer in the knowledge base. "
                "Please provide more details (like your order ID or date)."
            )
            with st.chat_message("assistant"):
                st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})

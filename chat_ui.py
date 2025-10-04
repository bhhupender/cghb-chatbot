import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss, pickle

# Load env
load_dotenv()
client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

# Load FAISS index + metadata
embedder = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
else:
    st.error("⚠️ No FAISS index found. Please upload documents and build index.")
    st.stop()

def query_rag(query, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    docs = []
    for idx in I[0]:
        if idx < len(metadata):
            docs.append(metadata[idx])
    return docs

def generate_answer(user_message: str, context_docs: list = []):
    context_text = "\n\n".join([f"SOURCE: {d['source']}\n{d['text']}" for d in context_docs])
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are CGHB Assistant. Reply in Hindi or English based only on context docs."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nUser: {user_message}"}
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="CGHB Chatbot", page_icon="🏠")
st.title("🏠 CGHB AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show messages
for msg in st.session_state.messages:
    if msg["sender"] == "user":
        st.chat_message("user").markdown(msg["text"])
    else:
        st.chat_message("assistant").markdown(msg["text"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📄 Sources: {', '.join(msg['sources'])}")

# Input box
if user_input := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"sender": "user", "text": user_input})
    st.chat_message("user").markdown(user_input)

    docs = query_rag(user_input, top_k=3)
    answer = generate_answer(user_input, docs)
    sources = [d["source"] for d in docs]

    st.session_state.messages.append({"sender": "bot", "text": answer, "sources": sources})
    st.chat_message("assistant").markdown(answer)
    if sources:
        st.caption(f"📄 Sources: {', '.join(sources)}")

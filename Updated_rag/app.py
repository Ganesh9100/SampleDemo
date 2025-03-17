import streamlit as st
import faiss
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === CONFIGURATION ===
INDEX_PATH = "faiss_index"
USE_RAM = False  # Should match the setting in indexing.py

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
if USE_RAM:
    index = faiss.IndexFlatL2(embed_model.get_sentence_embedding_dimension())
else:
    index = faiss.read_index(f"{INDEX_PATH}.faiss")
    with open(f"{INDEX_PATH}_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

# Load Gemma 3B model
@st.cache_resource
def load_gemma():
    return pipeline("text-generation", model="google/gemma-3b", device="cuda" if torch.cuda.is_available() else "cpu")

gemma_model = load_gemma()

def retrieve_relevant_chunks(query, n_results=3):
    """Finds the top similar chunks using FAISS."""
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, n_results)
    
    if USE_RAM:
        return [metadata[i]["text"] for i in indices[0] if i != -1]
    else:
        return [metadata[i]["filename"] for i in indices[0] if i != -1]

def generate_response(query):
    """Generates a response using RAG with Gemma 3B."""
    relevant_chunks = retrieve_relevant_chunks(query)
    
    if not relevant_chunks:
        return "I couldn't find relevant information in the knowledge base."

    context = "\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = gemma_model(prompt, max_length=300, do_sample=True)
    return response[0]["generated_text"]

# === STREAMLIT UI ===
st.title("📚 FAISS RAG Chatbot with Gemma 3B")
st.sidebar.header("Settings")

if st.sidebar.button("Re-index Knowledge Base"):
    import indexing
    indexing.index_knowledge_base("knowledge_base")

user_query = st.text_input("Ask a question:")
if st.button("Get Answer"):
    if user_query:
        response = generate_response(user_query)
        st.write(response)
    else:
        st.error("Please enter a question.")

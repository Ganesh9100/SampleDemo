# import streamlit as st
# import faiss
# import torch
# import pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# # === CONFIGURATION ===
# INDEX_PATH = "faiss_index"
# USE_RAM = False  # Should match the setting in indexing.py

# # Load embedding model
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load FAISS index
# if USE_RAM:
#     index = faiss.IndexFlatL2(embed_model.get_sentence_embedding_dimension())
# else:
#     index = faiss.read_index(f"{INDEX_PATH}.faiss")
#     with open(f"{INDEX_PATH}_metadata.pkl", "rb") as f:
#         metadata = pickle.load(f)

# # Load Gemma 3B model
# @st.cache_resource
# def load_gemma():
#     return pipeline("text-generation", model="google/gemma-3b", device="cuda" if torch.cuda.is_available() else "cpu")

# gemma_model = load_gemma()

# def retrieve_relevant_chunks(query, n_results=3):
#     """Finds the top similar chunks using FAISS."""
#     query_embedding = embed_model.encode([query])
#     distances, indices = index.search(query_embedding, n_results)
    
#     if USE_RAM:
#         return [metadata[i]["text"] for i in indices[0] if i != -1]
#     else:
#         return [metadata[i]["filename"] for i in indices[0] if i != -1]

# def generate_response(query):
#     """Generates a response using RAG with Gemma 3B."""
#     relevant_chunks = retrieve_relevant_chunks(query)
    
#     if not relevant_chunks:
#         return "I couldn't find relevant information in the knowledge base."

#     context = "\n".join(relevant_chunks)
#     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

#     response = gemma_model(prompt, max_length=300, do_sample=True)
#     return response[0]["generated_text"]

# # === STREAMLIT UI ===
# st.title("📚 FAISS RAG Chatbot with Gemma 3B")
# st.sidebar.header("Settings")

# if st.sidebar.button("Re-index Knowledge Base"):
#     import indexing
#     indexing.index_knowledge_base("knowledge_base")

# user_query = st.text_input("Ask a question:")
# if st.button("Get Answer"):
#     if user_query:
#         response = generate_response(user_query)
#         st.write(response)
#     else:
#         st.error("Please enter a question.")


import streamlit as st
import faiss
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

import numpy as np
metadata = np.load("faiss_index/metadata.pkl", allow_pickle=True)
print(metadata)  # Should print a list of document contents, NOT filenames.


# Load the Gemma model and tokenizer
@st.cache_resource
def load_gemma_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16)
    return tokenizer, model

# Load the FAISS index and metadata
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index("faiss_index/faiss.index")
    metadata = np.load("faiss_index/metadata.pkl", allow_pickle=True)  # List of document contents
    return index, metadata

# Retrieve relevant documents based on query
def retrieve_documents(query, index, metadata, retriever_model, top_k=3):
    query_embedding = retriever_model.encode([query], convert_to_numpy=True)

    if len(query_embedding.shape) == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)  # Ensure 2D

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):  # Check bounds
            doc_content = metadata[idx]  # Fetch document content
            results.append((doc_content, distances[0][i]))
        else:
            print(f"⚠️ Warning: FAISS returned out-of-bounds index {idx}")

    return results

def main():
    st.title("📚 Knowledge Base Q&A with Gemma 3 1B")

    # Load models and index
    tokenizer, gemma_model = load_gemma_model()
    index, metadata = load_index_and_metadata()
    retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

    query = st.text_input("Enter your question:")

    if query:
        results = retrieve_documents(query, index, metadata, retriever_model)
        print("Retrieved Results:", results)

        context = "\n".join([doc_content for doc_content, _ in results])

        if context.strip():
            input_text = f"Question: {query}\nContext: {context}\nAnswer:"
            inputs = tokenizer(input_text, return_tensors="pt").to(gemma_model.device)

            with torch.no_grad():
                outputs = gemma_model.generate(**inputs, max_new_tokens=150)

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            answer = "I couldn't find relevant documents for your query."

        st.write("### 🤖 AI Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()

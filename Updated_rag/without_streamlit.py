import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# === LOAD FAISS INDEX AND METADATA ===
index = faiss.read_index("faiss_index/faiss.index")
metadata = np.load("faiss_index/metadata.pkl", allow_pickle=True)  # Should contain document contents

# === LOAD GEMMA MODEL ===
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16)
model.eval()  # Set to evaluation mode (no training)

# === LOAD SENTENCE EMBEDDING MODEL ===
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

# === RETRIEVE DOCUMENTS BASED ON QUERY ===
def retrieve_documents(query, top_k=3):
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

# === GENERATE RESPONSE USING GEMMA ===
def generate_response(query):
    results = retrieve_documents(query)
    print("🔍 Retrieved Results:", results)

    context = "\n".join([doc_content for doc_content, _ in results])

    if context.strip():
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        answer = "I couldn't find relevant documents for your query."

    return answer

# === INTERACTIVE QUERY IN JUPYTER NOTEBOOK ===
while True:
    query = input("\n❓ Enter your question (or type 'exit' to stop): ")
    if query.lower() == "exit":
        break
    response = generate_response(query)
    print("\n🤖 AI Answer:", response)

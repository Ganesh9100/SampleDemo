import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load JSON Data
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract text content from JSON
documents = []
for item in data:  # Modify this based on your JSON structure
    content = item.get("description", "")  # Example: Extract 'description'
    documents.append(content)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to embeddings
embeddings = embed_model.encode(documents, convert_to_numpy=True)

# Create FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"✅ FAISS Index Created with {len(documents)} documents")


def retrieve_relevant_chunks(query, top_k=3):
    """Finds the most similar text chunks using FAISS."""
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = [documents[i] for i in indices[0] if i != -1]
    return results


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Gemma model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3b").to(device)

def generate_answer_gemma(query):
    """Retrieves relevant chunks and generates an answer using Gemma."""
    relevant_chunks = retrieve_relevant_chunks(query)
    
    context = "\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    query = input("Ask a question: ")
    if query.lower() == "exit":
        break
    answer = generate_answer_gemma(query)
    print("\n🤖 AI Answer:", answer, "\n")

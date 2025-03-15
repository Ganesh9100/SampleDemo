# import faiss
# import numpy as np
# import json
# from sentence_transformers import SentenceTransformer

# FAISS_INDEX_FILE = "phones.index"
# JSON_FILE = "phones.json"

# def create_faiss_index():
#     """Create FAISS index from phone data."""
#     print("Loading phone data...")
    
#     with open(JSON_FILE, "r", encoding="utf-8") as f:
#         phone_data = json.load(f)

#     if not phone_data:
#         raise ValueError("No phone data found. Ensure phones.json is not empty.")

#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     # Convert phone data to embeddings
#     texts = [f"{item['title']} {item['specs']} {item['price']}" for item in phone_data]
#     embeddings = model.encode(texts, convert_to_numpy=True)

#     print(f"Embedding shape: {embeddings.shape}")

#     # Create FAISS index
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)

#     # Save index
#     faiss.write_index(index, FAISS_INDEX_FILE)
#     print(f"FAISS index saved as {FAISS_INDEX_FILE}")

# if __name__ == "__main__":
#     create_faiss_index()


import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Define paths
KB_PATH = "knowledgebase"
INDEX_PATH = "faiss_index"

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read knowledge base files
documents = []
file_names = []

for file_name in os.listdir(KB_PATH):
    if file_name.endswith(".txt"):
        with open(os.path.join(KB_PATH, file_name), "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)
            file_names.append(file_name)

# Generate embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, os.path.join(INDEX_PATH, "faiss.index"))
with open(os.path.join(INDEX_PATH, "metadata.pkl"), "wb") as f:
    pickle.dump(file_names, f)

print("FAISS index created successfully!")
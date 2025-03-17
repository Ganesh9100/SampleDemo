import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# === CONFIGURATION ===
INDEX_PATH = "faiss_index"
USE_RAM = False  # Set to True for in-memory FAISS

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500):
    """Splits text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def load_text_file(file_path):
    """Reads plain text files."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_csv_file(file_path):
    """Reads CSV files and extracts text data."""
    df = pd.read_csv(file_path)
    return " ".join(df.astype(str).agg(" ".join, axis=1).tolist())

def load_json_file(file_path):
    """Reads JSON files assuming they contain text data."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return json.dumps(data)  # Convert JSON to string format

def index_knowledge_base(folder_path):
    """Processes all files, chunks text, and indexes them using FAISS."""
    texts, metadatas = [], []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith(".txt"):
            content = load_text_file(file_path)
        elif filename.endswith(".csv"):
            content = load_csv_file(file_path)
        elif filename.endswith(".json"):
            content = load_json_file(file_path)
        else:
            continue  # Skip unsupported file types
        
        chunks = chunk_text(content)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"filename": filename})
    
    if not texts:
        print("No valid data found for indexing.")
        return

    # Generate embeddings
    embeddings = embed_model.encode(texts)
    d = embeddings.shape[1]  # Embedding size

    # Initialize FAISS index
    index = faiss.IndexFlatL2(d) if USE_RAM else faiss.IndexIDMap(faiss.IndexFlatL2(d))

    # Store embeddings in FAISS
    ids = np.array(range(len(embeddings))).astype(np.int64)
    index.add_with_ids(embeddings, ids)

    # Save index to disk if not using RAM
    if not USE_RAM:
        faiss.write_index(index, f"{INDEX_PATH}.faiss")
        with open(f"{INDEX_PATH}_metadata.pkl", "wb") as f:
            pickle.dump(metadatas, f)
    
    print("✅ FAISS indexing complete!")

if __name__ == "__main__":
    index_knowledge_base("knowledge_base")

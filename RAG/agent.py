import streamlit as st
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_gemma_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16)
    return tokenizer, model

@st.cache_resource
def load_index_and_files():
    index = faiss.read_index('document_index.faiss')
    file_names = np.load('file_names.npy', allow_pickle=True)
    return index, file_names

def retrieve_documents(query, index, file_names, model, top_k=3):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return [(file_names[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

def main():
    st.title("Knowledge Base Q&A with Gemma 3 1B")
    tokenizer, gemma_model = load_gemma_model()
    index, file_names = load_index_and_files()
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

    query = st.text_input("Enter your question:")
    if query:
        results = retrieve_documents(query, index, file_names, retriever_model)
        context = ""
        for file_name, distance in results:
            with open(os.path.join('knowledge_base', file_name), 'r', encoding='utf-8') as file:
                context += file.read() + "\n"

        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt").to(gemma_model.device)
        outputs = gemma_model.generate(**inputs, max_new_tokens=150)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(answer)

if __name__ == "__main__":
    main()

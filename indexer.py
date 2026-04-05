import faiss
import pickle
import numpy as np
from pdf_processor import process_pdf
from embedder import get_embedding

def build_index(pdf_path):
    chunks = process_pdf(pdf_path)
    embedding = [get_embedding(chunk) for chunk in chunks]
    embedding = np.array(embedding)

    dimension = embedding.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding) # type: ignore

    faiss.write_index(index, 'index/faiss.index')
    with open('index/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)

    return index

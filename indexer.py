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

    # IndexFlatL2: brute-force exact search using L2 (Euclidean) distance.
    # Suitable here since corpus size is small (few hundred chunks from a PDF).
    # For large-scale use, consider IndexIVFFlat (cluster-based) or IndexHNSW (graph-based).
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding) # type: ignore

    faiss.write_index(index, 'index/faiss.index')
    with open('index/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)

    return index

import faiss
import pickle
import numpy as np
import os
from groq import Groq
from embedder import get_embedding
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    


def search(query, top_k = 3):
        index = faiss.read_index('index/faiss.index')
        with open('index/chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)

        query_embedding = get_embedding(query)
        query_embedding = np.array([query_embedding])

        distance, indices = index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({'text' : chunks[idx],
                            'distance' : distance[0][i]
                            })
            
        context = " ".join([r['text'] for r in results])
        context_str = f"Answer this question based only on the context below.\n\nContext: {context}\n\nQuestion: {query}"

        response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": context_str}]
        )
        
        return response.choices[0].message.content
# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions from uploaded PDF documents using LLMs.

## How it works
- User uploads a PDF document via the Streamlit interface
- Text is extracted using pdfplumber and split into chunks using LangChain
- Chunks are converted into vector embeddings using Sentence Transformers (MiniLM)
- Embeddings are stored in a FAISS index for fast similarity search
- User asks a question — it gets embedded and top 3 relevant chunks are retrieved
- Retrieved chunks + question are sent to Groq LLM (LLaMA3) to generate an accurate answer

## Tech Stack
- Python
- pdfplumber (PDF text extraction)
- LangChain (text chunking)
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS (Facebook AI Similarity Search)
- Groq API (LLaMA3 — llama-3.3-70b-versatile)
- Streamlit

## Project Structure
rag-knowledge-assistant/
├── app.py              # Streamlit UI
├── pdf_processor.py    # PDF extraction and chunking
├── embedder.py         # Text to embedding conversion
├── indexer.py          # Builds and saves FAISS index
├── retriever.py        # Retrieves chunks and calls Groq LLM
├── user_files/         # Uploaded PDFs stored here
├── index/
│   ├── faiss.index
│   └── chunks.pkl
└── requirements.txt

## Setup & Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Create a `.env` file in the root with your Groq API key:
> ```
> GROQ_API_KEY=your_key_here
> ```

# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions from uploaded PDF documents.

## How it works
- User uploads a PDF document
- Text is extracted using pdfplumber and split into chunks
- Chunks are embedded using Sentence Transformers (MiniLM)
- Embeddings are stored in a FAISS vector index
- User asks a question — it gets embedded and top 3 relevant chunks are retrieved
- Retrieved chunks + question are sent to Groq LLM (LLaMA3) to generate an accurate answer

## Tech Stack
Python, pdfplumber, LangChain, Sentence Transformers, FAISS, Groq API, Streamlit

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

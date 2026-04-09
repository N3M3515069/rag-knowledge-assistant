import os
import streamlit as st
from retriever import search
from indexer import build_index

os.makedirs("user_files", exist_ok=True)
os.makedirs("index", exist_ok=True)

st.title("RAG Knowledge Assistant")
st.write("Upload a PDF and ask questions about it.")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = "user_files/" + uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("PDF uploaded successfully! Building index...")
    build_index(pdf_path)
    st.success("Ready! Ask your question below.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Ask a question about your PDF:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):
        answer = search(query)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
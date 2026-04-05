import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
     
def process_pdf(path):

    texts = []
    with pdfplumber.open(path)as pdf:
        for pgs in pdf.pages:
            texts.append(pgs.extract_text())

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Max size of each chunk
    chunk_overlap=20,    # Number of overlapping characters between chunks
    length_function=len, # Function to measure length (default is len)
    separators=["\n\n", "\n", " ", ""] # Order of characters to try splitting on
)
    chunks = text_splitter.split_text(" ".join(texts))

    return(chunks)

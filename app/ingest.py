import os
import hashlib
import fitz
import markdown
from bs4 import BeautifulSoup

#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.embeddings import EmbeddingModel
from app.vector_store import collection

embedding_model = EmbeddingModel()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# PDF Reader
def read_pdf(path):

    text = ""

    doc = fitz.open(path)

    for page in doc:

        text += page.get_text()

    return text

# HTML Reader
def read_html(path):

    with open(path, "r", encoding="utf-8") as f:

        soup = BeautifulSoup(
            f.read(),
            "html.parser"
        )

    return soup.get_text()

# Markdown Reader
def read_markdown(path):

    with open(path, "r", encoding="utf-8") as f:

        html = markdown.markdown(
            f.read()
        )

    return BeautifulSoup(
        html,
        "html.parser"
    ).get_text()
    
# File Loader
def load_document(path):

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return read_pdf(path)

    elif ext in [".html", ".htm"]:
        return read_html(path)

    elif ext == ".md":
        return read_markdown(path)

    else:
        raise Exception("Unsupported file")

# Idempotent Hash
def document_hash(text):

    return hashlib.md5(
        text.encode()
    ).hexdigest()
    
def document_hash(text):

    return hashlib.md5(
        text.encode()
    ).hexdigest()
    
# Ingest Function
def ingest_document(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Document not found: {path}")

    text = load_document(path)

    doc_hash = document_hash(text)

    existing = collection.get(
        where={
            "document_hash": doc_hash
        }
    )

    if len(existing["ids"]) > 0:

        print("Already Indexed")

        return

    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):

        embedding = embedding_model.embed(chunk)

        collection.add(

            ids=[f"{doc_hash}_{i}"],

            embeddings=[embedding],

            documents=[chunk],

            metadatas=[

                {

                    "chunk": i,

                    "source": path,

                    "document_hash": doc_hash

                }

            ]
        )

    print("Document Indexed")
    
# Run Ingestion
if __name__ == "__main__":

    ingest_document(
        "./data/documents/sample.pdf"
    )
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from app.ingest import ingest_document
from app.rag import answer

app = FastAPI(
    title="Cost Efficient RAG"
)

# Allow CORS so browser/Swagger UI can fetch the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str


class Document(BaseModel):
    file_path: str


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "FastAPI app is up"
    }


@app.get("/health")

def health():

    return{

        "status":"running"
    }


@app.post("/ingest")
def ingest(doc: Document):
    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {doc.file_path}")
    ingest_document(doc.file_path)
    return {
        "message": "Document Indexed"
    }


@app.post("/query")

def query(data:Query):

    return answer(data.query)
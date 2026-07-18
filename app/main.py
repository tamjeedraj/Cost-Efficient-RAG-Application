from fastapi import FastAPI
from pydantic import BaseModel

from app.ingest import ingest_document
from app.rag import answer

app = FastAPI(
    title="Cost Efficient RAG"
)


class Query(BaseModel):

    question:str


class Document(BaseModel):

    path:str


@app.get("/health")

def health():

    return{

        "status":"running"
    }


@app.post("/ingest")

def ingest(doc:Document):

    ingest_document(doc.path)

    return{

        "message":"Document Indexed"
    }


@app.post("/query")

def query(data:Query):

    return answer(data.question)
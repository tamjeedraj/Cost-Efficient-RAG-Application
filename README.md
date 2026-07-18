# Cost-Efficient-RAG-Application
# Cost-Efficient RAG Application

A cost-efficient Retrieval-Augmented Generation (RAG) application built using a low-cost vector database. The system supports document ingestion, semantic search, grounded answers, source citations, and evaluation of retrieval and answer quality.

## Features

- PDF / HTML / Markdown document ingestion
- ChromaDB vector database
- Sentence Transformer embeddings
- Configurable chunk size and overlap
- Idempotent document ingestion
- Metadata-based filtering
- FastAPI REST API
- Top-K vector retrieval
- Grounded LLM answers
- Source chunk citations
- Evaluation metrics for retrieval and answers
- Latency and token usage logging

## Tech Stack

- Python
- FastAPI
- ChromaDB
- Sentence Transformers
- UV package manager
- LLM API

## Installation

Create environment and install dependencies:

```bash
uv sync
from app.vector_store import collection
from app.embeddings import EmbeddingModel
from app.config import TOP_K
#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os

embedding_model = EmbeddingModel()


# def get_llm():

#     if os.getenv("OPENAI_API_KEY"):

#         return ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0
#         )

#     return ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0
#     )

from langchain_google_genai import ChatGoogleGenerativeAI
import os


def get_llm():
    """Return a configured LLM or a dummy fallback when API keys are missing."""
    google_key = os.getenv("GOOGLE_API_KEY")

    if google_key:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_key,
            temperature=0
        )

    # Fallback dummy LLM to avoid raising 500s when no key is present
    class DummyLLM:
        def invoke(self, prompt):
            class Resp:
                content = "LLM not configured. Set GOOGLE_API_KEY or OPENAI_API_KEY."

            return Resp()

    return DummyLLM()

def retrieve(query, k=TOP_K):

    query_embedding = embedding_model.embed(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results

PROMPT = """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
reply exactly:

No relevant context found.

Context

{context}

Question

{question}

Answer:
"""

def answer(question):

    results = retrieve(question)

    docs = results["documents"][0]

    if len(docs) == 0:

        return {
            "answer":"No relevant context found.",
            "sources":[]
        }

    context = "\n\n".join(docs)

    llm = get_llm()

    prompt = PROMPT.format(

        context=context,

        question=question

    )

    try:
        response = llm.invoke(prompt)
        answer_text = response.content
    except Exception as e:
        # Build citations if available, then return error message instead of 500
        citations = []
        for meta in results.get("metadatas", [])[0] if results.get("metadatas") else []:
            citations.append({
                "source": meta.get("source"),
                "chunk": meta.get("chunk")
            })

        return {
            "answer": f"LLM invocation error: {e}",
            "sources": citations
        }

    citations = []
    for meta in results.get("metadatas", [])[0]:
        citations.append({
            "source": meta.get("source"),
            "chunk": meta.get("chunk")
        })

    return {
        "answer": answer_text,
        "sources": citations
    }

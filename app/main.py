"""FastAPI surface. Interactive docs at http://localhost:8000/docs once running."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.generation.answer import answer_question
from app.generation.llm import health as llm_health
from app.retrieval import store
from app.retrieval.search import search

app = FastAPI(
    title="Clause — Enterprise RAG Document Assistant",
    description="Grounded question answering over your own document corpus.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int | None = None
    use_rerank: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int | None = None
    use_rerank: bool = True


@app.get("/health")
def health():
    """Is the index built, and can we reach the model?"""
    return {"index": store.stats(), "llm": llm_health()}


@app.post("/search")
def search_endpoint(req: SearchRequest):
    """Retrieval only -- no LLM. The first thing to check when an answer is wrong."""
    hits = search(req.query, top_k=req.top_k, use_rerank=req.use_rerank)
    return {"query": req.query, "results": hits}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    """Full pipeline: retrieve, rerank, generate a cited answer."""
    return answer_question(req.question, top_k=req.top_k, use_rerank=req.use_rerank)

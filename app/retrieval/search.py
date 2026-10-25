"""Hybrid retrieval: vector + BM25, fused, then reranked.

This is the part that decides whether the whole system works. Three stages:

  1. recall    -- two cheap searches cast a wide net (CANDIDATES results each)
  2. fusion    -- Reciprocal Rank Fusion merges the two ranked lists
  3. precision -- a cross-encoder reranks the survivors down to TOP_K

Stage 3 is usually the single biggest accuracy win in the project. Phase 2's
eval harness is how you prove that rather than assume it.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.ingestion.chunk import contextualize
from app.retrieval import store
from app.retrieval.embed import embed_query, rerank

RRF_K = 60  # standard constant; dampens the influence of any single ranker


def _fuse(vector_hits: list[dict], keyword_hits: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: score by position in each list, not raw score.

    Raw scores from a cosine search and BM25 are not on the same scale, so
    averaging them is meaningless. Ranks are comparable; RRF uses those.
    """
    merged: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(vector_hits):
        row = merged.setdefault(hit["chunk_id"], dict(hit))
        row["fused_score"] = row.get("fused_score", 0.0) + 1.0 / (RRF_K + rank + 1)
        row["found_by"] = set(row.get("found_by", set())) | {"vector"}
    for rank, hit in enumerate(keyword_hits):
        row = merged.setdefault(hit["chunk_id"], dict(hit))
        row["fused_score"] = row.get("fused_score", 0.0) + 1.0 / (RRF_K + rank + 1)
        row["found_by"] = set(row.get("found_by", set())) | {"keyword"}
    return sorted(merged.values(), key=lambda r: r["fused_score"], reverse=True)


def search(
    query: str,
    top_k: int | None = None,
    candidates: int | None = None,
    use_rerank: bool = True,
    allowed_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    top_k = top_k or settings.top_k
    candidates = candidates or settings.candidates

    vector_hits = store.vector_search(embed_query(query), candidates, allowed_tags)
    keyword_hits = store.bm25_search(query, candidates)

    fused = _fuse(vector_hits, keyword_hits)[:candidates]
    if not fused:
        return []

    if use_rerank:
        # the reranker needs the document's identity too, or it scores every
        # contract's termination clause the same and can demote the right one
        passages = [
            contextualize(r["meta"].get("title", ""), r["meta"].get("section", ""), r["text"])
            for r in fused
        ]
        scores = rerank(query, passages)
        for row, s in zip(fused, scores):
            row["rerank_score"] = s
        fused.sort(key=lambda r: r["rerank_score"], reverse=True)

    results = fused[:top_k]
    for r in results:
        r["found_by"] = sorted(r.get("found_by", []))
    return results

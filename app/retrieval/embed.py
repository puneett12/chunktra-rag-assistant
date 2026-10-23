"""Local embedding and reranking models. Downloaded once, then run offline."""

from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    print(f"loading embedding model {settings.embed_model} ...")
    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder

    print(f"loading reranker {settings.rerank_model} ...")
    return CrossEncoder(settings.rerank_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    vecs = embedder().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 200
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    # bge models want this instruction prefix on the query side only
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return embedder().encode(prefixed, normalize_embeddings=True).tolist()


def rerank(query: str, passages: list[str]) -> list[float]:
    """Score each passage against the query. Higher is better."""
    if not passages:
        return []
    scores = reranker().predict([(query, p) for p in passages])
    return [float(s) for s in scores]

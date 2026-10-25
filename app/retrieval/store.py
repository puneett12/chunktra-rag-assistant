"""Persistence for the two indexes: Chroma for vectors, BM25 for keywords.

Both live on disk under data/processed/. No server, no account, no cost.
"""

from __future__ import annotations

import pickle
from typing import Any

from app.config import settings
from app.ingestion.chunk import Chunk

COLLECTION = "documents"
BM25_FILE = "bm25.pkl"


# --------------------------------------------------------------- vector store
def _client():
    import chromadb

    settings.index_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.index_path / "chroma"))


def collection():
    return _client().get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def reset_collection() -> None:
    c = _client()
    try:
        c.delete_collection(COLLECTION)
    except Exception:
        pass


def add_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    col = collection()
    col.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[{
            "doc_id": c.doc_id,
            "title": c.title,
            "page": c.page,
            "section": c.section,
            "source_path": c.source_path,
            "access_tags": ",".join(c.access_tags),
        } for c in chunks],
    )


def vector_search(vector: list[float], k: int, allowed_tags: list[str] | None = None
                  ) -> list[dict[str, Any]]:
    col = collection()
    # Phase 4: this is where the permission filter goes. Chroma supports
    # metadata `where` clauses, so filtering happens in the store -- before the
    # model ever sees a passage it should not have.
    res = col.query(query_embeddings=[vector], n_results=k)
    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        if allowed_tags is not None:
            tags = set(str(meta.get("access_tags", "")).split(","))
            if not tags & set(allowed_tags):
                continue
        out.append({
            "chunk_id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": meta,
            # chroma returns cosine distance; convert to a similarity
            "vector_score": 1.0 - float(res["distances"][0][i]),
        })
    return out


# ----------------------------------------------------------------- bm25 index
def build_bm25(chunks: list[Chunk]) -> None:
    from rank_bm25 import BM25Okapi

    corpus = [c.indexed_text().lower().split() for c in chunks]
    index = BM25Okapi(corpus)
    payload = {
        "index": index,
        "meta": [{"chunk_id": c.chunk_id, "text": c.text, "meta": {
            "doc_id": c.doc_id, "title": c.title, "page": c.page,
            "section": c.section, "source_path": c.source_path,
            "access_tags": ",".join(c.access_tags),
        }} for c in chunks],
    }
    settings.index_path.mkdir(parents=True, exist_ok=True)
    with open(settings.index_path / BM25_FILE, "wb") as fh:
        pickle.dump(payload, fh)


_bm25_cache: dict | None = None


def bm25_search(query: str, k: int) -> list[dict[str, Any]]:
    global _bm25_cache
    path = settings.index_path / BM25_FILE
    if not path.exists():
        return []
    if _bm25_cache is None:
        with open(path, "rb") as fh:
            _bm25_cache = pickle.load(fh)

    scores = _bm25_cache["index"].get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    out = []
    for i in ranked:
        if scores[i] <= 0:
            continue
        row = dict(_bm25_cache["meta"][i])
        row["bm25_score"] = float(scores[i])
        out.append(row)
    return out


def stats() -> dict[str, Any]:
    try:
        n = collection().count()
    except Exception:
        n = 0
    return {"chunks_indexed": n, "index_dir": str(settings.index_path)}

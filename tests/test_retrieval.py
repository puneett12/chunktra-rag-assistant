"""Retrieval tests that run without models or a real index.

The vector store and embedding model are replaced with small fakes, so these
check the logic — scoping, ranking, permissions — in milliseconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from app.retrieval import search as search_mod
from app.retrieval import store


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


class FakeCollection:
    """Mimics chroma's collection.get(where=..., include=[...])."""

    def __init__(self, rows):
        self.rows = rows
        self.last_where = None

    def get(self, where, include):
        self.last_where = where
        cond = where["doc_id"]
        wanted = set(cond["$in"]) if isinstance(cond, dict) else {cond}
        hits = [r for r in self.rows if r["meta"]["doc_id"] in wanted]
        return {
            "ids": [r["id"] for r in hits],
            "documents": [r["text"] for r in hits],
            "metadatas": [r["meta"] for r in hits],
            "embeddings": [r["emb"] for r in hits],
        }


ROWS = [
    {"id": "A::0", "text": "A governing law", "emb": _unit([1, 0, 0]),
     "meta": {"doc_id": "A", "access_tags": "public"}},
    {"id": "A::1", "text": "A notices", "emb": _unit([0, 1, 0]),
     "meta": {"doc_id": "A", "access_tags": "public"}},
    {"id": "A::2", "text": "A secret salary", "emb": _unit([0.9, 0.1, 0]),
     "meta": {"doc_id": "A", "access_tags": "hr"}},
    {"id": "B::0", "text": "B governing law", "emb": _unit([1, 0, 0]),
     "meta": {"doc_id": "B", "access_tags": "public"}},
]


@pytest.fixture
def fake_store(monkeypatch):
    col = FakeCollection(ROWS)
    monkeypatch.setattr(store, "collection", lambda: col)
    return col


def test_scoped_vector_search_stays_inside_the_document(fake_store):
    hits = store.vector_search_in_docs(_unit([1, 0, 0]), k=10, doc_ids=["A"])
    assert {h["meta"]["doc_id"] for h in hits} == {"A"}
    assert hits[0]["chunk_id"] == "A::0"          # most similar first


def test_scoped_vector_search_respects_k(fake_store):
    assert len(store.vector_search_in_docs(_unit([1, 0, 0]), k=1, doc_ids=["A"])) == 1


def test_scoped_vector_search_never_bypasses_permissions(fake_store):
    hits = store.vector_search_in_docs(_unit([1, 0, 0]), k=10, doc_ids=["A"],
                                       allowed_tags=["public"])
    assert "A::2" not in {h["chunk_id"] for h in hits}


def test_several_documents_use_an_in_filter(fake_store):
    hits = store.vector_search_in_docs(_unit([1, 0, 0]), k=10, doc_ids=["A", "B"])
    assert fake_store.last_where == {"doc_id": {"$in": ["A", "B"]}}
    assert {h["meta"]["doc_id"] for h in hits} == {"A", "B"}


def test_scoped_bm25_stays_inside_the_document(monkeypatch):
    from rank_bm25 import BM25Okapi

    texts = ["governing law delaware", "notices addresses", "governing law new york"]
    docs = ["A", "A", "B"]
    payload = {
        "index": BM25Okapi([t.split() for t in texts]),
        "meta": [{"chunk_id": f"{d}::{i}", "text": t, "meta": {"doc_id": d}}
                 for i, (t, d) in enumerate(zip(texts, docs))],
    }
    monkeypatch.setattr(store, "_bm25_cache", payload)
    monkeypatch.setattr(store.settings, "index_dir", str(Path(__file__).parent))  # any existing dir
    monkeypatch.setattr(store, "BM25_FILE", Path(__file__).name)                  # "exists" check passes

    unscoped = store.bm25_search("governing law", k=5)
    scoped = store.bm25_search("governing law", k=5, doc_ids=["A"])
    assert {r["meta"]["doc_id"] for r in unscoped} == {"A", "B"}
    assert {r["meta"]["doc_id"] for r in scoped} == {"A"}


def test_search_routes_doc_ids_to_both_halves(monkeypatch):
    seen = {}

    def fake_scoped(vector, k, doc_ids, allowed_tags=None):
        seen["vector"] = doc_ids
        return []

    def fake_bm25(query, k, doc_ids=None):
        seen["bm25"] = doc_ids
        return []

    monkeypatch.setattr(search_mod, "embed_query", lambda q: [1.0, 0.0, 0.0])
    monkeypatch.setattr(store, "vector_search_in_docs", fake_scoped)
    monkeypatch.setattr(store, "bm25_search", fake_bm25)
    search_mod.search("which law governs", doc_ids=["A"])
    assert seen == {"vector": ["A"], "bm25": ["A"]}

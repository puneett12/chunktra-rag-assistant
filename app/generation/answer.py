"""Grounded answering: the model may only use retrieved passages, and must cite.

The prompt does three jobs, and all three matter:
  - numbered sources, so the model has something concrete to cite
  - an explicit instruction to refuse when the context does not contain the answer
  - a citation format simple enough that it complies reliably

Refusing correctly is a feature. An assistant that says "the documents do not
cover this" is worth more than one that guesses well 80% of the time.
"""

from __future__ import annotations

from typing import Any

from app.generation.llm import LLMUnavailable, complete
from app.retrieval.search import search

SYSTEM = """You are a document analyst. You answer strictly from the numbered \
sources provided by the user, and never from your own knowledge.

Rules:
- Use ONLY the sources given. Do not add outside facts, context, or assumptions.
- Cite the source number in square brackets after every claim, like this [2].
- If the sources do not contain the answer, reply exactly: "The provided \
documents do not contain enough information to answer this."
- Quote the exact contractual or policy language when the wording matters.
- Be concise. No preamble, no restating the question."""

USER_TEMPLATE = """Sources:
{context}

Question: {question}

Answer using only the sources above, citing source numbers in brackets."""

REFUSAL = "The provided documents do not contain enough information to answer this."


def _format_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        m = h["meta"]
        header = f"[{i}] {m.get('title')} — page {m.get('page')}, {m.get('section')}"
        blocks.append(f"{header}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def _source_list(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, h in enumerate(hits, start=1):
        m = h["meta"]
        out.append({
            "n": i,
            "chunk_id": h["chunk_id"],
            "title": m.get("title"),
            "page": m.get("page"),
            "section": m.get("section"),
            "source_path": m.get("source_path"),
            "rerank_score": round(h.get("rerank_score", 0.0), 4),
            "found_by": h.get("found_by", []),
            "excerpt": h["text"][:400] + ("..." if len(h["text"]) > 400 else ""),
        })
    return out


def answer_question(
    question: str,
    top_k: int | None = None,
    use_rerank: bool = True,
    allowed_tags: list[str] | None = None,
) -> dict[str, Any]:
    hits = search(question, top_k=top_k, use_rerank=use_rerank, allowed_tags=allowed_tags)

    if not hits:
        return {
            "question": question,
            "answer": REFUSAL,
            "sources": [],
            "grounded": False,
        }

    prompt = USER_TEMPLATE.format(context=_format_context(hits), question=question)
    try:
        text = complete(SYSTEM, prompt)
    except LLMUnavailable as exc:
        return {
            "question": question,
            "answer": None,
            "error": str(exc),
            "sources": _source_list(hits),
            "grounded": False,
        }

    return {
        "question": question,
        "answer": text,
        "sources": _source_list(hits),
        "grounded": "do not contain enough information" not in text.lower(),
    }

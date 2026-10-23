"""Split a document into retrievable chunks.

Chunking is the highest-leverage thing in the whole pipeline. A naive
fixed-width split cuts clauses in half and retrieval never recovers. This
version splits on structure first (blank lines, numbered clause headings),
then packs those units up to a size budget with overlap.

Phase 3 is where you experiment here and measure the difference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.ingestion.parse import ParsedDoc

# "12.3  Termination." / "SECTION 8." / "ARTICLE IV" -- typical contract headings.
# Group 1 is the number, group 2 the short title that follows it.
HEADING = re.compile(
    r"^\s*((?:SECTION|ARTICLE)\s+[0-9IVXLC]+|[0-9]+(?:\.[0-9]+)*)\s*[.\)]?\s+([^.\n]{0,40})",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    page: int
    section: str
    source_path: str
    # Phase 4 will populate this with real roles; every chunk carries it from day one
    # so that adding permissions later is a filter change, not a re-index.
    access_tags: list[str]


def _split_units(text: str) -> list[str]:
    """Break text into the smallest sensible units: paragraphs and clauses."""
    units, buf = [], []
    for line in text.split("\n"):
        if HEADING.match(line) and buf:
            units.append("\n".join(buf).strip())
            buf = [line]
        elif not line.strip():
            if buf:
                units.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        units.append("\n".join(buf).strip())
    return [u for u in units if u]


def _looks_like_heading(text: str) -> bool:
    """True for "Service Levels" or "TERM AND TERMINATION"; false for a sentence.

    Numbered clauses in contracts come in two shapes: a real heading
    ("12.3 Renewal and Notice.") and a bare numbered sentence ("2.3 Halcyon may
    increase fees..."). Only the first should become a section label — otherwise
    the citation shows a sentence fragment instead of a section name.
    """
    words = text.split()
    if not words or len(words) > 5:
        return False
    if text.isupper():
        return True
    # Title Case: every word of 4+ letters starts capitalised
    return all(w[0].isupper() for w in words if len(w) > 3 and w[0].isalpha())


def _current_section(unit: str, fallback: str) -> str:
    """Return a short section label like "12.3 Renewal", or keep the previous one."""
    first = unit.split("\n", 1)[0].strip()
    m = HEADING.match(first)
    if not m:
        return fallback
    number = m.group(1).strip()
    # take only up to the first sentence break, then decide if it reads as a heading
    title = m.group(2).strip().rstrip(".").split(".")[0].strip()
    return f"{number} {title}" if _looks_like_heading(title) else number


def chunk_document(
    doc: ParsedDoc,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    units = _split_units(doc.text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    section = "(preamble)"      # running section as we walk the document
    buf_section = section       # section the CURRENT chunk started in
    offset = 0                  # char offset of the start of `buf` within doc.text
    search_from = 0

    def flush(start_offset: int, sect: str) -> None:
        if not buf:
            return
        body = "\n\n".join(buf).strip()
        if not body:
            return
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}::{len(chunks):04d}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=body,
            page=doc.page_for_offset(start_offset),
            section=sect,
            source_path=doc.source_path,
            access_tags=["public"],
        ))

    for unit in units:
        # track where this unit sits in the original text, for page numbers
        found = doc.text.find(unit[:60], search_from)
        if found != -1:
            search_from = found
        section = _current_section(unit, section)
        if not buf:
            offset = search_from
            buf_section = section

        # a single oversized unit becomes its own chunk, hard-split if needed
        if len(unit) > size * 1.6:
            flush(offset, buf_section)
            buf, buf_len = [], 0
            for i in range(0, len(unit), size - overlap):
                piece = unit[i:i + size]
                if piece.strip():
                    buf = [piece]
                    flush(search_from + i, section)
            buf, buf_len = [], 0
            offset = search_from + len(unit)
            buf_section = section
            continue

        if buf_len + len(unit) > size and buf:
            flush(offset, buf_section)
            # carry the tail of the previous chunk forward as overlap
            tail = "\n\n".join(buf)[-overlap:] if overlap else ""
            buf = [tail, unit] if tail else [unit]
            buf_len = len(tail) + len(unit)
            offset = search_from
            buf_section = section
        else:
            buf.append(unit)
            buf_len += len(unit)

    flush(offset, buf_section)
    return chunks

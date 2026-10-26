import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunk import chunk_document
from app.ingestion.parse import ParsedDoc, _clean

SAMPLE = """MASTER SERVICES AGREEMENT

1. Definitions. "Services" means the services described in each Statement of Work.

2. Term. This Agreement begins on the Effective Date and continues for three (3) years.

2.1 Renewal. This Agreement shall automatically renew for successive one-year terms
unless either party gives ninety (90) days written notice prior to the end of the
then-current term.

3. Governing Law. This Agreement is governed by the laws of the State of Delaware.
"""


def _doc(text=SAMPLE):
    return ParsedDoc("sample_msa", "Sample MSA", _clean(text), "data/raw/sample.txt")


def test_produces_chunks():
    chunks = chunk_document(_doc())
    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)


def test_chunk_ids_are_unique():
    chunks = chunk_document(_doc())
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_respects_size_budget():
    chunks = chunk_document(_doc(), size=200, overlap=40)
    # allow modest overshoot: a single indivisible unit may exceed the budget
    assert all(len(c.text) <= 200 * 1.8 for c in chunks)


def test_captures_section_headings():
    chunks = chunk_document(_doc(), size=150, overlap=20)
    sections = " ".join(c.section for c in chunks)
    assert any(ch.isdigit() for ch in sections)


def test_renewal_clause_survives_chunking():
    """The clause we most want to retrieve must stay intact in some chunk."""
    chunks = chunk_document(_doc(), size=400, overlap=80)
    joined = [c.text for c in chunks]
    assert any("ninety (90) days written notice" in t for t in joined)


def test_section_label_is_a_heading_not_a_sentence():
    """A numbered sentence must not become a section label."""
    chunks = chunk_document(_doc(), size=150, overlap=20)
    for c in chunks:
        assert len(c.section.split()) <= 6


def test_empty_document_is_safe():
    assert chunk_document(_doc("   \n\n  ")) == []


def test_indexed_text_carries_document_identity():
    """Every chunk must know which document it came from once indexed."""
    from app.ingestion.chunk import contextualize
    chunks = chunk_document(_doc())
    renewal = [c for c in chunks if "ninety (90) days written notice" in c.text][0]
    indexed = renewal.indexed_text()
    assert "Sample MSA" in indexed          # the document title travels with it
    assert renewal.text in indexed          # the original clause is intact
    assert "Sample MSA" not in renewal.text  # but the displayed text stays clean


def test_contextual_headers_can_be_disabled(monkeypatch):
    from app.config import settings
    from app.ingestion.chunk import contextualize
    monkeypatch.setattr(settings, "contextual_headers", False)
    assert contextualize("Title", "12.3", "body") == "body"


def test_overlap_never_starts_mid_word():
    """Overlap carried into a chunk must begin on a whole word.

    Slicing the overlap at a fixed character offset produced chunks starting
    "s Agreement shall be governed..." — ugly in a citation, and it caused the
    evaluator to miss answers the system had actually retrieved. Sweeps many
    overlap sizes so a cut can't land between words by luck.
    """
    import re
    clauses = [
        f"{n}. The supplier shall deliver {'consignment' if n % 2 else 'goods'} within "
        f"{n * 7} business days after receiving written notice from the purchaser."
        for n in range(1, 25)
    ]
    doc = ParsedDoc("t", "T", _clean("\n\n".join(clauses)), "t.txt")
    words = set(re.findall(r"\S+", doc.text))
    for overlap in range(31, 120, 7):
        for c in chunk_document(doc, size=260, overlap=overlap)[1:]:
            first = c.text.split()[0]
            assert first in words, f"overlap={overlap}: chunk starts mid-word: {c.text[:40]!r}"

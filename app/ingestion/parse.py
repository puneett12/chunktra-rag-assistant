"""Turn a file on disk into clean text plus per-page offsets.

Page offsets matter: they are how a citation can say "page 7" later on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    text: str
    source_path: str
    # page_breaks[i] = character offset in `text` where page i+1 begins
    page_breaks: list[int] = field(default_factory=list)

    def page_for_offset(self, offset: int) -> int:
        """Which 1-indexed page does this character offset fall on?"""
        page = 1
        for i, brk in enumerate(self.page_breaks):
            if offset >= brk:
                page = i + 1
            else:
                break
        return page


def _clean(text: str) -> str:
    # collapse the ragged whitespace PDF extraction leaves behind, but keep
    # paragraph breaks, because the chunker splits on them
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: Path) -> ParsedDoc:
    import fitz  # pymupdf

    doc = fitz.open(path)
    parts, breaks, cursor = [], [], 0
    for page in doc:
        breaks.append(cursor)
        page_text = page.get_text("text")
        parts.append(page_text)
        cursor += len(page_text) + 2  # +2 for the "\n\n" join below
    doc.close()
    return ParsedDoc(
        doc_id=path.stem,
        title=path.stem.replace("_", " ").replace("-", " "),
        text=_clean("\n\n".join(parts)),
        source_path=str(path),
        page_breaks=breaks,
    )


def parse_html(path: Path) -> ParsedDoc:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(errors="ignore"), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return ParsedDoc(
        doc_id=path.stem,
        title=(soup.title.string.strip() if soup.title and soup.title.string
               else path.stem.replace("-", " ")),
        text=_clean(soup.get_text("\n")),
        source_path=str(path),
    )


def parse_docx(path: Path) -> ParsedDoc:
    import docx

    d = docx.Document(str(path))
    body = "\n\n".join(p.text for p in d.paragraphs)
    return ParsedDoc(path.stem, path.stem.replace("-", " "), _clean(body), str(path))


def parse_text(path: Path) -> ParsedDoc:
    return ParsedDoc(
        path.stem, path.stem.replace("-", " "),
        _clean(path.read_text(errors="ignore")), str(path),
    )


PARSERS = {
    ".pdf": parse_pdf,
    ".html": parse_html,
    ".htm": parse_html,
    ".docx": parse_docx,
    ".txt": parse_text,
    ".md": parse_text,
}


def parse(path: Path) -> ParsedDoc | None:
    fn = PARSERS.get(path.suffix.lower())
    if fn is None:
        return None
    try:
        parsed = fn(path)
    except Exception as exc:  # a corrupt file should not kill the whole run
        print(f"  ! failed to parse {path.name}: {exc}")
        return None
    return parsed if parsed.text.strip() else None

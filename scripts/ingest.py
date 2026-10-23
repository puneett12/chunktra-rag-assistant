#!/usr/bin/env python3
"""Build the indexes from everything in data/raw/.

Run:  make ingest      (or: .venv/bin/python scripts/ingest.py)
Safe to re-run -- it rebuilds both indexes from scratch.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                      # noqa: E402
from app.ingestion.chunk import chunk_document       # noqa: E402
from app.ingestion.parse import parse                # noqa: E402
from app.retrieval import store                      # noqa: E402
from app.retrieval.embed import embed_passages       # noqa: E402

SUPPORTED = {".pdf", ".html", ".htm", ".docx", ".txt", ".md"}


def main() -> int:
    raw = settings.raw_path
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in SUPPORTED)

    if not files:
        print(f"No documents found in {raw}/")
        print("Try:  cp data/samples/*.txt data/raw/")
        return 1

    print(f"Found {len(files)} document(s) in {raw}/\n")
    t0 = time.time()

    all_chunks = []
    for i, path in enumerate(files, 1):
        doc = parse(path)
        if doc is None:
            continue
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  [{i}/{len(files)}] {path.name}: {len(chunks)} chunks")

    if not all_chunks:
        print("\nNothing to index -- every file failed to parse.")
        return 1

    print(f"\n{len(all_chunks)} chunks total. Embedding...")
    store.reset_collection()

    BATCH = 256
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        store.add_chunks(batch, embed_passages([c.text for c in batch]))
        print(f"  embedded {min(i + BATCH, len(all_chunks))}/{len(all_chunks)}")

    print("Building keyword (BM25) index...")
    store.build_bm25(all_chunks)

    print(f"\nDone in {time.time() - t0:.1f}s. {store.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

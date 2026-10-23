# Clause — Enterprise RAG Document Assistant

Ask a question about your own documents. Get an answer with the exact clause
quoted and a pointer to the page it came from — or an honest "the documents
don't cover this."

Runs entirely on a laptop. No API keys, no per-query cost.

**Status:** in development. Phase 1 of 8 — see the roadmap below.

---

## Roadmap

- [ ] **Phase 1** — Baseline RAG: parse, chunk, embed, retrieve, cite
- [ ] **Phase 2** — Evaluation harness and baseline accuracy numbers
- [ ] **Phase 3** — Chunking experiments, OCR, retrieval tuning
- [ ] **Phase 4** — Auth, roles, permission-filtered retrieval, audit log
- [ ] **Phase 5** — Autonomous agent with approval gates
- [ ] **Phase 6** — Error handling, caching, CI
- [ ] **Phase 7** — Deployment, demo, write-up

## Stack

Python 3.11+ · FastAPI · ChromaDB · sentence-transformers · rank-bm25 ·
Ollama · Streamlit. Everything free and running locally.

## Licence

MIT — see [LICENSE](LICENSE).

#!/usr/bin/env python3
"""Measure retrieval quality. Run this before and after every change.

Metrics:
  Recall@k  -- did the correct passage appear anywhere in the top k?
  MRR       -- how high up was it? 1.0 means always first.

Usage:
  make eval                                   # the hand-written question set
  python eval/run_eval.py --compare           # with and without the reranker
  python eval/run_eval.py --questions eval/questions_cuad.yaml --quiet
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app.config import settings  # noqa: E402
from app.retrieval.search import search  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.yaml"


_FOLD = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',   # curly quotes
    "–": "-", "—": "-", " ": " ",                  # dashes, nbsp
})


def _norm(text: str) -> str:
    """Forgiving about formatting, strict about content.

    Folds curly quotes and long dashes to plain ASCII and collapses whitespace,
    so a phrase still matches across a line break or a different PDF extractor.
    """
    return re.sub(r"\s+", " ", text.translate(_FOLD)).lower().strip()


def hit_position(results: list[dict], expect_doc: str | None,
                 expect_text: str | None) -> int | None:
    """1-indexed rank of the first correct result, or None if absent."""
    needle = _norm(expect_text) if expect_text else None
    for i, r in enumerate(results, start=1):
        doc_ok = (not expect_doc) or r["meta"].get("doc_id") == expect_doc
        text_ok = (not needle) or needle in _norm(r["text"])
        if doc_ok and text_ok:
            return i
    return None


@dataclass
class RunResult:
    recall: float
    mrr: float
    positions: list          # rank of the correct passage per question, or None
    doc_seen: list           # was the right DOCUMENT anywhere in the top k?
    returned: list           # what actually came back, kept for --show-misses
    latencies_ms: list = field(default_factory=list)

    def latency(self, pct: float) -> float:
        xs = sorted(self.latencies_ms)
        if not xs:
            return 0.0
        return xs[min(len(xs) - 1, int(round(pct / 100 * (len(xs) - 1))))]

    @property
    def mean_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0


def run(items: list[dict], k: int, use_rerank: bool, candidates: int | None = None,
        verbose: bool = True) -> RunResult:
    # The first search loads both models, which takes seconds. Pay that cost
    # before the clock starts or it swamps every latency figure.
    search("warm up", top_k=1, use_rerank=use_rerank, candidates=candidates)

    positions: list[int | None] = []
    doc_seen: list[bool] = []
    returned: list[list] = []
    latencies: list[float] = []
    for q in items:
        t0 = time.perf_counter()
        results = search(q["question"], top_k=k, use_rerank=use_rerank,
                         candidates=candidates)
        latencies.append((time.perf_counter() - t0) * 1000)
        returned.append(results)
        pos = hit_position(results, q.get("expect_doc"), q.get("expect_text"))
        positions.append(pos)
        doc_seen.append(any(r["meta"].get("doc_id") == q.get("expect_doc") for r in results))
        if verbose:
            mark = f"rank {pos}" if pos else "MISS"
            print(f"  [{mark:>7}] {q['question'][:66]}")
        elif len(positions) % 25 == 0 or len(positions) == len(items):
            # quiet mode still needs a heartbeat, or a long run looks frozen
            print(f"  ... {len(positions)}/{len(items)} questions", flush=True)

    found = [p for p in positions if p]
    recall = len(found) / len(positions) if positions else 0.0
    mrr = sum(1 / p for p in found) / len(positions) if positions else 0.0
    return RunResult(recall, mrr, positions, doc_seen, returned, latencies)


def _snip(text: str, n: int = 110) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("..." if len(flat) > n else "")


def show_misses(items: list[dict], positions: list, returned: list, limit: int) -> None:
    """For each miss: where the answer actually lives, and what came back instead.

    Aggregates say THAT retrieval failed; examples say WHY. Re-chunks only the
    documents being shown, so it costs seconds.
    """
    from app.config import settings
    from app.ingestion.chunk import chunk_document
    from app.ingestion.parse import parse

    files = {p.stem: p for p in settings.raw_path.iterdir() if p.is_file()}
    shown = 0
    for q, pos, results in zip(items, positions, returned):
        if pos or shown >= limit:
            continue
        shown += 1
        print(f"\n  MISS {shown}  [{q.get('category', '')}]  {q['question']}")
        print(f"    looking for: \"{q['expect_text']}\"")

        path = files.get(q.get("expect_doc", ""))
        doc = parse(path) if path else None
        chunks = chunk_document(doc) if doc else []
        target = next((c for c in chunks if _norm(q["expect_text"]) in _norm(c.text)), None)
        if target is None:
            print("    answer lives in: NO SINGLE CHUNK (split across a boundary)")
        else:
            print(f"    answer lives in: section {target.section!r}, {len(target.text)} chars, "
                  f"1 of {len(chunks)} chunks in this contract")
            print(f"      \"{_snip(target.text)}\"")

        print("    retrieved instead:")
        for r in results[:3]:
            m = r["meta"]
            where = "same doc " if m.get("doc_id") == q.get("expect_doc") else "OTHER DOC"
            print(f"      [{where}] section {str(m.get('section'))!r}: \"{_snip(r['text'], 80)}\"")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="run hybrid with and without rerank side by side")
    ap.add_argument("--questions", default=str(QUESTIONS),
                    help="question set to evaluate (default: eval/questions.yaml)")
    ap.add_argument("--quiet", action="store_true",
                    help="skip the per-question lines; print only the summary")
    ap.add_argument("--category", help='evaluate one category only, e.g. "Governing Law"')
    ap.add_argument("--show-misses", type=int, default=0, metavar="N",
                    help="print the first N misses: where the answer lives, what came back")
    args = ap.parse_args()

    qpath = Path(args.questions)
    print(f"Question set: {qpath.name}")
    spec = yaml.safe_load(qpath.read_text())
    items = [q for q in spec.get("questions", []) if not q.get("unanswerable")]
    refusals = [q for q in spec.get("questions", []) if q.get("unanswerable")]
    if args.category:
        items = [q for q in items if q.get("category", "").lower() == args.category.lower()]
        refusals = []
        print(f"Category filter: {args.category}")

    if not items:
        print("No answerable questions in eval/questions.yaml")
        return 1

    if args.compare:
        print(f"Comparing configurations over {len(items)} questions (k={args.k})\n")
        rows = []
        for label, rerank in [("hybrid (no rerank)", False), ("hybrid + rerank", True)]:
            r = run(items, args.k, rerank, verbose=False)
            rows.append((label, r.recall, r.mrr, r.mean_ms, r.latency(95)))
        print(f"  {'configuration':<22} {'Recall@' + str(args.k):>10} {'MRR':>8} "
              f"{'mean ms':>9} {'p95 ms':>8}")
        print(f"  {'-' * 61}")
        for label, recall, mrr, mean_ms, p95 in rows:
            print(f"  {label:<22} {recall:>9.1%} {mrr:>8.3f} {mean_ms:>9.0f} {p95:>8.0f}")
        print()
        print("Paste these into the results table in your README.")
        return 0

    print(f"Evaluating {len(items)} questions "
          f"(k={args.k}, rerank={'off' if args.no_rerank else 'on'}, "
          f"candidates={settings.candidates})\n")
    res = run(items, args.k, not args.no_rerank, verbose=not args.quiet)
    recall, mrr, positions, doc_seen, returned = (
        res.recall, res.mrr, res.positions, res.doc_seen, res.returned)

    print(f"\n{'-' * 56}")
    print(f"  Recall@{args.k}: {recall:.1%}   ({len([p for p in positions if p])}/{len(positions)})")
    print(f"  MRR       : {mrr:.3f}")
    print(f"  Latency   : mean {res.mean_ms:.0f} ms   p50 {res.latency(50):.0f} ms   "
          f"p95 {res.latency(95):.0f} ms   (retrieval + rerank, no LLM)")

    # Why did the misses miss? The two causes need opposite fixes:
    #   wrong document  -> the retriever can't tell contracts apart (titles, identity)
    #   wrong passage   -> it found the contract but not the clause (chunking, ranking)
    misses = [i for i, p in enumerate(positions) if not p]
    if misses:
        wrong_doc = sum(1 for i in misses if not doc_seen[i])
        wrong_passage = len(misses) - wrong_doc
        print(f"\n  Of {len(misses)} misses:")
        print(f"   never retrieved the right document   {wrong_doc:>4}  ({wrong_doc / len(misses):.0%})")
        print(f"   right document, wrong passage        {wrong_passage:>4}  ({wrong_passage / len(misses):.0%})")

    categories = sorted({q["category"] for q in items if q.get("category")})
    if categories:
        print(f"\n  {'By category:':<40} {'hits':>9} {'recall':>7}   {'wrong doc':>9} {'wrong passage':>13}")
        for cat in categories:
            idx = [i for i, q in enumerate(items) if q.get("category") == cat]
            hits = sum(1 for i in idx if positions[i])
            missed = [i for i in idx if not positions[i]]
            wd = sum(1 for i in missed if not doc_seen[i])
            print(f"   {cat:<38} {hits:>4}/{len(idx):<4} {hits / len(idx):>6.1%}"
                  f"   {wd:>9} {len(missed) - wd:>13}")
    if refusals:
        print(f"\n  {len(refusals)} unanswerable questions are listed but not scored here.")
        print("  Check those by hand — the system should decline, not invent.")
    print(f"{'-' * 56}")

    if args.show_misses:
        show_misses(items, positions, returned, args.show_misses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure retrieval quality. Run this before and after every change.

Metrics:
  Recall@k  -- did the correct passage appear anywhere in the top k?
  MRR       -- how high up was it? 1.0 means always first.

Usage:
  make eval                        # current configuration
  python eval/run_eval.py --compare   # all three configurations side by side
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app.retrieval.search import search  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.yaml"


def _norm(text: str) -> str:
    """Collapse all whitespace so a phrase still matches across a line break."""
    return re.sub(r"\s+", " ", text).lower()


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


def run(items: list[dict], k: int, use_rerank: bool, candidates: int | None = None,
        verbose: bool = True) -> tuple[float, float, list]:
    positions: list[int | None] = []
    for q in items:
        results = search(q["question"], top_k=k, use_rerank=use_rerank,
                         candidates=candidates)
        pos = hit_position(results, q.get("expect_doc"), q.get("expect_text"))
        positions.append(pos)
        if verbose:
            mark = f"rank {pos}" if pos else "MISS"
            print(f"  [{mark:>7}] {q['question'][:66]}")

    found = [p for p in positions if p]
    recall = len(found) / len(positions) if positions else 0.0
    mrr = sum(1 / p for p in found) / len(positions) if positions else 0.0
    return recall, mrr, positions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="run vector-only, hybrid, and hybrid+rerank side by side")
    args = ap.parse_args()

    spec = yaml.safe_load(QUESTIONS.read_text())
    items = [q for q in spec.get("questions", []) if not q.get("unanswerable")]
    refusals = [q for q in spec.get("questions", []) if q.get("unanswerable")]

    if not items:
        print("No answerable questions in eval/questions.yaml")
        return 1

    if args.compare:
        print(f"Comparing configurations over {len(items)} questions (k={args.k})\n")
        rows = []
        for label, rerank in [("hybrid (no rerank)", False), ("hybrid + rerank", True)]:
            recall, mrr, _ = run(items, args.k, rerank, verbose=False)
            rows.append((label, recall, mrr))
        print(f"  {'configuration':<22} {'Recall@' + str(args.k):>10} {'MRR':>8}")
        print(f"  {'-' * 42}")
        for label, recall, mrr in rows:
            print(f"  {label:<22} {recall:>9.1%} {mrr:>8.3f}")
        print()
        print("Paste these into the results table in your README.")
        return 0

    print(f"Evaluating {len(items)} questions "
          f"(k={args.k}, rerank={'off' if args.no_rerank else 'on'})\n")
    recall, mrr, positions = run(items, args.k, not args.no_rerank)

    print(f"\n{'-' * 56}")
    print(f"  Recall@{args.k}: {recall:.1%}   ({len([p for p in positions if p])}/{len(positions)})")
    print(f"  MRR       : {mrr:.3f}")
    if refusals:
        print(f"\n  {len(refusals)} unanswerable questions are listed but not scored here.")
        print("  Check those by hand — the system should decline, not invent.")
    print(f"{'-' * 56}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

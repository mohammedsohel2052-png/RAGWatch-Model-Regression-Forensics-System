"""
demo.py — Regression Detection Demo

Runs a self-contained demonstration that shows the full regression detection
cycle without needing any API keys. Uses the stub generator backend.

What it does:
  1. Establishes a healthy baseline (good retrieval, top_k=2)
  2. Simulates a regression (breaks retrieval, top_k=0)
  3. Compares the new run against the baseline → regression alerts fire

Run it:
  python demo.py
"""

import json
import os
import sys

# Force stub backend BEFORE importing generator (BACKEND is read at import time)
os.environ["GENERATOR_BACKEND"] = "stub"

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ragwatch"))

from ragwatch import Evaluator
from rag_core import load_chunks, embed_chunks, retrieve, embed_fn
from generator import generate

DEMO_DB = "demo_regression.db"
GOLDEN_DATASET_PATH = "golden_dataset.json"
DOC_PATH = "doc.md"


def build_pipeline(top_k: int):
    """Returns (retrieval_fn, generation_fn) for a given top_k setting."""
    chunks = load_chunks(DOC_PATH)
    chunks, chunk_embeddings = embed_chunks(chunks)

    def retrieval_fn(query: str) -> list[dict]:
        return retrieve(query, chunks, chunk_embeddings, top_k=top_k)

    def generation_fn(query: str, docs: list[dict]) -> dict:
        return generate(query, docs)

    return retrieval_fn, generation_fn


def print_separator(title: str = ""):
    line = "-" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}")
    else:
        print(line)


def main():
    print_separator("RAGWatch - Regression Detection Demo")
    print("  Backend: stub (no API key or Ollama required)")
    print("  Database:", DEMO_DB)

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    # ── Phase 1: Healthy Baseline ─────────────────────────────────────────────
    print_separator("Phase 1: Establishing Healthy Baseline  (top_k=2)")

    evaluator = Evaluator(db_path=DEMO_DB, embed_fn=embed_fn)
    retrieval_fn, generation_fn = build_pipeline(top_k=2)
    baseline_summary = evaluator.evaluate(golden_dataset, retrieval_fn, generation_fn)

    evaluator.promote_to_baseline(baseline_summary["run_id"])

    m = baseline_summary["metrics"]
    print(f"\n  Baseline Metrics:")
    print(f"    Context Precision : {m['avg_context_precision']:.3f}")
    print(f"    Context Recall    : {m['avg_context_recall']:.3f}")
    print(f"    Answer Relevancy  : {m['avg_answer_relevancy']:.3f}")
    print(f"    Faithfulness      : {m['faithfulness_score']:.3f}")

    # ── Phase 2: Simulate Regression ──────────────────────────────────────────
    print_separator("Phase 2: Simulating Regression  (top_k=0 - retrieval broken)")
    print("  Imagine a developer changed a config and retrieval is now disabled.")

    retrieval_fn_broken, generation_fn = build_pipeline(top_k=0)
    broken_summary = evaluator.evaluate(golden_dataset, retrieval_fn_broken, generation_fn)

    m2 = broken_summary["metrics"]
    print(f"\n  Broken Run Metrics:")
    print(f"    Context Precision : {m2['avg_context_precision']:.3f}  (was {m['avg_context_precision']:.3f})")
    print(f"    Context Recall    : {m2['avg_context_recall']:.3f}  (was {m['avg_context_recall']:.3f})")
    print(f"    Answer Relevancy  : {m2['avg_answer_relevancy']:.3f}  (was {m['avg_answer_relevancy']:.3f})")
    print(f"    Faithfulness      : {m2['faithfulness_score']:.3f}  (was {m['faithfulness_score']:.3f})")

    # ── Phase 3: Regression Detection ─────────────────────────────────────────
    print_separator("Phase 3: Regression Detection")

    alerts = evaluator.check_regressions(broken_summary)

    print_separator()
    if alerts:
        print(f"  [OK] Regression detection WORKED - {len(alerts)} alert(s) fired.")
        print("  In CI, this would:")
        print("    - Post a Slack message to your #alerts channel")
        print("    - Exit with code 1 (blocking the PR merge)")
        print("    - Upload the .db artifact to GitHub Actions for inspection")
    else:
        print("  [WARN] No regressions detected (thresholds may need adjustment).")

    print_separator()
    print(f"  Demo complete. Results saved to '{DEMO_DB}'.")
    print(f"  Run 'python -m ragwatch.ui' to view the dashboard at http://localhost:5050")
    print_separator()


if __name__ == "__main__":
    main()

# ragwatch

**ragwatch** is a lightweight, zero-friction Python SDK for evaluating, monitoring, and detecting regressions in RAG (Retrieval-Augmented Generation) pipelines.

Think of it as a mini [LangSmith](https://smith.langchain.com/) or [Braintrust](https://www.braintrustdata.com/) — built from scratch.

---

## Features

| Feature | Description |
|---|---|
| `@monitor` decorator | Drop on any function for instant latency & trace logging |
| `Evaluator` class | Run evaluations against a golden dataset in one call |
| Regression Detection | Auto-alerts when metrics drop below baseline thresholds |
| SQLite Storage | Human-readable, git-friendly local storage |
| Baseline Promotion | Promote any run to the reference baseline |

---

## Installation

```bash
pip install ragwatch
```

Or install from source (editable mode):

```bash
git clone https://github.com/mohammedsohel2052/ragwatch
pip install -e ./ragwatch
```

---

## Quick Start

### 1. Zero-Friction Monitoring with `@monitor`

```python
from ragwatch import monitor

@monitor(project_name="my-rag-bot")
def generate_answer(query, docs):
    # Your existing RAG generation code — unchanged
    return {"answer": "...", "latency_ms": 120}
```

Every call now logs a trace to `ragwatch.db` and prints:
```
[RAGWatch] ✓ Traced 'generate_answer' | 452ms | project='my-rag-bot'
```

### 2. Full Evaluation Pipeline

```python
from ragwatch import Evaluator

evaluator = Evaluator(db_path="eval_results.db", embed_fn=my_embed_fn)

summary = evaluator.evaluate(
    golden_dataset=dataset,           # list of {"question": ..., "expected_answer": ..., ...}
    retrieval_fn=my_retrieval_fn,     # fn(query) -> list[dict]
    generation_fn=my_generation_fn,   # fn(query, docs) -> {"answer": str, "latency_ms": float}
)
```

### 3. Regression Detection

```python
# Check current run against the stored baseline
alerts = evaluator.check_regressions(summary)

# If happy with results, promote as the new baseline
evaluator.promote_to_baseline(summary["run_id"])
```

### 4. RAGWatch UI Dashboard (Mini-LangSmith)

Visualize your evaluations and live telemetry instantly. From your project root, run:

```bash
python -m ragwatch.ui
```

Then open `http://localhost:5050` in your browser.

- **Evaluations Tab**: Track Precision, Recall, Relevancy, and Faithfulness over time.
- **Live Traces Tab**: See exactly what your `@monitor` decorator is capturing in real-time.

---

## Metrics Explained

| Metric | What it measures |
|---|---|
| **Context Precision** | Of retrieved chunks, what % were actually relevant? |
| **Context Recall** | Of all needed chunks, what % did we retrieve? |
| **Answer Relevancy** | Semantic similarity between generated & expected answer |
| **Faithfulness** | Did the model hallucinate when it shouldn't have? |

---

## Project Structure

```
ragwatch/
├── ragwatch/
│   ├── __init__.py       # Public API
│   ├── evaluator.py      # Main Evaluator class
│   ├── scorer.py         # Metric calculation functions
│   ├── storage.py        # SQLite persistence (runs + traces)
│   ├── monitor.py        # @monitor decorator
│   └── regression.py     # Regression alert logic
└── pyproject.toml
```

---

## Built By

Mohammed Sohel Patwari — [GitHub](https://github.com/mohammedsohel2052-png)

Inspired by [RAGAS](https://docs.ragas.io/), [LangSmith](https://smith.langchain.com/), and [Braintrust](https://www.braintrustdata.com/).

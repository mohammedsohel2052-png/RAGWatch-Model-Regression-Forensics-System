<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

<br>

<div align="center">
  <h1>RAGWatch: Model Regression & Forensics System</h1>
  <p><strong>A CI/CD-integrated evaluation, monitoring, and root-cause forensics pipeline for RAG systems.</strong></p>
</div>

---

## 🛑 The Problem: AI Regressions are Silent

LLMs are non-deterministic, and embedding models change. If a developer swaps an embedding model or tweaks a prompt to save costs, the system might suddenly start hallucinating or failing to retrieve context. 

Unlike traditional software where a bug causes a loud crash, **AI regressions cause worse answers** — which are completely invisible unless you measure them.

## 🚀 The Solution: RAGWatch

RAGWatch is a lightweight system that runs in your CI/CD pipeline and production environment to catch regressions before they reach your users. 

RAGWatch includes a full **LLM-as-a-Judge Forensics Engine** to automatically diagnose the root cause of regressions and a premium, Apple-inspired monochrome dashboard.

---

## 🌟 Key Features

### 1. Automated Regression Detection
Automatically scores an LLM pipeline against a fixed `golden_dataset.json` on every commit. Tracks historical metrics and alerts on regressions (e.g., dropping from 95% precision to 60%).

### 2. Four-Pillar Metrics Evaluation
We measure exactly where the pipeline is failing using RAGAS-style metrics:
- **Context Precision**: Of retrieved chunks, what % were actually relevant?
- **Context Recall**: Of all needed chunks, what % did the retriever find?
- **Answer Relevancy**: Semantic similarity between generated and expected answer.
- **Faithfulness**: Did the model hallucinate? Does it correctly say "I don't know"?

### 3. Root-Cause Forensics & Taxonomy
When a regression occurs, the **Forensics Engine** uses an LLM-as-a-judge to diagnose the failure into a 5-tier taxonomy:
1. `RETRIEVAL_MISS`: The retriever failed to find the right document.
2. `RETRIEVAL_NOISE`: The retriever found too much irrelevant junk, confusing the LLM.
3. `GENERATION_HALLUCINATION`: The LLM made up facts not in the context.
4. `GENERATION_MISUSE`: The context had the answer, but the LLM ignored it.
5. `AMBIGUOUS_GOLDEN`: The test question itself is flawed.

### 4. Human-in-the-Loop Feedback
Correct hallucinations directly in the UI dashboard and automatically append them to `golden_dataset.json` to prevent the regression from happening again.

### 5. Premium UI Dashboard
A fully local, zero-config Flask dashboard featuring a sleek, Apple-inspired black-and-white glassmorphism design.

---

## 📦 Installation

Install the SDK directly from the repository or via PyPI (coming soon):

```bash
# Clone the repository
git clone https://github.com/mohammedsohel2052-png/Model-regression-detection.git
cd Model-regression-detection

# Install the RAGWatch package
pip install -e ./ragwatch

# Install demo dependencies
pip install -r requirements.txt
```

---

## 💻 Quick Start

### 1. Run the Evaluation Pipeline
Run this in your CI/CD pipeline or locally to test your current code against the golden dataset.

```bash
# First run: establish a healthy baseline
python eval_runner.py --set-baseline

# Subsequent runs: evaluate and compare against baseline
python eval_runner.py
```

### 2. Launch the Premium Dashboard
View historical runs, live production traces, and run root-cause forensics.

```bash
python -m ragwatch.ui
# Opens http://localhost:5050
```

### 3. Run Root-Cause Forensics via CLI
```bash
python eval_runner.py --diagnose <trace_id>
```

---

## 🔌 Using the `@monitor` Decorator in Production

Instrument your production code with a single line of code. Every call automatically logs latency, traces, and output previews to the local `ragwatch.db`.

```python
from ragwatch import monitor

@monitor(project_name="insurance-rag")
def generate_answer(query, docs):
    # Your existing RAG function — unchanged
    return {"answer": "..."}
```

---

## 🏗️ Tech Stack

| Technology | Role |
|---|---|
| `pydantic` | Structured JSON extraction for the Forensics LLM-as-a-judge |
| `sentence-transformers` | Local text embeddings for answer relevancy |
| `numpy` | Cosine similarity for retrieval evaluation |
| `sqlite3` | Lightweight, zero-config run history storage |
| `openai` SDK | Backend-agnostic client (Works with OpenAI or Ollama) |
| `flask` | Premium UI dashboard |

---

Built by **Mohammed Sohel Patwari**  
Inspired by industry leaders like [RAGAS](https://docs.ragas.io/), [LangSmith](https://smith.langchain.com/), and [Braintrust](https://www.braintrustdata.com/).

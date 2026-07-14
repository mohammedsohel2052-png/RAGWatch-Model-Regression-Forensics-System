"""
rag_core.py — RAG Pipeline Core

Handles:
  - Loading doc.md and splitting it into chunks by ## headers
  - Embedding chunks using sentence-transformers (local, no API key needed)
  - Retrieving the top-k most relevant chunks for a query using cosine similarity
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Model (loaded once at import time) ──────────────────────────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[RAG] Loading embedding model '{_MODEL_NAME}'...")
        _model = SentenceTransformer(_MODEL_NAME)
        print("[RAG] Model loaded.")
    return _model


# ── Document Loading ─────────────────────────────────────────────────────────

def load_chunks(doc_path: str = "doc.md") -> list[dict]:
    """
    Reads doc.md and splits it into chunks by '## ' headers.
    Returns a list of dicts: {id, title, text}
    """
    with open(doc_path, "r", encoding="utf-8") as f:
        raw = f.read()

    sections = re.split(r"\n(?=## )", raw.strip())
    chunks = []
    for i, section in enumerate(sections):
        lines = section.strip().splitlines()
        title = lines[0].lstrip("#").strip() if lines else f"Section {i}"
        text = "\n".join(lines).strip()
        chunks.append({
            "id": f"chunk_{i}",
            "title": title,
            "text": text,
        })

    print(f"[RAG] Loaded {len(chunks)} chunks from '{doc_path}'.")
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    Adds an 'embedding' key to each chunk and returns a stacked embedding matrix.
    """
    model = _get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
    return chunks, embeddings


def embed_query(query: str) -> np.ndarray:
    """Returns a normalised embedding vector for a query string."""
    model = _get_model()
    return model.encode([query], normalize_embeddings=True)[0]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, chunks: list[dict], chunk_embeddings: np.ndarray, top_k: int = 2) -> list[dict]:
    """
    Finds the top-k most semantically similar chunks using cosine similarity.
    Because embeddings are L2-normalised, cosine similarity == dot product.

    Returns a list of chunk dicts (with id, title, text) sorted by relevance descending.
    """
    if top_k == 0:
        return []

    query_emb = embed_query(query)
    scores = chunk_embeddings @ query_emb  # shape: (n_chunks,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        result = {k: v for k, v in chunks[idx].items() if k != "embedding"}
        result["score"] = float(scores[idx])
        results.append(result)
    return results


# ── Convenience embed_fn for Evaluator ───────────────────────────────────────

def embed_fn(text: str) -> np.ndarray:
    """Single-string embed function compatible with ragwatch Evaluator."""
    return embed_query(text).reshape(1, -1)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunks = load_chunks()
    chunks, chunk_embs = embed_chunks(chunks)

    query = "How does my deductible affect my insurance premium?"
    results = retrieve(query, chunks, chunk_embs, top_k=2)

    print(f"\nQuery: {query}")
    print("Top retrieved chunks:")
    for r in results:
        print(f"  [{r['id']}] (score={r['score']:.3f}) {r['title']}")
        print(f"    {r['text'][:120]}...")

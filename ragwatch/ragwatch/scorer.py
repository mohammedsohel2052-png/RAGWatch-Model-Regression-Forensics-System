import numpy as np
from typing import Optional, Callable

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 2D arrays (1, dim) each."""
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def score_context_precision(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    relevant = set(retrieved_ids).intersection(set(expected_ids))
    return len(relevant) / len(retrieved_ids)

def score_context_recall(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    relevant = set(retrieved_ids).intersection(set(expected_ids))
    return len(relevant) / len(expected_ids)

def score_answer_relevancy(generated_answer: str, expected_answer: str, embed_fn: Callable) -> float:
    gen_emb = embed_fn(generated_answer)
    exp_emb = embed_fn(expected_answer)
    similarity = cosine_similarity(gen_emb, exp_emb)
    return float(max(0.0, similarity))

def score_faithfulness(generated_answer: str, expected_ids: list[str]) -> float:
    """
    Scores faithfulness in two scenarios:

    1. Trick question (expected_ids is empty):
       The correct answer is to say "I don't know" rather than hallucinating.
       Returns 1.0 if the model admitted ignorance, 0.0 if it hallucinated.

    2. Normal question (expected_ids is not empty):
       The model should give a substantive answer, not refuse unnecessarily.
       Returns 0.5 if the model said "I don't know" when it had context (over-refusal),
       otherwise 1.0.
    """
    lower_answer = generated_answer.lower()
    refusal_phrases = ["i don't know", "i do not know", "not mentioned", "not covered", "cannot answer"]
    is_refusal = any(phrase in lower_answer for phrase in refusal_phrases)

    if not expected_ids:
        # Trick question: reward refusal, penalise hallucination
        return 1.0 if is_refusal else 0.0
    else:
        # Normal question: reward substantive answer, penalise unnecessary refusal
        return 0.5 if is_refusal else 1.0

def score_run(golden_record: dict, retrieved_docs: list[dict], generated_answer: str, embed_fn: Callable = None) -> dict:
    expected_ids = golden_record.get("expected_source_chunk_ids", [])
    retrieved_ids = [doc.get("id") or doc.get("chunk_id") for doc in retrieved_docs]
    
    precision = score_context_precision(retrieved_ids, expected_ids)
    recall = score_context_recall(retrieved_ids, expected_ids)
    
    relevancy = 0.0
    if embed_fn and golden_record.get("expected_answer"):
        relevancy = score_answer_relevancy(generated_answer, golden_record["expected_answer"], embed_fn)
        
    faithfulness = score_faithfulness(generated_answer, expected_ids)
    
    return {
        "context_precision": precision,
        "context_recall": recall,
        "answer_relevancy": relevancy,
        "faithfulness": faithfulness
    }

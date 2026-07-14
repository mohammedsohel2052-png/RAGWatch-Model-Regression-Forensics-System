"""
generator.py — LLM Generation Layer

Strategy (tries in order):
  1. Ollama local server (http://localhost:11434) — FREE, no internet, no API key
  2. OpenAI API  (requires OPENAI_API_KEY in .env)
  3. Stub fallback — returns a deterministic canned answer for offline demos

Set GENERATOR_BACKEND=ollama | openai | stub in your .env to force a backend.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _get_backend() -> str:
    """Read BACKEND lazily so os.environ overrides set after import always work."""
    return os.getenv("GENERATOR_BACKEND", "auto").lower()

# ── Ollama backend ────────────────────────────────────────────────────────────

def _is_ollama_available() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def _generate_ollama(query: str, docs: list[dict]) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    context = "\n\n---\n\n".join(d["text"] for d in docs) if docs else "No relevant context found."
    prompt = (
        "You are an insurance expert assistant. "
        "Answer the question based ONLY on the provided context. "
        "If the context does not contain the answer, say 'I don't know'.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    latency_ms = (time.time() - t0) * 1000
    answer = resp.choices[0].message.content.strip()

    return {
        "answer": answer,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        "latency_ms": latency_ms,
        "backend": "ollama",
    }


# ── OpenAI backend ────────────────────────────────────────────────────────────

def _generate_openai(query: str, docs: list[dict]) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = "\n\n---\n\n".join(d["text"] for d in docs) if docs else "No relevant context found."
    system_prompt = (
        "You are an insurance expert assistant. "
        "Answer the question based ONLY on the provided context. "
        "If the context does not contain the answer, say 'I don't know'."
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

    t0 = time.time()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    latency_ms = (time.time() - t0) * 1000
    answer = resp.choices[0].message.content.strip()

    # Rough cost estimate (gpt-4o-mini pricing)
    prompt_tokens = resp.usage.prompt_tokens
    completion_tokens = resp.usage.completion_tokens
    cost_usd = (prompt_tokens * 0.00000015) + (completion_tokens * 0.0000006)

    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "backend": "openai",
    }


# ── Stub / offline fallback ───────────────────────────────────────────────────

_STUB_ANSWERS = {
    "deductible": "A deductible is the amount you pay out of pocket before your insurance coverage kicks in.",
    "premium": "A premium is the amount you pay to maintain your insurance policy, typically on a monthly or annual basis.",
    "comprehensive": "Comprehensive auto insurance covers damage from theft, vandalism, natural disasters, fire, falling objects, and animal collisions.",
    "liability": "Liability insurance covers costs when you are legally responsible for injuring someone or damaging their property.",
    "claim": "Filing a claim begins with notifying your insurer, providing documentation, and working with a claims adjuster.",
    "uninsured": "Uninsured Motorist coverage protects you if you are in an accident with a driver who has no insurance.",
    "exclusion": "A policy exclusion is a specific condition or event that is NOT covered by your insurance policy.",
    "collision": "Collision coverage pays for damage to your vehicle from hitting another car or object. Comprehensive covers non-collision events.",
    "rider": "A rider is an optional add-on to your standard insurance policy that provides additional coverage.",
    "gap": "Gap insurance covers the difference between your car's market value and remaining loan balance if totaled.",
    "credit": "Yes, in most states a lower credit score leads to higher premiums.",
    "roadside": "Roadside assistance covers towing, flat tire service, and battery jump starts.",
}


def _generate_stub(query: str, docs: list[dict]) -> dict:
    """Deterministic fallback for offline demos — no API key or Ollama needed."""
    t0 = time.time()
    query_lower = query.lower()

    # If no docs retrieved (trick question), say I don't know
    if not docs:
        answer = "I don't know, this topic is not covered in the provided documents."
    else:
        # Match a keyword from stub library
        answer = None
        for keyword, stub_answer in _STUB_ANSWERS.items():
            if keyword in query_lower:
                answer = stub_answer
                break
        if answer is None:
            answer = docs[0]["text"][:200].strip() + "..." if docs else "I don't know."

    latency_ms = (time.time() - t0) * 1000

    return {
        "answer": answer,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": latency_ms,
        "cost_usd": 0.0,
        "backend": "stub",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def generate(query: str, docs: list[dict]) -> dict:
    """
    Generate an answer for `query` given retrieved `docs`.
    Auto-selects the best available backend.
    """
    backend = _get_backend()

    if backend == "stub":
        return _generate_stub(query, docs)

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Set it in .env or use GENERATOR_BACKEND=stub.")
        return _generate_openai(query, docs)

    if backend == "ollama":
        return _generate_ollama(query, docs)

    # backend == "auto": try Ollama -> OpenAI -> stub
    if _is_ollama_available():
        print("[Generator] Using Ollama (local).")
        return _generate_ollama(query, docs)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("[Generator] Using OpenAI API.")
        return _generate_openai(query, docs)

    print("[Generator] No LLM backend available -- using stub (offline mode).")
    return _generate_stub(query, docs)

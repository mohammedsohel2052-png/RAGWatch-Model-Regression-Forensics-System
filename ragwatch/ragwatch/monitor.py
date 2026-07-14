import functools
import time
import uuid
from .storage import Storage

# Module-level storage cache to avoid creating a new DB connection per call
_storage_cache: dict[str, Storage] = {}

def monitor(project_name: str = "default", db_path: str = "ragwatch.db"):
    """
    Zero-friction @monitor decorator.

    Wraps any RAG generation function to automatically capture:
      - Function name, project, latency
      - A preview of the output
      - Persists a trace record to SQLite for the ragwatch dashboard

    Usage:
        @monitor(project_name="insurance-rag")
        def generate_answer(query, docs):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Lazily create a single Storage instance per db_path
            if db_path not in _storage_cache:
                _storage_cache[db_path] = Storage(db_path=db_path)
            storage = _storage_cache[db_path]

            start_time = time.time()

            # Execute the real function
            result = func(*args, **kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Build the trace record
            output_str = str(result.get("answer", result)) if isinstance(result, dict) else str(result)
            trace = {
                "trace_id": str(uuid.uuid4()),
                "project": project_name,
                "function": func.__name__,
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "output_preview": output_str[:200]
            }

            # Persist trace to SQLite
            storage.save_trace(trace)
            print(f"[RAGWatch] [OK] Traced '{func.__name__}' | {latency_ms:.0f}ms | project='{project_name}'")

            return result
        return wrapper
    return decorator

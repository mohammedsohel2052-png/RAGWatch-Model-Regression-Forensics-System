import uuid
import datetime
import os
from .storage import Storage
from .scorer import score_run
from .regression import check_for_regressions

class Evaluator:
    def __init__(self, db_path="ragwatch.db", embed_fn=None):
        """
        Initialize the RAGWatch Evaluator.
        embed_fn: A function that takes a string and returns a numpy array embedding.
        """
        self.storage = Storage(db_path)
        self.embed_fn = embed_fn

    def evaluate(self, golden_dataset: list[dict], retrieval_fn, generation_fn):
        """
        Runs the evaluation pipeline against a golden dataset.
        retrieval_fn: fn(query) -> list[dict] (returns retrieved chunks)
        generation_fn: fn(query, chunks) -> dict (returns {"answer": str, "prompt_tokens": int, "completion_tokens": int, "latency_ms": float})
        """
        print(f"Starting evaluation of {len(golden_dataset)} items...")
        
        results = []
        precisions, recalls, relevancies, faithfulnesses, latencies = [], [], [], [], []
        
        for i, item in enumerate(golden_dataset):
            print(f"[{i+1}/{len(golden_dataset)}] {item['question']}")
            
            # Execute pipeline
            retrieved_docs = retrieval_fn(item["question"])
            gen = generation_fn(item["question"], retrieved_docs)
            
            # Score
            scores = score_run(item, retrieved_docs, gen["answer"], self.embed_fn)
            
            precisions.append(scores["context_precision"])
            recalls.append(scores["context_recall"])
            relevancies.append(scores["answer_relevancy"])
            faithfulnesses.append(scores["faithfulness"])
            latencies.append(gen.get("latency_ms", 0))
            
            results.append({
                "question": item["question"],
                "answer": gen["answer"],
                "scores": scores
            })
            print(f"  -> {gen['answer'][:60]}...")
            
        run_id = str(uuid.uuid4())
        
        summary = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "commit_hash": os.environ.get("GITHUB_SHA", "local_run"),
            "metrics": {
                "avg_context_precision": sum(precisions) / len(precisions) if precisions else 0,
                "avg_context_recall": sum(recalls) / len(recalls) if recalls else 0,
                "avg_answer_relevancy": sum(relevancies) / len(relevancies) if relevancies else 0,
                "faithfulness_score": sum(faithfulnesses) / len(faithfulnesses) if faithfulnesses else 0,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            },
            "details": results
        }
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Precision: {summary['metrics']['avg_context_precision']:.2f}")
        print(f"Recall:    {summary['metrics']['avg_context_recall']:.2f}")
        print(f"Relevancy: {summary['metrics']['avg_answer_relevancy']:.2f}")
        print(f"Faithful:  {summary['metrics']['faithfulness_score']:.2f}")
        
        self.storage.save_run(summary)
        return summary

    def check_regressions(self, current_summary):
        baseline = self.storage.get_baseline_run()
        if not baseline:
            print("No baseline found.")
            return []
        return check_for_regressions(current_summary, baseline)

    def promote_to_baseline(self, run_id):
        self.storage.promote_to_baseline(run_id)

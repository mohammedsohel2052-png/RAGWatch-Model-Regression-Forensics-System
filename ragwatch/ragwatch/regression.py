def check_for_regressions(current_summary: dict, baseline_summary: dict, thresholds: dict = None) -> list[str]:
    """
    Compares the current run against the baseline and returns a list of regression alerts.
    """
    if thresholds is None:
        thresholds = {
            "avg_context_precision": 0.10,
            "avg_context_recall": 0.10,
            "avg_answer_relevancy": 0.05,
            "faithfulness_score": 0.001
        }
        
    alerts = []
    curr = current_summary["metrics"]
    base = baseline_summary["metrics"]
    
    for metric, threshold in thresholds.items():
        if metric in curr and metric in base:
            drop = base[metric] - curr[metric]
            if drop > threshold:
                alerts.append(f"Regression in {metric}: dropped by {drop:.2f} (threshold {threshold})")
                
    if alerts:
        print("\n[ALERT] REGRESSIONS DETECTED!")
        for a in alerts:
            print(f"  - {a}")
    else:
        print("\nNo regressions detected. Pipeline is stable.")
        
    return alerts

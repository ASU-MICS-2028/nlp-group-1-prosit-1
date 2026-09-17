"""
Evaluation metrics and benchmark comparisons for statistical and neural language models.
"""

from typing import List, Dict, Any
import numpy as np


def compute_cross_entropy(log_probabilities: List[float]) -> float:
    """
    Computes cross-entropy from a sequence of log probabilities.
    H(P, Q) = - (1 / N) * sum(log2(P(w)))
    """
    if not log_probabilities:
        return float("inf")
    return -float(np.mean(log_probabilities))


def format_evaluation_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Formats a dictionary of model evaluation results into a clean markdown table.

    Example input:
        {
            "Unigram (MLE)": {"Perplexity": 412.5, "OOV Rate": 0.05},
            "Bigram (Laplace)": {"Perplexity": 185.3, "OOV Rate": 0.05},
            "Trigram (Kneser-Ney)": {"Perplexity": 92.1, "OOV Rate": 0.05},
        }
    """
    if not results:
        return "No results."

    first_model = next(iter(results.keys()))
    metrics = list(results[first_model].keys())

    header = "| Model | " + " | ".join(metrics) + " |"
    divider = "| --- | " + " | ".join(["---"] * len(metrics)) + " |"

    rows = []
    for model_name, metric_dict in results.items():
        vals = [f"{metric_dict.get(m, 'N/A'):.2f}" if isinstance(metric_dict.get(m), (int, float)) else str(metric_dict.get(m)) for m in metrics]
        rows.append(f"| {model_name} | " + " | ".join(vals) + " |")

    return "\n".join([header, divider] + rows)

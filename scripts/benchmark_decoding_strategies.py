"""
Benchmark Decoding Strategies for Domain-Adapted DistilGPT2 (LoRA).
Quantifies the impact of repetition penalty, n-gram blocking, and temperature tuning
on suppressing degenerative phrase loops in agricultural advisory completions.
Saves empirical results to reports/decoding_strategies_benchmark.json.
"""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "models" / "domain_adapted_checkpoint"
REPORTS_DIR = REPO_ROOT / "reports"
BASE_MODEL_NAME = "distilgpt2"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_repetition_metrics(text: str):
    """Calculate distinct-1, distinct-2, and distinct-3 ratios to quantify repetition."""
    words = text.lower().split()
    if not words:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0, "word_count": 0}
    
    unigrams = words
    bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    
    d1 = len(set(unigrams)) / max(len(unigrams), 1)
    d2 = len(set(bigrams)) / max(len(bigrams), 1)
    d3 = len(set(trigrams)) / max(len(trigrams), 1)
    
    return {
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "distinct_3": round(d3, 4),
        "word_count": len(words)
    }


def run_decoding_benchmark():
    print("=== Benchmarking Decoding Strategies for LoRA-Adapted LLM ===")
    
    print(f"Loading checkpoint from: {CHECKPOINT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    adapted_model = PeftModel.from_pretrained(base_model, str(CHECKPOINT_DIR))
    adapted_model.eval()

    prompts = [
        "Question: why is crop rotation important in farming?\nAnswer:",
        "Question: What farming practice helps prevent soil erosion?\nAnswer:",
        "Question: How can farmers control fall armyworm in maize?\nAnswer:",
        "Question: What are the signs of nitrogen deficiency in crops?\nAnswer:",
    ]

    strategies = {
        "unpenalized_sampling": {
            "name": "Unpenalized Sampling (Baseline)",
            "params": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "do_sample": True}
        },
        "repetition_penalty_only": {
            "name": "Repetition Penalty (r=1.3)",
            "params": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.3, "no_repeat_ngram_size": 0, "do_sample": True}
        },
        "ngram_blocking": {
            "name": "Strict N-gram Blocking (N=3)",
            "params": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.0, "no_repeat_ngram_size": 3, "do_sample": True}
        },
        "conservative_agronomic": {
            "name": "Conservative Agronomic Sampling (Recommended)",
            "params": {"temperature": 0.35, "top_p": 0.85, "repetition_penalty": 1.25, "no_repeat_ngram_size": 3, "do_sample": True}
        },
        "deterministic_greedy": {
            "name": "Deterministic Greedy Search",
            "params": {"do_sample": False, "repetition_penalty": 1.25, "no_repeat_ngram_size": 3}
        }
    }

    benchmark_data = {
        "model": "distilgpt2 + LoRA (r=8, alpha=32)",
        "prompts_evaluated": len(prompts),
        "results": []
    }

    for prompt in prompts:
        prompt_entry = {
            "prompt": prompt.strip(),
            "evaluations": {}
        }
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        for strat_id, strat in strategies.items():
            with torch.no_grad():
                output = adapted_model.generate(
                    input_ids,
                    max_new_tokens=45,
                    pad_token_id=tokenizer.eos_token_id,
                    **strat["params"]
                )
            full_text = tokenizer.decode(output[0], skip_special_tokens=True)
            answer = full_text[len(prompt):].strip()
            metrics = calculate_repetition_metrics(answer)
            
            prompt_entry["evaluations"][strat_id] = {
                "strategy_name": strat["name"],
                "generated_answer": answer,
                "metrics": metrics
            }
            
        benchmark_data["results"].append(prompt_entry)

    # Compute average distinct-3 across all prompts per strategy
    strategy_averages = {}
    for strat_id, strat in strategies.items():
        d3_scores = [p["evaluations"][strat_id]["metrics"]["distinct_3"] for p in benchmark_data["results"]]
        strategy_averages[strat_id] = {
            "name": strat["name"],
            "avg_distinct_3": round(sum(d3_scores) / len(d3_scores), 4)
        }
    benchmark_data["strategy_averages"] = strategy_averages

    out_file = REPORTS_DIR / "decoding_strategies_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nBenchmark completed successfully! Saved to: {out_file}")
    print("\nSummary of Distinct-3 Ratios (Higher = Less Repetitive):")
    for s_id, s_info in strategy_averages.items():
        print(f"  - {s_info['name']}: {s_info['avg_distinct_3'] * 100:.1f}% unique trigrams")


if __name__ == "__main__":
    run_decoding_benchmark()

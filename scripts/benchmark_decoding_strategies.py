"""
Benchmark Decoding Strategies for Domain-Adapted DistilGPT2 (LoRA).
Measures how repetition penalty, n-gram blocking, and temperature change phrase looping in
agricultural completions, using Distinct-3 (unique word trigrams / all word trigrams per answer).

Distinct-3 measures repetition only, not whether the advice is correct. With no_repeat_ngram_size=3
the decoder is forbidden to repeat any token trigram, so Distinct-3 is close to 1 by construction.
Every sampled strategy is run with N_SAMPLES fixed seeds per prompt, so the averages are reproducible.

Run after src/train_domain_lora.py:  python scripts/benchmark_decoding_strategies.py
Saves empirical results to reports/decoding_strategies_benchmark.json.
"""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "models" / "domain_adapted_checkpoint"
REPORTS_DIR = REPO_ROOT / "reports"
BASE_MODEL_NAME = "distilgpt2"
RANDOM_SEED = 42
N_SAMPLES = 5  # seeds per (prompt, strategy); greedy search is deterministic and runs once

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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
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
            "name": "Low temperature + penalty + 3-gram block",
            "params": {"temperature": 0.35, "top_p": 0.85, "repetition_penalty": 1.25, "no_repeat_ngram_size": 3, "do_sample": True}
        },
        "deterministic_greedy": {
            "name": "Deterministic Greedy Search",
            "params": {"do_sample": False, "repetition_penalty": 1.25, "no_repeat_ngram_size": 3}
        }
    }

    benchmark_data = {
        "model": "distilgpt2 + LoRA (r=8, alpha=32), models/domain_adapted_checkpoint",
        "prompts_evaluated": len(prompts),
        "samples_per_prompt": N_SAMPLES,
        "note": "Distinct-3 measures repetition, not correctness. no_repeat_ngram_size=3 makes it ~1 by construction.",
        "results": []
    }

    for prompt in prompts:
        prompt_entry = {
            "prompt": prompt.strip(),
            "evaluations": {}
        }
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        for strat_id, strat in strategies.items():
            samples = []
            for i in range(N_SAMPLES if strat["params"]["do_sample"] else 1):
                set_seed(RANDOM_SEED + i)
                with torch.no_grad():
                    output = adapted_model.generate(
                        input_ids,
                        max_new_tokens=45,
                        pad_token_id=tokenizer.eos_token_id,
                        **strat["params"]
                    )
                answer = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                samples.append({"generated_answer": answer, "metrics": calculate_repetition_metrics(answer)})

            prompt_entry["evaluations"][strat_id] = {
                "strategy_name": strat["name"],
                "samples": samples,
                "mean_distinct_3": round(sum(x["metrics"]["distinct_3"] for x in samples) / len(samples), 4),
            }
            
        benchmark_data["results"].append(prompt_entry)

    # Compute average distinct-3 across all prompts per strategy
    strategy_averages = {}
    for strat_id, strat in strategies.items():
        d3_scores = [x["metrics"]["distinct_3"] for p in benchmark_data["results"] for x in p["evaluations"][strat_id]["samples"]]
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

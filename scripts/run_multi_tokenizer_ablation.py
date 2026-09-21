"""
Script to run multi-tokenizer ablation sweeps (N=1..6 across all 5 tokenizers)
for individual datasets and output structured tables for the Learning Journal.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizers import (
    WhitespaceTokenizer,
    UnicodeWordTokenizer,
    EweRuleStemmerTokenizer,
    CharacterTokenizer,
    SimpleBPETokenizer,
)
from src.experiment_runner import run_ngram_experiment


def run_dataset_sweep(
    train_path: str,
    test_path: str,
    dataset_name: str,
    output_json_path: str,
    max_order: int = 6,
    subsample_for_char: int = 4000,
) -> Dict[str, Any]:
    print(f"\n{'='*85}")
    print(f"RUNNING ALL 5 TOKENIZERS ON: {dataset_name}")
    print(f"{'='*85}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(test_path, "r", encoding="utf-8") as f:
        test_lines = [l.strip() for l in f if l.strip()]

    print(f"Loaded {len(train_lines)} train sentences, {len(test_lines)} test sentences.")

    # 1. Initialize tokenizers
    tokenizers = [
        ("Whitespace", WhitespaceTokenizer(), train_lines, test_lines),
        ("Unicode Word", UnicodeWordTokenizer(), train_lines, test_lines),
        ("Ewe Stemmer", EweRuleStemmerTokenizer(), train_lines, test_lines),
        (
            "Character",
            CharacterTokenizer(),
            train_lines[:subsample_for_char] if len(train_lines) > subsample_for_char else train_lines,
            test_lines[:subsample_for_char // 8] if len(test_lines) > subsample_for_char // 8 else test_lines,
        ),
    ]

    # BPE requires training first
    print("Training BPE tokenizer (150 merges)...")
    bpe = SimpleBPETokenizer(num_merges=150)
    bpe.train(train_lines[:3000] if len(train_lines) > 3000 else train_lines)
    tokenizers.append(("Byte-Pair Encoding (BPE)", bpe, train_lines, test_lines))

    sweep_results = {}

    for name, tok, tr_corpus, te_corpus in tokenizers:
        t0 = time.time()
        print(f"\n--- Running N=1..{max_order} for [{name}] ({len(tr_corpus)} sents) ---")
        res = run_ngram_experiment(
            train_corpus=tr_corpus,
            test_corpus=te_corpus,
            tokenizer=tok,
            max_order=max_order,
            smoothing="interpolation",
        )
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.2f}s")
        sweep_results[name] = res

    # Format output summary
    summary = {
        "dataset": dataset_name,
        "train_sentences": len(train_lines),
        "test_sentences": len(test_lines),
        "results": sweep_results,
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print markdown comparison table
    print(f"\n{'='*95}")
    print(f"MULTI-TOKENIZER SUMMARY MATRIX FOR {dataset_name} (Perplexity / Sparsity %)")
    print(f"{'='*95}")
    header = f"{'Order':<8} | {'Whitespace':<16} | {'Unicode Word':<16} | {'Ewe Stemmer':<16} | {'BPE (Subwords)':<16} | {'Character':<16}"
    print(header)
    print("-" * 95)

    for n in range(1, max_order + 1):
        order_str = f"{n}-gram" if n > 3 else ("Unigram" if n == 1 else ("Bigram" if n == 2 else "Trigram"))
        row_parts = [f"{order_str:<8}"]
        for tok_name in ["Whitespace", "Unicode Word", "Ewe Stemmer", "Byte-Pair Encoding (BPE)", "Character"]:
            tok_res = sweep_results[tok_name]
            entry = next((item for item in tok_res if item["order"] == n), None)
            if entry:
                val = f"{entry['perplexity']:.1f} ({entry['sparsity_pct']:.1f}%)"
            else:
                val = "N/A"
            row_parts.append(f"{val:<16}")
        print(" | ".join(row_parts))

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["1", "2", "3", "4", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ["2", "all"]:
        run_dataset_sweep(
            train_path="data/processed/dataset_2_json/train.txt",
            test_path="data/processed/dataset_2_json/test.txt",
            dataset_name="Dataset 2 (eweenglishsentence.json - Micro-Data)",
            output_json_path="reports/results_dataset_2_all_tokenizers.json",
            max_order=6,
        )

    if args.dataset in ["1", "all"]:
        run_dataset_sweep(
            train_path="data/processed/dataset_1_csv/train.txt",
            test_path="data/processed/dataset_1_csv/test.txt",
            dataset_name="Dataset 1 (EWE_ENGLISH.csv - Cultural Stories)",
            output_json_path="reports/results_dataset_1_all_tokenizers.json",
            max_order=6,
            subsample_for_char=4000,
        )

    if args.dataset in ["3", "all"]:
        run_dataset_sweep(
            train_path="data/processed/dataset_3_speech/train.txt",
            test_path="data/processed/dataset_3_speech/test.txt",
            dataset_name="Dataset 3 (Waxal Spoken Audio Transcripts - Oral Domain)",
            output_json_path="reports/results_dataset_3_all_tokenizers.json",
            max_order=6,
            subsample_for_char=4000,
        )

    if args.dataset in ["4", "all"]:
        run_dataset_sweep(
            train_path="data/processed/dataset_4_parquet/train.txt",
            test_path="data/processed/dataset_4_parquet/test.txt",
            dataset_name="Dataset 4 (Large-Scale Web & Scripture Corpus - 64k Sents)",
            output_json_path="reports/results_dataset_4_all_tokenizers.json",
            max_order=6,
            subsample_for_char=4000,
        )



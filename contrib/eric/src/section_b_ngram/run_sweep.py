"""
Runs the Section B sweep: 5 tokenizers x N=1..6, interpolated Kneser-Ney, on one dataset at a time.
Every tokenizer is trained on the same full training split and scored on the same validation and test
sentences. The best order per tokenizer is picked on validation perplexity.

Build the splits first (python -m src.section_b_ngram.build_datasets), then from the repo root:
    python -m src.section_b_ngram.run_sweep --dataset unified
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from src import ROOT
from src.section_b_ngram.ewe_tokenizers import (
    WhitespaceTokenizer,
    UnicodeWordTokenizer,
    EweRuleStemmerTokenizer,
    CharacterTokenizer,
    SimpleBPETokenizer,
)
from src.section_b_ngram.experiment_runner import run_ngram_experiment
from src.section_b_ngram.ngram import NGramLM
from src.section_b_ngram.preprocessing import build_vocabulary, replace_oov_tokens

RESULTS = ROOT / "results" / "section_b_ngram"
MAX_EVAL = 4000  # validation and test sentences scored per dataset, the same ones for every tokenizer

DATASETS = {
    "1": ("dataset_1_csv", "Dataset 1 (EWE_ENGLISH.csv)", "dataset_1.json"),
    "2": ("dataset_2_json", "Dataset 2 (eweenglishsentence.json, micro-data)", "dataset_2.json"),
    "3": ("dataset_3_speech", "Dataset 3 (Waxal speech transcriptions)", "dataset_3.json"),
    "4": ("dataset_4_parquet", "Dataset 4 (ewe_corpus.parquet, first 200k rows)", "dataset_4.json"),
    "unified": ("unified", "Unified corpus (all four sources)", "unified.json"),
}


def read_lines(path: Path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def discount_check(train, val, n):
    """RULES.md 1: compare the Ney-estimated discount with fixed values on validation, never on test."""
    tok = UnicodeWordTokenizer()
    tr, va = [tok.tokenize(x) for x in train], [tok.tokenize(x) for x in val]
    vocab, _ = build_vocabulary(tr, min_freq=2)
    tr, va = replace_oov_tokens(tr, vocab), replace_oov_tokens(va, vocab)
    out = {}
    for d in (None, 0.5, 0.75, 0.9):
        m = NGramLM(n=n, smoothing="kneser_ney", discount=d).fit(tr, vocab=vocab)
        out["ney" if d is None else str(d)] = round(m.perplexity(va), 2)
    return out


def run_dataset_sweep(key: str, max_order: int = 6) -> Dict[str, Any]:
    folder, label, out_name = DATASETS[key]
    data = ROOT / "data" / "processed" / folder
    train, val, test = (read_lines(data / f"{s}.txt") for s in ("train", "val", "test"))
    val_eval, test_eval = val[:MAX_EVAL], test[:MAX_EVAL]
    print(f"\n{label}: {len(train)} train, scoring {len(val_eval)} val and {len(test_eval)} test sentences", flush=True)

    bpe = SimpleBPETokenizer(num_merges=150)
    bpe.train(train)
    tokenizers = [WhitespaceTokenizer(), UnicodeWordTokenizer(), EweRuleStemmerTokenizer(), bpe, CharacterTokenizer()]

    results, best = {}, {}
    for tok in tokenizers:
        t0 = time.time()
        rows = run_ngram_experiment(train, val_eval, test_eval, tok, max_order=max_order, smoothing="kneser_ney")
        results[tok.name] = rows
        b = min(rows, key=lambda r: r["val_perplexity"])
        best[tok.name] = {k: b[k] for k in ("order", "val_perplexity", "perplexity", "per_word_perplexity")}
        print(f"  {tok.name:28s} best N={b['order']} (by val) test PPL {b['perplexity']} per-word {b['per_word_perplexity']}  [{time.time() - t0:.0f}s]", flush=True)

    summary = {
        "dataset": label,
        "smoothing": "interpolated Kneser-Ney (Ney discount per order); <unk> = tokens seen once in train",
        "train_sentences": len(train),
        "val_sentences_scored": len(val_eval),
        "test_sentences_scored": len(test_eval),
        "best_order_by_val": best,
        "discount_check_val_unicode_word": discount_check(train, val_eval, best["Unicode Word"]["order"]),
        "results": results,
    }
    with open(RESULTS / out_name, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nTest perplexity per token | per word, {label}")
    print(f"{'N':<3}" + "".join(f"{t.name[:22]:>34s}" for t in tokenizers))
    for n in range(1, max_order + 1):
        cells = [next(r for r in results[t.name] if r["order"] == n) for t in tokenizers]
        print(f"{n:<3}" + "".join(f"{c['perplexity']:>16.1f} | {c['per_word_perplexity']:>13.1f}" for c in cells))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args()
    for key in DATASETS if args.dataset == "all" else [args.dataset]:
        run_dataset_sweep(key)
